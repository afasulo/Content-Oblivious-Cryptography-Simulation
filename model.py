# simulation_project/model.py

import mesa

from . import config
from .core.oracle import Oracle
from .core.market_state import MakerState
from .core.vault import Vault
from .processing.mempool import Mempool
from .processing.block_producer import BlockProducer
from .agents.borrower_agent import BorrowerAgent # Ensure BorrowerAgent is imported
from .agents.keeper_agent import KeeperAgent
from .agents.mev_agent import MEVSearcherAgent
from .analysis import metrics

class MakerLiquidationModel(mesa.Model):
    """The main agent-based model for MakerDAO Liquidations."""
    def __init__(self, n_borrowers: int = config.N_BORROWERS,
                 n_keepers: int = config.N_KEEPERS,
                 n_mev_searchers: int = config.N_MEV_SEARCHERS,
                 ofp_mode: str = 'Baseline',
                 current_vdf_delay_t: float = config.VDF_DELAY_T,
                 market_shock_step: int = -1,
                 market_shock_factor: float = 1.0,
                 seed=None):

        super().__init__(seed=seed)

        self.num_borrowers = n_borrowers
        self.num_keepers = n_keepers
        self.num_mev_searchers = n_mev_searchers
        self.ofp_mode = ofp_mode
        self.market_shock_step = market_shock_step
        self.market_shock_factor = market_shock_factor


        if self.ofp_mode == 'VDF':
            config.VDF_DELAY_T = current_vdf_delay_t

        self.current_time = 0.0
        self.time_step_duration = float(config.TIME_STEP_DURATION_SECONDS)

        self.stage_list = [
            "env_update",
            "borrower_actions",
            "monitor_trigger",
            "bid_submit",
            "mev_scan_submit",
            "block_process",
            "state_update"
        ]

        self.oracle = Oracle(
            initial_price=config.INITIAL_ETH_PRICE,
            volatility=config.PRICE_VOLATILITY,
            time_step_seconds=self.time_step_duration,
            model=self
        )
        self.maker_state = MakerState(model=self)
        self.mempool = Mempool()
        self.block_producer = BlockProducer(model=self)

        self.completed_or_failed_auctions = []
        self.all_auctions_data = []
        self.total_value_liquidated = 0.0
        self.total_collateral_added_by_borrowers = 0.0
        self.total_debt_repaid_by_borrowers = 0.0


        for i in range(self.num_borrowers):
            vault_owner_id_str = f"BorrowerVault_{i}"
            initial_collateral = self.random.uniform(config.MIN_LIQUIDATION_RATIO * 5, config.MIN_LIQUIDATION_RATIO * 30)
            cr_buffer_low = 0.02
            cr_buffer_high = 0.40
            safe_cr_target = self.random.uniform(
                config.MIN_LIQUIDATION_RATIO + cr_buffer_low,
                config.MIN_LIQUIDATION_RATIO + cr_buffer_high
            )
            initial_debt = (initial_collateral * config.INITIAL_ETH_PRICE) / safe_cr_target
            initial_debt = max(1000.0, initial_debt)

            external_dai_holding = initial_debt * self.random.uniform(0.2, 0.6)
            external_coll_holding = initial_collateral * self.random.uniform(0.1, 0.3)

            vault = Vault(
                owner_id=vault_owner_id_str, initial_collateral=initial_collateral,
                initial_debt=initial_debt, ilk='ETH-A'
            )
            self.maker_state.add_vault(vault)
            _ = BorrowerAgent(model=self, vault=vault,
                              external_dai=external_dai_holding,
                              external_collateral=external_coll_holding)


        for i in range(self.num_keepers):
            keeper = KeeperAgent(model=self)
            initial_dai_keeper = self.random.uniform(100000, 500000)
            self.maker_state.add_agent_balance(keeper.unique_id, initial_dai_keeper)

        if self.num_mev_searchers > 0:
            for i in range(self.num_mev_searchers):
                mev_searcher = MEVSearcherAgent(model=self)
                initial_dai_mev = self.random.uniform(200000, 1000000)
                self.maker_state.add_agent_balance(mev_searcher.unique_id, initial_dai_mev)

        print(f"--- Model Initialized: OFP Mode = {self.ofp_mode}, VDF_T = {config.VDF_DELAY_T if self.ofp_mode == 'VDF' else 'N/A'}, ShockStep = {self.market_shock_step} ---")
        print(f"Agents Created: Borrowers={self.num_borrowers}, Keepers={self.num_keepers}, MEV={self.num_mev_searchers}. Total in model.agents: {len(self.agents)}")
        if config.VERBOSE_LOGGING:
            self._print_initial_vault_crs()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Steps": "steps", "Time": "current_time", "OFP_Mode": "ofp_mode",
                "OraclePrice": lambda m: m.oracle.current_price,
                "ActiveAuctions": lambda m: sum(len(c.active_auctions) for c in m.maker_state.clippers.values()),
                "CompletedAuctions": lambda m: len([r for r in m.completed_or_failed_auctions if r['status']=='Completed']),
                "FailedAuctions": lambda m: len([r for r in m.completed_or_failed_auctions if r['status']=='Failed']),
                "TotalValueLiquidated": "total_value_liquidated",
                "BadDebt": metrics.get_total_bad_debt, "MEVProfit": metrics.get_mev_profit,
                "KeeperProfit": metrics.get_keeper_profit, "AvgAuctionDuration": metrics.get_average_auction_duration,
                "AvgPriceEfficiency": metrics.get_average_price_efficiency, "OFPOverheadProxy": metrics.get_ofp_overhead_proxy,
                "TotalCollateralAddedByBorrowers": "total_collateral_added_by_borrowers",
                "TotalDebtRepaidByBorrowers": "total_debt_repaid_by_borrowers"
            },
            agent_reporters={
                "AgentType": lambda a: a.__class__.__name__,
                "DAI_Balance": lambda a: a.model.maker_state.get_agent_balance(a.unique_id) if not isinstance(a, BorrowerAgent) else getattr(a, 'external_dai_holdings', 0),
                "TotalProfit": lambda a: getattr(a, 'total_profit', 0),
                "GasSpent": lambda a: getattr(a, 'gas_spent', 0),
                "VaultCR": lambda a: a.vault.get_collateralization_ratio(a.model.oracle.current_price) if isinstance(a, BorrowerAgent) else None,
                "VaultStatus": lambda a: a.vault.status if isinstance(a, BorrowerAgent) else None,
                "VaultDebt": lambda a: a.vault.debt_amount if isinstance(a, BorrowerAgent) else None,
                "VaultCollateral": lambda a: a.vault.collateral_amount if isinstance(a, BorrowerAgent) else None,
            }
        )
        self.running = True
        self.datacollector.collect(self)

    def _print_initial_vault_crs(self):
        print("--- Initial Vault CRs (Sample) ---")
        sample_vault_owners = [v.owner_id for v in self.maker_state.vaults.values()][:min(5, len(self.maker_state.vaults))]
        for owner_id in sample_vault_owners:
            vault = self.maker_state.get_vault(owner_id)
            if vault:
                initial_cr = vault.get_collateralization_ratio(config.INITIAL_ETH_PRICE)
                print(f"Vault {vault.owner_id}: Initial CR = {initial_cr:.3f} "
                      f"(Debt: {vault.debt_amount:.2f}, Coll: {vault.collateral_amount:.2f} ETH)")
        print("---------------------------------")

    def record_auction_end(self, auction):
        if auction.status not in ["Completed", "Failed"]:
            print(f"Warning: record_auction_end called for auction {auction.id} with status {auction.status}")
            return

        collateral_sold = auction.initial_lot - auction.remaining_lot
        clearing_price = (auction.dai_raised / collateral_sold) if collateral_sold > 1e-9 else 0
        oracle_price_at_end = self.oracle.current_price
        price_efficiency = (clearing_price / oracle_price_at_end) if oracle_price_at_end > 1e-9 and clearing_price > 0 else 0
        
        current_run_value = 0 # Default
        if self.datacollector.model_vars.get('run'):
            current_run_value = self.datacollector.model_vars['run'][-1]
        elif hasattr(self, 'current_run_number_for_data'): # If passed from main_simulation
            current_run_value = self.current_run_number_for_data

        auction_data_point = {
            'auction_id': auction.id,
            'scenario': self.ofp_mode + (f"_T{config.VDF_DELAY_T}" if self.ofp_mode == 'VDF' else ("_Shock" if self.market_shock_step != -1 and self.steps >= self.market_shock_step else "")),
            'run': current_run_value,
            'ilk': auction.ilk,
            'status': auction.status,
            'tab_debt': auction.tab,
            'dai_raised': auction.dai_raised,
            'start_time_step': auction.start_time / self.time_step_duration,
            'end_time_step': self.current_time / self.time_step_duration,
            'duration_seconds': self.current_time - auction.start_time,
            'initial_lot': auction.initial_lot,
            'remaining_lot': auction.remaining_lot,
            'total_collateral_sold': collateral_sold,
            'oracle_price_at_kick': auction.start_price / config.PRICE_BUFFER if config.PRICE_BUFFER > 0 else auction.start_price, # Avoid division by zero
            'oracle_price_at_end': oracle_price_at_end,
            'start_auction_price': auction.start_price,
            'clearing_price': clearing_price,
            'num_bids': len(auction.bids),
            'price_efficiency': price_efficiency
        }
        self.all_auctions_data.append(auction_data_point)
        self.completed_or_failed_auctions.append(auction_data_point)

        clipper = self.maker_state.get_clipper(auction.ilk)
        if clipper and auction.id in clipper.active_auctions:
            del clipper.active_auctions[auction.id]

    def step(self):
        new_price = self.oracle.update_price(self.steps, self.market_shock_step, self.market_shock_factor)
        if config.VERBOSE_LOGGING:
             print(f"[Step {self.steps}, Time {self.current_time:.0f}] Oracle Price: {new_price:.2f}")

        self.agents.do("env_update")
        
        # CORRECTED: Call borrower_actions only on BorrowerAgent instances
        if BorrowerAgent in self.agents_by_type: # Check if there are any BorrowerAgents
            self.agents_by_type[BorrowerAgent].do("borrower_actions")
        
        self.agents.do("monitor_trigger")
        self.agents.do("bid_submit")
        self.agents.do("mev_scan_submit")
        transactions_for_block = self.mempool.get_transactions_for_block()
        if transactions_for_block:
            _ = self.block_producer.process_transactions(transactions_for_block)
        self.agents.do("block_process")
        self.agents.do("state_update")
        self.current_time += self.time_step_duration
        self.datacollector.collect(self)
        if self.steps >= config.SIMULATION_STEPS:
            self.running = False

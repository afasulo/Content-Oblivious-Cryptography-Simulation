# simulation_project/agents/keeper_agent.py

import mesa
from ..processing.transaction import Transaction
from .. import config # For config.VERBOSE_LOGGING

class KeeperAgent(mesa.Agent):
    """
    Monitors vaults, triggers liquidations (barks), and bids in auctions (takes)
    to profit from the liquidation process.
    """
    def __init__(self, model: mesa.Model,
                 profit_margin: float = config.KEEPER_PROFIT_MARGIN,
                 gas_strategy: str = 'medium'):
        super().__init__(model)
        self.acquired_collateral = {}
        self.profit_margin = float(profit_margin)
        self.gas_strategy = gas_strategy

        self.profit_from_bids = 0.0
        self.profit_from_incentives = 0.0
        self.gas_spent = 0.0

    @property
    def total_profit(self) -> float:
        return self.profit_from_bids + self.profit_from_incentives - self.gas_spent

    def get_dai_balance(self) -> float:
        return self.model.maker_state.get_agent_balance(self.unique_id)

    def get_gas_price(self) -> float:
        base = config.BASE_GAS_PRICE
        multiplier = 1.0
        if self.gas_strategy == 'high':
            multiplier = 1.5
        elif self.gas_strategy == 'low':
            multiplier = 0.8
        return base * multiplier * self.model.random.uniform(0.95, 1.05)

    def pay_gas(self, gas_units: float):
        eth_cost = gas_units * config.BASE_GAS_PRICE / (10**9)
        dai_equivalent_cost = eth_cost * self.model.oracle.current_price
        self.gas_spent += dai_equivalent_cost

    def record_incentive(self, tip: float, chip_factor: float, tab_auction_value: float):
        incentive_amount = float(tip) + (float(chip_factor) * float(tab_auction_value))
        self.profit_from_incentives += incentive_amount

    def update_state_post_take(self, ilk: str, collateral_received: float, dai_cost: float):
        self.acquired_collateral[ilk] = self.acquired_collateral.get(ilk, 0.0) + collateral_received
        if collateral_received > 1e-9:
            cost_price_per_unit = dai_cost / collateral_received
            estimated_sale_value = collateral_received * self.model.oracle.current_price
            profit_from_this_take = estimated_sale_value - dai_cost
            self.profit_from_bids += profit_from_this_take

    def env_update(self):
        pass

    def monitor_trigger(self):
        current_price = self.model.oracle.current_price
        if current_price <= 0:
            return

        for vault_id, vault in self.model.maker_state.vaults.items():
            if vault.status == "Active":
                cr = vault.get_collateralization_ratio(current_price)
                if cr < config.MIN_LIQUIDATION_RATIO:
                    if config.VERBOSE_LOGGING:
                        print(f"    >>>> [Keeper {self.unique_id}] Found liquidatable Vault {vault_id} (CR: {cr:.2f}). Submitting bark... <<<<")

                    ilk = vault.ilk
                    clipper = self.model.maker_state.get_clipper(ilk)
                    if not clipper:
                        # This is a warning about setup, should probably always be visible
                        print(f"    Warning: [Keeper {self.unique_id}] No clipper found for ilk {ilk} to bark Vault {vault_id}.")
                        continue

                    tab_to_raise = vault.debt_amount * (1 + config.LIQUIDATION_PENALTY)
                    collateral_to_seize = vault.collateral_amount

                    bark_tx = Transaction(
                        tx_type='bark',
                        sender_id=self.unique_id,
                        params={
                            'vault_id': vault_id,
                            'ilk': ilk,
                            'tab': tab_to_raise,
                            'collateral_to_seize': collateral_to_seize
                        },
                        gas_price=self.get_gas_price(),
                        submission_time=self.model.current_time
                    )
                    self.model.mempool.add_transaction(bark_tx)
                    return

    def bid_submit(self):
        current_time = self.model.current_time
        oracle_price = self.model.oracle.current_price
        if oracle_price <= 0:
            return

        current_dai_balance = self.get_dai_balance()
        if current_dai_balance <= 1e-6:
            return

        for ilk, clipper in self.model.maker_state.clippers.items():
            active_auction_items = list(clipper.active_auctions.items())
            for auction_id, auction in active_auction_items:
                if auction.status == "Active":
                    current_auction_price = auction.get_current_price(current_time)
                    target_buy_price = oracle_price * (1 - self.profit_margin)

                    if 0 < current_auction_price < target_buy_price:
                        if config.VERBOSE_LOGGING:
                            print(f"  [Keeper {self.unique_id}] Opportunity in Auction {auction_id} ({ilk}). "
                                  f"Auction Price: {current_auction_price:.2f} < Target Buy Price: {target_buy_price:.2f}")

                        desired_collateral_amount = auction.remaining_lot * 0.25
                        max_affordable_collateral = (current_dai_balance * 0.5) / current_auction_price if current_auction_price > 1e-9 else 0
                        min_bid_collateral_amount = 0.01

                        bid_collateral_amount = min(desired_collateral_amount, max_affordable_collateral, auction.remaining_lot)
                        bid_collateral_amount = max(bid_collateral_amount, min_bid_collateral_amount)
                        bid_collateral_amount = min(bid_collateral_amount, auction.remaining_lot)

                        if bid_collateral_amount > 1e-6:
                            max_price_for_bid = current_auction_price * 1.001

                            take_tx = Transaction(
                                tx_type='take',
                                sender_id=self.unique_id,
                                params={
                                    'auction_id': auction_id,
                                    'ilk': ilk,
                                    'amt_collateral': bid_collateral_amount,
                                    'max_price': max_price_for_bid
                                },
                                gas_price=self.get_gas_price(),
                                submission_time=current_time
                            )

                            if self.model.ofp_mode == 'TE':
                                if config.VERBOSE_LOGGING:
                                    print(f"    [Keeper {self.unique_id}] Encrypting 'take' tx for {auction_id} in TE mode.")
                                take_tx.is_encrypted = True

                            self.model.mempool.add_transaction(take_tx)
                            if config.VERBOSE_LOGGING:
                                print(f"    >>> [Keeper {self.unique_id}] Submitted 'take' tx for Auction {auction_id}. "
                                      f"Amt: {bid_collateral_amount:.4f}, MaxPrice: {max_price_for_bid:.2f} <<<")
                            return

    def mev_scan_submit(self):
        pass

    def block_process(self):
        pass

    def state_update(self):
        pass

    def __repr__(self):
        return (f"KeeperAgent(id={self.unique_id}, profit={self.total_profit:.2f}, "
                f"DAI={self.get_dai_balance():.2f}, gas_spent={self.gas_spent:.2f})")


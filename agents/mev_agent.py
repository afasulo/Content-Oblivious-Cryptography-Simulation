# simulation_project/agents/mev_agent.py

import mesa
from ..processing.transaction import Transaction
from .. import config # For config.VERBOSE_LOGGING

class MEVSearcherAgent(mesa.Agent):
    """
    Scans the mempool for profitable opportunities, specifically trying to
    front-run 'take' transactions submitted by other agents (e.g., Keepers).
    """
    def __init__(self, model: mesa.Model,
                 frontrun_margin: float = config.MEV_FRONTUN_MARGIN,
                 gas_boost_factor: float = config.MEV_GAS_BOOST):
        super().__init__(model)
        self.acquired_collateral = {}
        self.profit = 0.0 # Original name, used for profit from bids
        self.frontrun_margin = float(frontrun_margin)
        self.gas_boost_factor = float(gas_boost_factor)
        self.gas_spent = 0.0

    @property
    def total_profit(self) -> float:
        return self.profit - self.gas_spent

    def get_dai_balance(self) -> float:
        return self.model.maker_state.get_agent_balance(self.unique_id)

    def pay_gas(self, gas_units: float):
        eth_cost = gas_units * config.BASE_GAS_PRICE / (10**9)
        dai_equivalent_cost = eth_cost * self.model.oracle.current_price
        self.gas_spent += dai_equivalent_cost

    def update_state_post_take(self, ilk: str, collateral_received: float, dai_cost: float):
        self.acquired_collateral[ilk] = self.acquired_collateral.get(ilk, 0.0) + collateral_received
        if collateral_received > 0:
             cost_price = dai_cost / collateral_received
             profit_estimate = (self.model.oracle.current_price - cost_price) * collateral_received
             self.profit += profit_estimate

    def env_update(self):
        pass

    def monitor_trigger(self):
        pass

    def bid_submit(self):
        pass

    def mev_scan_submit(self):
        if self.model.ofp_mode != 'Baseline':
            return

        oracle_price = self.model.oracle.current_price
        if oracle_price <= 0:
            return

        current_dai_balance = self.get_dai_balance()
        if current_dai_balance <= 1e-6:
            return

        mempool_snapshot = self.model.mempool.view_transactions()

        for victim_tx in mempool_snapshot:
            victim_agent_candidate = self.model._agents.get(victim_tx.sender_id)

            if not (victim_tx.tx_type == 'take' and \
                    victim_agent_candidate is not None and \
                    victim_agent_candidate != self and \
                    not isinstance(victim_agent_candidate, MEVSearcherAgent)):
                continue

            victim_agent = victim_agent_candidate

            auction_id = victim_tx.params.get('auction_id')
            ilk = victim_tx.params.get('ilk')
            victim_amt = victim_tx.params.get('amt_collateral')
            victim_max_price = victim_tx.params.get('max_price')
            victim_gas_price = victim_tx.gas_price
            potential_buy_price = victim_max_price
            required_price = oracle_price * (1 - self.frontrun_margin)

            if 0 < potential_buy_price < required_price:
                if config.VERBOSE_LOGGING:
                    print(f"  [MEV {self.unique_id}] Found front-run target: Auction {auction_id}, Victim: {victim_tx.sender_id} ({victim_agent.__class__.__name__}), Price: {potential_buy_price:.2f}")

                my_bid_amt = victim_amt
                my_max_price = victim_max_price
                estimated_cost = my_bid_amt * potential_buy_price

                if estimated_cost <= current_dai_balance:
                    my_gas_price = victim_gas_price * self.gas_boost_factor

                    frontrun_tx = Transaction(
                        tx_type='take',
                        sender_id=self.unique_id,
                        params={'auction_id': auction_id, 'ilk': ilk, 'amt_collateral': my_bid_amt, 'max_price': my_max_price},
                        gas_price=my_gas_price,
                        submission_time=self.model.current_time
                    )
                    self.model.mempool.add_transaction(frontrun_tx)
                    if config.VERBOSE_LOGGING:
                        print(f"    >>> [MEV {self.unique_id}] Submitted FRONT-RUN 'take' tx for {auction_id}. Gas: {my_gas_price:.2f} > Victim Gas: {victim_gas_price:.2f} <<<")
                    return
                else:
                    if config.VERBOSE_LOGGING:
                        print(f"  [MEV {self.unique_id}] Opportunity found for {auction_id}, but insufficient DAI. "
                              f"Need: {estimated_cost:.2f}, Have: {current_dai_balance:.2f}")
    def block_process(self):
        pass

    def state_update(self):
        pass

    def __repr__(self):
        return (f"MEVSearcherAgent(id={self.unique_id}, profit={self.total_profit:.2f}, "
                f"DAI={self.get_dai_balance():.2f}, gas_spent={self.gas_spent:.2f})")

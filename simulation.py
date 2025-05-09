# simulation.py
# Complete Agent-Based Simulation for MakerDAO Liquidations with OFPs
# Corrected for Mesa 3.x + Fixed TE/VDF Bark/Take Logic

import mesa
import random
import numpy as np
import pandas as pd
import math
import time

# --- Global Constants & Parameters (ADJUSTED FOR ACTIVITY) ---
# Market & Simulation Setup
INITIAL_ETH_PRICE = 2000.0
PRICE_VOLATILITY = 0.50 # High Volatility - adjust as needed
PRICE_DRIFT = -0.10
N_BORROWERS = 50
N_KEEPERS = 5
N_MEV_SEARCHERS = 2
SIMULATION_STEPS = 1440
TIME_STEP_DURATION_SECONDS = 60

# MakerDAO Parameters
MIN_LIQUIDATION_RATIO = 1.50
LIQUIDATION_PENALTY = 0.13
PRICE_BUFFER = 1.20
AUCTION_DURATION_LIMIT = 1800
AUCTION_RESET_THRESHOLD = 0.60
KEEPER_INCENTIVE_TIP = 100
KEEPER_INCENTIVE_CHIP = 0.01

# PVT Curve Parameters
PVT_CURVE_TYPE = 'StairstepExponentialDecrease'
PVT_STEP_DURATION = 60
PVT_STEP_DECAY = 0.99
PVT_LINEAR_TAU = 3600

# OFP Parameters
TE_DECRYPTION_LATENCY = 0.2
VDF_DELAY_T = 5.0
VDF_VERIFICATION_GAS_COST = 2_000_000

# General Blockchain/Agent Parameters
BASE_GAS_PRICE = 20
KEEPER_PROFIT_MARGIN = 0.03
MEV_FRONTUN_MARGIN = 0.01
MEV_GAS_BOOST = 1.1

# Precision
STABLECOIN_PRECISION = 10**18
COLLATERAL_PRECISION = 10**18

# --- Environment Components ---

class Oracle:
    """Represents the price feed."""
    def __init__(self, initial_price, volatility, time_step_seconds, model):
        self.model = model
        self.current_price = initial_price
        self.volatility = volatility
        self.dt = time_step_seconds / (60 * 60 * 24 * 365)

    def update_price(self):
        mu = PRICE_DRIFT
        dW = self.model.random.normalvariate(0, math.sqrt(self.dt))
        effective_volatility = min(self.volatility, 1.0)
        dS = self.current_price * (mu * self.dt + effective_volatility * dW)
        self.current_price += dS
        self.current_price = max(self.current_price, 0.01)
        return self.current_price

class Vault:
    """Represents a Borrower's Vault."""
    def __init__(self, owner_id, initial_collateral, initial_debt):
        self.owner_id = owner_id
        self.collateral_amount = initial_collateral
        self.debt_amount = initial_debt
        self.status = "Active"
        self.ilk = 'ETH-A'

    def get_collateralization_ratio(self, current_price):
        if current_price <= 0: return float('inf')
        if self.debt_amount <= 1e-9: return float('inf')
        collateral_value = self.collateral_amount * current_price
        return collateral_value / self.debt_amount

class Auction:
    """Represents a single Liquidation 2.0 Dutch Auction."""
    def __init__(self, auction_id, ilk, lot_collateral, tab_debt, start_time, start_price,
                 pvt_curve_params, duration_limit, reset_threshold_factor, model):
        self.id = auction_id
        self.ilk = ilk
        self.initial_lot = lot_collateral
        self.remaining_lot = lot_collateral
        self.tab = tab_debt
        self.dai_raised = 0.0
        self.start_time = start_time
        self.start_price = start_price
        self.pvt_curve_params = pvt_curve_params
        self.duration_limit = duration_limit
        self.reset_threshold_factor = reset_threshold_factor
        self.status = "Active"
        self.bids = []
        self.model = model

    def get_elapsed_time(self, current_time):
        return max(0, current_time - self.start_time)

    def get_current_price(self, current_time):
        if self.status != "Active": return 0
        elapsed_time = self.get_elapsed_time(current_time)
        curve_type = self.pvt_curve_params.get('type')

        if curve_type == 'StairstepExponentialDecrease':
            step = self.pvt_curve_params.get('step', PVT_STEP_DURATION)
            cut = self.pvt_curve_params.get('cut', PVT_STEP_DECAY)
            if step <= 0: return 0
            num_steps_taken = math.floor(elapsed_time / step)
            current_price = self.start_price * (cut ** num_steps_taken)
        elif curve_type == 'LinearDecrease':
            tau = self.pvt_curve_params.get('tau', PVT_LINEAR_TAU)
            if tau <= 0: return 0
            price_drop_rate = self.start_price / tau
            current_price = self.start_price - price_drop_rate * elapsed_time
        else:
            print(f"Warning: Unknown PVT curve type: {curve_type} for Auction {self.id}")
            current_price = self.start_price
        return max(0, current_price)

class Clipper:
    """Simulates the Clipper contract for one collateral type (ilk)."""
    def __init__(self, ilk, model, pvt_params, buf, tail, cusp, tip, chip):
        self.ilk = ilk
        self.model = model
        self.active_auctions = {}
        self.next_auction_id = 0
        self.pvt_curve_params = pvt_params
        self.buf = buf
        self.tail = tail
        self.cusp_factor = cusp
        self.tip = tip
        self.chip = chip

    def kick(self, vault_urn, tab_debt, lot_collateral, current_time):
        auction_id = f"{self.ilk}-{self.next_auction_id}"
        self.next_auction_id += 1
        oracle_price = self.model.oracle.current_price
        if oracle_price <=0:
            print(f"    !!! [Time {current_time:.0f}] Clipper ({self.ilk}): Cannot kick auction {auction_id}. Oracle price <= 0 ({oracle_price:.2f}) !!!")
            return None, 0, 0
        start_price = oracle_price * self.buf
        new_auction = Auction(
            auction_id=auction_id, ilk=self.ilk, lot_collateral=lot_collateral, tab_debt=tab_debt,
            start_time=current_time, start_price=start_price, pvt_curve_params=self.pvt_curve_params,
            duration_limit=self.tail, reset_threshold_factor=self.cusp_factor, model=self.model
        )
        self.active_auctions[auction_id] = new_auction
        print(f"    >>> [Time {current_time:.0f}] Clipper ({self.ilk}): KICKED Auction {auction_id}. Lot: {lot_collateral:.4f}, Tab: {tab_debt:.2f}, Start Price: {start_price:.2f} <<<")
        self.model.total_value_liquidated += tab_debt
        return auction_id, self.tip, self.chip

    def take(self, auction_id, bidder_id, amt_collateral, max_price_per_unit, bidder_dai_balance, current_time):
        if auction_id not in self.active_auctions: return False, 0, 0, "Auction not found"
        auction = self.active_auctions[auction_id]
        if auction.status != "Active": return False, 0, 0, f"Auction status {auction.status}"

        current_price = auction.get_current_price(current_time)
        if current_price <= 1e-9: return False, 0, 0, "Auction price is zero"
        if current_price > max_price_per_unit:
            # ADDED: Specific check for TE/VDF modes if price mismatch happens
            if self.model.ofp_mode in ['TE', 'VDF']:
                print(f"      (Info {self.model.ofp_mode}): Bid failed for {bidder_id} on {auction_id} - Price {current_price:.2f} > Max {max_price_per_unit:.2f} (Latency effect?)")
            return False, 0, 0, "Price too high"

        actual_amt_collateral = min(amt_collateral, auction.remaining_lot)
        dai_needed_for_tab = max(0, auction.tab - auction.dai_raised)
        max_collateral_for_tab = dai_needed_for_tab / current_price if current_price > 1e-9 else float('inf')
        actual_amt_collateral = min(actual_amt_collateral, max_collateral_for_tab)
        dai_cost = actual_amt_collateral * current_price

        if dai_cost > bidder_dai_balance:
            affordable_collateral = bidder_dai_balance / current_price if current_price > 1e-9 else 0
            actual_amt_collateral = min(actual_amt_collateral, affordable_collateral)
            dai_cost = actual_amt_collateral * current_price

        if actual_amt_collateral <= 1e-9: return False, 0, 0, "Insufficient balance or adjusted amount too small"

        auction.remaining_lot -= actual_amt_collateral
        auction.dai_raised += dai_cost
        auction.bids.append({'bidder': bidder_id, 'time': current_time, 'amount_collateral': actual_amt_collateral,
                           'price_paid': current_price, 'dai_cost': dai_cost})
        print(f"    >>> [Time {current_time:.0f}] Bidder {bidder_id} SUCCESSFUL take on {auction_id}. Bought: {actual_amt_collateral:.4f} ETH @ {current_price:.2f} DAI/ETH. Cost: {dai_cost:.2f}. Raised: {auction.dai_raised:.2f}/{auction.tab:.2f} <<<")


        if auction.dai_raised >= auction.tab - 1e-6:
            auction.status = "Completed"
            print(f"    >>> [Time {current_time:.0f}] Auction {auction_id} COMPLETED. Raised: {auction.dai_raised:.2f}/{auction.tab:.2f}. Left: {auction.remaining_lot:.4f} <<<")
            self.model.record_auction_end(auction)

        return True, actual_amt_collateral, dai_cost, "Success"

    def check_stale_auctions(self, current_time):
         for auction_id, auction in list(self.active_auctions.items()):
            if auction.status != "Active": continue
            elapsed_time = auction.get_elapsed_time(current_time)
            current_price = auction.get_current_price(current_time)
            reset_price_threshold = auction.start_price * auction.reset_threshold_factor
            is_stale_tail = elapsed_time > auction.duration_limit
            is_stale_cusp = current_price > 0 and current_price < reset_price_threshold and elapsed_time > 0

            if is_stale_tail or is_stale_cusp:
                 reason = "DURATION (tail)" if is_stale_tail else "PRICE (cusp)"
                 print(f"    !!! [Time {current_time:.0f}] Auction {auction_id} hit {reason} limit. Status: Failed. Raised: {auction.dai_raised:.2f}/{auction.tab:.2f} !!!")
                 auction.status = "Failed"
                 self.model.record_auction_end(auction)

class MakerState:
    """Holds the overall state of the simulated MakerDAO system."""
    def __init__(self, model):
        self.model = model
        self.vaults = {}
        self.clippers = {
            'ETH-A': Clipper(ilk='ETH-A', model=model,
                             pvt_params={'type': PVT_CURVE_TYPE, 'step': PVT_STEP_DURATION, 'cut': PVT_STEP_DECAY, 'tau': PVT_LINEAR_TAU},
                             buf=PRICE_BUFFER, tail=AUCTION_DURATION_LIMIT, cusp=AUCTION_RESET_THRESHOLD,
                             tip=KEEPER_INCENTIVE_TIP, chip=KEEPER_INCENTIVE_CHIP)
        }
        self.agent_dai_balances = {}

    def add_vault(self, vault):
        self.vaults[vault.owner_id] = vault

    def add_agent_balance(self, agent_id, amount):
         self.agent_dai_balances[agent_id] = amount

    def get_agent_balance(self, agent_id):
         return self.agent_dai_balances.get(agent_id, 0)

    def update_agent_balance(self, agent_id, change):
         current_balance = self.get_agent_balance(agent_id)
         self.agent_dai_balances[agent_id] = max(0, current_balance + change)

class Transaction:
     """ Represents a transaction in the mempool or being processed. """
     def __init__(self, tx_type, sender_id, params, gas_price, submission_time):
         self.tx_type = tx_type
         self.sender_id = sender_id
         self.params = params
         self.gas_price = gas_price
         self.submission_time = submission_time
         self.status = "Pending"
         self.execution_time = -1
         self.is_encrypted = False # Only set for 'take' tx in TE mode
         self.vdf_reveal_time = -1 # Only used for 'take' tx in VDF mode

     def __lt__(self, other):
         if not isinstance(other, Transaction): return NotImplemented
         return self.gas_price > other.gas_price

class Mempool:
    """Simulates the transaction mempool."""
    def __init__(self):
        self.pending_txs = []

    def add_transaction(self, tx):
        self.pending_txs.append(tx)

    def get_transactions_for_block(self, max_txs=-1):
        try:
            sorted_txs = sorted(self.pending_txs, key=lambda tx: tx.gas_price, reverse=True)
        except AttributeError as e:
            print(f"Error sorting mempool - potentially contains non-Transaction object: {e}")
            print(f"Mempool contents: {self.pending_txs}")
            sorted_txs = [tx for tx in self.pending_txs if isinstance(tx, Transaction)]
        self.pending_txs = []
        return sorted_txs[:max_txs] if max_txs > 0 else sorted_txs

    def view_transactions(self):
         return list(self.pending_txs)

class BlockProducer:
    """Simulates block production and transaction processing."""
    def __init__(self, model):
        self.model = model

    def process_transactions(self, transactions):
        executed_txs = []
        if transactions:
            print(f"  [BlockProducer] Processing {len(transactions)} transactions...")
            # Debug Print: Show available agent IDs (keys from internal dict)
            # try:
            #     agent_keys = list(self.model.agents._agents.keys())
            #     key_types = [type(k).__name__ for k in agent_keys]
            #     print(f"    Available Agent IDs (Dict Keys Types): {key_types}") # Still shows objects
            # except Exception as e:
            #     print(f"    ERROR trying to list agent keys: {e}")
            # # Print actual unique_ids from the AgentSet for comparison
            # try:
            #     current_agent_ids = [(a.unique_id, type(a).__name__) for a in self.model.agents]
            #     print(f"    Available Agent unique_ids (from AgentSet): {sorted(current_agent_ids)}")
            # except Exception as e:
            #      print(f"    ERROR iterating agents for unique_ids: {e}")


        for tx in transactions:
            if not isinstance(tx, Transaction):
                print(f"    Warning: Found non-Transaction object in processing queue: {tx}. Skipping.")
                continue

            print(f"    Processing Tx: Type={tx.tx_type}, Sender={tx.sender_id}, Gas={tx.gas_price:.1f}")
            success = False
            details = "Processing Error"
            gas_cost_proxy = 0
            current_processing_time = self.model.current_time

            # --- Agent Lookup (Using Iteration) ---
            agent = None
            if isinstance(tx.sender_id, int):
                # print(f"      Attempting lookup for Sender ID: {tx.sender_id} (Type: {type(tx.sender_id)})") # Debug
                try:
                    agent = next((a for a in self.model.agents if a.unique_id == tx.sender_id), None)
                    if agent is None:
                         print(f"      >>>>>> Warning: Agent ID {tx.sender_id} not found via iteration. Skipping. <<<<<<")
                         # current_agent_ids_on_fail = [(a.unique_id, type(a).__name__) for a in self.model.agents] # Debug
                         # print(f"         IDs available at time of failure: {sorted(current_agent_ids_on_fail)}") # Debug
                         continue
                    # else: print(f"      Agent ID {tx.sender_id} found successfully via iteration: {agent}") # Debug
                except Exception as e:
                     print(f"      ERROR iterating agents to find ID {tx.sender_id}: {e}. Skipping.")
                     continue
            else:
                print(f"      Warning: Sender ID {tx.sender_id} is not an integer. Skipping.")
                continue
            # --- End Agent Lookup ---


            # --- OFP Delays/Checks ---
            # Apply delays only *after* agent is confirmed and *only* to relevant transaction types

            # Threshold Encryption Delay (Post-Ordering / Pre-Execution) - Only for 'take'
            if self.model.ofp_mode == 'TE' and tx.tx_type == 'take' and tx.is_encrypted:
                print(f"      TE Decryption Delay: {TE_DECRYPTION_LATENCY:.2f}s (Simulated). Content revealed.")
                tx.is_encrypted = False # Content now visible

            # Verifiable Delay Function Delay (Pre-Execution) - Only for 'take'
            if self.model.ofp_mode == 'VDF' and tx.tx_type == 'take':
                 if tx.vdf_reveal_time < 0: tx.vdf_reveal_time = tx.submission_time + VDF_DELAY_T
                 if current_processing_time < tx.vdf_reveal_time:
                      print(f"      VDF Delay Pending for 'take' Tx from {tx.sender_id}. Reveal: {tx.vdf_reveal_time:.0f} > Current: {current_processing_time:.0f}")
                      self.model.mempool.add_transaction(tx) # Re-queue
                      continue # Skip processing this tx now
                 gas_cost_proxy += VDF_VERIFICATION_GAS_COST
                 print(f"      VDF Delay Met for 'take' Tx. Added Verification Cost: {VDF_VERIFICATION_GAS_COST}")


            # --- Transaction Execution Logic ---
            if tx.tx_type == 'bark':
                # Bark txs execute immediately if selected (no TE/VDF applied above)
                vault_id = tx.params['vault_id']
                ilk = tx.params['ilk']
                tab = tx.params['tab']
                collateral = tx.params['collateral_to_seize']
                vault = self.model.maker_state.vaults.get(vault_id)

                if vault and vault.status == "Active":
                    clipper = self.model.maker_state.clippers.get(ilk)
                    if clipper:
                         vault.status = "Liquidating"
                         auction_id, tip, chip = clipper.kick(vault_id, tab, collateral, self.model.current_time)
                         if auction_id is not None:
                              if hasattr(agent, 'record_incentive'): agent.record_incentive(tip, chip, tab)
                              success = True
                              details = f"Auction {auction_id} kicked."
                              gas_cost_proxy += 500_000
                         else:
                              details = "Clipper kick failed (e.g., zero oracle price)."
                              vault.status = "Active"
                    else: details = f"Clipper for ilk {ilk} not found."
                elif not vault: details = f"Vault {vault_id} not found."
                else: details = f"Vault {vault_id} status {vault.status} (skipped)"

            elif tx.tx_type == 'take':
                # Take txs execute after potential TE/VDF delays handled above
                auction_id = tx.params['auction_id']
                ilk = tx.params['ilk']
                amt = tx.params['amt_collateral']
                max_price = tx.params['max_price']
                bidder_id = tx.sender_id

                clipper = self.model.maker_state.clippers.get(ilk)
                if clipper:
                     bidder_balance = self.model.maker_state.get_agent_balance(bidder_id)
                     bid_success, collateral_received, dai_cost, take_details = clipper.take( # Renamed local var
                         auction_id, bidder_id, amt, max_price, bidder_balance, self.model.current_time
                     )
                     details = take_details # Assign details from take function call
                     if bid_success:
                          success = True
                          self.model.maker_state.update_agent_balance(bidder_id, -dai_cost)
                          if hasattr(agent, 'update_state_post_take'):
                              agent.update_state_post_take(ilk, collateral_received, dai_cost)
                          gas_cost_proxy += 300_000
                     # ADDED Print: Check if TE/VDF take failed due to price after delay
                     elif not bid_success and self.model.ofp_mode in ['TE', 'VDF'] and details == "Price too high":
                           print(f"      (Info {self.model.ofp_mode}): 'take' from {bidder_id} on {auction_id} failed post-delay due to price.")

                else: details = f"Clipper for ilk {ilk} not found."

            # Finalize transaction status
            tx.status = "Executed" if success else "Failed"
            tx.execution_time = self.model.current_time
            executed_txs.append(tx)
            if hasattr(agent, 'pay_gas'): agent.pay_gas(gas_cost_proxy)
            print(f"      Tx Result: {tx.status}. Details: {details}")

        return executed_txs


# --- Agent Definitions (Mesa 3.x compatible) ---

class BorrowerAgent(mesa.Agent):
    """Represents a user with a Vault. Currently passive."""
    def __init__(self, model, vault):
        super().__init__(model)
        self.vault = vault

    def step(self): pass # No active logic needed for basic simulation

    # --- Stage Methods ---
    def env_update(self): pass
    def monitor_trigger(self): pass
    def bid_submit(self): pass
    def mev_scan_submit(self): pass
    def block_process(self): pass
    def state_update(self): pass # Changed from self.step()

class KeeperAgent(mesa.Agent):
    """Monitors vaults, triggers liquidations, and bids in auctions."""
    def __init__(self, model, profit_margin=KEEPER_PROFIT_MARGIN, gas_strategy='medium'):
        super().__init__(model)
        self.dai_balance = 0
        self.acquired_collateral = {}
        self.profit_margin = profit_margin
        self.gas_strategy = gas_strategy
        self.profit = 0.0
        self.gas_spent = 0.0

    def get_gas_price(self):
        base = BASE_GAS_PRICE
        mult = 1.0
        if self.gas_strategy == 'high': mult = 1.5
        if self.gas_strategy == 'low': mult = 0.8
        return base * mult * self.model.random.uniform(0.95, 1.05)

    def pay_gas(self, gas_amount_wei):
        eth_cost = gas_amount_wei * BASE_GAS_PRICE / (10**9)
        dai_equivalent_cost = eth_cost * self.model.oracle.current_price
        self.gas_spent += dai_equivalent_cost

    def record_incentive(self, tip, chip, tab):
        incentive = tip + (chip * tab)
        self.profit += incentive

    def update_state_post_take(self, ilk, collateral_received, dai_cost):
        self.dai_balance = self.model.maker_state.get_agent_balance(self.unique_id)
        self.acquired_collateral[ilk] = self.acquired_collateral.get(ilk, 0) + collateral_received
        if collateral_received > 0:
             cost_price = dai_cost / collateral_received
             profit_estimate = (self.model.oracle.current_price - cost_price) * collateral_received
             self.profit += profit_estimate

    # --- Stage Methods ---
    def env_update(self): pass
    def monitor_trigger(self):
        current_price = self.model.oracle.current_price
        if current_price <= 0: return

        for vault_id, vault in self.model.maker_state.vaults.items():
            if vault.status == "Active":
                cr = vault.get_collateralization_ratio(current_price)
                # RESTORED: Print CR check for EVERY active vault
                # print(f"  [Keeper {self.unique_id}] Checking Vault {vault_id}: Status={vault.status}, CR={cr:.4f} vs LR={MIN_LIQUIDATION_RATIO:.2f}")
                if cr < MIN_LIQUIDATION_RATIO:
                    print(f"    >>>> [Keeper {self.unique_id}] Found liquidatable Vault {vault_id} (CR: {cr:.2f}). Submitting bark... <<<<")
                    ilk = vault.ilk
                    clipper = self.model.maker_state.clippers.get(ilk)
                    if not clipper: continue
                    tab = vault.debt_amount * (1 + LIQUIDATION_PENALTY)
                    collateral_to_seize = vault.collateral_amount
                    bark_tx = Transaction(
                         tx_type='bark', sender_id=self.unique_id,
                         params={'vault_id': vault_id, 'ilk': ilk, 'tab': tab, 'collateral_to_seize': collateral_to_seize},
                         gas_price=self.get_gas_price(), submission_time=self.model.current_time
                    )
                    # FIXED: DO NOT encrypt 'bark' transactions in TE mode
                    # if self.model.ofp_mode == 'TE': bark_tx.is_encrypted = True # REMOVED
                    self.model.mempool.add_transaction(bark_tx)
                    return # Bark one per step

    def bid_submit(self):
        current_time = self.model.current_time
        oracle_price = self.model.oracle.current_price
        if oracle_price <= 0: return

        self.dai_balance = self.model.maker_state.get_agent_balance(self.unique_id)

        for ilk, clipper in self.model.maker_state.clippers.items():
            active_auction_items = list(clipper.active_auctions.items())
            for auction_id, auction in active_auction_items:
                if auction.status == "Active":
                    current_auction_price = auction.get_current_price(current_time)
                    target_buy_price = oracle_price * (1 - self.profit_margin)

                    if 0 < current_auction_price < target_buy_price:
                        print(f"  [Keeper {self.unique_id}] Opportunity in Auction {auction_id}. Price: {current_auction_price:.2f} < Target: {target_buy_price:.2f}")
                        # MODIFIED Bidding Strategy: Bid more aggressively
                        # Try to buy 25% of remaining lot, or spend 50% of DAI balance, whichever is less (min 0.1 ETH equiv)
                        desired_amt_lot = auction.remaining_lot * 0.25
                        max_affordable_amt = self.dai_balance * 0.5 / current_auction_price if current_auction_price > 1e-9 else 0
                        min_bid_eth = 0.1 # Minimum bid size to consider
                        
                        # Determine bid amount based on strategy and constraints
                        bid_amt = min(max(min_bid_eth, desired_amt_lot), max_affordable_amt, auction.remaining_lot)

                        if bid_amt > 1e-6: # Ensure meaningful bid
                            max_price = current_auction_price * 1.001
                            take_tx = Transaction(
                                tx_type='take', sender_id=self.unique_id,
                                params={'auction_id': auction_id, 'ilk': ilk, 'amt_collateral': bid_amt, 'max_price': max_price},
                                gas_price=self.get_gas_price(), submission_time=current_time
                            )
                            # FIXED: ONLY encrypt 'take' transactions in TE mode
                            if self.model.ofp_mode == 'TE':
                                print(f"    [Keeper {self.unique_id}] Encrypting 'take' tx for {auction_id} in TE mode.")
                                take_tx.is_encrypted = True

                            self.model.mempool.add_transaction(take_tx)
                            print(f"    >>> [Keeper {self.unique_id}] Submitted 'take' tx for {auction_id}. Amt: {bid_amt:.4f}, MaxPrice: {max_price:.2f} <<<")
                            return # Bid on one auction per step

    def mev_scan_submit(self): pass
    def block_process(self): pass
    def state_update(self): pass

class MEVSearcherAgent(mesa.Agent):
    """Scans mempool for profitable opportunities (front-running takes)."""
    def __init__(self, model, frontrun_margin=MEV_FRONTUN_MARGIN, gas_boost_factor=MEV_GAS_BOOST):
        super().__init__(model)
        self.dai_balance = 0
        self.acquired_collateral = {}
        self.profit = 0.0
        self.frontrun_margin = frontrun_margin
        self.gas_boost_factor = gas_boost_factor
        self.gas_spent = 0.0

    def pay_gas(self, gas_amount_wei):
        eth_cost = gas_amount_wei * BASE_GAS_PRICE / (10**9)
        dai_equivalent_cost = eth_cost * self.model.oracle.current_price
        self.gas_spent += dai_equivalent_cost

    def update_state_post_take(self, ilk, collateral_received, dai_cost):
        self.dai_balance = self.model.maker_state.get_agent_balance(self.unique_id)
        self.acquired_collateral[ilk] = self.acquired_collateral.get(ilk, 0) + collateral_received
        if collateral_received > 0:
             cost_price = dai_cost / collateral_received
             profit_estimate = (self.model.oracle.current_price - cost_price) * collateral_received
             self.profit += profit_estimate

    # --- Stage Methods ---
    def env_update(self): pass
    def monitor_trigger(self): pass
    def bid_submit(self): pass
    def mev_scan_submit(self): # Called during model's explicit MEV loop
        oracle_price = self.model.oracle.current_price
        if oracle_price <= 0: return

        mempool_view = self.model.mempool.view_transactions()
        self.dai_balance = self.model.maker_state.get_agent_balance(self.unique_id)

        if self.model.ofp_mode != 'Baseline': return

        for victim_tx in mempool_view:
            # MODIFIED Agent Lookup: Iterate through AgentSet
            victim_agent = None
            if isinstance(victim_tx.sender_id, int):
                try: victim_agent = next((a for a in self.model.agents if a.unique_id == victim_tx.sender_id), None)
                except Exception as e:
                     print(f"    [MEV {self.unique_id}] ERROR iterating agents to find ID {victim_tx.sender_id}: {e}")
                     continue
            else: continue

            if victim_tx.tx_type == 'take' and victim_agent and not isinstance(victim_agent, MEVSearcherAgent):
                auction_id = victim_tx.params['auction_id']
                ilk = victim_tx.params['ilk']
                victim_amt = victim_tx.params['amt_collateral']
                victim_max_price = victim_tx.params['max_price']
                victim_gas_price = victim_tx.gas_price
                potential_buy_price = victim_max_price
                required_price = oracle_price * (1 - self.frontrun_margin)

                if 0 < potential_buy_price < required_price:
                    print(f"  [MEV {self.unique_id}] Found front-run target: Auction {auction_id}, Victim: {victim_tx.sender_id}, Price: {potential_buy_price:.2f}")
                    my_bid_amt = victim_amt
                    my_max_price = victim_max_price
                    estimated_cost = my_bid_amt * potential_buy_price
                    if estimated_cost <= self.dai_balance:
                        my_gas_price = victim_gas_price * self.gas_boost_factor
                        frontrun_tx = Transaction(
                            tx_type='take', sender_id=self.unique_id,
                            params={'auction_id': auction_id, 'ilk': ilk, 'amt_collateral': my_bid_amt, 'max_price': my_max_price},
                            gas_price=my_gas_price, submission_time=self.model.current_time
                        )
                        # Take TXs are NOT encrypted by MEV agent itself in baseline
                        self.model.mempool.add_transaction(frontrun_tx)
                        print(f"    >>> [MEV {self.unique_id}] Submitted FRONT-RUN 'take' tx for {auction_id}. Gas: {my_gas_price:.2f} > Victim Gas: {victim_gas_price:.2f} <<<")
                        return

    def block_process(self): pass
    def state_update(self): pass

# --- Metric Helper Functions ---
# (No changes needed here)
def get_total_bad_debt(model):
    total_debt = 0
    for auction_result in model.completed_or_failed_auctions:
         shortfall = max(0, auction_result['tab'] - auction_result['dai_raised'])
         total_debt += shortfall
    return total_debt

def get_mev_profit(model):
     total_profit = 0
     for agent in model.agents:
          if isinstance(agent, MEVSearcherAgent):
               total_profit += agent.profit
     return total_profit

def get_keeper_profit(model):
     total_profit = 0
     for agent in model.agents:
          if isinstance(agent, KeeperAgent):
               total_profit += agent.profit
     return total_profit


# --- New Metric Helper Functions ---

def get_average_auction_duration(model):
    """Calculates the average duration of completed auctions."""
    completed_auctions = [
        a for a in model.completed_or_failed_auctions if a['status'] == 'Completed'
    ]
    if not completed_auctions:
        return 0
    total_duration = sum(a['duration'] for a in completed_auctions)
    return total_duration / len(completed_auctions)

def get_average_price_efficiency(model):
    """
    Calculates weighted average price efficiency for completed auctions.
    Price efficiency = (DAI raised / collateral sold) / oracle price at end
    A value of 1.0 means auctions sold at oracle price, > 1 is better, < 1 is worse
    """
    completed_auctions = [
        a for a in model.completed_or_failed_auctions 
        if a['status'] == 'Completed' and a.get('total_collateral_sold', 0) > 1e-6
    ]
    
    if not completed_auctions:
        return 0

    total_efficiency_score = 0
    total_collateral_weight = 0
    
    for auction in completed_auctions:
        collateral_sold = auction.get('total_collateral_sold', 0)
        dai_raised = auction.get('dai_raised', 0)
        oracle_price = auction.get('oracle_price_at_end', 0)
        
        if collateral_sold > 1e-6 and oracle_price > 1e-6:
            # Calculate weighted average auction price
            avg_auction_price = dai_raised / collateral_sold
            
            # Calculate efficiency ratio (auction price vs oracle price)
            efficiency = avg_auction_price / oracle_price
            
            # Add to weighted sum
            total_efficiency_score += efficiency * collateral_sold
            total_collateral_weight += collateral_sold
    
    if total_collateral_weight > 1e-6:
        return total_efficiency_score / total_collateral_weight
    else:
        return 0


def get_average_tx_latency(model_vars_df):
    """
    Estimates average end-to-end latency for transactions *if* execution times
    were logged appropriately. This requires modifying the simulation to log
    submission and execution times per transaction, which is not done by default.
    """
    # Placeholder Explanation: To calculate this, the DataCollector would need
    # to be configured to collect detailed transaction timing data (submit time,
    # execute time including OFP delays) potentially at the agent level or via
    # custom model reporting based on the processed transactions list.
    return 'N/A (Needs Data Mod)'

def get_ofp_overhead_proxy(model):
    """ Provides a proxy for OFP overhead based on the mode."""
    if model.ofp_mode == 'TE':
        # Proxy: TE Latency constant
        return f"{TE_DECRYPTION_LATENCY:.2f}s (Decryption Latency)"
    elif model.ofp_mode == 'VDF':
        # Proxy: VDF Delay T + Verification Cost (if used)
        return f"{VDF_DELAY_T:.1f}s (Delay T) + {VDF_VERIFICATION_GAS_COST:,} gas/tx (Verify Cost)"
    else: # Baseline
        return "0"

# --- End New Metric Helper Functions ---

# --- Main Model Class (Mesa 3.x compatible) ---
class MakerLiquidationModel(mesa.Model):
    """The main agent-based model for MakerDAO Liquidations."""
    def __init__(self, n_borrowers, n_keepers, n_mev_searchers, ofp_mode='Baseline', seed=None):
        super().__init__(seed=seed)
        self.num_borrowers = n_borrowers
        self.num_keepers = n_keepers
        self.num_mev_searchers = n_mev_searchers
        self.ofp_mode = ofp_mode
        self.current_time = 0
        self.time_step_duration = TIME_STEP_DURATION_SECONDS
        self.stage_list = ["env_update", "monitor_trigger", "bid_submit", "mev_scan_submit", "block_process", "state_update"]

        self.oracle = Oracle(INITIAL_ETH_PRICE, PRICE_VOLATILITY, self.time_step_duration, self)
        self.maker_state = MakerState(self)
        self.mempool = Mempool()
        self.block_producer = BlockProducer(self)
        self.completed_or_failed_auctions = []
        self.total_value_liquidated = 0.0

        # Create Agents
        for i in range(self.num_borrowers):
            vault_owner_id = f"B{i}"
            initial_collateral = self.random.uniform(5.0, 20.0)
            safe_cr = self.random.uniform(MIN_LIQUIDATION_RATIO + 0.02, 1.75)
            initial_debt = (initial_collateral * INITIAL_ETH_PRICE) / safe_cr if safe_cr > 0 else 1000
            initial_debt = max(1.0, initial_debt)
            vault = Vault(vault_owner_id, initial_collateral, initial_debt)
            self.maker_state.add_vault(vault)
            _ = BorrowerAgent(model=self, vault=vault)

        for i in range(self.num_keepers):
            # INCREASED Keeper Capital
            initial_dai = self.random.uniform(50000, 150000) # Start with more DAI
            a = KeeperAgent(model=self)
            self.maker_state.add_agent_balance(a.unique_id, initial_dai)
            a.dai_balance = initial_dai

        if self.num_mev_searchers > 0:
            for i in range(self.num_mev_searchers):
                initial_dai = self.random.uniform(100000, 300000) # MEV needs significant capital
                a = MEVSearcherAgent(model=self)
                self.maker_state.add_agent_balance(a.unique_id, initial_dai)
                a.dai_balance = initial_dai

        print("--- Initial Vault CRs (Sample) ---")
        for i in range(min(5, len(self.maker_state.vaults))):
            vault_id = f"B{i}"
            if vault_id in self.maker_state.vaults:
                vault = self.maker_state.vaults[vault_id]
                initial_cr = vault.get_collateralization_ratio(INITIAL_ETH_PRICE)
                print(f"Vault {vault_id}: Initial CR = {initial_cr:.3f} (Debt: {vault.debt_amount:.2f}, Coll: {vault.collateral_amount:.2f})")
            else:
                 print(f"Debug: Vault {vault_id} not found for initial CR print.")
        print("---------------------------------")

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Steps": "steps",
                "Time": "current_time",
                "OFP_Mode": "ofp_mode",
                "OraclePrice": lambda m: m.oracle.current_price,
                "ActiveAuctions": lambda m: sum(len(c.active_auctions) for c in m.maker_state.clippers.values()),
                "CompletedAuctions": lambda m: len([r for r in m.completed_or_failed_auctions if r['status']=='Completed']),
                "FailedAuctions": lambda m: len([r for r in m.completed_or_failed_auctions if r['status']=='Failed']),
                "TotalValueLiquidated": "total_value_liquidated",
                "BadDebt": get_total_bad_debt,
                "MEVProfit": get_mev_profit,
                "KeeperProfit": get_keeper_profit,
                # --- Added Metrics ---
                "AvgAuctionDuration": get_average_auction_duration,
                "AvgPriceEfficiency": get_average_price_efficiency, # Will show N/A for now
                "OFPOverheadProxy": get_ofp_overhead_proxy,
                # AvgTxLatency needs model_vars df, calculated post-run
            },
            agent_reporters={
                "AgentType": lambda a: a.__class__.__name__,
                "DAI_Balance": lambda a: a.model.maker_state.get_agent_balance(a.unique_id),
                "Profit": lambda a: getattr(a, 'profit', 0),
                "GasSpent": lambda a: getattr(a, 'gas_spent', 0),
            }
        )

        print(f"Model Initialized. OFP Mode: {self.ofp_mode}")
        print(f"Agents Created: {len(self.agents)}")

    def record_auction_end(self, auction):
     if auction.status not in ["Completed", "Failed"]: return
     
     # Calculate total collateral sold
     total_collateral_sold = auction.initial_lot - auction.remaining_lot
     
     self.completed_or_failed_auctions.append({
          'id': auction.id, 'ilk': auction.ilk, 'status': auction.status,
          'tab': auction.tab, 'dai_raised': auction.dai_raised,
          'start_time': auction.start_time, 'end_time': self.current_time,
          'duration': self.current_time - auction.start_time,
          'total_collateral_sold': total_collateral_sold,  # Store this new field
          'oracle_price_at_end': self.oracle.current_price,  # Also store oracle price at end
     })
     
     clipper = self.maker_state.clippers.get(auction.ilk)
     if clipper and auction.id in clipper.active_auctions:
         if auction.id in clipper.active_auctions: del clipper.active_auctions[auction.id]

    # --- Model Step Logic using Stages ---
    def env_update(self):
        _ = self.oracle.update_price()
        for clipper in self.maker_state.clippers.values():
             clipper.check_stale_auctions(self.current_time)

    def monitor_trigger(self): pass
    def bid_submit(self): pass
    def mev_scan_submit(self): pass
    def block_process(self): pass
    def state_update(self): pass

    def step(self):
        """Advance the model by one step."""
        self.env_update(); self.agents.do("env_update")
        self.agents.do("monitor_trigger")
        self.agents.do("bid_submit")
        mempool_snapshot = self.mempool.view_transactions()
        for agent in self.agents:
             if hasattr(agent, "mev_scan_submit"): agent.mev_scan_submit()
        transactions_in_block = self.mempool.get_transactions_for_block()
        if transactions_in_block: _ = self.block_producer.process_transactions(transactions_in_block)
        self.agents.do("block_process")
        self.agents.do("state_update")
        self.current_time += self.time_step_duration
        self.datacollector.collect(self)
        if self.steps >= SIMULATION_STEPS: self.running = False

# --- Main Execution Block (Corrected for NameError and Multiple Runs) ---
if __name__ == "__main__":
    # --- Make sure pandas is imported ---
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas library is required. Please install it (`pip install pandas`)")
        exit()
    # --- End pandas import check ---

    start_run_time = time.time()

    # --- Simulation Parameters ---
    NUM_RUNS = 5 # Number of runs per scenario (Increase for real research)
    scenarios = ['Baseline', 'TE', 'VDF']
    # Using only the default VDF_DELAY_T for this example
    vdf_delays_to_test = [1.0, 5.0, 30.0, 300.0] # Test 1s, 5s, 30s, 5min(300s)

    # --- Data Storage ---
    all_model_results = []
    all_keeper_final_profits = []

    # --- Main Loop for Multiple Runs ---
    total_simulations = NUM_RUNS * (len(scenarios) -1 + len(vdf_delays_to_test))
    current_simulation = 0

    for run in range(NUM_RUNS):
        print(f"\n===== Starting Run {run + 1} / {NUM_RUNS} =====")
        run_seed_base = random.randint(10000 * run, 10000 * (run + 1) -1)

        for scenario in scenarios:
            # Handle VDF delay variation if testing multiple delays
            if scenario == 'VDF':
                for delay in vdf_delays_to_test:
                    current_simulation += 1
                    print(f"\n--- Running Scenario: {scenario} (T={delay}s) --- (Sim {current_simulation}/{total_simulations})")
                    globals()['VDF_DELAY_T'] = delay
                    model_seed = run_seed_base + scenarios.index(scenario) * 100 + vdf_delays_to_test.index(delay)
                    scenario_label = f"{scenario}_T{delay}"

                    # Initialize and run model
                    model = MakerLiquidationModel(N_BORROWERS, N_KEEPERS, N_MEV_SEARCHERS, ofp_mode=scenario, seed=model_seed)
                    model.running = True
                    step_count = 0
                    while model.running:
                        try:
                            model.step()
                        except Exception as e:
                            print(f"\n!!!!! ERROR during {scenario_label} Run {run+1} Step {model.steps} !!!!!")
                            print(f"Error: {e}")
                            import traceback
                            traceback.print_exc()
                            model.running = False
                        step_count += 1
                        if not model.running : break
                        # Optional: Reduce print frequency
                        # if step_count % 500 == 0:
                        #      print(f"  ... Step {model.steps} ({scenario_label} Run {run+1}) ...")

                    # Collect Results
                    model_df = model.datacollector.get_model_vars_dataframe()
                    agent_df = model.datacollector.get_agent_vars_dataframe()

                    if not model_df.empty:
                         final_vars = model_df.iloc[-1].to_dict()
                         final_vars['run'] = run + 1
                         final_vars['scenario'] = scenario_label
                         all_model_results.append(final_vars)
                    else: print(f"Warning: Empty model_df for {scenario_label} Run {run+1}")

                    if not agent_df.empty:
                        final_step = agent_df.index.get_level_values('Step').max()
                        try: # Added try-except for potential empty index
                            final_agent_data = agent_df.xs(final_step, level='Step')
                            keeper_profits = final_agent_data[final_agent_data['AgentType'] == 'KeeperAgent']['Profit'].tolist()
                            all_keeper_final_profits.append({
                                'run': run + 1, 'scenario': scenario_label, 'keeper_profits': keeper_profits
                             })
                        except KeyError:
                             print(f"Warning: Could not access final step {final_step} in agent_df for {scenario_label} Run {run+1}")
                    else: print(f"Warning: Empty agent_df for {scenario_label} Run {run+1}")

            else: # Handle Baseline and TE scenarios
                current_simulation += 1
                print(f"\n--- Running Scenario: {scenario} --- (Sim {current_simulation}/{total_simulations})")
                model_seed = run_seed_base + scenarios.index(scenario)
                scenario_label = scenario

                # Initialize and run model
                model = MakerLiquidationModel(N_BORROWERS, N_KEEPERS, N_MEV_SEARCHERS, ofp_mode=scenario, seed=model_seed)
                model.running = True
                step_count = 0
                while model.running:
                    try:
                        model.step()
                    except Exception as e:
                        print(f"\n!!!!! ERROR during {scenario_label} Run {run+1} Step {model.steps} !!!!!")
                        print(f"Error: {e}")
                        import traceback
                        traceback.print_exc()
                        model.running = False
                    step_count += 1
                    if not model.running : break
                    # Optional: Reduce print frequency
                    # if step_count % 500 == 0:
                    #      print(f"  ... Step {model.steps} ({scenario_label} Run {run+1}) ...")

                # Collect Results
                model_df = model.datacollector.get_model_vars_dataframe()
                agent_df = model.datacollector.get_agent_vars_dataframe()

                if not model_df.empty:
                     final_vars = model_df.iloc[-1].to_dict()
                     final_vars['run'] = run + 1
                     final_vars['scenario'] = scenario_label
                     all_model_results.append(final_vars)
                else: print(f"Warning: Empty model_df for {scenario_label} Run {run+1}")

                if not agent_df.empty:
                    final_step = agent_df.index.get_level_values('Step').max()
                    try: # Added try-except
                        final_agent_data = agent_df.xs(final_step, level='Step')
                        keeper_profits = final_agent_data[final_agent_data['AgentType'] == 'KeeperAgent']['Profit'].tolist()
                        all_keeper_final_profits.append({
                            'run': run + 1, 'scenario': scenario_label, 'keeper_profits': keeper_profits
                         })
                    except KeyError:
                         print(f"Warning: Could not access final step {final_step} in agent_df for {scenario_label} Run {run+1}")
                else: print(f"Warning: Empty agent_df for {scenario_label} Run {run+1}")

    # --- Restore default VDF_DELAY_T if modified ---
    globals()['VDF_DELAY_T'] = vdf_delays_to_test[0] # Restore default

    end_run_time = time.time()
    print(f"\nTotal Simulation Run Time ({NUM_RUNS} runs): {end_run_time - start_run_time:.2f} seconds")

    # --- Aggregate and Analyze Results ---
    if not all_model_results:
        print("\nNo simulation results collected. Exiting analysis.")
    else:
        results_df = pd.DataFrame(all_model_results)

        # --- Define Helper Functions WITHIN scope ---
        def format_val(val, width=15):
            if isinstance(val, (int, float)):
                if isinstance(val, int): s = f"{val:,}"
                elif abs(val) > 1e7 or (abs(val) < 1e-3 and val != 0): s = f"{val:.2e}"
                else: s = f"{val:,.2f}"
            else: s = str(val)
            return f"{s:<{width}}"

        def format_agg(mean_val, std_val, width):
            if pd.isna(mean_val): return format_val("N/A", width)
            mean_str = format_val(mean_val, width=0)
            std_str = format_val(std_val, width=0) if pd.notna(std_val) else "0.00"
            combined = f"{mean_str} ± {std_str}"
            return format_val(combined, width)
        # --- End Helper Functions ---


        # Calculate Mean and Std Dev for key metrics
        metrics_to_aggregate = [
            "MEVProfit", "KeeperProfit", "BadDebt", "TotalValueLiquidated",
            "CompletedAuctions", "FailedAuctions", "AvgAuctionDuration","AvgPriceEfficiency"
        ]
        aggregation = results_df.groupby('scenario')[metrics_to_aggregate].agg(['mean', 'std'])

        # Calculate Keeper Profit Distribution Stats
        profit_stats = {}
        scenario_profits_all_runs = {}
        for record in all_keeper_final_profits:
            scenario = record['scenario']
            if scenario not in scenario_profits_all_runs: scenario_profits_all_runs[scenario] = []
            if record['keeper_profits']: scenario_profits_all_runs[scenario].extend(record['keeper_profits'])

        processed_scenarios = results_df['scenario'].unique()
        for scenario in processed_scenarios:
             all_profits = scenario_profits_all_runs.get(scenario, [])
             if all_profits:
                 profit_series = pd.Series(all_profits)
                 profit_stats[scenario] = {
                     'Mean Profit': profit_series.mean(),
                     'Median Profit': profit_series.median(),
                     'Std Dev Profit': profit_series.std(),
                     'Total Keepers': len(all_profits)
                 }
             else:
                 profit_stats[scenario] = {'Mean Profit': 0, 'Median Profit': 0, 'Std Dev Profit': 0, 'Total Keepers': 0}


        # --- Print Aggregated Results Table ---
        print("\n--- Aggregated Results Summary (Mean +/- Std Dev) ---")
        header_width = 18
        final_scenarios = aggregation.index.tolist() # Scenarios that actually have results

        # Use the format_val defined *above* this printing section
        header = f"{'Metric':<25} | " + " | ".join([format_val(s, header_width) for s in final_scenarios])
        print(header)
        print("-" * len(header))

        # Print aggregated model metrics (using format_agg which uses format_val)
        for metric in metrics_to_aggregate:
            if metric in aggregation.columns.get_level_values(0):
                 values_str = " | ".join([format_agg(aggregation.loc[s, (metric, 'mean')], aggregation.loc[s, (metric, 'std')], header_width) for s in final_scenarios])
                 print(f"{metric:<25} | {values_str}")
            else:
                 print(f"{metric:<25} | " + " | ".join([format_val("N/A", header_width)] * len(final_scenarios)))

        # Print Keeper Profit Distribution Stats (using format_val)
        print("-" * len(header))
        print(f"{'Keeper Profit Dist.':<25} | " + " | ".join([format_val("", header_width)] * len(final_scenarios)))
        for stat_name, display_name in [('Mean Profit', 'Mean Keeper Profit'), ('Median Profit', 'Median Keeper Profit'), ('Std Dev Profit', 'Std Dev Keeper Profit')]:
             values_str = " | ".join([format_val(profit_stats.get(s, {}).get(stat_name, 'N/A'), header_width) for s in final_scenarios])
             print(f"{display_name:<25} | {values_str}")

        # Print OFP Overhead Proxy (using format_val)
        print("-" * len(header))
        overhead_metric = "OFPOverheadProxy"
        overhead_values = {}
        if overhead_metric in results_df.columns:
             first_run_data = results_df.drop_duplicates(subset=['scenario'], keep='first').set_index('scenario')
             for s in final_scenarios:
                 if s in first_run_data.index: overhead_values[s] = first_run_data.loc[s, overhead_metric]
                 else: overhead_values[s] = "N/A"
        values_str = " | ".join([format_val(overhead_values.get(s, "N/A"), header_width) for s in final_scenarios])
        print(f"{overhead_metric:<25} | {values_str}")

        # Print metrics that still need data modification (using format_val)
        # print("-" * len(header))
        # for metric in ["AvgPriceEfficiency"]:
        #      values_str = " | ".join([format_val("N/A (Needs Data Mod)", header_width) for _ in final_scenarios])
        #      print(f"{metric:<25} | {values_str}")

        print("\nAggregated Analysis Complete.")
# simulation_project/processing/block_producer.py

from .transaction import Transaction
from ..agents.mev_agent import MEVSearcherAgent # For type checking
from .. import config # For config.VERBOSE_LOGGING

class BlockProducer:
    """Simulates block production and transaction processing logic."""
    def __init__(self, model):
        self.model = model

    def process_transactions(self, transactions: list[Transaction]) -> list[Transaction]:
        executed_txs_info = []

        if not transactions:
            return executed_txs_info

        # This is a high-level summary, keep it unconditional
        print(f"  [BlockProducer] Processing {len(transactions)} transactions for block at time {self.model.current_time:.0f}...")

        for tx in transactions:
            if not isinstance(tx, Transaction):
                # This is an unexpected state, keep it unconditional
                print(f"    Warning: Found non-Transaction object in processing queue: {tx}. Skipping.")
                continue

            if config.VERBOSE_LOGGING:
                print(f"    Processing Tx: Type={tx.tx_type}, Sender={tx.sender_id}, Gas={tx.gas_price:.1f}, Submitted={tx.submission_time:.0f}")

            success = False
            details = "Processing Error: Unknown"
            gas_cost_proxy_for_sender = 0
            current_processing_time = self.model.current_time

            agent = None
            if isinstance(tx.sender_id, int):
                try:
                    agent = next((a for a in self.model.agents if a.unique_id == tx.sender_id), None)
                    if agent is None:
                        # This is a critical failure for the transaction, keep unconditional
                        print(f"      >>>>>> Warning: Agent ID {tx.sender_id} (type: {type(tx.sender_id)}) not found via iteration in model.agents. Skipping transaction. <<<<<<")
                        available_agents_info = []
                        try:
                            for ag_debug in self.model.agents:
                                available_agents_info.append({'id': ag_debug.unique_id, 'type': ag_debug.__class__.__name__})
                            available_agents_info_sorted = sorted(available_agents_info, key=lambda x: x['id'])
                            print(f"         Available agent unique_ids at this time ({len(available_agents_info_sorted)} total): {available_agents_info_sorted}")
                        except Exception as e_debug:
                            print(f"         Error trying to list available agents for debugging: {e_debug}")
                        
                        tx.status = "Failed"
                        tx.execution_time = current_processing_time
                        details = f"Sender agent {tx.sender_id} not found."
                        executed_txs_info.append(tx)
                        print(f"      Tx Result: {tx.status}. Details: {details}") # Keep this result print
                        continue
                except Exception as e_lookup:
                    # Error during lookup is critical, keep unconditional
                    print(f"      ERROR during iterative agent lookup for ID {tx.sender_id}: {e_lookup}. Skipping transaction.")
                    tx.status = "Failed"
                    tx.execution_time = current_processing_time
                    details = f"Error looking up agent {tx.sender_id}."
                    executed_txs_info.append(tx)
                    print(f"      Tx Result: {tx.status}. Details: {details}") # Keep this result print
                    continue
            else:
                # Invalid sender ID type is critical, keep unconditional
                print(f"      Warning: Transaction sender_id '{tx.sender_id}' (type: {type(tx.sender_id)}) is not an integer. Skipping transaction.")
                tx.status = "Failed"
                tx.execution_time = current_processing_time
                details = f"Invalid sender_id type for agent {tx.sender_id}."
                executed_txs_info.append(tx)
                print(f"      Tx Result: {tx.status}. Details: {details}") # Keep this result print
                continue
            
            if config.VERBOSE_LOGGING:
                if self.model.ofp_mode == 'TE' and tx.tx_type == 'take' and tx.is_encrypted:
                    print(f"      TE Decryption: Content of 'take' tx from {tx.sender_id} revealed after {config.TE_DECRYPTION_LATENCY:.2f}s (simulated).")
            if tx.is_encrypted and self.model.ofp_mode == 'TE' and tx.tx_type == 'take': # Ensure this logic is outside verbose for state change
                 tx.is_encrypted = False


            if self.model.ofp_mode == 'VDF' and tx.tx_type == 'take':
                if tx.vdf_reveal_time < 0:
                    tx.vdf_reveal_time = tx.submission_time + config.VDF_DELAY_T
                if current_processing_time < tx.vdf_reveal_time:
                    if config.VERBOSE_LOGGING:
                        print(f"      VDF Delay Pending for 'take' Tx from {tx.sender_id}. "
                              f"Reveal Time: {tx.vdf_reveal_time:.0f} > Current Time: {current_processing_time:.0f}. Re-queuing.")
                    self.model.mempool.add_transaction(tx)
                    continue
                gas_cost_proxy_for_sender += config.VDF_VERIFICATION_GAS_COST
                if config.VERBOSE_LOGGING:
                    print(f"      VDF Delay Met for 'take' Tx from {tx.sender_id}. Added VDF Verification Gas: {config.VDF_VERIFICATION_GAS_COST}")

            maker_state = self.model.maker_state
            if tx.tx_type == 'bark':
                vault_id = tx.params.get('vault_id')
                ilk = tx.params.get('ilk')
                tab_to_raise = tx.params.get('tab')
                collateral_to_seize = tx.params.get('collateral_to_seize')
                vault = maker_state.get_vault(vault_id)
                if vault and vault.status == "Active":
                    clipper = maker_state.get_clipper(ilk)
                    if clipper:
                        vault.status = "Liquidating"
                        auction_id, tip_amount, chip_factor = clipper.kick(
                            vault_urn=vault_id, tab_debt=tab_to_raise,
                            lot_collateral=collateral_to_seize, current_time=current_processing_time
                        )
                        if auction_id is not None:
                            if hasattr(agent, 'record_incentive'):
                                agent.record_incentive(tip_amount, chip_factor, tab_to_raise)
                            success = True
                            details = f"Auction {auction_id} kicked for Vault {vault_id}."
                            gas_cost_proxy_for_sender += 500_000
                        else:
                            details = f"Clipper kick failed for Vault {vault_id} (e.g., zero oracle price)."
                            vault.status = "Active"
                    else:
                        details = f"Clipper for ilk {ilk} not found for Vault {vault_id}."
                elif not vault:
                    details = f"Vault {vault_id} not found for bark."
                else:
                    details = f"Vault {vault_id} status is {vault.status}, bark skipped."
            elif tx.tx_type == 'take':
                auction_id = tx.params.get('auction_id')
                ilk = tx.params.get('ilk')
                amt_collateral = tx.params.get('amt_collateral')
                max_price = tx.params.get('max_price')
                bidder_id = tx.sender_id
                clipper = maker_state.get_clipper(ilk)
                if clipper:
                    bidder_balance = maker_state.get_agent_balance(bidder_id)
                    bid_success, collateral_received, dai_cost, take_msg = clipper.take(
                        auction_id=auction_id, bidder_id=bidder_id, amt_collateral_to_buy=amt_collateral,
                        max_price_per_unit=max_price, bidder_dai_balance=bidder_balance,
                        current_time=current_processing_time
                    )
                    details = take_msg
                    if bid_success:
                        success = True
                        maker_state.update_agent_balance(bidder_id, -dai_cost)
                        if hasattr(agent, 'update_state_post_take'):
                            agent.update_state_post_take(ilk, collateral_received, dai_cost)
                        gas_cost_proxy_for_sender += 300_000
                    elif not bid_success and self.model.ofp_mode in ['TE', 'VDF'] and "Price too high" in details:
                        if config.VERBOSE_LOGGING: # This specific info can be verbose
                            print(f"      (Info {self.model.ofp_mode}): 'take' from {bidder_id} on {auction_id} failed post-delay due to price. Msg: {details}")
                else:
                    details = f"Clipper for ilk {ilk} (auction {auction_id}) not found for take."
            else:
                details = f"Unknown transaction type: {tx.tx_type}"

            tx.status = "Executed" if success else "Failed"
            tx.execution_time = current_processing_time
            executed_txs_info.append(tx)
            if hasattr(agent, 'pay_gas'):
                agent.pay_gas(gas_cost_proxy_for_sender)
            
            # Keep Tx Result summary unconditional as it's important
            print(f"      Tx Result: {tx.status}. Details: {details}")

        return executed_txs_info

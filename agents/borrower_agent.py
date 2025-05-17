# simulation_project/agents/borrower_agent.py

import mesa
from ..core.vault import Vault # Assuming Vault is in core directory
from .. import config # For borrower behavior parameters

class BorrowerAgent(mesa.Agent):
    """
    Represents a user with a Vault in the MakerDAO system.
    Can now attempt to save their vault if it becomes at risk by adding collateral or repaying debt.
    """
    def __init__(self, model: mesa.Model, vault: Vault, external_dai: float, external_collateral: float):
        super().__init__(model)
        self.vault = vault
        self.external_dai_holdings = external_dai  # Simulated DAI holdings outside the vault
        self.external_collateral_holdings = external_collateral # Simulated collateral holdings outside the vault

    def attempt_to_save_vault(self):
        """
        If CR is below a personal threshold (but above liquidation ratio),
        the borrower may attempt to add collateral or repay debt.
        This is a simplified model of saving behavior.
        """
        if self.vault.status != "Active": # Can only save active vaults
            return

        current_price = self.model.oracle.current_price
        if current_price <= 1e-9: # Cannot make decisions if price is zero
            return
            
        current_cr = self.vault.get_collateralization_ratio(current_price)
        
        # Personal threshold for saving: LR + a buffer (e.g., LR + 0.10)
        save_threshold_cr = config.MIN_LIQUIDATION_RATIO + config.BORROWER_SAVE_CR_THRESHOLD_ADD

        # Only attempt to save if CR is below the save threshold AND above the absolute liquidation ratio
        if config.MIN_LIQUIDATION_RATIO <= current_cr < save_threshold_cr:
            if self.model.random.random() < config.BORROWER_SAVE_ATTEMPT_PROB:
                # Target CR after saving: LR + a larger buffer (e.g., LR + 0.25)
                target_cr_after_save = config.MIN_LIQUIDATION_RATIO + config.BORROWER_RECAP_TARGET_CR_ADD
                action_taken = False

                # Priority 1: Try to add collateral (if it's the more CR-effective action or has holdings)
                if self.external_collateral_holdings > 1e-6:
                    # Calculate how much collateral is needed to reach target_cr_after_save
                    # TargetCollateralValue = TargetCR * Debt  => TargetCollateralAmount = (TargetCR * Debt) / Price
                    # CollateralToAdd = TargetCollateralAmount - CurrentCollateralAmount
                    required_collateral_value = target_cr_after_save * self.vault.debt_amount
                    target_collateral_amount = required_collateral_value / current_price
                    collateral_to_add = max(0, target_collateral_amount - self.vault.collateral_amount)
                    
                    actual_collateral_added = min(collateral_to_add, self.external_collateral_holdings)
                    
                    if actual_collateral_added > 1e-6:
                        self.vault.collateral_amount += actual_collateral_added
                        self.external_collateral_holdings -= actual_collateral_added
                        self.model.total_collateral_added_by_borrowers += actual_collateral_added
                        action_taken = True
                        if config.VERBOSE_LOGGING:
                            new_cr = self.vault.get_collateralization_ratio(current_price)
                            print(f"    --- Borrower {self.unique_id} (Vault {self.vault.owner_id}) ADDED {actual_collateral_added:.2f} COLL. "
                                  f"Old CR: {current_cr:.3f}, New CR: {new_cr:.3f} ---")
                
                # Priority 2: Try to repay debt (if no collateral added or not enough collateral, or if it's preferred)
                # Can attempt this even if collateral was added, if CR is still not at target.
                current_cr_after_coll_add = self.vault.get_collateralization_ratio(current_price) # Re-check CR
                if not action_taken or (action_taken and current_cr_after_coll_add < target_cr_after_save):
                    if self.external_dai_holdings > 1e-6:
                        # Calculate how much debt needs to be repaid to reach target_cr_after_save
                        # CurrentCollateralValue / (CurrentDebt - DebtToRepay) = TargetCR
                        # DebtToRepay = CurrentDebt - (CurrentCollateralValue / TargetCR)
                        current_collateral_value = self.vault.collateral_amount * current_price
                        target_debt_amount = current_collateral_value / target_cr_after_save if target_cr_after_save > 1e-9 else float('inf')
                        debt_to_repay = max(0, self.vault.debt_amount - target_debt_amount)
                        
                        actual_dai_repaid = min(debt_to_repay, self.external_dai_holdings, self.vault.debt_amount)

                        if actual_dai_repaid > 1e-6:
                            self.vault.debt_amount -= actual_dai_repaid
                            self.external_dai_holdings -= actual_dai_repaid
                            self.model.total_debt_repaid_by_borrowers += actual_dai_repaid
                            action_taken = True
                            if config.VERBOSE_LOGGING:
                                new_cr = self.vault.get_collateralization_ratio(current_price)
                                print(f"    --- Borrower {self.unique_id} (Vault {self.vault.owner_id}) REPAID {actual_dai_repaid:.2f} DAI. "
                                      f"Old CR (before this action): {current_cr_after_coll_add:.3f}, New CR: {new_cr:.3f} ---")
    
    # --- Stage Methods for model.agents.do("stage_name") ---
    def env_update(self):
        """Called when environment variables (like price) might change."""
        pass # Borrower might reassess vault health here if needed, but saving is in borrower_actions

    def borrower_actions(self): # New stage method
        """Borrower attempts to save their vault if at risk."""
        self.attempt_to_save_vault()

    def monitor_trigger(self):
        """Borrowers don't monitor to trigger liquidations on others."""
        pass 

    def bid_submit(self):
        """Borrowers don't typically bid on auctions in this model."""
        pass 

    def mev_scan_submit(self):
        """Borrowers are not MEV searchers."""
        pass 

    def block_process(self):
        """Called after a block of transactions has been processed."""
        pass 

    def state_update(self):
        """Called for agents to update their internal state at the end of a step."""
        pass

    def __repr__(self):
        return f"BorrowerAgent(id={self.unique_id}, vault_owner='{self.vault.owner_id}')"


# simulation_project/core/vault.py

import math

class Vault:
    """Represents a Borrower's Vault."""
    def __init__(self, owner_id: str, initial_collateral: float, initial_debt: float, ilk: str = 'ETH-A'):
        self.owner_id = owner_id
        self.collateral_amount = float(initial_collateral)
        self.debt_amount = float(initial_debt)
        self.status = "Active"  # Possible statuses: "Active", "Liquidating"
        self.ilk = ilk # Collateral type identifier

    def get_collateralization_ratio(self, current_price: float) -> float:
        """
        Calculates the current collateralization ratio (CR) of the vault.
        CR = (Collateral Amount * Current Price) / Debt Amount
        Returns float('inf') if debt is zero or price is zero/negative.
        """
        if current_price <= 0:
            return float('inf')
        if self.debt_amount <= 1e-9:  # Consider very small debt as effectively zero
            return float('inf')
        
        collateral_value = self.collateral_amount * current_price
        return collateral_value / self.debt_amount

    def __repr__(self):
        return (f"Vault(owner='{self.owner_id}', coll={self.collateral_amount:.2f}, "
                f"debt={self.debt_amount:.2f}, status='{self.status}', ilk='{self.ilk}')")


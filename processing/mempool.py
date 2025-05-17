# simulation_project/processing/mempool.py

from .transaction import Transaction # Assuming Transaction is in the same directory

class Mempool:
    """Simulates the transaction mempool."""
    def __init__(self):
        self.pending_txs = [] # List to store Transaction objects

    def add_transaction(self, tx: Transaction):
        """Adds a transaction to the mempool."""
        if not isinstance(tx, Transaction):
            print(f"Warning: Attempted to add non-Transaction object to mempool: {tx}")
            return
        self.pending_txs.append(tx)

    def get_transactions_for_block(self, max_txs: int = -1) -> list[Transaction]:
        """
        Retrieves transactions for the next block, sorted by gas price (descending).
        Clears the retrieved transactions from the mempool.
        
        Args:
            max_txs (int): Maximum number of transactions to include in the block. 
                           -1 means no limit (all pending transactions).
                           
        Returns:
            list[Transaction]: A list of transactions selected for the block.
        """
        try:
            # Sort by gas_price (descending) due to __lt__ in Transaction class
            # For multiple criteria (e.g., gas_price then submission_time), use a lambda key
            sorted_txs = sorted(self.pending_txs, reverse=True) # reverse=True because __lt__ makes it high-to-low
        except AttributeError as e:
            print(f"Error sorting mempool - potentially contains non-Transaction object: {e}")
            print(f"Mempool contents: {self.pending_txs}")
            # Filter out non-Transaction objects if any, then sort
            valid_txs = [tx for tx in self.pending_txs if isinstance(tx, Transaction)]
            sorted_txs = sorted(valid_txs, reverse=True)
        
        if max_txs > 0:
            selected_txs = sorted_txs[:max_txs]
            self.pending_txs = sorted_txs[max_txs:] # Keep remaining transactions
        else:
            selected_txs = sorted_txs
            self.pending_txs = [] # All transactions taken
            
        return selected_txs

    def view_transactions(self) -> list[Transaction]:
        """Returns a read-only view of the current pending transactions (copy)."""
        return list(self.pending_txs)

    def __len__(self):
        return len(self.pending_txs)


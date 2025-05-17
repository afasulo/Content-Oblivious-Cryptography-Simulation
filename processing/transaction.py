# simulation_project/processing/transaction.py

class Transaction:
    """ 
    Represents a transaction in the mempool or being processed. 
    This is a data class to hold transaction details.
    """
    def __init__(self, tx_type: str, sender_id, params: dict, 
                 gas_price: float, submission_time: float):
        self.tx_type = tx_type  # e.g., 'bark', 'take'
        self.sender_id = sender_id # unique_id of the agent submitting the transaction
        self.params = params # Dictionary of transaction-specific parameters
        self.gas_price = float(gas_price)
        self.submission_time = float(submission_time)
        
        self.status = "Pending"  # Possible statuses: "Pending", "Executed", "Failed"
        self.execution_time = -1.0 # Simulation time when executed or failed
        
        # OFP-specific attributes
        self.is_encrypted = False # True if this 'take' tx is encrypted (TE mode)
        self.vdf_reveal_time = -1.0 # Time when VDF output is revealed (VDF mode for 'take')

    def __lt__(self, other):
        """
        Comparison for sorting transactions, primarily by gas price (descending).
        Higher gas price means higher priority.
        """
        if not isinstance(other, Transaction):
            return NotImplemented
        # Higher gas price comes first
        return self.gas_price > other.gas_price

    def __repr__(self):
        return (f"Tx(type='{self.tx_type}', sender={self.sender_id}, gas={self.gas_price:.1f}, "
                f"status='{self.status}', submit_time={self.submission_time:.0f})")


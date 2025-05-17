# simulation_project/core/auction.py

import math
from .. import config

class Auction:
    """Represents a single Liquidation 2.0 Dutch Auction."""
    def __init__(self, auction_id: str, ilk: str, lot_collateral: float, tab_debt: float, 
                 start_time: float, start_price: float, pvt_curve_params: dict, 
                 duration_limit: int, reset_threshold_factor: float, model):
        self.id = auction_id
        self.ilk = ilk
        self.initial_lot = float(lot_collateral)  # Total collateral in the auction
        self.remaining_lot = float(lot_collateral) # Collateral yet to be sold
        self.tab = float(tab_debt) # Total DAI that needs to be raised (debt + penalty)
        self.dai_raised = 0.0    # DAI raised so far
        self.start_time = float(start_time) # Simulation time when auction started
        self.start_price = float(start_price) # Initial price per unit of collateral
        
        # PVT curve parameters (e.g., type, step, cut, tau)
        self.pvt_curve_params = pvt_curve_params 
        self.duration_limit = int(duration_limit) # Max duration of the auction (tail)
        self.reset_threshold_factor = float(reset_threshold_factor) # Price factor for cusp (cusp)
        
        self.status = "Active"  # Possible statuses: "Active", "Completed", "Failed"
        self.bids = [] # List to store bid details
        self.model = model # Mesa model instance for logging or accessing global state if needed

    def get_elapsed_time(self, current_time: float) -> float:
        """Calculates the time elapsed since the auction started."""
        return max(0, float(current_time) - self.start_time)

    def get_current_price(self, current_time: float) -> float:
        """
        Calculates the current price per unit of collateral based on the PVT curve.
        Returns 0 if auction is not active.
        """
        if self.status != "Active":
            return 0.0
        
        elapsed_time = self.get_elapsed_time(current_time)
        curve_type = self.pvt_curve_params.get('type', config.PVT_CURVE_TYPE)

        if curve_type == 'StairstepExponentialDecrease':
            step_duration = self.pvt_curve_params.get('step', config.PVT_STEP_DURATION)
            step_decay = self.pvt_curve_params.get('cut', config.PVT_STEP_DECAY)
            if step_duration <= 0: # Avoid division by zero or infinite loops
                return 0.0 
            num_steps_taken = math.floor(elapsed_time / step_duration)
            current_price = self.start_price * (step_decay ** num_steps_taken)
        elif curve_type == 'LinearDecrease':
            tau = self.pvt_curve_params.get('tau', config.PVT_LINEAR_TAU)
            if tau <= 0: # Avoid division by zero
                return 0.0
            price_drop_rate = self.start_price / tau
            current_price = self.start_price - (price_drop_rate * elapsed_time)
        else:
            # Fallback or error for unknown curve type
            print(f"Warning: Unknown PVT curve type: {curve_type} for Auction {self.id}. Using start price.")
            current_price = self.start_price
            
        return max(0, current_price) # Price cannot be negative

    def __repr__(self):
        return (f"Auction(id='{self.id}', ilk='{self.ilk}', lot={self.initial_lot:.2f}, "
                f"tab={self.tab:.2f}, status='{self.status}', raised={self.dai_raised:.2f}, "
                f"rem_lot={self.remaining_lot:.2f})")


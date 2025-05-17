# simulation_project/core/oracle.py

import math
from .. import config # Use relative import for config

class Oracle:
    """Represents the price feed."""
    def __init__(self, initial_price: float, volatility: float, time_step_seconds: int, model):
        self.model = model # model is a Mesa model instance
        self.current_price = float(initial_price)
        self.volatility = float(volatility)
        self.dt = float(time_step_seconds) / (60 * 60 * 24 * 365)
        self.price_drift = float(config.PRICE_DRIFT)
        self.shock_applied_this_step = False # To ensure shock is applied only once if called multiple times in a step

    def update_price(self, current_model_step: int, shock_trigger_step: int, shock_price_factor: float) -> float:
        """
        Updates the oracle price using a Geometric Brownian Motion (GBM) model.
        Includes an optional market shock at a specific step.

        Args:
            current_model_step (int): The current step number of the model.
            shock_trigger_step (int): The step at which the market shock should occur.
            shock_price_factor (float): The factor by which the price changes during the shock
                                     (e.g., 0.7 for a 30% drop, 1.3 for a 30% rise).

        Returns:
            float: The new current price.
        """
        # Apply GBM price change
        mu = self.price_drift
        dW = self.model.random.normalvariate(0, math.sqrt(self.dt))
        effective_volatility = min(self.volatility, 1.0)
        dS = self.current_price * (mu * self.dt + effective_volatility * dW)
        self.current_price += dS
        self.current_price = max(self.current_price, 0.01) # Floor price

        # Apply market shock if conditions are met and shock hasn't been applied this step
        if shock_trigger_step != -1 and current_model_step == shock_trigger_step and not self.shock_applied_this_step:
            price_before_shock = self.current_price
            self.current_price *= shock_price_factor # Apply multiplicative shock
            self.current_price = max(self.current_price, 0.01) # Ensure price doesn't go negative
            self.shock_applied_this_step = True # Mark shock as applied for this step
            if config.VERBOSE_LOGGING:
                print(f"    !!!! MARKET SHOCK APPLIED at step {current_model_step} !!!! "
                      f"Price before: {price_before_shock:.2f}, Factor: {shock_price_factor:.2f}, "
                      f"Price after: {self.current_price:.2f}")
        elif current_model_step != shock_trigger_step:
            self.shock_applied_this_step = False # Reset for next potential shock step

        return self.current_price

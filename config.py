# simulation_project/config.py

"""
Global constants and parameters for the MakerDAO Liquidation Simulation.
ADJUSTED FOR MORE REALISTIC PARAMETERS AND INCREASED RUNS.
"""

# --- Logging Configuration ---
VERBOSE_LOGGING = False  # Set to False to reduce console output during simulation steps


# Market & Simulation Setup
INITIAL_ETH_PRICE = 3200.0
PRICE_VOLATILITY = 0.75     # INCREASED: Higher volatility (e.g., 75% annualized)
PRICE_DRIFT = -0.03         # SLIGHT NEGATIVE DRIFT: To gently push CRs down.
N_BORROWERS = 75            # INCREASED: More borrowers for more potential events.
N_KEEPERS = 7               # INCREASED: More keepers for competition.
N_MEV_SEARCHERS = 2
SIMULATION_STEPS = 1440 # Run for 2 days to allow more dynamics to unfold.
TIME_STEP_DURATION_SECONDS = 60

# MakerDAO Parameters (Reflecting common ETH-A type settings)
MIN_LIQUIDATION_RATIO = 1.50 # Common Liquidation Ratio for ETH-A.
LIQUIDATION_PENALTY = 0.13   # Typical 'chop' (penalty) in MakerDAO.
PRICE_BUFFER = 1.20          # `buf` in Clipper: auction starts at oracle_price * 1.20. Standard.
AUCTION_DURATION_LIMIT = 2400 # `tail` in Clipper: max auction duration (30 minutes).
                              # This is relatively short for a full Dutch auction `tail`,
                              # implying focus on faster auction resolutions or reliance on `cusp`.
AUCTION_RESET_THRESHOLD = 0.60 # `cusp` in Clipper: price factor below which auction might reset/fail early.
KEEPER_INCENTIVE_TIP = 100   # `tip` in DAI for kicking an auction. Plausible flat incentive.
KEEPER_INCENTIVE_CHIP = 0.02 # `chip` percentage of collateral value as incentive (2%). Plausible.

# PVT (Price-Volume-Time) Curve Parameters for Dutch Auctions
PVT_CURVE_TYPE = 'StairstepExponentialDecrease' # Standard for MakerDAO Dutch Auctions.
PVT_STEP_DURATION = 45  # `step` for StairstepExponentialDecrease: duration of each price step in seconds. Common.
PVT_STEP_DECAY = 0.985  # `cut` for StairstepExponentialDecrease: price multiplier per step (1% drop).
PVT_LINEAR_TAU = 3600   # `tau` for LinearDecrease (not used if PVT_CURVE_TYPE is Stairstep).

# OFP Parameters
TE_DECRYPTION_LATENCY = 0.2  # Latency in seconds for Threshold Encryption decryption (hypothetical).
VDF_DELAY_T = 5.0            # Base delay in seconds for VDF (can be varied in main_simulation).
VDF_VERIFICATION_GAS_COST = 2_000_000 # Gas cost for VDF verification (estimate).

# General Blockchain/Agent Parameters
BASE_GAS_PRICE = 20         # Gwei per gas unit (moderate recent average).
KEEPER_PROFIT_MARGIN = 0.02 # Desired profit margin for Keepers (3%). Subjective behavioral param.
MEV_FRONTUN_MARGIN = 0.01   # Desired profit margin for MEV searchers (1%). Subjective.
MEV_GAS_BOOST = 1.1         # Factor by which MEV searchers boost gas (10%). Subjective.

# Borrower Agent Parameters (NEW)
BORROWER_SAVE_ATTEMPT_PROB = 0.20 # INCREASED: Probability a borrower attempts to save their vault.
BORROWER_SAVE_CR_THRESHOLD_ADD = 0.15 # Borrower tries to save if CR < (MIN_LR + THIS_VALUE) (e.g., < 1.65 if LR=1.50)
BORROWER_RECAP_TARGET_CR_ADD = 0.30   # When saving, borrower aims for CR = (MIN_LR + THIS_VALUE) (e.g., 1.80 if LR=1.50)

# Precision (for potential fixed-point arithmetic, though current sim uses floats)
STABLECOIN_PRECISION = 10**18
COLLATERAL_PRECISION = 10**18

# --- Parameters for Main Simulation Loop (can be overridden in main_simulation.py) ---
# NUM_RUNS will be set in main_simulation.py
SCENARIOS = ['Baseline', 'TE', 'VDF', 'MarketShock_Drop'] 
VDF_DELAYS_TO_TEST = [1.0, 5.0, 30.0, 300.0] # VDF T values to test (seconds)

NUM_RUNS = 750
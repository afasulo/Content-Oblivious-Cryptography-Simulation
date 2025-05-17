# simulation_project/analysis/metrics.py

"""
Functions to calculate various metrics from the simulation model's output.
These are typically used as model reporters in Mesa's DataCollector.
"""

from ..agents.mev_agent import MEVSearcherAgent # For type checking
from ..agents.keeper_agent import KeeperAgent   # For type checking
from .. import config # For OFP parameters

# --- Metric Helper Functions for DataCollector ---

def get_total_bad_debt(model) -> float:
    """
    Calculates the total bad debt from all completed or failed auctions.
    Bad debt occurs if an auction fails to raise the full 'tab' amount.
    """
    total_shortfall = 0.0
    # The auction_result dictionaries are created in model.record_auction_end
    # They now use the key 'tab_debt' instead of 'tab'.
    for auction_result in model.completed_or_failed_auctions:
        # CORRECTED KEY: Changed from 'tab' to 'tab_debt'
        shortfall = max(0, auction_result.get('tab_debt', 0) - auction_result.get('dai_raised', 0))
        total_shortfall += shortfall
    return total_shortfall

def get_mev_profit(model) -> float:
    """Calculates the total profit accumulated by all MEVSearcherAgents."""
    total_profit = 0.0
    for agent in model.agents:
        if isinstance(agent, MEVSearcherAgent):
            total_profit += getattr(agent, 'total_profit', 0.0)
    return total_profit

def get_keeper_profit(model) -> float:
    """Calculates the total profit accumulated by all KeeperAgents."""
    total_profit = 0.0
    for agent in model.agents:
        if isinstance(agent, KeeperAgent):
            total_profit += getattr(agent, 'total_profit', 0.0)
    return total_profit

def get_average_auction_duration(model) -> float:
    """
    Calculates the average duration of COMPLETED auctions.
    Excludes auctions that failed due to staleness.
    """
    completed_auctions = [
        a_res for a_res in model.completed_or_failed_auctions
        if a_res['status'] == 'Completed'
    ]
    if not completed_auctions:
        return 0.0

    total_duration = sum(a_res.get('duration_seconds', 0) for a_res in completed_auctions) # Ensure key 'duration_seconds' is used
    return total_duration / len(completed_auctions) if len(completed_auctions) > 0 else 0.0

def get_average_price_efficiency(model) -> float:
    """
    Calculates a weighted average of price efficiency for COMPLETED auctions.
    Price efficiency = (Clearing Price) / (Oracle Price at Auction End)
    Weighted by the amount of collateral sold in each auction.
    """
    completed_auctions_with_sales = [
        a_res for a_res in model.completed_or_failed_auctions
        if a_res['status'] == 'Completed' and a_res.get('total_collateral_sold', 0) > 1e-9
    ]

    if not completed_auctions_with_sales:
        return 0.0

    total_weighted_efficiency_score = 0.0
    total_collateral_weight = 0.0

    for auction_res in completed_auctions_with_sales:
        collateral_sold = auction_res.get('total_collateral_sold', 0)
        # Use the 'price_efficiency' pre-calculated in record_auction_end if available and reliable
        # Otherwise, recalculate here. Let's assume 'price_efficiency' key is in auction_res.
        efficiency_ratio = auction_res.get('price_efficiency', 0)

        total_weighted_efficiency_score += efficiency_ratio * collateral_sold
        total_collateral_weight += collateral_sold

    if total_collateral_weight > 1e-9:
        return total_weighted_efficiency_score / total_collateral_weight
    else:
        return 0.0

def get_ofp_overhead_proxy(model) -> str:
    """
    Provides a string representation of the proxy for OFP overhead based on the mode.
    """
    if model.ofp_mode == 'TE':
        return f"TE: {config.TE_DECRYPTION_LATENCY:.2f}s (Decryption Latency)"
    elif model.ofp_mode == 'VDF':
        # VDF_DELAY_T might be changed during multi-scenario runs, so read from config
        # or better, from the model instance if it stores its specific VDF_DELAY_T
        current_vdf_delay_t = getattr(model, 'current_vdf_delay_t', config.VDF_DELAY_T)
        return (f"VDF: {current_vdf_delay_t:.1f}s (Delay T) + "
                f"{config.VDF_VERIFICATION_GAS_COST:,} gas/tx (Verify Cost)")
    else: # Baseline or other modes like MarketShock (which might be based on Baseline)
        if model.ofp_mode == 'MarketShock_Drop': # Or however you label it
            return "Baseline (Market Shock): 0 (No OFP)"
        return "Baseline: 0 (No OFP)"

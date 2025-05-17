# simulation_project/analysis/__init__.py

"""
Makes the analysis components (metrics, plotting) importable.
"""
from .metrics import (
    get_total_bad_debt,
    get_mev_profit,
    get_keeper_profit,
    get_average_auction_duration,
    get_average_price_efficiency,
    get_ofp_overhead_proxy
    # get_average_tx_latency # This one is more complex and might need more data
)

from .plotting import (
    plot_distributions_and_boxplots,
    plot_keeper_profit_distributions,
    plot_scatter_relationships,
    plot_correlation_heatmaps,
    plot_pairplots,
    plot_risk_reward,
    format_val, # Helper for table printing
    format_agg  # Helper for table printing
)

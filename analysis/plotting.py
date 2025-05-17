# simulation_project/analysis/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # Import numpy for checking variance
import os # Import os for path joining

# REMOVED: from .model import MakerLiquidationModel # This was causing the ModuleNotFoundError

# --- Helper functions for formatting console output (from original script) ---
def format_val(val, width=15) -> str:
    """Formats a value for table printing."""
    if isinstance(val, (int, float)):
        if pd.isna(val):
            s = "N/A"
        elif isinstance(val, int):
            s = f"{val:,}"
        elif abs(val) > 1e7 or (abs(val) < 1e-3 and val != 0):
            s = f"{val:.2e}"
        else:
            s = f"{val:,.2f}"
    else:
        s = str(val)
    return f"{s:<{width}}"

def format_agg(mean_val, std_val, width) -> str:
    """Formats mean ± std dev for table printing."""
    if pd.isna(mean_val):
        return format_val("N/A", width)
    mean_str = format_val(mean_val, width=0) # width=0 to get unpadded string
    std_str = format_val(std_val, width=0) if pd.notna(std_val) else "0.00"
    combined = f"{mean_str} ± {std_str}"
    return format_val(combined, width)

# --- Plotting Functions ---

def plot_distributions_and_boxplots(results_df: pd.DataFrame, metrics_to_plot: list, scenario_order: list, output_dir: str = "."):
    """
    Generates and saves histogram distributions and box plots for specified metrics
    from the results DataFrame.
    """
    sns.set_theme(style="whitegrid")
    print(f"\n--- Generating Distributions and Box Plots for Run-level Metrics (Saving to {output_dir}) ---")

    for metric in metrics_to_plot:
        if metric not in results_df.columns:
            print(f"Warning: Metric '{metric}' not found in results_df. Skipping its plots.")
            continue

        # Check for data presence before plotting
        if results_df[metric].isnull().all() or results_df[metric].nunique() == 0:
            print(f"Warning: Metric '{metric}' has no data or no variance. Skipping histogram/KDE plot.")
        else:
            plt.figure(figsize=(12, 7))
            try:
                can_plot_kde = True
                # Check if data for KDE has variance for each group
                if results_df.groupby("scenario")[metric].nunique().min() <= 1:
                     if results_df.groupby("scenario")[metric].var(ddof=0).fillna(0).min() < 1e-9: # ddof=0 for population variance if only 1 sample
                        print(f"Warning: Metric '{metric}' has low/no variance for some scenarios. Plotting histogram without KDE.")
                        can_plot_kde = False

                if can_plot_kde:
                    sns.histplot(data=results_df, x=metric, hue="scenario", kde=True, multiple="layer", hue_order=scenario_order)
                else:
                    sns.histplot(data=results_df, x=metric, hue="scenario", kde=False, multiple="layer", hue_order=scenario_order)
                plt.title(f"Distribution of {metric} per Run (Overlaid)")
                plt.xlabel(metric)
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{metric}_distribution_overlay.png"))
                plt.close()

            except np.linalg.LinAlgError as e:
                print(f"Warning: LinAlgError for metric '{metric}' during KDE plotting. Plotting histogram without KDE. Error: {e}")
                plt.close() # Close the potentially broken plot
                plt.figure(figsize=(12, 7)) # Reopen
                sns.histplot(data=results_df, x=metric, hue="scenario", kde=False, multiple="layer", hue_order=scenario_order)
                plt.title(f"Distribution of {metric} per Run (Overlaid - No KDE)")
                plt.xlabel(metric)
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{metric}_distribution_overlay_no_kde.png"))
                plt.close()
            except Exception as e: # Catch any other plotting errors
                print(f"Error plotting histogram for metric '{metric}': {e}")
                plt.close() # Close plot and continue
                continue


        plt.figure(figsize=(12, 7))
        sns.boxplot(data=results_df, x="scenario", y=metric, order=scenario_order, showfliers=True)
        plt.title(f"Box Plot of {metric} by Scenario")
        plt.xlabel("Scenario")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot_scenario.png"))
        plt.close()

        plt.figure(figsize=(12, 7))
        sns.barplot(data=results_df, x="scenario", y=metric, order=scenario_order, capsize=.1, errorbar='ci')
        plt.title(f"Mean {metric} by Scenario (with 95% CI)")
        plt.xlabel("Scenario")
        plt.ylabel(f"Mean {metric}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_barplot_scenario.png"))
        plt.close()
    print("--- Run-level metric plots complete. ---")


def plot_keeper_profit_distributions(all_keeper_final_profits_records: list, scenario_order: list, output_dir: str = "."):
    sns.set_theme(style="whitegrid")
    print(f"\n--- Generating Plots for Individual Keeper Profits (Saving to {output_dir}) ---")

    keeper_profit_data_for_df = []
    for record in all_keeper_final_profits_records:
        scenario = record['scenario']
        run = record['run']
        profits_list = record.get('keeper_profits', [])
        if isinstance(profits_list, list):
            for profit in profits_list:
                keeper_profit_data_for_df.append({'scenario': scenario, 'run': run, 'individual_keeper_profit': profit})

    if not keeper_profit_data_for_df:
        print("No individual keeper profit data to plot.")
        return

    individual_keeper_profits_df = pd.DataFrame(keeper_profit_data_for_df)

    if individual_keeper_profits_df.empty or "individual_keeper_profit" not in individual_keeper_profits_df.columns:
        print("Individual keeper profits DataFrame is empty or missing required column.")
        return

    if individual_keeper_profits_df["individual_keeper_profit"].isnull().all() or individual_keeper_profits_df["individual_keeper_profit"].nunique() == 0:
        print(f"Warning: Individual keeper profits have no data or no variance. Skipping histogram/KDE plot.")
    else:
        plt.figure(figsize=(14, 8))
        try:
            can_plot_kde = True
            if individual_keeper_profits_df.groupby("scenario")["individual_keeper_profit"].nunique().min() <= 1:
                 if individual_keeper_profits_df.groupby("scenario")["individual_keeper_profit"].var(ddof=0).fillna(0).min() < 1e-9:
                    print("Warning: Individual keeper profits have low/no variance for some scenarios. Plotting histogram without KDE.")
                    can_plot_kde = False
            if can_plot_kde:
                sns.histplot(data=individual_keeper_profits_df, x="individual_keeper_profit", hue="scenario", kde=True, multiple="layer", hue_order=scenario_order)
            else:
                sns.histplot(data=individual_keeper_profits_df, x="individual_keeper_profit", hue="scenario", kde=False, multiple="layer", hue_order=scenario_order)
            plt.title("Distribution of Individual Keeper Profits (All Runs, Overlaid)")
            plt.xlabel("Individual Keeper Profit (DAI)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "individual_keeper_profit_distribution.png"))
            plt.close()
        except np.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError for individual keeper profits during KDE plotting. Plotting histogram without KDE. Error: {e}")
            plt.close(); plt.figure(figsize=(14, 8)) # Reset figure
            sns.histplot(data=individual_keeper_profits_df, x="individual_keeper_profit", hue="scenario", kde=False, multiple="layer", hue_order=scenario_order)
            plt.title("Distribution of Individual Keeper Profits (All Runs, Overlaid - No KDE)")
            plt.xlabel("Individual Keeper Profit (DAI)"); plt.ylabel("Frequency"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "individual_keeper_profit_distribution_no_kde.png")); plt.close()
        except Exception as e:
            print(f"Error plotting histogram for individual keeper profits: {e}"); plt.close()


    plt.figure(figsize=(14, 8))
    sns.boxplot(data=individual_keeper_profits_df, x="scenario", y="individual_keeper_profit", order=scenario_order, showfliers=True)
    plt.title("Box Plot of Individual Keeper Profits by Scenario")
    plt.xlabel("Scenario"); plt.ylabel("Individual Keeper Profit (DAI)"); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "individual_keeper_profit_boxplot.png")); plt.close()

    plt.figure(figsize=(14, 8))
    sns.violinplot(data=individual_keeper_profits_df, x="scenario", y="individual_keeper_profit", order=scenario_order, cut=0)
    plt.title("Violin Plot of Individual Keeper Profits by Scenario")
    plt.xlabel("Scenario"); plt.ylabel("Individual Keeper Profit (DAI)"); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "individual_keeper_profit_violinplot.png")); plt.close()
    print("--- Individual keeper profit plots complete. ---")


def plot_scatter_relationships(results_df: pd.DataFrame, scenario_order: list, output_dir: str = "."):
    sns.set_theme(style="whitegrid")
    print(f"\n--- Generating Scatter Plots for Metric Relationships (Saving to {output_dir}) ---")
    scatter_pairs = [
        ("AvgAuctionDuration", "AvgPriceEfficiency"), ("KeeperProfit", "BadDebt"),
        ("AvgPriceEfficiency", "KeeperProfit"), ("TotalValueLiquidated", "KeeperProfit")
    ]
    for x_metric, y_metric in scatter_pairs:
        if x_metric not in results_df.columns or y_metric not in results_df.columns:
            print(f"Warning: Metrics '{x_metric}' or '{y_metric}' not found for scatter plot. Skipping.")
            continue
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x=x_metric, y=y_metric, hue="scenario", alpha=0.7, hue_order=scenario_order, s=50)
        plt.title(f"{y_metric} vs. {x_metric} (per Run)"); plt.xlabel(x_metric); plt.ylabel(y_metric)
        plt.legend(title="Scenario"); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_{y_metric.lower()}_vs_{x_metric.lower()}.png")); plt.close()
    print("--- Scatter plots complete. ---")


def plot_correlation_heatmaps(results_df: pd.DataFrame, numerical_metrics_for_corr: list, scenario_order: list, output_dir: str = "."):
    sns.set_theme(style="whitegrid")
    print(f"\n--- Generating Correlation Heatmaps (Saving to {output_dir}) ---")
    valid_numerical_metrics = [m for m in numerical_metrics_for_corr if m in results_df.columns and results_df[m].nunique() > 1]
    if not valid_numerical_metrics or len(valid_numerical_metrics) < 2 :
        print("Skipping overall correlation heatmap due to insufficient data variance or too few metrics.")
    elif len(results_df[valid_numerical_metrics].drop_duplicates()) > 1:
        plt.figure(figsize=(12, 10)); overall_corr = results_df[valid_numerical_metrics].corr()
        sns.heatmap(overall_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, annot_kws={"size":8})
        plt.title("Overall Correlation Matrix of Key Metrics (All Scenarios)"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "correlation_heatmap_overall.png")); plt.close()
    else: print("Skipping overall correlation heatmap due to insufficient data variance.")

    for scenario_name in scenario_order:
        scenario_df = results_df[results_df['scenario'] == scenario_name]
        valid_scenario_metrics = [m for m in valid_numerical_metrics if m in scenario_df.columns and scenario_df[m].nunique() > 1]
        if not scenario_df.empty and len(valid_scenario_metrics) >= 2 and len(scenario_df[valid_scenario_metrics].drop_duplicates()) > 1:
            plt.figure(figsize=(12, 10)); scenario_corr = scenario_df[valid_scenario_metrics].corr()
            sns.heatmap(scenario_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, annot_kws={"size":8})
            plt.title(f"Correlation Matrix for Scenario: {scenario_name}"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"correlation_heatmap_{scenario_name.replace('.', '_')}.png")); plt.close()
        else: print(f"Skipping correlation heatmap for {scenario_name} due to insufficient data variance or too few metrics.")
    print("--- Correlation heatmaps complete. ---")


def plot_pairplots(results_df: pd.DataFrame, pairplot_metrics: list, scenario_order: list, output_dir: str = "."):
    sns.set_theme(style="ticks")
    print(f"\n--- Generating Pair Plot (Saving to {output_dir}) ---")
    valid_pairplot_metrics = [m for m in pairplot_metrics if m in results_df.columns and results_df[m].nunique() > 1]
    if not valid_pairplot_metrics or len(valid_pairplot_metrics) < 2:
        print("Not enough data variance or too few metrics for pairplot.")
        return
    pairplot_df_data = results_df[valid_pairplot_metrics + ["scenario"]]
    if not pairplot_df_data.empty and len(pairplot_df_data.drop_duplicates(subset=valid_pairplot_metrics)) > 1:
        try:
            pair_plot_fig = sns.pairplot(pairplot_df_data, hue="scenario", corner=True, hue_order=scenario_order, diag_kind="kde")
            pair_plot_fig.fig.suptitle("Pair Plot of Selected Metrics by Scenario", y=1.02)
            pair_plot_fig.savefig(os.path.join(output_dir, "pairplot_selected_metrics.png")); plt.close(pair_plot_fig.fig)
        except np.linalg.LinAlgError as e: print(f"Warning: LinAlgError during pairplot KDE: {e}. Consider if data has enough variance."); plt.close()
        except Exception as e: print(f"Error during pairplot generation: {e}"); plt.close()
    else: print("Not enough data variance for pairplot after filtering.")
    print("--- Pair plot complete. ---")


def plot_risk_reward(aggregated_results_df: pd.DataFrame, scenario_order: list, output_dir: str = "."):
    sns.set_theme(style="whitegrid")
    print(f"\n--- Generating Risk-Reward Plot for Keeper Profit (Saving to {output_dir}) ---")
    if 'KeeperProfit' in aggregated_results_df.columns.get_level_values(0):
        keeper_profit_summary = aggregated_results_df['KeeperProfit'][['mean', 'std']].reset_index()
        if keeper_profit_summary.empty or keeper_profit_summary['std'].isnull().all() or keeper_profit_summary['mean'].isnull().all():
            print("No valid aggregated KeeperProfit data (mean/std) for risk-reward plot."); return
        plt.figure(figsize=(10, 7))
        keeper_profit_summary_filtered = keeper_profit_summary[keeper_profit_summary['scenario'].isin(scenario_order)].dropna(subset=['mean', 'std'])
        if keeper_profit_summary_filtered.empty:
            print("No data for scenarios in scenario_order or all data has NaN mean/std for risk-reward plot."); plt.close(); return
        sns.scatterplot(data=keeper_profit_summary_filtered, x='std', y='mean', hue='scenario', s=150, style='scenario', hue_order=scenario_order, style_order=scenario_order, legend='full')
        for i in range(keeper_profit_summary_filtered.shape[0]):
            row = keeper_profit_summary_filtered.iloc[i]
            plt.text(row['std'] + 0.01 * keeper_profit_summary_filtered['std'].max(), row['mean'], row['scenario'], fontdict={'size':9})
        plt.title("Keeper Profit: Mean vs. Standard Deviation (Risk-Reward Proxy)"); plt.xlabel("Standard Deviation of Total Keeper Profit per Run (Risk/Variability)"); plt.ylabel("Mean Total Keeper Profit per Run (Reward)")
        plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "risk_reward_keeper_profit.png")); plt.close()
        print("--- Risk-reward plot complete. ---")
    else: print("KeeperProfit mean/std not found in aggregated_results_df for risk-reward plot.")

# NEW FUNCTION FOR CDF PLOTS
def plot_cdf_comparison(results_df: pd.DataFrame, metric: str, scenario_order: list, output_dir: str = ".", x_lim=None):
    """
    Generates and saves a Cumulative Distribution Function (CDF) plot for a given metric,
    compared across scenarios.
    """
    if metric not in results_df.columns:
        print(f"Warning: Metric '{metric}' not found for CDF plot. Skipping.")
        return
    if results_df[metric].isnull().all() or results_df[metric].nunique() < 2 : # Need at least 2 unique points for a meaningful CDF
        print(f"Warning: Metric '{metric}' has no data or not enough unique values for CDF plot. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=results_df, x=metric, hue="scenario", hue_order=scenario_order, complementary=False) # complementary=False for standard CDF
    plt.title(f"CDF of {metric} by Scenario")
    plt.xlabel(metric)
    plt.ylabel("Cumulative Probability (P(X <= x))")
    if x_lim:
        plt.xlim(x_lim)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_cdf_scenario.png"))
    plt.close()
    print(f"--- CDF plot for {metric} complete. ---")

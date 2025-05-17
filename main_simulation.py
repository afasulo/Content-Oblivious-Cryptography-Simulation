# simulation_project/main_simulation.py

import time
import random
import pandas as pd
import os
import gc

from .model import MakerLiquidationModel
from . import config
from .analysis import plotting

if __name__ == "__main__":
    start_full_run_time = time.time()

    OUTPUT_DIR = "simulation_outputs_enhanced_v2" # New directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Simulation outputs will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    num_runs_per_scenario = config.NUM_RUNS if hasattr(config, 'NUM_RUNS') else 100 # Default to 100 if not in config
    scenarios_to_run = config.SCENARIOS # Includes 'MarketShock_Drop' if added to config
    vdf_delays_to_test_for_vdf_scenario = config.VDF_DELAYS_TO_TEST

    MODEL_RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "aggregated_model_results_per_run_incremental.csv")
    ALL_AUCTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "all_auctions_data_detailed.csv") # New CSV for auction data
    KEEPER_PROFITS_CSV_PATH = os.path.join(OUTPUT_DIR, "all_keeper_final_profits_records.csv")


    all_keeper_final_profits_records = []
    first_model_data_write = True
    first_auction_data_write = True

    total_simulation_configs = 0
    scenario_details_list = []
    for scenario_name_loop in scenarios_to_run:
        if scenario_name_loop == 'VDF':
            for vdf_delay in vdf_delays_to_test_for_vdf_scenario:
                total_simulation_configs +=1
                scenario_details_list.append(f"{scenario_name_loop}_T{vdf_delay}")
        else: # Baseline, TE, MarketShock_Drop etc.
            total_simulation_configs +=1
            scenario_details_list.append(scenario_name_loop)

    total_simulations_to_run = num_runs_per_scenario * total_simulation_configs
    current_simulation_count = 0

    print(f"\nStarting {num_runs_per_scenario} runs for each of the following {len(scenario_details_list)} configurations:")
    for sd_label in scenario_details_list: print(f"  - {sd_label}")
    print(f"Total simulations to run: {total_simulations_to_run}")
    print(f"Model results will be incrementally saved to: {MODEL_RESULTS_CSV_PATH}")
    print(f"Detailed auction data will be saved to: {ALL_AUCTIONS_CSV_PATH}")
    print("=" * 50)

    for run_number in range(num_runs_per_scenario):
        print(f"\n===== STARTING RUN {run_number + 1} / {num_runs_per_scenario} =====")
        run_base_seed = random.randint(10000 * run_number, 10000 * (run_number + 1) - 1)

        for scenario_label_for_df in scenario_details_list: # Iterate through the generated labels
            current_simulation_count += 1
            print(f"\n--- Running: {scenario_label_for_df} (Run {run_number + 1}) --- "
                  f"(Sim {current_simulation_count}/{total_simulations_to_run})")

            # Determine model parameters based on scenario_label_for_df
            ofp_mode_for_model = scenario_label_for_df.split('_')[0] # Baseline, TE, VDF, MarketShock
            current_vdf_delay = None
            market_shock_step_param = -1
            market_shock_factor_param = 1.0

            if ofp_mode_for_model == 'VDF':
                try:
                    current_vdf_delay = float(scenario_label_for_df.split('_T')[-1])
                except ValueError:
                    print(f"Error parsing VDF delay from scenario label: {scenario_label_for_df}")
                    continue # Skip this misconfigured scenario
            elif ofp_mode_for_model == 'MarketShock': # Handles "MarketShock_Drop"
                ofp_mode_for_model = 'Baseline' # Market shock runs on a base OFP mode, e.g., Baseline
                market_shock_step_param = config.SIMULATION_STEPS // 2 # Shock halfway
                if "Drop" in scenario_label_for_df:
                     market_shock_factor_param = 0.65 # 35% drop
                # Add more shock types if needed (e.g., "MarketShock_Spike")
                print(f"    Market Shock Config: Step={market_shock_step_param}, Factor={market_shock_factor_param}")


            seed_offset = scenario_details_list.index(scenario_label_for_df)
            model_specific_seed = run_base_seed + seed_offset

            model_params = {
                "n_borrowers": config.N_BORROWERS,
                "n_keepers": config.N_KEEPERS,
                "n_mev_searchers": config.N_MEV_SEARCHERS,
                "ofp_mode": ofp_mode_for_model,
                "seed": model_specific_seed,
                "market_shock_step": market_shock_step_param,
                "market_shock_factor": market_shock_factor_param
            }
            if ofp_mode_for_model == 'VDF' and current_vdf_delay is not None:
                model_params["current_vdf_delay_t"] = current_vdf_delay

            model = MakerLiquidationModel(**model_params)

            step_count = 0
            while model.running and step_count < config.SIMULATION_STEPS:
                try:
                    model.step()
                except Exception as e:
                    print(f"\n!!!!! ERROR during {scenario_label_for_df} Run {run_number+1} Step {model.steps} !!!!!")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    model.running = False
                step_count += 1

            model_df_single_run = model.datacollector.get_model_vars_dataframe()
            if not model_df_single_run.empty:
                final_model_state = model_df_single_run.iloc[-1:].copy()
                final_model_state['run'] = run_number + 1
                final_model_state['scenario'] = scenario_label_for_df
                write_header_model = first_model_data_write or not os.path.exists(MODEL_RESULTS_CSV_PATH)
                final_model_state.to_csv(MODEL_RESULTS_CSV_PATH, mode='a', header=write_header_model, index=False)
                if write_header_model: first_model_data_write = False
            else:
                print(f"Warning: Empty model_df for {scenario_label_for_df} Run {run_number+1}.")
            del model_df_single_run

            agent_df_single_run = model.datacollector.get_agent_vars_dataframe()
            if agent_df_single_run is not None and not agent_df_single_run.empty:
                try:
                    final_step_agent_data = agent_df_single_run.xs(agent_df_single_run.index.get_level_values('Step').max(), level='Step')
                    keeper_profits_this_run = final_step_agent_data[
                        final_step_agent_data['AgentType'] == 'KeeperAgent'
                    ]['TotalProfit'].tolist()
                    all_keeper_final_profits_records.append({
                        'run': run_number + 1, 'scenario': scenario_label_for_df,
                        'keeper_profits': keeper_profits_this_run
                    })
                except KeyError:
                    print(f"Warning: Could not extract final step agent data for {scenario_label_for_df} Run {run_number+1}")
            else:
                print(f"Warning: Empty agent_df for {scenario_label_for_df} Run {run_number+1}")
            del agent_df_single_run

            # Save detailed auction data for this run
            if model.all_auctions_data:
                auctions_df_this_run = pd.DataFrame(model.all_auctions_data)
                # Add run number to this df if not already there (it's now added in record_auction_end)
                # auctions_df_this_run['run'] = run_number + 1 
                # auctions_df_this_run['scenario'] = scenario_label_for_df # Also added in record_auction_end
                write_header_auction = first_auction_data_write or not os.path.exists(ALL_AUCTIONS_CSV_PATH)
                auctions_df_this_run.to_csv(ALL_AUCTIONS_CSV_PATH, mode='a', header=write_header_auction, index=False)
                if write_header_auction: first_auction_data_write = False
            del model.all_auctions_data # Clear for next run

            del model
            gc.collect()

    end_full_run_time = time.time()
    total_duration_seconds = end_full_run_time - start_full_run_time
    print(f"\nTotal Simulation Execution Time ({num_runs_per_scenario} runs across all configs): "
          f"{total_duration_seconds:.2f} seconds ({total_duration_seconds/60:.2f} minutes)")

    if os.path.exists(MODEL_RESULTS_CSV_PATH):
        results_agg_df = pd.read_csv(MODEL_RESULTS_CSV_PATH)
        print(f"\nSuccessfully loaded aggregated model results from: {MODEL_RESULTS_CSV_PATH}")

        metrics_for_summary_table = [
            'TotalValueLiquidated', 'CompletedAuctions', 'FailedAuctions', 'BadDebt',
            'KeeperProfit', 'MEVProfit', 'AvgAuctionDuration', 'AvgPriceEfficiency',
            'TotalCollateralAddedByBorrowers', 'TotalDebtRepaidByBorrowers'
        ]
        actual_metrics_present = [m for m in metrics_for_summary_table if m in results_agg_df.columns]

        if actual_metrics_present:
            summary_stats = results_agg_df.groupby('scenario')[actual_metrics_present].agg(['mean', 'std'])
            print("\n--- Aggregated Results Summary (Mean +/- Std Dev over Runs) ---")
            header_width = 25 # Adjusted for potentially longer scenario names
            
            # Use scenario_details_list for consistent order in table header
            scenarios_in_summary = [s for s in scenario_details_list if s in summary_stats.index]

            header_line = f"{'Metric':<35} | " + " | ".join([plotting.format_val(s, header_width) for s in scenarios_in_summary])
            print(header_line)
            print("-" * len(header_line))

            for metric in actual_metrics_present:
                if metric in summary_stats.columns.get_level_values(0):
                    values_str = " | ".join([
                        plotting.format_agg(
                            summary_stats.loc[s, (metric, 'mean')],
                            summary_stats.loc[s, (metric, 'std')],
                            header_width
                        ) if s in summary_stats.index else plotting.format_val("N/A", header_width)
                        for s in scenarios_in_summary
                    ])
                    print(f"{metric:<35} | {values_str}")
                else:
                    print(f"{metric:<35} | " + " | ".join([plotting.format_val("N/A", header_width)] * len(scenarios_in_summary)))
            
            print("-" * len(header_line))
            overhead_metric_name = "OFPOverheadProxy"
            if overhead_metric_name in results_agg_df.columns:
                overhead_proxies = results_agg_df.drop_duplicates(subset=['scenario'])[['scenario', overhead_metric_name]].set_index('scenario')
                values_str = " | ".join([
                    plotting.format_val(
                        overhead_proxies.loc[s, overhead_metric_name],
                        header_width
                    ) if s in overhead_proxies.index else plotting.format_val("N/A", header_width)
                    for s in scenarios_in_summary
                ])
                print(f"{overhead_metric_name:<35} | {values_str}")
            print("\nAggregated Analysis Table Complete.")

            plot_scenario_order = scenarios_in_summary
            if not plot_scenario_order:
                plot_scenario_order = sorted(results_agg_df['scenario'].unique())

            plotting.plot_distributions_and_boxplots(results_agg_df, actual_metrics_present, plot_scenario_order, OUTPUT_DIR)
            if all_keeper_final_profits_records:
                 plotting.plot_keeper_profit_distributions(all_keeper_final_profits_records, plot_scenario_order, OUTPUT_DIR)
                 # Save keeper profits records to CSV
                 keeper_profits_df = pd.DataFrame(all_keeper_final_profits_records)
                 keeper_profits_df.to_csv(KEEPER_PROFITS_CSV_PATH, index=False)
                 print(f"All keeper final profits records saved to: {KEEPER_PROFITS_CSV_PATH}")


            plotting.plot_scatter_relationships(results_agg_df, plot_scenario_order, OUTPUT_DIR)
            corr_metrics = [m for m in actual_metrics_present if m not in ["OFPOverheadProxy"]]
            plotting.plot_correlation_heatmaps(results_agg_df, corr_metrics, plot_scenario_order, OUTPUT_DIR)
            
            pairplot_subset = ["AvgPriceEfficiency", "KeeperProfit", "BadDebt", "AvgAuctionDuration", "TotalValueLiquidated"]
            pairplot_subset_present = [m for m in pairplot_subset if m in results_agg_df.columns]
            if pairplot_subset_present:
                plotting.plot_pairplots(results_agg_df, pairplot_subset_present, plot_scenario_order, OUTPUT_DIR)
            
            # Add new CDF plots
            if "KeeperProfit" in actual_metrics_present:
                plotting.plot_cdf_comparison(results_agg_df, "KeeperProfit", plot_scenario_order, OUTPUT_DIR)
            if "AvgPriceEfficiency" in actual_metrics_present:
                 # Filter out runs with 0 efficiency for a more meaningful CDF if many such runs exist
                plot_df_eff = results_agg_df[results_agg_df["AvgPriceEfficiency"] > 0]
                if not plot_df_eff.empty:
                    plotting.plot_cdf_comparison(plot_df_eff, "AvgPriceEfficiency", plot_scenario_order, OUTPUT_DIR)
                else:
                    print("No positive AvgPriceEfficiency data to plot CDF.")


            if not summary_stats.empty:
                plotting.plot_risk_reward(summary_stats, plot_scenario_order, OUTPUT_DIR)
            print(f"\n--- Data Visualizations Complete. Plots saved to '{OUTPUT_DIR}' directory. ---")
        else:
            print("No valid metrics found in the aggregated results for detailed analysis.")
    else:
        print(f"\nModel results CSV not found at {MODEL_RESULTS_CSV_PATH}. Skipping analysis and plotting.")

    print("\n===== SIMULATION SCRIPT COMPLETE =====")

import pandas as pd
import sys
from typing import List

def _process_single_csv(file_path: str) -> pd.DataFrame:
    """
    Helper function.
    Compute per-scenario PDMS scores from an EPDMS result CSV file,
    and return a DataFrame with all original columns plus a new 'pdms_score' column.

    Args:
        file_path (str): Path to the EPDMS result CSV file.

    Returns:
        pd.DataFrame: DataFrame with all original data and 'pdms_score' column.
                      Returns an empty DataFrame if the file is invalid or has no valid scenarios.
    """
    try:
        epdms_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Skipping.", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}. Skipping.", file=sys.stderr)
        return pd.DataFrame()

    # 2. Check required columns
    required_cols = [
        'valid', 'token', 'no_at_fault_collisions', 'drivable_area_compliance',
        'ego_progress', 'time_to_collision_within_bound', 'history_comfort'
    ]
    missing_cols = [col for col in required_cols if col not in epdms_df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns in '{file_path}': {', '.join(missing_cols)}. Skipping file.", file=sys.stderr)
        return pd.DataFrame()

    # 3. Filter valid scenarios, excluding summary/average rows
    valid_scenarios_df = epdms_df[
        (epdms_df['valid'] == True) &
        (~epdms_df['token'].str.contains('average|score', case=False, na=False))
    ].copy()

    if valid_scenarios_df.empty:
        print(f"Info: No valid scenario data found in '{file_path}'.")
        return pd.DataFrame()

    # 4. Define PDMS v1 weights
    W_EP = 5.0
    W_TTC = 5.0
    W_HC = 2.0
    TOTAL_WEIGHT_PDMS = W_EP + W_TTC + W_HC

    # 5. Compute two core components of PDMS v1
    multiplier_prod = (
        valid_scenarios_df['no_at_fault_collisions'] *
        valid_scenarios_df['drivable_area_compliance']
    )

    weighted_sum = (
        W_EP * valid_scenarios_df['ego_progress'] +
        W_TTC * valid_scenarios_df['time_to_collision_within_bound'] +
        W_HC * valid_scenarios_df['history_comfort']
    )

    weighted_avg = weighted_sum / TOTAL_WEIGHT_PDMS

    # 6. Compute per-scenario PDMS score
    valid_scenarios_df['pdms_score'] = multiplier_prod * weighted_avg

    return valid_scenarios_df

def generate_best_of_n_report(file_paths: List[str], output_csv_path: str):
    """
    Compute "best-of-n" PDMS scores from multiple EPDMS result CSV files,
    and save a CSV report containing only the best result for each token.

    Args:
        file_paths (List[str]): List of paths to EPDMS result CSV files.
        output_csv_path (str): Path to save the final "best-of-n" report CSV.
    """
    if not file_paths:
        print("Error: No input file paths provided.", file=sys.stderr)
        return

    print(f"Processing {len(file_paths)} files for Best-of-N report...")

    # 1. Read all files, compute per-scenario PDMS, and collect into a list
    all_scenario_dfs_list = []
    for file_path in file_paths:
        scenario_df = _process_single_csv(file_path)
        if not scenario_df.empty:
            all_scenario_dfs_list.append(scenario_df)

    if not all_scenario_dfs_list:
        print("Error: No valid data could be processed from any of the files.", file=sys.stderr)
        return

    # 2. Concatenate all DataFrames
    combined_df = pd.concat(all_scenario_dfs_list, ignore_index=True)

    if 'token' not in combined_df.columns:
         print("Error: 'token' column not found in processed data.", file=sys.stderr)
         return

    print(f"Total valid scenarios found across all files: {len(combined_df)}")

    # 3. Find the "best" row for each token (highest PDMS score)
    #    - Sort by 'pdms_score' descending
    #    - Deduplicate by 'token', keeping the first (highest score)
    best_of_n_df = combined_df.sort_values(by='pdms_score', ascending=False)
    best_of_n_df = best_of_n_df.drop_duplicates(subset=['token'], keep='first')

    num_unique_tokens = len(best_of_n_df)
    if num_unique_tokens == 0:
        print("Error: No unique tokens found after processing.", file=sys.stderr)
        return

    # 4. Compute the final average of the "best" scores
    final_best_of_n_pdms_score = best_of_n_df['pdms_score'].mean()

    # Append an average row
    average_row = best_of_n_df.mean(numeric_only=True)
    average_row['token'] = 'average_best_of_n'
    best_of_n_df = pd.concat([best_of_n_df, pd.DataFrame([average_row])], ignore_index=True)


    # 5. Save "best-of-n" results to CSV
    try:
        best_of_n_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    except Exception as e:
        print(f"Error: Failed to save output CSV to '{output_csv_path}': {e}", file=sys.stderr)
        return

    # 6. Print final summary
    print("\n--- Best-of-N PDMS Report Generation Complete ---")
    print(f"Total files processed (N): {len(file_paths)}")
    print(f"Unique tokens/scenarios found: {num_unique_tokens}")
    print(f"Final Best-of-N PDMS Score: {final_best_of_n_pdms_score:.4f}")
    print(f"Best-of-N results saved to: {output_csv_path}")
    print("-------------------------------------------------")


# --- Script entry point ---
if __name__ == "__main__":
    list_of_csv_files = []
    output_csv_path = ""

    generate_best_of_n_report(list_of_csv_files, output_csv_path)

import pandas as pd
import numpy as np

# List of Hugging Face Parquet URLs
hf_urls = [
    # Qwen7b
    "hf://datasets/RAGEVALUATION-HJKMY/Qwen7b_ragbench_techqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/Qwen7b_ragbench_emanual_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/Qwen7b_ragbench_delucionqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    # Deepseek7b
    "hf://datasets/RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_techqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_delucionqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_emanual_400row_mistake_added/data/train-00000-of-00001.parquet",
    # Mistral8b
    "hf://datasets/RAGEVALUATION-HJKMY/Mistral8b_ragbench_techqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/Mistral8b_ragbench_emanual_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/Mistral8b_ragbench_delucionqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    # llama8b
    "hf://datasets/RAGEVALUATION-HJKMY/Llama8b_ragbench_techqa_400row_mistake_added_evaluated/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/Llama8b_ragbench_delucionqa_400row_mistake_added/data/train-00000-of-00001.parquet",
    "hf://datasets/RAGEVALUATION-HJKMY/Llama8b_ragbench_emanual_400row_mistake_added_evaluated/data/train-00000-of-00001.parquet",
]

# Initialize a list to store all dataframes
all_dfs = []

# First, load and concatenate all dataframes
for file_path in hf_urls:
    try:
        print(f"\nLoading: {file_path}")
        df = pd.read_parquet(file_path)

        # Extract dataset name from the URL
        full_dataset_name = file_path.split("/")[4]
        parts = full_dataset_name.split("_")
        dataset_name = parts[2] if len(parts) >= 3 else "Unknown"

        # Add dataset column to the dataframe (no model column)
        df["dataset"] = dataset_name

        # Store the dataframe for later concatenation
        all_dfs.append(df)

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Concatenate all dataframes vertically
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv("combined_rag_datasets.csv", index=False)
    print("✅ Saved: combined_rag_datasets.csv")
else:
    print("No dataframes to concatenate!")
    exit()

# Now calculate metrics on the combined dataset
all_results = []

# Identify all datasets in the combined data
datasets = combined_df["dataset"].unique()

# Get metric names from the combined dataframe
metric_columns = [
    col
    for col in combined_df.columns
    if col.startswith("ground_truth_") and col.endswith("_score")
]
metric_names = [
    col.replace("ground_truth_", "").replace("_score", "") for col in metric_columns
]

print(
    f"\nCalculating metrics for {len(metric_names)} metrics across {len(datasets)} datasets..."
)

# Calculate metrics for each dataset and metric (no model dimension)
for dataset in datasets:
    # Filter the combined dataframe
    filtered_df = combined_df[combined_df["dataset"] == dataset]

    if filtered_df.empty:
        print(f"No data for dataset {dataset}. Skipping...")
        continue

    # Calculate metrics for each metric type
    for metric in metric_names:
        try:
            gt_scores = filtered_df[f"ground_truth_{metric}_score"].dropna()
            re_scores = filtered_df[f"Correct_{metric}_score"].dropna()
            in_scores = filtered_df[f"Incorrect_{metric}_score"].dropna()

            # Skip if any category has insufficient data
            if len(gt_scores) < 2 or len(in_scores) < 2:
                print(f"Insufficient data for {dataset}/{metric}. Skipping...")
                continue

            gt_mean = gt_scores.mean()
            re_mean = re_scores.mean()
            in_mean = in_scores.mean()

            gt_var = gt_scores.var(ddof=1)  # Sample variance
            re_var = re_scores.var(ddof=1)
            in_var = in_scores.var(ddof=1)

            gt_sigma = gt_scores.var(ddof=0)  # Population variance
            re_sigma = re_scores.var(ddof=0)
            in_sigma = in_scores.var(ddof=0)

            # Calculate Cohen's d using pooled standard deviation
            pooled_sd = np.sqrt(
                (gt_var * (len(gt_scores) - 1) + in_var * (len(in_scores) - 1))
                / (len(gt_scores) + len(in_scores) - 2)
            )
            cohen_d = (gt_mean - in_mean) / pooled_sd if pooled_sd != 0 else np.nan

            # Calculate Variance Ratio (VR)
            vr = re_sigma / gt_sigma if gt_sigma != 0 else np.nan

            all_results.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "ground_truth_mean": gt_mean,
                    "ground_truth_variance(s^2)": gt_var,
                    "ground_truth_variance(sigma^2)": gt_sigma,
                    "rewrite_mean": re_mean,
                    "rewrite_variance(s^2)": re_var,
                    "rewrite_variance(sigma^2)": re_sigma,
                    "wrong_mean": in_mean,
                    "wrong_variance(s^2)": in_var,
                    "wrong_variance(sigma^2)": in_sigma,
                    "Cohen's d": cohen_d,
                    "VR": vr,
                    "n_ground_truth": len(gt_scores),
                    "n_rewrite": len(re_scores),
                    "n_wrong": len(in_scores),
                }
            )
        except KeyError as e:
            print(f"Metric {metric} missing columns for {dataset}. Error: {e}")

    print(f"Completed calculations for dataset {dataset}")

# Calculate metrics for the entire combined dataset (across all datasets)
print("\nCalculating overall metrics across all datasets...")

for metric in metric_names:
    try:
        gt_scores = combined_df[f"ground_truth_{metric}_score"].dropna()
        re_scores = combined_df[f"Correct_{metric}_score"].dropna()
        in_scores = combined_df[f"Incorrect_{metric}_score"].dropna()

        if len(gt_scores) < 2 or len(in_scores) < 2:
            print(f"Insufficient overall data for {metric}. Skipping...")
            continue

        gt_mean = gt_scores.mean()
        re_mean = re_scores.mean()
        in_mean = in_scores.mean()

        gt_var = gt_scores.var(ddof=1)
        re_var = re_scores.var(ddof=1)
        in_var = in_scores.var(ddof=1)

        gt_sigma = gt_scores.var(ddof=0)
        re_sigma = re_scores.var(ddof=0)
        in_sigma = in_scores.var(ddof=0)

        # Calculate Cohen's d
        pooled_sd = np.sqrt(
            (gt_var * (len(gt_scores) - 1) + in_var * (len(in_scores) - 1))
            / (len(gt_scores) + len(in_scores) - 2)
        )
        cohen_d = (gt_mean - in_mean) / pooled_sd if pooled_sd != 0 else np.nan

        # Calculate VR
        vr = re_sigma / gt_sigma if gt_sigma != 0 else np.nan

        all_results.append(
            {
                "Dataset": "ALL",
                "Metric": metric,
                "ground_truth_mean": gt_mean,
                "ground_truth_variance(s^2)": gt_var,
                "ground_truth_variance(sigma^2)": gt_sigma,
                "rewrite_mean": re_mean,
                "rewrite_variance(s^2)": re_var,
                "rewrite_variance(sigma^2)": re_sigma,
                "wrong_mean": in_mean,
                "wrong_variance(s^2)": in_var,
                "wrong_variance(sigma^2)": in_sigma,
                "Cohen's d": cohen_d,
                "VR": vr,
                "n_ground_truth": len(gt_scores),
                "n_rewrite": len(re_scores),
                "n_wrong": len(in_scores),
            }
        )
    except KeyError as e:
        print(f"Metric {metric} missing columns in combined dataset. Error: {e}")

# Convert all results to DataFrame and export
final_df = pd.DataFrame(all_results)

# Sort by dataset and metric for better readability
final_df = final_df.sort_values(by=["Dataset", "Metric"])

# Export statistics
final_df.to_csv("metric_validation_summary_by_dataset.csv", index=False)
print("✅ Saved: metric_validation_summary_by_dataset.csv")

# Create a pivot table for easier cross-comparisons
pivot_df = final_df.pivot_table(
    index=["Dataset"], columns="Metric", values=["Cohen's d", "VR"], aggfunc="first"
)

# Export the pivot table
pivot_df.to_csv("metric_validation_pivot_by_dataset.csv")
print("✅ Saved: metric_validation_pivot_by_dataset.csv")

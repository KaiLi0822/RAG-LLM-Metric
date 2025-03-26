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
    "hf://datasets/RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_emanual_400row_mistake_added/data/train-00000-of-00001.parquet"
]

# Initialize the final results list
all_results = []

for file_path in hf_urls:
    try:
        print(f"\nProcessing: {file_path}")
        df = pd.read_parquet(file_path)

        # Extract dataset/model names from the URL (or override manually if needed)
        full_dataset_name = file_path.split("/")[4]
        parts = full_dataset_name.split("_")
        dataset_name = parts[2] if len(parts) >= 3 else "Unknown"
        model_name = parts[0]

        # Step 1: Obtain metric names
        metric_columns = [col for col in df.columns if col.startswith("ground_truth_") and col.endswith("_score")]
        metric_names = [col.replace("ground_truth_", "").replace("_score", "") for col in metric_columns]

        # Step 2: Compute stats for each metric
        for metric in metric_names:
            try:
                gt_scores = df[f"ground_truth_{metric}_score"].dropna()
                re_scores = df[f"Correct_{metric}_score"].dropna()
                in_scores = df[f"Incorrect_{metric}_score"].dropna()

                gt_mean = gt_scores.mean()
                re_mean = re_scores.mean()
                in_mean = in_scores.mean()

                gt_var = gt_scores.var()
                re_var = re_scores.var()
                in_var = in_scores.var()

                gt_sigma = gt_scores.var(ddof=0)
                re_sigma = re_scores.var(ddof=0)
                in_sigma = in_scores.var(ddof=0)

                pooled_sd = np.sqrt((gt_var + in_var) / 2)
                cohen_d = (gt_mean - in_mean) / pooled_sd if pooled_sd != 0 else np.nan
                vr = re_sigma / gt_sigma if gt_sigma != 0 else np.nan

                all_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
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
                    "VR": vr
                })
            except KeyError:
                print(f"Metric {metric} missing columns. Skipping...")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Convert all results to DataFrame and export
final_df = pd.DataFrame(all_results)
final_df.to_csv("metric_validation_summary_all.csv", index=False)
print("✅ Saved: metric_validation_summary_all.csv")

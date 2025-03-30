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

all_results = []

for file_path in hf_urls:
    try:
        print(f"\n🔍 Processing: {file_path}")
        df = pd.read_parquet(file_path)

        # Extract dataset/model names
        full_dataset_name = file_path.split("/")[4]
        parts = full_dataset_name.split("_")
        dataset_name = parts[2] if len(parts) >= 3 else "Unknown"
        model_name = parts[0]

        # Step 1: Identify metric names
        metric_columns = [col for col in df.columns if col.startswith("ground_truth_")]
        metric_names = []

        for col in metric_columns:
            raw = col.removeprefix("ground_truth_")
            if raw.endswith("_score"):
                metric = raw.removesuffix("_score")
            else:
                metric = raw

            # Fix duplicated suffixes like refusal_accuracy_refusal_accuracy
            parts_metric = metric.split("_")
            if len(parts_metric) >= 2 and parts_metric[-1] == parts_metric[-2]:
                metric = "_".join(parts_metric[:-1])

            metric_names.append(metric)

        metric_names = list(set(metric_names))  # Remove duplicates

        # Step 2: Process each metric
        for metric in metric_names:
            try:
                def resolve_col(prefix):
                    # Match any col starting with prefix_metric
                    prefix_string = f"{prefix}_{metric}"
                    matches = [col for col in df.columns if col.startswith(prefix_string)]
                    if not matches:
                        raise KeyError(f"{prefix_string} column for '{metric}' not found")
                    return df[matches[0]].dropna()

                gt_scores = resolve_col("ground_truth")
                re_scores = resolve_col("Correct")
                in_scores = resolve_col("Incorrect")

                # Mean and variance calculations
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

            except KeyError as e:
                print(f"⚠️ Metric '{metric}' skipped — {e}")

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

# Save to CSV
final_df = pd.DataFrame(all_results)
final_df.to_csv("metric_validation_summary_all.csv", index=False)
print("✅ Saved: metric_validation_summary_all.csv")

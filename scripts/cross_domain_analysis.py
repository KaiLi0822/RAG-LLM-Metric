import re
import pandas as pd
import numpy as np
from datasets import load_dataset


# 1. LOAD DATASETS (DeepSeek7b, 3 Domains)
techqa = load_dataset(
    "RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_techqa_400row_mistake_added",
    split="train"
).to_pandas()

emanual = load_dataset(
    "RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_emanual_400row_mistake_added",
    split="train"
).to_pandas()

delucionqa = load_dataset(
    "RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_delucionqa_400row_mistake_added",
    split="train"
).to_pandas()

# Put them into a dictionary for convenience
domain_dfs = {
    "techqa": techqa,
    "emanual": emanual,
    "delucionqa": delucionqa
}

# 2. DETECT METRIC NAMES AUTOMATICALLY
# Using regex to find columns that start with Correct_, Incorrect_, ground_truth_
metric_pattern = re.compile(r'^(Correct|Incorrect|ground_truth)_(.+)$')
sample_cols = techqa.columns.tolist()

# This will extract the second capture group (the metric name after the underscore)
metric_names = sorted({
    match.group(2)
    for col in sample_cols
    if (match := metric_pattern.match(col))
})

print("Detected metric_names:", metric_names)


# 3. SET THRESHOLDS FOR EACH METRIC
# These are EXAMPLE threshold values. Adjust as needed.
# If a metric is missing from this dictionary, we'll use a default threshold of 0.5

custom_thresholds = {
    # 1. Adherence/Faithfulness
    "Adherence_Faithfulness_faithfulness_score": 0.7,
    # 2. Answer Equivalence
    #    (This is an int64. If it's an exact match scenario, you might set threshold=1)
    "answer_equivalence_equivalence_score": 1.0,
    # 3. Answer Similarity
    "answer_similarity": 0.8,
    # 4. Coherence
    "COHERENCE_coherence_score": 0.7,
    # 5. Context Relevance
    "Context_Relevance_relevance_score": 0.5,
    # 6. Context Utilization
    "Context_Utilization_context_utilization_score": 0.4,
    # 7. Engagement
    "engagement_engagement_score": 0.5,
    # 8. Factual Accuracy
    "FACTUAL_ACCURACY_accuracy_score": 0.8,
    # 9. Factual Correctness (F1)
    "factual_correctness_F1_score": 0.9,
    # 10. Key Point Completeness
    "key_point_completeness_score": 0.7,
    # 11. Key Point Hallucination
    #     (A “lower is better” metric might need a different approach,
    #      but we'll assume higher->better for demonstration.)
    "key_point_hallucination_score": 0.5,
    # 12. Key Point Irrelevant
    #     (If lower is better, you might do the inverse or skip it; for now assume higher->better.)
    "key_point_irrelevant_score": 0.5,
    # 13. Learning Facilitation
    "learning_facilitation_learning_facilitation_score": 0.6,
    # 14. Refusal Accuracy
    #     (If it is an int for correct/incorrect, you might set it to 1 for passing.)
    "refusal_accuracy_refusal_accuracy": 1.0
}

default_threshold = 0.5  # used if not found in custom_thresholds


# 4. HELPER FUNCTION: Calculate Pass Rate

def calculate_pass_rate(df, metric_col, threshold):
    """
    Counts how many rows have metric_col >= threshold, then divides by total rows.
    Returns the pass rate (float).
    """
    if metric_col not in df.columns:
        return np.nan
    total_rows = len(df)
    if total_rows == 0:
        return 0.0

    pass_count = (df[metric_col] >= threshold).sum()
    return pass_count / total_rows


# 5. MAIN ANALYSIS: Compute Pass Rates & Domain Robustness

results = []
model_name = "DeepSeek7b"

for metric in metric_names:
    # The dataset columns are "Correct_<metric>", "Incorrect_<metric>", "ground_truth_<metric>"
    rewrite_col = f"Correct_{metric}"
    wrong_col   = f"Incorrect_{metric}"
    gt_col      = f"ground_truth_{metric}"

    # Look up threshold; fallback to default if missing
    threshold = custom_thresholds.get(metric, default_threshold)

    # For each of the 3 domains, compute pass rate
    pass_rates_rewrite = {}
    pass_rates_wrong = {}
    pass_rates_gt = {}

    for domain_name, df in domain_dfs.items():
        pr_rewrite = calculate_pass_rate(df, rewrite_col, threshold)
        pr_wrong   = calculate_pass_rate(df, wrong_col, threshold)
        pr_gt      = calculate_pass_rate(df, gt_col, threshold)

        pass_rates_rewrite[domain_name] = pr_rewrite
        pass_rates_wrong[domain_name]   = pr_wrong
        pass_rates_gt[domain_name]      = pr_gt

    # Convert to arrays in a fixed domain order
    domain_order = ["techqa", "emanual", "delucionqa"]
    rewrite_vals = np.array([pass_rates_rewrite[d] for d in domain_order])
    wrong_vals   = np.array([pass_rates_wrong[d]   for d in domain_order])
    gt_vals      = np.array([pass_rates_gt[d]      for d in domain_order])

    # Function to compute median, MAD, RMAD
    def compute_mad_rmad(values):
        med = np.median(values)
        abs_dev = np.abs(values - med)
        mad = np.median(abs_dev)
        if med == 0:
            rmad = np.nan
        else:
            rmad = mad / med
        return med, mad, rmad

    # rewrite
    rewrite_median, rewrite_MAD, rewrite_RMAD = compute_mad_rmad(rewrite_vals)
    # wrong
    wrong_median, wrong_MAD, wrong_RMAD = compute_mad_rmad(wrong_vals)
    # ground_truth
    gt_median, gt_MAD, gt_RMAD = compute_mad_rmad(gt_vals)

    row = {
        "Metric": metric,
        "Model": model_name,

        # Re-write pass rates
        "techqa_rewrite_PassRate": rewrite_vals[0],
        "emanual_rewrite_PassRate": rewrite_vals[1],
        "delucionqa_rewrite_PassRate": rewrite_vals[2],
        "rewrite_median(PassRate)": rewrite_median,
        "rewrite_MAD": rewrite_MAD,
        "rewrite_RMAD": rewrite_RMAD,

        # Wrong pass rates
        "techqa_wrong_PassRate": wrong_vals[0],
        "emanual_wrong_PassRate": wrong_vals[1],
        "delucionqa_wrong_PassRate": wrong_vals[2],
        "wrong_median(PassRate)": wrong_median,
        "wrong_MAD": wrong_MAD,
        "wrong_RMAD": wrong_RMAD,

        # Ground Truth pass rates
        "techqa_ground_truth_PassRate": gt_vals[0],
        "emanual_ground_truth_PassRate": gt_vals[1],
        "delucionqa_ground_truth_PassRate": gt_vals[2],
        "ground_truth_median(PassRate)": gt_median,
        "ground_truth_MAD": gt_MAD,
        "ground_truth_RMAD": gt_RMAD
    }

    results.append(row)


# 6. CREATE DATAFRAME & SAVE

final_df = pd.DataFrame(results)
final_df.to_csv("domain_robustness_evaluation.csv", index=False)

print("✅ Done! Saved 'domain_robustness_evaluation.csv' with domain robustness stats.")
print(final_df.head())

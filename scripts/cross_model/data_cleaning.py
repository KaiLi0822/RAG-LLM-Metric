from datasets import load_dataset
import pandas as pd
import re

# Load datasets from Hugging Face
deepseek = load_dataset(
    "RAGEVALUATION-HJKMY/DeepSeek7b_ragbench_delucionqa_400row_mistake_added",
    split="train",
).to_pandas()
llama = load_dataset(
    "RAGEVALUATION-HJKMY/Llama8b_ragbench_delucionqa_400row_mistake_added",
    split="train",
).to_pandas()
mistral = load_dataset(
    "RAGEVALUATION-HJKMY/Mistral8b_ragbench_delucionqa_400row_mistake_added",
    split="train",
).to_pandas()
qwen = load_dataset(
    "RAGEVALUATION-HJKMY/Qwen7b_ragbench_delucionqa_400row_mistake_added", split="train"
).to_pandas()

# Extract metrics from DeepSeek dataset
metric_pattern = re.compile(r"^(Correct|Incorrect|ground_truth)_(.+)$")
metric_names = sorted(
    {match.group(2) for col in deepseek.columns if (match := metric_pattern.match(col))}
)

# Base columns (use DeepSeek arbitrarily)
base_df = deepseek[["id", "question", "documents", "response"]].copy()
base_df["rewrite"] = deepseek["Paraphrased"]
base_df["wrong"] = deepseek["Incorrect"]

# Create one dataset per metric
combined_datasets = {}

for metric in metric_names:
    df = base_df.copy()
    df["metric_name"] = metric

    # Add DeepSeek scores
    df["deepseek7b_ground_truth"] = deepseek[f"ground_truth_{metric}"]
    df["deepseek7b_rewrite"] = deepseek[f"Correct_{metric}"]
    df["deepseek7b_wrong"] = deepseek[f"Incorrect_{metric}"]

    # Add LLaMA scores
    df["llama8b_ground_truth"] = llama[f"ground_truth_{metric}"]
    df["llama8b_rewrite"] = llama[f"Correct_{metric}"]
    df["llama8b_wrong"] = llama[f"Incorrect_{metric}"]

    # Add Mistral scores
    df["mistral8b_ground_truth"] = mistral[f"ground_truth_{metric}"]
    df["mistral8b_rewrite"] = mistral[f"Correct_{metric}"]
    df["mistral8b_wrong"] = mistral[f"Incorrect_{metric}"]

    # Add Qwen scores
    df["qwen7b_ground_truth"] = qwen[f"ground_truth_{metric}"]
    df["qwen7b_rewrite"] = qwen[f"Correct_{metric}"]
    df["qwen7b_wrong"] = qwen[f"Incorrect_{metric}"]

    combined_datasets[metric] = df

# Optionally, save each dataset
for metric, df in combined_datasets.items():
    filename = f"combined_{metric}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

# Show how many datasets were created
print(f"✅ Generated {len(combined_datasets)} metric-specific datasets.")

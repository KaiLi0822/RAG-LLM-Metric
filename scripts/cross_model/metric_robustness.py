import pandas as pd
import os

# Configuration
input_folder = "analysis_data/cross_model/cosine_similarity"
output_file = "analysis_data/cross_model/similarity_passrate.csv"
COSINE_THRESHOLD = 0.95

# All possible cosine similarity columns
all_sim_cols = [
    'deepseek7b_llama8b_cos_sim',
    'deepseek7b_mistral8b_cos_sim',
    'deepseek7b_qwen7b_cos_sim',
    'llama8b_mistral8b_cos_sim',
    'llama8b_qwen7b_cos_sim',
    'mistral8b_qwen7b_cos_sim'
]

results = []

# Process each file
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)
        all_rows = len(df)

        # 🧠 Adjust similarity columns per metric
        if filename == "combined_answer_equivalence_equivalence_score.csv":
            sim_cols = [col for col in all_sim_cols if not col.startswith("deepseek7b")]
        else:
            sim_cols = all_sim_cols

        # A row is invalid only if all selected similarity values are -1
        invalid_row_mask = (df[sim_cols] == -1).all(axis=1)
        invalid_row_count = invalid_row_mask.sum()

        # Valid rows
        valid_df = df[~invalid_row_mask]
        valid_row_count = len(valid_df)

        # Count rows passing the threshold
        passed_rows = 0
        if valid_row_count > 0:
            for _, row in valid_df[sim_cols].iterrows():
                valid_values = row[row != -1]
                if not valid_values.empty and valid_values.min() > COSINE_THRESHOLD:
                    passed_rows += 1

        passrate = passed_rows / valid_row_count if valid_row_count > 0 else 0.0

        results.append({
            "metric": os.path.splitext(filename)[0],
            "passrate": passrate,
            "passed_row_counts": passed_rows,
            "valid_row_counts": valid_row_count,
            "invalid_row_counts": invalid_row_count,
            "all_rows": all_rows
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"✅ Summary saved to {output_file} with threshold = {COSINE_THRESHOLD}")

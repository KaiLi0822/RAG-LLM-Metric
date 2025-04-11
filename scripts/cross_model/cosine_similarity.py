import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Directory containing the 14 CSV files
input_folder = "analysis_data/cross_model/data_cleaning"  # Replace with your folder
output_folder = "analysis_data/cross_model/cosine_similarity"
os.makedirs(output_folder, exist_ok=True)

models = ["deepseek7b", "llama8b", "mistral8b", "qwen7b"]
model_pairs = list(combinations(models, 2))

# Process each file
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        # Extract model-related columns (last 12)
        last_12_cols = df.columns[-12:]
        df_selected = df[last_12_cols].copy()

        # Create empty similarity columns
        for model1, model2 in model_pairs:
            df_selected[f"{model1}_{model2}_cos_sim"] = -1.0

        # Precompute columns per model (once, outside the loop)
        model_col_map = {
            model: [
                col
                for col in df_selected.columns
                if model in col and "_cos_sim" not in col
            ]
            for model in models
        }

        # Compute cosine similarity row by row
        for idx, row in df_selected.iterrows():
            model_vectors = {}

            # Extract vector per model
            for model in models:
                values = row[model_col_map[model]].values.astype(float)
                if np.any(values == -1) or np.any(pd.isna(values)):
                    model_vectors[model] = None
                else:
                    model_vectors[model] = values.reshape(1, -1)

            # Compute cosine similarity for each pair (if both vectors are valid)
            for model1, model2 in model_pairs:
                vec1 = model_vectors[model1]
                vec2 = model_vectors[model2]

                if vec1 is not None and vec2 is not None:
                    sim = cosine_similarity(vec1, vec2)[0][0]
                    df_selected.at[idx, f"{model1}_{model2}_cos_sim"] = sim
                else:
                    df_selected.at[idx, f"{model1}_{model2}_cos_sim"] = -1.0

        # Combine original + new similarity columns
        df_combined = pd.concat([df, df_selected.iloc[:, -6:]], axis=1)

        # Save to new CSV file
        output_path = os.path.join(output_folder, filename)
        df_combined.to_csv(output_path, index=False)
        print(f"✅ Processed and saved: {output_path}")

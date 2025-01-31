import pandas as pd

# Data for models and their scores
import json
json_file = 'scores.json'



with open(json_file) as f:
    data = json.load(f) 

models = list(data.keys())

prompt_count = len(data[models[0]])

def index_to_letter(index):
    return chr(index + 65)

data_prompt = {f'Prompt {index_to_letter(i)}': [data[model][i] for model in models] for i in range(prompt_count)}
data = {"Model": models, **data_prompt}


"""
data = {
    "Model": [
        "stelterlab/phi-4-AWQ",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "casperhansen/llama-3-8b-instruct-awq",
        "internlm/internlm3-8b-instruct-awq",
        "lurker18/Ministral_8B_Instruct_2410_AWQ_4bit",
        "tiiuae/Falcon3-10B-Instruct-AWQ",
        "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "CohereForAI/c4ai-command-r7b-12-2024",
    ],
    "Prompt A": [77.77, 75.42, 67.87, 65.76, 66.62, 79.52, 82.36, 64.99],
    "Prompt B": [81.18, 75.56, 77.04, 75.73, 75.34, 81.70, 81.60, 66.95],
    "Prompt C": [76.79, 75.15, 76.63, 72.66, 76.17, 81.09, 82.75, 63.82],
    "Prompt D": [72.86, 74.37, 73.07, 73.09, 74.73, 82.04, 80.57, 64.95],
    "Prompt E": [74.25, 66.87, 46.94, 71.16, 66.94, 79.99, 79.52, 57.38],
    "Prompt F": [57.11, 52.07, 54.09, 66.90, 51.41, 66.41, 58.30, 56.83],
}
"""




# Create DataFrame for the first sheet
df_models = pd.DataFrame(data)

# Calculate the best prompt for each model
df_models["Best Prompt"] = df_models.iloc[:, 1:-1].idxmax(axis=1)
df_models["Best Score"] = df_models.iloc[:, 1:-1].max(axis=1)

# Create a comparison DataFrame for the second sheet
comparison_data = {
    "Model": df_models["Model"],
    "Best Prompt": df_models["Best Prompt"],
    "Best Score": df_models["Best Score"],
}
comparison_df = pd.DataFrame(comparison_data)

# Generate Excel file
file_path = "./model_scores_comparison_3.xlsx"
with pd.ExcelWriter(file_path) as writer:
    df_models.to_excel(writer, sheet_name="Model Scores", index=False)
    comparison_df.to_excel(writer, sheet_name="Best Prompt Comparison", index=False)


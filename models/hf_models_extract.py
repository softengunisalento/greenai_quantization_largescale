import csv
from huggingface_hub import list_models

# Define the number of models to fetch
NUM_MODELS = 50
CSV_FILE = 'huggingface_models.csv'
SORT = "likes7d"

# Fetch models from Hugging Face Hub
models = list_models(sort=SORT, direction=-1, limit=NUM_MODELS)

# Create a CSV file and write model names and tasks
with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
		writer = csv.writer(file)
		# Write the header
		writer.writerow(['Model Name', 'First Task'])

		# Write the model data
		for model in models:
				model_name = model.modelId
				tasks = model.pipeline_tag if model.pipeline_tag else 'Unknown'	# Check if 'pipeline_tag' exists
				writer.writerow([model_name, tasks])

print(f'Successfully saved the first {NUM_MODELS} models and their tasks to {CSV_FILE}.')

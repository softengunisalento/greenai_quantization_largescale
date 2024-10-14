import csv
from huggingface_hub import list_models

# Define the number of models to fetch
num_models = 50

# Fetch models from Hugging Face Hub
models = list_models(sort="downloads", direction=-1, limit=50)

# Create a CSV file and write model names and tasks
csv_file = 'huggingface_models.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
		writer = csv.writer(file)
		# Write the header
		writer.writerow(['Model Name', 'First Task'])

		# Write the model data
		for model in models:
				model_name = model.modelId
				tasks = model.pipeline_tag if model.pipeline_tag else 'Unknown'	# Check if 'pipeline_tag' exists
				writer.writerow([model_name, tasks])

print(f'Successfully saved the first {num_models} models and their tasks to {csv_file}.')

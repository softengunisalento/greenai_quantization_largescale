from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
import tempfile
from tqdm import tqdm
import pandas as pd
import csv

### Constants
MODEL_ID = "BAAI/bge-base-en-v1.5"
DATASET_ID = "LLukas22/nq-simplified"
TMP_FNAME = "LLukas22_nq_simplified"
SPECIFIC_FILE = "test.json"
FILE_NAME = "baai_bge_base_en_v1_5"
CSV_FILE = "db_logits_" + FILE_NAME + ".csv"

### Functions
def load_json_file(dataset_id, file_name):
	""" Function to extract and process the specific file """
	# Create a consistent temp directory name
	temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
	existing_dir = None
	
	# Check if the temp directory already exists and has content
	if os.path.exists(temp_dir) and os.listdir(temp_dir):
		print(f"Found existing temporary directory: {temp_dir}")
		# Try to find the extracted directory
		for item in os.listdir(temp_dir):
			item_path = os.path.join(temp_dir, item)
			if os.path.isdir(item_path):
				existing_dir = item_path
				print(f"Using existing directory: {existing_dir}")
				break
	
	# If no extracted directory found, we need to download and extract
	if existing_dir is None:
		# Download the specific file with tqdm progress bar
		from huggingface_hub import hf_hub_download

		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename=file_name,
			repo_type="dataset",
			local_files_only=False,
			resume_download=True
		)
		
		# Create the temp directory if it doesn't exist
		os.makedirs(temp_dir, exist_ok=True)
	
		print(f"Downloaded file to: {file_path}")

		# Set existing_dir to temp_dir since we're using the downloaded file directly
		existing_dir = temp_dir
	
	# Now we need to find the actual path to the JSON file
	if file_path is None:
		# Try to find the file in the existing directory
		for root, dirs, files in os.walk(existing_dir):
			for file in files:
				if file == file_name:
					file_path = os.path.join(root, file)
					break
			if file_path:
				break
	
	if not file_path:
		print(f"Could not find {file_name} in the directory structure")
		exit()
	
	# Load the JSON data
	data = []
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			data = [json.loads(line) for line in f]
		print(f"Loaded {len(data)} records from {file_name}")
	except Exception as e:
		print(f"Error loading {file_name}: {e}")
	
	return data

def csv_results():
	""" Print MSE statistics """
	mse_series = pd.Series(list(mse_values.values()))

	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Record index", "MSE", "Max Difference"])
		
		for idx in mse_values:
			writer.writerow([idx, mse_values[idx], max_diff_values[idx]])
		
		writer.writerow(["Min MSE", "Max MSE", "Mean MSE", "Median MSE", "Standard Deviation"])
		writer.writerow([mse_series.min(), mse_series.max(), mse_series.mean(), mse_series.median(), mse_series.std()])

	print(f"Results saved to {CSV_FILE}")

# Load the tokenizer and model from sentence-transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
	model,  # the model instance
	{torch.nn.Linear},  # specify the layers to quantize
	dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Load dataset
print(f"Processing {SPECIFIC_FILE}...")
json_files = load_json_file(DATASET_ID, SPECIFIC_FILE)

if not json_files:
	print(f"No json files found in {SPECIFIC_FILE}")
	exit()

# Dictionary to store MSE values for each file
mse_values = {}
max_diff_values = {}

# Process each json file
print(f"Found {len(json_files)} json files, processing...")
for idx, json_file in enumerate(tqdm(json_files, desc="Processing json files")):
	question_array = json_file["question"]
	
	# Prepare input data
	inputs = tokenizer(
		question_array, 
		padding=True, 
		truncation=True, 
		return_tensors="pt"
	)
	
	# Forward pass through the original model
	with torch.no_grad():
		original_logits = model(**inputs)

	# Extract last hidden state from the original outputs
	original_hidden_states = original_logits.last_hidden_state
	
	# Forward pass through the quantized model
	with torch.no_grad():
		quantized_logits = model_quantized(**inputs)

	# Extract last hidden state from the quantized outputs
	quantized_hidden_states = quantized_logits.last_hidden_state
	
	# Calculate the mean squared error (MSE) between the logits
	mse = torch.mean((original_hidden_states - quantized_hidden_states) ** 2).item()
	
	# Calculate the maximum absolute difference
	max_diff = torch.max(torch.abs(original_hidden_states - quantized_hidden_states)).item()
	
	# Store values for later analysis
	mse_values[idx] = mse
	max_diff_values[idx] = max_diff

csv_results()
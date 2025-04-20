from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import json
import torch
import argparse
import tempfile
import pandas as pd


### Constants
MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
DATASET_ID = "LLukas22/nq-simplified"
TMP_FNAME = "LLukas22_nq_simplified"
SPECIFIC_FILE = "test.json"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class QuestionDataset(Dataset):
	def __init__(self, data, tokenizer, max_length):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		question = self.data[idx]["question"]
		return {
			"question": question,
			"idx": idx
		}

def collate_fn(batch):
	"""Custom collate function for DataLoader"""
	questions = [item["question"] for item in batch]
	indices = [item["idx"] for item in batch]
	return {
		"questions": questions,
		"indices": indices
	}

def load_json_file(dataset_id, file_name):
	""" Function to extract and process the specific file """
	# Create a consistent temp directory name
	temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
	existing_dir = None
	file_path = None
	
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

def csv_results(mse_values, max_diff_values):
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

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	
	# Load the tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID)
	
	# Set the device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Move the original model to the correct device
	model = model.to(device)
	
	# Apply dynamic quantization to the model
	if args.quantization_bits == 4:
		model_quantized = AutoModelForSequenceClassification.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
		model_quantized = AutoModelForMaskedLM.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map=device
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")
	
	print(f"Models loaded to {device}")

	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	json_data = load_json_file(DATASET_ID, SPECIFIC_FILE)

	if not json_data:
		print(f"No data found in {SPECIFIC_FILE}")
		exit()

	# Create dataset and dataloader
	dataset = QuestionDataset(json_data, tokenizer, args.max_lenght_tokenizer)
	
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True, 
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device.type == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True
	)

	# Dictionary to store MSE values for each file
	mse_values = {}
	max_diff_values = {}

	# Process batches
	print(f"Running {len(dataset)} records inference with batch size {args.batch_size} on {device}")

	for batch in tqdm(dataloader, desc="Comparing model outputs"):
		questions = batch["questions"]
		indices = batch["indices"]
		
		# Tokenize input batch
		inputs = tokenizer(
			questions, 
			padding=True, 
			truncation=True, 
			max_length=args.max_lenght_tokenizer,
			return_tensors="pt"
		).to(device)
		
		# Forward pass through both models
		with torch.no_grad():
			original_logits = model(**inputs)

			quantized_logits = model_quantized(
				**inputs,
				output_hidden_states=True
			)

		# Extract last hidden states 
		original_hidden_states = original_logits.last_hidden_state
		quantized_hidden_states = quantized_logits.hidden_states[-1]
		
		# Calculate MSE and max difference for each example in the batch
		for i, idx in enumerate(indices):
			# Calculate the mean squared error (MSE) between the outputs
			mse = torch.mean((original_hidden_states[i] - quantized_hidden_states[i]) ** 2).item()
			
			# Calculate the maximum absolute difference
			max_diff = torch.max(torch.abs(original_hidden_states[i] - quantized_hidden_states[i])).item()
			
			# Store values for later analysis
			mse_values[idx] = mse
			max_diff_values[idx] = max_diff

	csv_results(mse_values, max_diff_values)

	# Free cache
	torch.cuda.empty_cache()


if __name__ == "__main__":
	main()
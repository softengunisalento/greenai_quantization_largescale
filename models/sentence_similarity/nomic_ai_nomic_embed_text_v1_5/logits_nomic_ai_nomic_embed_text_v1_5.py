from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import random
import argparse
import tempfile
import pandas as pd
from huggingface_hub import hf_hub_download


### Constants
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DATASET_ID = "mandarjoshi/trivia_qa"
TMP_FNAME = "mandarjoshi_trivia_qa"
SPECIFIC_FILE = "rc/test-00000-of-00004.parquet"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class TriviaQADataset(Dataset):
	"""Dataset for text embedding evaluation"""
	
	def __init__(self, df, tokenizer, max_length):
		# Use question or question + context as input
		self.texts = df['question'].tolist()
		self.tokenizer = tokenizer
		self.max_length = max_length
	
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = self.texts[idx]
		
		# Skip if text is too short (handled in collate_fn)
		if len(text.split()) < 3:
			return None
			
		# Tokenize the text
		inputs = self.tokenizer(
			text, 
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		)
		
		input_ids = inputs["input_ids"][0]
		attention_mask = inputs["attention_mask"][0]
		
		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"text": text
		}

def collate_fn(batch):
	"""Custom collate function to filter out None values"""
	# Filter out None values (skipped examples)
	batch = [item for item in batch if item is not None]
	
	if not batch:
		return None
	
	# Get max length in this batch
	max_length = max(len(item["input_ids"]) for item in batch)
	
	# Prepare lists for batched data
	input_ids = []
	attention_masks = []
	texts = []
	
	# Pad sequences
	for item in batch:
		# Get current length
		curr_len = len(item["input_ids"])
		padding_len = max_length - curr_len
		
		# Pad input_ids (ensure we're using long tensors)
		padded_input_ids = torch.cat([
			item["input_ids"], 
			torch.tensor([tokenizer.pad_token_id] * padding_len, dtype=torch.long)
		])
		input_ids.append(padded_input_ids)
		
		# Pad attention_mask
		padded_attention_mask = torch.cat([
			item["attention_mask"],
			torch.zeros(padding_len, dtype=torch.long)
		])
		attention_masks.append(padded_attention_mask)
		
		# Store original text
		texts.append(item["text"])
	
	return {
		"input_ids": torch.stack(input_ids),
		"attention_mask": torch.stack(attention_masks),
		"texts": texts
	}

def load_parquet_file(dataset_id, file_name):
	""" Function to extract and process the specific parquet file """
	# Create a consistent temp directory name
	temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
	file_path = None
	
	# Check if the temp directory already exists and has content
	if os.path.exists(temp_dir) and os.listdir(temp_dir):
		print(f"Found existing temporary directory: {temp_dir}")
		# Try to find the file in the existing directory
		for root, dirs, files in os.walk(temp_dir):
			for file in files:
				if file == os.path.basename(file_name):
					file_path = os.path.join(root, file)
					print(f"Found existing file: {file_path}")
					break
			if file_path:
				break
	
	# If no file found, we need to download
	if file_path is None:
		# Download the specific file with progress
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
	
	if not file_path:
		print(f"Could not find {file_name} in the directory structure")
		exit()
	
	# Load the parquet data
	try:
		df = pd.read_parquet(file_path)
		print(f"Loaded {len(df)} records from {file_name}")
		return df
	except Exception as e:
		print(f"Error loading {file_name}: {e}")
		return None

def csv_results(mse_values, max_diff_values):
	""" Save results to CSV """
	mse_series = pd.Series(list(mse_values.values()))

	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		# writer.writerow(["Record index", "MSE", "Max Difference"])
		
		# for idx in mse_values:
		# 	writer.writerow([idx, mse_values[idx], max_diff_values[idx]])
		
		writer.writerow(["Min MSE", "Max MSE", "Mean MSE", "Median MSE", "Standard Deviation"])
		writer.writerow([mse_series.min(), mse_series.max(), mse_series.mean(), mse_series.median(), mse_series.std()])

	print(f"Results saved to {CSV_FILE}")

def cosine_similarity(tensor1, tensor2):
	"""Calculate cosine similarity between two tensors"""
	# Normalize the tensors
	tensor1_norm = tensor1 / tensor1.norm(dim=-1, keepdim=True)
	tensor2_norm = tensor2 / tensor2.norm(dim=-1, keepdim=True)
	
	# Calculate cosine similarity
	return (tensor1_norm * tensor2_norm).sum(dim=-1)

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="Nomic Embed model comparison with quantization")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_length_tokenizer', type=int, default=512, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='quantization bits (8 for quantization)')
	return parser.parse_args()


### Main
def main():
	global tokenizer  # Make tokenizer global so collate_fn can access it
	
	# Load the tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, safe_serialization=True)
	
	# Set the device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Move the original model to the correct device
	model = model.to(device)
	
	# Apply dynamic quantization to the model
	if args.quantization_bits == 4:
		model_quantized = AutoModel.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map="auto",
			trust_remote_code=True
		)
	elif args.quantization_bits == 8:
		model_quantized = AutoModel.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map="auto",
			trust_remote_code=True
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")
	
	print(f"Models loaded to {device}")
	
	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
	
	if df is None or len(df) == 0:
		print(f"No data found in {SPECIFIC_FILE}")
		exit()
	
	# Create dataset and dataloader
	dataset = TriviaQADataset(df, tokenizer, args.max_length_tokenizer)
	
	dataloader = DataLoader(
		dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device.type == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True if args.num_workers > 0 else False
	)
	
	# Dictionaries to store comparison values for each record
	mse_values = {}
	max_diff_values = {}
	counter = 0
	
	print(f"Running inference with batch size {args.batch_size} on {device}")
	
	# Get the actual device where model_quantized is located
	if hasattr(model_quantized, 'device'):
		quant_device = model_quantized.device
	else:
		# For models with device_map, we can check the device of the first parameter
		first_param_device = next(model_quantized.parameters()).device
		quant_device = first_param_device
	
	print(f"Original model on: {device}, Quantized model on: {quant_device}")
	
	for batch in tqdm(dataloader, desc="Comparing model outputs"):
		# Skip empty batches
		if batch is None:
			continue
			
		# Move data to appropriate devices
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		
		# Move tensors to the same device as the quantized model
		input_ids_quant = batch["input_ids"].to(quant_device)
		attention_mask_quant = batch["attention_mask"].to(quant_device)

		# Forward pass through both models
		with torch.no_grad():
			original_outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)
			
			quantized_outputs = model_quantized(
				input_ids=input_ids_quant,
				attention_mask=attention_mask_quant
			)
		
		# Extract embeddings (these could be last_hidden_state or pooled output depending on model)
		if hasattr(original_outputs, 'last_hidden_state'):
			original_embeddings = original_outputs.last_hidden_state
			quantized_embeddings = quantized_outputs.last_hidden_state
		else:
			# If the model returns a tuple, assume the first element is the embedding
			original_embeddings = original_outputs[0]
			quantized_embeddings = quantized_outputs[0]
		
		# Move quantized embeddings to the same device as original for comparison
		quantized_embeddings = quantized_embeddings.to(device)
		
		# Calculate metrics for each example in the batch
		for i in range(len(input_ids)):
			# Calculate the mean squared error (MSE) between the outputs
			mse = torch.mean((original_embeddings[i] - quantized_embeddings[i]) ** 2).item()
			
			# Calculate the maximum absolute difference
			max_diff = torch.max(torch.abs(original_embeddings[i] - quantized_embeddings[i])).item()
			
			# Store values for later analysis
			mse_values[counter] = mse
			max_diff_values[counter] = max_diff
			counter += 1
	
	# Save results to CSV
	csv_results(mse_values, max_diff_values)

	# Free cache
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


if __name__ == "__main__":
	args = arg_commandline()
	main()
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import random
import argparse
import tempfile
import pandas as pd


### Constants
MODEL_ID = "google-bert/bert-base-multilingual-cased"
DATASET_ID = "legacy-datasets/wikipedia"
TMP_FNAME = "legacy_datasets_wikipedia"
SPECIFIC_FILE = "train-00009-of-00041.parquet"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"
MASK_PROB = 0.15  # Probability to mask a token


### Functions
class MaskedDataset(Dataset):
	"""Dataset for masked language modeling task"""
	
	def __init__(self, df, tokenizer):
		self.texts = df['text'].tolist()
		self.tokenizer = tokenizer
	
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = self.texts[idx]
		
		# Skip if text is too short (handled in collate_fn)
		if len(text.split()) < 5:
			return None
			
		# Tokenize the text
		inputs = self.tokenizer(
			text, 
			truncation=True,
			max_length=args.max_lenght_tokenizer,
			return_tensors="pt"
		)
		
		input_ids = inputs["input_ids"][0].tolist()
		attention_mask = inputs["attention_mask"][0].tolist()

		# Don't mask special tokens
		special_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
		
		# Get positions of tokens that can be masked (not special tokens)
		maskable_positions = [i for i, token_id in enumerate(input_ids) 
							if token_id not in special_tokens]
		
		# Skip if not enough maskable tokens (handled in collate_fn)
		if len(maskable_positions) < 3:
			return None
			
		# Randomly select positions to mask based on probability
		num_to_mask = max(1, int(len(maskable_positions) * MASK_PROB))
		positions_to_mask = random.sample(maskable_positions, num_to_mask)
		
		# Create masked version and labels
		masked_input_ids = input_ids.copy()
		labels = [-100] * len(input_ids) # -100 means don't compute loss for this position
		
		for pos in positions_to_mask:
			# Record the original token for the label
			labels[pos] = input_ids[pos]
			# Replace with [MASK] token
			masked_input_ids[pos] = self.tokenizer.mask_token_id
		
		# Convert to tensors (ensure long type for input_ids)
		return {
			"input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_mask, dtype=torch.long),
			"labels": torch.tensor(labels, dtype=torch.long),
			"mask_positions": positions_to_mask
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
	labels = []
	mask_positions_list = []
	
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
		
		# Pad labels
		padded_labels = torch.cat([
			item["labels"],
			torch.tensor([-100] * padding_len, dtype=torch.long)
		])
		labels.append(padded_labels)
		
		# No padding needed for mask positions
		mask_positions_list.append(item["mask_positions"])
	
	return {
		"input_ids": torch.stack(input_ids),
		"attention_mask": torch.stack(attention_masks),
		"labels": torch.stack(labels),
		"mask_positions": mask_positions_list
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
				if file == file_name:
					file_path = os.path.join(root, file)
					print(f"Found existing file: {file_path}")
					break
			if file_path:
				break
	
	# If no file found, we need to download
	if file_path is None:
		# Download the specific file with progress
		from huggingface_hub import hf_hub_download

		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename=f"data/20220301.en/{file_name}",
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
	parser.add_argument('-b', '--batch_size', type=int, default=128, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	
	global tokenizer # Make tokenizer global so collate_fn can access it
	
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
	df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
	
	if df is None or len(df) == 0:
		print(f"No data found in {SPECIFIC_FILE}")
		exit()
	
	# Create dataset and dataloader
	dataset = MaskedDataset(df, tokenizer)
	
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
	
	# Dictionary to store MSE values for each record
	mse_values = {}
	max_diff_values = {}
	counter = 0
	
	print(f"Running fill-mask inference with batch size {args.batch_size} on {device}")
	
	for batch in tqdm(dataloader, desc="Comparing model outputs"):
		# Skip empty batches
		if batch is None:
			continue
			
		# Move data to device
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		
		# Forward pass through both models
		with torch.no_grad():
			original_outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)
			
			quantized_outputs = model_quantized(
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True
			)
		
		# Extract last hidden states
		original_hidden_states = original_outputs.last_hidden_state
		quantized_hidden_states = quantized_outputs.hidden_states[-1]
		
		# Calculate MSE and max difference for each example in the batch
		for i in range(len(input_ids)):
			# Calculate the mean squared error (MSE) between the outputs
			mse = torch.mean((original_hidden_states[i] - quantized_hidden_states[i]) ** 2).item()
			
			# Calculate the maximum absolute difference
			max_diff = torch.max(torch.abs(original_hidden_states[i] - quantized_hidden_states[i])).item()
			
			# Store values for later analysis
			mse_values[counter] = mse
			max_diff_values[counter] = max_diff
			counter += 1
	
	# Save results to CSV
	csv_results(mse_values, max_diff_values)

	# Free cache
	torch.cuda.empty_cache()


if __name__ == "__main__":
	args = arg_commandline()
	main()
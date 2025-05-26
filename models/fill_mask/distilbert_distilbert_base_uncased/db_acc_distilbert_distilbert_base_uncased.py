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
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "legacy-datasets/wikipedia"
TMP_FNAME = "legacy_datasets_wikipedia"
SPECIFIC_FILE = "train-00009-of-00041.parquet"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"
MASK_PROB = 0.15 # Probability to mask a token


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

def calculate_accuracy(original_model, model_int4, model_int8, dataloader, tokenizer, device):
	"""Calculate MLM accuracy for all models and measure the drops"""
	original_correct = 0
	int4_correct = 0
	int8_correct = 0
	total_predictions = 0
	
	# Dictionaries to store per-example prediction correctness
	original_preds = {}
	int4_preds = {}
	int8_preds = {}
	
	example_counter = 0
	
	for batch in tqdm(dataloader, desc="Evaluating accuracy"):
		if batch is None:
			continue
		
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		labels = batch["labels"].to(device)
		mask_positions = batch["mask_positions"]
		
		with torch.no_grad():
			# Forward pass through all models
			original_outputs = original_model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels
			)
			
			int4_outputs = model_int4(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels
			)
			
			int8_outputs = model_int8(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels
			)
		
		# For each example in the batch
		for i in range(len(input_ids)):
			example_id = example_counter
			example_counter += 1
			
			# Initialize prediction lists for this example
			original_preds[example_id] = []
			int4_preds[example_id] = []
			int8_preds[example_id] = []
			
			# Get masked positions for this example
			positions = mask_positions[i]
			
			# For each masked position
			for pos in positions:
				if pos >= len(labels[i]) or labels[i][pos] == -100:  # Skip tokens that weren't masked
					continue
					
				# Get the correct token
				correct_token = labels[i][pos].item()
				
				# Get model predictions
				original_logits = original_outputs.logits[i, pos]
				int4_logits = int4_outputs.logits[i, pos]
				int8_logits = int8_outputs.logits[i, pos]
				
				# Get top prediction
				original_pred = torch.argmax(original_logits).item()
				int4_pred = torch.argmax(int4_logits).item()
				int8_pred = torch.argmax(int8_logits).item()
				
				# Store prediction correctness (1 for correct, 0 for incorrect)
				original_correct_pred = 1 if original_pred == correct_token else 0
				int4_correct_pred = 1 if int4_pred == correct_token else 0
				int8_correct_pred = 1 if int8_pred == correct_token else 0
				
				# Append to example prediction lists
				original_preds[example_id].append(original_correct_pred)
				int4_preds[example_id].append(int4_correct_pred)
				int8_preds[example_id].append(int8_correct_pred)
				
				# Count correct predictions
				original_correct += original_correct_pred
				int4_correct += int4_correct_pred
				int8_correct += int8_correct_pred
				
				total_predictions += 1
	
	# Calculate accuracies
	original_accuracy = original_correct / total_predictions if total_predictions > 0 else 0
	int4_accuracy = int4_correct / total_predictions if total_predictions > 0 else 0
	int8_accuracy = int8_correct / total_predictions if total_predictions > 0 else 0
	
	# Calculate accuracy drops
	int4_drop = original_accuracy - int4_accuracy
	int8_drop = original_accuracy - int8_accuracy
	
	# Calculate per-example accuracy difference statistics
	example_int4_drops = []
	example_int8_drops = []
	
	for example_id in original_preds:
		if len(original_preds[example_id]) > 0:
			# Calculate per-example accuracy
			example_orig_acc = sum(original_preds[example_id]) / len(original_preds[example_id])
			example_int4_acc = sum(int4_preds[example_id]) / len(int4_preds[example_id])
			example_int8_acc = sum(int8_preds[example_id]) / len(int8_preds[example_id])
			
			# Calculate drops
			example_int4_drops.append(example_orig_acc - example_int4_acc)
			example_int8_drops.append(example_orig_acc - example_int8_acc)
	
	# Calculate statistics of per-example drops
	int4_drop_mean = sum(example_int4_drops) / len(example_int4_drops) if example_int4_drops else 0
	int4_drop_std = (sum((x - int4_drop_mean) ** 2 for x in example_int4_drops) / len(example_int4_drops)) ** 0.5 if example_int4_drops else 0
	
	int8_drop_mean = sum(example_int8_drops) / len(example_int8_drops) if example_int8_drops else 0
	int8_drop_std = (sum((x - int8_drop_mean) ** 2 for x in example_int8_drops) / len(example_int8_drops)) ** 0.5 if example_int8_drops else 0
	
	return {
		"original_accuracy": original_accuracy,
		"int4_accuracy": int4_accuracy,
		"int8_accuracy": int8_accuracy,
		"int4_drop": int4_drop,
		"int8_drop": int8_drop,
		"int4_drop_percentage": (int4_drop / original_accuracy) * 100 if original_accuracy > 0 else 0,
		"int8_drop_percentage": (int8_drop / original_accuracy) * 100 if original_accuracy > 0 else 0,
		"int4_drop_std": int4_drop_std,
		"int8_drop_std": int8_drop_std,
		"total_predictions": total_predictions
	}

def calculate_hidden_state_mse(original_model, model_int4, model_int8, dataloader, device):
	"""Calculate MSE and max difference between original and quantized models"""
	mse_values_int4 = {}
	mse_values_int8 = {}
	max_diff_values_int4 = {}
	max_diff_values_int8 = {}
	counter = 0
	
	for batch in tqdm(dataloader, desc="Comparing hidden states"):
		# Skip empty batches
		if batch is None:
			continue
			
		# Move data to device
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		
		# Forward pass through all models
		with torch.no_grad():
			original_outputs = original_model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True
			)
			
			int4_outputs = model_int4(
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True
			)
			
			int8_outputs = model_int8(
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True
			)
		
		# Extract last hidden states
		original_hidden_states = original_outputs.hidden_states[-1]
		int4_hidden_states = int4_outputs.hidden_states[-1]
		int8_hidden_states = int8_outputs.hidden_states[-1]
		
		# Calculate MSE and max difference for each example in the batch
		for i in range(len(input_ids)):
			# Calculate MSE for int4
			mse_int4 = torch.mean((original_hidden_states[i] - int4_hidden_states[i]) ** 2).item()
			max_diff_int4 = torch.max(torch.abs(original_hidden_states[i] - int4_hidden_states[i])).item()
			
			# Calculate MSE for int8
			mse_int8 = torch.mean((original_hidden_states[i] - int8_hidden_states[i]) ** 2).item()
			max_diff_int8 = torch.max(torch.abs(original_hidden_states[i] - int8_hidden_states[i])).item()
			
			# Store values
			mse_values_int4[counter] = mse_int4
			mse_values_int8[counter] = mse_int8
			max_diff_values_int4[counter] = max_diff_int4
			max_diff_values_int8[counter] = max_diff_int8
			counter += 1
	
	return {
		"mse_values_int4": mse_values_int4,
		"mse_values_int8": mse_values_int8,
		"max_diff_values_int4": max_diff_values_int4,
		"max_diff_values_int8": max_diff_values_int8
	}

def csv_results(mse_results, accuracy_results):
	""" Print MSE statistics and accuracy results """
	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# Write accuracy results first (most important)
		writer.writerow(["Accuracy Metrics"])
		writer.writerow(["Model", "Accuracy", "Accuracy Drop", "Accuracy Drop (%)", "Drop Std Dev"])
		writer.writerow(["Original", accuracy_results["original_accuracy"], "N/A", "N/A", "N/A"])
		writer.writerow(["Int4", accuracy_results["int4_accuracy"], 
						 accuracy_results["int4_drop"], 
						 accuracy_results["int4_drop_percentage"],
						 accuracy_results["int4_drop_std"]])
		writer.writerow(["Int8", accuracy_results["int8_accuracy"], 
						 accuracy_results["int8_drop"], 
						 accuracy_results["int8_drop_percentage"],
						 accuracy_results["int8_drop_std"]])
		writer.writerow(["Total predictions", accuracy_results["total_predictions"]])
		
		writer.writerow([])  # Empty row for separation
		
		if mse_results:
			# MSE statistics
			mse_int4_series = pd.Series(list(mse_results["mse_values_int4"].values()))
			mse_int8_series = pd.Series(list(mse_results["mse_values_int8"].values()))
			
			writer.writerow(["Hidden State MSE Metrics"])
			writer.writerow(["Metric", "Int4", "Int8", "Difference (Int4-Int8)"])
			writer.writerow(["Min MSE", mse_int4_series.min(), mse_int8_series.min(), 
							mse_int4_series.min() - mse_int8_series.min()])
			writer.writerow(["Max MSE", mse_int4_series.max(), mse_int8_series.max(), 
							mse_int4_series.max() - mse_int8_series.max()])
			writer.writerow(["Mean MSE", mse_int4_series.mean(), mse_int8_series.mean(), 
							mse_int4_series.mean() - mse_int8_series.mean()])
			writer.writerow(["Median MSE", mse_int4_series.median(), mse_int8_series.median(), 
							mse_int4_series.median() - mse_int8_series.median()])
			writer.writerow(["Std Dev MSE", mse_int4_series.std(), mse_int8_series.std(), 
							mse_int4_series.std() - mse_int8_series.std()])
			
			# Max difference statistics
			max_diff_int4_series = pd.Series(list(mse_results["max_diff_values_int4"].values()))
			max_diff_int8_series = pd.Series(list(mse_results["max_diff_values_int8"].values()))
			
			writer.writerow([])  # Empty row for separation
			writer.writerow(["Hidden State Max Difference Metrics"])
			writer.writerow(["Metric", "Int4", "Int8", "Difference (Int4-Int8)"])
			writer.writerow(["Min Max-Diff", max_diff_int4_series.min(), max_diff_int8_series.min(), 
							max_diff_int4_series.min() - max_diff_int8_series.min()])
			writer.writerow(["Max Max-Diff", max_diff_int4_series.max(), max_diff_int8_series.max(), 
							max_diff_int4_series.max() - max_diff_int8_series.max()])
			writer.writerow(["Mean Max-Diff", max_diff_int4_series.mean(), max_diff_int8_series.mean(), 
							max_diff_int4_series.mean() - max_diff_int8_series.mean()])
			writer.writerow(["Median Max-Diff", max_diff_int4_series.median(), max_diff_int8_series.median(), 
							max_diff_int4_series.median() - max_diff_int8_series.median()])
			writer.writerow(["Std Dev Max-Diff", max_diff_int4_series.std(), max_diff_int8_series.std(), 
							max_diff_int4_series.std() - max_diff_int8_series.std()])

	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="Compare original, int4, and int8 quantized models")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')

	return parser.parse_args()


### Main
def main():
	global args, tokenizer
	args = arg_commandline()
	
	# Load the tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	
	# Set the device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Load the original model
	print(f"Loading original model...")
	model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
	model = model.to(device)
	
	# Load 4-bit quantized model
	print(f"Loading 4-bit quantized model...")
	model_int4 = AutoModelForMaskedLM.from_pretrained(
		MODEL_ID,
		load_in_4bit=True,
		device_map=device
	)
	
	# Load 8-bit quantized model
	print(f"Loading 8-bit quantized model...")
	model_int8 = AutoModelForMaskedLM.from_pretrained(
		MODEL_ID,
		load_in_8bit=True,
		device_map=device
	)
	
	print(f"All models loaded to {device}")
	
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
	
	# Calculate hidden state MSE
	print("Calculating hidden state differences...")
	mse_results = calculate_hidden_state_mse(model, model_int4, model_int8, dataloader, device)
	
	# Calculate accuracy drop for both quantization methods
	print("Calculating accuracy drops...")
	accuracy_results = calculate_accuracy(model, model_int4, model_int8, dataloader, tokenizer, device)
	
	# Print results to console
	print("\nAccuracy Results:")
	print(f"Original model accuracy: {accuracy_results['original_accuracy']:.4f}")
	print(f"Int4 model accuracy: {accuracy_results['int4_accuracy']:.4f} (drop: {accuracy_results['int4_drop']:.4f}, {accuracy_results['int4_drop_percentage']:.2f}%)")
	print(f"Int8 model accuracy: {accuracy_results['int8_accuracy']:.4f} (drop: {accuracy_results['int8_drop']:.4f}, {accuracy_results['int8_drop_percentage']:.2f}%)")
	print(f"Total masked token predictions: {accuracy_results['total_predictions']}")
	
	# Save results to CSV
	csv_results(mse_results, accuracy_results)
	
	# Generate a comparison report
	if accuracy_results["int4_accuracy"] > accuracy_results["int8_accuracy"]:
		print("\nQuantization Comparison: Int4 outperforms Int8 in accuracy!")
	elif accuracy_results["int8_accuracy"] > accuracy_results["int4_accuracy"]:
		print("\nQuantization Comparison: Int8 outperforms Int4 in accuracy!")
	else:
		print("\nQuantization Comparison: Int4 and Int8 have identical accuracy.")
		
	if mse_results:
		int4_mse_mean = pd.Series(list(mse_results["mse_values_int4"].values())).mean()
		int8_mse_mean = pd.Series(list(mse_results["mse_values_int8"].values())).mean()
		
		if int4_mse_mean < int8_mse_mean:
			print("Int4 has lower mean MSE than Int8, suggesting better hidden state preservation.")
		else:
			print("Int8 has lower mean MSE than Int4, suggesting better hidden state preservation.")

	# Free cache
	torch.cuda.empty_cache()


if __name__ == "__main__":
	args = arg_commandline()
	main()
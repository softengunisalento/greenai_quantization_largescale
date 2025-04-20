from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import time
import torch
import argparse
import tempfile
import pandas as pd
from codecarbon import EmissionsTracker


### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "google/electra-base-discriminator"
DATASET_ID = "ade-benchmark-corpus/ade_corpus_v2"
TMP_FNAME = "ade_benchmark_corpus_ade_corpus_v2"
SPECIFIC_FILE = "train-00000-of-00001.parquet"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class ADEDataset(Dataset):
	"""Dataset for ADE classification"""
	
	def __init__(self, dataframe, tokenizer, max_length=128):
		self.tokenizer = tokenizer
		# Assuming the dataframe has 'text' and 'label' columns
		# Adjust column names as needed based on the actual dataset
		self.texts = dataframe['text'].tolist()
		self.labels = dataframe['label'].tolist() if 'label' in dataframe.columns else [0] * len(dataframe)
		self.max_length = max_length
		
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = self.texts[idx]
		label = self.labels[idx]
		
		# Skip if text is too short
		if len(text.split()) < 5:
			# Return a dummy sample that will be filtered later
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
			"labels": torch.tensor(label, dtype=torch.long),
			"text": text
		}

def collate_fn(batch):
	"""Custom collate function to handle None values and pad sequences"""
	# Filter out None values
	batch = [item for item in batch if item is not None]
	
	if not batch:
		return None
	
	# Get max length in this batch
	max_length = max(len(item["input_ids"]) for item in batch)
	
	# Prepare lists for batched data
	input_ids = []
	attention_masks = []
	labels = []
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
		
		# No padding needed for labels
		labels.append(item["labels"])
		
		# No padding needed for text
		texts.append(item["text"])
	
	return {
		"input_ids": torch.stack(input_ids),
		"attention_mask": torch.stack(attention_masks),
		"labels": torch.stack(labels),
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
		from huggingface_hub import hf_hub_download

		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename=f"Ade_corpus_v2_classification/{file_name}",
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

def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def calculate_co2_emissions(energy_joules):
	""" Calculate CO2 emissions (in grams) """
	# Convert joules to megajoules (1 MJ = 1e6 J)
	energy_megajoules = energy_joules / 1e6
	return energy_megajoules * CARBON_INTENSITY

def measure_energy_consumption(func):
	""" Decorator to measure energy consumption of a function """
	def wrapper(*args, **kwargs):
		# Start the CodeCarbon tracker
		tracker = EmissionsTracker(
			log_level="error",
			save_to_file=False
		)
		tracker.start()
		
		# Start the inference tracker
		start_time = time.time()
		
		# Call the function
		result = func(*args, **kwargs)
		
		# Measure the after inference time
		end_time = time.time()
		inference_time = end_time - start_time
		
		# Stop the CodeCarbon tracker
		tracker.stop()
		
		co2_emissions = calculate_co2_emissions(tracker._total_energy.kWh)
		
		# kWh to Wh
		cpu_energy_Wh = tracker._total_cpu_energy.kWh * 1000
		gpu_energy_Wh = tracker._total_gpu_energy.kWh * 1000
		ram_energy_Wh = tracker._total_ram_energy.kWh * 1000
		total_energy_Wh = tracker._total_energy.kWh * 1000
		
		with open(CSV_FILE, mode='a', newline='') as file:
			writer = csv.writer(file)
			
			# Write metrics
			writer.writerow(["Inference s", "CPU Wh", "GPU Wh", "RAM Wh", "Energy Wh", "CO2 Emissions gr"])
			writer.writerow([inference_time, cpu_energy_Wh, gpu_energy_Wh, ram_energy_Wh, total_energy_Wh, co2_emissions])
		
		return result
	return wrapper

def csv_results(model, results_list):
	""" Save classification results to CSV """
	# Check the memory footprint of the model
	mem_footprint = model.get_memory_footprint()
	
	# Checking the size of the model
	module_sizes = evaluate_size(model)
	
	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)

		# Write files stats
		writer.writerow(["Example ID", "Text", "Predicted Label", "Confidence Score"])
		for result in results_list:
			writer.writerow([
				result["id"], 
				result["text"], 
				result["predicted_label"],
				result["confidence_score"]
			])
		
		# Write metrics
		writer.writerow(["Footprint MBs", "Size GB"])
		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])
	
	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	
	global tokenizer  # Make tokenizer global so collate_fn can access it

	# Load the tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID, num_labels=2)  # Adjust num_labels based on dataset
	
	# Move to GPU if available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)

	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
	
	if df is None or df.empty:
		print(f"No data found in {SPECIFIC_FILE}")
		return
	
	# Create dataset and dataloader
	dataset = ADEDataset(df, tokenizer)
	dataloader = DataLoader(
		dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True if args.num_workers > 0 else False
	)
	
	results = []
	counter = 0
	
	print(f"Running classification inference with DataLoader...")
	
	# Process batches using DataLoader
	model.eval()
	for batch in tqdm(dataloader, desc="Processing batches"):
		# Skip empty batches
		if batch is None:
			continue
			
		# Move data to device
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		labels = batch["labels"].to(device)
		texts = batch["texts"]
		
		# Forward pass
		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			
			hidden_states = outputs.last_hidden_state
			cls_output = hidden_states[:, 0, :]
			logits = torch.nn.Linear(cls_output.size(-1), 2).to(device)(cls_output)
			
		# Get predictions
		predictions = torch.softmax(logits, dim=1)
		predicted_labels = torch.argmax(predictions, dim=1)
		confidence_scores = torch.max(predictions, dim=1).values
		
		# Process each example in the batch
		for i, (predicted_label, confidence_score) in enumerate(zip(predicted_labels.cpu().tolist(), confidence_scores.cpu().tolist())):
			example_results = {
				"id": counter,
				"text": texts[i],
				"predicted_label": predicted_label,
				"confidence_score": confidence_score
			}
			counter += 1
			results.append(example_results)
	
	csv_results(model, results)

	# Free cache
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


if __name__ == "__main__":
	# Check if the -m option is provided
	args = arg_commandline()
	if args.mgpu:
		# If -m is provided, wrap main with energy measurement
		main_with_timing = measure_energy_consumption(main)
		main_with_timing()
	else:
		# Otherwise, just call main directly
		main()
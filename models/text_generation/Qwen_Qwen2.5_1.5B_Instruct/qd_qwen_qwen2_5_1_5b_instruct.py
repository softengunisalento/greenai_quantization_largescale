from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import os
import math
import torch
import csv
import time
import json
import argparse
import pandas as pd
import pandas as pd
from codecarbon import EmissionsTracker


### Constants
N_BYTES = 1 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()
N_TOKENS = 512
COMPARISON_FRACTION = 0.25


### Functions
class XP3Dataset(Dataset):
	"""Dataset for the xP3 dataset"""
	
	def __init__(self, df, tokenizer, max_length, fraction=1.0):
		# Extract text from the dataset (e.g., 'text' field or article field)
		# Use only a fraction of the dataset if specified
		if fraction < 1.0:
			n_samples = math.ceil(len(df) * fraction)
			df = df.sample(n=n_samples, random_state=42)  # Use fixed seed for reproducibility
			print(f"Using {n_samples} samples ({fraction*100:.1f}% of the dataset)")
		self.texts = df['inputs'].tolist()  # Assuming the text is stored in the 'text' column
		self.tokenizer = tokenizer
		self.max_length = max_length
	
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = self.texts[idx]
		
		# Skip if text is too short
		if len(text.split()) < 5:
			return None
			
		# Return the text and index for processing in collate_fn
		return {
			"text": text,
			"idx": idx
		}

def collate_fn(batch):
	"""Custom collate function for DataLoader"""
	# Filter out None values (skipped examples)
	batch = [item for item in batch if item is not None]
	
	if not batch:
		return None
	
	texts = [item["text"] for item in batch]
	indices = [item["idx"] for item in batch]
	
	return {
		"texts": texts,
		"indices": indices
	}

def load_jsonl_file(dataset_id, file_name):
	"""Function to extract and process the specific JSONL file"""
	# Check if the file exists in the HuggingFace dataset
	file_path = None
	
	from huggingface_hub import hf_hub_download
	file_path = hf_hub_download(
		repo_id=dataset_id,
		filename=f"en/{file_name}",
		repo_type="dataset",
		local_files_only=False,
		resume_download=True
	)
	
	# Load the JSONL data
	try:
		with open(file_path, 'r') as f:
			data = [json.loads(line) for line in f]
		df = pd.DataFrame(data)
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

def calculate_energy_consumption(power_watts, time_seconds):
	""" Calculate energy consumption in Joules """
	return power_watts * time_seconds

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

		# measure the power consumption after inference
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

		# check the memory footprint of the model
		mem_footprint = model_quantized.get_memory_footprint()

		# checking the size of the model
		module_sizes = evaluate_size(model_quantized)

		# Add energy columns to existing CSV
		try:
			# Check if file exists and has content
			file_exists = os.path.exists(CSV_FILE)
			has_content = False
			
			if file_exists:
				try:
					df = pd.read_csv(CSV_FILE)
					has_content = len(df) > 0
				except:
					has_content = False
			
			# Create new row with energy data
			energy_data = {
				'footprint_mb': mem_footprint/1e+6,
				'sizes_gb': module_sizes/1e+9,
				'inference_time_s': inference_time,
				'cpu_wh': cpu_energy_Wh,
				'gpu_wh': gpu_energy_Wh,
				'ram_wh': ram_energy_Wh,
				'total_energy_wh': total_energy_Wh,
				'co2_emissions_gr': co2_emissions
			}
			
			if has_content:
				# Read existing data and append new row
				df = pd.read_csv(CSV_FILE)
				
				# Add missing columns with NaN if they don't exist
				# for col in energy_data.keys():
				# 	if col not in df.columns:
				# 		df[col] = pd.NA
				
				# Create new row - copy last row and update with energy data
				new_row = df.iloc[-1].copy() if len(df) > 0 else pd.Series()
				for key, value in energy_data.items():
					new_row[key] = value
				
				# Append new row
				df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
			else:
				# Create new DataFrame with energy data
				df = pd.DataFrame([energy_data])
			
			# Write back to CSV
			df.to_csv(CSV_FILE, index=False)
			
		except Exception as e:
			print(f"Error adding energy data: {e}")
			# Fallback to simple append
			with open(CSV_FILE, mode='a', newline='') as file:
				writer = csv.writer(file)
				if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
					writer.writerow(["footprint_mb", "sizes_gb", "inference_time_s", "cpu_wh", "gpu_wh", "ram_wh", "total_energy_wh", "co2_emissions_gr"])
				writer.writerow([inference_time, cpu_energy_Wh, gpu_energy_Wh, ram_energy_Wh, total_energy_Wh, co2_emissions])

		return result
	return wrapper

# def csv_results(model, predictions):
# 	""" Print statistics """
# 	# check the memory footprint of the model
# 	mem_footprint = model.get_memory_footprint()
	
# 	# checking the size of the model
# 	module_sizes = evaluate_size(model)
# 	csv_exists = os.path.exists(CSV_FILE)
# 	with open(CSV_FILE, mode='a', newline='') as file:
# 		writer = csv.writer(file)

# 		# if not csv_exists:
# 		# 	writer.writerow(["Record index", "Prediction", "Comparison Fraction"])
		
# 		# for idx in predictions:
# 		# 	writer.writerow([idx, predictions[idx], COMPARISON_FRACTION])
		
# 		# write metrics
# 		if not csv_exists:
# 			writer.writerow(["footprint_mb", "sizes_gb"])
# 		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

# 	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser for quantization comparison")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=16, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_length_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int8')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	global model_quantized

	print(f"Running comparison on {COMPARISON_FRACTION*100:.1f}% of the dataset")

	# Load the tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	tokenizer.pad_token = tokenizer.eos_token
	
	# Set the device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"device used: {device}")
	
	# Apply quantization to the model
	if args.quantization_bits == 4:
		model_quantized = AutoModelForCausalLM.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
		model_quantized = AutoModelForCausalLM.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map=device
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")
	
	model_quantized.config.pad_token_id = tokenizer.pad_token_id
	
	print(f"Models loaded to {device}")
	
	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	df = load_jsonl_file(DATASET_ID, SPECIFIC_FILE)
	
	if df is None or len(df) == 0:
		print(f"No data found in {SPECIFIC_FILE}")
		exit()
	
	# Create dataset and dataloader
	dataset = XP3Dataset(df, tokenizer, args.max_length_tokenizer, fraction=COMPARISON_FRACTION)
	
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
	predictions = {}
	counter = 0
	
	print(f"Running inference with batch size {args.batch_size} on {device}")
	
	for batch in tqdm(dataloader, desc="Comparing model outputs"):
		# Skip empty batches
		if batch is None:
			continue
			
		# Move data to device
		texts = batch["texts"]
		
		# Tokenize inputs
		inputs = tokenizer(
			texts,
			padding=True,
			truncation=True,
			return_tensors="pt"
		).to(device)

		outputs = model_quantized.generate(**inputs, max_new_tokens=N_TOKENS)
		predictions[counter] = {tokenizer.decode(outputs[0], skip_special_tokens=True)}
		counter += 1



	# Free cache
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
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import os
import csv
import time
import json
import argparse
import pandas as pd
import tempfile
from codecarbon import EmissionsTracker


### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "intfloat/multilingual-e5-small"
DATASET_ID = "LLukas22/nq-simplified"
TMP_FNAME = "LLukas22_nq_simplified"
SPECIFIC_FILE = "test.json"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()


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
		mem_footprint = model.get_memory_footprint()

		# checking the size of the model
		module_sizes = evaluate_size(model)

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

# def csv_results(model, tensors_list):
# 	""" Print statistics """
# 	# check the memory footprint of the model
# 	mem_footprint = model.get_memory_footprint()
	
# 	# checking the size of the model
# 	module_sizes = evaluate_size(model)
# 	csv_exists = os.path.exists(CSV_FILE)
# 	with open(CSV_FILE, mode='a', newline='') as file:
# 		writer = csv.writer(file)
		
# 		# if not csv_exists:
# 		# 	writer.writerow(["Record index", "Tensors"])
		
# 		# for idx in tensors_list:
# 		# 	writer.writerow([idx, tensors_list[idx]])
		
# 		# write metrics
# 		if not csv_exists:
# 			writer.writerow(["footprint_mb", "sizes_gb"])
# 		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

# 	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=256, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	global model

	# Load the tokenizer and model from sentence-transformers
	model = AutoModel.from_pretrained(MODEL_ID)
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	
	# Move to GPU if not quantized
	device = 'cuda' if torch.cuda.is_available() and not any('quantized' in name for name, _ in model.named_modules()) else 'cpu'
	model = model.to(device)

	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	json_data = load_json_file(DATASET_ID, SPECIFIC_FILE)

	if not json_data:
		print(f"No json files found in {SPECIFIC_FILE}")
		exit()

	# Create dataset and dataloader
	dataset = QuestionDataset(json_data, tokenizer, args.max_lenght_tokenizer)
	
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True, 
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True
	)

	tensors_list = {}
	counter = 0

	# Process batches using DataLoader
	model.eval()
	print(f"Found {len(json_data)} json files, processing")
	for batch in tqdm(dataloader, desc="Processing batches"):
		# Skip empty batches
		if batch is None:
			continue
			
		# Move data to device (ensure correct tensor types)
		question_array = batch["questions"]
		
		# Prepare input data
		inputs = tokenizer(
			question_array, 
			padding=True, 
			truncation=True, 
			return_tensors="pt"
		).to(device)
		
		# Forward pass
		with torch.no_grad():
			model_output = model(**inputs)
			embeddings = model_output[0][:, 0]
		
		tensors_list[counter] = torch.nn.functional.normalize(embeddings, p=2, dim=1).tolist()
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
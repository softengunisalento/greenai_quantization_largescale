from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torch
import csv
import time
import json
import argparse
import pandas as pd
import pandas as pd
from codecarbon import EmissionsTracker

### Constants
N_BYTES = 1
CARBON_INTENSITY = 475
MODEL_ID = "google-t5/t5-small"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()

### Functions
class XP3Dataset(Dataset):
	def __init__(self, df, tokenizer, max_length):
		self.texts = df['inputs'].tolist()
		self.tokenizer = tokenizer
		self.max_length = max_length
	
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = self.texts[idx]
		if len(text.split()) < 5:
			return None
		return {
			"text": text,
			"idx": idx
	}

def collate_fn(batch):
	batch = [item for item in batch if item is not None]
	if not batch:
		return None
	texts = [item["text"] for item in batch]
	indices = [item["idx"] for item in batch]
	return {"texts": texts, "indices": indices}

def load_jsonl_file(dataset_id, file_name):
	from huggingface_hub import hf_hub_download
	file_path = hf_hub_download(
		repo_id=dataset_id,
		filename=f"en/{file_name}",
		repo_type="dataset",
		local_files_only=False,
		resume_download=True
	)
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
# 		# 	writer.writerow(["Record index", "Prediction"])
		
# 		# for idx in predictions:
# 		# 	writer.writerow([idx, predictions[idx]])
		
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
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_length_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int8')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	global model

	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	
	# Apply dynamic quantization to the model
	if args.quantization_bits == 4:
		model_quantized = AutoModelForSeq2SeqLM.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
		model_quantized = AutoModelForSeq2SeqLM.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map=device
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")

	print(f"Loading dataset file: {SPECIFIC_FILE}")
	df = load_jsonl_file(DATASET_ID, SPECIFIC_FILE)

	if df is None or len(df) == 0:
		print("Empty dataset")
		return

	dataset = XP3Dataset(df, tokenizer, args.max_length_tokenizer)

	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=(device.type == 'cuda'),
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True
	)

	predictions = {}
	counter = 0

	print(f"Starting inference with batch size {args.batch_size}")

	for batch in tqdm(dataloader, desc="Running T5 inference"):
		if batch is None:
			continue

		inputs = tokenizer(batch["texts"], return_tensors="pt",padding=True, truncation=True).to(device)

		outputs = model_quantized.generate(**inputs)

		for i, output in enumerate(outputs):
			predictions[counter] = tokenizer.decode(output, skip_special_tokens=True)
			counter += 1


	torch.cuda.empty_cache()

if __name__ == "__main__":
	args = arg_commandline()
	if args.mgpu:
		main_with_timing = measure_energy_consumption(main)
		main_with_timing()
	else:
		main()

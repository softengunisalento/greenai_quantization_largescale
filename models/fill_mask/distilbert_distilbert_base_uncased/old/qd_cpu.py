from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import argparse
import os
import time
import tempfile
from tqdm import tqdm
import csv
import pandas as pd
from codecarbon import EmissionsTracker
import random

### Constants
N_BYTES = 1 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "legacy-datasets/wikipedia"
TMP_FNAME = "legacy_datasets_wikipedia"
SPECIFIC_FILE = "train-00009-of-00041.parquet"
FILE_NAME = "distilbert_distilbert_base_uncased"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv" + FILE_NAME + ".csv"
MASK_PROB = 0.15  # Probability to mask a token

### Functions
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

def preprocess_for_fill_mask(texts, tokenizer, max_length=128):
	"""
	Preprocess text data for fill-mask task by randomly masking tokens
	"""
	processed_examples = []
	
	for text in tqdm(texts, desc="Preprocessing for fill-mask"):
		# Skip if text is too short
		if len(text.split()) < 5:
			continue
			
		# Tokenize the text
		inputs = tokenizer(
			text, 
			truncation=True,
			max_length=max_length,
			return_tensors="pt"
		)
		
		input_ids = inputs["input_ids"][0].tolist()
		
		# Don't mask special tokens
		special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
		
		# Get positions of tokens that can be masked (not special tokens)
		maskable_positions = [i for i, token_id in enumerate(input_ids) 
							 if token_id not in special_tokens]
		
		# Skip if not enough maskable tokens
		if len(maskable_positions) < 3:
			continue
			
		# Randomly select positions to mask based on probability
		num_to_mask = max(1, int(len(maskable_positions) * MASK_PROB))
		positions_to_mask = random.sample(maskable_positions, num_to_mask)
		
		# Create masked version and labels
		masked_input_ids = input_ids.copy()
		labels = [-100] * len(input_ids)  # -100 means don't compute loss for this position
		
		for pos in positions_to_mask:
			# Record the original token for the label
			labels[pos] = input_ids[pos]
			# Replace with [MASK] token
			masked_input_ids[pos] = tokenizer.mask_token_id
		
		# Store the example
		processed_examples.append({
			"input_ids": masked_input_ids,
			"labels": labels,
			"attention_mask": inputs["attention_mask"][0].tolist()
		})
	
	return processed_examples

def get_gpu_power():
	""" Get the power usage of the GPU """
	power_usage = os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits').read().strip()
	return power_usage

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

def verbose_measurements(model):
	""" Function to print verbose output """
	# check the memory footprint of the model
	mem_footprint = model.get_memory_footprint()
	
	# checking the size of the model
	module_sizes = evaluate_size(model)
	
	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# write stats
		writer.writerow(["Footprint MBs", "Size GB"])
		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

def measure_energy_consumption(func):
	""" Decorator to measure energy consumption of a function """
	def wrapper(*args, **kwargs):
		# measure the power consumption before and after inference
		power_before = get_gpu_power()
		start_time = time.time()
		
		# Call the function
		result = func(*args, **kwargs)
		
		# measure the power consumption after inference
		end_time = time.time()
		inference_time = end_time - start_time
		
		power_after = get_gpu_power()
		average_power_usage = (power_after + power_before) / 2
		
		energy_joules = calculate_energy_consumption(average_power_usage, inference_time)
		
		co2_emissions = calculate_co2_emissions(energy_joules)
		
		with open(CSV_FILE, mode='a', newline='') as file:
			writer = csv.writer(file)
			
			# write stats
			writer.writerow(["Inference s", "Average Power Consumption W","Energy J", "CO2 Emissions gr"])
			writer.writerow([inference_time, average_power_usage, energy_joules, co2_emissions])
		
		return result
	return wrapper

def csv_results(results_list):
	""" Save fill-mask results to CSV """
	
	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Example ID", "Masked Text", "Top Predictions"])
		
		for result in results_list:
			writer.writerow([
				result["id"], 
				result["masked_text"], 
				"; ".join([f"{p['token_str']} ({p['score']:.4f})" for p in result["predictions"]])
			])
	
	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose of basic measurement')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	# Load the tokenizer and model
	model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

	# Apply dynamic quantization to the model
	model_quantized = torch.quantization.quantize_dynamic(
		model,  # the model instance
		{torch.nn.Linear},  # specify the layers to quantize
		dtype=torch.qint8  # quantized dtype (8-bit int)
	)
	
	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
	
	if df is None or df.empty:
		print(f"No data found in {SPECIFIC_FILE}")
		exit()
	
	# Extract text from the dataframe (the main content column is 'text')
	texts = df['text'].tolist()
	
	# Preprocess texts for fill-mask task
	processed_examples = preprocess_for_fill_mask(texts, tokenizer)
	
	# Process each example
	results = []
	
	print(f"Running fill-mask inference on {len(processed_examples)} processed examples...")
	for idx, example in enumerate(tqdm(processed_examples, desc="Fill-mask inference")):
		# Convert to tensors
		input_ids = torch.tensor([example["input_ids"]])
		attention_mask = torch.tensor([example["attention_mask"]])
		
		# Get positions of mask tokens
		mask_positions = [i for i, token_id in enumerate(example["input_ids"]) 
						 if token_id == tokenizer.mask_token_id]
		
		# Skip if no masks found (shouldn't happen based on preprocessing)
		if not mask_positions:
			continue
		
		# Forward pass through the model
		with torch.no_grad():
			outputs = model_quantized(input_ids=input_ids, attention_mask=attention_mask)
			predictions = outputs.logits
		
		# Get the decoded text with masks
		masked_text = tokenizer.decode(example["input_ids"])
		
		# For each mask position, get top 5 predictions
		example_results = {
			"id": idx,
			"masked_text": masked_text,
			"predictions": []
		}
		
		for mask_pos in mask_positions:
			# Get the predicted token scores for this position
			mask_token_logits = predictions[0, mask_pos]
			
			# Get top 5 tokens
			top_5_tokens = torch.topk(mask_token_logits, 5, dim=0)
			
			# Convert to a list of predictions
			for token_id, score in zip(top_5_tokens.indices.tolist(), top_5_tokens.values.tolist()):
				token_str = tokenizer.decode([token_id])
				example_results["predictions"].append({
					"token_str": token_str,
					"token_id": token_id,
					"score": float(score),
					"position": mask_pos
				})
		
		results.append(example_results)
	
	csv_results(results)

	if args.verbose:
		# If -v is provided, call main with verbose output
		verbose_measurements(model_quantized)


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
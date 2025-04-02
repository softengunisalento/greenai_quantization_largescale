from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import os
import time
import json
import tempfile
from tqdm import tqdm
import csv

### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "BAAI/bge-small-en-v1.5"
DATASET_ID = "LLukas22/nq-simplified"
TMP_FNAME = "LLukas22_nq_simplified"
SPECIFIC_FILE = "test.json"
FILE_NAME = "baai_bge_small_en_v1_5"
CSV_FILE = "db_or_" + FILE_NAME + ".csv"

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

def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def get_gpu_power():
	""" Get the power usage of the GPU """
	power_usage = os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits').read().strip()
	return float(power_usage)

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

def csv_results(tensors_list):
	""" Print statistics """

	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Record index", "Tensors"])
		
		for idx in tensors_list:
			writer.writerow([idx, tensors_list[idx]])

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

	# Load the tokenizer and model from sentence-transformers
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID)

	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	json_files = load_json_file(DATASET_ID, SPECIFIC_FILE)

	if not json_files:
		print(f"No json files found in {SPECIFIC_FILE}")
		exit()
	

	tensors_list = {}

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
			model_output = model(**inputs)
			embeddings = model_output[0][:, 0]
		
		tensors_list[idx] = torch.nn.functional.normalize(embeddings, p=2, dim=1).tolist()

	csv_results(tensors_list)

	if args.verbose:
		# If -v is provided, call main with verbose output
		verbose_measurements(model)


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
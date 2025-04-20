from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
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
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "legacy-datasets/wikipedia"
TMP_FNAME = "legacy_datasets_wikipedia"
SPECIFIC_FILE = "train-00009-of-00041.parquet"
FILE_NAME = "distilbert_distilbert_base_uncased"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv" + FILE_NAME + ".csv"
MASK_PROB = 0.15  # Probability to mask a token
BATCH_SIZE = 512   # Batch size for DataLoader
NUM_WORKERS = 8   # Number of worker processes for data loading
BATCHES_X_WORKER = 4 # Number of batches loaded in advance by each worker
MAX_LENGTH_TOKENIZER = 128 # Maximum sequence length for tokenization

### Dataset and DataLoader Classes
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
			max_length=MAX_LENGTH_TOKENIZER,
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
		labels = [-100] * len(input_ids)  # -100 means don't compute loss for this position
		
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
			"masked_text": self.tokenizer.decode(masked_input_ids)
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
	masked_texts = []
	
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
		
		# No padding needed for text
		masked_texts.append(item["masked_text"])
	
	return {
		"input_ids": torch.stack(input_ids),
		"attention_mask": torch.stack(attention_masks),
		"labels": torch.stack(labels),
		"masked_texts": masked_texts
	}

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

def get_gpu_power(gpu_id=0):
	""" Get the power usage of the GPU """
	power_usage = os.popen(f'nvidia-smi --id={gpu_id} --query-gpu=power.draw --format=csv,noheader,nounits').read().strip()
	if not power_usage:
		return 0.0
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
		# Get the current GPU ID if it exists in kwargs
		gpu_id = kwargs.get('gpu_id', 0)
		
		# Check if CUDA is available
		if not torch.cuda.is_available():
			print("WARNING: CUDA is not available, energy measurements may not be accurate")
		
		# Measure the power consumption before and after inference
		power_readings = []
		
		def record_power():
			power_readings.append(get_gpu_power(gpu_id))
		
		# Start background power monitoring thread
		import threading
		import time
		
		stop_monitoring = False
		
		def power_monitor():
			while not stop_monitoring:
				record_power()
				time.sleep(0.1)  # Sample every 100ms
		
		monitor_thread = threading.Thread(target=power_monitor)
		monitor_thread.daemon = True
		monitor_thread.start()
		
		start_time = time.time()
		
		# Call the function
		result = func(*args, **kwargs)
		
		# Stop power monitoring
		stop_monitoring = True
		monitor_thread.join(timeout=1.0)
		
		end_time = time.time()
		inference_time = end_time - start_time
		
		# Calculate average power usage
		if power_readings:
			average_power_usage = sum(power_readings) / len(power_readings)
		else:
			average_power_usage = 0
			print("WARNING: No power readings collected")
		
		energy_joules = calculate_energy_consumption(average_power_usage, inference_time)
		co2_emissions = calculate_co2_emissions(energy_joules)
		
		with open(CSV_FILE, mode='a', newline='') as file:
			writer = csv.writer(file)
			
			# write stats
			writer.writerow(["Inference s", "Average Power Consumption W","Energy J", "CO2 Emissions gr", "GPU ID"])
			writer.writerow([inference_time, average_power_usage, energy_joules, co2_emissions, gpu_id])
		
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
	parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='batch size for processing')
	parser.add_argument('-w', '--workers', type=int, default=NUM_WORKERS, help='number of worker processes for data loading')
	parser.add_argument('--prefetch_factor', type=int, default=BATCHES_X_WORKER, help='number of batches loaded in advance by each worker')
	parser.add_argument('--max_length', type=int, default=MAX_LENGTH_TOKENIZER, help='maximum sequence length for tokenization')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
    global tokenizer  # Make tokenizer global so collate_fn can access it
    args = arg_commandline()

    # Load the tokenizer and model
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Set the device - for static quantization, we need to use CPU
    device = torch.device('cpu')  # Static quantization works on CPU
    model = model.to(device)
    
    print(f"Model loaded to {device}")
    
    # Load dataset
    print(f"Processing {SPECIFIC_FILE}...")
    df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
    
    if df is None or df.empty:
        print(f"No data found in {SPECIFIC_FILE}")
        exit()
    
    # Create dataset and dataloader
    dataset = MaskedDataset(df, tokenizer)
    
    # Create DataLoader for calibration
    calibration_dataloader = DataLoader(
        dataset, 
        batch_size=32,  # Smaller batch size for calibration
        shuffle=True, 
        num_workers=args.workers,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    # Regular dataloader for inference
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=False,  # No pin_memory for CPU
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    # Get example inputs for tracing (from the first valid batch)
    example_inputs = None
    for batch in calibration_dataloader:
        if batch is not None:
            example_inputs = (batch["input_ids"], batch["attention_mask"])
            break
    
    if example_inputs is None:
        print("Could not get example inputs from dataset.")
        exit()
    
    # Instead of FX graph mode, use eager mode quantization which is more compatible with HuggingFace models
    model.eval()
    
    # Step 1: Fuse modules (if applicable)
    # For transformers, we typically don't need explicit fusion as it's handled internally
    
    # Step 2: Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Step 3: Prepare for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Step 4: Calibrate with representative data
    print("Calibrating model with representative data...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_dataloader):
            if batch is None:
                continue
            
            # Forward pass for calibration
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            model_prepared(input_ids=input_ids, attention_mask=attention_mask)
            
            # Limit calibration to a few batches
            if i >= 10:
                break
    
    # Step 5: Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = torch.quantization.convert(model_prepared)
    
    # Use the quantized model for inference
    model = quantized_model
    model.eval()
    
    # Process batches
    results = []
    counter = 0
    
    print(f"Running fill-mask inference with quantized model, batch size {args.batch_size}")
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Skip empty batches
        if batch is None:
            continue
            
        # Get data (device is CPU for quantized model)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        masked_texts = batch["masked_texts"]
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = outputs.logits
        
        # Process each example in the batch
        for i, input_ids_i in enumerate(input_ids):
            masked_text = masked_texts[i]
            # Get positions of mask tokens
            mask_positions = [j for j, token_id in enumerate(input_ids_i.tolist()) 
                           if token_id == tokenizer.mask_token_id]
            
            # Skip if no masks found
            if not mask_positions:
                continue
            
            # For each mask position, get top 5 predictions
            example_results = {
                "id": counter,
                "masked_text": masked_text,
                "predictions": []
            }
            counter += 1
            
            for mask_pos in mask_positions:
                # Get the predicted token scores for this position
                mask_token_logits = predictions[i, mask_pos]
                
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
        # Compare model size before and after quantization
        original_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
        original_size = evaluate_size(original_model)
        quantized_size = evaluate_size(model)
        
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model Type", "Size (GB)", "Size Reduction %"])
            writer.writerow(["Original", original_size/1e+9, "0%"])
            writer.writerow(["Quantized", quantized_size/1e+9, f"{(1 - quantized_size/original_size) * 100:.2f}%"])
        
        print(f"Original model size: {original_size/1e+9:.2f} GB")
        print(f"Quantized model size: {quantized_size/1e+9:.2f} GB")
        print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")
        
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

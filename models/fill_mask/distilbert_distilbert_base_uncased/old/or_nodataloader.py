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
BATCH_SIZE = 512  # Adjust based on your GPU memory

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

def preprocess_for_fill_mask_gpu(texts, tokenizer, model, max_length=128, batch_size=32):
    """
    Preprocess text data for fill-mask task by randomly masking tokens
    Utilizing GPU for batch processing
    """
    # Determine device (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() and not any('quantized' in name for name, _ in model.named_modules()) else 'cpu'
    print(f"Using {device} for preprocessing")
    
    # Process in batches
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    masked_texts = []
    
    # Calculate number of batches
    num_examples = len(texts)
    
    for batch_idx in tqdm(range(BATCH_SIZE), desc="Preprocessing batches for fill-mask"):
        # Calculate start and end indices for this batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_examples)
        
        # Get the current batch of texts
        batch_texts = texts[start_idx:end_idx]
        
        # Skip empty or very short texts
        batch_texts = [text for text in batch_texts if text and len(text.split()) >= 5]
        
        if not batch_texts:
            continue
        
        # Tokenize the batch on GPU
        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        batch_input_ids = batch_encodings["input_ids"]
        batch_attention_masks = batch_encodings["attention_mask"]
        
        # Create masked versions and labels for each example in the batch
        batch_masked_input_ids = batch_input_ids.clone()
        batch_labels = torch.ones_like(batch_input_ids) * -100  # -100 means don't compute loss
        
        # Don't mask special tokens
        special_tokens = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id], device=device)
        
        # Process each example in the batch
        for i in range(batch_input_ids.size(0)):
            input_ids = batch_input_ids[i]
            
            # Create a mask for tokens that can be masked (not special tokens)
            can_mask = ~torch.isin(input_ids, special_tokens)
            maskable_positions = torch.nonzero(can_mask, as_tuple=True)[0]
            
            # Skip if not enough maskable tokens
            if len(maskable_positions) < 3:
                continue
                
            # Randomly select positions to mask based on probability
            num_to_mask = max(1, int(len(maskable_positions) * MASK_PROB))
            
            # Ensure we don't try to sample more positions than available
            num_to_mask = min(num_to_mask, len(maskable_positions))
            
            # Get random indices from maskable positions
            mask_indices = torch.randperm(len(maskable_positions), device=device)[:num_to_mask]
            positions_to_mask = maskable_positions[mask_indices]
            
            # Record the original tokens for the labels
            batch_labels[i, positions_to_mask] = input_ids[positions_to_mask]
            
            # Replace with [MASK] token
            batch_masked_input_ids[i, positions_to_mask] = tokenizer.mask_token_id
        
        # Move back to CPU for storage
        batch_masked_input_ids_cpu = batch_masked_input_ids.cpu().tolist()
        batch_attention_masks_cpu = batch_attention_masks.cpu().tolist()
        batch_labels_cpu = batch_labels.cpu().tolist()
        
        # Decode masked texts
        for i in range(len(batch_masked_input_ids_cpu)):
            input_ids_list.append(batch_masked_input_ids_cpu[i])
            attention_mask_list.append(batch_attention_masks_cpu[i])
            labels_list.append(batch_labels_cpu[i])
            masked_texts.append(tokenizer.decode(batch_masked_input_ids_cpu[i]))
    
    print(f"Preprocessed {len(input_ids_list)} examples with GPU acceleration")
    
    return {
        "input_ids": input_ids_list,
        "attention_masks": attention_mask_list,
        "labels": labels_list,
        "masked_texts": masked_texts
    }

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
    
    # Determine device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() and not any('quantized' in name for name, _ in model.named_modules()) else 'cpu'
    model = model.to(device)
    
    # Load dataset
    print(f"Processing {SPECIFIC_FILE}...")
    df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
    
    if df is None or df.empty:
        print(f"No data found in {SPECIFIC_FILE}")
        exit()
    
    # Extract text from the dataframe
    texts = df['text'].tolist()
    
    # Use GPU-accelerated preprocessing
    processed_data = preprocess_for_fill_mask_gpu(texts, tokenizer, model, batch_size=32)
    
    # Process in batches for inference
    results = []
    
    print(f"Running fill-mask inference on {len(processed_data['input_ids'])} processed examples...")
    
    # Calculate number of batches
    num_examples = len(processed_data['input_ids'])
    num_batches = (num_examples + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    for batch_idx in tqdm(range(num_batches), desc="Processing inference batches"):
        # Calculate start and end indices for this batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_examples)
        
        # Get the current batch
        batch_input_ids = processed_data['input_ids'][start_idx:end_idx]
        batch_attention_masks = processed_data['attention_masks'][start_idx:end_idx]
        batch_masked_texts = processed_data['masked_texts'][start_idx:end_idx]
        
        # Convert to tensors and move to device
        input_ids_tensor = torch.tensor(batch_input_ids).to(device)
        attention_mask_tensor = torch.tensor(batch_attention_masks).to(device)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            predictions = outputs.logits
        
        # Process each example in the batch
        for i, (input_ids, masked_text) in enumerate(zip(batch_input_ids, batch_masked_texts)):
            # Get positions of mask tokens
            mask_positions = [j for j, token_id in enumerate(input_ids) 
                            if token_id == tokenizer.mask_token_id]
            
            # Skip if no masks found
            if not mask_positions:
                continue
            
            # For each mask position, get top 5 predictions
            example_results = {
                "id": start_idx + i,
                "masked_text": masked_text,
                "predictions": []
            }
            
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

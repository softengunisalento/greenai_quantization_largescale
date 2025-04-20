from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import argparse
import pandas as pd
import json


### Constants
MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-en-fr"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class XP3Dataset(Dataset):
    """Dataset for the xP3 dataset"""
    
    def __init__(self, df, tokenizer):
        # Extract text from the dataset
        self.inputs = df['inputs'].tolist()  # Source text for translation
        self.targets = df['targets'].tolist()  # Target text for translation
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # Skip if text is too short (handled in collate_fn)
        if len(input_text.split()) < 5:
            return None
            
        # Tokenize the text
        inputs = self.tokenizer(
            input_text, 
            truncation=True,
            padding=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            truncation=True,
            padding=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "target_ids": targets["input_ids"][0]
        }

def collate_fn(batch):
    """Custom collate function to filter out None values"""
    # Filter out None values (skipped examples)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Get max length in this batch
    max_input_length = max(len(item["input_ids"]) for item in batch)
    max_target_length = max(len(item["target_ids"]) for item in batch)
    
    # Prepare lists for batched data
    input_ids = []
    attention_masks = []
    target_ids = []
    
    # Pad sequences
    for item in batch:
        # Pad input_ids
        curr_input_len = len(item["input_ids"])
        padding_input_len = max_input_length - curr_input_len
        
        padded_input_ids = torch.cat([
            item["input_ids"], 
            torch.tensor([tokenizer.pad_token_id] * padding_input_len, dtype=torch.long)
        ])
        input_ids.append(padded_input_ids)
        
        # Pad attention_mask
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(padding_input_len, dtype=torch.long)
        ])
        attention_masks.append(padded_attention_mask)
        
        # Pad target_ids
        curr_target_len = len(item["target_ids"])
        padding_target_len = max_target_length - curr_target_len
        
        padded_target_ids = torch.cat([
            item["target_ids"],
            torch.tensor([tokenizer.pad_token_id] * padding_target_len, dtype=torch.long)
        ])
        target_ids.append(padded_target_ids)
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "target_ids": torch.stack(target_ids)
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
    parser = argparse.ArgumentParser(description="argument parser for quantization comparison on translation tasks")

    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
    parser.add_argument('-t', '--max_length_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
    parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int8')

    return parser.parse_args()

### Main
def main():
    # Make the args variable accessible globally
    global args
    args = arg_commandline()
    
    # Make tokenizer global for use in collate_fn
    global tokenizer
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the original model to the correct device
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
    
    print(f"Models loaded to {device}")
    
    # Load dataset
    print(f"Processing {SPECIFIC_FILE}...")
    df = load_jsonl_file(DATASET_ID, SPECIFIC_FILE)
    
    if df is None or len(df) == 0:
        print(f"No data found in {SPECIFIC_FILE}")
        exit()
    
    # Create dataset and dataloader
    dataset = XP3Dataset(df, tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Dictionary to store MSE values for each record
    mse_values = {}
    max_diff_values = {}
    counter = 0
    
    print(f"Running inference with batch size {args.batch_size} on {device}")
    
    for batch in tqdm(dataloader, desc="Comparing model outputs"):
        # Skip empty batches
        if batch is None:
            continue
            
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        # Forward pass through both models
        with torch.no_grad():
            # For original model
            original_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids
            )
            
            # For quantized model
            quantized_outputs = model_quantized(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids
            )
        
        # Extract logits
        original_logits = original_outputs.logits
        quantized_logits = quantized_outputs.logits
        
        # Calculate MSE and max difference for each example in the batch
        for i in range(len(input_ids)):
            # Flatten the logits for this example
            original_logits_flat = original_logits[i].view(-1)
            quantized_logits_flat = quantized_logits[i].view(-1)
            
            # Calculate the mean squared error (MSE) between the outputs
            mse = torch.mean((original_logits_flat - quantized_logits_flat) ** 2).item()
            
            # Calculate the maximum absolute difference
            max_diff = torch.max(torch.abs(original_logits_flat - quantized_logits_flat)).item()
            
            # Store values for later analysis
            mse_values[counter] = mse
            max_diff_values[counter] = max_diff
            counter += 1
    
    # Save results to CSV
    csv_results(mse_values, max_diff_values)

    # Free cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
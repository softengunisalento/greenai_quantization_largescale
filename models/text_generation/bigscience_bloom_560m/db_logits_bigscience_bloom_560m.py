from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import argparse
import pandas as pd
import json

### Constants
MODEL_ID = "bigscience/bloom-560m"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"

### Functions
class XP3Dataset(Dataset):
    """Dataset for the xP3 dataset"""
    
    def __init__(self, df, tokenizer):
        # Extract text from the dataset (e.g., 'text' field or article field)
        self.texts = df['inputs'].tolist()  # Assuming the text is stored in the 'text' column
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
            padding=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0]
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
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks)
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
    parser = argparse.ArgumentParser(description="argument parser for quantization comparison")

    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
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
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the original model to the correct device
    model = model.to(device)
    
    # Apply dynamic quantization to the model
    if args.quantization_bits == 4:
        model_quantized = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            device_map=device
        )
    elif args.quantization_bits == 8:
        model_quantized = AutoModelForSequenceClassification.from_pretrained(
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
        persistent_workers=True
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
        
        # Forward pass through both models
        with torch.no_grad():
            # For original model
            original_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # For quantized model
            quantized_outputs = model_quantized(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Extract last hidden states
        original_hidden_states = original_outputs.logits
        quantized_hidden_states = quantized_outputs.logits
        
        # Calculate MSE and max difference for each example in the batch
        for i in range(len(input_ids)):
            # Calculate the mean squared error (MSE) between the outputs
            mse = torch.mean((original_hidden_states[i] - quantized_hidden_states[i]) ** 2).item()
            
            # Calculate the maximum absolute difference
            max_diff = torch.max(torch.abs(original_hidden_states[i] - quantized_hidden_states[i])).item()
            
            # Store values for later analysis
            mse_values[counter] = mse
            max_diff_values[counter] = max_diff
            counter += 1
    
    # Save results to CSV
    csv_results(mse_values, max_diff_values)

    # Free cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

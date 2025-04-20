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
MODEL_ID = "bigscience/bloomz-560m"
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
        self.indices = list(range(len(self.texts)))  # Add indices for tracking examples
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Skip if text is too short (handled in collate_fn)
        if len(text.split()) < 5:
            return None
        
        return {
            "text": text,
            "index": self.indices[idx]
        }

def collate_fn(batch):
    """Custom collate function to filter out None values"""
    # Filter out None values (skipped examples)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Collect texts and indices
    texts = [item["text"] for item in batch]
    indices = [item["index"] for item in batch]
    
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

def calculate_accuracy(original_model, model_int4, model_int8, dataloader, tokenizer, device):
    """Calculate MLM accuracy for all models and measure the drops"""
    original_correct = 0
    int4_correct = 0
    int8_correct = 0
    total_predictions = 0
    
    # Dictionaries to store per-example prediction correctness
    original_preds = {}
    int4_preds = {}
    int8_preds = {}
    
    for batch in tqdm(dataloader, desc="Evaluating accuracy"):
        if batch is None:
            continue
        
        # Get texts from the batch and tokenize them
        texts = batch["texts"]
        indices = batch["indices"]
        
        # Tokenize inputs
        encoded_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        ).to(device)
        
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        
        # Create a copy for masks and labels
        labels = input_ids.clone()
        masked_inputs = input_ids.clone()
        
        # Apply masking - mask 15% of tokens
        mask_positions = []
        for i in range(len(input_ids)):
            positions = []
            for j in range(1, len(input_ids[i]) - 1):  # Skip special tokens
                # Only mask non-padding tokens
                if input_ids[i][j] != tokenizer.pad_token_id and torch.rand(1).item() < 0.15:
                    masked_inputs[i][j] = tokenizer.mask_token_id
                    positions.append(j)
                else:
                    # For non-masked positions, use -100 as label to ignore in loss
                    labels[i][j] = -100
            mask_positions.append(positions)
        
        # Make predictions with all models
        with torch.no_grad():
            # Forward pass through all models
            original_outputs = original_model(
                input_ids=masked_inputs,
                attention_mask=attention_mask
            )
            
            int4_outputs = model_int4(
                input_ids=masked_inputs,
                attention_mask=attention_mask
            )
            
            int8_outputs = model_int8(
                input_ids=masked_inputs,
                attention_mask=attention_mask
            )
        
        # For each example in the batch
        for i in range(len(input_ids)):
            example_id = indices[i]
            
            # Initialize prediction lists for this example
            original_preds[example_id] = []
            int4_preds[example_id] = []
            int8_preds[example_id] = []
            
            # Get masked positions for this example
            positions = mask_positions[i]
            
            # For each masked position
            for pos in positions:
                # Get the correct token
                correct_token = labels[i][pos].item()
                if correct_token == -100:
                    continue
                
                # Get model predictions
                original_logits = original_outputs.logits[i, pos]
                int4_logits = int4_outputs.logits[i, pos]
                int8_logits = int8_outputs.logits[i, pos]
                
                # Get top prediction
                original_pred = torch.argmax(original_logits).item()
                int4_pred = torch.argmax(int4_logits).item()
                int8_pred = torch.argmax(int8_logits).item()
                
                # Store prediction correctness (1 for correct, 0 for incorrect)
                original_correct_pred = 1 if original_pred == correct_token else 0
                int4_correct_pred = 1 if int4_pred == correct_token else 0
                int8_correct_pred = 1 if int8_pred == correct_token else 0
                
                # Append to example prediction lists
                original_preds[example_id].append(original_correct_pred)
                int4_preds[example_id].append(int4_correct_pred)
                int8_preds[example_id].append(int8_correct_pred)
                
                # Count correct predictions
                original_correct += original_correct_pred
                int4_correct += int4_correct_pred
                int8_correct += int8_correct_pred
                
                total_predictions += 1
    
    # Calculate accuracies
    original_accuracy = original_correct / total_predictions if total_predictions > 0 else 0
    int4_accuracy = int4_correct / total_predictions if total_predictions > 0 else 0
    int8_accuracy = int8_correct / total_predictions if total_predictions > 0 else 0
    
    # Calculate accuracy drops
    int4_drop = original_accuracy - int4_accuracy
    int8_drop = original_accuracy - int8_accuracy
    
    # Calculate per-example accuracy difference statistics
    example_int4_drops = []
    example_int8_drops = []
    
    for example_id in original_preds:
        if len(original_preds[example_id]) > 0:
            # Calculate per-example accuracy
            example_orig_acc = sum(original_preds[example_id]) / len(original_preds[example_id])
            example_int4_acc = sum(int4_preds[example_id]) / len(int4_preds[example_id])
            example_int8_acc = sum(int8_preds[example_id]) / len(int8_preds[example_id])
            
            # Calculate drops
            example_int4_drops.append(example_orig_acc - example_int4_acc)
            example_int8_drops.append(example_orig_acc - example_int8_acc)
    
    # Calculate statistics of per-example drops
    int4_drop_mean = sum(example_int4_drops) / len(example_int4_drops) if example_int4_drops else 0
    int4_drop_std = (sum((x - int4_drop_mean) ** 2 for x in example_int4_drops) / len(example_int4_drops)) ** 0.5 if example_int4_drops else 0
    
    int8_drop_mean = sum(example_int8_drops) / len(example_int8_drops) if example_int8_drops else 0
    int8_drop_std = (sum((x - int8_drop_mean) ** 2 for x in example_int8_drops) / len(example_int8_drops)) ** 0.5 if example_int8_drops else 0
    
    return {
        "original_accuracy": original_accuracy,
        "int4_accuracy": int4_accuracy,
        "int8_accuracy": int8_accuracy,
        "int4_drop": int4_drop,
        "int8_drop": int8_drop,
        "int4_drop_percentage": (int4_drop / original_accuracy) * 100 if original_accuracy > 0 else 0,
        "int8_drop_percentage": (int8_drop / original_accuracy) * 100 if original_accuracy > 0 else 0,
        "int4_drop_std": int4_drop_std,
        "int8_drop_std": int8_drop_std,
        "total_predictions": total_predictions
    }

def calculate_hidden_state_mse(original_model, model_int4, model_int8, dataloader, tokenizer, device):
    """Calculate MSE between hidden states of original and quantized models"""
    mse_values_int4 = {}
    mse_values_int8 = {}
    max_diff_values_int4 = {}
    max_diff_values_int8 = {}
    
    for batch in tqdm(dataloader, desc="Comparing hidden states"):
        if batch is None:
            continue
            
        texts = batch["texts"]
        indices = batch["indices"]
        
        # Tokenize input batch
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass through all models to get hidden states
        with torch.no_grad():
            original_outputs = original_model(**inputs, output_hidden_states=True)
            int4_outputs = model_int4(**inputs, output_hidden_states=True)
            int8_outputs = model_int8(**inputs, output_hidden_states=True)
        
        # Extract last hidden states
        original_hidden_states = original_outputs.last_hidden_state \
            if hasattr(original_outputs, 'last_hidden_state') else original_outputs.hidden_states[-1]
        int4_hidden_states = int4_outputs.hidden_states[-1]
        int8_hidden_states = int8_outputs.hidden_states[-1]
        
        # Calculate MSE and max difference for each example in the batch
        for i, idx in enumerate(indices):
            # Calculate MSE between original and int4
            mse_int4 = torch.mean((original_hidden_states[i] - int4_hidden_states[i]) ** 2).item()
            max_diff_int4 = torch.max(torch.abs(original_hidden_states[i] - int4_hidden_states[i])).item()
            
            # Calculate MSE between original and int8
            mse_int8 = torch.mean((original_hidden_states[i] - int8_hidden_states[i]) ** 2).item()
            max_diff_int8 = torch.max(torch.abs(original_hidden_states[i] - int8_hidden_states[i])).item()
            
            # Store values for later analysis
            mse_values_int4[idx] = mse_int4
            mse_values_int8[idx] = mse_int8
            max_diff_values_int4[idx] = max_diff_int4
            max_diff_values_int8[idx] = max_diff_int8
    
    return {
        "mse_values_int4": mse_values_int4,
        "mse_values_int8": mse_values_int8,
        "max_diff_values_int4": max_diff_values_int4,
        "max_diff_values_int8": max_diff_values_int8
    }

def csv_results(mse_results, accuracy_results):
    """ Print MSE statistics and accuracy results """
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write accuracy results first (most important)
        writer.writerow(["Accuracy Metrics"])
        writer.writerow(["Model", "Accuracy", "Accuracy Drop", "Accuracy Drop (%)", "Drop Std Dev"])
        writer.writerow(["Original", accuracy_results["original_accuracy"], "N/A", "N/A", "N/A"])
        writer.writerow(["Int4", accuracy_results["int4_accuracy"],
                        accuracy_results["int4_drop"],
                        accuracy_results["int4_drop_percentage"],
                        accuracy_results["int4_drop_std"]])
        writer.writerow(["Int8", accuracy_results["int8_accuracy"],
                        accuracy_results["int8_drop"],
                        accuracy_results["int8_drop_percentage"],
                        accuracy_results["int8_drop_std"]])
        writer.writerow(["Total predictions", accuracy_results["total_predictions"]])
        writer.writerow([])  # Empty row for separation
        
        # MSE statistics
        mse_int4_series = pd.Series(list(mse_results["mse_values_int4"].values()))
        mse_int8_series = pd.Series(list(mse_results["mse_values_int8"].values()))
        
        writer.writerow(["Hidden State MSE Metrics"])
        writer.writerow(["Metric", "Int4", "Int8", "Difference (Int4-Int8)"])
        writer.writerow(["Min MSE", mse_int4_series.min(), mse_int8_series.min(),
                       mse_int4_series.min() - mse_int8_series.min()])
        writer.writerow(["Max MSE", mse_int4_series.max(), mse_int8_series.max(),
                       mse_int4_series.max() - mse_int8_series.max()])
        writer.writerow(["Mean MSE", mse_int4_series.mean(), mse_int8_series.mean(),
                       mse_int4_series.mean() - mse_int8_series.mean()])
        writer.writerow(["Median MSE", mse_int4_series.median(), mse_int8_series.median(),
                       mse_int4_series.median() - mse_int8_series.median()])
        writer.writerow(["Std Dev MSE", mse_int4_series.std(), mse_int8_series.std(),
                       mse_int4_series.std() - mse_int8_series.std()])
        
        # Max difference statistics
        max_diff_int4_series = pd.Series(list(mse_results["max_diff_values_int4"].values()))
        max_diff_int8_series = pd.Series(list(mse_results["max_diff_values_int8"].values()))
        
        writer.writerow([])  # Empty row for separation
        writer.writerow(["Hidden State Max Difference Metrics"])
        writer.writerow(["Metric", "Int4", "Int8", "Difference (Int4-Int8)"])
        writer.writerow(["Min Max-Diff", max_diff_int4_series.min(), max_diff_int8_series.min(),
                       max_diff_int4_series.min() - max_diff_int8_series.min()])
        writer.writerow(["Max Max-Diff", max_diff_int4_series.max(), max_diff_int8_series.max(),
                       max_diff_int4_series.max() - max_diff_int8_series.max()])
        writer.writerow(["Mean Max-Diff", max_diff_int4_series.mean(), max_diff_int8_series.mean(),
                       max_diff_int4_series.mean() - max_diff_int8_series.mean()])
        writer.writerow(["Median Max-Diff", max_diff_int4_series.median(), max_diff_int8_series.median(),
                       max_diff_int4_series.median() - max_diff_int8_series.median()])
        writer.writerow(["Std Dev Max-Diff", max_diff_int4_series.std(), max_diff_int8_series.std(),
                       max_diff_int4_series.std() - max_diff_int8_series.std()])
    
    print(f"Results saved to {CSV_FILE}")

def arg_commandline():
    """ Create an argument parser object """
    parser = argparse.ArgumentParser(description="argument parser for quantization comparison")

    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
    parser.add_argument('-t', '--max_length_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
    # New argument for accuracy evaluation
    parser.add_argument('--eval_accuracy', action='store_true', help='evaluate model accuracy using masked language modeling')
    parser.add_argument('--eval_mse', action='store_true', help='evaluate MSE between hidden states of models')

    return parser.parse_args()

### Main
def main():
    # Make the args variable accessible globally
    global args
    args = arg_commandline()
    
    # Make tokenizer global for use in collate_fn
    global tokenizer
    
    # Load the tokenizer and models
    print(f"Loading tokenizer and models from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load original model
    original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    original_model = original_model.to(device)
    
    # Apply dynamic quantization to the models
    # Load int4 model
    model_int4 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        device_map=device
    )
    
    # Load int8 model
    model_int8 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        load_in_8bit=True,
        device_map=device
    )
    
    print(f"All models loaded successfully")
    
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
    
    # Initialize results dictionaries
    mse_results = None
    accuracy_results = None
    
    # Evaluate accuracy if enabled
    accuracy_results = calculate_accuracy(
        original_model, model_int4, model_int8, dataloader, tokenizer, device
    )
    
    # Evaluate hidden state MSE if enabled
    mse_results = calculate_hidden_state_mse(
        original_model, model_int4, model_int8, dataloader, tokenizer, device
    )

    # Write results to CSV
    csv_results(mse_results, accuracy_results)

    # Free cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
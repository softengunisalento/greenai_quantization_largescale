from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import argparse
import pandas as pd
import json
import gc
import numpy as np


### Constants
MODEL_ID = "google-t5/t5-small"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class XP3Dataset(Dataset):
    """Dataset for the xP3 dataset"""
    
    def __init__(self, df, tokenizer, max_length):
        # Extract text from the dataset
        self.inputs = df['inputs'].tolist()  # Source text for translation
        self.targets = df['targets'].tolist()  # Target text for translation
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # Skip if text is too short
        if len(input_text.split()) < 5:
            return None
            
        return {
            "input_text": input_text,
            "target_text": target_text,
            "id": idx
        }

def collate_fn(batch):
    """Custom collate function to filter out None values"""
    # Filter out None values (skipped examples)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    return {
        "input_text": [item["input_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "id": [item["id"] for item in batch]
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

def calculate_accuracy(original_model, model_int4, model_int8, dataloader, device, tokenizer):
    """Calculate generation accuracy for all models and measure the drops"""
    # Initialize dictionaries to store generations
    original_generations = {}
    int4_generations = {}
    int8_generations = {}
    
    # Track the datasets targets for reference
    targets = {}
    
    for batch in tqdm(dataloader, desc="Evaluating generation accuracy"):
        if batch is None:
            continue
            
        # Process each example individually to avoid OOM
        for i in range(len(batch["id"])):
            try:
                example_id = batch["id"][i]
                input_text = batch["input_text"][i]
                target_text = batch["target_text"][i]
                
                print(f"Processing example: {example_id}")
                
                # Tokenize input
                inputs = tokenizer(
                    input_text, 
                    truncation=True,
                    padding=True,
                    max_length=args.max_length_tokenizer,
                    return_tensors="pt"
                ).to(device)
                
                # Store the target for reference
                targets[example_id] = target_text
                
                # Generate text with original model
                with torch.no_grad():
                    original_outputs = original_model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"],
                        max_length=args.max_length_tokenizer
                    )
                    original_generation = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
                
                # Generate text with INT4 model (if available)
                if model_int4 is not None:
                    with torch.no_grad():
                        int4_outputs = model_int4.generate(
                            input_ids=inputs["input_ids"], 
                            attention_mask=inputs["attention_mask"],
                            max_length=args.max_length_tokenizer
                        )
                        int4_generation = tokenizer.decode(int4_outputs[0], skip_special_tokens=True)
                else:
                    int4_generation = "Model not available"
                
                # Generate text with INT8 model
                with torch.no_grad():
                    int8_outputs = model_int8.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"],
                        max_length=args.max_length_tokenizer
                    )
                    int8_generation = tokenizer.decode(int8_outputs[0], skip_special_tokens=True)
                
                # Store generations
                original_generations[example_id] = original_generation
                int4_generations[example_id] = int4_generation
                int8_generations[example_id] = int8_generation
                
                print(f"Original: {original_generation[:100]}...")
                if model_int4 is not None:
                    print(f"INT4: {int4_generation[:100]}...")
                print(f"INT8: {int8_generation[:100]}...")
                print(f"Target: {target_text[:100]}...")
                
            except Exception as e:
                print(f"Failed to process example {example_id}: {str(e)}")
    
    # Calculate similarity metrics
    int4_exact_match = 0
    int8_exact_match = 0
    
    int4_char_similarity = []
    int8_char_similarity = []
    
    for example_id in original_generations:
        orig = original_generations[example_id]
        int4 = int4_generations[example_id] if model_int4 is not None else ""
        int8 = int8_generations[example_id]
        
        # Count exact matches
        if model_int4 is not None and orig == int4:
            int4_exact_match += 1
        if orig == int8:
            int8_exact_match += 1
        
        # Calculate character-level similarity
        if len(orig) > 0:
            if model_int4 is not None:
                int4_sim = sum(c1 == c2 for c1, c2 in zip(orig, int4[:len(orig)])) / len(orig)
                int4_char_similarity.append(int4_sim)
                
            int8_sim = sum(c1 == c2 for c1, c2 in zip(orig, int8[:len(orig)])) / len(orig)
            int8_char_similarity.append(int8_sim)
    
    # Calculate metrics
    total_examples = len(original_generations)
    
    # Calculate INT4 metrics if the model is available
    if model_int4 is not None:
        int4_exact_match_rate = int4_exact_match / total_examples if total_examples > 0 else 0
        int4_avg_char_similarity = sum(int4_char_similarity) / len(int4_char_similarity) if int4_char_similarity else 0
        int4_drop = 1 - int4_exact_match_rate
        int4_char_drop = 1 - int4_avg_char_similarity
    else:
        int4_exact_match_rate = 0
        int4_avg_char_similarity = 0
        int4_drop = 1.0
        int4_char_drop = 1.0
        
    # Calculate INT8 metrics
    int8_exact_match_rate = int8_exact_match / total_examples if total_examples > 0 else 0
    int8_avg_char_similarity = sum(int8_char_similarity) / len(int8_char_similarity) if int8_char_similarity else 0
    int8_drop = 1 - int8_exact_match_rate
    int8_char_drop = 1 - int8_avg_char_similarity
    
    return {
        "total_examples": total_examples,
        "int4_exact_match_rate": int4_exact_match_rate,
        "int8_exact_match_rate": int8_exact_match_rate,
        "int4_avg_char_similarity": int4_avg_char_similarity,
        "int8_avg_char_similarity": int8_avg_char_similarity,
        "int4_drop": int4_drop,
        "int8_drop": int8_drop,
        "int4_char_drop": int4_char_drop,
        "int8_char_drop": int8_char_drop,
        "original_generations": original_generations,
        "int4_generations": int4_generations,
        "int8_generations": int8_generations,
        "targets": targets
    }

def calculate_hidden_state_mse(original_model, model_int4, model_int8, dataloader, device, tokenizer):
    """Calculate MSE and max difference between original and quantized models"""
    # Dictionaries to store MSE values and max differences for each quantized model
    mse_values_int4 = {}
    mse_values_int8 = {}
    max_diff_values_int4 = {}
    max_diff_values_int8 = {}
    
    for batch in tqdm(dataloader, desc="Evaluating hidden state differences"):
        if batch is None:
            continue
            
        # Process each example individually
        for i in range(len(batch["id"])):
            try:
                example_id = batch["id"][i]
                input_text = batch["input_text"][i]
                
                print(f"Processing example: {example_id}")
                
                # Tokenize input
                inputs = tokenizer(
                    input_text, 
                    truncation=True,
                    padding=True,
                    max_length=args.max_length_tokenizer,
                    return_tensors="pt"
                ).to(device)
                
                # Prepare decoder input IDs (usually the decoder start token)
                decoder_input_ids = torch.tensor([[original_model.config.decoder_start_token_id]]).to(device)
                
                # Get outputs from original model (logits)
                with torch.no_grad():
                    original_outputs = original_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        decoder_input_ids=decoder_input_ids
                    ).logits.detach().cpu().numpy()
                
                # Get outputs from INT4 model if available
                if model_int4 is not None:
                    with torch.no_grad():
                        int4_outputs = model_int4(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            decoder_input_ids=decoder_input_ids
                        ).logits.detach().cpu().numpy()
                        
                        # Calculate INT4 metrics
                        mse_int4 = np.mean((original_outputs - int4_outputs) ** 2)
                        max_diff_int4 = np.max(np.abs(original_outputs - int4_outputs))
                        
                        # Store INT4 results
                        mse_values_int4[example_id] = float(mse_int4)
                        max_diff_values_int4[example_id] = float(max_diff_int4)
                        
                        print(f"INT4 MSE: {mse_int4:.6f}, Max Diff: {max_diff_int4:.6f}")
                
                # Get outputs from INT8 model
                with torch.no_grad():
                    int8_outputs = model_int8(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        decoder_input_ids=decoder_input_ids
                    ).logits.detach().cpu().numpy()
                    
                    # Calculate INT8 metrics
                    mse_int8 = np.mean((original_outputs - int8_outputs) ** 2)
                    max_diff_int8 = np.max(np.abs(original_outputs - int8_outputs))
                    
                    # Store INT8 results
                    mse_values_int8[example_id] = float(mse_int8)
                    max_diff_values_int8[example_id] = float(max_diff_int8)
                    
                    print(f"INT8 MSE: {mse_int8:.6f}, Max Diff: {max_diff_int8:.6f}")
                
            except Exception as e:
                print(f"Failed to process example {example_id}: {str(e)}")
    
    return {
        "mse_values_int4": mse_values_int4,
        "mse_values_int8": mse_values_int8,
        "max_diff_values_int4": max_diff_values_int4,
        "max_diff_values_int8": max_diff_values_int8
    }

def csv_results(mse_results, accuracy_results):
    """ Write MSE statistics and accuracy results to CSV """
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write accuracy results first (most important)
        writer.writerow(["Text Generation Accuracy Metrics"])
        writer.writerow(["Metric", "Original", "Int4", "Int8"])
        writer.writerow(["Exact Match Rate", "1.0", 
                        accuracy_results["int4_exact_match_rate"], 
                        accuracy_results["int8_exact_match_rate"]])
        writer.writerow(["Character Similarity", "1.0", 
                        accuracy_results["int4_avg_char_similarity"], 
                        accuracy_results["int8_avg_char_similarity"]])
        writer.writerow(["Exact Match Drop", "N/A", 
                        accuracy_results["int4_drop"], 
                        accuracy_results["int8_drop"]])
        writer.writerow(["Character Drop", "N/A", 
                        accuracy_results["int4_char_drop"], 
                        accuracy_results["int8_char_drop"]])
        writer.writerow(["Total examples processed", accuracy_results["total_examples"]])
        
        writer.writerow([])  # Empty row for separation
        
        if mse_results and "mse_values_int4" in mse_results and mse_results["mse_values_int4"]:
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
        
        writer.writerow([])  # Empty row for separation
        
        # Write a few example generations for comparison
        writer.writerow(["Example Generations"])
        writer.writerow(["Example ID", "Original", "Int4", "Int8", "Target"])
        
        # Take up to 5 examples to showcase
        sample_ids = list(accuracy_results["original_generations"].keys())[:5]
        for example_id in sample_ids:
            writer.writerow([
                example_id,
                accuracy_results["original_generations"][example_id][:100] + "...",
                accuracy_results["int4_generations"][example_id][:100] + "..." if accuracy_results["int4_generations"] else "N/A",
                accuracy_results["int8_generations"][example_id][:100] + "...",
                accuracy_results["targets"][example_id][:100] + "..."
            ])

    print(f"Results saved to {CSV_FILE}")

def arg_commandline():
    """ Create an argument parser object """
    parser = argparse.ArgumentParser(description="argument parser for quantization comparison on text generation tasks")

    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=1, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
    parser.add_argument('-t', '--max_length_tokenizer', type=int, default=512, help='maximum sequence length for tokenization')
    parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='bit value to quantize: 4 or 8')
    parser.add_argument('-m', '--max_examples', type=int, default=20, help='maximum number of examples to process')
    parser.add_argument('--skip_mse', action='store_true', help='skip MSE calculation to save time')
    parser.add_argument('--skip_int4', action='store_true', help='skip INT4 model loading and evaluation')

    return parser.parse_args()

### Main
def main():
    # Make the args variable accessible globally
    global args
    args = arg_commandline()
    
    # Load the tokenizer and model
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the original model
    print(f"Loading original model from {MODEL_ID}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    
    # Load 4-bit quantized model (if requested)
    model_int4 = None
    if not args.skip_int4:
        try:
            print(f"Loading 4-bit quantized model...")
            model_int4 = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID,
                load_in_4bit=True,
                device_map=device
            )
        except Exception as e:
            print(f"Failed to load INT4 model: {e}")
            print("Continuing without INT4 model...")
    else:
        print("Skipping INT4 model as requested...")
    
    # Load 8-bit quantized model
    print(f"Loading 8-bit quantized model...")
    model_int8 = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        load_in_8bit=True,
        device_map=device
    )
    
    print(f"Models loaded successfully")
    
    # Load dataset
    print(f"Loading dataset: {DATASET_ID}, file: {SPECIFIC_FILE}")
    df = load_jsonl_file(DATASET_ID, SPECIFIC_FILE)
    
    if df is None or len(df) == 0:
        print(f"No data found in {SPECIFIC_FILE}")
        exit()
    
    # Limit the number of examples if specified
    if args.max_examples > 0 and len(df) > args.max_examples:
        print(f"Limiting to {args.max_examples} examples")
        df = df.head(args.max_examples)
    
    # Create dataset and dataloader
    dataset = XP3Dataset(df, tokenizer, args.max_length_tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Keep order for reproducibility 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Process MSE calculation if not skipped
    mse_results = {}
    if not args.skip_mse:
        print("Calculating hidden state differences...")
        mse_results = calculate_hidden_state_mse(model, model_int4, model_int8, dataloader, device, tokenizer)
    
    # Calculate generation accuracy
    print("Calculating generation accuracy...")
    accuracy_results = calculate_accuracy(model, model_int4, model_int8, dataloader, device, tokenizer)
    
    # Print results to console
    print("\nAccuracy Results:")
    print(f"Original model used as ground truth reference (accuracy: 1.0)")
    if model_int4 is not None:
        print(f"Int4 model exact match rate: {accuracy_results['int4_exact_match_rate']:.4f} (drop: {accuracy_results['int4_drop']:.4f})")
        print(f"Int4 model character similarity: {accuracy_results['int4_avg_char_similarity']:.4f} (character-level drop: {accuracy_results['int4_char_drop']:.4f})")
    else:
        print("Int4 model not available for comparison")
    print(f"Int8 model exact match rate: {accuracy_results['int8_exact_match_rate']:.4f} (drop: {accuracy_results['int8_drop']:.4f})")
    print(f"Int8 model character similarity: {accuracy_results['int8_avg_char_similarity']:.4f} (character-level drop: {accuracy_results['int8_char_drop']:.4f})")
    print(f"Total examples processed: {accuracy_results['total_examples']}")
    
    # Save results to CSV
    csv_results(mse_results, accuracy_results)
    
    # Generate a comparison report
    if model_int4 is not None:
        if accuracy_results["int4_exact_match_rate"] > accuracy_results["int8_exact_match_rate"]:
            print("\nQuantization Comparison: Int4 outperforms Int8 in generation accuracy!")
        elif accuracy_results["int8_exact_match_rate"] > accuracy_results["int4_exact_match_rate"]:
            print("\nQuantization Comparison: Int8 outperforms Int4 in generation accuracy!")
        else:
            print("\nQuantization Comparison: Int4 and Int8 have identical generation accuracy.")
            
        if mse_results and "mse_values_int4" in mse_results and mse_results["mse_values_int4"]:
            int4_mse_mean = pd.Series(list(mse_results["mse_values_int4"].values())).mean()
            int8_mse_mean = pd.Series(list(mse_results["mse_values_int8"].values())).mean()
            
            if int4_mse_mean < int8_mse_mean:
                print("Int4 has lower mean MSE than Int8, suggesting better hidden state preservation.")
            else:
                print("Int8 has lower mean MSE than Int4, suggesting better hidden state preservation.")

    # Free cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Clean up
    del model
    if model_int4 is not None:
        del model_int4
    del model_int8
    gc.collect()


if __name__ == "__main__":
    main()
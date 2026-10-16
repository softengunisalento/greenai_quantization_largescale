from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import argparse
import pandas as pd
import json
from sacrebleu import sentence_bleu
import numpy as np

### Constants
MODEL_ID = "openai-community/gpt2"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()

### Functions
class XP3Dataset(Dataset):
    """Dataset for the xP3 dataset"""
    
    def __init__(self, df, tokenizer):
        self.inputs = df['inputs'].tolist()
        self.targets = df['targets'].tolist() if 'targets' in df.columns else None
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx] if self.targets else ""
        
        # Skip if text is too short
        if len(input_text.split()) < 5:
            return None
            
        # Tokenize the input text
        inputs = self.tokenizer(
            input_text, 
            truncation=True,
            padding=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "input_text": input_text,
            "target_text": target_text
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
    input_texts = []
    target_texts = []
    
    # Pad sequences
    for item in batch:
        # Get current length
        curr_len = len(item["input_ids"])
        padding_len = max_length - curr_len
        
        # Pad input_ids
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
        
        # Store text data
        input_texts.append(item["input_text"])
        target_texts.append(item["target_text"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "input_texts": input_texts,
        "target_texts": target_texts
    }

def generate_text(model, tokenizer, input_ids, attention_mask, max_new_tokens=50):
    """Generate text using the model"""
    with torch.no_grad():
        # Generate text
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Decode generated text (remove input part)
        generated_texts = []
        for i in range(len(input_ids)):
            # Get only the newly generated tokens
            new_tokens = generated_ids[i][len(input_ids[i]):]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        return generated_texts

def calculate_bleu_scores(references, hypotheses):
    """Calculate BLEU scores for a batch of texts"""
    bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        if not ref.strip() or not hyp.strip():
            bleu_scores.append(0.0)
        else:
            # Calculate sentence-level BLEU score
            score = sentence_bleu(hyp, [ref]).score
            bleu_scores.append(score)
    
    return bleu_scores

def load_jsonl_file(dataset_id, file_name):
    """Function to extract and process the specific JSONL file"""
    from huggingface_hub import hf_hub_download
    
    try:
        file_path = hf_hub_download(
            repo_id=dataset_id,
            filename=f"en/{file_name}",
            repo_type="dataset",
            local_files_only=False,
            resume_download=True
        )
        
        # Load the JSONL data
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} records from {file_name}")
        return df
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

def csv_results(bleu_scores_original, bleu_scores_quantized, bleu_differences):
    """Save BLEU score results to CSV"""
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Calculate statistics
        orig_series = pd.Series(bleu_scores_original)
        quant_series = pd.Series(bleu_scores_quantized)
        diff_series = pd.Series(bleu_differences)

        writer.writerow(["Min BLEU", "Max BLEU", "Mean BLEU", "Median BLEU", "Standard Deviation"])
        writer.writerow([diff_series.min(), diff_series.max(), diff_series.mean(), diff_series.median(), diff_series.std()])

    print(f"Results saved to {CSV_FILE}")

def arg_commandline():
    """ Create an argument parser object """
    parser = argparse.ArgumentParser(description="BLEU score evaluation for quantized text generation models")

    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
    parser.add_argument('-t', '--max_length_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
    parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int8')
    parser.add_argument('-g', '--max_new_tokens', type=int, default=50, help='maximum number of new tokens to generate')
    parser.add_argument('-n', '--num_samples', type=int, default=1000, help='number of samples to evaluate (for faster testing)')

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
    tokenizer.padding_side = 'left'
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original model - FIXED: Using AutoModelForCausalLM instead of AutoModelForSequenceClassification
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device used: {device}")

    # Move the original model to the correct device
    model = model.to(device)
    
    # Apply dynamic quantization to the model - FIXED: Using AutoModelForCausalLM
    if args.quantization_bits == 4:
        model_quantized = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            device_map="auto"  # Changed from device to "auto" for better compatibility
        )
    elif args.quantization_bits == 8:
        model_quantized = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
            device_map="auto"  # Changed from device to "auto" for better compatibility
        )
    else:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    
    print(f"Models loaded to {device}")
    
    # Load dataset
    print(f"Processing {SPECIFIC_FILE}...")
    df = load_jsonl_file(DATASET_ID, SPECIFIC_FILE)
    
    if df is None or len(df) == 0:
        print(f"No data found in {SPECIFIC_FILE}")
        return
    
    # Limit dataset size for faster evaluation
    if len(df) > args.num_samples:
        df = df.head(args.num_samples)
        print(f"Limited dataset to {args.num_samples} samples")
    
    # Create dataset and dataloader
    dataset = XP3Dataset(df, tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=args.prefetch_factor
    )
    
    # Lists to store BLEU scores
    bleu_scores_original = []
    bleu_scores_quantized = []
    bleu_differences = []
    
    print(f"Running text generation evaluation with batch size {args.batch_size} on {device}")
    
    # Set models to evaluation mode
    model.eval()
    model_quantized.eval()
    
    for batch in tqdm(dataloader, desc="Evaluating text generation with BLEU"):
        # Skip empty batches
        if batch is None:
            continue
            
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        input_texts = batch["input_texts"]
        target_texts = batch["target_texts"]
        
        # Generate text with original model
        original_generated = generate_text(
            model, tokenizer, input_ids, attention_mask, args.max_new_tokens
        )
        
        # Generate text with quantized model
        quantized_generated = generate_text(
            model_quantized, tokenizer, input_ids, attention_mask, args.max_new_tokens
        )
        
        # Calculate BLEU scores
        # If we have target texts, use them as references
        if target_texts and all(target_texts):
            references = target_texts
        else:
            # If no targets, use original model output as reference
            references = original_generated
        
        # Calculate BLEU scores for original model
        bleu_orig = calculate_bleu_scores(references, original_generated)
        
        # Calculate BLEU scores for quantized model
        bleu_quant = calculate_bleu_scores(references, quantized_generated)
        
        # Calculate differences
        bleu_diff = [orig - quant for orig, quant in zip(bleu_orig, bleu_quant)]
        
        # Store results
        bleu_scores_original.extend(bleu_orig)
        bleu_scores_quantized.extend(bleu_quant)
        bleu_differences.extend(bleu_diff)
    
    # Save results to CSV
    csv_results(bleu_scores_original, bleu_scores_quantized, bleu_differences)
    
    # Free cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
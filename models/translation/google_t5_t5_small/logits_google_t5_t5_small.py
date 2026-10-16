from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
MODEL_ID = "google-t5/t5-small"
DATASET_ID = "bigscience/xP3"
TMP_FNAME = "bigscience_xP3"
SPECIFIC_FILE = "xp3_GEM_wiki_lingua_ar_en_test_xp3longwritearticle.jsonl"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()


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
        
        # Skip if text is too short
        if len(input_text.split()) < 5:
            return None
            
        # Tokenize the input text for generation
        inputs = self.tokenizer(
            input_text, 
            truncation=True,
            padding=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt"
        )
        
        # Keep target text as string for BLEU evaluation
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "target_text": target_text,
            "input_text": input_text
        }

def collate_fn(batch):
    """Custom collate function to filter out None values"""
    # Filter out None values (skipped examples)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Get max length in this batch
    max_input_length = max(len(item["input_ids"]) for item in batch)
    
    # Prepare lists for batched data
    input_ids = []
    attention_masks = []
    target_texts = []
    input_texts = []
    
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
        
        # Keep target texts as strings
        target_texts.append(item["target_text"])
        input_texts.append(item["input_text"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "target_texts": target_texts,
        "input_texts": input_texts
    }

def load_jsonl_file(dataset_id, file_name):
    """Function to extract and process the specific JSONL file"""
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

def generate_text(model, tokenizer, input_ids, attention_mask, device):
    """Generate text using the model"""
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_generation_length,
            num_beams=args.num_beams,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_texts = []
    for i, generated_sequence in enumerate(generated_ids):
        # Remove input tokens from generated sequence
        input_len = input_ids[i].shape[0]
        generated_tokens = generated_sequence[input_len:]
        
        generated_text = tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        generated_texts.append(generated_text)
    
    return generated_texts

def calculate_bleu_scores(references, hypotheses):
    """Calculate BLEU scores for a batch of translations"""
    sentence_bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        # Calculate sentence-level BLEU score
        score = sentence_bleu(hyp, [ref]).score
        sentence_bleu_scores.append(score)
    
    return sentence_bleu_scores

def csv_results(results_data):
    """Save results to CSV file"""
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Calculate and write statistics
        original_bleu_scores = [d['original_bleu'] for d in results_data]
        quantized_bleu_scores = [d['quantized_bleu'] for d in results_data]
        bleu_differences = [d['bleu_diff'] for d in results_data]

        writer.writerow(["Min BLEU", "Max BLEU", "Mean BLEU", "Median BLEU", "Standard Deviation"])
        writer.writerow([np.min(bleu_differences), np.max(bleu_differences), np.mean(bleu_differences), np.median(bleu_differences), np.std(bleu_differences)])

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
    
    # Add parameters for text generation
    parser.add_argument('-g', '--max_generation_length', type=int, default=512, help='maximum length for generated text')
    parser.add_argument('-n', '--num_beams', type=int, default=4, help='number of beams for beam search')
    parser.add_argument('-l', '--limit_samples', type=int, default=5000, help='limit number of samples to evaluate (for faster testing)')

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
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device used: {device}")

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
    
    # Limit samples for faster evaluation if specified
    if args.limit_samples > 0 and len(df) > args.limit_samples:
        df = df.head(args.limit_samples)
        print(f"Limited to {len(df)} samples for evaluation")
    
    # Create dataset and dataloader
    dataset = XP3Dataset(df, tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Keep order for consistent evaluation
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # List to store all results
    results_data = []
    counter = 0
    
    print(f"Running text generation and BLEU evaluation with batch size {args.batch_size} on {device}")
    
    for batch in tqdm(dataloader, desc="Evaluating model generations"):
        # Skip empty batches
        if batch is None:
            continue
            
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_texts = batch["target_texts"]
        input_texts = batch["input_texts"]
        
        # Generate text with original model
        original_generated = generate_text(model, tokenizer, input_ids, attention_mask, device)
        
        # Generate text with quantized model
        quantized_generated = generate_text(model_quantized, tokenizer, input_ids, attention_mask, device)
        
        # Calculate BLEU scores
        original_bleu_scores = calculate_bleu_scores(target_texts, original_generated)
        quantized_bleu_scores = calculate_bleu_scores(target_texts, quantized_generated)
        
        # Store results
        for i in range(len(input_texts)):
            bleu_diff = original_bleu_scores[i] - quantized_bleu_scores[i]
            
            results_data.append({
                'index': counter,
                'original_bleu': original_bleu_scores[i],
                'quantized_bleu': quantized_bleu_scores[i],
                'bleu_diff': bleu_diff,
                'input_text': input_texts[i],
                'reference_text': target_texts[i],
                'original_generated': original_generated[i],
                'quantized_generated': quantized_generated[i]
            })
            counter += 1
    
    # Save results to CSV
    csv_results(results_data)

    # Free cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
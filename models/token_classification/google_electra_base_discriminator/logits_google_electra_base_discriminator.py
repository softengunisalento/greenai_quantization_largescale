from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import argparse
import tempfile
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from seqeval.metrics import accuracy_score as seqeval_accuracy_score


### Constants
MODEL_ID = "google/electra-base-discriminator"
DATASET_ID = "ade-benchmark-corpus/ade_corpus_v2"
TMP_FNAME = "ade_benchmark_corpus_ade_corpus_v2"
SPECIFIC_FILE = "train-00000-of-00001.parquet"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()


### Functions
class ADEDataset(Dataset):
    """Dataset for ADE corpus classification task"""
    
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist() if 'labels' in df.columns else None
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Skip if text is too short
        if len(text.split()) < 5:
            return None
            
        # Tokenize the text
        inputs = self.tokenizer(
            text, 
            truncation=True,
            max_length=args.max_length_tokenizer,
            return_tensors="pt",
            is_split_into_words=False,
            padding=False
        )
        
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text
        }
        
        # Add labels if available
        if self.labels is not None:
            labels = self.labels[idx]
            # Convert labels to tensor if they're not already
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            result["labels"] = labels
            
        return result

def collate_fn(batch):
    """Custom collate function to filter out None values and pad sequences"""
    # Filter out None values (skipped examples)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Get max length in this batch
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Prepare lists for batched data
    input_ids = []
    attention_masks = []
    texts = []
    labels = []
    
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
        
        texts.append(item["text"])
        
        # Handle labels if present
        if "labels" in item:
            label = item["labels"]
            if len(label) < max_length:
                # Pad labels with -100 (ignored in loss calculation)
                padded_label = torch.cat([
                    label,
                    torch.tensor([-100] * padding_len, dtype=torch.long)
                ])
            else:
                padded_label = label[:max_length]
            labels.append(padded_label)
    
    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "texts": texts
    }
    
    if labels:
        result["labels"] = torch.stack(labels)
    
    return result

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
            filename=f"Ade_corpus_v2_classification/{file_name}",
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

def get_predictions(model, input_ids, attention_mask, id2label):
    """Get NER predictions from model"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Convert predictions to labels
    batch_predictions = []
    for i, pred_sequence in enumerate(predictions):
        pred_labels = []
        for j, pred_id in enumerate(pred_sequence):
            if attention_mask[i][j] == 1:  # Only consider non-padded tokens
                pred_labels.append(id2label[pred_id.item()])
        batch_predictions.append(pred_labels)
    
    return batch_predictions

def align_predictions_with_labels(predictions, labels, id2label):
    """Align predictions with true labels for evaluation"""
    aligned_predictions = []
    aligned_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        pred_labels = []
        true_labels = []
        
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:  # Skip padded tokens
                pred_labels.append(id2label[pred_id])
                true_labels.append(id2label[label_id])
        
        aligned_predictions.append(pred_labels)
        aligned_labels.append(true_labels)
    
    return aligned_predictions, aligned_labels

def calculate_metrics(true_labels, pred_labels):
    """Calculate various NER metrics"""
    
    # Flatten for token-level metrics
    flat_true = [label for sent in true_labels for label in sent]
    flat_pred = [label for sent in pred_labels for label in sent]
    
    # Token-level metrics
    token_accuracy = accuracy_score(flat_true, flat_pred)
    token_precision = precision_score(flat_true, flat_pred, average='weighted', zero_division=0)
    token_recall = recall_score(flat_true, flat_pred, average='weighted', zero_division=0)
    token_f1 = f1_score(flat_true, flat_pred, average='weighted', zero_division=0)
    
    # Entity-level metrics using seqeval
    entity_accuracy = seqeval_accuracy_score(true_labels, pred_labels)
    entity_precision = seqeval_precision_score(true_labels, pred_labels, zero_division=0)
    entity_recall = seqeval_recall_score(true_labels, pred_labels, zero_division=0)
    entity_f1 = seqeval_f1_score(true_labels, pred_labels, zero_division=0)
    
    return {
        'token_accuracy': token_accuracy,
        'token_precision': token_precision,
        'token_recall': token_recall,
        'token_f1': token_f1,
        'entity_accuracy': entity_accuracy,
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'entity_f1': entity_f1
    }

def csv_results(original_metrics, quantized_metrics, comparison_metrics):
    """Save evaluation results to CSV"""
    
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers
        writer.writerow(["Model", "Token Accuracy", "Token Precision", "Token Recall", "Token F1",
                        "Entity Accuracy", "Entity Precision", "Entity Recall", "Entity F1"])
        
        # Write original model metrics
        writer.writerow(["Original", 
                        original_metrics['token_accuracy'],
                        original_metrics['token_precision'],
                        original_metrics['token_recall'],
                        original_metrics['token_f1'],
                        original_metrics['entity_accuracy'],
                        original_metrics['entity_precision'],
                        original_metrics['entity_recall'],
                        original_metrics['entity_f1']])
        
        # Write quantized model metrics
        writer.writerow(["Quantized", 
                        quantized_metrics['token_accuracy'],
                        quantized_metrics['token_precision'],
                        quantized_metrics['token_recall'],
                        quantized_metrics['token_f1'],
                        quantized_metrics['entity_accuracy'],
                        quantized_metrics['entity_precision'],
                        quantized_metrics['entity_recall'],
                        quantized_metrics['entity_f1']])
        
        # Write comparison (differences)
        writer.writerow([])
        writer.writerow(["Metric", "Difference (Original - Quantized)", "Relative Change (%)"])
        
        for metric in original_metrics.keys():
            diff = original_metrics[metric] - quantized_metrics[metric]
            rel_change = (diff / original_metrics[metric]) * 100 if original_metrics[metric] != 0 else 0
            writer.writerow([metric, diff, rel_change])
    
    print(f"Results saved to {CSV_FILE}")

def arg_commandline():
    """Create an argument parser object"""
    parser = argparse.ArgumentParser(description="Evaluate NER model performance with quantization")

    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
    parser.add_argument('-t', '--max_length_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
    parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')
    parser.add_argument('-s', '--sample_size', type=int, default=None, help='number of samples to evaluate (None for all)')

    return parser.parse_args()


### Main
def main():
    global tokenizer
    
    # Load the tokenizer and model for token classification
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    
    # Get label mappings
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")

    # Move the original model to the correct device
    model = model.to(device)
    
    # Apply quantization to the model
    if args.quantization_bits == 4:
        model_quantized = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            device_map=device
        )
    elif args.quantization_bits == 8:
        model_quantized = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
            device_map=device
        )
    else:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    
    print(f"Models loaded to {device}")
    
    # Load dataset
    print(f"Processing {SPECIFIC_FILE}...")
    df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
    
    if df is None or len(df) == 0:
        print(f"No data found in {SPECIFIC_FILE}")
        exit()
    
    # Sample data if specified
    if args.sample_size is not None:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        print(f"Sampled {len(df)} records for evaluation")
    
    # Create dataset and dataloader
    dataset = ADEDataset(df, tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Lists to store all predictions and labels
    all_original_predictions = []
    all_quantized_predictions = []
    all_true_labels = []
    
    print(f"Running evaluation with batch size {args.batch_size} on {device}")
    
    # Set models to evaluation mode
    model.eval()
    model_quantized.eval()
    
    for batch in tqdm(dataloader, desc="Evaluating model performance"):
        # Skip empty batches
        if batch is None:
            continue
            
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Get predictions from both models
        original_predictions = get_predictions(model, input_ids, attention_mask, id2label)
        quantized_predictions = get_predictions(model_quantized, input_ids, attention_mask, id2label)
        
        # If we have labels, use them for evaluation
        if "labels" in batch:
            labels = batch["labels"].to(device)
            # Convert labels to the same format as predictions
            true_labels = []
            for i, label_sequence in enumerate(labels):
                label_list = []
                for j, label_id in enumerate(label_sequence):
                    if attention_mask[i][j] == 1 and label_id != -100:
                        label_list.append(id2label[label_id.item()])
                true_labels.append(label_list)
            
            all_true_labels.extend(true_labels)
        
        all_original_predictions.extend(original_predictions)
        all_quantized_predictions.extend(quantized_predictions)
    
    # Calculate metrics if we have true labels
    if all_true_labels:
        print("\nCalculating metrics with ground truth labels...")
        
        original_metrics = calculate_metrics(all_true_labels, all_original_predictions)
        quantized_metrics = calculate_metrics(all_true_labels, all_quantized_predictions)
        
        # Calculate differences
        comparison_metrics = {}
        for metric in original_metrics.keys():
            comparison_metrics[metric] = original_metrics[metric] - quantized_metrics[metric]
        
        # Print results
        print(f"\nOriginal Model Performance:")
        for metric, value in original_metrics.items():
            print(f"  {metric}: {value}")
        
        print(f"\nQuantized Model Performance:")
        for metric, value in quantized_metrics.items():
            print(f"  {metric}: {value}")
        
        print(f"\nPerformance Differences (Original - Quantized):")
        for metric, value in comparison_metrics.items():
            print(f"  {metric}: {value}")
        
        # Save results to CSV
        csv_results(original_metrics, quantized_metrics, comparison_metrics)
    
    else:
        print("No ground truth labels available. Comparing predictions between models...")
        
        # Calculate agreement between models
        total_tokens = sum(len(pred) for pred in all_original_predictions)
        matching_tokens = sum(
            sum(1 for orig, quant in zip(orig_pred, quant_pred) if orig == quant)
            for orig_pred, quant_pred in zip(all_original_predictions, all_quantized_predictions)
        )
        
        agreement = matching_tokens / total_tokens if total_tokens > 0 else 0
        print(f"\nPrediction Agreement between Original and Quantized models: {agreement}")
        
        # Save basic comparison
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Prediction Agreementetric", "Total Tokens Compared", "Matching Tokens"])
            writer.writerow([agreement, total_tokens, matching_tokens])
    
    # Free cache
    torch.cuda.empty_cache()
    print("\nEvaluation completed!")


if __name__ == "__main__":
    args = arg_commandline()
    main()
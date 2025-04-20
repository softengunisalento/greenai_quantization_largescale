from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import argparse
import csv
import os
import time
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from codecarbon import EmissionsTracker
from sklearn.metrics.pairwise import cosine_similarity

### Constants
N_BYTES = 1
CARBON_INTENSITY = 475  # Global average carbon intensity
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DATASET_ID = "mandarjoshi/trivia_qa"
TMP_FNAME = "mandarjoshi_trivia_qa"
SPECIFIC_FILE = "test-00000-of-00004.parquet"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]
        
        # Process the text
        try:
            return {
                "text": item["text"],
                "question_id": item["question_id"]
            }
        except Exception as e:
            print(f"Error loading text {item['question_id']}: {e}")
            # Return a placeholder to avoid crashing
            return {
                "text": "Error loading text",
                "question_id": item["question_id"] + "_ERROR"
            }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        "texts": [item["text"] for item in batch],
        "question_ids": [item["question_id"] for item in batch]
    }

def process_parquet_file(dataset_id, file_name):
    """ Function to download and process the specific parquet file with authentication """
    # Create a consistent temp directory name
    temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download the specific file with tqdm progress bar
    from huggingface_hub import hf_hub_download

    try:
        file_path = hf_hub_download(
            repo_id=dataset_id,
            filename=f"rc/{file_name}",
            repo_type="dataset",
            local_files_only=False,
            resume_download=True
        )
        
        print(f"Downloaded file to: {file_path}")
        
        # Read the parquet file
        data = pd.read_parquet(file_path)
        print(f"Loaded {len(data)} records from parquet file")
        
        # Process the data
        text_data = []
        max_texts = 500  # Limit to 500 texts for faster testing
        
        # Check columns and adjust based on actual data structure
        if 'question' in data.columns and 'qid' in data.columns:
            for idx, row in tqdm(data.iloc[:max_texts].iterrows(), desc="Processing text data"):
                text_data.append({
                    "text": row['question'],
                    "question_id": row['qid']
                })
        else:
            print(f"Warning: Expected columns not found. Available columns: {data.columns}")
            # Adapt to actual data structure
            for idx, row in tqdm(data.iloc[:max_texts].iterrows(), desc="Processing text data"):
                # Use first column as text and index as ID if structure doesn't match
                text_data.append({
                    "text": str(row.iloc[0]),
                    "question_id": f"text_{idx}"
                })
                
        return text_data
        
    except Exception as e:
        print(f"Error downloading or processing dataset: {e}")
        return []

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on the model output"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def evaluate_size(model):
    """ Evaluate the size of the model """
    # calculating the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # return bytes
    return total_params * N_BYTES

def calculate_energy_consumption(power_watts, time_seconds):
    """ Calculate energy consumption in Joules """
    return power_watts * time_seconds

def calculate_co2_emissions(energy_joules):
    """ Calculate CO2 emissions (in grams) """
    # Convert joules to megajoules (1 MJ = 1e6 J)
    energy_megajoules = energy_joules / 1e6
    return energy_megajoules * CARBON_INTENSITY

def measure_energy_consumption(func):
    """ Decorator to measure energy consumption of a function """
    def wrapper(*args, **kwargs):
        # Start the CodeCarbon tracker
        tracker = EmissionsTracker(
            log_level="error",
            save_to_file=False
        )
        tracker.start()
        
        # Start the inference tracker
        start_time = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # measure the power consumption after inference
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Stop the CodeCarbon tracker
        tracker.stop()
        
        co2_emissions = calculate_co2_emissions(tracker._total_energy.kWh)
        
        # kWh to Wh
        cpu_energy_Wh = tracker._total_cpu_energy.kWh * 1000
        gpu_energy_Wh = tracker._total_gpu_energy.kWh * 1000
        ram_energy_Wh = tracker._total_ram_energy.kWh * 1000
        total_energy_Wh = tracker._total_energy.kWh * 1000

        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # write stats
            writer.writerow(["Inference s", "CPU Wh", "GPU Wh", "RAM Wh", "Energy Wh", "CO2 Emissions gr"])
            writer.writerow([inference_time, cpu_energy_Wh, gpu_energy_Wh, ram_energy_Wh, total_energy_Wh, co2_emissions])

        return result
    return wrapper

def generate_embeddings(texts, model, tokenizer, device, batch_size=32):
    """Generate embeddings for a list of texts"""
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            
        # Mean pooling
        embeddings = mean_pooling(outputs, encoded['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(embeddings.cpu())
    
    # Concatenate all batch embeddings
    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        return torch.tensor([])

def calculate_similarities(query_embeddings, document_embeddings):
    """Calculate cosine similarity between query and document embeddings"""
    # Convert to numpy for sklearn's cosine_similarity
    query_np = query_embeddings.numpy()
    doc_np = document_embeddings.numpy()
    
    # Calculate similarities
    similarities = cosine_similarity(query_np, doc_np)
    
    return similarities

def csv_results(model, embeddings_dict):
    """ Save results to CSV """
    # Check the memory footprint of the model
    mem_footprint = model.get_memory_footprint() if hasattr(model, 'get_memory_footprint') else 0

    # Checking the size of the model
    module_sizes = evaluate_size(model)

    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write embeddings stats
        writer.writerow(["Question ID", "Embedding Norm"])

        for question_id in embeddings_dict:
            # Calculate embedding norm as a simple metric to save
            embedding_norm = torch.norm(embeddings_dict[question_id]).item()
            writer.writerow([question_id, embedding_norm])
        
        # Write metrics
        writer.writerow([])  # Empty row as separator
        writer.writerow(["Footprint MBs", "Size GB"])
        writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

    print(f"Results saved to {CSV_FILE}")

def arg_commandline():
    """ Create an argument parser object """
    parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

    # Add arguments/options
    parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
    # Add parameters for dataloader
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
    parser.add_argument('-t', '--max_length_tokenizer', type=int, default=512, help='maximum sequence length for tokenization')
    parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='quantization bits (8 for quantization)')

    # Parse the arguments
    return parser.parse_args()


### Main
def main():
    args = arg_commandline()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Apply dynamic quantization to the model
    if args.quantization_bits == 4:
        model_quantized = AutoModel.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    elif args.quantization_bits == 8:
        model_quantized = AutoModel.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    
    model_quantized.eval()

    # Download and process texts from the parquet file
    print(f"Processing {SPECIFIC_FILE} from {DATASET_ID}...")
    text_data = process_parquet_file(DATASET_ID, SPECIFIC_FILE)

    if not text_data:
        print("No text data found. Exiting.")
        exit()
    
    # Create dataset and dataloader
    dataset = TextDataset(
        text_data, 
        tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Dictionary to store embeddings for each question
    embeddings_dict = {}

    # Process each batch of texts
    print(f"Running inference on {len(text_data)} texts with {args.quantization_bits}-bit precision")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing texts")):
        try:
            # Tokenize text
            encoded_texts = tokenizer(
                batch["texts"],
                padding=True,
                truncation=True,
                max_length=args.max_length_tokenizer,
                return_tensors='pt'
            ).to(device)

            # Inference:
            with torch.no_grad():  # no need to calculate gradients
                model_output = model_quantized(**encoded_texts)

            # Perform mean pooling - ensuring both tensors are on the same device
            embeddings = mean_pooling(model_output, encoded_texts['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Detach and move to CPU before storing
            embeddings_cpu = embeddings.detach().cpu()
            
            # Store embeddings
            for question_id, embedding in zip(batch["question_ids"], embeddings_cpu):
                embeddings_dict[question_id] = embedding

        except Exception as e:
            print(f"Failed to process batch {batch_idx}: {str(e)}")
    
    # Save results
    csv_results(model_quantized, embeddings_dict)


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
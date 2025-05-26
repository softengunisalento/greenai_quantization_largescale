from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from torch.utils.data import Dataset, DataLoader
import gc
import os
import csv
import torch
import argparse
import requests
import pandas as pd
import tempfile
import zipfile
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from huggingface_hub import hf_hub_download

### Constants
MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_ID = "sayakpaul/indoor-scenes"
TMP_FNAME = "indoor_scenes_dataset"
CANDIDATE_LABELS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"

### Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_files, processor):
        self.image_files = image_files
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        item = self.image_files[idx]
        
        # Open and process the image
        try:
            image = Image.open(item["path"]).convert("RGB")
            return {
                "image": image,
                "file_name": item["file_name"]
            }
        except Exception as e:
            print(f"Error loading {item['file_name']}: {e}")
            # Return a placeholder to avoid crashing
            return {
                "image": Image.new('RGB', (224, 224), color='black'),
                "file_name": item["file_name"] + "_ERROR"
            }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        "images": [item["image"] for item in batch],
        "file_names": [item["file_name"] for item in batch]
    }

def process_dataset(dataset_id, cache_dir=None):
    """ Download dataset files using hf_hub_download """
    print(f"Downloading dataset '{dataset_id}'...")
    
    # Create a consistent temp directory name if not provided
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
        os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download the images archive
        file_path = hf_hub_download(
            repo_id=dataset_id,
            filename="Images.zip",  # The main image archive
            repo_type="dataset",
            local_files_only=False,
            cache_dir=cache_dir
        )
        
        print(f"Successfully downloaded dataset to: {file_path}")
        
        # Extract the files if needed
        extracted_dir = os.path.join(cache_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Check if we need to unzip 
        if file_path.endswith('.zip') and not os.listdir(extracted_dir):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            print("Extraction complete!")
        
        return extracted_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Fallback to another dataset if the first fails
        try:
            fallback_dataset = "frgfm/imagenette"
            file_path = hf_hub_download(
                repo_id=fallback_dataset,
                filename="imagenette2-320.zip",
                repo_type="dataset",
                local_files_only=False,
                cache_dir=cache_dir
            )
            
            # Extract the fallback dataset
            fallback_extracted = os.path.join(cache_dir, "fallback_extracted")
            os.makedirs(fallback_extracted, exist_ok=True)
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(fallback_extracted)
                
            return fallback_extracted
            
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            
            # Create some placeholder dummy images for testing
            dummy_dir = os.path.join(cache_dir, "dummy_images")
            os.makedirs(dummy_dir, exist_ok=True)
            
            # Create 10 dummy images
            for i in range(10):
                dummy_img = Image.new('RGB', (224, 224), color=(i*20, 100, 200))
                dummy_img.save(os.path.join(dummy_dir, f'dummy_image_{i}.jpg'))

            return dummy_dir

def get_image_files(dataset_dir, max_images=100):
    """ Collect image files from dataset directory """
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG', '.PNG']
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append({
                    "path": os.path.join(root, file),
                    "file_name": file
                })
                
                # Limit the number of images to process
                if len(image_files) >= max_images:
                    return image_files
    
    return image_files

def csv_results(mse_values, max_diff_values):
    """ Save results to CSV file """
    mse_series = pd.Series(list(mse_values.values()))

    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "MSE", "Max Difference"])
        
        for idx in mse_values:
            writer.writerow([idx, mse_values[idx], max_diff_values[idx]])
        
        writer.writerow(["Min MSE", "Max MSE", "Mean MSE", "Median MSE", "Standard Deviation"])
        writer.writerow([mse_series.min(), mse_series.max(), mse_series.mean(), mse_series.median(), mse_series.std()])

    print(f"Results saved to {CSV_FILE}")

def arg_commandline():
    parser = argparse.ArgumentParser(description="CLIP Quantization Evaluation on Indoor Scenes Dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='size of the batches for DataLoader')
    parser.add_argument("-n", "--num_images", type=int, default=100, help='maximum number of images to process')
    parser.add_argument('-q', '--quantization_bits', type=int, default=8, 
                        help='value to quantize (4 for int4 or 8 for int8)')
    parser.add_argument('-c', '--cache_dir', type=str, default=None, 
                        help='directory to cache downloaded dataset')
    return parser.parse_args()

### Main
def main():
    args = arg_commandline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download and prepare the dataset
    dataset_dir = process_dataset(DATASET_ID, args.cache_dir)
    image_files = get_image_files(dataset_dir, args.num_images)
    print(f"Found {len(image_files)} images to process")

    # Load processor and models
    print("Loading CLIP processor and models...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Load original model
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Load quantized model
    if args.quantization_bits == 4:
        print("Loading 4-bit quantized model...")
        model_quantized = AutoModelForZeroShotImageClassification.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            device_map=device
        )
    elif args.quantization_bits == 8:
        print("Loading 8-bit quantized model...")
        model_quantized = AutoModelForZeroShotImageClassification.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
            device_map=device
        )
    else:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    
    model_quantized.eval()

    # Dataset + DataLoader
    dataset = ImageDataset(image_files, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=0  # Increase if needed for your system
    )

    mse_values = {}
    max_diff_values = {}

    print(f"Evaluating with {args.quantization_bits}-bit quantization...")
    for batch in tqdm(dataloader, desc="Processing"):
        images = batch["images"]
        file_names = batch["file_names"]

        inputs = processor(images=images, text=CANDIDATE_LABELS, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            original_outputs = model(**inputs)
            quantized_outputs = model_quantized(**inputs)

        logits_orig = original_outputs.logits_per_image.cpu()
        logits_quant = quantized_outputs.logits_per_image.cpu()

        for i in range(len(file_names)):
            mse = torch.mean((logits_orig[i] - logits_quant[i]) ** 2).item()
            max_diff = torch.max(torch.abs(logits_orig[i] - logits_quant[i])).item()
            mse_values[file_names[i]] = mse
            max_diff_values[file_names[i]] = max_diff

        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    csv_results(mse_values, max_diff_values)

if __name__ == "__main__":
    main()
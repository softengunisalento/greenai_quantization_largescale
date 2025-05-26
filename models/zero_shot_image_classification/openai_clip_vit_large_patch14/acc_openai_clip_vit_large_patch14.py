from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from torch.utils.data import Dataset, DataLoader
import gc
import os
import csv
import torch
import argparse
import pandas as pd
import tempfile
import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download

### Constants
MODEL_ID = "openai/clip-vit-large-patch14"
DATASET_ID = "sayakpaul/indoor-scenes"
TMP_FNAME = "indoor_scenes_dataset"
CANDIDATE_LABELS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"

### Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_files, processor, max_images=100):
        self.image_files = image_files[:max_images] if max_images else image_files
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

def calculate_accuracy(original_model, model_int4, model_int8, dataloader, device, processor):
    """Calculate classification accuracy for all models and measure the drops"""
    # Initialize dictionaries to store predictions
    original_predictions = {}
    int4_predictions = {}
    int8_predictions = {}
    
    for batch in tqdm(dataloader, desc="Evaluating classification accuracy"):
        images = batch["images"]
        file_names = batch["file_names"]

        # Process inputs
        inputs = processor(images=images, text=CANDIDATE_LABELS, return_tensors="pt", padding=True).to(device)
        
        # Get predictions from original model
        with torch.no_grad():
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits_per_image
            original_probs = torch.softmax(original_logits, dim=-1)
            original_pred_classes = torch.argmax(original_probs, dim=-1).cpu().numpy()
            
        # Get predictions from INT4 model
        with torch.no_grad():
            int4_outputs = model_int4(**inputs)
            int4_logits = int4_outputs.logits_per_image
            int4_probs = torch.softmax(int4_logits, dim=-1)
            int4_pred_classes = torch.argmax(int4_probs, dim=-1).cpu().numpy()
            
        # Get predictions from INT8 model
        with torch.no_grad():
            int8_outputs = model_int8(**inputs)
            int8_logits = int8_outputs.logits_per_image
            int8_probs = torch.softmax(int8_logits, dim=-1)
            int8_pred_classes = torch.argmax(int8_probs, dim=-1).cpu().numpy()
        
        # Store predictions
        for i, file_name in enumerate(file_names):
            original_predictions[file_name] = {
                "class_id": int(original_pred_classes[i]),
                "class_name": CANDIDATE_LABELS[original_pred_classes[i]],
                "confidence": float(original_probs[i][original_pred_classes[i]])
            }
            
            int4_predictions[file_name] = {
                "class_id": int(int4_pred_classes[i]),
                "class_name": CANDIDATE_LABELS[int4_pred_classes[i]],
                "confidence": float(int4_probs[i][int4_pred_classes[i]])
            }
            
            int8_predictions[file_name] = {
                "class_id": int(int8_pred_classes[i]),
                "class_name": CANDIDATE_LABELS[int8_pred_classes[i]],
                "confidence": float(int8_probs[i][int8_pred_classes[i]])
            }
    
    # Calculate accuracy metrics
    int4_match = 0
    int8_match = 0
    
    int4_conf_similarity = []
    int8_conf_similarity = []
    
    for file_name in original_predictions:
        orig = original_predictions[file_name]["class_id"]
        int4 = int4_predictions[file_name]["class_id"]
        int8 = int8_predictions[file_name]["class_id"]
        
        # Count matches
        if orig == int4:
            int4_match += 1
        if orig == int8:
            int8_match += 1
        
        # Calculate confidence similarity
        orig_conf = original_predictions[file_name]["confidence"]
        int4_conf = int4_predictions[file_name]["confidence"]
        int8_conf = int8_predictions[file_name]["confidence"]
        
        int4_conf_similarity.append(abs(orig_conf - int4_conf) / orig_conf if orig_conf > 0 else 0)
        int8_conf_similarity.append(abs(orig_conf - int8_conf) / orig_conf if orig_conf > 0 else 0)
    
    # Calculate metrics
    total_files = len(original_predictions)
    int4_match_rate = int4_match / total_files if total_files > 0 else 0
    int8_match_rate = int8_match / total_files if total_files > 0 else 0
    
    int4_avg_conf_similarity = 1 - (sum(int4_conf_similarity) / len(int4_conf_similarity) if int4_conf_similarity else 0)
    int8_avg_conf_similarity = 1 - (sum(int8_conf_similarity) / len(int8_conf_similarity) if int8_conf_similarity else 0)
    
    # Calculate drops
    int4_drop = 1 - int4_match_rate
    int8_drop = 1 - int8_match_rate
    
    int4_conf_drop = 1 - int4_avg_conf_similarity
    int8_conf_drop = 1 - int8_avg_conf_similarity
    
    return {
        "total_files": total_files,
        "int4_match_rate": int4_match_rate,
        "int8_match_rate": int8_match_rate,
        "int4_avg_conf_similarity": int4_avg_conf_similarity,
        "int8_avg_conf_similarity": int8_avg_conf_similarity,
        "int4_drop": int4_drop,
        "int8_drop": int8_drop,
        "int4_conf_drop": int4_conf_drop,
        "int8_conf_drop": int8_conf_drop,
        "original_predictions": original_predictions,
        "int4_predictions": int4_predictions,
        "int8_predictions": int8_predictions
    }

def calculate_hidden_state_mse(original_model, model_int4, model_int8, dataloader, device, processor):
    """Calculate MSE and max difference between original and quantized models"""
    # Dictionaries to store MSE values and max differences for each quantized model
    mse_values_int4 = {}
    mse_values_int8 = {}
    max_diff_values_int4 = {}
    max_diff_values_int8 = {}

    for batch in tqdm(dataloader, desc="Calculating hidden state differences"):
        images = batch["images"]
        file_names = batch["file_names"]

        inputs = processor(images=images, text=CANDIDATE_LABELS, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # Get outputs from original model
            original_outputs = original_model(**inputs).logits_per_image.detach().cpu().numpy()
            
            # Get outputs from INT4 model
            int4_outputs = model_int4(**inputs).logits_per_image.detach().cpu().numpy()
            
            # Get outputs from INT8 model
            int8_outputs = model_int8(**inputs).logits_per_image.detach().cpu().numpy()
            
            # Calculate metrics for each image
            for i, file_name in enumerate(file_names):
                # INT4 model metrics
                mse_int4 = np.mean((original_outputs[i] - int4_outputs[i]) ** 2)
                max_diff_int4 = np.max(np.abs(original_outputs[i] - int4_outputs[i]))
                
                # INT8 model metrics
                mse_int8 = np.mean((original_outputs[i] - int8_outputs[i]) ** 2)
                max_diff_int8 = np.max(np.abs(original_outputs[i] - int8_outputs[i]))
                
                # Store results
                mse_values_int4[file_name] = float(mse_int4)
                max_diff_values_int4[file_name] = float(max_diff_int4)
                
                mse_values_int8[file_name] = float(mse_int8)
                max_diff_values_int8[file_name] = float(max_diff_int8)

        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return {
        "mse_values_int4": mse_values_int4,
        "mse_values_int8": mse_values_int8,
        "max_diff_values_int4": max_diff_values_int4,
        "max_diff_values_int8": max_diff_values_int8
    }

def csv_results(mse_results, accuracy_results):
    """ Print MSE statistics and accuracy results """
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write accuracy results first (most important)
        writer.writerow(["Classification Accuracy Metrics"])
        writer.writerow(["Metric", "Original", "Int4", "Int8"])
        writer.writerow(["Match Rate", "1.0", 
                        accuracy_results["int4_match_rate"], 
                        accuracy_results["int8_match_rate"]])
        writer.writerow(["Confidence Similarity", "1.0", 
                        accuracy_results["int4_avg_conf_similarity"], 
                        accuracy_results["int8_avg_conf_similarity"]])
        writer.writerow(["Match Drop", "N/A", 
                        accuracy_results["int4_drop"], 
                        accuracy_results["int8_drop"]])
        writer.writerow(["Confidence Drop", "N/A", 
                        accuracy_results["int4_conf_drop"], 
                        accuracy_results["int8_conf_drop"]])
        writer.writerow(["Total files processed", accuracy_results["total_files"]])
        
        writer.writerow([])  # Empty row for separation
        
        if mse_results:
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
    parser = argparse.ArgumentParser(description="CLIP Quantization Evaluation on Indoor Scenes Dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='size of the batches for DataLoader')
    parser.add_argument("-n", "--num_images", type=int, default=100, help='maximum number of images to process')
    parser.add_argument('-q', '--quantization_bits', type=int, default=0, 
                        help='specific quantization to analyze (4 for int4, 8 for int8, 0 for both)')
    parser.add_argument('-c', '--cache_dir', type=str, default=None, 
                        help='directory to cache downloaded dataset')
    parser.add_argument('-w', '--num_workers', type=int, default=4, 
                        help='number of workers for DataLoader')
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

    # Load INT4 model
    print("Loading 4-bit quantized model...")
    model_int4 = AutoModelForZeroShotImageClassification.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        device_map=device
    )
    model_int4.eval()
    
    # Load INT8 model
    print("Loading 8-bit quantized model...")
    model_int8 = AutoModelForZeroShotImageClassification.from_pretrained(
        MODEL_ID,
        load_in_8bit=True,
        device_map=device
    )
    model_int8.eval()

    # Dataset + DataLoader
    dataset = ImageDataset(image_files, processor, args.num_images)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Calculate hidden state MSE
    print("Calculating hidden state differences...")
    mse_results = calculate_hidden_state_mse(model, model_int4, model_int8, dataloader, device, processor)

    # Calculate accuracy for both quantization methods
    print("Calculating accuracy...")
    accuracy_results = calculate_accuracy(model, model_int4, model_int8, dataloader, device, processor)
    
    # Print results to console
    print("\nAccuracy Results:")
    print(f"Original model used as ground truth reference (accuracy: 1.0)")
    print(f"Int4 model match rate: {accuracy_results['int4_match_rate']:.4f} (drop: {accuracy_results['int4_drop']:.4f})")
    print(f"Int8 model match rate: {accuracy_results['int8_match_rate']:.4f} (drop: {accuracy_results['int8_drop']:.4f})")
    print(f"Int4 model confidence similarity: {accuracy_results['int4_avg_conf_similarity']:.4f} (drop: {accuracy_results['int4_conf_drop']:.4f})")
    print(f"Int8 model confidence similarity: {accuracy_results['int8_avg_conf_similarity']:.4f} (drop: {accuracy_results['int8_conf_drop']:.4f})")
    print(f"Total files processed: {accuracy_results['total_files']}")
    
    # Save results to CSV
    csv_results(mse_results, accuracy_results)
    
    # Generate a comparison report
    if accuracy_results["int4_match_rate"] > accuracy_results["int8_match_rate"]:
        print("\nQuantization Comparison: Int4 outperforms Int8 in classification accuracy!")
    elif accuracy_results["int8_match_rate"] > accuracy_results["int4_match_rate"]:
        print("\nQuantization Comparison: Int8 outperforms Int4 in classification accuracy!")
    else:
        print("\nQuantization Comparison: Int4 and Int8 have identical classification accuracy.")
        
    if mse_results:
        int4_mse_mean = pd.Series(list(mse_results["mse_values_int4"].values())).mean()
        int8_mse_mean = pd.Series(list(mse_results["mse_values_int8"].values())).mean()
        
        if int4_mse_mean < int8_mse_mean:
            print("Int4 has lower mean MSE than Int8, suggesting better hidden state preservation.")
        else:
            print("Int8 has lower mean MSE than Int4, suggesting better hidden state preservation.")

    # Free cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
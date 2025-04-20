from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import tarfile
import argparse
import tempfile
import pandas as pd
import gc
from PIL import Image
from huggingface_hub import login, hf_hub_download


### Constants
MODEL_ID = "google/vit-base-patch16-224-in21k"
DATASET_ID = "ILSVRC/imagenet-1k"
TMP_FNAME = "imagenet_val_images"
SPECIFIC_FILE = "val_images.tar.gz"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"
HF_TOKEN = "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"


### Functions
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

def process_tar_file(dataset_id, file_name):
	""" Function to extract and process the specific tar file with authentication """
	# Login with token
	print("Authenticating with Hugging Face...")
	login(token=HF_TOKEN)
	
	# Create a consistent temp directory name
	temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
	extracted_dir = os.path.join(temp_dir, "images")
	
	# Check if the temp directory already exists and has content
	if os.path.exists(extracted_dir) and os.listdir(extracted_dir):
		print(f"Found existing temporary directory: {extracted_dir}")
	else:
		# Download the specific file with tqdm progress bar
		try:
			file_path = hf_hub_download(
				repo_id=dataset_id,
				filename=f"data/{file_name}",
				repo_type="dataset",
				local_files_only=False,
				resume_download=True,
				token=HF_TOKEN
			)
			
			# Create the temp directory if it doesn't exist
			os.makedirs(extracted_dir, exist_ok=True)
			
			# Extract image data from the tar file with tqdm progress
			with tarfile.open(file_path, 'r:gz') as tar:
				members = tar.getmembers()
				print(f"Extracting {len(members)} files from tar archive...")
				
				for member in tqdm(members, desc="Extracting files"):
					tar.extract(member, path=extracted_dir)
				
				print(f"Extracted to directory: {extracted_dir}")
		except Exception as e:
			print(f"Error downloading or extracting dataset: {e}")
	
	# Process the extracted files
	image_data = []
	
	# Find all image files (supporting common image formats)
	image_files = []
	image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG']
	
	for root, dirs, files in os.walk(extracted_dir):
		for file in files:
			if any(file.lower().endswith(ext) for ext in image_extensions):
				image_files.append(os.path.join(root, file))
	
	print(f"Found {len(image_files)} image files")
	
	# Prepare the image data structure
	max_images = 500  # Limit to 500 images for faster testing
	for img_path in tqdm(image_files[:max_images], desc="Processing image files"):
		file = os.path.basename(img_path)
		image_data.append({
			"path": img_path,
			"file_name": file
		})
	
	return image_data

def calculate_accuracy(original_model, model_int4, model_int8, dataloader, device, processor):
    """Calculate classification accuracy for all models and measure the drops"""
    original_correct = 0
    int4_correct = 0
    int8_correct = 0
    total_predictions = 0
    
    # Dictionaries to store per-example prediction correctness
    original_preds = {}
    int4_preds = {}
    int8_preds = {}
    
    example_counter = 0
    
    for batch in tqdm(dataloader, desc="Evaluating accuracy"):
        if batch is None:
            continue
        
        images = batch["images"]
        
        # Process images for model input
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Forward pass through all models
            original_outputs = original_model(**inputs)
            int4_outputs = model_int4(**inputs)
            int8_outputs = model_int8(**inputs)
            
            # Get predictions (highest scoring class)
            original_preds_batch = torch.argmax(original_outputs.logits, dim=1)
            int4_preds_batch = torch.argmax(int4_outputs.logits, dim=1)
            int8_preds_batch = torch.argmax(int8_outputs.logits, dim=1)
            
            # For testing without ground truth, we'll compare int4 and int8 to the original model
            # In a real scenario, you'd compare to actual labels
            for i in range(len(images)):
                example_id = example_counter
                example_counter += 1
                
                original_pred = original_preds_batch[i].item()
                int4_pred = int4_preds_batch[i].item()
                int8_pred = int8_preds_batch[i].item()
                
                # Check if quantized models match the original model's prediction
                int4_correct_pred = 1 if int4_pred == original_pred else 0
                int8_correct_pred = 1 if int8_pred == original_pred else 0
                
                # Store predictions
                original_preds[example_id] = original_pred
                int4_preds[example_id] = [int4_pred, int4_correct_pred]
                int8_preds[example_id] = [int8_pred, int8_correct_pred]
                
                # Count correct predictions (matching the original model)
                int4_correct += int4_correct_pred
                int8_correct += int8_correct_pred
                total_predictions += 1
    
    # Calculate accuracies (compared to original model)
    original_accuracy = 1.0  # By definition, original matches itself 100%
    int4_accuracy = int4_correct / total_predictions if total_predictions > 0 else 0
    int8_accuracy = int8_correct / total_predictions if total_predictions > 0 else 0
    
    # Calculate accuracy drops
    int4_drop = original_accuracy - int4_accuracy
    int8_drop = original_accuracy - int8_accuracy
    
    # Since we don't have per-example multiple predictions, standard deviation isn't applicable
    # in the same way as in the MLM task
    
    return {
        "original_accuracy": original_accuracy,
        "int4_accuracy": int4_accuracy,
        "int8_accuracy": int8_accuracy,
        "int4_drop": int4_drop,
        "int8_drop": int8_drop,
        "int4_drop_percentage": (int4_drop / original_accuracy) * 100 if original_accuracy > 0 else 0,
        "int8_drop_percentage": (int8_drop / original_accuracy) * 100 if original_accuracy > 0 else 0,
        "int4_drop_std": 0,  # Not applicable in this simplified comparison
        "int8_drop_std": 0,  # Not applicable in this simplified comparison
        "total_predictions": total_predictions
    }

def calculate_hidden_state_mse(original_model, model_int4, model_int8, dataloader, device, processor):
    """Calculate MSE and max difference between original and quantized models"""
    mse_values_int4 = {}
    mse_values_int8 = {}
    max_diff_values_int4 = {}
    max_diff_values_int8 = {}
    counter = 0
    
    for batch in tqdm(dataloader, desc="Comparing hidden states"):
        # Skip empty batches
        if batch is None:
            continue
            
        # Get images and file names
        images = batch["images"]
        file_names = batch["file_names"]
            
        # Prepare inputs
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        # Forward pass through all models
        with torch.no_grad():
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits.cpu()

            int4_outputs = model_int4(**inputs)
            int4_logits = int4_outputs.logits.cpu()

            int8_outputs = model_int8(**inputs)
            int8_logits = int8_outputs.logits.cpu()
        
        # Calculate MSE and max difference for each example in the batch
        for i in range(len(images)):
            # Calculate MSE and max difference
            original_slice = original_logits[i]
            int4_slice = int4_logits[i]
            int8_slice = int8_logits[i]

            # Calculate MSE for int4
            mse_int4 = torch.mean((original_slice - int4_slice) ** 2).item()
            max_diff_int4 = torch.max(torch.abs(original_slice - int4_slice)).item()
            
            # Calculate MSE for int8
            mse_int8 = torch.mean((original_slice - int8_slice) ** 2).item()
            max_diff_int8 = torch.max(torch.abs(original_slice - int8_slice)).item()
            
            # Store values
            mse_values_int4[counter] = mse_int4
            mse_values_int8[counter] = mse_int8
            max_diff_values_int4[counter] = max_diff_int4
            max_diff_values_int8[counter] = max_diff_int8
            counter += 1
    
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
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="Compare original, int4, and int8 quantized models")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=125, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-n', '--num_images', type=int, default=100, help='maximum number of images to process')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	# Load processor and model
	processor = AutoImageProcessor.from_pretrained(MODEL_ID)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device)
	
	# Load 4-bit quantized model
	print(f"Loading 4-bit quantized model...")
	model_int4 = AutoModelForImageClassification.from_pretrained(
		MODEL_ID,
		load_in_4bit=True,
		device_map=device
	)
	
	# Load 8-bit quantized model
	print(f"Loading 8-bit quantized model...")
	model_int8 = AutoModelForImageClassification.from_pretrained(
		MODEL_ID,
		load_in_8bit=True,
		device_map=device
	)
	
	print(f"All models loaded to {device}")
	
	# Get image files
	# Extract images from the tar file
	print(f"Processing {SPECIFIC_FILE} from {DATASET_ID}...")
	image_files = process_tar_file(DATASET_ID, SPECIFIC_FILE)

	if not image_files:
		print("No image files found. Exiting.")
		exit()

	# Limit the number of images to process if specified
	if args.num_images > 0 and args.num_images < len(image_files):
		image_files = image_files[:args.num_images]
		print(f"Limited to processing {args.num_images} images")

	# Create dataset and dataloader
	dataset = ImageDataset(image_files, processor)
	
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device.type == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True if args.num_workers > 0 else False
	)
	
	# Calculate hidden state MSE
	print("Calculating hidden state differences...")
	mse_results = calculate_hidden_state_mse(model, model_int4, model_int8, dataloader, device, processor)
	
	# Calculate accuracy drop for both quantization methods
	print("Calculating accuracy drops...")
	accuracy_results = calculate_accuracy(model, model_int4, model_int8, dataloader, device, processor)
	
	# Print results to console
	print("\nAccuracy Results:")
	print(f"Original model accuracy: {accuracy_results['original_accuracy']:.4f}")
	print(f"Int4 model accuracy: {accuracy_results['int4_accuracy']:.4f} (drop: {accuracy_results['int4_drop']:.4f}, {accuracy_results['int4_drop_percentage']:.2f}%)")
	print(f"Int8 model accuracy: {accuracy_results['int8_accuracy']:.4f} (drop: {accuracy_results['int8_drop']:.4f}, {accuracy_results['int8_drop_percentage']:.2f}%)")
	print(f"Total masked token predictions: {accuracy_results['total_predictions']}")
	
	# Save results to CSV
	csv_results(mse_results, accuracy_results)
	
	# Generate a comparison report
	if accuracy_results["int4_accuracy"] > accuracy_results["int8_accuracy"]:
		print("\nQuantization Comparison: Int4 outperforms Int8 in accuracy!")
	elif accuracy_results["int8_accuracy"] > accuracy_results["int4_accuracy"]:
		print("\nQuantization Comparison: Int8 outperforms Int4 in accuracy!")
	else:
		print("\nQuantization Comparison: Int4 and Int8 have identical accuracy.")
		
	if mse_results:
		int4_mse_mean = pd.Series(list(mse_results["mse_values_int4"].values())).mean()
		int8_mse_mean = pd.Series(list(mse_results["mse_values_int8"].values())).mean()
		
		if int4_mse_mean < int8_mse_mean:
			print("Int4 has lower mean MSE than Int8, suggesting better hidden state preservation.")
		else:
			print("Int8 has lower mean MSE than Int4, suggesting better hidden state preservation.")

	# Free cache
	torch.cuda.empty_cache()


if __name__ == "__main__":
	args = arg_commandline()
	main()
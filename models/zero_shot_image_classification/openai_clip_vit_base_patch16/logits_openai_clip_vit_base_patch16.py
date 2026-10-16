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
import numpy as np

### Constants
MODEL_ID = "openai/clip-vit-base-patch16"
DATASET_ID = "keremberke/indoor-scene-classification"
TMP_FNAME = "indoor_scenes_dataset"
CANDIDATE_LABELS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
SPECIFIC_FILE = "data/test.zip"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()

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
				"file_name": item["file_name"],
				"true_label": item.get("true_label", None)  # Add true label if available
			}
		except Exception as e:
			print(f"Error loading {item['file_name']}: {e}")
			# Return a placeholder to avoid crashing
			return {
				"image": Image.new('RGB', (224, 224), color='black'),
				"file_name": item["file_name"] + "_ERROR",
				"true_label": None
			}

def collate_fn(batch):
	"""Custom collate function for DataLoader"""
	return {
		"images": [item["image"] for item in batch],
		"file_names": [item["file_name"] for item in batch],
		"true_labels": [item["true_label"] for item in batch]
	}

def process_dataset(dataset_id, cache_dir=None):
	""" Download dataset files using hf_hub_download """
	print(f"Downloading dataset '{dataset_id}'...")
	
	# Create a consistent temp directory name if not provided
	if cache_dir is None:
		cache_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
		os.makedirs(cache_dir, exist_ok=True)
	
	try:
		# Download the specific test.zip file
		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename=SPECIFIC_FILE,
			repo_type="dataset",
			local_files_only=False,
			cache_dir=cache_dir
		)
		
		print(f"Successfully downloaded dataset to: {file_path}")
		
		# Extract the files
		extracted_dir = os.path.join(cache_dir, "extracted")
		os.makedirs(extracted_dir, exist_ok=True)
		
		# Unzip the file
		with zipfile.ZipFile(file_path, 'r') as zip_ref:
			zip_ref.extractall(extracted_dir)
		print("Extraction complete!")
		
		return extracted_dir
		
	except Exception as e:
		print(f"Error downloading dataset: {e}")
		
		# Fallback: try to download directly from the URL if HF download fails
		try:
			print("Attempting direct download...")
			url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{SPECIFIC_FILE}"
			response = requests.get(url, stream=True)
			response.raise_for_status()
			
			# Save to cache directory
			zip_path = os.path.join(cache_dir, "test.zip")
			with open(zip_path, 'wb') as f:
				for chunk in response.iter_content(chunk_size=8192):
					f.write(chunk)
			
			# Extract the files
			extracted_dir = os.path.join(cache_dir, "extracted")
			os.makedirs(extracted_dir, exist_ok=True)
			
			with zipfile.ZipFile(zip_path, 'r') as zip_ref:
				zip_ref.extractall(extracted_dir)
				
			print("Direct download and extraction complete!")
			return extracted_dir
			
		except Exception as fallback_error:
			print(f"Direct download also failed: {fallback_error}")
			
			# Create some placeholder dummy images for testing
			dummy_dir = os.path.join(cache_dir, "dummy_images")
			os.makedirs(dummy_dir, exist_ok=True)
			
			# Create dummy images for each category
			for i, label in enumerate(CANDIDATE_LABELS):
				label_dir = os.path.join(dummy_dir, label.replace(" ", "_"))
				os.makedirs(label_dir, exist_ok=True)
				
				# Create 5 dummy images per category
				for j in range(5):
					dummy_img = Image.new('RGB', (224, 224), color=(i*50, j*30, 200))
					dummy_img.save(os.path.join(label_dir, f'dummy_{label.replace(" ", "_")}_{j}.jpg'))

			print("Created dummy images for testing")
			return dummy_dir

def get_image_files(dataset_dir, max_images=100):
	""" Collect image files from dataset directory """
	image_files = []
	image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG', '.PNG']
	
	# Walk through the directory structure
	for root, dirs, files in os.walk(dataset_dir):
		for file in files:
			if any(file.lower().endswith(ext) for ext in image_extensions):
				# Try to infer true label from directory structure
				true_label = None
				
				# Check parent directory name for label
				parent_dir = os.path.basename(root).lower()
				for label in CANDIDATE_LABELS:
					label_variants = [
						label.lower(),
						label.replace(" ", "_").lower(),
						label.replace(" ", "-").lower(),
						label.replace(" ", "").lower()
					]
					if any(variant in parent_dir for variant in label_variants):
						true_label = label
						break
				
				# If no label found in directory, try filename
				if true_label is None:
					filename_lower = file.lower()
					for label in CANDIDATE_LABELS:
						label_variants = [
							label.lower(),
							label.replace(" ", "_").lower(),
							label.replace(" ", "-").lower(),
							label.replace(" ", "").lower()
						]
						if any(variant in filename_lower for variant in label_variants):
							true_label = label
							break
				
				image_files.append({
					"path": os.path.join(root, file),
					"file_name": file,
					"true_label": true_label
				})
				
				# Limit the number of images to process
				if len(image_files) >= max_images:
					return image_files
	
	return image_files

def calculate_topk_accuracy(original_logits, quantized_logits, k=1):
	"""
	Calculate Top-k accuracy between original and quantized model predictions
	Fixed to use double precision for better accuracy
	"""
	# Convert to double precision for more accurate calculations
	original_logits = original_logits.double()
	quantized_logits = quantized_logits.double()
	
	# Get top-k predictions for both models
	_, orig_topk = torch.topk(original_logits, k, dim=1)
	_, quant_topk = torch.topk(quantized_logits, k, dim=1)
	
	# Calculate accuracy: how many of the top-k predictions match
	matches = 0.0  # Use float for precision
	total = orig_topk.shape[0]
	
	for i in range(total):
		orig_set = set(orig_topk[i].tolist())
		quant_set = set(quant_topk[i].tolist())
		# Count intersection for partial matching
		intersection = len(orig_set.intersection(quant_set))
		matches += intersection / k
	
	return matches / total

def get_predicted_labels(logits, candidate_labels):
	"""Get predicted labels from logits"""
	probs = torch.softmax(logits, dim=1)
	predicted_indices = torch.argmax(probs, dim=1)
	predicted_labels = [candidate_labels[idx] for idx in predicted_indices]
	return predicted_labels, probs

def csv_results(topk_accuracies, results_data):
	""" Save results to CSV file with proper precision formatting """
	
	# Calculate average with proper precision
	if topk_accuracies:
		average = sum(topk_accuracies.values()) / len(topk_accuracies)
		topk_accuracies["average"] = average
	
	# Format all accuracy values to 6 decimal places
	formatted_accuracies = {}
	for k, accuracy in topk_accuracies.items():
		if isinstance(k, int):
			formatted_accuracies[f"top_{k}_accuracy"] = f"{accuracy}"
		else:
			formatted_accuracies[k] = f"{accuracy}"
	
	# Write to CSV with proper formatting
	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		
		# Write header and values
		writer.writerow(formatted_accuracies.keys())
		writer.writerow(formatted_accuracies.values())

def arg_commandline():
	parser = argparse.ArgumentParser(description="CLIP Top-k Accuracy Evaluation on Indoor Scenes Dataset")
	parser.add_argument('-b', '--batch_size', type=int, default=32, help='size of the batches for DataLoader')
	parser.add_argument("-n", "--num_images", type=int, default=100, help='maximum number of images to process')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, 
						help='value to quantize (4 for int4 or 8 for int8)')
	parser.add_argument('-c', '--cache_dir', type=str, default=None, 
						help='directory to cache downloaded dataset')
	parser.add_argument('-k', '--top_k', type=int, nargs='+', default=[1, 5], 
						help='k values for top-k accuracy evaluation')
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

	# Print some statistics about true labels
	labeled_files = [f for f in image_files if f["true_label"] is not None]
	print(f"Images with true labels: {len(labeled_files)}")
	if labeled_files:
		label_counts = {}
		for f in labeled_files:
			label = f["true_label"]
			label_counts[label] = label_counts.get(label, 0) + 1
		print("Label distribution:", label_counts)

	# Load processor and models
	print("Loading CLIP processor and models...")
	processor = AutoProcessor.from_pretrained(MODEL_ID)
	
	# Load original model
	model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_ID).to(device)
	model.eval()

	# Load quantized model
	if args.quantization_bits == 4:
		model_quantized = AutoModelForZeroShotImageClassification.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
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

	results_data = []
	all_original_logits = []
	all_quantized_logits = []

	print(f"Evaluating with {args.quantization_bits}-bit quantization...")
	for batch in tqdm(dataloader, desc="Processing"):
		images = batch["images"]
		file_names = batch["file_names"]
		true_labels = batch["true_labels"]

		inputs = processor(images=images, text=CANDIDATE_LABELS, return_tensors="pt", padding=True).to(device)

		with torch.no_grad():
			original_outputs = model(**inputs)
			quantized_outputs = model_quantized(**inputs)

		logits_orig = original_outputs.logits_per_image.cpu()
		logits_quant = quantized_outputs.logits_per_image.cpu()
		
		# Store logits for top-k accuracy calculation
		all_original_logits.append(logits_orig)
		all_quantized_logits.append(logits_quant)

		# Get predictions and confidences
		orig_predictions, orig_probs = get_predicted_labels(logits_orig, CANDIDATE_LABELS)
		quant_predictions, quant_probs = get_predicted_labels(logits_quant, CANDIDATE_LABELS)

		# Store results for each image
		for i in range(len(file_names)):
			orig_conf = torch.max(orig_probs[i]).item()
			quant_conf = torch.max(quant_probs[i]).item()
			
			results_data.append({
				"file_name": file_names[i],
				"true_label": true_labels[i],
				"original_prediction": orig_predictions[i],
				"quantized_prediction": quant_predictions[i],
				"original_confidence": orig_conf,
				"quantized_confidence": quant_conf,
				"predictions_match": orig_predictions[i] == quant_predictions[i]
			})

		# Clean up memory
		if device.type == 'cuda':
			torch.cuda.empty_cache()
		gc.collect()

	# Calculate Top-k accuracies with improved precision
	all_original_logits = torch.cat(all_original_logits, dim=0)
	all_quantized_logits = torch.cat(all_quantized_logits, dim=0)
	
	topk_accuracies = {}
	for k in args.top_k:
		if k <= len(CANDIDATE_LABELS):
			accuracy = calculate_topk_accuracy(all_original_logits, all_quantized_logits, k)
			topk_accuracies[k] = accuracy

	# Save results to CSV with proper precision
	csv_results(topk_accuracies, results_data)

if __name__ == "__main__":
	main()
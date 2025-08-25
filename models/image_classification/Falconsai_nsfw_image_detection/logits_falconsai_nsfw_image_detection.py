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
MODEL_ID = "Falconsai/nsfw_image_detection"
DATASET_ID = "ILSVRC/imagenet-1k"
TMP_FNAME = "imagenet_val_images"
SPECIFIC_FILE = "val_images.tar.gz"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"
HF_TOKEN = "HF_TOKEN"


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
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="Image classification model quantization comparison")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=4, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')
	parser.add_argument('-n', '--num_images', type=int, default=100, help='maximum number of images to process')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	# Load processor and model
	processor = AutoImageProcessor.from_pretrained(MODEL_ID)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device)

	if args.quantization_bits == 4:
		model_quantized = AutoModelForImageClassification.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
		model_quantized = AutoModelForImageClassification.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map=device
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")
	
	print(f"Both models loaded - Original on {device}, Quantized model ready")

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

	# Dictionary to store MSE values for each file
	all_mse_values = {}
	all_max_diff_values = {}

	# Process each batch of images
	print(f"Running inference on {len(image_files)} images with batch size {args.batch_size} on {device}")

	for batch in tqdm(dataloader, desc="Processing batches"):
		try:
			# Get batch data
			images = batch["images"]
			file_names = batch["file_names"]
			
			# Prepare inputs
			inputs = processor(images=images, return_tensors="pt").to(device)
			
			# Run inference with original model
			with torch.no_grad():
				original_outputs = model(**inputs)
				original_logits = original_outputs.logits.cpu()
			
			# Run inference with quantized model
			with torch.no_grad():
				quantized_outputs = model_quantized(**inputs)
				quantized_logits = quantized_outputs.logits.cpu()
			
			# Calculate metrics for each image in the batch
			for j, file_name in enumerate(file_names):
				# Calculate MSE and max difference
				original_slice = original_logits[j:j+1]
				quantized_slice = quantized_logits[j:j+1]
				
				mse = torch.mean((original_slice - quantized_slice) ** 2).item()
				max_diff = torch.max(torch.abs(original_slice - quantized_slice)).item()
				
				all_mse_values[file_name] = float(mse)
				all_max_diff_values[file_name] = float(max_diff)
				
		except Exception as e:
			print(f"Failed to process batch: {str(e)}")
		
		# Clean up memory
		if device.type == 'cuda':
			torch.cuda.empty_cache()
		gc.collect()

	# Save results
	csv_results(all_mse_values, all_max_diff_values)

if __name__ == "__main__":
	main()
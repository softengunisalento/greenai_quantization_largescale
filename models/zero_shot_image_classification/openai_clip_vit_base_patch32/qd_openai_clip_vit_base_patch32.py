from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import argparse
import pandas as pd
import csv
import os
import time
import tempfile
from tqdm import tqdm
from codecarbon import EmissionsTracker


### Constants
N_BYTES = 1
CARBON_INTENSITY = 475
MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_ID = "sayakpaul/indoor-scenes"
TMP_FNAME = "indoor_scenes_dataset"
CANDIDATE_LABELS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()


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

def process_dataset(dataset_id, cache_dir=None):
	""" Download dataset files using hf_hub_download """
	print(f"Downloading dataset '{dataset_id}'...")
	
	# Create a consistent temp directory name if not provided
	if cache_dir is None:
		cache_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
		os.makedirs(cache_dir, exist_ok=True)
	
	try:
		# Download the images archive - this dataset has a file called "indoor-scenes-cvpr-2019.zip"
		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename="Images.zip",  # The main image archive
			repo_type="dataset",
			local_files_only=False,
			cache_dir=cache_dir
		)
		
		print(f"Successfully downloaded dataset to: {file_path}")
		
		# Extract the files if needed (this will depend on if the download already unzipped)
		extracted_dir = os.path.join(cache_dir, "extracted")
		os.makedirs(extracted_dir, exist_ok=True)
		
		# Check if we need to unzip 
		if file_path.endswith('.zip') and not os.listdir(extracted_dir):
			import zipfile
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
			
			import zipfile
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

def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

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

		# check the memory footprint of the model
		mem_footprint = model_quantized.get_memory_footprint()

		# checking the size of the model
		module_sizes = evaluate_size(model_quantized)

		# Add energy columns to existing CSV
		try:
			# Check if file exists and has content
			file_exists = os.path.exists(CSV_FILE)
			has_content = False
			
			if file_exists:
				try:
					df = pd.read_csv(CSV_FILE)
					has_content = len(df) > 0
				except:
					has_content = False
			
			# Create new row with energy data
			energy_data = {
				'footprint_mb': mem_footprint/1e+6,
				'sizes_gb': module_sizes/1e+9,
				'inference_time_s': inference_time,
				'cpu_wh': cpu_energy_Wh,
				'gpu_wh': gpu_energy_Wh,
				'ram_wh': ram_energy_Wh,
				'total_energy_wh': total_energy_Wh,
				'co2_emissions_gr': co2_emissions
			}
			
			if has_content:
				# Read existing data and append new row
				df = pd.read_csv(CSV_FILE)
				
				# Add missing columns with NaN if they don't exist
				# for col in energy_data.keys():
				# 	if col not in df.columns:
				# 		df[col] = pd.NA
				
				# Create new row - copy last row and update with energy data
				new_row = df.iloc[-1].copy() if len(df) > 0 else pd.Series()
				for key, value in energy_data.items():
					new_row[key] = value
				
				# Append new row
				df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
			else:
				# Create new DataFrame with energy data
				df = pd.DataFrame([energy_data])
			
			# Write back to CSV
			df.to_csv(CSV_FILE, index=False)
			
		except Exception as e:
			print(f"Error adding energy data: {e}")
			# Fallback to simple append
			with open(CSV_FILE, mode='a', newline='') as file:
				writer = csv.writer(file)
				if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
					writer.writerow(["footprint_mb", "sizes_gb", "inference_time_s", "cpu_wh", "gpu_wh", "ram_wh", "total_energy_wh", "co2_emissions_gr"])
				writer.writerow([inference_time, cpu_energy_Wh, gpu_energy_Wh, ram_energy_Wh, total_energy_Wh, co2_emissions])

		return result
	return wrapper

# def csv_results(model, predictions):
# 	""" Print statistics """
# 	# check the memory footprint of the model
# 	mem_footprint = model.get_memory_footprint()

# 	# checking the size of the model
# 	module_sizes = evaluate_size(model)
# 	csv_exists = os.path.exists(CSV_FILE)
# 	with open(CSV_FILE, mode='a', newline='') as file:
# 		writer = csv.writer(file)
		
# 		# write files stats
# 		# if not csv_exists:
# 		# 	writer.writerow(["File Name", "Prediction", "Confidence Score"])

# 		# for img_path, result in predictions.items():
# 		# 	writer.writerow([img_path, result["label"], result["score"]])
		
# 		# write metrics
# 		if not csv_exists:
# 			writer.writerow(["footprint_mb", "sizes_gb"])
# 		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

# 	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="Room classification with energy consumption measurements")
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')
	parser.add_argument('-n', '--num_images', type=int, default=100, help='maximum number of images to process')

	return parser.parse_args()


def main():
	args = arg_commandline()
	global model_quantized

	# Load the CLIP model and processor
	processor = CLIPProcessor.from_pretrained(MODEL_ID)
	
	# Move to GPU if available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if args.quantization_bits == 4:
		model_quantized = CLIPModel.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
		model_quantized = CLIPModel.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map=device
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")

	# Download and extract the dataset
	dataset_dir = process_dataset(DATASET_ID)
	print(f"Working with dataset directory: {dataset_dir}")

	# Collect image files (limited to num_images)
	image_files = get_image_files(dataset_dir, args.num_images)
	
	print(f"Found {len(image_files)} image files")
	
	if len(image_files) == 0:
		print("No images found! Check if the dataset contains image files.")
		return
	
	# Create dataset and dataloader
	dataset = ImageDataset(image_files, processor)
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device == 'cuda' else False,
		prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
		persistent_workers=True if args.num_workers > 0 else False
	)

	# Dictionary to store predictions
	predictions = {}

	# Format the candidate labels for CLIP
	text_inputs = processor(
		text=CANDIDATE_LABELS, 
		padding=True, 
		return_tensors="pt"
	).to(device)

	# Process each batch of images
	for batch_idx, batch in enumerate(tqdm(dataloader, desc="Classifying rooms")):
		try:
			# Process the images
			image_inputs = processor(
				images=batch["images"],
				padding=True,
				return_tensors="pt"
			).to(device)

			# Run inference with CLIP
			with torch.no_grad():
				outputs = model_quantized(**image_inputs, **text_inputs)
				
				# Get similarity scores
				logits_per_image = outputs.logits_per_image
				probs = logits_per_image.softmax(dim=1)
				
				# Get the predicted class for each image
				for i, (file_name, prob) in enumerate(zip(batch["file_names"], probs)):
					# Get the most likely class
					max_idx = prob.argmax().item()
					confidence = prob[max_idx].item()
					predicted_label = CANDIDATE_LABELS[max_idx]
					
					predictions[file_name] = {
						"label": predicted_label,
						"score": confidence
					}

		except Exception as e:
			print(f"Failed to process batch {batch_idx}: {str(e)}")
	
	# Save the results to a CSV file


if __name__ == "__main__":
	args = arg_commandline()
	if args.mgpu:
		# If -m is provided, wrap main with energy measurement
		main_with_timing = measure_energy_consumption(main)
		main_with_timing()
	else:
		# Otherwise, just call main directly
		main()
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import torch
import argparse
import pandas as pd
import csv
import os
import time
from tqdm import tqdm
from codecarbon import EmissionsTracker

### Constants
N_BYTES = 1  # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475  # Global average carbon intensity
MODEL_ID = "Falconsai/nsfw_image_detection"
DATASET_ID = "ILSVRC/imagenet-1k"
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

		# Process the already-loaded image
		try:
			image = item["image"].convert("RGB")
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

def load_validation_images(dataset_id, max_images=500):
	""" Stream the validation split (imagenet-1k is parquet-backed on the Hub,
	not a raw tar.gz - streaming avoids materializing the full dataset) """
	ds = load_dataset(dataset_id, split="validation", streaming=True)

	image_data = []
	for i, example in enumerate(tqdm(ds, total=max_images, desc="Streaming validation images")):
		if i >= max_images:
			break
		image_data.append({
			"image": example["image"],
			"file_name": f"val_{i:05d}"
		})

	print(f"Found {len(image_data)} image files")
	return image_data

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
# 		# 	writer.writerow(["File Name", "Prediction"])

# 		# for img_path in predictions:
# 		# 	writer.writerow([img_path, predictions[img_path]])
		
# 		# write metrics
# 		if not csv_exists:
# 			writer.writerow(["footprint_mb", "sizes_gb"])
# 		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

# 	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=0, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	global model_quantized

	# Load the image processor, the model, and get images from the dataset
	processor = AutoImageProcessor.from_pretrained(MODEL_ID)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"device used: {device}")

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

	# Extract images from the tar file
	print(f"Processing validation split from {DATASET_ID}...")
	image_files = load_validation_images(DATASET_ID)

	if not image_files:
		print("No image files found. Exiting.")
		exit()
	
	# Create dataset and dataloader
	dataset = ImageDataset(
		image_files, 
		processor
	)
	
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

	# Dictionary to store MSE values for each file
	predictions = {}

	# Process each image
	print(f"Running inference on {len(image_files)} images")
	for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing images")):
		try:
			# Open and process the image
			inputs = processor(
				images=batch["images"],
				return_tensors="pt"
			).to(device)

			# Inference:
			with torch.no_grad():  # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
				outputs = model_quantized(**inputs)

			# Get the predicted class
			predicted_classes = outputs.logits.argmax(dim=-1).tolist()


			# Get the class names
			class_names = model_quantized.config.id2label
			for file_name, pred_idx in zip(batch["file_names"], predicted_classes):
				predictions[file_name] = class_names[pred_idx]

		except Exception as e:
			print(f"Failed to process image {batch_idx}: {str(e)}")

	# Save results



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

from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import argparse
import csv
import os
import time
import tarfile
import tempfile
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm
from codecarbon import EmissionsTracker

### Constants
N_BYTES = 4  # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475  # Global average carbon intensity
MODEL_ID = "microsoft/resnet-152"
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
		totale_energy_Wh = tracker._total_energy.kWh * 1000

		with open(CSV_FILE, mode='a', newline='') as file:
			writer = csv.writer(file)
			
			# write stats
			writer.writerow(["Inference s", "CPU Wh", "GPU Wh", "RAM Wh", "Energy Wh", "CO2 Emissions gr"])
			writer.writerow([inference_time,cpu_energy_Wh, gpu_energy_Wh, ram_energy_Wh, totale_energy_Wh, co2_emissions])

		return result
	return wrapper

def csv_results(model, predictions):
	""" Print statistics """
	# check the memory footprint of the model
	mem_footprint = model.get_memory_footprint()

	# checking the size of the model
	module_sizes = evaluate_size(model)

	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		
		# write files stats
		writer.writerow(["File Name", "Prediction"])

		for img_path in predictions:
			writer.writerow([img_path, predictions[img_path]])
		
		# write metrics
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

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	# Load the image processor, the model, and get images from the dataset
	processor = AutoImageProcessor.from_pretrained(MODEL_ID)
	model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
	
	# Move to GPU if not quantized
	device = 'cuda' if torch.cuda.is_available() and not any('quantized' in name for name, _ in model.named_modules()) else 'cpu'
	model = model.to(device)

	# Extract images from the tar file
	print(f"Processing {SPECIFIC_FILE} from {DATASET_ID}...")
	image_files = process_tar_file(DATASET_ID, SPECIFIC_FILE)

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
		prefetch_factor=args.prefetch_factor,
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
				outputs = model(**inputs)

			# Get the predicted class
			predicted_classes = outputs.logits.argmax(dim=-1).tolist()


			# Get the class names
			class_names = model.config.id2label
			for file_name, pred_idx in zip(batch["file_names"], predicted_classes):
				predictions[file_name] = class_names[pred_idx]

		except Exception as e:
			print(f"Failed to process image {batch_idx}: {str(e)}")

	# Save results
	csv_results(model, predictions)


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

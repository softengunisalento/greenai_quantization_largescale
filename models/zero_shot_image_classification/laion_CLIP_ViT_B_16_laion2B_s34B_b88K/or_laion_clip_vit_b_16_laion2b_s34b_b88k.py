import open_clip
import torch
from PIL import Image
import argparse
import os
import time

### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
IMG_PATH = "./dog.jpg"
CANDIDATE_LABELS = ["cat", "dog", "car", "person"]
DEVICE = "cpu"

### Functions
def evaluate_footprint(model):
	""" Evaluate the footprint of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def get_embeddings(sentences, tokenizer, model):
	"""Function to get sentence embeddings"""
	encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
	# Inference:
	with torch.no_grad(): # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
		model_output = model(**encoded_input)
		# Perform pooling. In this case, cls pooling.
		embeddings = model_output[0][:, 0]
	embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
	return embeddings

def get_gpu_power():
	""" Get the power usage of the GPU """
	power_usage = os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits').read().strip()
	return float(power_usage)

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
		# measure the power consumption before and after inference
		power_before = get_gpu_power()
		start_time = time.time()

		# Call the function
		result = func(*args, **kwargs)

		# measure the power consumption after inference
		end_time = time.time()
		inference_time = end_time - start_time
		print(f"> Original Inference Time: {inference_time} seconds")

		power_after = get_gpu_power()
		average_power_usage = (power_after + power_before) / 2
		print(f"> Original Average Power Consumption: {average_power_usage} Watts")

		energy_joules = calculate_energy_consumption(average_power_usage, inference_time)
		print(f"> Original Energy Consumption: {energy_joules} Joules")

		co2_emissions = calculate_co2_emissions(energy_joules)
		print(f"> Original CO2 Emissions: {co2_emissions} grams")
		return result
	return wrapper

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	arg_commandline()

	# Load the model, the tokenizer and the image
	model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
	tokenizer = open_clip.get_tokenizer(MODEL_ID)
	image = Image.open(IMG_PATH)

	# Preprocess the image
	image_tensor = preprocess_val(image).unsqueeze(0).to(DEVICE)

	# Tokenize the text prompt
	text_tensor = tokenizer(CANDIDATE_LABELS)

	# Move tensors to the same device as the model
	model.to(DEVICE)
	text_tensor = text_tensor.to(DEVICE)

	# Inference:
	with torch.no_grad(): # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
		image_features = model.encode_image(image_tensor)
		text_features = model.encode_text(text_tensor)

	# Normalize the features
	image_features /= image_features.norm(dim=-1, keepdim=True)
	print(f"> Original image features: {image_features}")
	text_features /= text_features.norm(dim=-1, keepdim=True)
	print(f"> Original text features: {text_features}")

	# Calculate similarity for each text prompt with matrix multiplication
	similarity = (image_features @ text_features.T)
	print(f"> Original similarity scores: {similarity}")

	#Get the similarity score
	result_similarity = similarity[0,-1].item()
	print(f"> Original image classification: {result_similarity}")

	# check the memory footprint of the model
	mem_footprint = evaluate_footprint(model)
	print(f"> Original Footprint: {mem_footprint/1e+6} MBs")

	# checking the size of the model
	module_sizes = evaluate_size(model)
	print(f"> Original model size is {module_sizes/1e+9} GB")


if __name__ == "__main__":
	# Check if the -m option is provided
	if arg_commandline().mgpu:
		# If -m is provided, wrap main with energy measurement
		main_with_timing = measure_energy_consumption(main)
		main_with_timing()
	else:
		# Otherwise, just call main directly
		main()
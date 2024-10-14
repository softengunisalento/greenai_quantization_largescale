from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import argparse
import os
import time

### Constants
N_BYTES = 1 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
DATASET_ID = "hf-internal-testing/librispeech_asr_demo"
DATASET_CONFIG = "clean"
DATA_TYPE = "audio"

### Functions
def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

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
		print(f"> Dynamically quantized Inference Time: {inference_time} seconds")

		power_after = get_gpu_power()
		average_power_usage = (power_after + power_before) / 2
		print(f"> Dynamically quantized Average Power Consumption: {average_power_usage} Watts")

		energy_joules = calculate_energy_consumption(average_power_usage, inference_time)
		print(f"> Dynamically quantized Energy Consumption: {energy_joules} Joules")

		co2_emissions = calculate_co2_emissions(energy_joules)
		print(f"> Dynamically quantized CO2 Emissions: {co2_emissions} grams")
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

	# loading the dataset
	dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", trust_remote_code=True)
	dataset = dataset.sort("id")
	sampling_rate = dataset.features[DATA_TYPE].sampling_rate

	# loading the Feature Extractor and Model
	feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
	model = ASTForAudioClassification.from_pretrained(MODEL_ID)

	# Apply dynamic quantization to the model
	model_quantized = torch.quantization.quantize_dynamic(
		model,  # the model instance
		{torch.nn.Linear},  # specify the layers to quantize
		dtype=torch.qint8  # quantized dtype (8-bit int)
	)

	# audio file is decoded on the fly
	inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

	# Inference:
	with torch.no_grad(): # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
		logits = model_quantized(**inputs).logits

	# get the predicted class and label
	predicted_class_ids = torch.argmax(logits, dim=-1).item()
	predicted_label = model_quantized.config.id2label[predicted_class_ids]
	print(f"> Dynamically quantized prediction label: {predicted_label}")

	# compute loss for the target label
	target_label = model_quantized.config.id2label[0]
	inputs["labels"] = torch.tensor([model_quantized.config.label2id[target_label]])
	loss = model_quantized(**inputs).loss
	print(f"> Dynamically quantized loss: {loss.item()}")

	# check the memory footprint of the model
	mem_footprint = model_quantized.get_memory_footprint()
	print(f"> Dynamically quantized Footprint: {mem_footprint/1e+6} MBs")

	# checking the size of the model
	module_sizes = evaluate_size(model_quantized)
	print(f"> Dynamically quantized model size is {module_sizes/1e+9} GB")


if __name__ == "__main__":
	# Check if the -m option is provided
	if arg_commandline().mgpu:
		# If -m is provided, wrap main with energy measurement
		main_with_timing = measure_energy_consumption(main)
		main_with_timing()
	else:
		# Otherwise, just call main directly
		main()
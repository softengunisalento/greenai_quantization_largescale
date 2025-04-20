from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import argparse
import os
import time

### Constants
N_BYTES = 1 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
INPUT_SENTENCE = ["That is a happy person"]
SENTENCES_TO_COMPARE = [
	"That is a happy dog",
	"That is a very happy person",
	"Today is a sunny day"
] # list of sentences to compare with the input sentence

### Functions
def verbose_measurements(model_quantized):
	""" Function to print verbose output """
	# check the memory footprint of the model
	mem_footprint = model_quantized.get_memory_footprint()
	print(f"> Dynamically quantized Footprint: {mem_footprint/1e+6} MBs")

	# checking the size of the model
	module_sizes = evaluate_size(model_quantized)
	print(f"> Dynamically quantized model size is: {module_sizes/1e+9} GB")

def get_gpu_power():
	""" Get the power usage of the GPU """
	power_usage = os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits').read().strip()
	return power_usage

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
	parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose of basic measurement')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	# Load the tokenizer and model from sentence-transformers
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID)

	# Apply dynamic quantization to the model
	model_quantized = torch.quantization.quantize_dynamic(
		model,  # the model instance
		{torch.nn.Linear},  # specify the layers to quantize
		dtype=torch.qint8  # quantized dtype (8-bit int)
	)

	# Get embeddings for source and comparison sentences
	source_embedding = get_embeddings(INPUT_SENTENCE, tokenizer, model_quantized)
	comparison_embeddings = get_embeddings(SENTENCES_TO_COMPARE, tokenizer, model_quantized)

	# Calculate cosine similarities
	cosine_similarities = cosine_similarity(source_embedding, comparison_embeddings)
	for sentence, similarity in zip(SENTENCES_TO_COMPARE, cosine_similarities[0]):
		print(f"> Dynamically quantized Similarity between '{INPUT_SENTENCE}' and '{sentence}': {similarity:.4f}")

	if args.verbose:
		# If -v is provided, call main with verbose output
		verbose_measurements(model_quantized)


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
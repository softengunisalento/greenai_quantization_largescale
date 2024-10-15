from huggingface_hub import hf_hub_download
import fasttext
import psutil
import torch
import nltk
import numpy as np
import argparse
import os
import time

### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "facebook/fasttext-language-identification"
INPUT_TEXT = "ciao sono matteo" # Text to generate
QUANTO_WEIGHTS = "int8"
TOKEN_LANGUAGE = "english"
MODEL_FILENAME = "model.bin"

### Functions
def verbose_measurements(model):
	""" Function to print verbose output """
	...

def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def named_module_tensors(module):
	"""Function to yield `_data` and `._scale` attributes"""
	for named_parameter in module.named_parameters(recurse=False):
		name, val = named_parameter
		if hasattr(val,"_data") or hasattr(val,"_scale"):
			if hasattr(val,"_data"):
				yield name + "._data", val._data
			if hasattr(val,"_scale"):
				yield name + "._scale", val._scale
		else:
			yield named_parameter

	for named_buffer in module.named_buffers(recurse=False):
		yield named_buffer

def dtype_byte_size(dtype):
	"""Function to return the size of a dtype"""
	if dtype == torch.float32:
		return 4
	elif dtype == torch.float64:
		return 8
	elif dtype == torch.int8:
		return 1
	elif dtype == torch.int16:
		return 2
	elif dtype == torch.int32:
		return 4
	elif dtype == torch.int64:
		return 8
	elif dtype == torch.bool:
		return 1
	else:
		raise ValueError(f"dtype is not a valid dtype: {dtype}.")

def compute_module_sizes(model):
	"""Compute the size of each module in the model. For fastText models, we need a different approach to get the size as it doesn't have 'named_parameters'"""
	total_size = 0

	# Get the size of the vocabulary (embedding matrix)
	vocab_size = len(model.get_words())
	embedding_dim = model.get_dimension()
	total_size += vocab_size * embedding_dim * np.dtype(np.float32).itemsize # Assuming float32 embeddings

	return {'': total_size}

def get_memory_usage():
	"""Function to get memory usage"""
	process = psutil.Process()
	mem = process.memory_info().rss / float(2 ** 20)	# in MB
	return mem

def predict_language(text, config, model):
	"""Define a function to tokenize and predict language"""
	tokens = nltk.word_tokenize(text, language=config.tokenization_language)
	tokenized_text = ' '.join(tokens)
	predictions = model.predict(tokenized_text, k=5)	# k is the number of top predictions to return
	languages = [label.replace('__label__', '') for label in predictions[0]]
	confidences = predictions[1]
	return list(zip(languages, confidences))

### Classes
class Config:
	"""Define configuration directly in the script"""
	fasttext_model_path = None
	tokenization_language = TOKEN_LANGUAGE
	quantization_weights = QUANTO_WEIGHTS

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
	parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose of basic measurement')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	config = Config()

	# Download the pre-trained model if not already downloaded
	if not config.fasttext_model_path:
		config.fasttext_model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILENAME)

	# Load the FastText model
	model = fasttext.load_model(config.fasttext_model_path)

	# Apply quantization if specified
	if config.quantization_weights == QUANTO_WEIGHTS:
		model.quantize(input=None, retrain=False, cutoff=0, dsub=2)

	# Download the NLTK tokenizer data
	nltk.download('punkt')

	# Test the function with a sample text
	predictions = predict_language(INPUT_TEXT, config, model)
	print("> Dynamic quantized Predictions:")
	print(predictions)

	# Print memory usage after loading model
	print(f"> Dynamic quantized Footprint: {get_memory_usage()} MBs")

	# check the model size
	module_sizes = compute_module_sizes(model)
	print(f"> The dynamic quantized model size is: {module_sizes[''] * 1e-9} GB")

	if args.verbose:
		# If -v is provided, call main with verbose output
		verbose_measurements(model)


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
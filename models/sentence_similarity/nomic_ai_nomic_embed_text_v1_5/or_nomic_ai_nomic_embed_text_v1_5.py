import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import argparse
import os
import time

### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DOCUMENTS = [
	"Artificial intelligence is transforming the world.",
	"Natural language processing enables machines to understand human language.",
	"Quantum computing is the future of technology."
] # list of documents to search for
SEARCH_QUERIES = [
	"How is AI changing the world?",
	"Future of quantum computing?"
] # list of search queries to find in the documents

### Functions
def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def mean_pooling(model_output, attention_mask):
	"""Perform mean pooling on the model output"""
	token_embeddings = model_output[0]
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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

	# Load tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, safe_serialization=True)
	model.eval()

	# Tokenize input documents and queries
	encoded_documents = tokenizer(DOCUMENTS, padding=True, truncation=True, return_tensors='pt')
	encoded_queries = tokenizer(SEARCH_QUERIES, padding=True, truncation=True, return_tensors='pt')

	# Inference:
	with torch.no_grad(): # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
		model_output_docs = model(**encoded_documents)
		model_output_queries = model(**encoded_queries)

	# Perform mean pooling of the model
	document_embeddings = mean_pooling(model_output_docs, encoded_documents['attention_mask'])
	query_embeddings = mean_pooling(model_output_queries, encoded_queries['attention_mask'])

	# Perform normalization of document and query embeddings
	document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
	query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

	# Calculate cosine similarity between query and document embeddings
	similarities = cosine_similarity(query_embeddings.numpy(), document_embeddings.numpy())

	# Print results
	for i, query in enumerate(SEARCH_QUERIES):
		print(f"> Original Query: {query}")
		similarity_scores = similarities[i]
		for j, score in enumerate(similarity_scores):
			print(f"	Document {j+1}: {DOCUMENTS[j]} (Score: {score:.4f})")

	# check the memory footprint of the model
	mem_footprint = model.get_memory_footprint()
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
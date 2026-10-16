from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import argparse
import pandas as pd
import os
import csv
import time
import tempfile
import pandas as pd
from codecarbon import EmissionsTracker


### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "sentence-transformers/all-MiniLM-L12-v2"
DATASET_ID = "mandarjoshi/trivia_qa"
TMP_FNAME = "mandarjoshi_trivia_qa"
SPECIFIC_FILE = "rc/test-00000-of-00004.parquet"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()


### Classes
class TriviaDataset(Dataset):
	"""Dataset for sentence embedding from TriviaQA"""
	
	def __init__(self, dataframe, tokenizer, max_length=128):
		self.tokenizer = tokenizer
		self.texts = []
		
		# Extract relevant columns based on the structure of TriviaQA
		if 'question' in dataframe.columns:
			self.texts = dataframe['question'].tolist()
		elif 'text' in dataframe.columns:
			self.texts = dataframe['text'].tolist()
		else:
			# Try to find the first text column
			text_columns = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
			if text_columns:
				self.texts = dataframe[text_columns[0]].tolist()
				
		self.max_length = max_length
		
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = self.texts[idx]
		
		# Skip if text is too short
		if len(str(text).split()) < 5:
			# Return a dummy sample that will be filtered later
			return None
			
		# Tokenize the text
		inputs = self.tokenizer(
			str(text), 
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
			padding="max_length"
		)
		
		return {
			"input_ids": inputs["input_ids"][0],
			"attention_mask": inputs["attention_mask"][0],
			"text": str(text)
		}


### Functions
def collate_fn(batch):
	"""Custom collate function to handle None values"""
	# Filter out None values
	batch = [item for item in batch if item is not None]
	
	if not batch:
		return None
	
	# Prepare lists for batched data
	input_ids = []
	attention_masks = []
	texts = []
	
	for item in batch:
		input_ids.append(item["input_ids"])
		attention_masks.append(item["attention_mask"])
		texts.append(item["text"])
	
	return {
		"input_ids": torch.stack(input_ids),
		"attention_mask": torch.stack(attention_masks),
		"texts": texts
	}

def load_parquet_file(dataset_id, file_name):
	""" Function to extract and process the specific parquet file """
	# Create a consistent temp directory name
	temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
	file_path = None
	
	# Check if the temp directory already exists and has content
	if os.path.exists(temp_dir) and os.listdir(temp_dir):
		print(f"Found existing temporary directory: {temp_dir}")
		# Try to find the file in the existing directory
		for root, dirs, files in os.walk(temp_dir):
			for file in files:
				if file == file_name.split("/")[-1]:
					file_path = os.path.join(root, file)
					print(f"Found existing file: {file_path}")
					break
			if file_path:
				break
	
	# If no file found, we need to download
	if file_path is None:
		# Download the specific file with progress
		from huggingface_hub import hf_hub_download

		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename=file_name,
			repo_type="dataset",
			local_files_only=False,
			resume_download=True
		)
		
		# Create the temp directory if it doesn't exist
		os.makedirs(temp_dir, exist_ok=True)
		print(f"Downloaded file to: {file_path}")
	
	if not file_path:
		print(f"Could not find {file_name} in the directory structure")
		exit()
	
	# Load the parquet data
	try:
		df = pd.read_parquet(file_path)
		print(f"Loaded {len(df)} records from {file_name}")
		return df
	except Exception as e:
		print(f"Error loading {file_name}: {e}")
		return None

def evaluate_size(model):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())

	# return bytes
	return total_params * N_BYTES

def calculate_energy_consumption(power_watts, time_seconds):
	""" Calculate energy consumption in Joules """
	return power_watts * time_seconds

def calculate_co2_emissions(energy):
	""" Calculate CO2 emissions (in grams) """
	# Convert joules to megajoules (1 MJ = 1e6 J)
	energy_megajoules = energy / 1e6
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
		mem_footprint = model.get_memory_footprint()

		# checking the size of the model
		module_sizes = evaluate_size(model)

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

# def csv_results(model, results_list):
# 	""" Save sentence embedding similarity results to CSV """
# 	# check the memory footprint of the model
# 	mem_footprint = model.get_memory_footprint()
	
# 	# checking the size of the model
# 	module_sizes = evaluate_size(model)
# 	csv_exists = os.path.exists(CSV_FILE)
# 	with open(CSV_FILE, mode='a', newline='') as file:
# 		writer = csv.writer(file)

# 		# Write headers for similarity results
# 		# if not csv_exists:
# 		# 	writer.writerow(["Query Text", "Compared Text", "Similarity Score"])
		
# 		# # Write each similarity result
# 		# for result in results_list:
# 		# 	writer.writerow([
# 		# 		result["query_text"], 
# 		# 		result["compared_text"], 
# 		# 		result["similarity"]
# 		# 	])
		
# 		# # Write model metrics
# 		# if not csv_exists:
# 		# 	writer.writerow([""])  # Empty row for separation
# 		if not csv_exists:
# 			writer.writerow(["footprint_mb", "sizes_gb"])
# 		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])
	
# 	print(f"Results saved to {CSV_FILE}")

def process_batch_embeddings(batch, model, device):
	"""Process a batch to compute embeddings"""
	# Move tensors to device
	input_ids = batch["input_ids"].to(device)
	attention_mask = batch["attention_mask"].to(device)
	
	# Forward pass without gradients
	with torch.no_grad():
		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		# Get embeddings from CLS token
		embeddings = outputs[0][:, 0]
		# Normalize embeddings
		embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
	
	return embeddings

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')
	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-n', '--num_samples', type=int, default=100, help='number of samples to use for similarity comparison')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()
	global tokenizer, model
	
	# Load the tokenizer and model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	model = AutoModel.from_pretrained(MODEL_ID)
	
	# Move to GPU if available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f"Using device: {device}")
	model = model.to(device)
	
	# Load dataset
	print(f"Processing {SPECIFIC_FILE}...")
	df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)
	
	if df is None or df.empty:
		print(f"No data found in {SPECIFIC_FILE}")
		return
	
	# Create dataset and dataloader
	dataset = TriviaDataset(df, tokenizer)
	dataloader = DataLoader(
		dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device == 'cuda' else False,
		prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
		persistent_workers=True if args.num_workers > 0 else False
	)
	
	results = []
	comparison_count = 0
	query_texts = []
	query_embeddings = []
	
	print(f"Extracting query texts for embedding comparisons...")
	
	# First, collect a set of query texts to use as reference points
	num_queries = min(10, args.num_samples // 10)
	
	# Process batches to extract query texts
	for batch in tqdm(dataloader, desc="Collecting query texts", total=num_queries):
		if batch is None or len(query_texts) >= num_queries:
			continue
			
		# Add the first text from this batch as a query
		query_texts.append(batch["texts"][0])
		
		# Break once we have enough queries
		if len(query_texts) >= num_queries:
			break
	
	print(f"Computing embeddings for {len(query_texts)} query texts...")
	
	# Compute embeddings for all query texts at once
	model.eval()
	for query_text in query_texts:
		# Tokenize the query text
		encoded_input = tokenizer([query_text], padding=True, truncation=True, return_tensors='pt')
		encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
		
		# Compute embedding
		with torch.no_grad():
			outputs = model(**encoded_input)
			embedding = outputs[0][:, 0]  # CLS token embedding
			# Normalize embedding
			embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
			query_embeddings.append(embedding)
	
	print(f"Running embedding similarity comparisons...")
	
	# Now process batches for comparison
	current_query_idx = 0
	for batch in tqdm(dataloader, desc="Computing similarities"):
		# Skip empty batches
		if batch is None or comparison_count >= args.num_samples:
			continue
			
		# Get embeddings for this batch
		batch_embeddings = process_batch_embeddings(batch, model, device)
		texts = batch["texts"]
		
		# Get current query embedding
		query_embedding = query_embeddings[current_query_idx]
		query_text = query_texts[current_query_idx]
		
		# Compute similarities with all texts in this batch
		similarities = cosine_similarity(
			query_embedding.cpu().numpy(), 
			batch_embeddings.cpu().numpy()
		)[0]
		
		# Add all similarities to results
		for text, similarity in zip(texts, similarities):
			if comparison_count < args.num_samples and text != query_text:
				results.append({
					"query_text": query_text,
					"compared_text": text,
					"similarity": float(similarity)
				})
				comparison_count += 1
		
		# Move to next query when we have enough comparisons for this one
		if comparison_count % (args.num_samples // num_queries) == 0 and current_query_idx < len(query_texts) - 1:
			current_query_idx += 1
	
	# Save all results to CSV


	# Free cache
	torch.cuda.empty_cache()


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
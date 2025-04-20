from transformers import AutoTokenizer, AutoModel
import torch
import os
import tempfile
from tqdm import tqdm
import pandas as pd
import csv
import random

### Constants
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "legacy-datasets/wikipedia"
TMP_FNAME = "legacy_datasets_wikipedia"
SPECIFIC_FILE = "train-00009-of-00041.parquet"
FILE_NAME = "distilbert_distilbert_base_uncased"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv" + FILE_NAME + ".csv"
MASK_PROB = 0.15  # Probability to mask a token

### Functions
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
				if file == file_name:
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
			filename=f"data/20220301.en/{file_name}",
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

def preprocess_for_fill_mask(texts, tokenizer, max_length=128):
	"""
	Preprocess text data for fill-mask task by randomly masking tokens
	"""
	processed_examples = []
	
	for text in tqdm(texts, desc="Preprocessing for fill-mask"):
		# Skip if text is too short
		if len(text.split()) < 5:
			continue
			
		# Tokenize the text
		inputs = tokenizer(
			text, 
			truncation=True,
			max_length=max_length,
			return_tensors="pt"
		)
		
		input_ids = inputs["input_ids"][0].tolist()
		
		# Don't mask special tokens
		special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
		
		# Get positions of tokens that can be masked (not special tokens)
		maskable_positions = [i for i, token_id in enumerate(input_ids) 
							 if token_id not in special_tokens]
		
		# Skip if not enough maskable tokens
		if len(maskable_positions) < 3:
			continue
			
		# Randomly select positions to mask based on probability
		num_to_mask = max(1, int(len(maskable_positions) * MASK_PROB))
		positions_to_mask = random.sample(maskable_positions, num_to_mask)
		
		# Create masked version and labels
		masked_input_ids = input_ids.copy()
		labels = [-100] * len(input_ids)  # -100 means don't compute loss for this position
		
		for pos in positions_to_mask:
			# Record the original token for the label
			labels[pos] = input_ids[pos]
			# Replace with [MASK] token
			masked_input_ids[pos] = tokenizer.mask_token_id
		
		# Store the example
		processed_examples.append({
			"input_ids": masked_input_ids,
			"labels": labels,
			"attention_mask": inputs["attention_mask"][0].tolist()
		})
	
	return processed_examples

def csv_results():
	""" Print MSE statistics """
	mse_series = pd.Series(list(mse_values.values()))

	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Record index", "MSE", "Max Difference"])
		
		for idx in mse_values:
			writer.writerow([idx, mse_values[idx], max_diff_values[idx]])
		
		writer.writerow(["Min MSE", "Max MSE", "Mean MSE", "Median MSE", "Standard Deviation"])
		writer.writerow([mse_series.min(), mse_series.max(), mse_series.mean(), mse_series.median(), mse_series.std()])

	print(f"Results saved to {CSV_FILE}")


# Load the tokenizer and model from sentence-transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
	model,  # the model instance
	{torch.nn.Linear},  # specify the layers to quantize
	dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Load dataset
print(f"Processing {SPECIFIC_FILE}...")
df = load_parquet_file(DATASET_ID, SPECIFIC_FILE)

if df is None or len(df) == 0:
	print(f"No data found in {SPECIFIC_FILE}")
	exit()
	
# Extract text from the dataframe (the main content column is 'text')
texts = df['text'].tolist()

# Preprocess texts for fill-mask task
processed_examples = preprocess_for_fill_mask(texts, tokenizer)

# Dictionary to store MSE values for each record
mse_values = {}
max_diff_values = {}

print(f"Running fill-mask inference on {len(processed_examples)} processed examples...")
for idx, example in enumerate(tqdm(processed_examples, desc="Fill-mask inference")):
	# Convert to tensors
	input_ids = torch.tensor([example["input_ids"]])
	attention_mask = torch.tensor([example["attention_mask"]])
	
	# Get positions of mask tokens
	mask_positions = [i for i, token_id in enumerate(example["input_ids"]) 
						if token_id == tokenizer.mask_token_id]
	
	# Skip if no masks found (shouldn't happen based on preprocessing)
	if not mask_positions:
		continue
	
	# Forward pass through the model
	with torch.no_grad():
		original_logits = model(input_ids=input_ids, attention_mask=attention_mask)

	# Extract last hidden state from the original outputs
	original_hidden_states = original_logits.last_hidden_state
	
	# Forward pass through the model
	with torch.no_grad():
		quantized_logits = model_quantized(input_ids=input_ids, attention_mask=attention_mask)

	# Extract last hidden state from the quantized outputs
	quantized_hidden_states = quantized_logits.last_hidden_state
	
	# Calculate the mean squared error (MSE) between the logits
	mse = torch.mean((original_hidden_states - quantized_hidden_states) ** 2).item()
	
	# Calculate the maximum absolute difference
	max_diff = torch.max(torch.abs(original_hidden_states - quantized_hidden_states)).item()
	
	# Store values for later analysis
	mse_values[idx] = mse
	max_diff_values[idx] = max_diff

csv_results()
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import csv
import torch
import tarfile
import librosa
import argparse
import tempfile
import pandas as pd
import gc
import numpy as np


### Constants
MODEL_ID = "openai/whisper-small"
DATASET_ID = "MLCommons/unsupervised_peoples_speech"
TMP_FNAME = "MLCommons_unsupervised_peoples_speech"
SPECIFIC_FILE = "000030.tar"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
class AudioDataset(Dataset):
	def __init__(self, audio_files, processor, max_length, max_audio_seconds=30):
		self.audio_files = audio_files
		self.processor = processor
		self.max_length = max_length
		self.max_audio_seconds = max_audio_seconds

	def __len__(self):
		return len(self.audio_files)

	def __getitem__(self, idx):
		item = self.audio_files[idx]
		
		# Limit audio length to reduce memory usage
		audio = item["array"]
		sr = item["sampling_rate"]
		max_samples = int(self.max_audio_seconds * sr)
		
		if len(audio) > max_samples:
			audio = audio[:max_samples]
			
		return {
			"array": audio,
			"sampling_rate": sr,
			"file_name": item["file_name"]
		}

def collate_fn(batch):
	"""Custom collate function for DataLoader"""
	return {
		"array": [item["array"] for item in batch],
		"sampling_rate": batch[0]["sampling_rate"],
		"file_name": [item["file_name"] for item in batch]
	}

def process_tar_file(dataset_id, file_name):
	""" Function to extract and process the specific tar file """
	# Create a consistent temp directory name
	temp_dir = os.path.join(tempfile.gettempdir(), TMP_FNAME)
	extracted_dir = None
	
	# Check if the temp directory already exists and has content
	if os.path.exists(temp_dir) and os.listdir(temp_dir):
		print(f"Found existing temporary directory: {temp_dir}")
		# Try to find the extracted directory
		for item in os.listdir(temp_dir):
			item_path = os.path.join(temp_dir, item)
			if os.path.isdir(item_path):
				extracted_dir = item_path
				print(f"Using existing extracted directory: {extracted_dir}")
				break
	
	# If no extracted directory found, we need to download and extract
	if extracted_dir is None:
		# Download the specific file with tqdm progress bar
		from huggingface_hub import hf_hub_download
		
		file_path = hf_hub_download(
			repo_id=dataset_id,
			filename=f"audio/{file_name}",
			repo_type="dataset",
			local_files_only=False,
			resume_download=True
		)
		
		# Create the temp directory if it doesn't exist
		os.makedirs(temp_dir, exist_ok=True)
		
		# Extract audio data from the tar file with tqdm progress
		with tarfile.open(file_path, 'r') as tar:
			members = tar.getmembers()
			print(f"Extracting {len(members)} files from tar archive...")
			
			for member in tqdm(members, desc="Extracting files"):
				tar.extract(member, path=temp_dir)
			
			# Get the extracted directory name (usually the same as the tar name without extension)
			extracted_dir = os.path.join(temp_dir, os.path.splitext(file_name)[0])
			if not os.path.exists(extracted_dir):
				# If the directory name doesn't match the tar name, find the first directory
				for item in os.listdir(temp_dir):
					item_path = os.path.join(temp_dir, item)
					if os.path.isdir(item_path):
						extracted_dir = item_path
						break
			
			print(f"Extracted to directory: {extracted_dir}")
	
	# Process the extracted files
	audio_data = []
	
	# Find all MP3 files
	mp3_files = []
	for root, dirs, files in os.walk(extracted_dir):
		for file in files:
			if file.endswith('.mp3'):
				mp3_files.append(os.path.join(root, file))
	
	print(f"Found {len(mp3_files)} MP3 files")
	
	# Process each MP3 file with tqdm progress
	for mp3_path in tqdm(mp3_files, desc="Loading MP3 files"):
		file = os.path.basename(mp3_path)
		try:
			# Load the audio file using librosa
			audio, sr = librosa.load(mp3_path, sr=None)
			
			audio_data.append({
				"array": audio,
				"sampling_rate": sr,
				"file_name": file
			})
		except Exception as e:
			print(f"Error loading {file}: {e}")
	
	return audio_data

def csv_results(mse_values, max_diff_values):
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

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add parameters for dataloader - Reduced defaults to help with memory issues
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')
	parser.add_argument('-a', '--max_audio_seconds', type=int, default=30, help='maximum audio length in seconds to process')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	processor = WhisperProcessor.from_pretrained(MODEL_ID)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

	if args.quantization_bits == 4:
		model_quantized = WhisperForConditionalGeneration.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map=device
		)
	elif args.quantization_bits == 8:
		model_quantized = WhisperForConditionalGeneration.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map=device
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")
	
	print(f"Both models loaded to {device}")

	# Extract audio from the specific tar file
	print(f"Processing {SPECIFIC_FILE}...")
	audio_files = process_tar_file(DATASET_ID, SPECIFIC_FILE)

	if not audio_files:
		print(f"No audio files found in {SPECIFIC_FILE}")
		exit()

	# Create dataset and dataloader with reduced batch size to avoid OOM
	dataset = AudioDataset(
		audio_files, 
		processor, 
		args.max_lenght_tokenizer,
		max_audio_seconds=args.max_audio_seconds
	)
	
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size, # Use a smaller batch size
		shuffle=False, # Changed to False for more predictable memory usage
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device.type == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True if args.num_workers > 0 else False
	)

	# Dictionary to store MSE values for each file
	all_mse_values = {}
	all_max_diff_values = {}

	# Process each audio file
	print(f"Running {len(audio_files)} audio files inference with batch size {args.batch_size} on {device}")

	# Process files in smaller chunks to avoid OOM
	print("Trying with a smaller chunk size...")
	chunk_size = 1 # Process one file at a time

	for batch_idx, batch in enumerate(dataloader):
		
		# Try to process the batch (1 file at a time)
		for i in tqdm(range(len(batch["file_name"]))):
			try:
				mse_values = {}
				max_diff_values = {}

				# Create a single-file batch
				single_batch = {
					"array": [batch["array"][i]],
					"sampling_rate": batch["sampling_rate"],
					"file_name": [batch["file_name"][i]]
				}
				
				# Use a much smaller batch size to process files sequentially
				file_names = single_batch["file_name"]
				audio_array = single_batch["array"]
				sampling_rate = single_batch["sampling_rate"]

				# Chunk the data
				for i in range(0, len(file_names), chunk_size):
					chunk_files = file_names[i:i+chunk_size]
					chunk_audio = audio_array[i:i+chunk_size]

					# Check if the model requires resampling of the audio
					if processor.feature_extractor.sampling_rate != sampling_rate:
						chunk_audio = [librosa.resample(
							audio, 
							orig_sr=sampling_rate, 
							target_sr=processor.feature_extractor.sampling_rate
						) for audio in chunk_audio]
						current_sampling_rate = processor.feature_extractor.sampling_rate
					else:
						current_sampling_rate = sampling_rate
						
					# Tokenize input batch
					inputs = processor.feature_extractor(
						chunk_audio, 
						sampling_rate=current_sampling_rate, 
						return_tensors="pt"
					).to(device)
					
					# Prepare decoder input IDs
					decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)
					
					with torch.no_grad():
						original_outputs = model(
							input_features=inputs.input_features, 
							decoder_input_ids=decoder_input_ids
						).logits.detach().cpu().numpy()
					
					# Explicitly convert to half precision (float16) to match quantized model
					inputs.input_features = inputs.input_features.half().to(device)
					
					with torch.no_grad():
						quantized_outputs = model_quantized(
							input_features=inputs.input_features, 
							decoder_input_ids=decoder_input_ids
						).logits.detach().cpu().numpy()
					
					# Calculate metrics
					for j, file_name in enumerate(chunk_files):
						original_slice = original_outputs[j:j+1]
						quantized_slice = quantized_outputs[j:j+1]
						
						# Calculate MSE and max difference
						mse = np.mean((original_slice - quantized_slice) ** 2)
						max_diff = np.max(np.abs(original_slice - quantized_slice))
						
						mse_values[file_name] = float(mse)
						max_diff_values[file_name] = float(max_diff)
				
				# Update the global dictionaries
				all_mse_values.update(mse_values)
				all_max_diff_values.update(max_diff_values)
				
			except Exception as inner_e:
				print(f"Failed to process file {batch['file_name'][i]}: {str(inner_e)}")
		
		torch.cuda.empty_cache()
		gc.collect()

	# Save results
	csv_results(all_mse_values, all_max_diff_values)

if __name__ == "__main__":
	main()
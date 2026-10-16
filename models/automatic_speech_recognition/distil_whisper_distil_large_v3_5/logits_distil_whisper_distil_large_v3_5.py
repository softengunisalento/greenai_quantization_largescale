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
from jiwer import wer, cer  # For WER calculation
import re


### Constants
MODEL_ID = "distil-whisper/distil-large-v3.5"
DATASET_ID = "MLCommons/unsupervised_peoples_speech"
TMP_FNAME = "MLCommons_unsupervised_peoples_speech"
SPECIFIC_FILE = "000030.tar"
from pathlib import Path
CSV_FILE = Path(__file__).with_suffix('.csv').resolve()


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

def normalize_text(text):
	"""Normalize text for WER calculation"""
	# Convert to lowercase
	text = text.lower()
	# Remove punctuation and extra whitespace
	text = re.sub(r'[^\w\s]', '', text)
	text = re.sub(r'\s+', ' ', text)
	return text.strip()

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
			
				try:
					delfile = "/tmp/MLCommons_unsupervised_peoples_speech/LibrivoxM4bCollectionAudiobooks_7/DasBildnisDesDorianGray01-09.mp3"
					if os.path.exists(delfile):
						os.remove(delfile)
						print(f"File '{delfile}' has been deleted successfully.")
				except PermissionError:
					print(f"Permission denied: Cannot delete '{delfile}'.")
				except Exception as e:
					print(f"Error deleting file: {e}")
			
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

def csv_results(wer_values, cer_values, transcription_results):
	""" Save WER and CER statistics to CSV """
	wer_series = pd.Series(list(wer_values.values()))
	cer_series = pd.Series(list(cer_values.values()))

	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# Write summary statistics
		writer.writerow(["Metric", "Min", "Max", "Mean", "Median", "Std Dev"])
		writer.writerow([
			"WER", 
			wer_series.min(), 
			wer_series.max(), 
			wer_series.mean(), 
			wer_series.median(), 
			wer_series.std()
		])
		writer.writerow([
			"CER", 
			cer_series.min(), 
			cer_series.max(), 
			cer_series.mean(), 
			cer_series.median(), 
			cer_series.std()
		])

	print(f"Results saved to {CSV_FILE}")

def generate_transcription(model, processor, input_features, device):
	"""Generate transcription using the model"""
	with torch.no_grad():
		# Generate tokens
		predicted_ids = model.generate(
			input_features,
			max_length=448,  # Standard max length for Whisper
			num_beams=1,     # Greedy decoding for consistency
			do_sample=False
		)
		
		# Decode to text
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
		return transcription

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="Evaluate Whisper model quantization using WER")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=1, help='size of the batches for DataLoader (recommended: 1 for WER evaluation)')
	parser.add_argument('-w', '--num_workers', type=int, default=4, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=2, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=448, help='maximum sequence length for tokenization')
	parser.add_argument('-q', '--quantization_bits', type=int, default=8, help='value to quantize in int4 or int8')
	parser.add_argument('-a', '--max_audio_seconds', type=int, default=30, help='maximum audio length in seconds to process')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	processor = WhisperProcessor.from_pretrained(MODEL_ID)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"device used: {device}")

	# Load original model
	model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

	# Load quantized model
	if args.quantization_bits == 4:
		model_quantized = WhisperForConditionalGeneration.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map="auto",
			torch_dtype=torch.float16
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

	# Create dataset and dataloader
	dataset = AudioDataset(
		audio_files, 
		processor, 
		args.max_lenght_tokenizer,
		max_audio_seconds=args.max_audio_seconds
	)
	
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=True if device.type == 'cuda' else False,
		prefetch_factor=args.prefetch_factor,
		persistent_workers=True if args.num_workers > 0 else False
	)

	# Dictionaries to store results
	all_wer_values = {}
	all_cer_values = {}
	all_transcription_results = {}

	# Process each audio file
	print(f"Running transcription inference on {len(audio_files)} audio files with batch size {args.batch_size} on {device}")

	for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
		try:
			# Process each file in the batch
			for i in range(len(batch["file_name"])):
				file_name = batch["file_name"][i]
				audio_array = batch["array"][i]
				sampling_rate = batch["sampling_rate"]
				
				# Resample if necessary
				if processor.feature_extractor.sampling_rate != sampling_rate:
					audio_array = librosa.resample(
						audio_array, 
						orig_sr=sampling_rate, 
						target_sr=processor.feature_extractor.sampling_rate
					)
					current_sampling_rate = processor.feature_extractor.sampling_rate
				else:
					current_sampling_rate = sampling_rate
				
				# Prepare input features
				inputs = processor.feature_extractor(
					[audio_array], 
					sampling_rate=current_sampling_rate, 
					return_tensors="pt"
				).to(device)
				
				# Generate transcription with original model
				original_transcription = generate_transcription(
					model, processor, inputs.input_features, device
				)
				
				# Generate transcription with quantized model
				# Convert to appropriate precision for quantized model
				quantized_inputs = inputs.input_features.half()
				
				quantized_transcription = generate_transcription(
					model_quantized, processor, quantized_inputs, device
				)
				
				# Normalize transcriptions for WER calculation
				original_normalized = normalize_text(original_transcription)
				quantized_normalized = normalize_text(quantized_transcription)
				
				# Calculate WER and CER
				if original_normalized and quantized_normalized:
					try:
						wer_score = wer(original_normalized, quantized_normalized)
						cer_score = cer(original_normalized, quantized_normalized)
					except Exception as e:
						print(f"Error calculating WER/CER for {file_name}: {e}")
						wer_score = 1.0  # Maximum error
						cer_score = 1.0
				else:
					# Handle empty transcriptions
					if not original_normalized and not quantized_normalized:
						wer_score = 0.0
						cer_score = 0.0
					else:
						wer_score = 1.0
						cer_score = 1.0
				
				# Store results
				all_wer_values[file_name] = wer_score
				all_cer_values[file_name] = cer_score
				all_transcription_results[file_name] = {
					'original': original_transcription,
					'quantized': quantized_transcription
				}
				
		except Exception as e:
			print(f"Failed to process batch {batch_idx}: {str(e)}")
			continue
		
		# Clean up GPU memory
		if device.type == 'cuda':
			torch.cuda.empty_cache()
		gc.collect()

	# Save results to CSV
	csv_results(all_wer_values, all_cer_values, all_transcription_results)

if __name__ == "__main__":
	main()
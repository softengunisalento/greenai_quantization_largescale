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
MODEL_ID = "distil-whisper/distil-large-v3.5"
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

def calculate_accuracy(original_model, model_int4, model_int8, dataloader, device, processor):
	"""Calculate transcription accuracy for all models and measure the drops"""
	# Initialize dictionaries to store transcriptions
	original_transcriptions = {}
	int4_transcriptions = {}
	int8_transcriptions = {}
	
	for batch in tqdm(dataloader, desc="Evaluating transcription accuracy"):
		# Process each file individually
		for i in range(len(batch["file_name"])):
			try:
				file_name = batch["file_name"][i]
				print(f"Processing file: {file_name}")
				
				# Get audio
				audio = batch["array"][i]
				sampling_rate = batch["sampling_rate"]
				
				# Check if the model requires resampling of the audio
				if processor.feature_extractor.sampling_rate != sampling_rate:
					audio = librosa.resample(
						audio, 
						orig_sr=sampling_rate, 
						target_sr=processor.feature_extractor.sampling_rate
					)
					current_sampling_rate = processor.feature_extractor.sampling_rate
				else:
					current_sampling_rate = sampling_rate
				
				# Process audio
				inputs = processor.feature_extractor(
					[audio], 
					sampling_rate=current_sampling_rate, 
					return_tensors="pt"
				).to(device)
				
				# Generate transcriptions with original model
				with torch.no_grad():
					original_result = original_model.generate(inputs=inputs.input_features)
					original_transcription = processor.batch_decode(original_result, skip_special_tokens=True)[0]
				
				# Explicitly convert to half precision for the quantized models
				half_inputs = inputs.input_features.half().to(device)
				
				# Generate transcriptions with INT4 model
				with torch.no_grad():
					int4_result = model_int4.generate(inputs=half_inputs)
					int4_transcription = processor.batch_decode(int4_result, skip_special_tokens=True)[0]
				
				# Generate transcriptions with INT8 model
				with torch.no_grad():
					int8_result = model_int8.generate(inputs=half_inputs)
					int8_transcription = processor.batch_decode(int8_result, skip_special_tokens=True)[0]
				
				# Store transcriptions
				original_transcriptions[file_name] = original_transcription
				int4_transcriptions[file_name] = int4_transcription
				int8_transcriptions[file_name] = int8_transcription
				
				print(f"Original: {original_transcription}")
				print(f"INT4: {int4_transcription}")
				print(f"INT8: {int8_transcription}")
				
			except Exception as e:
				print(f"Failed to process file {file_name}: {str(e)}")
	
	# Calculate similarity metrics
	int4_exact_match = 0
	int8_exact_match = 0
	
	int4_char_similarity = []
	int8_char_similarity = []
	
	for file_name in original_transcriptions:
		orig = original_transcriptions[file_name]
		int4 = int4_transcriptions[file_name]
		int8 = int8_transcriptions[file_name]
		
		# Count exact matches
		if orig == int4:
			int4_exact_match += 1
		if orig == int8:
			int8_exact_match += 1
		
		# Calculate character-level similarity (simple implementation)
		# A more sophisticated approach would use Levenshtein distance
		if len(orig) > 0:
			int4_sim = sum(c1 == c2 for c1, c2 in zip(orig, int4[:len(orig)])) / len(orig)
			int8_sim = sum(c1 == c2 for c1, c2 in zip(orig, int8[:len(orig)])) / len(orig)
			
			int4_char_similarity.append(int4_sim)
			int8_char_similarity.append(int8_sim)
	
	# Calculate metrics
	total_files = len(original_transcriptions)
	int4_exact_match_rate = int4_exact_match / total_files if total_files > 0 else 0
	int8_exact_match_rate = int8_exact_match / total_files if total_files > 0 else 0
	
	int4_avg_char_similarity = sum(int4_char_similarity) / len(int4_char_similarity) if int4_char_similarity else 0
	int8_avg_char_similarity = sum(int8_char_similarity) / len(int8_char_similarity) if int8_char_similarity else 0
	
	# Calculate drops
	int4_drop = 1 - int4_exact_match_rate
	int8_drop = 1 - int8_exact_match_rate
	
	int4_char_drop = 1 - int4_avg_char_similarity
	int8_char_drop = 1 - int8_avg_char_similarity
	
	return {
		"total_files": total_files,
		"int4_exact_match_rate": int4_exact_match_rate,
		"int8_exact_match_rate": int8_exact_match_rate,
		"int4_avg_char_similarity": int4_avg_char_similarity,
		"int8_avg_char_similarity": int8_avg_char_similarity,
		"int4_drop": int4_drop,
		"int8_drop": int8_drop,
		"int4_char_drop": int4_char_drop,
		"int8_char_drop": int8_char_drop,
		"original_transcriptions": original_transcriptions,
		"int4_transcriptions": int4_transcriptions,
		"int8_transcriptions": int8_transcriptions
	}

def calculate_hidden_state_mse(original_model, model_int4, model_int8, dataloader, device, processor):
	"""Calculate MSE and max difference between original and quantized models"""
	# Dictionaries to store MSE values and max differences for each quantized model
	mse_values_int4 = {}
	mse_values_int8 = {}
	max_diff_values_int4 = {}
	max_diff_values_int8 = {}

	# Process files in smaller chunks to avoid OOM
	print("Processing files one at a time...")
	
	for batch in dataloader:
		# Process each file individually
		for i in range(len(batch["file_name"])):
			try:
				file_name = batch["file_name"][i]
				print(f"Processing file: {file_name}")
				
				# Create a single-file batch
				audio = batch["array"][i]
				sampling_rate = batch["sampling_rate"]
				
				# Check if the model requires resampling of the audio
				if processor.feature_extractor.sampling_rate != sampling_rate:
					audio = librosa.resample(
						audio, 
						orig_sr=sampling_rate, 
						target_sr=processor.feature_extractor.sampling_rate
					)
					current_sampling_rate = processor.feature_extractor.sampling_rate
				else:
					current_sampling_rate = sampling_rate
					
				# Tokenize input
				inputs = processor.feature_extractor(
					[audio], 
					sampling_rate=current_sampling_rate, 
					return_tensors="pt"
				).to(device)
				
				# Prepare decoder input IDs
				decoder_input_ids = torch.tensor([[original_model.config.decoder_start_token_id]]).to(device)
				
				# Get outputs from original model
				with torch.no_grad():
					original_outputs = original_model(
						input_features=inputs.input_features, 
						decoder_input_ids=decoder_input_ids
					).logits.detach().cpu().numpy()
				
				# Explicitly convert to half precision (float16) to match quantized model
				half_inputs = inputs.input_features.half().to(device)
				
				# Get outputs from INT4 model
				with torch.no_grad():
					int4_outputs = model_int4(
						input_features=half_inputs, 
						decoder_input_ids=decoder_input_ids
					).logits.detach().cpu().numpy()
				
				# Get outputs from INT8 model
				with torch.no_grad():
					int8_outputs = model_int8(
						input_features=half_inputs, 
						decoder_input_ids=decoder_input_ids
					).logits.detach().cpu().numpy()
				
				# Calculate metrics
				# INT4 model metrics
				mse_int4 = np.mean((original_outputs - int4_outputs) ** 2)
				max_diff_int4 = np.max(np.abs(original_outputs - int4_outputs))
				
				# INT8 model metrics
				mse_int8 = np.mean((original_outputs - int8_outputs) ** 2)
				max_diff_int8 = np.max(np.abs(original_outputs - int8_outputs))
				
				# Store results
				mse_values_int4[file_name] = float(mse_int4)
				max_diff_values_int4[file_name] = float(max_diff_int4)
				
				mse_values_int8[file_name] = float(mse_int8)
				max_diff_values_int8[file_name] = float(max_diff_int8)
				
				print(f"Completed file: {file_name}, MSE INT4: {mse_int4:.6f}, MSE INT8: {mse_int8:.6f}")
				
			except Exception as e:
				print(f"Failed to process file {batch['file_name'][i]}: {str(e)}")
	
	return {
		"mse_values_int4": mse_values_int4,
		"mse_values_int8": mse_values_int8,
		"max_diff_values_int4": max_diff_values_int4,
		"max_diff_values_int8": max_diff_values_int8
	}

def csv_results(mse_results, accuracy_results):
	""" Print MSE statistics and accuracy results """
	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		
		# Write accuracy results first (most important)
		writer.writerow(["Transcription Accuracy Metrics"])
		writer.writerow(["Metric", "Original", "Int4", "Int8"])
		writer.writerow(["Exact Match Rate", "1.0", 
						accuracy_results["int4_exact_match_rate"], 
						accuracy_results["int8_exact_match_rate"]])
		writer.writerow(["Character Similarity", "1.0", 
						accuracy_results["int4_avg_char_similarity"], 
						accuracy_results["int8_avg_char_similarity"]])
		writer.writerow(["Exact Match Drop", "N/A", 
						accuracy_results["int4_drop"], 
						accuracy_results["int8_drop"]])
		writer.writerow(["Character Drop", "N/A", 
						accuracy_results["int4_char_drop"], 
						accuracy_results["int8_char_drop"]])
		writer.writerow(["Total files processed", accuracy_results["total_files"]])
		
		writer.writerow([])  # Empty row for separation
		
		if mse_results:
			# MSE statistics
			mse_int4_series = pd.Series(list(mse_results["mse_values_int4"].values()))
			mse_int8_series = pd.Series(list(mse_results["mse_values_int8"].values()))
			
			writer.writerow(["Hidden State MSE Metrics"])
			writer.writerow(["Metric", "Int4", "Int8", "Difference (Int4-Int8)"])
			writer.writerow(["Min MSE", mse_int4_series.min(), mse_int8_series.min(), 
							mse_int4_series.min() - mse_int8_series.min()])
			writer.writerow(["Max MSE", mse_int4_series.max(), mse_int8_series.max(), 
							mse_int4_series.max() - mse_int8_series.max()])
			writer.writerow(["Mean MSE", mse_int4_series.mean(), mse_int8_series.mean(), 
							mse_int4_series.mean() - mse_int8_series.mean()])
			writer.writerow(["Median MSE", mse_int4_series.median(), mse_int8_series.median(), 
							mse_int4_series.median() - mse_int8_series.median()])
			writer.writerow(["Std Dev MSE", mse_int4_series.std(), mse_int8_series.std(), 
							mse_int4_series.std() - mse_int8_series.std()])
			
			# Max difference statistics
			max_diff_int4_series = pd.Series(list(mse_results["max_diff_values_int4"].values()))
			max_diff_int8_series = pd.Series(list(mse_results["max_diff_values_int8"].values()))
			
			writer.writerow([])  # Empty row for separation
			writer.writerow(["Hidden State Max Difference Metrics"])
			writer.writerow(["Metric", "Int4", "Int8", "Difference (Int4-Int8)"])
			writer.writerow(["Min Max-Diff", max_diff_int4_series.min(), max_diff_int8_series.min(), 
							max_diff_int4_series.min() - max_diff_int8_series.min()])
			writer.writerow(["Max Max-Diff", max_diff_int4_series.max(), max_diff_int8_series.max(), 
							max_diff_int4_series.max() - max_diff_int8_series.max()])
			writer.writerow(["Mean Max-Diff", max_diff_int4_series.mean(), max_diff_int8_series.mean(), 
							max_diff_int4_series.mean() - max_diff_int8_series.mean()])
			writer.writerow(["Median Max-Diff", max_diff_int4_series.median(), max_diff_int8_series.median(), 
							max_diff_int4_series.median() - max_diff_int8_series.median()])
			writer.writerow(["Std Dev Max-Diff", max_diff_int4_series.std(), max_diff_int8_series.std(), 
							max_diff_int4_series.std() - max_diff_int8_series.std()])

	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add parameters for dataloader
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='size of the batches for DataLoader')
	parser.add_argument('-w', '--num_workers', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('-f', '--prefetch_factor', type=int, default=4, help='number of batches loaded in advance by each worker')
	parser.add_argument('-t', '--max_lenght_tokenizer', type=int, default=128, help='maximum sequence length for tokenization')
	parser.add_argument('-a', '--max_audio_seconds', type=int, default=30, help='maximum audio length in seconds to process')

	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	processor = WhisperProcessor.from_pretrained(MODEL_ID)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

	# Load 4-bit quantized model
	print(f"Loading 4-bit quantized model...")
	model_int4 = WhisperForConditionalGeneration.from_pretrained(
		MODEL_ID,
		load_in_4bit=True,
		device_map=device
	)
	
	# Load 8-bit quantized model
	print(f"Loading 8-bit quantized model...")
	model_int8 = WhisperForConditionalGeneration.from_pretrained(
		MODEL_ID,
		load_in_8bit=True,
		device_map=device
	)
	
	print(f"All models loaded to {device}")

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

	# Process each audio file
	print(f"Running {len(audio_files)} audio files inference with batch size {args.batch_size} on {device}")
	
	# Calculate hidden state MSE
	print("Calculating hidden state differences...")
	mse_results = calculate_hidden_state_mse(model, model_int4, model_int8, dataloader, device, processor)

	# Calculate accuracy drop for both quantization methods
	print("Calculating accuracy drops...")
	accuracy_results = calculate_accuracy(model, model_int4, model_int8, dataloader, device, processor)
	
	# Print results to console
	print("\nAccuracy Results:")
	print(f"Original model used as ground truth reference (accuracy: 1.0)")
	print(f"Int4 model exact match rate: {accuracy_results['int4_exact_match_rate']:.4f} (drop: {accuracy_results['int4_drop']:.4f})")
	print(f"Int8 model exact match rate: {accuracy_results['int8_exact_match_rate']:.4f} (drop: {accuracy_results['int8_drop']:.4f})")
	print(f"Int4 model character similarity: {accuracy_results['int4_avg_char_similarity']:.4f} (character-level drop: {accuracy_results['int4_char_drop']:.4f})")
	print(f"Int8 model character similarity: {accuracy_results['int8_avg_char_similarity']:.4f} (character-level drop: {accuracy_results['int8_char_drop']:.4f})")
	print(f"Total files processed: {accuracy_results['total_files']}")
	
	# Save results to CSV
	csv_results(mse_results, accuracy_results)
	
	# Generate a comparison report
	if accuracy_results["int4_exact_match_rate"] > accuracy_results["int8_exact_match_rate"]:
		print("\nQuantization Comparison: Int4 outperforms Int8 in transcription accuracy!")
	elif accuracy_results["int8_exact_match_rate"] > accuracy_results["int4_exact_match_rate"]:
		print("\nQuantization Comparison: Int8 outperforms Int4 in transcription accuracy!")
	else:
		print("\nQuantization Comparison: Int4 and Int8 have identical transcription accuracy.")
		
	if mse_results:
		int4_mse_mean = pd.Series(list(mse_results["mse_values_int4"].values())).mean()
		int8_mse_mean = pd.Series(list(mse_results["mse_values_int8"].values())).mean()
		
		if int4_mse_mean < int8_mse_mean:
			print("Int4 has lower mean MSE than Int8, suggesting better hidden state preservation.")
		else:
			print("Int8 has lower mean MSE than Int4, suggesting better hidden state preservation.")

	# Free cache
	torch.cuda.empty_cache()

if __name__ == "__main__":
	main()
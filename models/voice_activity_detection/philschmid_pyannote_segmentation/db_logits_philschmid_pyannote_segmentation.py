from pyannote.audio import Inference, Model
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
MODEL_ID = "philschmid/pyannote-segmentation"
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
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = Model.from_pretrained(MODEL_ID).to(device)
	inference = Inference(model)

	if args.quantization_bits == 4:
		model_quantized = Model.from_pretrained(
			MODEL_ID,
			load_in_4bit=True,
			device_map="auto"
		)
	elif args.quantization_bits == 8:
		model_quantized = Model.from_pretrained(
			MODEL_ID,
			load_in_8bit=True,
			device_map="auto"
		)
	else:
		raise ValueError("Only 4-bit and 8-bit quantization are supported.")
	
	quantized_inference = Inference(model_quantized)
	
	print(f"Both models ready on {device}")

	# Extract audio from the specific tar file
	print(f"Processing {SPECIFIC_FILE}...")
	audio_files = process_tar_file(DATASET_ID, SPECIFIC_FILE)

	if not audio_files:
		print(f"No audio files found in {SPECIFIC_FILE}")
		exit()

	dataset = AudioDataset(
		audio_files,
		processor=None,
		max_length=args.max_lenght_tokenizer,
		max_audio_seconds=args.max_audio_seconds
	)

	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=(device.type == 'cuda'),
		prefetch_factor=args.prefetch_factor,
		persistent_workers=(args.num_workers > 0)
	)

	all_mse_values = {}
	all_max_diff_values = {}

	print(f"Running inference on {len(audio_files)} audio files using {device}")

	for batch_idx, batch in enumerate(dataloader):
		for i in tqdm(range(len(batch["file_name"]))):
			try:
				file_name = batch["file_name"][i]
				audio = batch["array"][i]
				sampling_rate = batch["sampling_rate"]

				# Convert audio to PyTorch tensor and add batch dimension
				waveform_tensor = torch.tensor(audio).unsqueeze(0)

				# Run inference with original model
				embedding_original = inference(
					{"waveform": waveform_tensor, "sample_rate": sampling_rate}
				).data

				# Run inference with quantized model
				embedding_quantized = quantized_inference(
					{"waveform": waveform_tensor, "sample_rate": sampling_rate}
				).data

				# Compute error metrics
				mse = np.mean((embedding_original - embedding_quantized) ** 2)
				max_diff = np.max(np.abs(embedding_original - embedding_quantized))

				all_mse_values[file_name] = float(mse)
				all_max_diff_values[file_name] = float(max_diff)

			except Exception as inner_e:
				print(f"Failed to process file {batch['file_name'][i]}: {str(inner_e)}")

		torch.cuda.empty_cache()
		gc.collect()

	# Save results to CSV
	csv_results(all_mse_values, all_max_diff_values)


if __name__ == "__main__":
	main()

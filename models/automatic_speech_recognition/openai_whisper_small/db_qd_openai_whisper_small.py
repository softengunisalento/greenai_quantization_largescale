from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import argparse
import os
import time
import tarfile
import librosa
import tempfile
from tqdm import tqdm
import csv

### Constants
N_BYTES = 1 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "openai/whisper-small"
DATASET_ID = "MLCommons/unsupervised_peoples_speech"
TMP_FNAME = "MLCommons_unsupervised_peoples_speech"
SPECIFIC_FILE = "000030.tar"
FILE_NAME = "openai_whisper_small"
CSV_FILE = "db_qd_" + FILE_NAME + ".csv"

### FunctionsFunctions
def process_tar_file(dataset_id, file_name):
	""" Function to extract and process the specific tar file """
	# Create a temp directory name
	temp_dir = tempfile.gettempdir() # system temp directory
	temp_dir = os.path.join(temp_dir, TMP_FNAME)
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

def verbose_measurements(model):
	""" Function to print verbose output """
	# check the memory footprint of the model
	mem_footprint = model.get_memory_footprint()

	# checking the size of the model
	module_sizes = evaluate_size(model)

	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# write stats
		writer.writerow(["Footprint MBs", "Size GB"])
		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

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

		power_after = get_gpu_power()
		average_power_usage = (power_after + power_before) / 2

		energy_joules = calculate_energy_consumption(average_power_usage, inference_time)

		co2_emissions = calculate_co2_emissions(energy_joules)

		with open(CSV_FILE, mode='a', newline='') as file:
			writer = csv.writer(file)
			
			# write stats
			writer.writerow(["Inference s", "Average Power Consumption W","Energy J", "CO2 Emissions gr"])
			writer.writerow([inference_time, average_power_usage, energy_joules, co2_emissions])

		return result
	return wrapper

def csv_results(transcriptions_s_tokens, transcriptions):
	""" Print statistics """

	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# write files stats
		writer.writerow(["File Name", "Transcription", "Transcription with special tokens"])

		for file_name in transcriptions:
			writer.writerow([file_name, transcriptions[file_name], transcriptions_s_tokens[file_name]])

	print(f"Results saved to {CSV_FILE}")

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

	# Load the Whisper model and processor
	processor = WhisperProcessor.from_pretrained(MODEL_ID)
	model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

	# Apply dynamic quantization to the model
	model_quantized = torch.quantization.quantize_dynamic(
		model,  # the model instance
		{torch.nn.Linear},  # specify the layers to quantize
		dtype=torch.qint8  # quantized dtype (8-bit int)
	)

	# Extract audio from the specific tar file
	print(f"Processing {SPECIFIC_FILE}...")
	audio_files = process_tar_file(DATASET_ID, SPECIFIC_FILE)

	if not audio_files:
		print(f"No audio files found in {SPECIFIC_FILE}")
		exit()

	transcriptions_s_tokens = {}
	transcriptions = {}
	
	# Process each audio file
	print(f"Found {len(audio_files)} audio files, processing...")
	for idx, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
		audio_array = audio_file["array"]
		sampling_rate = audio_file["sampling_rate"]
		file_name = audio_file["file_name"]
		
		# Check if the model requires resampling of the audio
		if processor.feature_extractor.sampling_rate != sampling_rate:
			audio_array = librosa.resample(
				audio_array, 
				orig_sr=sampling_rate, 
				target_sr=processor.feature_extractor.sampling_rate
			)
			sampling_rate = processor.feature_extractor.sampling_rate
		
		# Prepare input data
		inputs = processor(
			audio_array, 
			sampling_rate=sampling_rate, 
			return_tensors="pt"
		).input_features
		
		# Generate token ids
		predicted_ids = model_quantized.generate(inputs)

		# Decode token ids to text
		transcriptions_s_tokens[file_name] = processor.batch_decode(predicted_ids, skip_special_tokens=False)
		transcriptions[file_name] = processor.batch_decode(predicted_ids, skip_special_tokens=True)

	csv_results(transcriptions_s_tokens, transcriptions)

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
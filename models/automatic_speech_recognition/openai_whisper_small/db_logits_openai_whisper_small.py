from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
import tarfile
import librosa
import tempfile
from tqdm import tqdm
import pandas as pd
import csv

### Constants
MODEL_ID = "openai/whisper-small"
DATASET_ID = "MLCommons/unsupervised_peoples_speech"
TMP_FNAME = "MLCommons_unsupervised_peoples_speech"
SPECIFIC_FILE = "000030.tar"
FILE_NAME = "openai_whisper_small"
CSV_FILE = "db_logits_" + FILE_NAME + ".csv"


### Functions
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

def csv_results():
	""" Print MSE statistics """
	mse_series = pd.Series(list(mse_values.values()))

	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["File Name", "MSE", "Max Difference"])
		
		for file_name in mse_values:
			writer.writerow([file_name, mse_values[file_name], max_diff_values[file_name]])
		
		writer.writerow(["Min MSE", "Max MSE", "Mean MSE", "Median MSE", "Standard Deviation"])
		writer.writerow([mse_series.min(), mse_series.max(), mse_series.mean(), mse_series.median(), mse_series.std()])

	print(f"Results saved to {CSV_FILE}")

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

# Dictionary to store MSE values for each file
mse_values = {}
max_diff_values = {}

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
	inputs = processor.feature_extractor(
		audio_array, 
		sampling_rate=sampling_rate, 
		return_tensors="pt"
	)

	# Prepare dummy decoder input IDs
	decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
	
	# Forward pass through the original model
	with torch.no_grad():
		original_logits = model(input_features=inputs.input_features, decoder_input_ids=decoder_input_ids).logits
	
	# Forward pass through the quantized model
	with torch.no_grad():
		quantized_logits = model_quantized(input_features=inputs.input_features, decoder_input_ids=decoder_input_ids).logits
	
	# Calculate the mean squared error (MSE) between the logits
	mse = torch.mean((original_logits - quantized_logits) ** 2).item()
	
	# Calculate the maximum absolute difference
	max_diff = torch.max(torch.abs(original_logits - quantized_logits)).item()
	
	# Store values for later analysis
	mse_values[file_name] = mse
	max_diff_values[file_name] = max_diff

csv_results()
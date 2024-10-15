from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch

### Constants
MODEL_ID = "openai/whisper-small"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_CONFIG = "clean"
DATASET_TYPE = "audio"

# Load a dummy dataset from the Hugging Face Datasets library
dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation")
sample = dataset[0][DATASET_TYPE]

# Load the Whisper model and processor
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Process the audio sample
inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

# Prepare dummy decoder input IDs
decoder_input_ids = torch.zeros((inputs.shape[0], 1), dtype=torch.long)  # 1 for a start token

# Forward pass through the original model
with torch.no_grad():
    original_logits = model(input_features=inputs, decoder_input_ids=decoder_input_ids).logits

# Forward pass through the quantized model
with torch.no_grad():
    quantized_logits = model_quantized(input_features=inputs, decoder_input_ids=decoder_input_ids).logits

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_logits - quantized_logits) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")

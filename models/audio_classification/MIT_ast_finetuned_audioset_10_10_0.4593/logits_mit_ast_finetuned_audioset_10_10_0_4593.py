from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch

### Constants
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
DATASET_ID = "hf-internal-testing/librispeech_asr_demo"
DATASET_CONFIG = "clean"
DATA_TYPE = "audio"

# Load the dataset
dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", trust_remote_code=True)
dataset = dataset.sort("id")
sampling_rate = dataset.features[DATA_TYPE].sampling_rate

# Load the Feature Extractor and Original Model
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = ASTForAudioClassification.from_pretrained(MODEL_ID)

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Prepare input data
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

# Forward pass through the original model
with torch.no_grad():
    original_logits = model(**inputs).logits

# Forward pass through the quantized model
with torch.no_grad():
    quantized_logits = model_quantized(**inputs).logits

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_logits - quantized_logits) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")

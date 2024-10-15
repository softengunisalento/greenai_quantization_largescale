from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

### Constants
MODEL_ID = "google-t5/t5-small"
INPUT_TEXT = "translate English to French: How are you?"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# Tokenize the input sentence
inputs = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids

# Generate logits from the original model
with torch.no_grad():
    original_outputs = model(input_ids=inputs, labels=inputs)
    original_logits = original_outputs.logits

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Generate logits from the original model
with torch.no_grad():
    quantized_outputs = model_quantized(input_ids=inputs, labels=inputs)
    quantized_logits = quantized_outputs.logits

# Flatten the logits for MSE calculation
original_logits_flat = original_logits.view(-1)
quantized_logits_flat = quantized_logits.view(-1)

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_logits_flat - quantized_logits_flat) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")

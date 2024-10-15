from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

### Constants
MODEL_ID = "bigscience/bloomz-560m"
INPUT_TEXT = "Give me a short introduction to large language model."
TORCH_DTYPE = "float32"
DEVICE_MAP = "cpu"

# Load the tokenizer and model from sentence-transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE_MAP)

# Tokenize input documents and queries
inputs = tokenizer(INPUT_TEXT, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Set the model to evaluation mode
model.eval()

# Forward pass through the original model
with torch.no_grad():
    original_outputs = model(**inputs)

# Extract last hidden state from the original outputs
original_hidden_states = original_outputs.logits

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Forward pass through the quantized model
with torch.no_grad():
    quantized_outputs = model_quantized(**inputs)

# Extract last hidden state from the quantized outputs
quantized_hidden_states = quantized_outputs.logits

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_hidden_states - quantized_hidden_states) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")
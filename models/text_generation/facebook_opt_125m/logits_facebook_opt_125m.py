from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

### Constants
MODEL_ID = "facebook/opt-125m"
PROMPT = "Once upon a time in a faraway land"

# Load the tokenizer and model from sentence-transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# Tokenize input documents and queries
inputs = tokenizer(PROMPT, return_tensors='pt')

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
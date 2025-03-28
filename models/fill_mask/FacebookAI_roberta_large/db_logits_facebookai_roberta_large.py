from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

### Constants
MODEL_ID = "FacebookAI/roberta-large"
PARQUET_FILE = "../train-00009-of-00041.parquet"  # Path to dataset file

### Functions
def load_dataset(parquet_file):
	"""Load text from the Parquet file"""
	df = pd.read_parquet(parquet_file)
	return df['text'].tolist()  # Extract text column as list


# Load the tokenizer and model from sentence-transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

# Load dataset
texts = load_dataset(PARQUET_FILE)
print(f"> Loaded {len(texts)} texts from {PARQUET_FILE}")

# Process each text
for i, text in enumerate(texts[:100]):  # Limit to first 5 for demonstration
    print(f"\n> Processing Text {i+1}: {text[:100]}...")  # Show snippet

    inputs = tokenizer(text, return_tensors='pt')

# Set the model to evaluation mode
model.eval()

# Forward pass through the original model
with torch.no_grad():
    original_outputs = model(**inputs)

# Extract last hidden state from the original outputs
original_hidden_states = original_outputs.last_hidden_state

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
quantized_hidden_states = quantized_outputs.last_hidden_state

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_hidden_states - quantized_hidden_states) ** 2)

print(f"{mse.item()}")

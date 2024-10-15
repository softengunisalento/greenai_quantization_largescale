from transformers import AutoTokenizer, AutoModel
import torch

### Constants
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
SEARCH_QUERIES = [
	"How is AI changing the world?",
	"Future of quantum computing?"
] # list of search queries to find in the documents

# Load the tokenizer and model from sentence-transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, safe_serialization=True)

# Tokenize input documents and queries
encoded_inputs = tokenizer(SEARCH_QUERIES, padding=True, truncation=True, return_tensors='pt')

# Set the model to evaluation mode
model.eval()

# Forward pass through the original model
with torch.no_grad():
    original_outputs = model(**encoded_inputs)

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
    quantized_outputs = model_quantized(**encoded_inputs)

# Extract last hidden state from the quantized outputs
quantized_hidden_states = quantized_outputs.last_hidden_state

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_hidden_states - quantized_hidden_states) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")
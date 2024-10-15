from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import torch

### Constants
MODEL_ID = "openai/clip-vit-base-patch32"
IMG_PATH = "./dog.jpg"
CANDIDATE_LABELS = ["cat", "dog", "car", "person"]

# Load the processor, the model and the image
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_ID)
image = Image.open(IMG_PATH)

# Process the image and the candidate labels
inputs = processor(images=image, text=CANDIDATE_LABELS, return_tensors="pt")

# Set the model to evaluation mode
model.eval()

# Forward pass through the original model
with torch.no_grad():
    original_outputs = model(**inputs)

# Extract last hidden state from the original outputs
original_hidden_states = original_outputs[0]

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
quantized_hidden_states = quantized_outputs[0]

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_hidden_states - quantized_hidden_states) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")

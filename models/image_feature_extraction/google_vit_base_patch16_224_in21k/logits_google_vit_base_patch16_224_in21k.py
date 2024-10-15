from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

### Constants
MODEL_ID = "google/vit-base-patch16-224-in21k"
IMG_PATH = "./dog.jpg"

# Load the processor, the model and the image
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
image = Image.open(IMG_PATH)

# Process the image and the candidate labels
inputs = processor(images=image, return_tensors="pt")

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Forward pass through the original model
with torch.no_grad():
    original_outputs = model(**inputs)
    original_logits = original_outputs.logits

# Forward pass through the quantized model
with torch.no_grad():
    quantized_outputs = model_quantized(**inputs)
    quantized_logits = quantized_outputs.logits

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_logits - quantized_logits) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")

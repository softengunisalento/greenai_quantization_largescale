import open_clip
import torch
from PIL import Image

### Constants
MODEL_ID = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
IMG_PATH = "./dog.jpg"
CANDIDATE_LABELS = ["cat", "dog", "car", "person"]
DEVICE = "cpu"

# Load the model, the tokenizer and the image
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
tokenizer = open_clip.get_tokenizer(MODEL_ID)
image = Image.open(IMG_PATH)

# Preprocess the image
image_tensor = preprocess_val(image).unsqueeze(0).to(DEVICE)

# Tokenize the text prompt
text_tensor = tokenizer(CANDIDATE_LABELS).to(DEVICE)

# Move tensors to the same device as the model
model.to(DEVICE)

# Inference:
with torch.no_grad(): # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
    original_image_features = model.encode_image(image_tensor)
    original_text_features = model.encode_text(text_tensor)

# Normalize the features
original_image_features /= original_image_features.norm(dim=-1, keepdim=True)
original_text_features /= original_text_features.norm(dim=-1, keepdim=True)

# Calculate similarity for original model
original_similarity = (original_image_features @ original_text_features.T)

# Get the similarity score for the last label
original_result_similarity = original_similarity[0, -1].item()
print(f"> Original image classification: {original_result_similarity}")

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Move tensors to the same device as the model
model_quantized.to(DEVICE)

# Inference:
with torch.no_grad(): # no need to calculate gradients (disabling to reduce memory usage, speed up computations)
    quantized_image_features = model_quantized.encode_image(image_tensor)
    quantized_text_features = model_quantized.encode_text(text_tensor)

# Normalize the features
quantized_image_features /= quantized_image_features.norm(dim=-1, keepdim=True)
quantized_text_features /= quantized_text_features.norm(dim=-1, keepdim=True)

# Calculate similarity for quantized model
quantized_similarity = (quantized_image_features @ quantized_text_features.T)

# Get the similarity score for the last label
quantized_result_similarity = quantized_similarity[0, -1].item()
print(f"> Original image classification: {quantized_result_similarity}")

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_similarity - quantized_similarity) ** 2)

print(f"> Mean Squared Error between original and quantized model logits: {mse.item()}")

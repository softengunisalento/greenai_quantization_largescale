from pyannote.audio import Inference, Model
import librosa
import torch

### Constants
MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Load the model and processor
model = Model.from_pretrained(MODEL_ID) # Use Model class from pyannote.audio
inference = Inference(MODEL_ID)

# Load a demo audio file from librosa
demo_audio_path = librosa.example('trumpet')

# Extract speaker embeddings from the original model
original_embeddings = inference(demo_audio_path)

# Get the embeddings data from the original model
original_embedding_data = original_embeddings.data
print(f"> Original speaker embeddings shape: {original_embedding_data.shape}")
print(f"> Original speaker embeddings: {original_embedding_data}")

# Set the model to evaluation mode
model.eval()

# Apply dynamic quantization to the model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Extract speaker embeddings from the quantized model
quantized_inference = Inference(model_quantized)
quantized_embeddings = quantized_inference(demo_audio_path)

# Get the embeddings data from the quantized model
quantized_embedding_data = quantized_embeddings.data
print(f"> Quantized speaker embeddings shape: {quantized_embedding_data.shape}")
print(f"> Quantized speaker embeddings: {quantized_embedding_data}")

# Convert embeddings data to PyTorch tensors
original_tensor = torch.tensor(original_embedding_data)
quantized_tensor = torch.tensor(quantized_embedding_data)

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_tensor - quantized_tensor) ** 2)

print(f"{mse.item()}")
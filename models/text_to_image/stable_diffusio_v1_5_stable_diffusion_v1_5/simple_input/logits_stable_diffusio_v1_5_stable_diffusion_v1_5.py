from diffusers import StableDiffusionPipeline
import torch

### Constants
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
PROMPT = "a photo of an astronaut riding a horse on mars"
IMG_PATH = "astronaut_rides_horse.png"
QUANT_IMG_PATH = "quantized_astronaut_rides_horse.png"

### Functions
def extract_logits(unet, tokenizer, text_encoder, prompt, dtype):
    """ Extract logits from the last layer of the UNet model """
    unet.eval()

    # Tokenize and encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Get text embeddings
    text_embeddings = text_encoder(inputs.input_ids)[0]

    # Generate random noise as input to the model
    noise = torch.randn((1, unet.config.in_channels, 64, 64), dtype=dtype).to("cpu")

    # Define diffusion timestep
    timesteps = torch.tensor([50], dtype=torch.long).to("cpu")  # Example timestep

    # Pass through the UNet model
    with torch.no_grad():
        unet_output = unet(noise, timesteps, encoder_hidden_states=text_embeddings)

    # Return the sample from the UNet output (logits)
    return unet_output.sample

### Main

# Load the model
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
pipeline = pipeline.to("cpu")

# Apply dynamic quantization to the model
quanto_pipeline = torch.quantization.quantize_dynamic(
    pipeline.unet,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)
quanto_pipeline = quanto_pipeline.to("cpu")

# Generate logits for both original
logits_original = extract_logits(pipeline.unet, pipeline.tokenizer, pipeline.text_encoder, PROMPT, torch.float32)
logits_quantized = extract_logits(quanto_pipeline, pipeline.tokenizer, pipeline.text_encoder, PROMPT, torch.float32)

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((logits_original - logits_quantized) ** 2)

print(f"{mse.item()}")

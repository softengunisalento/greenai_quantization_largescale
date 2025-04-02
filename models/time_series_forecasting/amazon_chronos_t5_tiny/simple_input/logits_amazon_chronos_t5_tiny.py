import pandas as pd
from chronos import ChronosPipeline
import torch

### Constants
MODEL_ID = "amazon/chronos-t5-tiny"
HISTORICAL_DATA = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
DEVICE_MAP = "cpu"

# Load the model and dataset
pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE_MAP, torch_dtype=torch.float32)
df = pd.read_csv(HISTORICAL_DATA)

# Prepare the context data
context = torch.tensor(df["#Passengers"])
prediction_length = 12

# Make forecast with original model
forecast = pipeline.predict(context, prediction_length)

# Apply dynamic quantization to the model
model = pipeline.model  # extract the underlying model
model_quantized = torch.quantization.quantize_dynamic(
    model,  # the model instance
    {torch.nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8  # quantized dtype (8-bit int)
)

# Update the pipeline model with the quantized model
pipeline.model = model_quantized

# Make forecast with quantized model
quantized_forecast = pipeline.predict(context, prediction_length)

# Compute logits (flatten the forecast for MSE calculation)
original_logits = forecast.flatten()
quantized_logits = quantized_forecast.flatten()

# Calculate the mean squared error (MSE) between the logits
mse = torch.mean((original_logits - quantized_logits) ** 2)

print(f"{mse.item()}")

from chronos import ChronosPipeline
import os
import csv
import torch
import pandas as pd


### Constants
MODEL_ID = "amazon/chronos-t5-large"
HISTORICAL_DATA = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"
PREDICTION_LENGTH = 12
DEVICE = "cpu"


### Funtions
def csv_results(mse_values, max_diff_values):
	""" Print MSE statistics """
	mse_series = pd.Series(list(mse_values.values()))

	with open(CSV_FILE, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Record index", "MSE", "Max Difference"])
		
		for idx in mse_values:
			writer.writerow([idx, mse_values[idx], max_diff_values[idx]])
		
		writer.writerow(["Min MSE", "Max MSE", "Mean MSE", "Median MSE", "Standard Deviation"])
		writer.writerow([mse_series.min(), mse_series.max(), mse_series.mean(), mse_series.median(), mse_series.std()])

	print(f"Results saved to {CSV_FILE}")


### Main
def main():
	# Load the model and dataset
	pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.float32)
	df = pd.read_csv(HISTORICAL_DATA)

	# Prepare the context data
	context = torch.tensor(df["#Passengers"])

	# Make forecast with original model
	forecast = pipeline.predict(context, PREDICTION_LENGTH)

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
	quantized_forecast = pipeline.predict(context, PREDICTION_LENGTH)

	mse_values = {}
	max_diff_values = {}

	# Compute logits (flatten the forecast for MSE calculation)
	original_logits = forecast.flatten()
	quantized_logits = quantized_forecast.flatten()

	# Calculate the mean squared error (MSE) between the logits
	mse_values[0] = torch.mean((original_logits - quantized_logits) ** 2).item()
	max_diff_values[0] = torch.max(torch.abs(original_logits - quantized_logits)).item()

	csv_results(mse_values, max_diff_values)

if __name__ == "__main__":
	main()
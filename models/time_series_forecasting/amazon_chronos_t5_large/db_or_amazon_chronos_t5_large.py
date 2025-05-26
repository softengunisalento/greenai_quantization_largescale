from chronos import ChronosPipeline
import os
import csv
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker


### Constants
N_BYTES = 4 # Number of bytes that take each parameter (4=float32, 1=int8)
CARBON_INTENSITY = 475 # Global average carbon intensity
MODEL_ID = "amazon/chronos-t5-large"
HISTORICAL_DATA = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
CSV_FILE = os.path.splitext(os.path.basename(__file__))[0] + ".csv"


### Functions
def evaluate_size(pipeline):
	""" Evaluate the size of the model """
	# calculating the total number of parameters
	total_params = sum(p.numel() for p in pipeline.model.parameters())

	# return bytes
	return total_params * N_BYTES

def calculate_co2_emissions(energy_joules):
	""" Calculate CO2 emissions (in grams) """
	# Convert joules to megajoules (1 MJ = 1e6 J)
	energy_megajoules = energy_joules / 1e6
	return energy_megajoules * CARBON_INTENSITY

def measure_energy_consumption(func):
	""" Decorator to measure energy consumption of a function """
	def wrapper(*args, **kwargs):
		# Start the CodeCarbon tracker
		tracker = EmissionsTracker(
			log_level="error",
			save_to_file=False
		)
		tracker.start()
		
		# Start the inference tracker
		start_time = time.time()

		# Call the function
		result = func(*args, **kwargs)

		# measure the power consumption after inference
		end_time = time.time()
		inference_time = end_time - start_time
		
		# Stop the CodeCarbon tracker
		tracker.stop()
		
		co2_emissions = calculate_co2_emissions(tracker._total_energy.kWh)
		
		# kWh to Wh
		cpu_energy_Wh = tracker._total_cpu_energy.kWh * 1000
		gpu_energy_Wh = tracker._total_gpu_energy.kWh * 1000
		ram_energy_Wh = tracker._total_ram_energy.kWh * 1000
		totale_energy_Wh = tracker._total_energy.kWh * 1000

		with open(CSV_FILE, mode='a', newline='') as file:
			writer = csv.writer(file)
			
			# write stats
			writer.writerow(["Inference s", "CPU Wh", "GPU Wh", "RAM Wh", "Energy Wh", "CO2 Emissions gr"])
			writer.writerow([inference_time,cpu_energy_Wh, gpu_energy_Wh, ram_energy_Wh, totale_energy_Wh, co2_emissions])

		return result
	return wrapper

def csv_results(pipeline, df):
	""" Print statistics """
	# check the memory footprint of the model
	mem_footprint = df.memory_usage(deep=True).sum()

	# checking the size of the model
	module_sizes = evaluate_size(pipeline)

	with open(CSV_FILE, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# write files stats
		writer.writerow(["File Name"])
		writer.writerow([HISTORICAL_DATA])
		
		# write metrics
		writer.writerow(["Footprint MBs", "Size GB"])
		writer.writerow([mem_footprint/1e+6, module_sizes/1e+9])

	print(f"Results saved to {CSV_FILE}")

def arg_commandline():
	""" Create an argument parser object """
	parser = argparse.ArgumentParser(description="argument parser to use custom energy consumption measurements")

	# Add arguments/options
	parser.add_argument('-m', '--mgpu', action='store_true', help='enable custom energy consumption measurement function')

	# Parse the arguments
	return parser.parse_args()


### Main
def main():
	args = arg_commandline()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load the model and dataset
	pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.bfloat16)
	df = pd.read_csv(HISTORICAL_DATA)

	# make a forecast
	context = torch.tensor(df["#Passengers"])
	prediction_length = 12
	forecast = pipeline.predict(context, prediction_length)	# shape [num_series, num_samples, prediction_length]

	# visualize the forecast
	forecast_index = range(len(df), len(df) + prediction_length)
	low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

	plt.figure(figsize=(8, 4))
	plt.plot(df["#Passengers"], color="royalblue", label="historical data")
	plt.plot(forecast_index, median, color="tomato", label="median forecast")
	plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
	plt.legend()
	plt.grid()
	plt.show()

	csv_results(pipeline, df)


if __name__ == "__main__":
	# Check if the -m option is provided
	args = arg_commandline()
	if args.mgpu:
		# If -m is provided, wrap main with energy measurement
		main_with_timing = measure_energy_consumption(main)
		main_with_timing()
	else:
		# Otherwise, just call main directly
		main()
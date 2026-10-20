#!/usr/bin/env python3
"""
Rebuild a combined per-model comparison table (same 61-column shape as the
existing old_csv.csv) from the fresh or_*.csv / qd_*.csv / logits_*.csv files
scattered under models/<task_type>/<model_name>/.

Does NOT touch any of the source or_/qd_/logits_ CSVs or the benchmark
scripts - read-only input, single new output file.

Usage:
    python merge_results.py [models_dir] [old_csv_path] [-o output.csv]
"""

import argparse
import csv
from pathlib import Path

ENERGY_COLS = ["footprint_mb", "sizes_gb", "inference_time_s", "cpu_wh", "gpu_wh", "ram_wh", "total_energy_wh", "co2_emissions_gr"]

# Metric labels whose "mean accuracy" value is an error-like quantity
# (lower is better) and therefore gets inverted: normalized = 100 - mean*100.
# Everything else (topk, miou, cos sim, blank, ...) is similarity-like
# (higher is better): normalized = mean*100.
INVERT_LABELS = {"wer", "mse", "bleu", "blue"}

# Only these labels have a confidently-verified single-row "Min X / Max X /
# Mean X / Median X / Std Dev" (or WER/CER two-row) shape in their
# logits_*.py script. Every other label (topk, blank, or a label not found
# in old_csv.csv at all) is left blank rather than guessed.
CONFIDENT_LABELS = {"wer", "mse", "bleu", "blue", "miou", "cos sim"}


def is_blank_row(row):
	return all(c.strip() == "" for c in row)


def strip_avg_rows(rows):
	"""Drop any (blank row, followed-by average row) pairs appended in-place
	by csv_avg.py, so repeated runs of that tool don't corrupt our means."""
	cleaned = []
	i = 0
	while i < len(rows):
		if is_blank_row(rows[i]):
			i += 2  # skip the blank row and the average row right after it
			continue
		cleaned.append(rows[i])
		i += 1
	return cleaned


def column_means(rows, header):
	"""Column-wise mean over whatever cells parse as float in each column
	independently (mirrors results.py's robust-per-column approach)."""
	means = {}
	for col_idx, col_name in enumerate(header):
		vals = []
		for row in rows:
			if col_idx >= len(row):
				continue
			try:
				vals.append(float(row[col_idx]))
			except ValueError:
				continue
		means[col_name] = (sum(vals) / len(vals)) if vals else None
	return means


def read_energy_csv(csv_path):
	if not csv_path.exists():
		return [], []
	with open(csv_path, newline="") as f:
		all_rows = list(csv.reader(f))
	if not all_rows:
		return [], []
	header, rows = all_rows[0], all_rows[1:]
	return header, strip_avg_rows(rows)


def avg_or(model_dir, or_name):
	header, rows = read_energy_csv(model_dir / f"{or_name}.csv")
	if not rows:
		return {c: None for c in ENERGY_COLS}
	return column_means(rows, header)


def classify_qd_rows(rows, header, or_sizes_gb, model_label=""):
	"""Classify each qd_ row as int8/int4/unknown by its own sizes_gb ratio
	against the model's FP32 baseline, NOT by row position (even/odd).

	Row-position parity (wrapper_rep.py's "-q 4 then -q 8" per repetition)
	is not reliable in practice: bitsandbytes can silently fall back to
	8-bit when 4-bit isn't supported for a given layer/hardware (no error
	raised), and manual top-up runs at different times (different rep
	counts, different sessions) can desync the global alternation for a
	file built from multiple separate wrapper_rep.py invocations.

	bitsandbytes INT8 reports a very consistent ~25% of the FP32 sizes_gb
	(1 byte/param, same element count). INT4 packs 2 values/byte, so its
	ratio varies by architecture (which fraction of params are in
	quantizable Linear layers) - there's no single expected INT4 ratio,
	only "reliably below the INT8 band"."""
	size_idx = header.index("sizes_gb") if "sizes_gb" in header else None
	int4_rows, int8_rows, unknown_rows = [], [], []
	for row in rows:
		ratio = None
		if size_idx is not None and or_sizes_gb:
			try:
				ratio = float(row[size_idx]) / or_sizes_gb * 100.0
			except (ValueError, ZeroDivisionError, IndexError):
				ratio = None
		if ratio is None:
			unknown_rows.append(row)
		elif 22.0 <= ratio <= 28.0:
			int8_rows.append(row)
		elif ratio < 22.0:
			int4_rows.append(row)
		else:
			unknown_rows.append(row)

	# Fallback for architectures where a large non-quantizable fraction
	# (e.g. big embedding tables bitsandbytes never touches) pushes both
	# ratios well above the fixed 22-28% INT8 band - only kicks in when the
	# fixed bands caught nothing at all, and only on a clean, balanced
	# two-value split (smaller size -> int4, larger -> int8).
	if not int4_rows and not int8_rows and unknown_rows and size_idx is not None:
		by_value = {}
		for row in unknown_rows:
			try:
				val = float(row[size_idx])
			except (ValueError, IndexError):
				continue
			by_value.setdefault(val, []).append(row)
		if len(by_value) == 2:
			lo_val, hi_val = sorted(by_value)
			lo_rows, hi_rows = by_value[lo_val], by_value[hi_val]
			if min(len(lo_rows), len(hi_rows)) >= 0.2 * len(unknown_rows):
				int4_rows, int8_rows = lo_rows, hi_rows
				unknown_rows = [r for r in unknown_rows if r not in lo_rows and r not in hi_rows]
				print(f"  NOTE [{model_label}]: ratios ({lo_val/or_sizes_gb*100:.1f}% / {hi_val/or_sizes_gb*100:.1f}%) fall outside the fixed INT8 band but split cleanly into two balanced clusters - classified by relative size (smaller=int4, larger=int8) instead")

	if unknown_rows:
		print(f"  WARNING [{model_label}]: {len(unknown_rows)}/{len(rows)} qd_ rows have an unrecognized sizes_gb ratio (neither ~25% INT8 nor a lower INT4 band) - excluded from both int4/int8 averages")
	if rows and not int4_rows:
		print(f"  WARNING [{model_label}]: 0/{len(rows)} qd_ rows classify as INT4 (all INT8 or unknown) - INT4 quantization likely never actually ran (silent backend fallback)")
	return int4_rows, int8_rows, unknown_rows


def avg_qd(model_dir, qd_name, or_sizes_gb, model_label=""):
	"""Split rows into int4/int8 by measured size ratio (see
	classify_qd_rows), not by row position."""
	header, rows = read_energy_csv(model_dir / f"{qd_name}.csv")
	if not rows:
		empty = {c: None for c in ENERGY_COLS}
		return dict(empty), dict(empty)
	int4_rows, int8_rows, _ = classify_qd_rows(rows, header, or_sizes_gb, model_label)
	return column_means(int4_rows, header), column_means(int8_rows, header)


def find_mean_in_stats_row(rows, header):
	"""Single-row 'Min X,Max X,Mean X,Median X,...' shape: grab the cell
	under whichever header column name starts with 'Mean'."""
	if not rows or not header:
		return None
	mean_idx = next((i for i, h in enumerate(header) if h.strip().lower().startswith("mean")), None)
	if mean_idx is None or mean_idx >= len(rows[0]):
		return None
	try:
		return float(rows[0][mean_idx])
	except ValueError:
		return None


def find_mean_in_metric_row(rows, header, metric_name):
	"""Two-row 'Metric,Min,Max,Mean,Median,Std Dev' shape (WER/CER style):
	find the row whose first cell matches metric_name, grab its Mean cell."""
	if not rows or not header:
		return None
	mean_idx = next((i for i, h in enumerate(header) if h.strip().lower() == "mean"), None)
	if mean_idx is None:
		return None
	for row in rows:
		if row and row[0].strip().upper() == metric_name:
			try:
				return float(row[mean_idx])
			except (ValueError, IndexError):
				return None
	return None


def split_into_run_blocks(all_rows, is_header_row):
	blocks = []
	current_header, current_rows = None, []
	for row in all_rows:
		if row and is_header_row(row):
			if current_header is not None:
				blocks.append((current_header, current_rows))
			current_header, current_rows = row, []
		elif current_header is not None:
			current_rows.append(row)
	if current_header is not None:
		blocks.append((current_header, current_rows))
	return blocks


def extract_mean_accuracy(logits_path, label):
	"""Returns (mean_int8, mean_int4) or (None, None) if not confidently
	extractable for this label/shape. Block 0 of the logits CSV corresponds
	to the default single run (-q 8); block 1, if present, is a later -q 4
	run via wrapper_rep_logits.py."""
	if label not in CONFIDENT_LABELS or not logits_path.exists():
		return None, None

	with open(logits_path, newline="") as f:
		all_rows = list(csv.reader(f))
	if not all_rows:
		return None, None

	if label == "wer":
		blocks = split_into_run_blocks(all_rows, lambda row: row[0] == "Metric")
		means = [find_mean_in_metric_row(rows, header, "WER") for header, rows in blocks]
	else:
		blocks = split_into_run_blocks(all_rows, lambda row: row[0].strip().lower().startswith("min"))
		means = [find_mean_in_stats_row(rows, header) for header, rows in blocks]

	mean_int8 = means[0] if len(means) > 0 else None
	mean_int4 = means[1] if len(means) > 1 else None
	return mean_int8, mean_int4


def normalize(mean_value, label):
	if mean_value is None:
		return None
	return (100.0 - mean_value * 100.0) if label in INVERT_LABELS else (mean_value * 100.0)


def pct(a, b):
	if a is None or b is None or b == 0:
		return None
	return a / b * 100.0


def diff(a, b):
	if a is None or b is None:
		return None
	return a - b


def fmt(value, is_pct=False):
	if value is None:
		return ""
	return f"{value:.4f}%" if is_pct else repr(value)


def load_old_csv_labels(old_csv_path):
	labels = {}
	if not old_csv_path.exists():
		return labels
	with open(old_csv_path, newline="") as f:
		rows = list(csv.reader(f))
	if not rows:
		return labels
	header = rows[0]
	name_idx = header.index("Model Name")
	metric_idx = header.index("acc metrics")
	for row in rows[1:]:
		if len(row) > max(name_idx, metric_idx):
			labels[row[name_idx]] = row[metric_idx].strip()
	return labels


def get_model_id(or_script_path):
	text = or_script_path.read_text(errors="ignore")
	for line in text.splitlines():
		line = line.strip()
		if line.startswith("MODEL_ID"):
			_, _, rhs = line.partition("=")
			return rhs.split("#")[0].strip().strip('"').strip("'")
	return None


def build_row(model_dir, old_labels):
	or_scripts = list(model_dir.glob("or_*.py"))
	if not or_scripts:
		return None
	or_script = or_scripts[0]
	base = or_script.stem[len("or_"):]
	qd_name = f"qd_{base}"
	logits_name = f"logits_{base}"

	model_id = get_model_id(or_script)
	if model_id is None:
		return None

	task = model_dir.parent.name.replace("_", "-")

	or_avg = avg_or(model_dir, or_script.stem)
	int4_avg, int8_avg = avg_qd(model_dir, qd_name, or_avg.get("sizes_gb"), model_label=model_id)

	or_size, int8_size, int4_size = or_avg.get("sizes_gb"), int8_avg.get("sizes_gb"), int4_avg.get("sizes_gb")
	or_fp, int8_fp, int4_fp = or_avg.get("footprint_mb"), int8_avg.get("footprint_mb"), int4_avg.get("footprint_mb")
	or_inf, int8_inf, int4_inf = or_avg.get("inference_time_s"), int8_avg.get("inference_time_s"), int4_avg.get("inference_time_s")
	or_cpu, int8_cpu, int4_cpu = or_avg.get("cpu_wh"), int8_avg.get("cpu_wh"), int4_avg.get("cpu_wh")
	or_gpu, int8_gpu, int4_gpu = or_avg.get("gpu_wh"), int8_avg.get("gpu_wh"), int4_avg.get("gpu_wh")
	or_ram, int8_ram, int4_ram = or_avg.get("ram_wh"), int8_avg.get("ram_wh"), int4_avg.get("ram_wh")
	or_energy, int8_energy, int4_energy = or_avg.get("total_energy_wh"), int8_avg.get("total_energy_wh"), int4_avg.get("total_energy_wh")
	or_co2, int8_co2, int4_co2 = or_avg.get("co2_emissions_gr"), int8_avg.get("co2_emissions_gr"), int4_avg.get("co2_emissions_gr")

	param = (or_size * 1e9 / 4.0) if or_size is not None else None

	label = old_labels.get(model_id, "")
	mean_int8, mean_int4 = extract_mean_accuracy(model_dir / f"{logits_name}.csv", label)
	norm_int8, norm_int4 = normalize(mean_int8, label), normalize(mean_int4, label)

	co2_emitted_int8 = pct(int8_co2, or_co2)
	co2_emitted_int4 = pct(int4_co2, or_co2)
	size_pct_int8 = pct(int8_size, or_size)
	size_pct_int4 = pct(int4_size, or_size)
	inf_pct_int8 = pct(int8_inf, or_inf)
	inf_pct_int4 = pct(int4_inf, or_inf)
	inf_pctchg_int8 = pct(diff(int8_inf, or_inf), or_inf)
	inf_pctchg_int4 = pct(diff(int4_inf, int8_inf), int8_inf)

	# Returned as a flat list, in exact column order - HEADER has two
	# genuinely-duplicate-named "Inference s %" column pairs (matching
	# old_csv.csv's own duplication), so a dict keyed by name can't hold it.
	return [
		model_id,
		task,
		"",  # db link - not reliably derivable from the scripts, left blank
		fmt(param),
		label,
		fmt(norm_int8, is_pct=True), fmt(norm_int4, is_pct=True),
		fmt(mean_int8), fmt(mean_int4),
		fmt(pct(int8_fp, or_fp)), fmt(pct(int4_fp, int8_fp)),
		fmt(or_fp), fmt(int8_fp), fmt(int4_fp),
		fmt(100.0 - size_pct_int8, is_pct=True) if size_pct_int8 is not None else "",
		fmt(100.0 - size_pct_int4, is_pct=True) if size_pct_int4 is not None else "",
		fmt(size_pct_int8, is_pct=True), fmt(size_pct_int4, is_pct=True),
		fmt(diff(int8_size, or_size)), fmt(diff(int4_size, int8_size)),
		fmt(or_size), fmt(int8_size), fmt(int4_size),
		fmt(inf_pct_int8, is_pct=True), fmt(inf_pct_int4, is_pct=True),
		fmt(inf_pctchg_int8, is_pct=True), fmt(inf_pctchg_int4, is_pct=True),
		fmt(or_inf), fmt(int8_inf), fmt(int4_inf),
		fmt(diff(int8_cpu, or_cpu)), fmt(diff(int4_cpu, int8_cpu)),
		fmt(or_cpu), fmt(int8_cpu), fmt(int4_cpu),
		fmt(diff(int8_gpu, or_gpu)), fmt(diff(int4_gpu, int8_gpu)),
		fmt(or_gpu), fmt(int8_gpu), fmt(int4_gpu),
		fmt(diff(int8_ram, or_ram)), fmt(diff(int4_ram, int8_ram)),
		fmt(or_ram), fmt(int8_ram), fmt(int4_ram),
		fmt(pct(int8_energy, or_energy), is_pct=True), fmt(pct(int4_energy, or_energy), is_pct=True),
		fmt(diff(int8_energy, or_energy)), fmt(diff(int4_energy, or_energy)),
		fmt(or_energy), fmt(int8_energy), fmt(int4_energy),
		fmt(100.0 - co2_emitted_int8, is_pct=True) if co2_emitted_int8 is not None else "",
		fmt(100.0 - co2_emitted_int4, is_pct=True) if co2_emitted_int4 is not None else "",
		fmt(co2_emitted_int8, is_pct=True), fmt(co2_emitted_int4, is_pct=True),
		fmt(diff(int8_co2, or_co2)), fmt(diff(int4_co2, or_co2)),
		fmt(or_co2), fmt(int8_co2), fmt(int4_co2),
	]


HEADER = [
	"Model Name", "First Task", "db \U0001F917", "param", "acc metrics",
	"normalized int8", "normalized int4", "mean accuracy int8", "mean accuracy int4",
	"Footprint MB % int8", "Footprint MB % int4", "or footprint", "int8 footprint", "int4 footprint",
	"% Size saved int8", "% Size saved int4", "% Size int8", "% Size int4",
	"Size GB int8", "Size GB int4", "or size", "int8 size", "int4 size",
	"Inference s % int8", "Inference s % int4", "Inference s % int8", "Inference s % int4",
	"or inference", "int8 inference", "int4 inference",
	"CPU Wh int8", "CPU Wh int4", "or cpu", "int8 cpu", "int4 cpu",
	"GPU Wh int8", "GPU Wh int4", "or gpu", "int8 gpu", "int4 gpu",
	"RAM Wh int8", "RAM Wh int4", "or ram", "int8 ram", "int4 ram",
	"% Energy Wh int8", "% Energy Wh int4", "Energy Wh int8", "Energy Wh int4",
	"or energy", "int8 energy", "int4 energy",
	"% CO2 saved int8", "% CO2 saved int4", "% CO2 emitted int8", "% CO2 emitted int4",
	"CO2 g int8", "CO2 g int4", "or CO2", "int8 CO2", "int4 CO2",
]


def main():
	parser = argparse.ArgumentParser(description="Rebuild old_csv.csv-shaped comparison table from fresh or_/qd_/logits_ CSVs")
	parser.add_argument("models_dir", nargs="?", default="models")
	parser.add_argument("old_csv", nargs="?", default="old_csv.csv")
	parser.add_argument("-o", "--output", default="merged_results.csv")
	args = parser.parse_args()

	models_dir = Path(args.models_dir)
	old_labels = load_old_csv_labels(Path(args.old_csv))

	rows = []
	for or_script in sorted(models_dir.rglob("or_*.py")):
		row = build_row(or_script.parent, old_labels)
		if row is not None:
			assert len(row) == len(HEADER), f"column count mismatch for {or_script}: {len(row)} vs {len(HEADER)}"
			rows.append(row)

	with open(args.output, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(HEADER)
		writer.writerows(rows)

	print(f"Wrote {len(rows)} model rows to {args.output}")


if __name__ == "__main__":
	main()

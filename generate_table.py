#!/usr/bin/env python3
"""
Recompute the quantization significance table (table.tex) from fresh
or_*.csv / qd_*.csv data under models/<task_type>/<model_name>/.

Per model, per quantization level (int8, int4), compares the raw
per-repetition `total_energy_wh` values: FP32 (or_*.csv) vs quantized
(qd_*.csv, even/odd split per wrapper_rep.py's -q4-then-q8 convention).

- Both sides have >=2 values -> "repeated executions" bucket: Mann-Whitney U
  test (normal approximation, two-sided, p<0.05 = significant) + Cliff's
  delta effect size.
- Both sides have exactly 1 value -> "single execution" bucket: descriptive
  %-of-FP32 only, no test possible.
- Either side has 0 values -> excluded, reported separately.

Read-only on all CSVs; does not touch benchmark scripts or source data.

Usage:
    python generate_table.py [models_dir] [-o table.tex]
"""

import argparse
import csv
import math
from pathlib import Path

ENERGY_COL = "total_energy_wh"

# time_series_forecasting has no macroclass in the original table (no
# "Time Series" column exists) - excluded from this analysis, reported
# separately rather than silently folded into another class.
TASK_MACROCLASS = {
	"automatic_speech_recognition": "Speech Processing",
	"voice_activity_detection": "Speech Processing",
	"feature_extraction": "Feature Engineering",
	"sentence_similarity": "Feature Engineering",
	"image_feature_extraction": "Feature Engineering",
	"text_generation": "Sequence Transformation",
	"translation": "Sequence Transformation",
	"fill_mask": "Sequence Transformation",
	"token_classification": "Sequence Transformation",
	"image_classification": "Computer Vision",
	"image_segmentation": "Computer Vision",
	"zero_shot_image_classification": "Computer Vision",
}
MACROCLASS_ORDER = ["Speech Processing", "Feature Engineering", "Sequence Transformation", "Computer Vision"]


def is_blank_row(row):
	return all(c.strip() == "" for c in row)


def strip_avg_rows(rows):
	cleaned = []
	i = 0
	while i < len(rows):
		if is_blank_row(rows[i]):
			i += 2
			continue
		cleaned.append(rows[i])
		i += 1
	return cleaned


def read_energy_column(csv_path):
	"""Returns list of float total_energy_wh values, or [] if missing."""
	if not csv_path.exists():
		return []
	with open(csv_path, newline="") as f:
		all_rows = list(csv.reader(f))
	if not all_rows:
		return []
	header, rows = all_rows[0], strip_avg_rows(all_rows[1:])
	if ENERGY_COL not in header:
		return []
	idx = header.index(ENERGY_COL)
	vals = []
	for row in rows:
		if idx >= len(row):
			continue
		try:
			vals.append(float(row[idx]))
		except ValueError:
			continue
	return vals


def read_qd_energy_columns(csv_path):
	"""Returns (int4_vals, int8_vals) via the even/odd row convention."""
	if not csv_path.exists():
		return [], []
	with open(csv_path, newline="") as f:
		all_rows = list(csv.reader(f))
	if not all_rows:
		return [], []
	header, rows = all_rows[0], strip_avg_rows(all_rows[1:])
	if ENERGY_COL not in header:
		return [], []
	idx = header.index(ENERGY_COL)
	int4_rows, int8_rows = rows[0::2], rows[1::2]

	def extract(rs):
		vals = []
		for row in rs:
			if idx >= len(row):
				continue
			try:
				vals.append(float(row[idx]))
			except ValueError:
				continue
		return vals

	return extract(int4_rows), extract(int8_rows)


def median(vals):
	s = sorted(vals)
	n = len(s)
	if n == 0:
		return None
	mid = n // 2
	return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def percentile(vals, p):
	"""Linear-interpolation percentile, matching numpy's default."""
	s = sorted(vals)
	n = len(s)
	if n == 0:
		return None
	if n == 1:
		return s[0]
	rank = p / 100.0 * (n - 1)
	lo = math.floor(rank)
	hi = math.ceil(rank)
	if lo == hi:
		return s[int(rank)]
	frac = rank - lo
	return s[lo] * (1 - frac) + s[hi] * frac


def mann_whitney_u(x, y):
	"""Two-sided Mann-Whitney U test, normal approximation with tie
	correction and continuity correction. Returns (U, p_value)."""
	n1, n2 = len(x), len(y)
	combined = [(v, 0) for v in x] + [(v, 1) for v in y]
	combined.sort(key=lambda t: t[0])

	ranks = [0.0] * len(combined)
	i = 0
	while i < len(combined):
		j = i
		while j < len(combined) and combined[j][0] == combined[i][0]:
			j += 1
		avg_rank = (i + 1 + j) / 2.0  # 1-indexed rank average over the tie
		for k in range(i, j):
			ranks[k] = avg_rank
		i = j

	rank_sum_x = sum(r for r, (v, grp) in zip(ranks, combined) if grp == 0)
	u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
	u2 = n1 * n2 - u1
	u = min(u1, u2)

	mean_u = n1 * n2 / 2.0

	# tie correction term
	tie_groups = {}
	for v, _ in combined:
		tie_groups[v] = tie_groups.get(v, 0) + 1
	n_total = n1 + n2
	tie_term = sum(t ** 3 - t for t in tie_groups.values())
	var_u = (n1 * n2 / 12.0) * ((n_total + 1) - tie_term / (n_total * (n_total - 1))) if n_total > 1 else 0

	if var_u <= 0:
		return u, 1.0

	std_u = math.sqrt(var_u)
	z = (u - mean_u + (0.5 if u < mean_u else -0.5)) / std_u  # continuity correction
	p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
	p = max(0.0, min(1.0, p))
	return u, p


def cliffs_delta(x, y):
	"""δ = (#(x_i>y_j) - #(x_i<y_j)) / (n*m). Positive => x tends larger."""
	n, m = len(x), len(y)
	if n == 0 or m == 0:
		return None
	gt = lt = 0
	for xi in x:
		for yj in y:
			if xi > yj:
				gt += 1
			elif xi < yj:
				lt += 1
	return (gt - lt) / (n * m)


def classify_model(fp32_vals, quant_vals, alpha=0.05):
	"""Returns dict with bucket ('repeated'/'single'/'none'), pct_median,
	significance ('red'/'neutral'/'inc'/None), abs_delta (or None)."""
	n_fp32, n_quant = len(fp32_vals), len(quant_vals)
	if n_fp32 == 0 or n_quant == 0:
		return {"bucket": "none"}

	med_fp32, med_quant = median(fp32_vals), median(quant_vals)
	pct = (med_quant / med_fp32 * 100.0) if med_fp32 else None

	if n_fp32 >= 2 and n_quant >= 2:
		_, p = mann_whitney_u(fp32_vals, quant_vals)
		delta = cliffs_delta(fp32_vals, quant_vals)
		if p < alpha:
			sig = "red" if med_quant < med_fp32 else "inc"
		else:
			sig = "neutral"
		return {"bucket": "repeated", "pct_median": pct, "sig": sig, "abs_delta": abs(delta) if delta is not None else None}
	else:
		return {"bucket": "single", "pct_median": pct}


def get_model_id(or_script_path):
	text = or_script_path.read_text(errors="ignore")
	for line in text.splitlines():
		line = line.strip()
		if line.startswith("MODEL_ID"):
			_, _, rhs = line.partition("=")
			return rhs.split("#")[0].strip().strip('"').strip("'")
	return None


def collect_models(models_dir):
	"""Returns list of dicts: {model_id, task, macroclass, or_vals,
	int4_vals, int8_vals} for every model under a mapped task folder."""
	results = []
	excluded_unmapped = []
	for or_script in sorted(models_dir.rglob("or_*.py")):
		model_dir = or_script.parent
		task = model_dir.parent.name
		base = or_script.stem[len("or_"):]
		model_id = get_model_id(or_script) or base

		if task not in TASK_MACROCLASS:
			excluded_unmapped.append((model_id, task))
			continue

		or_vals = read_energy_column(model_dir / f"{or_script.stem}.csv")
		int4_vals, int8_vals = read_qd_energy_columns(model_dir / f"qd_{base}.csv")

		results.append({
			"model_id": model_id,
			"task": task,
			"macroclass": TASK_MACROCLASS[task],
			"or_vals": or_vals,
			"int4_vals": int4_vals,
			"int8_vals": int8_vals,
		})
	return results, excluded_unmapped


def build_subtable(models, quant_key, bucket_filter):
	"""For a given quant level ('int8_vals'/'int4_vals') and bucket
	('repeated'/'single'), returns per-macroclass aggregate dict."""
	by_class = {mc: [] for mc in MACROCLASS_ORDER}
	totals_by_class = {mc: 0 for mc in MACROCLASS_ORDER}

	for m in models:
		mc = m["macroclass"]
		totals_by_class[mc] += 1
		result = classify_model(m["or_vals"], m[quant_key])
		if result["bucket"] == bucket_filter:
			by_class[mc].append(result)

	rows = []
	grand_pct, grand_red, grand_neutral, grand_inc, grand_delta, grand_n, grand_total = [], 0, 0, 0, [], 0, 0
	for mc in MACROCLASS_ORDER:
		entries = by_class[mc]
		n = len(entries)
		total = totals_by_class[mc]
		pcts = [e["pct_median"] for e in entries if e["pct_median"] is not None]
		row = {"macroclass": mc, "n": n, "total": total, "pcts": pcts}
		if bucket_filter == "repeated":
			reds = sum(1 for e in entries if e["sig"] == "red")
			neutrals = sum(1 for e in entries if e["sig"] == "neutral")
			incs = sum(1 for e in entries if e["sig"] == "inc")
			deltas = [e["abs_delta"] for e in entries if e["abs_delta"] is not None]
			row.update({"red": reds, "neutral": neutrals, "inc": incs, "deltas": deltas})
			grand_red += reds
			grand_neutral += neutrals
			grand_inc += incs
			grand_delta.extend(deltas)
		rows.append(row)
		grand_pct.extend(pcts)
		grand_n += n
		grand_total += total

	grand = {"n": grand_n, "total": grand_total, "pcts": grand_pct}
	if bucket_filter == "repeated":
		grand.update({"red": grand_red, "neutral": grand_neutral, "inc": grand_inc, "deltas": grand_delta})
	return rows, grand


def fmt_pct_iqr(pcts):
	if not pcts:
		return "--"
	med = median(pcts)
	q1, q3 = percentile(pcts, 25), percentile(pcts, 75)
	return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def fmt_delta(deltas):
	if not deltas:
		return "--"
	return f"{median(deltas):.2f}"


def render_latex(int8_repeated, int8_repeated_grand, int8_single, int8_single_grand,
                  int4_repeated, int4_repeated_grand, int4_single, int4_single_grand):
	lines = []
	lines.append(r"\begin{tabular}{|l|c|c|c|c|c|c|}")
	lines.append(r"\hline")
	lines.append(r"\multirow{2}{*}{\textbf{Task macroclass}} & \multirow{2}{*}{\textbf{Models}} & \multirow{2}{*}{\textbf{\makecell{Median energy\\(\% FP32) [IQR]}}} & \multicolumn{3}{c|}{\textbf{Outcome}} & \multirow{2}{*}{\textbf{\makecell{Median\\$|\delta|$}}} \\")
	lines.append(r"\cline{4-6}")
	lines.append(r" & & & \textbf{\makecell{Sig.\\red.}} & \textbf{Neutral} & \textbf{\makecell{Sig.\\inc.}} & \\")
	lines.append(r"\hline")

	lines.append(r"\multicolumn{7}{|c|}{\textit{(a) INT8 vs. FP32 --- repeated executions, statistical comparison}} \\")
	lines.append(r"\hline")
	for row in int8_repeated:
		lines.append(f"{row['macroclass']:<24} & {row['n']} / {row['total']}   & {fmt_pct_iqr(row['pcts'])} & {row['red']}  & {row['neutral']} & {row['inc']} & {fmt_delta(row['deltas'])} \\\\")
	lines.append(r"\hline")
	g = int8_repeated_grand
	lines.append(f"\\textbf{{All}} & \\textbf{{{g['n']} / {g['total']}}} & \\textbf{{{fmt_pct_iqr(g['pcts'])}}} & \\textbf{{{g['red']}}} & \\textbf{{{g['neutral']}}} & \\textbf{{{g['inc']}}} & \\textbf{{{fmt_delta(g['deltas'])}}} \\\\")
	lines.append(r"\hline")

	lines.append(r"\multicolumn{7}{|c|}{\textit{(b) INT8 vs. FP32 --- single execution, descriptive only}} \\")
	lines.append(r"\hline")
	for row in int8_single:
		lines.append(f"{row['macroclass']:<24} & {row['n']} / {row['total']}  & {fmt_pct_iqr(row['pcts'])} & \\multicolumn{{4}}{{c|}}{{}} \\\\")
	lines.append(r"\hline")
	g = int8_single_grand
	lines.append(f"\\textbf{{All}} & \\textbf{{{g['n']} / {g['total']}}} & \\textbf{{{fmt_pct_iqr(g['pcts'])}}} & \\multicolumn{{4}}{{c|}}{{}} \\\\")
	lines.append(r"\hline")

	lines.append(r"\multicolumn{7}{|c|}{\textit{(c) INT4 vs. FP32 --- repeated executions, statistical comparison}} \\")
	lines.append(r"\hline")
	for row in int4_repeated:
		lines.append(f"{row['macroclass']:<24} & {row['n']} / {row['total']}   & {fmt_pct_iqr(row['pcts'])} & {row['red']} & {row['neutral']} & {row['inc']} & {fmt_delta(row['deltas'])} \\\\")
	lines.append(r"\hline")
	g = int4_repeated_grand
	lines.append(f"\\textbf{{All}} & \\textbf{{{g['n']} / {g['total']}}} & \\textbf{{{fmt_pct_iqr(g['pcts'])}}} & \\textbf{{{g['red']}}} & \\textbf{{{g['neutral']}}} & \\textbf{{{g['inc']}}} & \\textbf{{{fmt_delta(g['deltas'])}}} \\\\")
	lines.append(r"\hline")

	lines.append(r"\multicolumn{7}{|c|}{\textit{(d) INT4 vs. FP32 --- single execution, descriptive only}} \\")
	lines.append(r"\hline")
	for row in int4_single:
		lines.append(f"{row['macroclass']:<24} & {row['n']} / {row['total']}  & {fmt_pct_iqr(row['pcts'])} & \\multicolumn{{4}}{{c|}}{{}} \\\\")
	lines.append(r"\hline")
	g = int4_single_grand
	lines.append(f"\\textbf{{All}} & \\textbf{{{g['n']} / {g['total']}}} & \\textbf{{{fmt_pct_iqr(g['pcts'])}}} & \\multicolumn{{4}}{{c|}}{{}} \\\\")
	lines.append(r"\hline")
	lines.append(r"\end{tabular}")
	lines.append(r"\end{table*}")
	return "\n".join(lines)


def main():
	parser = argparse.ArgumentParser(description="Recompute table.tex from fresh or_/qd_ CSV data")
	parser.add_argument("models_dir", nargs="?", default="models")
	parser.add_argument("-o", "--output", default="table.tex")
	args = parser.parse_args()

	models_dir = Path(args.models_dir)
	models, excluded_unmapped = collect_models(models_dir)

	none_bucket = []
	for m in models:
		for quant_key, label in (("int8_vals", "int8"), ("int4_vals", "int4")):
			if classify_model(m["or_vals"], m[quant_key])["bucket"] == "none":
				none_bucket.append((m["model_id"], label))

	int8_rep, int8_rep_g = build_subtable(models, "int8_vals", "repeated")
	int8_sgl, int8_sgl_g = build_subtable(models, "int8_vals", "single")
	int4_rep, int4_rep_g = build_subtable(models, "int4_vals", "repeated")
	int4_sgl, int4_sgl_g = build_subtable(models, "int4_vals", "single")

	latex = render_latex(int8_rep, int8_rep_g, int8_sgl, int8_sgl_g,
	                      int4_rep, int4_rep_g, int4_sgl, int4_sgl_g)

	Path(args.output).write_text(latex + "\n")

	print(f"Models mapped into macroclasses: {len(models)}")
	if excluded_unmapped:
		print(f"Excluded (no macroclass mapping, e.g. time_series_forecasting): {excluded_unmapped}")
	if none_bucket:
		print(f"Zero-data exclusions (model, quant_level) with no usable rows at all: {none_bucket}")
	print(f"Wrote {args.output}")


if __name__ == "__main__":
	main()

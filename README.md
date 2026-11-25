# Post-Training Quantization for Green AI

This academic thesis investigates the impact of Deep Learning model compression
techniques, specifically **post-training quantization (PTQ)**, on both model
efficiency and environmental sustainability. The experiments apply the
quantization back-end that the Hugging Face Transformers library exposes by
default (**bitsandbytes**) — `LLM.int8()` at eight bits, and 4-bit FP4 with
block-wise scaling of the weights — to **43 publicly available models covering
12 tasks**, grouped into four macroclasses: speech processing, feature
engineering, sequence transformation, and computer vision. Model size, output
fidelity with respect to the FP32 baseline, and energy consumption (estimated
with **CodeCarbon**) are measured over 30 repeated executions per
configuration.

### News
This paper is in review at Springer Computing.

---

## Usage

Clone the repository and install:

```bash
git clone https://anonymous.4open.science/r/greenai_quantization_largescale-CB10
cd greenai_quantization_largescale-CB10
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Repository layout

Each benchmarked model lives in `models/<task>/<model>/` with three scripts:

* `logits_<model>.py` — computes the output fidelity (divergence from the FP32
  outputs with the measure suited to the task);
* `or_<model>.py` — runs the FP32 (original precision) baseline;
* `qd_<model>.py` — runs the quantized variant (`-q 8` for INT8, `-q 4` for
  FP4).

With `-m` enabled, every executed repetition is recorded with its inference
time, energy consumption (CPU/GPU/RAM, via CodeCarbon) and model footprint,
and appended to the script's CSV. Example:

```bash
cd models/automatic_speech_recognition/distil_whisper_distil_large_v3_5
python3 or_distil_whisper_distil_large_v3_5.py -m
python3 qd_distil_whisper_distil_large_v3_5.py -m -q 4
python3 qd_distil_whisper_distil_large_v3_5.py -m -q 8
```

The quantization is the standard Hugging Face loading call, as in every
released script:

```python
if args.quantization_bits == 4:
  model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    device_map=device
  )
elif args.quantization_bits == 8:
  model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,
    device_map=device
  )
```

---

## Performance Benchmarks

### Test Environment

* **CPU**: AMD Ryzen 9 7900X (12 cores / 24 threads)
* **GPU**: NVIDIA GeForce RTX 4090 (AD102), driver 535.230.02
* **Software**: Python, PyTorch, Hugging Face Transformers, bitsandbytes
* **Energy estimation**: CodeCarbon

### Results Summary

* **Model size** — 75% reduction at INT8 for every model, median of 83.5% at
  FP4.
* **Energy** — 30 of the 43 models save significantly at INT8 (median 84.63%
  of the FP32 baseline); FP4 costs on average 3.1% more than INT8, with the
  penalty concentrated on encoder and embedding workloads.
* **Output fidelity** — the average INT8 fidelity loss of the most resilient
  tasks is 0.23%; FP4 costs about 18% on average.

The per-run raw values are in the CSV files under `models/`; the aggregate
statistical comparison per task macroclass is in `table.tex`, regenerated
with `generate_table.py`. The merged per-model table is rebuilt with
`merge_results.py` (`huggingface_db_models.csv`).

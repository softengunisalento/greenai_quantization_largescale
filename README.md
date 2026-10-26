# Quantization Large-scale for AI Models

This academic thesis investigates the impact of Deep Learning model compression techniques, specifically **post-training** quantization to INT8 and INT4 precision, on both model efficiency and environmental sustainability. The research analyzes a broad selection of models across various tasks like speech processing, computer vision, and text generation, using a GPU-accelerated setup. Key findings highlight the trade-offs between accuracy, model size, inference speed, and energy consumption, revealing that INT4 compression, while reducing model size, can sometimes increase energy usage depending on hardware optimization, such as the NVIDIA RTX 4090 favoring INT8 operations. The study underscores the crucial relationship between hardware capabilities and the effectiveness of quantization techniques in achieving Green AI objectives.

---

## Usage

Clone the repository and install manually:

```bash
git clone https://github.com/lonelygunter/greenai
cd quantization-toolkit
```

Create a venv and than:

```bash
pip install -r requirements.txt
```

To run the python scripts:

```bash
python3 logits_distil_whisper_distil_large_v3_5.py
python3 or_distil_whisper_distil_large_v3_5.py -m
python3 qd_distil_whisper_distil_large_v3_5.py -m -q 4
python3 qd_distil_whisper_distil_large_v3_5.py -m -q 8
```


Here's a simple example of applying PTQ to a PyTorch model:

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
else:
  raise ValueError("Only 4-bit and 8-bit quantization are supported.")
```

---

## Supported Frameworks

* PyTorch
* TensorFlow

---

## Quantization Methods Implemented

* **Post-Training Quantization (PTQ)**

  * Dynamic quantization
  * Per-layer quantization

---

## Performance Benchmarks

The evaluation of model quantization highlights significant improvements in model efficiency and environmental impact, especially in terms of size reduction, inference speed, and energy consumption, though not without trade-offs in accuracy.

### Test Environment

* **CPU**: AMD Ryzen 9 7900X
* **GPU**: NVIDIA GeForce RTX 4090
* **Frameworks**: PyTorch 3.13

---

## Results Summary

*Figures below are size/inference-time/energy as a percentage of the FP32 baseline
(100% = no change, measured on the Test Environment above), pulled directly from
`merged_results.csv`. Per-model accuracy/MSE figures are not included here — the
model→metric-label mapping (`old_csv.csv`) used to build them was lost and has not
been rebuilt; see the full `table.tex` for the aggregate, per-macroclass statistical
comparison instead.*

### Feature Extraction and Image Classification

**Model size** shrinks predictably under quantization — INT8 always lands at exactly
25% of FP32 (1 byte/param vs. 4), INT4 varies by architecture (how much of the model
sits in quantizable `nn.Linear` layers). `microsoft/resnet-50` is the extreme case:
being almost entirely convolutional, only ~24% of its size is recoverable even under
INT4, versus the 75–85% typical of Linear-heavy transformer models.

| Model                  | Tasks | INT8 Size | INT4 Size |
| ----------------------- | ----- | --------- | --------- |
| BAAI/bge-base-en-v1.5  | FE    | 25.00%    | 15.30%    |
| BAAI/bge-small-en-v1.5 | FE    | 25.00%    | 17.04%    |
| microsoft/resnet-50    | IC    | 25.00%    | 24.00%    |

Size reduction does **not** translate into faster or greener inference on this setup,
though — see below.

### Sentence Transformers and Energy Impact

Energy impact is mixed and model-dependent, not uniformly favorable. `BAAI/bge-base-en-v1.5`
genuinely benefits (INT8 drops energy to 62% of FP32); most others show energy roughly
flat to modestly *higher* under quantization, and `microsoft/resnet-50` is a clear
outlier at 4.4× FP32 energy under both INT8 and INT4.

| Model             | INT8 Energy % | INT4 Energy % |
| ----------------- | -------------- | -------------- |
| all-MiniLM-L6-v2  | 111.14%       | 111.89%       |
| all-mpnet-base-v2 | 89.57%        | 102.75%       |

### Speed of Inference

Inference speed is the sharpest counter-intuitive result in this dataset: quantization
frequently makes inference **slower**, not faster, on the RTX 4090 test rig —
consistent with this project's own thesis framing (INT4/INT8 kernel support is
hardware-dependent, and this GPU is optimized for INT8 tensor-core paths, not every
bitsandbytes quantized-layer kernel). `microsoft/resnet-50` is the most dramatic case:
INT8 inference takes **5.85×** FP32's time, INT4 **5.92×** — both clear regressions,
not improvements. `facebook/opt-125m` similarly regresses (INT8 at 4.94× FP32 time,
INT4 at 2.44×, better than INT8 but still far slower than unquantized). Smaller
sentence-transformer models are closer to parity (within ~5–12% either direction).

### Size vs Performance

The net picture: **size reduction is the one consistently-delivered benefit of
quantization in this benchmark** — energy and inference-time gains are the exception,
not the rule, and several models (especially Conv-heavy vision models like resnet-50)
see both energy and speed get measurably *worse* under both INT8 and INT4. Models
dominated by `nn.Linear` layers (sentence-transformer/embedding architectures) see the
largest, most dependable size reductions (75–85%); whether that is worth deploying
depends on whether size or runtime cost is the actual constraint for a given use case.
# Deep Learning Model Quantization for Green AI 



## Methodology

The research methodology is based on recent advancements in quantization and calibration techniques. A constraint was applied to use only **post-training models**, meaning **post-training quantization (PTQ)** was the chosen technique. This approach offers greater generalization as compression techniques can be applied to any model without retraining, saving time and electrical energy.

Reducing the number of bits in model representation results in lower memory usage, reduced energy consumption, and faster matrix multiplication operations using integer arithmetic. This also allows models to be run on devices with limited resources, such as edge devices and smartphones.

The applied methodology consists of four main steps:
1.  **Model Selection:** Identifying widely used models.
2.  **Model Compression:** Applying quantization techniques.
3.  **Inference Execution:** Running inference on compressed models.
4.  **Measurement Collection:** Gathering data on energy consumption and accuracy.

The technical approach involves:
*   Comparing **INT8 and INT4 precision levels** against the original FP32 models. PyTorch's post-training static and dynamic quantization was used, noting that PyTorch has no native INT4 support on CPU.
*   Utilizing **NVIDIA TensorRT** for hardware-optimized kernels supporting INT8/INT4 on specific hardware. The hardware used was an **NVIDIA GeForce RTX 4090 (AD102)** GPU with Nvidia driver 535.230.02 and an **AMD Ryzen 9 7900X** CPU.
*   Implementing models selected from the **Hugging Face portal** using the **Hugging Face Transformers library**. This library provides an interface for working with architectures in NLP and beyond, simplifies dataset management, and facilitates integration with frameworks like PyTorch/TensorFlow, supporting operations like quantization. It allowed for rapid experimentation with INT8 and INT4 configurations.
*   Using PyTorch's `Dataset` and `DataLoader` for efficient data loading and preprocessing. Optimization features like `pin_memory` and `num_workers`/`prefetch_factor` were used for faster computation and parallel processing. Specific validation/test files or training samples were downloaded from datasets.
*   Adopting **CodeCarbon**, an open-source tool, to estimate energy consumption (kWh) during inference. It uses `pyRAPL` (for CPU) and `pynvml` (for NVIDIA GPU).
*   Measuring inference speed using the `time` library.
*   Evaluating model accuracy loss using the **Mean Squared Error (MSE)** between the raw outputs (logits) of the original and quantized models. A low MSE indicates good preservation of model behavior, while high values signal significant distortions. The MSE was calculated using `torch.mean((logits_or[i] - logits_intX[i]) ** 2).item()`.
*   Measuring model size reduction (Footprint) in MB or GB.

A total of **42 models** were selected from Hugging Face based on current trends. These models were classified into **12 different tasks**: Speech Processing, Text Representation, Foundation Models, Computer Vision, Multimodal Models, and Specialized Models.
Datasets widely used in scientific literature for evaluating quantization impact were used. These datasets cover audio, text, and image domains and include examples like MLCommons/unsupervised_peoples_speech for Speech Processing, LLukas22/nq-simplified for NLP, and ILSVRC/imagenet-1k for Computer Vision.

The experimental pipeline was implemented via a Python script that evaluates quantization impact in terms of accuracy (MSE), energy efficiency (CPU/GPU/RAM consumption), and model memory size. The setup involved initialization of original and quantized models, data loading with optimized batch sizes, workers, and prefetch factors, and wrapping the execution with timing and CodeCarbon trackers.

## Run scripts

- original model:
```python
python3 db_or_... -m
```
- quantized INT8 model:
```python
python3 db_qd_... -m
```
- quantized INT$ model:
```python
python3 db_qd_... -m -q 4
```
















# Quantization Large-scale for AI Models

This academic thesis investigates the impact of Deep Learning model compression techniques, specifically **post-training** quantization to INT8 and INT4 precision, on both model efficiency and environmental sustainability. The research analyzes a broad selection of models across various tasks like speech processing, computer vision, and text generation, using a GPU-accelerated setup. Key findings highlight the trade-offs between accuracy, model size, inference speed, and energy consumption, revealing that INT4 compression, while reducing model size, can sometimes increase energy usage depending on hardware optimization, such as the NVIDIA RTX 4090 favoring INT8 operations. The study underscores the crucial relationship between hardware capabilities and the effectiveness of quantization techniques in achieving Green AI objectives.

---

## 📚 Usage

Clone the repository and install manually:

```bash
git clone https://github.com/lonelygunter/greenai
cd quantization-toolkit
```

Create a venv and than:

```bash
pip install -r requirements.txt
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

### Feature Extraction and Image Classification

Quantization delivers strong results for **feature extraction tasks**, such as sentence embedding. For example, `BAAI/bge-small-en-v1.5` exhibits only an 8% increase in MSE while achieving an 80% reduction in model size. Similarly, in **image classification** tasks (e.g., `microsoft/resnet-50`), quantization achieves a 75% memory footprint reduction with minimal impact on accuracy, as summarized in Table \[tab\:fea\_iclass].

| Model                  | Tasks | INT8 mean MSE | INT4 mean MSE | INT8 Size | INT4 Size |
| ---------------------- | ----- | ------------- | ------------- | --------- | --------- |
| BAAI/bge-base-en-v1.5  | FE    | 4.48e-7       | 1.54e-6       | 25.00%    | 15.30%    |
| BAAI/bge-small-en-v1.5 | FE    | 7.80e-7       | 1.64e-6       | 25.00%    | 17.04%    |
| microsoft/resnet-50    | IC    | 7.16e-8       | 9.60e-9       | 25.00%    | 25.00%    |

### Sentence Transformers and Energy Impact

In **sentence similarity** tasks, INT8 quantization offers clear environmental advantages. For several models, energy consumption is reduced by up to 50% with negligible loss in accuracy. `all-mpnet-base-v2`, for example, reduces energy usage to 92.78% while maintaining strong accuracy (Table \[tab\:sen\_tra]).

| Model             | INT8 mean MSE | INT4 mean MSE | INT8 Energy % | INT4 Energy % |
| ----------------- | ------------- | ------------- | ------------- | ------------- |
| all-MiniLM-L6-v2  | 1.18e-7       | 8.43e-7       | 100.06%       | 111.10%       |
| all-mpnet-base-v2 | 3.09e-7       | 2.17e-7       | 92.78%        | 119.35%       |

### Quantization vs Accuracy

Trade-offs become evident when transitioning from INT8 to INT4. A general trend is observed where INT4 quantization leads to higher MSE—up to 110% increase in some cases (e.g., `BAAI/bge-small-en-v1.5`). This suggests INT4 precision may not be ideal for all models unless accuracy degradation is acceptable. However, INT4 also brings aggressive size reductions, often between **75%–85%**, especially in transformer-based models.

### Speed of Inference

The inference speed improvements are **task and architecture dependent**. Quantization generally accelerates vision models (e.g., `resnet-50` improves by 11.6% with INT8 and 38.2% with INT4), while some NLP models experience slowdowns—particularly under INT4—due to hardware dependencies. Notably, the `facebook/opt-125m` model exhibits a 557% speedup with INT8, indicating potential architectural advantages under quantization.

### Size vs Performance

Sentence transformers such as `all-MiniLM-L6-v2` show **optimal trade-offs**, achieving **\~75% size reduction** with minimal impact on accuracy or speed. These results make such models excellent candidates for deployment in edge environments or resource-constrained settings.
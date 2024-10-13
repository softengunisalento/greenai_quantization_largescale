# 🌱 Green AI

The idea is to use an **horizontal approach** where we understand **the effectiveness of compression algorithms** on a large scale, through **quantization techniques** (without optimizing them), to understand what data we derive in terms of **energy consumption** and **carbon footprint**, being careful to **preserve**, as much as possible, the **same accuracy** of the original model.

- use **50+ models** from any domain
    - without training them
- **evaluation** the models, checking:
    - **energy consumption**
    - **carbon footprint**
    - **accuracy**

## Vertical approach:

- **anomaly detection’s** domain + testing
- **evaluate** as much as possible 2 o 3 models
- shift inference over time (domain specific problem)
- check the **tradeoff** between carbon footprint savings and requirements

**IMPORTANT**: have to create a lot of **chart and statistics** to make understand the **value added** to the committee

## 🪛 Quantization

> Quantization is a technique to **reduce the computational and memory costs** of running inference by representing the weights and activations with **low-precision** data types like `int8` instead of the usual `float32` .
>
>Reducing the number of bits means the resulting model requires **less memory storage**, consumes **less energy** (in theory), and operations like **matrix multiplication** can be performed much faster with integer arithmetic. It also allows to run models on embedded devices, which sometimes only support integer data types.

The two most common quantization cases are `float32 -> float16` and `float32 -> int8`.

### Quantization to int8

The idea is to find the best way to project our range `[a, b]` of `float32` values to the `int8` space.

Let’s consider a float `x` in `[a, b]`, then we can write the following quantization scheme:

$$ x = S * (x_q - Z) $$

where:

- `x_q` is the quantized int8 value associated to `x`
- `S` and `Z` are the quantization parameters
  - `S` is the scale, and is a positive `float32`
  - `Z` is called the zero-point, it is the `int8` value corresponding to the value 0 in the `float32` realm. This is important to be able to represent exactly the value 0 because it is used everywhere throughout machine learning models.

so:

$$ x_q = round(x/S + Z) $$

And `float32` values outside of the `[a, b]` range are clipped to the closest representable value, so for any floating-point number `x`:

$$ x_q = clip(round(x/S + Z), round(a/S + Z), round(b/S + Z)) $$

### Calibration

Calibration is the step during quantization where the `float32` ranges are computed. For **weights it is quite easy** since the actual range is known at quantization-time. But it is less clear for **activations**, and different approaches exist:

- Post training **dynamic quantization**: the range for each activation is **computed on the fly at runtime**. While this gives **great results** without too much work, it can be a bit **slower than static quantization** because of the overhead introduced by computing the range each time
- Post training **static quantization**: the range for each activation is **computed in advance at quantization-time**, typically by passing representative data through the model and recording the activation values. In practice, the steps are:
  1. Observers are put on activations to record their values.
  2. A certain number of forward passes on a calibration dataset is done (around 200 examples is enough).
  3. The ranges for each computation are computed according to some calibration technique.
- Quantization aware training (**not an option for us**): the range for each activation is computed at training-time. But “fake quantize” operators are used instead of observers: they record values just as observers do, but they also simulate the error induced by quantization to let the model adapt to it.

#### Define calibration techniques

For both post training **static quantization** and **quantization aware training**, it is necessary to define calibration techniques, the most common are:

- **Min-max**: the computed range is [min observed value, max observed value], this works well with **weights**.
- **Moving average min-max**: the computed range is [moving average min observed value, moving average max observed value], this works well with **activations**.
- **Histogram**: records a histogram of values **along with min and max values**, then chooses according to some criterion:
  - Entropy: the range is computed as the one minimizing the error between the full-precision and the quantized data.
  - Mean Square Error: the range is computed as the one minimizing the mean square error between the full-precision and the quantized data.
  - Percentile: the range is computed using a given percentile value p on the observed values. The idea is to try to have p% of the observed values in the computed range. While this is possible when doing affine quantization, it is not always possible to exactly match that when doing symmetric quantization. You can check how it is done in ONNX Runtime for more details.

### Pratical steps to quantize a model to int8

1. **Choose which operators** to quantize. Good operators to quantize are the one dominating it terms of computation time, for instance linear projections and matrix multiplications.
2. **Try post-training dynamic quantization**, **if** it is **fast enough stop** here, otherwise continue to step 3.
3. Try post-training **static quantization** which can be **faster** than dynamic quantization but often with a **drop in terms of accuracy**. Apply observers to your models in places where you want to quantize.
4. Choose a **calibration technique** and perform it.
5. Convert the model to its quantized form: the observers are removed and the **float32 operators are converted to their int8 coutnerparts**.
6. Evaluate the quantized model: is the **accuracy good enough?** If yes, stop here, otherwise start again at step 3 but with **quantization aware training** this time.
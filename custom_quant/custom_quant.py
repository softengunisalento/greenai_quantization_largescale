import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import time

# Import the quantization code from the previous artifact
# Assuming all the functions are defined in a module called 'quantizer'
# For this example, I'll redefine the key functions needed

# Constants
SYM_ZERO_POINT = 0

def linear_dequantization(q_tensor, scale, zero_point):
    """Dequantize a tensor using linear quantization"""
    return scale * (q_tensor.float() - zero_point)

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """Quantize a tensor with a given scale and zero point"""
    q_tensor = torch.round(tensor / scale + zero_point).to(dtype)
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    return torch.clamp(q_tensor, q_min, q_max)

def get_q_scale_symmetric(tensor, dtype=torch.int8):
    """Function that evaluates the scale between r_max and q_max"""
    q_max = torch.iinfo(dtype).max
    r_max = tensor.abs().max().item()
    scale = r_max / q_max
    return scale

def linear_q_symmetric(tensor, dtype=torch.int8):
    """Function that linearly quantizes a symmetric tensor"""
    scale = get_q_scale_symmetric(tensor, dtype)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, SYM_ZERO_POINT, dtype)
    return quantized_tensor, scale

class W8A16LinearLayer(nn.Module):
    """Custom Linear layer with 8-bit weight quantization"""
    def __init__(self, in_features, out_features, bias=True, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use register_buffer for quantized weights
        self.register_buffer('q_weight', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
    def quantize(self, weight_tensor):
        """Quantize the weights of the layer"""
        q_weight, scale = linear_q_symmetric(weight_tensor)
        self.q_weight.copy_(q_weight)
        self.weight_scale.copy_(torch.tensor(scale))
        
    def forward(self, x):
        """Forward pass with quantized weights"""
        deq_weight = linear_dequantization(self.q_weight, self.weight_scale, SYM_ZERO_POINT)
        return nn.functional.linear(x, deq_weight, self.bias)

def quantize_model_selectively(model, layer_patterns_to_quantize=None, layer_patterns_to_exclude=None):
    """Quantize specific layers of a model based on pattern matching"""
    if layer_patterns_to_exclude is None:
        layer_patterns_to_exclude = []
    
    # Function to check if a name matches any pattern
    def matches_pattern(name, patterns):
        if patterns is None:
            return False
        return any(pattern in name for pattern in patterns)
    
    # Clone the model to avoid modifying the original
    quantized_model = model
    
    # Collect all linear layers
    linear_layers = []
    
    def collect_linear_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                linear_layers.append((full_name, child))
            else:
                collect_linear_layers(child, full_name)
    
    collect_linear_layers(quantized_model)
    
    # Quantize selected layers
    for name, layer in linear_layers:
        # Skip if in exclude list
        if matches_pattern(name, layer_patterns_to_exclude):
            continue
            
        # Quantize if in include list or include list is None
        if layer_patterns_to_quantize is None or matches_pattern(name, layer_patterns_to_quantize):
            # Get parent module and attribute name
            parent_path, attr_name = name.rsplit('.', 1)
            parent = quantized_model
            
            for part in parent_path.split('.'):
                if part:  # Skip empty parts
                    parent = getattr(parent, part)
            
            # Create quantized layer
            old_bias = layer.bias
            old_weight = layer.weight
            new_layer = W8A16LinearLayer(layer.in_features, layer.out_features, 
                                        old_bias is not None, layer.weight.dtype)
            
            # Quantize weights
            new_layer.quantize(old_weight)
            
            # Set bias if exists
            if old_bias is not None:
                new_layer.bias = old_bias
                
            # Replace original layer with quantized one
            setattr(parent, attr_name, new_layer)
    
    return quantized_model

def evaluate_model_size(model):
    """Evaluate the size of the model in MB"""
    total_params = 0
    total_size = 0
    
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size()
        total_params += param.numel()
        total_size += param_size
        
    for name, buffer in model.named_buffers():
        buffer_size = buffer.numel() * buffer.element_size()
        total_size += buffer_size
        
    return total_size / (1024 * 1024)  # Convert to MB

# Simple example dataset for benchmarking
class SimpleDataset(Dataset):
    def __init__(self, tokenizer, text_samples, max_length=128):
        self.encodings = tokenizer(text_samples, truncation=True, padding='max_length', 
                                 max_length=max_length, return_tensors='pt')
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def benchmark_model(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Benchmark model performance on a small dataset"""
    # Prepare a small dataset for benchmarking
    text_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "I love natural language processing and deep learning.",
        "Transformers are powerful models for various NLP tasks.",
        "Machine learning is revolutionizing many industries.",
        "Quantization reduces model size while maintaining accuracy."
    ] * 20  # Repeat to create a larger dataset
    
    dataset = SimpleDataset(tokenizer, text_samples)
    dataloader = DataLoader(dataset, batch_size=8)
    
    model = model.to(device)
    model.eval()
    
    # Warm-up
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            _ = model(**inputs)
        break
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            _ = model(**inputs)
    
    elapsed_time = time.time() - start_time
    
    return {
        "inference_time": elapsed_time,
        "samples_per_second": len(dataset) / elapsed_time
    }

# Main code to load and quantize DistilBERT
def main():
    print("Loading DistilBERT model...")
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Print original model structure
    print("\nOriginal model structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")
    
    # Calculate original model size
    original_size = evaluate_model_size(model)
    print(f"\nOriginal model size: {original_size:.2f} MB")
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    original_benchmark = benchmark_model(model, tokenizer)
    print(f"Original model inference time: {original_benchmark['inference_time']:.4f} seconds")
    print(f"Original model throughput: {original_benchmark['samples_per_second']:.2f} samples/second")
    
    # Quantize model - selectively quantize different parts of the model
    print("\nQuantizing model...")
    
    # Example 1: Quantize all linear layers
    quantized_model_all = quantize_model_selectively(model)
    
    # Example 2: Only quantize attention layers
    quantized_model_attn = quantize_model_selectively(
        model, 
        layer_patterns_to_quantize=["attention"]
    )
    
    # Example 3: Quantize all except output layer
    quantized_model_selective = quantize_model_selectively(
        model,
        layer_patterns_to_exclude=["output"]
    )
    
    # Calculate quantized model sizes
    quantized_size_all = evaluate_model_size(quantized_model_all)
    quantized_size_attn = evaluate_model_size(quantized_model_attn)
    quantized_size_selective = evaluate_model_size(quantized_model_selective)
    
    print(f"\nQuantized model (all layers) size: {quantized_size_all:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size_all/original_size) * 100:.2f}%")
    
    print(f"\nQuantized model (attention only) size: {quantized_size_attn:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size_attn/original_size) * 100:.2f}%")
    
    print(f"\nQuantized model (selective) size: {quantized_size_selective:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size_selective/original_size) * 100:.2f}%")
    
    # Benchmark fully quantized model
    print("\nBenchmarking fully quantized model...")
    quantized_benchmark = benchmark_model(quantized_model_all, tokenizer)
    print(f"Quantized model inference time: {quantized_benchmark['inference_time']:.4f} seconds")
    print(f"Quantized model throughput: {quantized_benchmark['samples_per_second']:.2f} samples/second")
    print(f"Speedup: {quantized_benchmark['samples_per_second'] / original_benchmark['samples_per_second']:.2f}x")
    
    # Basic accuracy test - compare model outputs
    print("\nComparing model outputs...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # Original model output
    model = model.to(device)
    with torch.no_grad():
        original_output = model(**inputs).last_hidden_state
    
    # Quantized model output
    quantized_model_all = quantized_model_all.to(device)
    with torch.no_grad():
        quantized_output = quantized_model_all(**inputs).last_hidden_state
    
    # Calculate cosine similarity of outputs
    def cosine_similarity(tensor1, tensor2):
        return torch.nn.functional.cosine_similarity(
            tensor1.view(-1), tensor2.view(-1), dim=0
        ).item()
    
    similarity = cosine_similarity(original_output, quantized_output)
    print(f"Output similarity (cosine): {similarity:.6f}")
    
    # Calculate mean squared error
    mse = torch.mean((original_output - quantized_output) ** 2).item()
    print(f"Mean squared error: {mse:.6f}")
    
    print("\nQuantization complete!")

if __name__ == "__main__":
    main()
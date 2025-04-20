from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig
import torch

QUANTIZATION_BITS=4
MODEL_NAME="distilbert/distilbert-base-uncased"

def load_quantized_model(quantization_bits):
    """
    Load a dynamically quantized model with GPU support using bitsandbytes.
    :param model_name: Name of the pre-trained model from Hugging Face
    :param quantization_bits: Number of bits for quantization (4 or 8)
    :return: Quantized model and tokenizer
    """
    if quantization_bits not in [4, 8]:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    
    print(f"Loading model {MODEL_NAME} with {quantization_bits}-bit quantization...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True if quantization_bits == 4 else False,
        load_in_8bit=True if quantization_bits == 8 else False,
    )

    if QUANTIZATION_BITS == 4:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=device
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer, device

def test_model(model, tokenizer, device, text="Hello, how are you?"):
    """
    Run a simple forward pass to test the model
    :param model: The loaded quantized model
    :param tokenizer: The corresponding tokenizer
    :param device: Device to run inference on
    :param text: Input text for testing
    """

    torch.set_printoptions(threshold=float("inf"))
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    print("Model output:", outputs)

if __name__ == "__main__":
    model, tokenizer, device = load_quantized_model(QUANTIZATION_BITS)  # Change to 8 for 8-bit quantization
    test_model(model, tokenizer, device)

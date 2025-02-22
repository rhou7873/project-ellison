from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from optimum.exporters.onnx import export
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def load_huggingface_model(model_name: str):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loaded '{model_name}'")

    return model, tokenizer
    

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    model, tokenizer = load_huggingface_model(model_name=MODEL_NAME)
# src/models/__init__.py

from .tabcnn import TabCNN
from .fretnet import FretNet
# from .crnn import CRNN             
# from .transformer import Transformer   

MODEL_REGISTRY = {
    "tabcnn": TabCNN,
    "fretnet": FretNet,
    # "crnn": CRNN,
    # "transformer": Transformer
}

def get_model(model_name, model_params):
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' bulunamadı. Seçenekler: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**model_params)
# src/models/__init__.py

from typing import Type
import torch.nn as nn

from .base_model import BaseTabModel
from .tabcnn import TabCNN
from .fretnet import FretNet
from .transformer import Transformer

# Model ismini sınıfla eşleştiren merkezi bir kayıt listesi
MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "TabCNN": TabCNN,
    "FretNet": FretNet,
    "Transformer": Transformer,
}

def get_model(config: dict) -> nn.Module:
    """
    Acts as a factory to instantiate a model based on the configuration.

    This function reads the model name and parameters from the config,
    finds the corresponding model class in the registry, and initializes it,
    passing the full config object to the model's constructor.

    Args:
        config (dict): The complete experiment configuration dictionary.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    model_name = config['model']['name']
    model_params = config['model']['params']
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered in MODEL_REGISTRY.")

    model_class = MODEL_REGISTRY[model_name]
    
    return model_class(config=config, **model_params)
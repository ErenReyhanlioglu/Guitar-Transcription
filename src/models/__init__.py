# src/models/__init__.py

from typing import Type
import torch.nn as nn

from .base_model import BaseTabModel
from .tabcnn import TabCNN
from .fretnet import FretNet
from .transformer import Transformer

MODEL_REGISTRY: dict[str, Type[BaseTabModel]] = {
    "TabCNN": TabCNN,
    "FretNet": FretNet,
    "Transformer": Transformer,
}

def get_model(config: dict) -> BaseTabModel:
    """
    Acts as a factory to instantiate a model based on the configuration.

    This function reads all necessary parameters from the config and 
    initializes the correct model class.
    """
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered in MODEL_REGISTRY.")

    active_feature_name = config['data']['active_feature']
    feature_def = config['feature_definitions'][active_feature_name]
    
    init_params = {
        "in_channels": feature_def['in_channels'],
        "num_freq": feature_def['num_freq'],
        "num_strings": config['instrument']['num_strings'],
        "num_classes": config['data']['num_classes'],
        "config": config,
        **config['model']['params'] 
    }

    model_class = MODEL_REGISTRY[model_name]
    
    return model_class(**init_params)
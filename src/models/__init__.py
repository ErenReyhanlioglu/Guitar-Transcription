from typing import Type
import torch.nn as nn
import logging

from .base_model import BaseTabModel
from .cnn_mtl import cnn_mtl

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, Type[BaseTabModel]] = {
    "CNN_MTL": cnn_mtl,  
    "cnn_mtl": cnn_mtl,  
}

def get_model(config: dict) -> BaseTabModel:
    """
    Acts as a factory to instantiate a model based on the configuration.
    This version is updated to pass the entire config to the models,
    allowing them to build themselves dynamically.
    """
    model_name = config['model']['name']
    
    if model_name in MODEL_REGISTRY:
        target_name = model_name
    elif model_name.lower() in MODEL_REGISTRY:
        target_name = model_name.lower()
    else:
        raise ValueError(f"Model '{model_name}' is not registered in MODEL_REGISTRY. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[target_name]
    logger.info(f"Creating model: '{target_name}' from factory.")
    
    return model_class(config=config)
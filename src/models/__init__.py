# src/models/__init__.py
from typing import Type
import torch.nn as nn
import logging

from .base_model import BaseTabModel
from .tabcnn import TabCNN
from .fretnet import FretNet
from .transformer import Transformer

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, Type[BaseTabModel]] = {
    "TabCNN": TabCNN,
    "FretNet": FretNet,
    "Transformer": Transformer,
}

def get_model(config: dict) -> BaseTabModel:
    """
    Acts as a factory to instantiate a model based on the configuration.
    This version is updated to pass the entire config to the models,
    allowing them to build themselves dynamically.
    """
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered in MODEL_REGISTRY.")

    model_class = MODEL_REGISTRY[model_name]
    logger.info(f"Creating model: '{model_name}' from factory.")
    return model_class(config=config)
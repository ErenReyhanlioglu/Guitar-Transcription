"""
Purpose:
    This __init__.py file serves as a "model factory". It centralizes the
    model creation process, allowing the main experiment script to instantiate
    any model architecture simply by providing its name as a string (from a config file).

Dependencies:
    - Imports all model classes from the other files in this directory
      (e.g., TabCNN, CRNN, Transformer).

Current Status:
    - Contains a `get_model` function that takes a model name and a dictionary of
      parameters, then returns the corresponding initialized PyTorch model.

Future Plans:
    - [ ] Add more robust error handling for invalid model names or missing parameters.
"""
from .tabcnn import TabCNN

def get_model(model_name, model_params):
    if model_name.lower() == "tabcnn":
        return TabCNN(**model_params)
    else:
        raise ValueError(f"Model {model_name} not found.")
# src/utils/config_helpers.py

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def process_config(config):
    """
    Loads the raw config dictionary, derives necessary parameters,
    and returns the fully processed config.
    """
    max_fret = config['data']['max_fret']
    include_silence = config['data']['include_silence']

    num_fret_classes = max_fret + 1
    silence_class_idx = num_fret_classes
    total_model_classes = num_fret_classes + 1 if include_silence else num_fret_classes

    config['data']['silence_class'] = silence_class_idx
    config['model']['params']['num_classes'] = total_model_classes

    return config

def get_optimizer(model, config):
    """
    Creates an optimizer for the given model based on the "active_optimizer" 
    key in the configuration.
    """
    optimizer_config_root = config.get('training', {}).get('optimizer')

    if not optimizer_config_root:
        raise ValueError("Optimizer configuration block not found in config.")

    active_optimizer_name = optimizer_config_root.get('active_optimizer')
    if not active_optimizer_name:
        raise ValueError("'active_optimizer' not specified in config.")

    try:
        specific_config = optimizer_config_root['configurations'][active_optimizer_name]
        params = specific_config.get('params', {})
    except KeyError:
        raise ValueError(f"Configuration for active optimizer '{active_optimizer_name}' not found in yaml.")

    print(f"Initializing active optimizer: {active_optimizer_name} with params: {params}")

    name_lower = active_optimizer_name.lower()
    
    if name_lower == 'adam':
        return optim.Adam(model.parameters(), **params)
    
    elif name_lower == 'adamw':
        return optim.AdamW(model.parameters(), **params)
    
    else:
        raise ValueError(f"Optimizer '{active_optimizer_name}' not supported.")

def get_scheduler(optimizer, config):
    """
    Creates a learning rate scheduler based on the "active_scheduler" key
    in the configuration.
    """
    scheduler_config_root = config.get('training', {}).get('scheduler')

    if not scheduler_config_root:
        print("Scheduler configuration block not found.")
        return None

    active_scheduler_name = scheduler_config_root.get('active_scheduler')

    if not active_scheduler_name or active_scheduler_name.lower() == 'none':
        print("No active scheduler specified. Proceeding without a scheduler.")
        return None
        
    try:
        specific_config = scheduler_config_root['configurations'][active_scheduler_name]
        params = specific_config.get('params', {})
    except KeyError:
        raise ValueError(f"Configuration for active scheduler '{active_scheduler_name}' not found in yaml.")

    print(f"Initializing active scheduler: {active_scheduler_name} with params: {params}")

    name_lower = active_scheduler_name.lower()
    
    if name_lower == 'steplr':
        return StepLR(optimizer, **params)
    
    elif name_lower == 'reducelronplateau':
        return ReduceLROnPlateau(optimizer, **params)
    
    else:
        raise ValueError(f"Scheduler '{active_scheduler_name}' not supported.")
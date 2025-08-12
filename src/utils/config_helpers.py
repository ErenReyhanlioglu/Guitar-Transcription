import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def process_config(config: dict) -> dict:
    active_dataset_name = config['dataset']
    print(f"✅ Active dataset selected: '{active_dataset_name}'")
    
    dataset_specific_config = config['dataset_configs'][active_dataset_name]
    feature_definitions_config = config['feature_definitions']
    
    config['data'].update(dataset_specific_config)
    config['data']['features'] = feature_definitions_config
    
    active_prep_mode = config['data']['active_preparation_mode']
    prep_mode_params = config['data']['configurations'][active_prep_mode]
    
    config['data']['preparation_mode'] = active_prep_mode 
    
    config['data'].update(prep_mode_params)
    print(f"✅ Active data preparation mode: '{active_prep_mode}' with params: {prep_mode_params}")

    active_loss_name = config['loss']['active_loss']
    loss_params = config['loss']['configurations'][active_loss_name]
    config['loss']['type'] = active_loss_name
    config['loss'].update(loss_params)
    print(f"✅ Active loss function: '{active_loss_name}' with params: {loss_params}")

    active_feature_name = config['data']['active_feature']
    feature_params = config['data']['features'][active_feature_name]
    config['model']['params']['in_channels'] = feature_params['in_channels']
    config['model']['params']['num_freq'] = feature_params['num_freq']
    
    print(f"✅ Active feature set to '{active_feature_name}' with params: "
          f"in_channels={feature_params['in_channels']}, num_freq={feature_params['num_freq']}")

    num_frets = config['instrument']['num_frets']
    num_fret_classes = num_frets + 1
    silence_class_idx = num_fret_classes
    total_model_classes = num_fret_classes + 1
    
    config['data']['num_classes'] = total_model_classes
    config['data']['silence_class'] = silence_class_idx
    config['model']['params']['num_classes'] = total_model_classes
    config['model']['params']['num_strings'] = config['instrument']['num_strings']
    
    del config['data']['configurations']
    del config['loss']['configurations']
    
    print("✅ Config processing complete.")
    return config

def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config['active_optimizer']
    params = optimizer_config['configurations'][optimizer_name]['params']
    
    print(f"Initializing optimizer: {optimizer_name} with params: {params}")
    
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), **params)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    scheduler_config = config['training'].get('scheduler')
    if not scheduler_config:
        return None
        
    scheduler_name = scheduler_config.get('active_scheduler')
    if not scheduler_name or scheduler_name.lower() == 'none':
        return None
        
    params = scheduler_config['configurations'][scheduler_name].get('params', {})
    print(f"Initializing scheduler: {scheduler_name} with params: {params}")

    if scheduler_name.lower() == 'steplr':
        return StepLR(optimizer, **params)
    elif scheduler_name.lower() == 'reducelronplateau':
        params.pop('monitor', None)
        return ReduceLROnPlateau(optimizer, **params)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not supported.")
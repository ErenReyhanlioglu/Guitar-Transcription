import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import logging  

logger = logging.getLogger(__name__)

def process_config(config: dict) -> dict:
    """
    Parses the raw config dictionary, setting up derived parameters 
    and preparing it for the training pipeline.
    This version is updated to handle a list of active features.
    """
    active_dataset = config['dataset']
    logger.info(f"Active dataset selected: '{active_dataset}'")
    config['data'].update(config['dataset_configs'][active_dataset])

    active_prep_mode = config['data']['active_preparation_mode']
    prep_mode_params = config['data']['configurations'][active_prep_mode]
    config['data'].update(prep_mode_params)
    logger.info(f"Active data preparation mode: '{active_prep_mode}' with {prep_mode_params}")

    active_loss_name = config['loss']['active_loss']
    loss_params = config['loss']['configurations'][active_loss_name]
    config['loss'].update(loss_params)
    logger.info(f"Active loss function: '{active_loss_name}' with {loss_params}")

    if 'active_features' not in config['data']:
        raise ValueError("'data.active_features' listesi config dosyasında bulunamadı.")
        
    active_features_list = config['data']['active_features']
    logger.info(f"Active features set to: {active_features_list}")

    for feature in active_features_list:
        if feature not in config['feature_definitions']:
            raise ValueError(f"Aktif özellik '{feature}', 'feature_definitions' içinde tanımlanmamış.")

    num_frets = config['instrument']['num_frets']
    silence_class_idx = num_frets + 1  #  19 + 1 = 20
    total_model_classes = num_frets + 2 # 19 + 2 = 21 (0-19. frets + silence)

    config['data']['silence_class'] = silence_class_idx 
    config['data']['num_classes'] = total_model_classes
    config['model']['params']['num_classes'] = total_model_classes
    config['model']['params']['num_strings'] = config['instrument']['num_strings']
    
    del config['data']['configurations']
    del config['loss']['configurations']
    
    logger.info("Config processing complete.")
    return config

def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Initializes the optimizer based on the config."""
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config['active_optimizer']
    params = optimizer_config['configurations'][optimizer_name]['params']
    
    logger.info(f"Initializing optimizer: {optimizer_name} with {params}")
    
    optimizer_name_lower = optimizer_name.lower()
    if optimizer_name_lower == 'adam':
        return optim.Adam(model.parameters(), **params)
    elif optimizer_name_lower == 'adamw':
        return optim.AdamW(model.parameters(), **params)
    else:
        logger.error(f"Optimizer '{optimizer_name}' not supported.")
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Initializes the learning rate scheduler based on the config."""
    scheduler_config = config['training'].get('scheduler')
    if not scheduler_config:
        return None
        
    scheduler_name = scheduler_config.get('active_scheduler')
    if not scheduler_name or scheduler_name.lower() == 'none':
        logger.info("No learning rate scheduler will be used.")
        return None
        
    params = scheduler_config['configurations'][scheduler_name].get('params', {})
    logger.info(f"Initializing scheduler: {scheduler_name} with {params}")
    
    scheduler_name_lower = scheduler_name.lower()
    if scheduler_name_lower == 'steplr':
        return StepLR(optimizer, **params)
    elif scheduler_name_lower == 'reducelronplateau':
        safe_params = params.copy()
        safe_params.pop('monitor', None) 
        return ReduceLROnPlateau(optimizer, **safe_params)
    else:
        logger.error(f"Scheduler '{scheduler_name}' not supported.")
        raise ValueError(f"Scheduler '{scheduler_name}' not supported.")
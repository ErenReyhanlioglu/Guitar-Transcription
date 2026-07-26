import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

def resolve_device(config: dict) -> torch.device:
    """
    Resolves config['training']['device'] against actual CUDA availability.
    Falls back to CPU if CUDA was requested but isn't available, logging a
    clear warning first — CPU works for small/debug runs but is impractical
    for a full six-fold training run (the paper's runs used an A100 GPU).
    """
    requested = config['training']['device']
    if requested == 'cuda' and not torch.cuda.is_available():
        logger.warning(
            "Config requests device='cuda' but no CUDA GPU is available — "
            "falling back to CPU. This works for quick/debug runs, but a full "
            "six-fold training run on CPU will be far too slow to be practical."
        )
        return torch.device('cpu')
    return torch.device(requested)

def process_config(config: dict) -> dict:
    """
    Processes the config, filling in missing fields and validating its format.
    """
    active_dataset = config['dataset']
    if active_dataset in config:
        logger.info(f"Active dataset config loaded for: '{active_dataset}'")
        config['data'].update(config[active_dataset])

    if 'active_features' not in config['data']:
        raise ValueError("'data.active_features' not found in the config.")

    raw_features = config['data']['active_features']

    if isinstance(raw_features, dict):
        active_features_list = [k for k, v in raw_features.items() if v]
    elif isinstance(raw_features, list):
        active_features_list = raw_features
    else:
        raise ValueError("'active_features' format not recognized (must be a List or Dict).")

    config['data']['active_features'] = active_features_list
    logger.info(f"Active features set to: {active_features_list}")

    for feature in active_features_list:
        if feature not in config['feature_definitions']:
            raise ValueError(f"Active feature '{feature}' is not defined in 'feature_definitions'.")

    num_frets = config['instrument']['num_frets']
    silence_class_val = config['instrument'].get('silence_class', num_frets + 1)
    
    if 'params' not in config['model']:
        config['model']['params'] = {}
        
    if 'num_classes' not in config['model']['params']:
        config['model']['params']['num_classes'] = silence_class_val + 1

    config['model']['params']['num_strings'] = config['instrument']['num_strings']
    
    logger.info(f"Config processing complete. Num Classes: {config['model']['params']['num_classes']}")
    return config

def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Builds an optimizer for the model. Supports differential learning rates.
    Logs if 'loss_lr' is present in the config (used inside Trainer).
    """
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config['active_optimizer']
    
    if 'configurations' in optimizer_config:
        opt_spec_config = optimizer_config['configurations'][optimizer_name]
    else:
        opt_spec_config = optimizer_config
        
    params = opt_spec_config['params']
    use_differential_lr = opt_spec_config.get('differential_lr', False)
    
    if use_differential_lr:
        logger.info(f"Initializing optimizer with DIFFERENTIAL LR: {optimizer_name}")
        
        backbone_params = []
        head_params = []
        
        # nn.Module.__getattr__ is typed to return `Tensor | Module`, so pyright can't
        # know these hasattr-guarded submodules are Modules (not Tensors) — they always
        # are here, since feature_branches/heads/etc. are only ever assigned nn.Module
        # subtypes in cnn_mtl.py.
        try:
            # 1. Backbone (Feature Branches)
            if hasattr(model, 'feature_branches'):
                backbone_params.extend(list(model.feature_branches.parameters()))  # type: ignore[reportAttributeAccessIssue]
            elif hasattr(model, 'backbone'): # support for the old structure
                backbone_params.extend(list(model.backbone.parameters()))  # type: ignore[reportAttributeAccessIssue]

            # 2. Projections (included in the backbone group if present)
            if hasattr(model, 'projections') and model.projections:
                 backbone_params.extend(list(model.projections.parameters()))  # type: ignore[reportAttributeAccessIssue]

            # 3. Heads (all heads)
            if hasattr(model, 'heads'):
                head_params.extend(list(model.heads.parameters()))  # type: ignore[reportAttributeAccessIssue]
            elif hasattr(model, 'head'): # support for the old structure
                head_params.extend(list(model.head.parameters()))  # type: ignore[reportAttributeAccessIssue]

            if not backbone_params or not head_params:
                raise AttributeError("Could not parse model parameters (feature_branches/heads not found).")

            # Build parameter groups
            param_groups = [
                {'params': backbone_params, 'lr': params['backbone_lr']},
                {'params': head_params, 'lr': params['head_lr']}
            ]

            # Extract other parameters like weight decay (excluding lr)
            base_params = {k: v for k, v in params.items() if 'lr' not in k}
            
            log_msg = f" -> Backbone LR: {params['backbone_lr']}, Head LR: {params['head_lr']}"
            if 'loss_lr' in params:
                log_msg += f", Loss LR: {params['loss_lr']} (Will be used in Trainer)"
            logger.info(log_msg)
            
        except AttributeError as e:
            logger.error(f"Differential LR error: incompatible model structure! {e}")
            raise e
        
        # Initialize the optimizer
        if optimizer_name.lower() == 'adam':
            return optim.Adam(param_groups, **base_params)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(param_groups, **base_params)

    else:
        # Standard (single-LR) optimizer
        logger.info(f"Initializing optimizer: {optimizer_name} with {params}")
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model.parameters(), **params)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model.parameters(), **params)

    raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """
    Builds the learning-rate scheduler.
    """
    scheduler_config = config['training'].get('scheduler')
    if not scheduler_config:
        return None
        
    scheduler_name = scheduler_config.get('active_scheduler')
    if not scheduler_name or scheduler_name.lower() == 'none':
        logger.info("No learning rate scheduler will be used.")
        return None
        
    if 'configurations' in scheduler_config:
        params = scheduler_config['configurations'][scheduler_name].get('params', {})
    else:
        params = scheduler_config.get('params', {})

    logger.info(f"Initializing scheduler: {scheduler_name} with {params}")
    
    scheduler_name_lower = scheduler_name.lower()
    
    if scheduler_name_lower == 'steplr':
        return StepLR(optimizer, **params)
        
    elif scheduler_name_lower == 'reducelronplateau':
        # The monitor parameter is used inside the trainer, not passed to the scheduler.
        init_params = params.copy()
        if 'monitor' in init_params:
            del init_params['monitor']
        if 'mode' in init_params and 'mode' not in ['min', 'max']:
             # Default value
             init_params['mode'] = 'min'
             
        return ReduceLROnPlateau(optimizer, **init_params)
    
    raise ValueError(f"Scheduler '{scheduler_name}' not supported.")
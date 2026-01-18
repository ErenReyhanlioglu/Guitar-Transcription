import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

def process_config(config: dict) -> dict:
    """
    Config dosyasını işleyerek eksik alanları doldurur ve formatı doğrular.
    """
    active_dataset = config['dataset']
    if active_dataset in config:
        logger.info(f"Active dataset config loaded for: '{active_dataset}'")
        config['data'].update(config[active_dataset])
    
    if 'active_features' not in config['data']:
        raise ValueError("'data.active_features' config dosyasında bulunamadı.")
    
    raw_features = config['data']['active_features']
    
    if isinstance(raw_features, dict):
        active_features_list = [k for k, v in raw_features.items() if v]
    elif isinstance(raw_features, list):
        active_features_list = raw_features
    else:
        raise ValueError("'active_features' formatı tanınamadı (Liste veya Dict olmalı).")

    config['data']['active_features'] = active_features_list
    logger.info(f"Active features set to: {active_features_list}")

    for feature in active_features_list:
        if feature not in config['feature_definitions']:
            raise ValueError(f"Aktif özellik '{feature}', 'feature_definitions' içinde tanımlanmamış.")

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
    Model için optimizer oluşturur. Differential Learning Rate destekler.
    Config'de 'loss_lr' varsa loglar (Trainer içinde kullanılır).
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
        
        try:
            # 1. Backbone (Feature Branches)
            if hasattr(model, 'feature_branches'):
                backbone_params.extend(list(model.feature_branches.parameters()))
            elif hasattr(model, 'backbone'): # Eski yapı desteği
                backbone_params.extend(list(model.backbone.parameters()))
            
            # 2. Projections (Varsa Backbone grubuna dahil edilir)
            if hasattr(model, 'projections') and model.projections:
                 backbone_params.extend(list(model.projections.parameters()))

            # 3. Heads (Tüm kafalar)
            if hasattr(model, 'heads'):
                head_params.extend(list(model.heads.parameters()))
            elif hasattr(model, 'head'): # Eski yapı desteği
                head_params.extend(list(model.head.parameters()))

            if not backbone_params or not head_params:
                raise AttributeError("Model parametreleri ayrıştırılamadı (feature_branches/heads bulunamadı).")

            # Parametre Gruplarını Oluştur
            param_groups = [
                {'params': backbone_params, 'lr': params['backbone_lr']},
                {'params': head_params, 'lr': params['head_lr']}
            ]
            
            # Weight Decay vb. diğer parametreleri ayıkla (lr hariç)
            base_params = {k: v for k, v in params.items() if 'lr' not in k}
            
            log_msg = f" -> Backbone LR: {params['backbone_lr']}, Head LR: {params['head_lr']}"
            if 'loss_lr' in params:
                log_msg += f", Loss LR: {params['loss_lr']} (Will be used in Trainer)"
            logger.info(log_msg)
            
        except AttributeError as e:
            logger.error(f"Differential LR hatası: Model yapısı uyumsuz! {e}")
            raise e
        
        # Optimizer'ı başlat
        if optimizer_name.lower() == 'adam':
            return optim.Adam(param_groups, **base_params)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(param_groups, **base_params)

    else:
        # Standart (Tek LR) Optimizer
        logger.info(f"Initializing optimizer: {optimizer_name} with {params}")
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model.parameters(), **params)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model.parameters(), **params)

    raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """
    Learning Rate Scheduler oluşturur.
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
        # Monitor parametresi trainer içinde kullanılır, scheduler'a verilmez.
        init_params = params.copy()
        if 'monitor' in init_params:
            del init_params['monitor']
        if 'mode' in init_params and 'mode' not in ['min', 'max']:
             # Varsayılan değer
             init_params['mode'] = 'min'
             
        return ReduceLROnPlateau(optimizer, **init_params)
    
    raise ValueError(f"Scheduler '{scheduler_name}' not supported.")
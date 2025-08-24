import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

def process_config(config: dict) -> dict:
    """
    Ham konfigürasyon sözlüğünü işler, türetilmiş parametreleri ayarlar
    ve eğitim süreci için hazır hale getirir. Bu versiyon, iç içe geçmiş
    'configurations' anahtarlarını doğru bir şekilde yönetir.
    """
    active_dataset = config['dataset']
    logger.info(f"Active dataset selected: '{active_dataset}'")
    config['data'].update(config['dataset_configs'][active_dataset])

    active_prep_mode = config['data']['active_preparation_mode']
    prep_mode_params = config['data']['configurations'][active_prep_mode]
    logger.info(f"Active data preparation mode: '{active_prep_mode}' with {prep_mode_params}")
    config['data'].update(prep_mode_params)

    active_loss_name = config['loss']['active_loss']
    loss_params = config['loss']['configurations'][active_loss_name]
    logger.info(f"Active primary loss configuration set to '{active_loss_name}' with {loss_params}")
    config['loss'].update(loss_params)

    if 'active_features' not in config['data']:
        raise ValueError("'data.active_features' listesi config dosyasında bulunamadı.")
    active_features_list = config['data']['active_features']
    logger.info(f"Active features set to: {active_features_list}")
    for feature in active_features_list:
        if feature not in config['feature_definitions']:
            raise ValueError(f"Aktif özellik '{feature}', 'feature_definitions' içinde tanımlanmamış.")

    num_frets = config['instrument']['num_frets']
    silence_class_idx = num_frets + 1  # 0-19 frets, 20. index "no-note"
    total_model_classes = num_frets + 2 # 0-19 ffrest + "no-note" + "out-of-range" 

    config['data']['silence_class'] = silence_class_idx
    config['data']['num_classes'] = total_model_classes
    if 'params' not in config['model']:
        config['model']['params'] = {}
    config['model']['params']['num_classes'] = total_model_classes
    config['model']['params']['num_strings'] = config['instrument']['num_strings']

    if 'configurations' in config['data']:
        del config['data']['configurations']
    if 'configurations' in config['loss']:
        del config['loss']['configurations']
    
    logger.info("Config processing complete.")
    return config


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Config dosyasına göre optimizer'ı başlatır.
    Diferansiyel öğrenme oranını (differential_lr) destekler.
    """
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config['active_optimizer']
    opt_spec_config = optimizer_config['configurations'][optimizer_name]
    params = opt_spec_config['params']
    
    use_differential_lr = opt_spec_config.get('differential_lr', False)
    
    if use_differential_lr:
        logger.info(f"Initializing optimizer with DIFFERENTIAL LR: {optimizer_name}")
        try:
            param_groups = [
                {'params': model.backbone.parameters(), 'lr': params['backbone_lr']},
                {'params': model.head.parameters(), 'lr': params['head_lr']}
            ]
            base_params = {k: v for k, v in params.items() if 'lr' not in k}
            logger.info(f"  -> Backbone LR: {params['backbone_lr']}, Head LR: {params['head_lr']}, Other params: {base_params}")
        except AttributeError as e:
            logger.error("Diferansiyel LR için modelde 'backbone' ve 'head' attribute'ları bulunamadı!")
            raise AttributeError(f"Modelinizde diferansiyel öğrenme oranını desteklemek için 'backbone' ve 'head' adında katman grupları tanımlanmalıdır. Hata: {e}")
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(param_groups, **base_params)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(param_groups, **base_params)

    else:
        logger.info(f"Initializing optimizer: {optimizer_name} with {params}")
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model.parameters(), **params)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model.parameters(), **params)

    raise ValueError(f"Optimizer '{optimizer_name}' not supported.")


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Config dosyasına göre learning rate scheduler'ını başlatır."""
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
        init_params = params.copy()
        if 'monitor' in init_params:
            del init_params['monitor']
        return ReduceLROnPlateau(optimizer, **init_params)
    
    raise ValueError(f"Scheduler '{scheduler_name}' not supported.")
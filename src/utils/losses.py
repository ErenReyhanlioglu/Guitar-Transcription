import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, 
            reduction='none', 
            ignore_index=self.ignore_index, 
            weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            active_elements = (targets != self.ignore_index).sum()
            return focal_loss.sum() / (active_elements + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskUncertaintyLoss(nn.Module):
    """
    Dynamic Uncertainty Weighting.
    Creates a learnable sigma parameter for each active task.
    """
    def __init__(self, active_tasks):
        super().__init__()
        self.active_tasks = set(active_tasks)
        
        # Initialize log variance parameters to 0.0 (variance=1.0)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in active_tasks
        })
        
    def forward(self, losses_dict):
        total_loss = 0
        
        for task_name, loss_val in losses_dict.items():
            if task_name in self.log_vars:
                log_var = self.log_vars[task_name]
                precision = torch.exp(-log_var)
                
                # Formula: 1/(2*sigma^2) * Loss + log(sigma)
                weighted_loss = 0.5 * precision * loss_val + 0.5 * log_var
                total_loss += weighted_loss
            else:
                # If task is not in UW list (unexpected), add raw loss
                total_loss += loss_val
                
        return total_loss

class CombinedLoss(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.config = config
        self.loss_config = config['loss']
        
        # --- CONFIG FIX ---
        # Heads parametresini esnek bir şekilde bul
        model_params = config['model'].get('params', {})
        if 'heads' in model_params:
            self.heads_config = model_params['heads']
        else:
            self.heads_config = config['model'].get('heads', {}) # Fallback
            
        self.device = config['training']['device']
                
        tab_conf = self.loss_config.get('tablature_loss', {})
        self.tab_loss_fn = FocalLoss(
            gamma=tab_conf.get('params', {}).get('gamma', 2.0),
            ignore_index=tab_conf.get('params', {}).get('ignore_index', -1),
            alpha=class_weights.to(self.device) if class_weights is not None else None
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss() # For Hand Position

        self.active_tasks = ['tablature'] 
        
        if self.heads_config.get('hand_position', {}).get('enabled', False):
            self.active_tasks.append('hand_position')
        if self.heads_config.get('string_activity', {}).get('enabled', False):
            self.active_tasks.append('string_activity')
        if self.heads_config.get('pitch_class', {}).get('enabled', False):
            self.active_tasks.append('pitch_class')
        if self.heads_config.get('multipitch', {}).get('enabled', False): 
            self.active_tasks.append('multipitch')
        if self.heads_config.get('onset', {}).get('enabled', False):
            self.active_tasks.append('onset')
        

        logger.info(f"Loss Active Tasks: {self.active_tasks}")

        self.strategy = self.loss_config.get('weighting_strategy', 'static')
        
        if self.strategy == 'uncertainty':
            self.uncertainty_wrapper = MultiTaskUncertaintyLoss(self.active_tasks)
            logger.info("Strategy: Uncertainty Weighting (Learnable Sigmas)")
        else:
            logger.info("Strategy: Static Weighting (Fixed Config Weights)")

    def forward(self, outputs, targets):
            losses = {}
                    
            # --- 1. Tablature Loss ---
            if 'tab_logits' in outputs:
                tab_logits = outputs['tab_logits'] # Beklenen: (B*T, S, C) veya (B*T, S*C)
                tab_targets = targets['tablature'] # Beklenen: (B*T, S)
                
                # Logits Shape Düzeltme: (N, S*C) gelirse -> (N, S, C) yap
                if tab_logits.dim() == 2:
                    # Muhtemelen (Batch*Time, Strings*Classes)
                    S = self.config['instrument']['num_strings']
                    C = self.config['model']['params']['num_classes']
                    tab_logits = tab_logits.view(-1, S, C)
                
                # Loss için Flattening: 
                # Logits -> (Total_Samples, Classes) -> (2000 * 6, 21)
                # Targets -> (Total_Samples)         -> (2000 * 6)
                
                # (Batch*Time, Strings, Classes) -> Permute -> (Batch*Time, Classes, Strings) -> Yanlış olur
                # Doğrusu: Her bir (Tel, Zaman) çifti bağımsız bir örnektir.
                
                # Logits: (N, S, C) -> (N*S, C)
                flat_logits = tab_logits.reshape(-1, tab_logits.shape[-1])
                
                # Targets: (N, S) -> (N*S)
                flat_targets = tab_targets.reshape(-1)
                
                losses['tablature'] = self.tab_loss_fn(flat_logits, flat_targets)

            # --- 2. Hand Position Loss ---
            if 'hand_pos_logits' in outputs:
                # Logits: (N, Classes), Target: (N,)
                # Zaten data loader bunu (B*T, C) ve (B*T) olarak veriyor olmalı
                losses['hand_position'] = self.ce(outputs['hand_pos_logits'], targets['hand_pos_target'])
                
            # --- 3. String Activity Loss ---
            if 'activity_logits' in outputs:
                # BCE Loss (Logits ve Targets aynı boyutta olmalı)
                # Logits: (N, 6), Target: (N, 6)
                losses['string_activity'] = self.bce(outputs['activity_logits'], targets['activity_target'])
                
            # --- 4. Pitch Class Loss ---
            if 'pitch_class_logits' in outputs:
                losses['pitch_class'] = self.bce(outputs['pitch_class_logits'], targets['pitch_class_target'])
                
            # --- 5. Multipitch Loss ---
            if 'multipitch_logits' in outputs:
                losses['multipitch'] = self.bce(outputs['multipitch_logits'], targets['multipitch_target'])
                
            # --- 6. Onset Loss ---
            if 'onset_logits' in outputs:
                losses['onset'] = self.bce(outputs['onset_logits'], targets['onset_target'])
                        
            # --- Loss Aggregation ---
            if self.strategy == 'uncertainty':
                total_loss = self.uncertainty_wrapper(losses)
            else:
                total_loss = losses.get('tablature', 0.0)
                for key in losses:
                    if key == 'tablature': continue
                    weight_key = f"{key}_loss" 
                    w = self.loss_config.get(weight_key, {}).get('weight', 1.0)
                    total_loss += w * losses[key]

            loss_dict_out = {'total_loss': total_loss}
            for k, v in losses.items():
                loss_dict_out[f"{k}_loss"] = v
                
            return loss_dict_out
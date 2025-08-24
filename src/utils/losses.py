import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        valid_mask = target != self.ignore_index
        if self.reduction == 'mean':
            return focal_loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0).to(input.device)
        elif self.reduction == 'sum':
            return focal_loss[valid_mask].sum()
        else:
            return focal_loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomSoftmaxTablatureLoss(nn.Module): # Bu özelleştirilecek
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, targets):
        return self.loss_fn(preds, targets)

class CustomLogisticTablatureLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        loss_params = config['loss'] 
        instrument_params = config['instrument']
        self.lmbda = loss_params.get('lmbda', 0.0)
        self.num_strings = instrument_params['num_strings']
        self.num_frets = instrument_params['num_frets'] + 1 # num_classes_per_string is num_frets + silence
        self.use_focal = loss_params.get('use_focal', False)
        
        if self.use_focal:
            focal_params = loss_params.get('focal_params', {})
            self.bce_equivalent = BinaryFocalLoss(
                alpha=focal_params.get('alpha', 0.25),
                gamma=focal_params.get('gamma', 2.0)
            )
        else:
            self.bce_equivalent = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        main_loss = self.bce_equivalent(preds, targets)
        
        inhibition_loss = 0.0
        if self.lmbda > 0:
            probs = torch.sigmoid(preds)
            probs_reshaped = probs.view(-1, self.num_strings, self.num_frets)
            sum_probs_per_string = torch.sum(probs_reshaped, dim=-1)
            inhibition_loss = torch.mean(torch.pow(F.relu(sum_probs_per_string - 1), 2))

        return main_loss + self.lmbda * inhibition_loss

class CombinedLoss(nn.Module):
    def __init__(self, config: dict, class_weights: torch.Tensor = None):
        super().__init__()
        self.config = config
        self.loss_config = config['loss']
        self.primary_loss_config = self.loss_config 
        self.aux_loss_config = self.loss_config.get('auxiliary_loss', {})
        self.aux_enabled = self.aux_loss_config.get('enabled', False)
        
        self.class_weights = class_weights
        if self.class_weights is not None:
            logger.info(f"CombinedLoss initialized with external class weights of shape: {self.class_weights.shape}")

        self.primary_loss_fn = self._create_primary_loss()
        
        if self.aux_enabled:
            self.auxiliary_loss_fn = self._create_auxiliary_loss()
            self.aux_weight = self.aux_loss_config.get('weight', 0.4)

    def _create_primary_loss(self):
        loss_type = self.primary_loss_config['type']
        logger.info(f"Creating primary loss of type: '{loss_type}'")
        
        if loss_type == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(ignore_index=-1)
        elif loss_type == "CustomSoftmaxTablatureLoss":
            logger.info(" -> Initializing CustomSoftmaxTablatureLoss (currently a wrapper for CrossEntropy).")
            return CustomSoftmaxTablatureLoss(ignore_index=-1)
        elif loss_type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        elif loss_type == "CustomLogisticTablatureLoss":
            logger.info(" -> Initializing CustomLogisticTablatureLoss with internal config.")
            return CustomLogisticTablatureLoss(self.config)
        else:
            raise ValueError(f"Unsupported primary loss type: {loss_type}")

    def _create_auxiliary_loss(self):
        loss_type = self.aux_loss_config.get('type', 'BCEWithLogitsLoss')
        logger.info(f"Creating auxiliary loss of type: '{loss_type}'")
        if loss_type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss() 
        else:
            raise ValueError(f"Unsupported auxiliary loss type: {loss_type}")
    
    def forward(self, model_output, batch):
        tablature_target = batch['tablature']
        
        if self.aux_enabled:
            tab_logits, multipitch_logits = model_output
        else:
            tab_logits = model_output
            
        if self.config['loss']['active_loss'] == 'softmax_groups':
            if tab_logits.dim() == 4: # (B, T, S, C) output
                B, T, S, C = tab_logits.shape
                # Prep Target: (B, S, T) -> (B*T, S)
                tablature_target = tablature_target.permute(0, 2, 1).reshape(-1, S)
            elif tab_logits.dim() == 2: # TabCNN gibi (B_flat, S*C) çıktısı
                S = self.config['instrument']['num_strings']
                C = self.config['model']['params']['num_classes']
                # (B_flat, S*C) -> (B_flat, 1, S, C)
                tab_logits = tab_logits.view(-1, 1, S, C)
                # Target: (B_flat, S), (bkz: data_loader.py, collate_fn)
            else:
                raise ValueError(f"Unsupported tab_logits dimension: {tab_logits.dim()}")

            B, T, S, C = tab_logits.shape

            logger.debug(f"Unified shapes for loss calculation: tab_logits={tab_logits.shape}, tablature_target={tablature_target.shape}")

            total_string_loss = 0.0
            for s in range(S):
                pred_s = tab_logits[:, :, s, :].contiguous().view(-1, C)
                target_s = tablature_target[:, s].contiguous().view(-1)
                
                weight_s = self.class_weights[s].to(pred_s.device) if self.class_weights is not None else None
                if self.primary_loss_config.get('use_focal', False):
                    focal_params = self.primary_loss_config.get('focal_params', {})
                    gamma = focal_params.get('gamma', 2.0)
                    loss_calculator = MultiClassFocalLoss(gamma=gamma, weight=weight_s, ignore_index=-1)
                else:
                    loss_calculator = nn.CrossEntropyLoss(weight=weight_s, ignore_index=-1)
                string_loss = loss_calculator(pred_s, target_s)
                total_string_loss += string_loss
            primary_loss = total_string_loss / S
        else: 
            primary_loss = self.primary_loss_fn(tab_logits, tablature_target)

        if self.aux_enabled:
            multipitch_target = batch['multipitch_target']

            if multipitch_logits.dim() == 3 and multipitch_target.dim() == 3: 
              multipitch_target = multipitch_target.permute(0, 2, 1)

            logger.debug(f"Auxiliary loss shapes: multipitch_logits={multipitch_logits.shape}, multipitch_target={multipitch_target.shape}")
            aux_loss = self.auxiliary_loss_fn(multipitch_logits, multipitch_target)
            total_loss = primary_loss + self.aux_weight * aux_loss
            
            logger.debug(f"Loss - Total: {total_loss.item():.4f} (Primary: {primary_loss.item():.4f} + Aux: {aux_loss.item():.4f} * {self.aux_weight})")
            return {"total_loss": total_loss, "primary_loss": primary_loss, "aux_loss": aux_loss}
        else:
            logger.debug(f"Loss - Primary (Total): {primary_loss.item():.4f}")
            return {"total_loss": primary_loss, "primary_loss": primary_loss, "aux_loss": torch.tensor(0.0).to(primary_loss.device)}
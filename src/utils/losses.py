import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logger.debug(f"[FocalLoss] Inputs - input: {describe(input)}, target: {describe(target)}")
        ce_loss = F.cross_entropy(input, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SoftmaxGroups(nn.Module):
    def __init__(self, num_groups, group_size):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        logger.info(f"SoftmaxGroups loss initialized with num_groups={num_groups}, group_size={group_size}")

    def forward(self, preds, targets, class_weights=None, use_focal=False, focal_gamma=2.0):
        logger.debug(f"[SoftmaxGroups] --- loss calculation start ---")
        logger.debug(f"[SoftmaxGroups] Inputs - preds: {describe(preds)}, targets: {describe(targets)}, class_weights: {describe(class_weights)}")
        logger.debug(f"[SoftmaxGroups] Options - use_focal: {use_focal}, focal_gamma: {focal_gamma}")
        
        B, T, SxC = preds.shape
        S = self.num_groups
        C = self.group_size

        preds_reshaped = preds.view(B, T, S, C)
        logger.debug(f"[SoftmaxGroups] Reshaped preds: {describe(preds_reshaped)}")
        
        total_loss = 0.0

        for s in range(S):
            pred_s = preds_reshaped[:, :, s, :].contiguous().view(-1, C)
            target_s = targets[:, :, s].contiguous().view(-1)
            
            if s == 0:
                logger.debug(f"  -> For first string (s=0) - pred_s: {describe(pred_s)}, target_s: {describe(target_s)}")
            
            weight_s = class_weights[s].to(pred_s.device) if class_weights is not None else None
            
            if use_focal:
                ce = F.cross_entropy(pred_s, target_s, weight=weight_s, reduction='none', ignore_index=-1)
                valid_ce = ce[target_s != -1]
                if valid_ce.numel() > 0:
                    pt = torch.exp(-valid_ce)
                    loss = ((1 - pt) ** focal_gamma) * valid_ce
                    total_loss += loss.mean()
            else:
                loss_fn = nn.CrossEntropyLoss(weight=weight_s, ignore_index=-1)
                total_loss += loss_fn(pred_s, target_s)

        final_loss = total_loss / S
        logger.debug(f"[SoftmaxGroups] Returning final loss: {final_loss.item():.6f}")
        return final_loss

class LogisticBankLoss(nn.Module):
    def __init__(self, num_strings, num_classes, lmbda=1.0, use_focal=False, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.use_focal = use_focal
        self.lmbda = lmbda
        self.num_strings = num_strings
        self.num_frets = num_classes - 1
        
        logger.info(f"LogisticBankLoss initialized.")
        if self.use_focal:
            self.focal_loss_func = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"  -> Using BinaryFocalLoss (gamma={focal_gamma}, alpha={focal_alpha})")
        else:
            logger.info(f"  -> Using standard BCEWithLogitsLoss.")
        logger.info(f"  -> Inhibition lambda set to: {self.lmbda}")

    def forward(self, preds, targets_logistic, class_weights=None):
        logger.debug(f"[LogisticBankLoss] --- loss calculation start ---")
        logger.debug(f"[LogisticBankLoss] Inputs - preds: {describe(preds)}, targets_logistic: {describe(targets_logistic)}, class_weights: {describe(class_weights)}")

        pos_weight = None
        if class_weights is not None:
            pos_weight = class_weights[:, :-1].reshape(-1).to(preds.device)
            logger.debug(f"  -> Derived pos_weight for BCE: {describe(pos_weight)}")
        
        if self.use_focal:
            main_loss = self.focal_loss_func(preds, targets_logistic, pos_weight=pos_weight)
        else:
            main_loss = F.binary_cross_entropy_with_logits(preds, targets_logistic, pos_weight=pos_weight)
        
        logger.debug(f"  -> Main loss component: {main_loss.item():.6f}")

        inhibition_loss = 0.0
        if self.lmbda > 0:
            probs = torch.sigmoid(preds)
            probs_reshaped = probs.view(-1, self.num_strings, self.num_frets)
            logger.debug(f"  -> Reshaped probs for inhibition: {describe(probs_reshaped)}")
            sum_probs_per_string = torch.sum(probs_reshaped, dim=-1)
            inhibition_loss = torch.mean(torch.pow(F.relu(sum_probs_per_string - 1), 2))
            logger.debug(f"  -> Inhibition loss component: {inhibition_loss.item():.6f}")

        total_loss = main_loss + self.lmbda * inhibition_loss
        logger.debug(f"[LogisticBankLoss] Returning final loss: {total_loss.item():.6f} (main: {main_loss.item():.4f} + lambda*{self.lmbda} * inhibition: {inhibition_loss.item():.4f})")
        return total_loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, pos_weight=None):
        logger.debug(f"[BinaryFocalLoss] Inputs - inputs: {describe(inputs)}, targets: {describe(targets)}")
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='none')
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
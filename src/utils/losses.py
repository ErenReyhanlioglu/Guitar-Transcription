import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
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

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.num_groups, self.group_size)
        x = F.softmax(x, dim=-1)
        return x.view(B, T, D)

    def get_loss(self, preds, targets, class_weights=None, focal=False, gamma=2.0, include_silence=True):
        B, T, SxC = preds.shape
        S = self.num_groups
        C = self.group_size

        preds = preds.view(B, T, S, C)
        total_loss = 0.0

        for s in range(S):
            pred_s = preds[:, :, s, :].contiguous().view(-1, C)
            target_s = targets[:, :, s].contiguous().view(-1)

            if not include_silence:
                mask = target_s != -1
                pred_s = pred_s[mask]
                target_s = target_s[mask]

            weight_s = class_weights[s].to(pred_s.device) if class_weights is not None else None

            if focal:
                ce = F.cross_entropy(pred_s, target_s, weight=weight_s, reduction='none', ignore_index=-1)
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** gamma) * ce
                if len(loss) > 0:
                    total_loss += loss.mean()
            else:
                loss_fn = nn.CrossEntropyLoss(weight=weight_s, ignore_index=-1)
                total_loss += loss_fn(pred_s, target_s)

        return total_loss / S

class LogisticBankLoss(nn.Module):
    def __init__(self, num_strings, lmbda=1.0, use_focal=False, focal_alpha=0.25, focal_gamma=2.0): 
        super().__init__()
        
        if use_focal:
            print("LogisticBankLoss is using BinaryFocalLoss.")
            self.loss_func = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            print("LogisticBankLoss is using standard BCEWithLogitsLoss.")
            self.loss_func = nn.BCEWithLogitsLoss()
            
        self.lmbda = lmbda
        self.num_strings = num_strings

    def forward(self, preds, targets_logistic): 
        main_loss = self.loss_func(preds, targets_logistic)

        inhibition = 0.0
        if self.lmbda > 0:
            probs = torch.sigmoid(preds)
            if preds.dim() > 1:
                probs_reshaped = probs.view(probs.shape[0], self.num_strings, -1)
                inhibition = torch.mean(torch.pow(F.relu(torch.sum(probs_reshaped, dim=-1) - 1), 2))
            else:
                probs_reshaped = probs.view(self.num_strings, -1)
                inhibition = torch.mean(torch.pow(F.relu(torch.sum(probs_reshaped, dim=-1) - 1), 2))

        return main_loss + self.lmbda * inhibition

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
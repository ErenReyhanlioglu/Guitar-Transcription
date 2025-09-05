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

    def forward(self, input, target, class_weights=None): 
        ce_loss = F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index, reduction='none')
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

class CustomSoftmaxTablatureLoss(nn.Module):
    """
    Hata türüne (Miss, False Alarm, Substitution) göre farklı cezalar
    uygulayan özel bir Cross-Entropy loss fonksiyonu.
    """
    def __init__(self, miss_penalty=1.5, false_alarm_penalty=1.5, substitution_penalty=1.0, 
                 silence_class=20, ignore_index=-1):
        super().__init__()
        self.miss_penalty = miss_penalty
        self.false_alarm_penalty = false_alarm_penalty
        self.substitution_penalty = substitution_penalty
        self.silence_class = silence_class
        self.ignore_index = ignore_index
        logger.info(f"CustomSoftmaxTablatureLoss initialized with penalties: "
                    f"Miss={miss_penalty}, FA={false_alarm_penalty}, Sub={substitution_penalty}")

    def forward(self, preds, targets, class_weights=None):
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0).to(preds.device)

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        per_sample_loss = F.cross_entropy(preds, targets, weight=class_weights, reduction='none')

        with torch.no_grad():
            pred_classes = torch.argmax(preds, dim=1)
            
            miss_mask = (targets != self.silence_class) & (pred_classes == self.silence_class)
            false_alarm_mask = (targets == self.silence_class) & (pred_classes != self.silence_class)
            sub_mask = (targets != self.silence_class) & \
                       (pred_classes != self.silence_class) & \
                       (targets != pred_classes)

            penalties = torch.ones_like(targets, dtype=torch.float)
            penalties[miss_mask] = self.miss_penalty
            penalties[false_alarm_mask] = self.false_alarm_penalty
            penalties[sub_mask] = self.substitution_penalty

        weighted_loss = per_sample_loss * penalties
        return weighted_loss.mean()

class CustomPenalizedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, miss_penalty=1.5, false_alarm_penalty=1.5, substitution_penalty=1.0, 
                 silence_class=20, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.miss_penalty = miss_penalty
        self.false_alarm_penalty = false_alarm_penalty
        self.substitution_penalty = substitution_penalty
        self.silence_class = silence_class
        self.ignore_index = ignore_index
        logger.info("CustomPenalizedFocalLoss initialized.")

    def forward(self, preds, targets, class_weights=None):
        valid_mask = targets != self.ignore_index
        if not valid_mask.any(): return torch.tensor(0.0).to(preds.device)
        preds, targets = preds[valid_mask], targets[valid_mask]

        # 1. Temel Cross-Entropy loss'u hesapla
        ce_loss = F.cross_entropy(preds, targets, weight=class_weights, reduction='none')

        # 2. Focal Loss ağırlığını (pt) hesapla
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        # 3. Özel ceza ağırlıklarını hesapla
        with torch.no_grad():
            pred_classes = torch.argmax(preds, dim=1)
            miss_mask = (targets != self.silence_class) & (pred_classes == self.silence_class)
            fa_mask = (targets == self.silence_class) & (pred_classes != self.silence_class)
            sub_mask = (targets != self.silence_class) & (pred_classes != self.silence_class) & (targets != pred_classes)
            penalties = torch.ones_like(targets, dtype=torch.float)
            penalties[miss_mask] = self.miss_penalty
            penalties[fa_mask] = self.false_alarm_penalty
            penalties[sub_mask] = self.substitution_penalty

        # 4. Tüm bileşenleri birleştir
        final_loss = penalties * focal_term * ce_loss
        return final_loss.mean()

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
        self.instrument_config = config['instrument']
        
        self.active_loss_strategy = self.loss_config['active_loss']
        self.active_loss_config = self.loss_config['configurations'][self.active_loss_strategy]
        
        self.class_weights = class_weights
        if self.class_weights is not None and self.active_loss_config.get('use_class_weights', False):
            logger.info(f"CombinedLoss initialized with class weights of shape: {self.class_weights.shape}")

        self.primary_loss_fn = self._create_primary_loss()
        self._setup_auxiliary_losses()

    def _create_primary_loss(self):
        loss_type = self.active_loss_config.get('type')
        use_focal = self.active_loss_config.get('use_focal', False)
        
        logger.info(f"Creating primary loss for '{self.active_loss_strategy}' strategy...")
        
        if self.active_loss_strategy != 'softmax_groups':
            raise ValueError(f"This advanced loss setup is only for 'softmax_groups'.")

        penalty_config = self.active_loss_config.get('custom_softmax_penalties', {})
        focal_params = self.active_loss_config.get('focal_params', {})
        silence_cls = self.instrument_config['silence_class']

        # 1. BİRLEŞİK DURUM: Özel cezalar VE Focal Loss bir arada
        if loss_type == "CustomSoftmaxTablatureLoss" and use_focal:
            logger.info("--> Activating COMBINED loss: CustomPenalizedFocalLoss")
            return CustomPenalizedFocalLoss(
                gamma=focal_params.get('gamma', 2.0),
                miss_penalty=penalty_config.get('miss', 1.5),
                false_alarm_penalty=penalty_config.get('false_alarm', 1.5),
                substitution_penalty=penalty_config.get('substitution', 1.0),
                silence_class=silence_cls,
                ignore_index=-1
            )
        
        # 2. SADECE ÖZEL CEZA DURUMU
        elif loss_type == "CustomSoftmaxTablatureLoss":
            logger.info("--> Activating penalty-based loss: CustomSoftmaxTablatureLoss")
            return CustomSoftmaxTablatureLoss(
                miss_penalty=penalty_config.get('miss', 1.5),
                false_alarm_penalty=penalty_config.get('false_alarm', 1.5),
                substitution_penalty=penalty_config.get('substitution', 1.0),
                silence_class=silence_cls,
                ignore_index=-1
            )

        # 3. SADECE FOCAL LOSS DURUMU
        elif use_focal:
            logger.info("--> Activating standard Focal Loss: MultiClassFocalLoss")
            return MultiClassFocalLoss(
                gamma=focal_params.get('gamma', 2.0),
                ignore_index=-1
            )
        
        # 4. VARSAYILAN DURUM: Standart Cross Entropy
        elif loss_type == "CrossEntropyLoss":
            logger.info("--> Activating default loss: CrossEntropyLoss")
            return nn.CrossEntropyLoss(ignore_index=-1)
            
        else:
            raise ValueError(f"Unsupported loss type or combination for softmax_groups: {loss_type} with use_focal={use_focal}")

    def _create_helper_loss(self, loss_key: str):
        """Yardımcı görevler için loss fonksiyonu oluşturur."""
        loss_config = self.loss_config.get(loss_key, {})
        loss_type = loss_config.get('type', 'BCEWithLogitsLoss')
        logger.info(f"Creating helper loss '{loss_key}' of type: '{loss_type}'")
        if loss_type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported helper loss type: {loss_type}")

    def _setup_auxiliary_losses(self):
        """Tüm yardımcı loss'ları config'e göre ayarlar."""
        self.aux_enabled = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        if self.aux_enabled:
            self.auxiliary_loss_fn = self._create_helper_loss('auxiliary_loss')
            self.aux_weight = self.loss_config.get('auxiliary_loss', {}).get('weight', 0.4)

        self.onset_enabled = self.loss_config.get('onset_loss', {}).get('enabled', False)
        if self.onset_enabled:
            self.onset_loss_fn = self._create_helper_loss('onset_loss')
            self.onset_weight = self.loss_config.get('onset_loss', {}).get('weight', 0.5)

        self.offset_enabled = self.loss_config.get('offset_loss', {}).get('enabled', False)
        if self.offset_enabled:
            self.offset_loss_fn = self._create_helper_loss('offset_loss')
            self.offset_weight = self.loss_config.get('offset_loss', {}).get('weight', 0.5)
    
    def forward(self, model_output: dict, batch: dict) -> dict:
        # --- 1. Birincil (Primary) Loss Hesaplaması ---
        tab_logits = model_output['tab_logits']
        tablature_target = batch['tablature']

        if self.active_loss_strategy == 'softmax_groups':
            S = self.instrument_config['num_strings']
            C = self.config['model']['params']['num_classes']

            if tab_logits.dim() == 4: # FretNet gibi dinamik model çıktısı: (B, T, S, C)
                # Hedef verisini (B, S, T) -> (B*T, S) formatına dönüştür.
                # 'permute' işleminden sonra 'view' yerine 'reshape' kullanmak daha güvenlidir.
                tablature_target_reshaped = tablature_target.permute(0, 2, 1).reshape(-1, S)
                
                # Logits'i (B*T, S, C) formatına dönüştür
                logits_reshaped = tab_logits.reshape(-1, S, C)

            elif tab_logits.dim() == 2: # TabCNN gibi statik model çıktısı: (B*T, S*C)
                # Logits'i (B*T, S, C) formatına dönüştür
                logits_reshaped = tab_logits.reshape(-1, S, C)
                # Hedef veri (B*T, S) olarak geldiği varsayılır.
                tablature_target_reshaped = tablature_target
            else:
                raise ValueError(f"Desteklenmeyen 'tab_logits' boyutu: {tab_logits.dim()}")
            
            total_string_loss = 0.0
            use_class_weights = self.active_loss_config.get('use_class_weights', False)

            for s in range(S):
                pred_s = logits_reshaped[:, s, :]
                target_s = tablature_target_reshaped[:, s]

                class_weights_s = None
                if use_class_weights and self.class_weights is not None:
                    class_weights_s = self.class_weights[s].to(pred_s.device)

                if isinstance(self.primary_loss_fn, nn.CrossEntropyLoss):
                    loss_calculator = nn.CrossEntropyLoss(weight=class_weights_s, ignore_index=-1)
                    total_string_loss += loss_calculator(pred_s, target_s)
                else:
                    # Özel kayıp fonksiyonları için class_weights parametresini aktar
                    total_string_loss += self.primary_loss_fn(pred_s, target_s, class_weights=class_weights_s)
            
            primary_loss = total_string_loss / S
        else:
            # Diğer loss stratejileri için (örneğin 'logistic_bce')
            primary_loss = self.primary_loss_fn(tab_logits, tablature_target)

        # --- 2. Yardımcı (Auxiliary) Loss Hesaplamaları ve Toplam Loss'un Oluşturulması ---
        total_loss = primary_loss
        loss_dict = {"primary_loss": primary_loss.detach()}

        if self.aux_enabled and 'multipitch_logits' in model_output:
            multipitch_logits = model_output['multipitch_logits']
            multipitch_target = batch['multipitch_target']
            
            # Aux loss için boyutları eşleştir
            if multipitch_logits.dim() == 3 and multipitch_target.dim() == 3:
                # Logits boyutu (B, T, C_p) ise, hedefi (B, C_p, T) yap.
                # BCEWithLogitsLoss'un beklediği format (N, C) şeklindedir.
                # Bu nedenle permute gerekebilir.
                if multipitch_logits.shape[1] != multipitch_target.shape[1]:
                    multipitch_target = multipitch_target.permute(0, 2, 1)
            
            aux_loss = self.auxiliary_loss_fn(multipitch_logits, multipitch_target)
            total_loss += self.aux_weight * aux_loss
            loss_dict['aux_loss'] = aux_loss.detach()
                
        if self.onset_enabled and 'onset_logits' in model_output:
            onset_logits = model_output['onset_logits']
            onset_target = batch['onset_target']
            onset_loss = self.onset_loss_fn(onset_logits, onset_target)
            total_loss += self.onset_weight * onset_loss
            loss_dict['onset_loss'] = onset_loss.detach()

        if self.offset_enabled and 'offset_logits' in model_output:
            offset_logits = model_output['offset_logits']
            offset_target = batch['offset_target']
            offset_loss = self.offset_loss_fn(offset_logits, offset_target)
            total_loss += self.offset_weight * offset_loss
            loss_dict['offset_loss'] = offset_loss.detach()

        loss_dict['total_loss'] = total_loss
        return loss_dict
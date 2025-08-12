import torch
import numpy as np
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

from .utils.metrics import compute_tablature_metrics, compute_multipitch_metrics, finalize_output
from .utils.losses import SoftmaxGroups, LogisticBankLoss
from .utils.guitar_profile import GuitarProfile
from .utils.experiment import save_model_summary, generate_experiment_report 
from .utils.callbacks import EarlyStopping
from .utils.analyze_errors import analyze as run_error_analysis
from .utils.agt_tools import tablature_to_logistic, logistic_to_tablature

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, experiment_path, class_weights=None, writer=None, initial_history=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.experiment_path = experiment_path
        self.writer = writer
        self.class_weights = class_weights
        
        self.guitar_profile = GuitarProfile(self.config['instrument'])
        self.loss_config = self.config['loss']
        self.data_config = self.config['data']
        self.training_config = self.config['training']
        self.preparation_mode = self.data_config.get('preparation_mode', 'windowing')

        self.loss_fn = self._setup_loss_function()
        
        active_feature_name = self.data_config['active_feature']
        self.feature_key = self.config['feature_definitions'][active_feature_name]['key']

        self.history = initial_history if initial_history else self._create_empty_history()
        
        self.use_amp = self.training_config.get('use_mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = self._setup_early_stopping()

    def _setup_loss_function(self):
        loss_type = self.loss_config.get('type', 'softmax_groups')
        if loss_type == 'softmax_groups':
            return SoftmaxGroups(
                num_groups=self.config['instrument']['num_strings'],
                group_size=self.config['data']['num_classes']
            )
        elif loss_type == 'logistic_bank':
            return LogisticBankLoss(
                num_strings=self.config['instrument']['num_strings'],
                lmbda=self.loss_config.get('lmbda', 1.0)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def _setup_early_stopping(self):
        es_config = self.training_config.get('early_stopping', {})
        if es_config.get('enabled', False):
            monitor = es_config.get('monitor', 'val_tab_f1')
            print(f"Early stopping enabled. Monitoring '{monitor}'")
            return EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0),
                monitor=monitor,
                mode=es_config.get('mode', 'max')
            )
        return None

    def _create_empty_history(self):
        return {
            "train_loss": [], "val_loss": [], "train_tab_f1": [], "val_tab_f1": [],
            "train_tab_precision": [], "val_tab_precision": [], "train_tab_recall": [], "val_tab_recall": [],
            "train_mp_f1": [], "val_mp_f1": [], 
            "train_mp_precision": [], "val_mp_precision": [], 
            "train_mp_recall": [], "val_mp_recall": [],       
            "lr": []
        }
    
    def _compute_loss(self, logits, targets, batch):
        if isinstance(logits, dict):
            tab_logits = logits['tablature']
            onset_logits = logits['onsets']
            tab_targets = targets
            onset_targets = batch.get('onsets')

            if onset_targets is None:
                raise ValueError("Model is predicting onsets, but 'onsets' key not found in batch.")

            tab_loss = self._compute_single_loss(tab_logits, tab_targets, batch)
            onset_loss = self._compute_single_loss(onset_logits, onset_targets.to(self.device), batch)
            
            onset_weight = self.loss_config.get('onset_loss_weight', 1.0)
            return tab_loss + onset_weight * onset_loss
        else:
            return self._compute_single_loss(logits, targets, batch)

    def _compute_single_loss(self, logits, targets, batch):
        loss_type = self.loss_config.get('type')
        targets = targets.to(self.device)
        
        B, T, S, C_out = logits.shape

        if loss_type == 'softmax_groups':
            preds_reshaped = logits.reshape(B, T, -1)
            return self.loss_fn.get_loss(
                preds=preds_reshaped, 
                targets=targets, 
                class_weights=self.class_weights,
                focal=self.loss_config.get('use_focal', False),
                gamma=self.loss_config.get('focal_loss_gamma', 2.0)
            )
        
        elif loss_type == 'logistic_bank':
            labels_logistic = tablature_to_logistic(targets.permute(0, 2, 1), self.guitar_profile, silence=False)
            labels_logistic = labels_logistic.to(self.device, dtype=torch.float32)
            
            labels_logistic = labels_logistic.permute(0, 2, 1).reshape(B, T, -1)
            
            preds_reshaped = logits.reshape(B, T, -1)

            return self.loss_fn(preds_reshaped.reshape(-1, preds_reshaped.shape[-1]), 
                                labels_logistic.reshape(-1, labels_logistic.shape[-1]))
        
        return 0.0

    def _prepare_model_input(self, batch):
        return batch[self.feature_key].to(self.device)
    
    def _run_epoch(self, data_loader, is_training=True):
        self.model.train(is_training)
        
        total_loss = 0
        all_logits_list, all_targets_list = [], []

        desc = "Training" if is_training else "Validating"
        for batch in tqdm(data_loader, desc=desc):
            inputs = self._prepare_model_input(batch)
            targets_full = batch['tablature'].to(self.device) 
            
            with torch.set_grad_enabled(is_training):
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    if self.preparation_mode == 'windowing':
                        logits = self.model(inputs)
                        loss = self._compute_loss(logits, targets_full.permute(0, 2, 1), batch)
                        
                    elif self.preparation_mode == 'framify':
                        framify_win_size = self.data_config.get('framify_window_size', 9)
                        pad_amount = framify_win_size // 2
                        
                        inputs_padded = F.pad(inputs, (pad_amount, pad_amount), 'constant', 0)
                        
                        unfolded = inputs_padded.unfold(3, framify_win_size, 1) # (B, C, F, T_new, W)
                        unfolded = unfolded.permute(0, 3, 1, 2, 4) # (B, T, C, F, W)
                        B, T, C, n_freqs, W = unfolded.shape
                        
                        input_frames = unfolded.reshape(B * T, C, n_freqs, W)
                        target_frames = targets_full.permute(0, 2, 1).reshape(B * T, self.model.num_strings)

                        logits_flat = self.model(input_frames) # Model (B*T, S*C) veya (B*T, S*(C-1)) döndürür
                        
                        # Çıktıyı (B, T, S, C_out) formatına getirerek kayıp fonksiyonuyla uyumlu hale getir
                        # Not: C_out, silence sınıfı içerip içermemesine göre değişir
                        if self.loss_config['type'] == 'logistic_bank':
                            num_output_classes = self.model.num_classes - 1
                        else:
                            num_output_classes = self.model.num_classes
                            
                        logits = logits_flat.reshape(B, T, self.model.num_strings, num_output_classes)
                        
                        # Kayıp fonksiyonuna hedefleri (B, T, S) formatında ver
                        loss = self._compute_loss(logits, targets_full.permute(0, 2, 1), batch)

                    else:
                        raise ValueError(f"Unsupported preparation_mode: {self.preparation_mode}")

                if is_training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                
                if isinstance(logits, dict): 
                    logits = logits['tablature']

                # Logit'leri ve hedefleri (B, T, S, C) ve (B, T, S) formatında tutarlılık için yeniden şekillendir
                B, T, S, C_out = logits.shape
                all_logits_list.append(logits.reshape(-1, S, C_out).cpu().detach())
                all_targets_list.append(targets_full.permute(0, 2, 1).reshape(-1, S).cpu().detach())


        if not all_logits_list:
            return {'loss': 0, 'tab_f1': 0, 'tab_precision': 0, 'tab_recall': 0, 'mp_f1': 0, 'mp_precision': 0, 'mp_recall': 0}

        all_logits = torch.cat(all_logits_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)

        metrics = self._calculate_metrics(all_logits, all_targets)
        metrics['loss'] = total_loss / len(data_loader) 
        return metrics

    def train(self, start_epoch=0, best_val_metric=-1):
        self.model.to(self.device)
        monitor_metric_key = self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
        if best_val_metric == -1:
            best_val_metric = -np.inf if (self.early_stopper and self.early_stopper.mode == 'max') else np.inf

        for epoch in range(start_epoch, self.training_config['epochs']):
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics = self._run_epoch(self.val_loader, is_training=False)
            
            val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
            train_metrics['train_loss'] = train_metrics.pop('loss')

            self._update_history(train_metrics, val_metrics_prefixed)
            self._log_epoch(epoch, train_metrics, val_metrics_prefixed)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics_prefixed[monitor_metric_key])
            
            current_metric = val_metrics_prefixed[monitor_metric_key]
            is_best = (self.early_stopper.mode == 'max' and current_metric > best_val_metric) or \
                      (self.early_stopper.mode == 'min' and current_metric < best_val_metric)
            if is_best:
                best_val_metric = current_metric
            
            self._save_checkpoint(epoch, best_val_metric, is_best)

            if self.early_stopper:
                self.early_stopper(current_metric)
                if self.early_stopper.early_stop:
                    print("Early stopping triggered.")
                    break
        
        self._finalize_training()
        return self.model, self.history
    
    def _calculate_metrics(self, logits, targets):
        silence_class = self.data_config['silence_class']
        
        if self.loss_config['type'] == 'logistic_bank':
            preds_tab = logistic_to_tablature(
                torch.sigmoid(logits).cpu(), self.guitar_profile, silence=False
            )
        else:
            preds_tab = finalize_output(
                logits,
                silence_class=silence_class,
                return_shape="logits", 
                mask_silence=not self.config['metrics']['include_silence']
            )

        tab_metrics = compute_tablature_metrics(
            preds_tab, 
            targets,
            self.config['metrics']['include_silence']
        )
        
        mp_metrics = compute_multipitch_metrics(
            preds_tab, 
            targets, 
            self.guitar_profile
        )
        
        return {
            'tab_f1': tab_metrics['overall_f1'],
            'tab_precision': tab_metrics['overall_precision'],
            'tab_recall': tab_metrics['overall_recall'],
            'mp_f1': mp_metrics['multipitch_f1'],
            'mp_precision': mp_metrics['multipitch_precision'],
            'mp_recall': mp_metrics['multipitch_recall']
        }
      
    def _update_history(self, train_metrics, val_metrics):
        self.history["train_loss"].append(train_metrics['train_loss'])
        self.history["val_loss"].append(val_metrics['val_loss'])
        self.history["train_tab_f1"].append(train_metrics['tab_f1'])
        self.history["val_tab_f1"].append(val_metrics['val_tab_f1'])
        self.history["train_tab_precision"].append(train_metrics['tab_precision'])
        self.history["val_tab_precision"].append(val_metrics['val_tab_precision'])
        self.history["train_tab_recall"].append(train_metrics['tab_recall'])
        self.history["val_tab_recall"].append(val_metrics['val_tab_recall'])
        self.history["train_mp_f1"].append(train_metrics['mp_f1'])
        self.history["val_mp_f1"].append(val_metrics['val_mp_f1'])
        self.history["train_mp_precision"].append(train_metrics['mp_precision'])
        self.history["val_mp_precision"].append(val_metrics['val_mp_precision'])
        self.history["train_mp_recall"].append(train_metrics['mp_recall'])
        self.history["val_mp_recall"].append(val_metrics['val_mp_recall'])
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

    def _log_epoch(self, epoch, train_metrics, val_metrics):
        epoch_log = (
            f"Epoch {epoch+1:02d}/{self.training_config['epochs']} | "
            f"Train Loss: {train_metrics['train_loss']:.4f}, Train Tab F1: {train_metrics['tab_f1']:.4f}, Train MP F1: {train_metrics['mp_f1']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f}, Val Tab F1: {val_metrics['val_tab_f1']:.4f}, "
            f"Val MP F1: {val_metrics['val_mp_f1']:.4f}, Val MP P: {val_metrics['val_mp_precision']:.4f}, Val MP R: {val_metrics['val_mp_recall']:.4f} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        print(epoch_log)
        if self.writer:
            self.writer.add_scalars('Loss', {'train': train_metrics['train_loss'], 'validation': val_metrics['val_loss']}, epoch)
            self.writer.add_scalars('Tablature_F1', {'train': train_metrics['tab_f1'], 'validation': val_metrics['val_tab_f1']}, epoch)
            self.writer.add_scalars('Multi-pitch_F1', {'train': train_metrics['mp_f1'], 'validation': val_metrics['val_mp_f1']}, epoch)
            self.writer.add_scalars('Multi-pitch_Precision', {'train': train_metrics['mp_precision'], 'validation': val_metrics['val_mp_precision']}, epoch)
            self.writer.add_scalars('Multi-pitch_Recall', {'train': train_metrics['mp_recall'], 'validation': val_metrics['val_mp_recall']}, epoch)
            self.writer.add_scalar('Parameters/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
    def _save_checkpoint(self, epoch, best_val_metric, is_best):
        state = {
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': best_val_metric, 'history': self.history
        }
        latest_path = os.path.join(self.experiment_path, 'checkpoint_latest.pt')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.experiment_path, 'model_best.pt')
            torch.save(self.model.state_dict(), best_path)
            monitor_key = self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
            print(f"New best model with {monitor_key}={best_val_metric:.4f} saved to: {best_path}")

    def _finalize_training(self):
        print("\nTraining finished.")

        try:
            run_error_analysis(self.experiment_path, val_loader=self.val_loader)
        except Exception as e:
            print(f"\nERROR: Could not run final error analysis. Reason: {e}")

        if self.writer:
            self.writer.close()
        history_path = os.path.join(self.experiment_path, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved: {history_path}")
        print("Saving model summary...")
        save_model_summary(self.model, self.config, self.experiment_path)

        generate_experiment_report(
            model=self.model,
            history=self.history,
            val_loader=self.val_loader,
            config=self.config,
            experiment_path=self.experiment_path,
            device=self.device
        )
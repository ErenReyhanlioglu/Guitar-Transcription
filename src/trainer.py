import torch
import numpy as np
import os
import json
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

from src.utils.logger import describe
from src.utils.metrics import compute_tablature_metrics, compute_multipitch_metrics, finalize_output, compute_octave_tolerant_metrics, apply_duration_threshold
from src.utils.losses import SoftmaxGroups, LogisticBankLoss
from src.utils.guitar_profile import GuitarProfile
from src.utils.experiment import save_model_summary, generate_experiment_report 
from src.utils.callbacks import EarlyStopping
from src.utils.analyze_errors import analyze as run_error_analysis
from src.utils.agt_tools import tablature_to_logistic, logistic_to_tablature

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, experiment_path, class_weights=None, writer=None, initial_history=None, log_filter=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.log_filter = log_filter
        self.experiment_path = experiment_path
        self.writer = writer
        self.class_weights = class_weights
        
        self.guitar_profile = GuitarProfile(self.config['instrument'])
        self.loss_config = self.config['loss']
        self.data_config = self.config['data']
        self.training_config = self.config['training']
        self.preparation_mode = self.data_config['active_preparation_mode']
        
        self.loss_fn = self._setup_loss_function()
        
        active_feature_name = self.data_config['active_feature']
        self.feature_key = self.config['feature_definitions'][active_feature_name]['key']
        self.history = initial_history if initial_history else self._create_empty_history()
        
        self.use_amp = self.training_config.get('use_mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = self._setup_early_stopping()
        
        logger.info("Trainer initialized.")
        logger.info(f"  -> Using device: {self.device}")
        logger.info(f"  -> Mixed precision (AMP) enabled: {self.use_amp}")
        logger.info(f"  -> Data preparation mode: {self.preparation_mode}")

    def _setup_loss_function(self):
        loss_type = self.loss_config['active_loss']
        if loss_type == 'softmax_groups':
            return SoftmaxGroups(
                num_groups=self.config['instrument']['num_strings'],
                group_size=self.config['data']['num_classes']
            )
        elif loss_type == 'logistic_bank':
            return LogisticBankLoss(
                num_strings=self.config['instrument']['num_strings'],
                num_classes=self.config['data']['num_classes'],
                lmbda=self.loss_config.get('lmbda', 1.0),
                use_focal=self.loss_config.get('use_focal', False),
                focal_gamma=self.loss_config.get('focal_gamma', 2.0),
                focal_alpha=self.loss_config.get('focal_alpha', 0.25)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def _setup_early_stopping(self):
        es_config = self.training_config.get('early_stopping', {})
        if es_config.get('enabled', False):
            monitor = es_config.get('monitor', 'val_tab_f1')
            logger.info(f"Early stopping enabled. Monitoring '{monitor}' (patience={es_config.get('patience', 10)}, mode='{es_config.get('mode', 'max')}')")
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
            "train_mp_f1": [], "val_mp_f1": [], "train_mp_precision": [], "val_mp_precision": [], 
            "train_mp_recall": [], "val_mp_recall": [], "train_octave_f1": [], "val_octave_f1": [],
            "train_octave_precision": [], "val_octave_precision": [], "train_octave_recall": [], "val_octave_recall": [],
            "lr": [], "train_correctly_discarded_segments": [], "val_correctly_discarded_segments": [],
            "train_accidentally_discarded_segments": [], "val_accidentally_discarded_segments": []
        }
    
    def _compute_loss(self, logits, targets, batch):
        logger.debug(f"[_compute_loss] Received logits: {describe(logits)}, targets: {describe(targets)}")
        if isinstance(logits, dict):
            tab_logits = logits['tablature']
            onset_logits = logits.get('onsets')
            tab_targets = targets
            onset_targets = batch.get('onsets')
            
            if onset_logits is not None and onset_targets is None:
                raise ValueError("Model is predicting onsets, but 'onsets' key not found in batch.")

            tab_loss = self._compute_single_loss(tab_logits, tab_targets)
            logger.debug(f"  -> Tablature loss component: {tab_loss.item():.4f}")
            
            total_loss = tab_loss
            if onset_logits is not None:
                onset_loss = self._compute_single_loss(onset_logits, onset_targets.to(self.device))
                onset_weight = self.loss_config.get('onset_loss_weight', 1.0)
                logger.debug(f"  -> Onset loss component: {onset_loss.item():.4f} (weight: {onset_weight})")
                total_loss += onset_weight * onset_loss
            return total_loss
        else:
            return self._compute_single_loss(logits, targets)

    def _compute_single_loss(self, logits, targets):
        loss_type = self.loss_config['active_loss']
        targets = targets.to(self.device)
        logger.debug(f"[_compute_single_loss] Processing with '{loss_type}'. Logits: {describe(logits)}, Targets on device: {describe(targets)}")
        
        if loss_type == 'softmax_groups':
            B, T, S, C_out = logits.shape
            preds_reshaped = logits.reshape(B, T, -1) 
            return self.loss_fn(
                preds=preds_reshaped, targets=targets, class_weights=self.class_weights,
                use_focal=self.loss_config.get('use_focal', False), focal_gamma=self.loss_config.get('focal_gamma', 2.0)
            )
        
        elif loss_type == 'logistic_bank':
            targets_logistic = tablature_to_logistic(targets, self.model.num_strings, self.model.num_classes).to(self.device)
            logger.debug(f"  -> Converted targets to logistic bank format: {describe(targets_logistic)}")
            
            # FretNet/Transformer (4D) ve TabCNN (2D) çıktılarını yönet
            if len(logits.shape) == 4:
                preds_reshaped = logits.view(logits.size(0) * logits.size(1), -1)
            else:
                preds_reshaped = logits
            
            logger.debug(f"  -> Reshaped preds for loss_fn: {describe(preds_reshaped)}")
            return self.loss_fn(preds_reshaped, targets_logistic.view(-1, preds_reshaped.shape[-1]), class_weights=self.class_weights)

    def _prepare_model_input(self, batch):
        return batch[self.feature_key].to(self.device)
    
    def _run_epoch(self, data_loader, is_training=True):
        self.model.train(is_training)
        total_loss = 0
        all_logits_list, all_targets_list = [], []
        desc = "Training" if is_training else "Validating"
        
        for i, batch in enumerate(tqdm(data_loader, desc=desc, leave=False)):
            inputs = self._prepare_model_input(batch)
            targets_full = batch['tablature']
            
            if i == 0: logger.debug(f"First batch input: {describe(inputs)}, First batch targets: {describe(targets_full)}")
            
            with torch.set_grad_enabled(is_training):
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    if self.preparation_mode == 'windowing':
                        logits = self.model(inputs)
                        targets = targets_full.permute(0, 2, 1)
                        loss = self._compute_loss(logits, targets, batch)
                    
                    elif self.preparation_mode == 'framify':
                        framify_win_size = self.config['data'].get('framify_window_size', 9)
                        pad_amount = framify_win_size // 2
                        inputs_padded = F.pad(inputs, (0, 0, pad_amount, pad_amount), 'constant', 0)
                        logger.debug(f"  [Framify] Padded inputs: {describe(inputs_padded)}")
                        
                        unfolded = inputs_padded.unfold(2, framify_win_size, 1).permute(0, 2, 1, 3, 4)
                        logger.debug(f"  [Framify] Unfolded inputs: {describe(unfolded)}")
                        B, T, C, F, W = unfolded.shape
                        
                        input_frames = unfolded.reshape(B * T, C, F, W)
                        logger.debug(f"  [Framify] Reshaped for model input: {describe(input_frames)}")
                        
                        logits_flat = self.model(input_frames)
                        logger.debug(f"  [Framify] Flat logits from model: {describe(logits_flat)}")

                        num_output_classes = -1
                        if self.loss_config['active_loss'] == 'softmax_groups':
                            num_output_classes = self.config['data']['num_classes']
                        elif self.loss_config['active_loss'] == 'logistic_bank':
                            num_output_classes = self.config['data']['num_classes'] - 1
                            
                        logits = logits_flat.view(B, T, self.model.num_strings, num_output_classes)
                        logger.debug(f"  [Framify] Reshaped logits to 4D: {describe(logits)}")
                        
                        targets = targets_full.permute(0, 2, 1)
                        loss = self._compute_loss(logits, targets, batch)
                    else:
                        raise ValueError(f"Unsupported preparation_mode: {self.preparation_mode}")

                if is_training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item()
            
            final_logits = logits['tablature'] if isinstance(logits, dict) else logits
            B_dim, T_dim, S_dim, C_out_dim = final_logits.shape
            all_logits_list.append(final_logits.reshape(-1, S_dim, C_out_dim).cpu().detach())
            all_targets_list.append(targets_full.permute(0, 2, 1).reshape(-1, S_dim).cpu().detach())

        if not all_logits_list: return self._create_empty_history()

        all_logits = torch.cat(all_logits_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)
        logger.debug(f"Metrics calculation for {desc} epoch. All logits: {describe(all_logits)}, All targets: {describe(all_targets)}")
        
        metrics = self._calculate_metrics(all_logits, all_targets)
        metrics['loss'] = total_loss / len(data_loader)
        logger.debug(f"Epoch metrics calculated: {describe(metrics)}")
        return metrics

    def _calculate_metrics(self, logits, targets):
        logger.debug(f"[_calculate_metrics] Received concatenated inputs - logits: {describe(logits)}, targets: {describe(targets)}")
        silence_class = self.config['data']['silence_class']
        
        if self.loss_config['active_loss'] == 'logistic_bank':
            logits_flat = logits.reshape(logits.shape[0], -1)
            probs = torch.sigmoid(logits_flat)
            pred_threshold = self.config['post_processing'].get('prediction_threshold', 0.5)
            preds_tab_raw = logistic_to_tablature(
                probs, self.model.num_strings, self.model.num_classes, threshold=pred_threshold
            )
            logger.debug(f"  -> Converted logistic logits to tablature preds: {describe(preds_tab_raw)}")
        else: 
            preds_tab_raw = finalize_output(logits, silence_class=silence_class, return_shape="tablature")
        
        min_duration = self.config.get('post_processing', {}).get('min_duration_frames', 0)
        if min_duration > 0:
            preds_tab_processed, threshold_stats = apply_duration_threshold(preds_tab_raw, targets, min_duration, silence_class)
        else:
            preds_tab_processed, threshold_stats = preds_tab_raw, {'correctly_discarded_segments': 0, 'accidentally_discarded_segments': 0}
        
        tab_metrics = compute_tablature_metrics(preds_tab_processed, targets, self.config['metrics']['include_silence'])
        mp_metrics = compute_multipitch_metrics(preds_tab_processed, targets, self.guitar_profile)
        octave_metrics = compute_octave_tolerant_metrics(preds_tab_processed, targets, self.config['instrument']['tuning'], silence_class)
        
        final_metrics = {
            'tab_f1': tab_metrics.get('overall_f1', 0.0), 'tab_precision': tab_metrics.get('overall_precision', 0.0), 'tab_recall': tab_metrics.get('overall_recall', 0.0),
            'mp_f1': mp_metrics.get('multipitch_f1', 0.0), 'mp_precision': mp_metrics.get('multipitch_precision', 0.0), 'mp_recall': mp_metrics.get('multipitch_recall', 0.0),
            'octave_f1': octave_metrics.get('octave_f1', 0.0), 'octave_precision': octave_metrics.get('octave_precision', 0.0), 'octave_recall': octave_metrics.get('octave_recall', 0.0)
        }
        final_metrics.update(threshold_stats)
        return final_metrics
      
    def train(self, start_epoch=0, best_val_metric=-1):
        self.model.to(self.device)
        monitor_metric_key = self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
        if best_val_metric == -1:
            best_val_metric = -np.inf if (self.early_stopper and self.early_stopper.mode == 'max') else np.inf
        
        epochs = self.training_config['epochs']
        logger.info(f"--- Starting training for {epochs} epochs ---")

        for epoch in range(start_epoch, epochs):
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics = self._run_epoch(self.val_loader, is_training=False)
            
            val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
            train_metrics_prefixed = {f"train_{k}": v for k, v in train_metrics.items()}

            self._update_history(train_metrics_prefixed, val_metrics_prefixed)
            self._log_epoch(epoch, epochs, train_metrics_prefixed, val_metrics_prefixed)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau): self.scheduler.step(val_metrics_prefixed[monitor_metric_key])
                else: self.scheduler.step()
            
            current_metric = val_metrics_prefixed[monitor_metric_key]
            is_best = (self.early_stopper.mode == 'max' and current_metric > best_val_metric) or \
                      (self.early_stopper.mode == 'min' and current_metric < best_val_metric)
            if is_best:
                best_val_metric = current_metric
            
            self._save_checkpoint(epoch, best_val_metric, is_best)

            if self.early_stopper:
                self.early_stopper(current_metric)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break
        
        self._finalize_training()
        return self.model, self.history
    
    def _update_history(self, train_metrics, val_metrics):
        for key in self.history.keys():
            current_train_val = train_metrics.get(key)
            current_val_val = val_metrics.get(f"val_{key.split('_', 1)[-1]}" if key.startswith('train') else key)
            if key == 'lr':
                self.history[key].append(self.optimizer.param_groups[0]['lr'])
            elif key.startswith('train_'):
                self.history[key].append(current_train_val if current_train_val is not None else 0)
            elif key.startswith('val_'):
                self.history[key].append(current_val_val if current_val_val is not None else 0)

    def _log_epoch(self, epoch, total_epochs, train_metrics, val_metrics):
        if self.config.get('logging_and_checkpointing', {}).get('log_epoch_summary', True):
            lr = self.optimizer.param_groups[0]['lr']
            tm, vm = train_metrics, val_metrics
            
            log_message = (
                f"\n--- Epoch {epoch+1:02d}/{total_epochs} ---\n"
                f"LR: {lr:.6f}\n"
                f"Losses                | Train: {tm['train_loss']:.4f}, Validation: {vm['val_loss']:.4f}\n"
                f"-------------------------\n"
                f"Tab Metrics (Strict)\n"
                f"  ├─ F1                | Train: {tm['train_tab_f1']:.4f}, Validation: {vm['val_tab_f1']:.4f}\n"
                f"  ├─ Precision         | Train: {tm['train_tab_precision']:.4f}, Validation: {vm['val_tab_precision']:.4f}\n"
                f"  └─ Recall            | Train: {tm['train_tab_recall']:.4f}, Validation: {vm['val_tab_recall']:.4f}\n"
                f"Multi-pitch Metrics\n"
                f"  ├─ F1                | Train: {tm['train_mp_f1']:.4f}, Validation: {vm['val_mp_f1']:.4f}\n"
                f"  ├─ Precision         | Train: {tm['train_mp_precision']:.4f}, Validation: {vm['val_mp_precision']:.4f}\n"
                f"  └─ Recall            | Train: {tm['train_mp_recall']:.4f}, Validation: {vm['val_mp_recall']:.4f}\n"
                f"Octave Tolerant Metrics\n"
                f"  ├─ F1                | Train: {tm['train_octave_f1']:.4f}, Validation: {vm['val_octave_f1']:.4f}\n"
                f"  ├─ Precision         | Train: {tm['train_octave_precision']:.4f}, Validation: {vm['val_octave_precision']:.4f}\n"
                f"  └─ Recall            | Train: {tm['train_octave_recall']:.4f}, Validation: {vm['val_octave_recall']:.4f}"
            )

            if self.config.get('post_processing', {}).get('min_duration_frames', 0) > 0:
                train_correctly = int(tm.get('train_correctly_discarded_segments', 0))
                val_correctly = int(vm.get('val_correctly_discarded_segments', 0))
                train_accidentally = int(tm.get('train_accidentally_discarded_segments', 0))
                val_accidentally = int(vm.get('val_accidentally_discarded_segments', 0))
                log_message += (
                    f"\nPost-Processing Stats (Segments Discarded)\n"
                    f"  ├─ Correctly (FP)    | Train: {train_correctly}, Validation: {val_correctly}\n"
                    f"  └─ Accidentally (TP) | Train: {train_accidentally}, Validation: {val_accidentally}"
                )
            
            logger.info(log_message)

        if self.writer:
            epoch_num = epoch + 1
            self.writer.add_scalars('Loss', {'train': train_metrics['train_loss'], 'validation': val_metrics['val_loss']}, epoch_num)
            self.writer.add_scalars('Tablature_F1', {'train': train_metrics['train_tab_f1'], 'validation': val_metrics['val_tab_f1']}, epoch_num)
            self.writer.add_scalar('Parameters/learning_rate', self.optimizer.param_groups[0]['lr'], epoch_num)

    def _save_checkpoint(self, epoch, best_val_metric, is_best):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                 'best_val_metric': best_val_metric, 'history': self.history}
        latest_path = os.path.join(self.experiment_path, 'checkpoint_latest.pt')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.experiment_path, 'model_best.pt')
            torch.save(self.model.state_dict(), best_path)
            monitor_key = self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
            logger.info(f"New best model with {monitor_key}={best_val_metric:.4f} saved to: {best_path}")

    def _finalize_training(self):
        logger.info("--- Training finished ---")
        try:
            logger.info("Running final error analysis...")
            run_error_analysis(self.experiment_path, val_loader=self.val_loader)
        except Exception as e:
            logger.error(f"Could not run final error analysis. Reason: {e}", exc_info=True)
        if self.writer:
            self.writer.close()
        history_path = os.path.join(self.experiment_path, "history.json")
        with open(history_path, 'w') as f: json.dump(self.history, f, indent=4)
        logger.info(f"Training history saved: {history_path}")
        logger.info("Saving model summary...")
        save_model_summary(self.model, self.config, self.experiment_path)
        logger.info("Generating final experiment report...")
        generate_experiment_report(model=self.model, history=self.history, val_loader=self.val_loader,
                                   config=self.config, experiment_path=self.experiment_path, device=self.device)
        logger.info("Experiment finalized.")
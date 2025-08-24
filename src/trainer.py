import torch
import numpy as np
import os
import json
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.utils.logger import describe
from src.utils.metrics import compute_tablature_metrics, compute_multipitch_metrics, compute_octave_tolerant_metrics
from src.utils.losses import CombinedLoss
from src.utils.guitar_profile import GuitarProfile
from src.utils.experiment import save_model_summary, generate_experiment_report
from src.utils.callbacks import EarlyStopping
from src.utils.analyze_errors import analyze as run_error_analysis
from src.utils.agt_tools import logistic_to_tablature

logger = logging.getLogger(__name__)

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
        
        self.loss_config = self.config['loss']
        self.training_config = self.config['training']
        self.instrument_config = self.config['instrument']
        self.post_processing_config = self.config['post_processing']
        self.metrics_config = self.config['metrics']
        self.guitar_profile = GuitarProfile(self.instrument_config)
        
        self.loss_fn = CombinedLoss(config=self.config, class_weights=class_weights)
        self.history = initial_history if initial_history else self._create_empty_history()
        
        self.use_amp = self.training_config.get('use_mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = self._setup_early_stopping()
        
        logger.info("Trainer initialized with the new modular structure.")

    def _setup_early_stopping(self):
        es_config = self.training_config.get('early_stopping', {})
        if es_config.get('enabled', False):
            monitor = es_config.get('monitor', 'val_tab_f1')
            logger.info(f"Early stopping enabled. Monitoring '{monitor}'")
            return EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0),
                monitor=monitor,
                mode=es_config.get('mode', 'max')
            )
        return None

    def _create_empty_history(self):
        keys_with_phases = [
            "loss_total", "loss_primary", "loss_aux",
            "tab_f1", "tab_precision", "tab_recall",
            "tab_f1_macro", "tab_precision_macro", "tab_recall_macro", 
            "mp_f1", "mp_precision", "mp_recall",
            "octave_f1", "octave_precision", "octave_recall",
            "octave_f1_macro", "octave_precision_macro", "octave_recall_macro"
        ]
        
        history = {f"{phase}_{key}": [] for phase in ["train", "val"] for key in keys_with_phases}
        history["lr"] = [] 
        
        return history
    
    def _prepare_model_input(self, batch):
        return {key: tensor.to(self.device) for key, tensor in batch['features'].items()}
    
    def _run_epoch(self, data_loader, is_training=True):
        self.model.train(is_training)
        epoch_losses = {"total": 0.0, "primary": 0.0, "aux": 0.0}
        all_tab_logits_list, all_tab_targets_list = [], []
        
        aux_enabled = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        desc = "Training" if is_training else "Validating"
        
        for batch in tqdm(data_loader, desc=desc, leave=False):
            inputs = self._prepare_model_input(batch)
            for key in batch:
                if key != 'features':
                    batch[key] = batch[key].to(self.device)

            with torch.set_grad_enabled(is_training):
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    model_output = self.model(**inputs)
                    loss_dict = self.loss_fn(model_output, batch)
                    loss_to_backprop = loss_dict['total_loss']
                
                if is_training and torch.isfinite(loss_to_backprop):
                    self.scaler.scale(loss_to_backprop).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if torch.isfinite(loss_to_backprop):
                epoch_losses["total"] += loss_dict['total_loss'].item()
                epoch_losses["primary"] += loss_dict['primary_loss'].item()
                epoch_losses["aux"] += loss_dict['aux_loss'].item()
            
            tab_logits = model_output[0] if aux_enabled else model_output
            all_tab_logits_list.append(tab_logits.cpu().detach())
            all_tab_targets_list.append(batch['tablature'].cpu().detach())

        epoch_metrics = self._calculate_all_metrics(all_tab_logits_list, all_tab_targets_list)
        
        num_batches = len(data_loader) if len(data_loader) > 0 else 1
        epoch_metrics['loss_total'] = epoch_losses["total"] / num_batches
        epoch_metrics['loss_primary'] = epoch_losses["primary"] / num_batches
        epoch_metrics['loss_aux'] = epoch_losses["aux"] / num_batches
        
        return epoch_metrics

    def _calculate_all_metrics(self, logits_list, targets_list):
        all_logits = torch.cat(logits_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        S = self.instrument_config['num_strings']
        C = self.model.num_classes

        if self.loss_config['active_loss'] == 'softmax_groups':
            if all_logits.dim() == 4: # FretNet (B, T, S, C) output
                # (B, T, S, C) -> (B, S, T, C) -> (B, S, T)
                preds_tab = torch.argmax(all_logits.permute(0, 2, 1, 3), dim=-1)
            elif all_logits.dim() == 2: # TabCNN (B_flat, S*C) output
                # (B_flat, S*C) -> (B_flat, S, C) -> (B_flat, S)
                preds_tab = torch.argmax(all_logits.view(-1, S, C), dim=-1)
            else:
                raise ValueError(f"Unsupported all_logits dimension for metrics: {all_logits.dim()}")
        else: # logistic_bank
            # ... (Bu kısım gelecekteki logistic bank kullanımı için)
            pass

        logger.debug(f"Shape after argmax: preds_tab={preds_tab.shape}, all_targets={all_targets.shape}")

        # TabCNN target (B_flat, S)
        # FretNet target (B, S, T) 
        if all_targets.dim() == 3: # FretNet target
            targets_flat = all_targets.permute(0, 2, 1).reshape(-1, S)
        else: # TabCNN target
            targets_flat = all_targets

        # preds_tab de FretNet için (B,S,T)'den (B*T,S)'ye çevrilmeli
        if preds_tab.dim() == 3:
            preds_tab_flat = preds_tab.permute(0, 2, 1).reshape(-1, S)
        else:
            preds_tab_flat = preds_tab
        
        logger.debug(f"Shape after flattening for metrics: preds_tab_flat={preds_tab_flat.shape}, targets_flat={targets_flat.shape}")

        tab_metrics_raw = compute_tablature_metrics(preds_tab_flat, targets_flat, self.metrics_config.get('include_silence', False))
        mp_metrics_raw = compute_multipitch_metrics(preds_tab_flat, targets_flat, self.guitar_profile)
        octave_metrics_raw = compute_octave_tolerant_metrics(preds_tab_flat, targets_flat, self.instrument_config['tuning'], self.instrument_config['num_frets'] + 1)
        
        final_metrics = {
        'tab_f1': tab_metrics_raw.get('overall_f1', 0.0),
        'tab_precision': tab_metrics_raw.get('overall_precision', 0.0),
        'tab_recall': tab_metrics_raw.get('overall_recall', 0.0),
        'tab_f1_macro': tab_metrics_raw.get('overall_f1_macro', 0.0),
        'tab_precision_macro': tab_metrics_raw.get('overall_precision_macro', 0.0),
        'tab_recall_macro': tab_metrics_raw.get('overall_recall_macro', 0.0),
        
        'mp_f1': mp_metrics_raw.get('multipitch_f1', 0.0),
        'mp_precision': mp_metrics_raw.get('multipitch_precision', 0.0),
        'mp_recall': mp_metrics_raw.get('multipitch_recall', 0.0),
        
        'octave_f1': octave_metrics_raw.get('octave_f1', 0.0),
        'octave_precision': octave_metrics_raw.get('octave_precision', 0.0),
        'octave_recall': octave_metrics_raw.get('octave_recall', 0.0),
        'octave_f1_macro': octave_metrics_raw.get('octave_f1_macro', 0.0),
        'octave_precision_macro': octave_metrics_raw.get('octave_precision_macro', 0.0),
        'octave_recall_macro': octave_metrics_raw.get('octave_recall_macro', 0.0),
        }
        
        return final_metrics

    def train(self, start_epoch=0, best_val_metric=-1):
        self.model.to(self.device)
        monitor_metric_key = self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
        
        if best_val_metric == -1:
            best_val_metric = -np.inf if (self.early_stopper and self.early_stopper.mode == 'max') else np.inf
        
        epochs = self.training_config['epochs']
        logger.info(f"--- Starting training from epoch {start_epoch} for {epochs} total epochs ---")

        for epoch in range(start_epoch, epochs):
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics = self._run_epoch(self.val_loader, is_training=False)
            
            self._update_and_log_history(epoch, epochs, train_metrics, val_metrics)
            
            current_metric_val = val_metrics[monitor_metric_key.replace('val_','')]
            
            if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(current_metric_val)
            
            is_best = (self.early_stopper.mode == 'max' and current_metric_val > best_val_metric) or \
                      (self.early_stopper.mode == 'min' and current_metric_val < best_val_metric)

            if is_best:
                best_val_metric = current_metric_val
            
            self._save_checkpoint(epoch, best_val_metric, is_best)

            if self.early_stopper:
                self.early_stopper(current_metric_val)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break
        
        self._finalize_training()
        return self.model, self.history

    def _update_and_log_history(self, epoch, total_epochs, train_metrics, val_metrics):
        for key in self.history.keys():
            if key == 'lr':
                self.history[key].append(self.optimizer.param_groups[0]['lr'])
            elif key.startswith('train_'):
                self.history[key].append(train_metrics.get(key.replace('train_', ''), 0.0))
            elif key.startswith('val_'):
                self.history[key].append(val_metrics.get(key.replace('val_', ''), 0.0))

        lr = self.history['lr'][-1]
        tm, vm = train_metrics, val_metrics
        
        log_message = (
        f"\n--- Epoch {epoch+1:02d}/{total_epochs} ---\n"
        f"LR: {lr:.6f}\n"
        f"Losses              | Train: {tm['loss_total']:.4f}, Validation: {vm['loss_total']:.4f}\n" 
        f"--------------------------------------------------\n"
        f"Tab Metrics (F1 Weighted / Macro)\n"
        f"  ├─ F1              | Train: {tm.get('tab_f1', 0):.4f} / {tm.get('tab_f1_macro', 0):.4f}, Validation: {vm.get('tab_f1', 0):.4f} / {vm.get('tab_f1_macro', 0):.4f}\n"
        f"  ├─ Precision       | Train: {tm.get('tab_precision', 0):.4f}, Validation: {vm.get('tab_precision', 0):.4f}\n"
        f"  └─ Recall          | Train: {tm.get('tab_recall', 0):.4f}, Validation: {vm.get('tab_recall', 0):.4f}\n"
        f"Multi-pitch Metrics\n"
        f"  ├─ F1              | Train: {tm.get('mp_f1', 0):.4f}, Validation: {vm.get('mp_f1', 0):.4f}\n"
        f"  ├─ Precision       | Train: {tm.get('mp_precision', 0):.4f}, Validation: {vm.get('mp_precision', 0):.4f}\n"
        f"  └─ Recall          | Train: {tm.get('mp_recall', 0):.4f}, Validation: {vm.get('mp_recall', 0):.4f}\n"
        f"Octave Tolerant Metrics (F1 Weighted / Macro)\n"
        f"  ├─ F1              | Train: {tm.get('octave_f1', 0):.4f} / {tm.get('octave_f1_macro', 0):.4f}, Validation: {vm.get('octave_f1', 0):.4f} / {vm.get('octave_f1_macro', 0):.4f}\n"
        f"  ├─ Precision       | Train: {tm.get('octave_precision', 0):.4f}, Validation: {vm.get('octave_precision', 0):.4f}\n"
        f"  └─ Recall          | Train: {tm.get('octave_recall', 0):.4f}, Validation: {vm.get('octave_recall', 0):.4f}"
        )
        logger.info(log_message)

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
        if self.writer: self.writer.close()
        
        history_path = os.path.join(self.experiment_path, "history.json")
        try:
            with open(history_path, 'w') as f: json.dump(self.history, f, indent=4)
            logger.info(f"Training history saved: {history_path}")
        except Exception as e:
            logger.error(f"Could not save history to {history_path}. Reason: {e}")
            
        try:
            logger.info("Running final error analysis...")
            run_error_analysis(
                experiment_path=self.experiment_path,
                val_loader=self.val_loader,
            )
        except Exception as e:
            logger.error(f"Could not run final error analysis. Reason: {e}", exc_info=True)
            
        logger.info("Saving model summary...")
        save_model_summary(self.model, self.config, self.experiment_path)
        
        logger.info("Generating final experiment report...")
        try:
            generate_experiment_report(model=self.model, history=self.history, val_loader=self.val_loader,
                                      config=self.config, experiment_path=self.experiment_path, device=self.device,
                                      profile=self.guitar_profile)
        except Exception as e:
            logger.error(f"Could not generate experiment report. Reason: {e}", exc_info=True)

        logger.info("Experiment finalized.")

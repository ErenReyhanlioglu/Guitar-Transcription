# src/trainer.py

import torch
import numpy as np
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from .utils.metrics import accuracy, compute_tablature_metrics
from .utils.experiment import save_model_summary
from .utils.callbacks import EarlyStopping

class Trainer:
    """
    Handles the model training, validation, and checkpointing loop.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, experiment_path, class_weights=None, writer=None, initial_history=None):
        """
        Initializes the Trainer object.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.experiment_path = experiment_path
        self.class_weights = class_weights
        self.writer = writer

        if initial_history:
            self.history = initial_history
        else:
            self.history = {
                "train_loss": [], "val_loss": [], 
                "train_acc": [], "val_acc": [], 
                "train_f1": [], "val_f1": [],
                "lr": [] 
            }

        self.use_amp = self.config['training'].get('use_mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) is activated.")

        self.early_stopper = None
        es_config = self.config.get('training', {}).get('early_stopping', {})
        if es_config.get('enabled', False):
            self.early_stopper = EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0),
                monitor=es_config.get('monitor', 'val_loss'),
                mode=es_config.get('mode', 'min')
            )
            print(f"Early stopping enabled. Monitoring '{self.early_stopper.monitor}'")

    def _compute_loss(self, logits, targets):
        """Computes the loss using the model's SoftmaxGroups layer."""
        B, T, S, C = logits.shape
        logits_flat = logits.view(B, T, S * C)
        targets_permuted = targets.permute(0, 2, 1)
        
        loss = self.model.softmax_groups.get_loss(
            preds=logits_flat,
            targets=targets_permuted,
            class_weights=self.class_weights,
            focal=self.config['loss'].get('use_focal', True),
            gamma=self.config['loss'].get('focal_loss_gamma', 2.0),
            include_silence=self.config['data']['include_silence']
        )
        return loss

    def _evaluate(self, data_loader):
        """Evaluates the model on a given data loader and returns a dictionary of metrics."""
        self.model.eval()
        total_loss, total_batches = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in data_loader:
                cqt = batch["cqt"].to(self.device)
                tab = batch["tablature"].to(self.device)
                
                with autocast(enabled=self.use_amp):
                    logits = self.model(cqt, apply_softmax=False)
                    loss = self._compute_loss(logits, tab)

                total_loss += loss.item()
                total_batches += 1
                all_preds.append(logits.cpu())
                all_targets.append(tab.cpu())

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        
        if not all_preds:
            return {'val_loss': avg_loss, 'val_acc': 0, 'val_f1': 0}
            
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        include_silence = self.config['data']['include_silence']
        silence_class = self.config['data'].get('silence_class', 20)
        
        acc = accuracy(all_preds, all_targets, include_silence, silence_class).item()
        _, _, f1 = compute_tablature_metrics(all_preds, all_targets, include_silence, silence_class)

        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1}
    
    def _save_history(self):
        """Saves the training history dictionary to a JSON file."""
        history_path = os.path.join(self.experiment_path, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved: {history_path}")

    def _save_checkpoint(self, epoch, best_val_f1, is_best=False, is_interval=False):
        """Saves a training checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_f1': best_val_f1,
            'history': self.history
        }

        latest_path = os.path.join(self.experiment_path, 'checkpoint_latest.pt')
        torch.save(state, latest_path)
        
        if is_best:
            best_path = os.path.join(self.experiment_path, 'model_best.pt')
            torch.save(self.model.state_dict(), best_path)
            print(f"✨ New best model with F1={best_val_f1:.4f} saved to: {best_path}")
        
        if is_interval:
            interval_path = os.path.join(self.experiment_path, 'model_checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            os.makedirs(os.path.dirname(interval_path), exist_ok=True)
            torch.save(state, interval_path)
            print(f"Interval checkpoint for epoch {epoch+1} saved to: {interval_path}")


    def train(self, start_epoch=0, best_val_f1=-1):
        """Runs the main training loop."""
        self.model.to(self.device)

        for epoch in range(start_epoch, self.config['training']['epochs']):
            self.model.train()
            train_losses = []

            for batch in self.train_loader:
                cqt = batch["cqt"].to(self.device)
                tab = batch["tablature"].to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True) 
                
                with autocast(enabled=self.use_amp):
                    logits = self.model(cqt, apply_softmax=False)
                    loss = self._compute_loss(logits, tab)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_losses.append(loss.item())
                        
            avg_train_loss = np.mean(train_losses)
            val_metrics = self._evaluate(self.val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_metrics['val_loss'])
            self.history["val_acc"].append(val_metrics['val_acc'])
            self.history["val_f1"].append(val_metrics['val_f1'])
            self.history['lr'].append(current_lr)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    monitor_key = self.config.get('training', {}).get('scheduler', {}).get('params', {}).get('monitor', 'val_loss')
                    self.scheduler.step(val_metrics[monitor_key])
                else:
                    self.scheduler.step()

            print(f"Epoch {epoch+1:02d}/{self.config['training']['epochs']} -> "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, Val F1: {val_metrics['val_f1']:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('F1_Score/validation', val_metrics['val_f1'], epoch)
                self.writer.add_scalar('Accuracy/validation', val_metrics['val_acc'], epoch)
                self.writer.add_scalar('Parameters/learning_rate', current_lr, epoch)

            is_best = val_metrics['val_f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['val_f1']
            
            interval = self.config.get('training', {}).get('logging_and_checkpointing', {}).get('save_epoch_checkpoint_interval', 0)
            is_interval = interval > 0 and (epoch + 1) % interval == 0

            self._save_checkpoint(epoch, best_val_f1, is_best=is_best, is_interval=is_interval)

            if self.early_stopper:
                monitor_metric_value = val_metrics[self.early_stopper.monitor]
                self.early_stopper(monitor_metric_value)
                if self.early_stopper.early_stop:
                    print("Early stopping criteria met. Ending training.")
                    break

        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")
            
        self._save_history()
        print("Saving final model summary...")
        save_model_summary(self.model, self.config, self.experiment_path)

        return self.model, self.history
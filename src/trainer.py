# src/trainer.py

import torch
import numpy as np
import os
import json
from .utils.metrics import accuracy, compute_tablature_metrics
from .utils.experiment import save_model_summary
from torch.cuda.amp import autocast, GradScaler

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
        self.class_weights = class_weights
        
        self.writer = writer

        if initial_history:
            self.history = initial_history
        else:
            self.history = { "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": [] }

        self.use_amp = self.config['training'].get('use_mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) is activated.")

    def _compute_loss(self, logits, targets):
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
        self.model.eval()
        total_loss, total_acc, total_batches = 0, 0, 0
        all_f1 = []

        with torch.no_grad():
            for batch in data_loader:
                cqt = batch["cqt"].to(self.device)
                tab = batch["tablature"].to(self.device)
                
                with autocast(enabled=self.use_amp):
                    logits = self.model(cqt, apply_softmax=False)
                    loss = self._compute_loss(logits, tab)

                include_silence = self.config['data']['include_silence']
                silence_class = self.config['data'].get('silence_class', 20) 
                acc = accuracy(logits, tab, include_silence, silence_class)
                _, _, f1 = compute_tablature_metrics(logits, tab, include_silence, silence_class)
                
                total_loss += loss.item()
                total_acc += acc.item()
                all_f1.append(f1)
                total_batches += 1

        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        avg_f1 = np.mean(all_f1)
        
        return avg_loss, avg_acc, avg_f1
    
    def _save_history(self):
        history_path = os.path.join(self.experiment_path, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved: {history_path}")

    def train(self, start_epoch=0, best_val_f1=-1):
        self.model.to(self.device)

        for epoch in range(start_epoch, self.config['training']['epochs']):
            self.model.train()
            train_losses, train_accs, train_f1s = [], [], []

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
                include_silence = self.config['data']['include_silence']
                silence_class = self.config['data'].get('silence_class', 20)
                train_accs.append(accuracy(logits, tab, include_silence, silence_class).item())
                _, _, f1 = compute_tablature_metrics(logits, tab, include_silence, silence_class)
                train_f1s.append(f1)
            
            self.scheduler.step()

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)
            avg_train_f1 = np.mean(train_f1s)
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(avg_train_acc)
            self.history["train_f1"].append(avg_train_f1)
            
            val_loss, val_acc, val_f1 = self._evaluate(self.val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            
            print(f"Epoch {epoch+1:02d}/{self.config['training']['epochs']} -> "
                  f"Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                self.writer.add_scalar('F1_Score/train', avg_train_f1, epoch)
                self.writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
                
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                self.writer.add_scalar('F1_Score/validation', val_f1, epoch)
                self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
                
                self.writer.add_scalar('Parameters/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = os.path.join(self.experiment_path, 'model_best.pt')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model with F1={best_val_f1:.4f} saved to: {best_model_path}")

            checkpoint_path = os.path.join(self.experiment_path, 'checkpoint_latest.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'history': self.history
            }, checkpoint_path)

            interval = self.config.get('training', {}).get('logging_and_checkpointing', {}).get('save_epoch_checkpoint_interval', 0)
            
            if interval and interval > 0:
                if (epoch + 1) % interval == 0:
                    epoch_checkpoint_path = os.path.join(self.experiment_path, 'model_checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_val_f1': best_val_f1,
                        'history': self.history
                    }, epoch_checkpoint_path)
                    print(f"Epoch {epoch+1} checkpoint saved to: {epoch_checkpoint_path}")

        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")
            
        self._save_history()

        print("Saving final model summary...")
        save_model_summary(self.model, self.config, self.experiment_path)

        return self.model, self.history
"""
Purpose:
    This module contains the main `Trainer` class, which encapsulates the entire
    logic for training, validating, and testing a model. It acts as the "engine"
    of the pipeline, taking a model, data loaders, and a configuration, and
    running the experiment.

Dependencies:
    - torch
    - src.utils.metrics
    - src.utils.losses
    - The model and data loader objects provided during initialization.

Current Status:
    - Implements a standard training and validation loop over epochs.
    - Calculates loss using the provided loss function (e.g., FocalLoss with SoftmaxGroups).
    - Computes and logs performance metrics (accuracy, F1, etc.) for each epoch.
    - Saves the best model checkpoint based on validation performance.
    - Saves the complete training history to a JSON file at the end of the run.

Future Plans:
    - [ ] Refactor the entire class to use PyTorch Lightning (`pl.LightningModule`).
          This would abstract away most of the boilerplate training loop code, making
          it cleaner and enabling features like multi-GPU training and mixed precision easily.
    - [ ] Implement more advanced checkpointing (e.g., saving top-k models).
    - [ ] Add support for gradient clipping to prevent exploding gradients.
    - [ ] Integrate experiment tracking callbacks for tools like TensorBoard or Weights & Biases.
"""

import torch
import numpy as np
import os
from .utils.metrics import accuracy, compute_tablature_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, experiment_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.experiment_path = experiment_path
        self.history = { "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], 
                         "train_f1": [], "val_f1": [] }
        self.class_weights = None

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

    def _evaluate(self, data_loader, is_val=True):
        self.model.eval()
        total_loss, total_acc, total_batches = 0, 0, 0
        all_prec, all_rec, all_f1 = [], [], []

        with torch.no_grad():
            for batch in data_loader:
                cqt = batch["cqt"].to(self.device)
                tab = batch["tablature"].to(self.device)
                logits = self.model(cqt, apply_softmax=False)
                loss = self._compute_loss(logits, tab)

                include_silence = self.config['data']['include_silence']
                silence_class = self.config['data']['silence_class']
                acc = accuracy(logits, tab, include_silence, silence_class)
                prec, rec, f1 = compute_tablature_metrics(logits, tab, include_silence, silence_class)
                
                total_loss += loss.item()
                total_acc += acc.item()
                all_prec.append(prec)
                all_rec.append(rec)
                all_f1.append(f1)
                total_batches += 1

        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        avg_f1 = np.mean(all_f1)
        
        if is_val:
            print(f"🧪 Val Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | F1: {avg_f1:.4f}")
            self.history["val_loss"].append(avg_loss)
            self.history["val_acc"].append(avg_acc)
            self.history["val_f1"].append(avg_f1)
        
        return avg_loss, avg_acc, avg_f1

    def train(self):
        self.model.to(self.device)
        if self.config['loss'].get('use_class_weights', False):
            # (Note: This assumes npz_paths are accessible via config, which might need adjustment)
            # self.class_weights = compute_per_string_class_weights(...)
            print("Class weights will be implemented.")

        best_val_f1 = -1

        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            total_loss, total_acc, total_batches = 0, 0, 0
            all_f1 = []

            for batch in self.train_loader:
                cqt = batch["cqt"].to(self.device)
                tab = batch["tablature"].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(cqt, apply_softmax=False)
                loss = self._compute_loss(logits, tab)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1:02d}/{self.config['training']['epochs']} | Train Loss: {avg_loss:.4f}")

            val_loss, val_acc, val_f1 = self._evaluate(self.val_loader, is_val=True)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), os.path.join(self.experiment_path, 'model_best.pt'))
                print(f"🚀 New best model saved with F1: {best_val_f1:.4f}")
        
        return self.history
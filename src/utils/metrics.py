"""
Purpose:
    This module is a collection of functions used to evaluate the performance
    of the transcription models.

Dependencies:
    - torch
    - sklearn.metrics

Current Status:
    - `accuracy`: Computes overall frame-wise accuracy.
    - `compute_tablature_metrics`: Computes macro-averaged Precision, Recall, and F1-score.
    - `stringwise_classification_report`: Generates a detailed report of metrics for each
      of the 6 strings individually.

Future Plans:
    - [ ] Implement more advanced transcription-specific metrics, such as onset-aware
          F1-scores or multi-pitch evaluation metrics.
    - [ ] Add metrics that allow for a tolerance of +/- 1 fret.
"""

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

def finalize_output(logits, silence_class, return_shape="targets", mask_silence=False):
    preds = logits.argmax(dim=-1)
    if mask_silence:
        preds[preds == silence_class] = -1
    if return_shape == "targets":
        preds = preds.permute(0, 2, 1)
    return preds

def accuracy(logits, targets, include_silence, silence_class):
    preds = finalize_output(logits, return_shape="logits", mask_silence=not include_silence, silence_class=silence_class)
    targets = targets.permute(0, 2, 1)

    if not include_silence:
        mask = (targets != -1)
        if mask.sum() == 0:
            return torch.tensor(1.0)
        correct = (preds == targets) & mask
        return correct.sum().float() / mask.sum().float()
    else:
        correct = (preds == targets)
        return correct.sum().float() / targets.numel()

def compute_tablature_metrics(logits, targets, include_silence, silence_class):
    preds = finalize_output(logits, return_shape="logits", mask_silence=not include_silence, silence_class=silence_class)
    targets = targets.permute(0, 2, 1)

    if not include_silence:
        mask = (targets != -1)
        if mask.sum() == 0:
            return 1.0, 1.0, 1.0
        preds_flat = preds[mask].cpu().numpy()
        targets_flat = targets[mask].cpu().numpy()
    else:
        preds_flat = preds.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()

    precision = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)

    return precision, recall, f1

def compute_per_string_class_weights(
    npz_path_list, num_strings, num_classes,
    silence_class, silence_factor, include_silence
):
    string_counters = [Counter() for _ in range(num_strings)]

    for path in npz_path_list:
        data = np.load(path, allow_pickle=True)
        if "tablature" not in data:
            continue

        tab = data["tablature"]
        tab = np.where(tab == -1, silence_class, tab)

        for s in range(num_strings):
            string_counters[s].update(tab[s].tolist())

    weights = []
    for s in range(num_strings):
        total_counts = string_counters[s]
        class_weights = []

        for cls in range(num_classes):
            count = total_counts.get(cls, 1)
            w = 1.0 / np.log1p(count)
            class_weights.append(w)

        w_tensor = torch.tensor(class_weights, dtype=torch.float32)

        if silence_class < len(w_tensor):
            if include_silence:
                w_tensor[silence_class] *= silence_factor
            else:
                w_tensor[silence_class] = 0.0

        w_tensor = w_tensor / w_tensor.sum() * num_classes
        weights.append(w_tensor)

    return weights
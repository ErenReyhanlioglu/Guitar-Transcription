import torch
import numpy as np
import scipy.ndimage as ndimage
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from tqdm import tqdm

from .agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch

def finalize_output(logits, silence_class, return_shape="targets", mask_silence=False):
    preds = logits.argmax(dim=-1)
    
    if mask_silence:
        preds[preds == silence_class] = -1
        
    if return_shape == "targets":
        preds = preds.permute(0, 2, 1) # (B, T, S) -> (B, S, T)
        
    return preds

def accuracy(logits, targets, include_silence, silence_class):
    preds = finalize_output(logits, silence_class, return_shape="logits", mask_silence=not include_silence)

    if not include_silence:
        mask = (targets != -1)
        if mask.sum() == 0:
            return torch.tensor(1.0) 
        correct = (preds == targets) & mask
        return correct.sum().float() / mask.sum().float()
    else:
        correct = (preds == targets)
        return correct.sum().float() / targets.numel()

def compute_tablature_metrics(preds, targets, include_silence):
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    results = {}
    preds_flat = preds_np.flatten()
    targets_flat = targets_np.flatten()
    
    if not include_silence:
        mask = (targets_flat != -1)
        if mask.sum() == 0: 
             results['overall_f1'] = 1.0
             results['overall_precision'] = 1.0
             results['overall_recall'] = 1.0
        else:
            preds_flat = preds_flat[mask]
            targets_flat = targets_flat[mask]
    
    if len(targets_flat) > 0:
        p = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
        r = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
        f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
        results['overall_f1'] = f1
        results['overall_precision'] = p
        results['overall_recall'] = r

    per_string_metrics = {}
    if targets_np.ndim > 1: 
        num_strings = targets_np.shape[1] 
        for s in range(num_strings):
            preds_s = preds_np[:, s].flatten()
            targets_s = targets_np[:, s].flatten()

            if not include_silence:
                mask_s = (targets_s != -1)
                if mask_s.sum() == 0:
                    continue 
                preds_s = preds_s[mask_s]
                targets_s = targets_s[mask_s]
            
            if len(targets_s) > 0:
                f1_s = f1_score(targets_s, preds_s, average='macro', zero_division=0)
                per_string_metrics[f'string_{s}_f1'] = f1_s
    
    results['per_string'] = per_string_metrics
    
    return results

def compute_multipitch_metrics(preds_tab, targets, profile):
    preds_tab_transposed = torch.from_numpy(preds_tab.cpu().numpy().T)
    targets_transposed = torch.from_numpy(targets.cpu().numpy().T)

    preds_smp = tablature_to_stacked_multi_pitch(preds_tab_transposed, profile)
    targets_smp = tablature_to_stacked_multi_pitch(targets_transposed, profile)
    
    preds_mp = stacked_multi_pitch_to_multi_pitch(preds_smp)
    targets_mp = stacked_multi_pitch_to_multi_pitch(targets_smp)
    
    preds_flat = preds_mp.flatten()
    targets_flat = targets_mp.flatten()
    
    p = precision_score(targets_flat, preds_flat, average='binary', zero_division=0)
    r = recall_score(targets_flat, preds_flat, average='binary', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='binary', zero_division=0)
    
    return {'multipitch_f1': f1, 'multipitch_precision': p, 'multipitch_recall': r}

from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

def compute_per_string_class_weights(
    npz_path_list: list[str], 
    num_strings: int, 
    num_classes: int,
    silence_class: int, 
    silence_factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    string_counters = [Counter() for _ in range(num_strings)]

    print("Calculating class weights from dataset...")
    for path in tqdm(npz_path_list, desc="Analyzing files"):
        try:
            with np.load(path, allow_pickle=True) as data:
                if "tablature" not in data:
                    continue

                tab = data["tablature"]
                tab = np.where(tab == -1, silence_class, tab)

                for s in range(num_strings):
                    string_counters[s].update(tab[s].tolist())
        except Exception as e:
            print(f"Warning: Could not process file {path}. Error: {e}")
            continue

    final_weights = []
    final_counts = []

    for s in range(num_strings):
        total_counts_for_string = string_counters[s]
        
        current_string_weights = []
        current_string_counts = []

        for cls in range(num_classes):
            count = total_counts_for_string.get(cls, 0)
            
            if count == 0:
                weight = 0.0
            else:
                weight = 1.0 / (np.log1p(count) + 1e-9)
            
            current_string_weights.append(weight)
            current_string_counts.append(count)

        weights_tensor = torch.tensor(current_string_weights, dtype=torch.float32)

        if silence_class < len(weights_tensor):
            weights_tensor[silence_class] *= silence_factor

        if weights_tensor.sum() > 0:
            weights_tensor = (weights_tensor / weights_tensor.sum()) * num_classes
        
        final_weights.append(weights_tensor)
        final_counts.append(torch.tensor(current_string_counts, dtype=torch.int32))
        
    return torch.stack(final_weights), torch.stack(final_counts)

def compute_octave_tolerant_metrics(preds_tab, targets_tab, tuning, silence_class, num_pitch_classes=12):
    """
    Computes F1, Precision, and Recall scores that are tolerant to octave errors.
    The new representation for a note is (string_index, pitch_class).
    Silence is treated as its own unique class.
    
    Args:
        preds_tab (torch.Tensor): Predictions tensor of shape (N, num_strings).
        targets_tab (torch.Tensor): Targets tensor of shape (N, num_strings).
        tuning (list or np.array): MIDI numbers for open strings.
        silence_class (int): The integer label for the silence class.
        num_pitch_classes (int): Number of pitch classes in an octave (usually 12).
        
    Returns:
        dict: A dictionary containing octave-tolerant F1, precision, and recall.
    """
    if not isinstance(preds_tab, torch.Tensor):
        preds_tab = torch.from_numpy(preds_tab)
    if not isinstance(targets_tab, torch.Tensor):
        targets_tab = torch.from_numpy(targets_tab)
        
    device = preds_tab.device
    tuning_tensor = torch.tensor(tuning, device=device).unsqueeze(0)
    
    valid_preds_mask = (preds_tab != silence_class) & (preds_tab != -1)
    valid_targets_mask = (targets_tab != silence_class) & (targets_tab != -1)

    midi_preds_full = tuning_tensor + preds_tab
    midi_targets_full = tuning_tensor + targets_tab

    midi_preds = torch.zeros_like(preds_tab)
    midi_targets = torch.zeros_like(targets_tab)
    midi_preds[valid_preds_mask] = midi_preds_full[valid_preds_mask]
    midi_targets[valid_targets_mask] = midi_targets_full[valid_targets_mask]

    octave_preds = midi_preds % num_pitch_classes
    octave_preds[preds_tab == silence_class] = num_pitch_classes
    octave_preds[preds_tab == -1] = -1

    octave_targets = midi_targets % num_pitch_classes
    octave_targets[targets_tab == silence_class] = num_pitch_classes
    octave_targets[targets_tab == -1] = -1

    preds_flat = octave_preds.cpu().numpy().flatten()
    targets_flat = octave_targets.cpu().numpy().flatten()
    
    active_mask = (targets_flat != -1)
    preds_flat = preds_flat[active_mask]
    targets_flat = targets_flat[active_mask]
    
    if len(targets_flat) == 0:
        return {'octave_f1': 1.0, 'octave_precision': 1.0, 'octave_recall': 1.0}
        
    p = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    r = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
    
    return {'octave_f1': f1, 'octave_precision': p, 'octave_recall': r}

def apply_duration_threshold(preds_tab, targets, min_duration_frames, silence_class):
    """
    Applies a duration-based threshold and tracks its effects.

    Args:
        preds_tab (torch.Tensor): Raw tablature predictions, shape (N, num_strings).
        targets (torch.Tensor): Ground truth tablature, shape (N, num_strings).
        min_duration_frames (int): Minimum duration to keep a note segment.
        silence_class (int): Integer for the silence class.

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The processed tablature tensor.
            - A dictionary with statistics about discarded segments.
    """
    processed_preds = preds_tab.clone()
    num_strings = preds_tab.shape[1]

    stats = {
        'correctly_discarded_segments': 0, 
        'accidentally_discarded_segments': 0 
    }

    for s in range(num_strings):
        string_preds = preds_tab[:, s]
        string_targets = targets[:, s]
        
        labels, num_features = ndimage.label(string_preds != silence_class)
        
        if num_features > 0:
            segment_sizes = np.bincount(labels.ravel())
            small_segments_labels = np.where(segment_sizes < min_duration_frames)[0]
            
            for label in small_segments_labels:
                if label == 0: continue

                segment_mask = (labels == label)
                
                predicted_fret = string_preds[segment_mask][0].item()
                
                ground_truth_in_segment = string_targets[segment_mask]
                
                correct_frames_in_segment = torch.sum(ground_truth_in_segment == predicted_fret)
                is_segment_correct = (correct_frames_in_segment.item() / len(ground_truth_in_segment)) >= 0.5

                if is_segment_correct:
                    stats['accidentally_discarded_segments'] += 1
                else:
                    stats['correctly_discarded_segments'] += 1
                
                processed_preds[:, s][segment_mask] = silence_class
                
    return processed_preds, stats
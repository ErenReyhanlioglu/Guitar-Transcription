import torch
import numpy as np
import scipy.ndimage as ndimage
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from tqdm import tqdm
import logging

from .logger import describe
from .agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch

logger = logging.getLogger(__name__)

def finalize_output(logits, silence_class, return_shape="targets", mask_silence=False):
    logger.debug(f"[finalize_output] Input logits: {describe(logits)}")
    preds = logits.argmax(dim=-1)
    if mask_silence:
        preds[preds == silence_class] = -1
    if return_shape == "targets":
        preds = preds.permute(0, 2, 1)
    logger.debug(f"[finalize_output] Returning preds: {describe(preds)}")
    return preds

def accuracy(logits, targets, include_silence, silence_class):
    logger.debug(f"[accuracy] Inputs - logits: {describe(logits)}, targets: {describe(targets)}")
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
    logger.debug(f"[tab_metrics] Inputs - preds: {describe(preds)}, targets: {describe(targets)}")
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    results = {}
    
    preds_flat = preds_np.flatten()
    targets_flat = targets_np.flatten()
    
    if not include_silence:
        mask = (targets_flat != -1) 
        if mask.sum() == 0: 
            return {'overall_f1': 1.0, 'overall_precision': 1.0, 'overall_recall': 1.0,
                    'overall_f1_macro': 1.0, 'overall_precision_macro': 1.0, 'overall_recall_macro': 1.0,
                    'per_string': {}}
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]
        
    if len(targets_flat) > 0:
        p_w = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0)
        r_w = recall_score(targets_flat, preds_flat, average='weighted', zero_division=0)
        f1_w = f1_score(targets_flat, preds_flat, average='weighted', zero_division=0)
        
        p_m = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
        r_m = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
        f1_m = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
        
        results['overall_precision'] = p_w
        results['overall_recall'] = r_w
        results['overall_f1'] = f1_w
        results['overall_precision_macro'] = p_m
        results['overall_recall_macro'] = r_m
        results['overall_f1_macro'] = f1_m

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
                f1_s_w = f1_score(targets_s, preds_s, average='weighted', zero_division=0)
                f1_s_m = f1_score(targets_s, preds_s, average='macro', zero_division=0)
                per_string_metrics[f'string_{s}_f1'] = f1_s_w 
                per_string_metrics[f'string_{s}_f1_macro'] = f1_s_m 
                
    results['per_string'] = per_string_metrics
    logger.debug(f"[tab_metrics] Returning results: {describe(results)}")
    return results

def compute_multipitch_metrics(preds_tab, targets, profile):
    logger.debug(f"[mp_metrics] Inputs - preds_tab: {describe(preds_tab)}, targets: {describe(targets)}")
    
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
    
    results = {'multipitch_f1': f1, 'multipitch_precision': p, 'multipitch_recall': r}
    logger.debug(f"[mp_metrics] Returning results: {describe(results)}")
    return results

def compute_octave_tolerant_metrics(preds_tab, targets_tab, tuning, silence_class, num_pitch_classes=12):
    if not isinstance(preds_tab, torch.Tensor): preds_tab = torch.from_numpy(preds_tab)
    if not isinstance(targets_tab, torch.Tensor): targets_tab = torch.from_numpy(targets_tab)
    logger.debug(f"[octave_metrics] Inputs - preds_tab: {describe(preds_tab)}, targets_tab: {describe(targets_tab)}")
    
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
    octave_preds[~valid_preds_mask] = -1
    octave_preds[preds_tab == silence_class] = num_pitch_classes 

    octave_targets = midi_targets % num_pitch_classes
    octave_targets[~valid_targets_mask] = -1
    octave_targets[targets_tab == silence_class] = num_pitch_classes

    preds_flat = octave_preds.cpu().numpy().flatten()
    targets_flat = octave_targets.cpu().numpy().flatten()
    
    active_mask = (targets_flat != -1) & (targets_flat != num_pitch_classes)
    preds_flat = preds_flat[active_mask]
    targets_flat = targets_flat[active_mask]
    
    if len(targets_flat) == 0:
        return {'octave_f1': 1.0, 'octave_precision': 1.0, 'octave_recall': 1.0,
                'octave_f1_macro': 1.0, 'octave_precision_macro': 1.0, 'octave_recall_macro': 1.0}
    
    p_w = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    r_w = recall_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    f1_w = f1_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    
    p_m = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    r_m = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    f1_m = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
    
    return {'octave_precision': p_w, 'octave_recall': r_w, 'octave_f1': f1_w,
            'octave_precision_macro': p_m, 'octave_recall_macro': r_m, 'octave_f1_macro': f1_m}

def compute_tablature_error_scores(preds, targets, silence_class):
    """
    Calculates frame-wise tablature error scores based on predictions and targets.
    All inputs are expected to be 2D Tensors/Arrays of shape (N, S).
    """
    logger.debug(f"[error_scores] Computing error scores...")
    
    if isinstance(preds, torch.Tensor):
        preds_np = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()

    preds_flat = preds_np.flatten()
    targets_flat = targets_np.flatten()

    n_ref = np.sum(targets_flat != silence_class)

    if n_ref == 0:
        n_est = np.sum(preds_flat != silence_class)
        e_fa = n_est / (n_est + 1e-9) 
        return {
            'tab_error_substitution': 0.0,
            'tab_error_miss': 0.0,
            'tab_error_false_alarm': e_fa,
            'tab_error_total': e_fa
        }

    miss_error_frames = np.sum((targets_flat != silence_class) & (preds_flat == silence_class))
    false_alarm_frames = np.sum((targets_flat == silence_class) & (preds_flat != silence_class))
    substitution_error_frames = np.sum(
        (targets_flat != silence_class) & 
        (preds_flat != silence_class) & 
        (targets_flat != preds_flat)
    )

    e_sub = substitution_error_frames / n_ref
    e_miss = miss_error_frames / n_ref
    e_fa = false_alarm_frames / n_ref
    e_tot = e_sub + e_miss + e_fa

    results = {
        'tab_error_substitution': e_sub,
        'tab_error_miss': e_miss,
        'tab_error_false_alarm': e_fa,
        'tab_error_total': e_tot
    }
    
    logger.debug(f"[error_scores] Returning results: {results}")
    return results

def compute_per_string_class_weights(
    npz_path_list: list[str], 
    num_strings: int, 
    num_classes: int,
    silence_class: int, 
    silence_factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    string_counters = [Counter() for _ in range(num_strings)]
    logger.info("Calculating class weights from the dataset...")
    for path in tqdm(npz_path_list, desc="Analyzing files for class weights"):
        try:
            with np.load(path, allow_pickle=True) as data:
                if "tablature" not in data:
                    continue
                tab = data["tablature"]
                tab = np.where(tab == -1, silence_class, tab)
                for s in range(num_strings):
                    string_counters[s].update(tab[s].tolist())
        except Exception as e:
            logger.warning(f"Could not process file {path} for class weights. Error: {e}")
            continue
    final_weights, final_counts = [], []
    for s in range(num_strings):
        total_counts_for_string = string_counters[s]
        current_string_weights, current_string_counts = [], []
        for cls in range(num_classes):
            count = total_counts_for_string.get(cls, 0)
            weight = 0.0 if count == 0 else 1.0 / (np.log1p(count) + 1e-9)
            current_string_weights.append(weight)
            current_string_counts.append(count)
        weights_tensor = torch.tensor(current_string_weights, dtype=torch.float32)
        if silence_class < len(weights_tensor):
            weights_tensor[silence_class] *= silence_factor
        if weights_tensor.sum() > 0:
            weights_tensor = (weights_tensor / weights_tensor.sum()) * num_classes
        final_weights.append(weights_tensor)
        final_counts.append(torch.tensor(current_string_counts, dtype=torch.int32))
    final_weights_tensor = torch.stack(final_weights)
    final_counts_tensor = torch.stack(final_counts)
    results_tuple = (final_weights_tensor, final_counts_tensor)
    logger.debug(f"Class weights calculation returning: {describe(results_tuple)}")
    logger.debug(f"  -> Weights: {describe(results_tuple[0])}")
    logger.debug(f"  -> Counts: {describe(results_tuple[1])}")
    logger.info("Class weights calculation finished.")
    return results_tuple

def apply_duration_threshold(preds_tab, targets, min_duration_frames, silence_class):
    logger.debug(f"[duration_filter] Applying filter with min_duration: {min_duration_frames} to preds: {describe(preds_tab)}")
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
    if stats['correctly_discarded_segments'] > 0 or stats['accidentally_discarded_segments'] > 0:
        logger.debug(f"[duration_filter] Stats: {describe(stats)}")
    
    results_tuple = (processed_preds, stats)
    logger.debug(f"[duration_filter] Returning: {describe(results_tuple)}")
    return results_tuple
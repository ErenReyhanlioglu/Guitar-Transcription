import torch
import numpy as np
import scipy.ndimage as ndimage
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from tqdm import tqdm
import logging
from .logger import describe
from .agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch

EPSILON = 1e-9 

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

def compute_tablature_metrics(preds, targets, include_silence, silence_class):
    """
    - overall_precision/recall/f1: LOGISTIC (ikili aktivasyon) MICRO (Evaluator mantığı)
      -> tablature etiketlerini silence hariç one-hot'a çevir, tüm eksenlerde flatten et, TP/pred/gt say.
    - *_weighted ve *_macro: Mevcut MULTICLASS yöntem korunur (trainer uyumu için).
    - per_string: Aynen korunur (weighted + macro).
    """
    logger.debug(f"[tab_metrics] Inputs - preds: {describe(preds)}, targets: {describe(targets)}")

    def _ensure_TS(x: torch.Tensor) -> torch.Tensor:
        # (B,T,S) -> (T,S); (T,S) 
        if x.ndim == 3:
            B, T, S = x.shape
            return x.reshape(-1, S)
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Expected (T,S) or (B,T,S), got {x.shape}")

    def _compact_labels_ex_silence(all_vals: np.ndarray, silence: int):
        uniq = np.unique(all_vals)
        eff = [u for u in uniq if u != silence and u >= 0]
        eff_sorted = sorted(eff)
        return {c:i for i,c in enumerate(eff_sorted)}, len(eff_sorted)

    def _to_logistic_no_silence(tab_np: np.ndarray, silence: int):
        T, S = tab_np.shape
        mapper, C_eff = _compact_labels_ex_silence(tab_np, silence)
        if C_eff == 0:
            return np.zeros((T, S, 0), dtype=np.int8)  
        act = np.zeros((T, S, C_eff), dtype=np.int8)
        valid = (tab_np != silence) & (tab_np >= 0)
        rows, cols = np.where(valid)
        eff_idx = np.vectorize(lambda x: mapper[x])(tab_np[valid])
        act[rows, cols, eff_idx] = 1
        return act

    preds_TS   = _ensure_TS(preds)
    targets_TS = _ensure_TS(targets)

    preds_np   = preds_TS.detach().cpu().numpy()
    targets_np = targets_TS.detach().cpu().numpy()

    pred_act = _to_logistic_no_silence(preds_np,   silence_class)  # (T,S,C_eff_pred)
    ref_act  = _to_logistic_no_silence(targets_np, silence_class)  # (T,S,C_eff_ref)

    C_eff = max(pred_act.shape[2], ref_act.shape[2]) if pred_act.ndim == 3 and ref_act.ndim == 3 else 0
    if C_eff == 0:
        results = {
            'overall_precision': 1.0, 'overall_recall': 1.0, 'overall_f1': 1.0,
            'overall_precision_weighted': 1.0, 'overall_recall_weighted': 1.0, 'overall_f1_weighted': 1.0,
            'overall_precision_macro': 1.0, 'overall_recall_macro': 1.0, 'overall_f1_macro': 1.0,
            'per_string': {}
        }
        logger.debug(f"[tab_metrics] Returning results (empty classes fastpath): {describe(results)}")
        return results

    if pred_act.shape[2] < C_eff:
        pad = ((0,0),(0,0),(0, C_eff - pred_act.shape[2]))
        pred_act = np.pad(pred_act, pad, mode='constant')
    if ref_act.shape[2] < C_eff:
        pad = ((0,0),(0,0),(0, C_eff - ref_act.shape[2]))
        ref_act = np.pad(ref_act, pad, mode='constant')

    est_f = pred_act.reshape(-1)
    ref_f = ref_act.reshape(-1)

    tp   = np.sum(est_f * ref_f)
    pred = np.sum(est_f)
    gt   = np.sum(ref_f)

    eps = 1e-9
    p_micro_log = float(tp / (pred + eps))
    r_micro_log = float(tp / (gt   + eps))
    f1_micro_log= float(0.0 if (p_micro_log + r_micro_log) == 0 else (2*p_micro_log*r_micro_log)/(p_micro_log+r_micro_log+eps))

    results = {
        'overall_precision': p_micro_log,
        'overall_recall':    r_micro_log,
        'overall_f1':        f1_micro_log
    }

    preds_flat   = preds_np.reshape(-1)
    targets_flat = targets_np.reshape(-1)

    if not include_silence:
        mask = (targets_flat != silence_class)
        if mask.sum() == 0:
            results.update({
                'overall_precision_weighted': 1.0, 'overall_recall_weighted': 1.0, 'overall_f1_weighted': 1.0,
                'overall_precision_macro': 1.0,    'overall_recall_macro': 1.0,    'overall_f1_macro': 1.0,
                'per_string': {}
            })
            logger.debug(f"[tab_metrics] Returning results (no-non-silence fastpath): {describe(results)}")
            return results
        preds_flat   = preds_flat[mask]
        targets_flat = targets_flat[mask]

    uniq_lbls = np.unique(np.concatenate([preds_flat, targets_flat]))
    uniq_lbls = uniq_lbls[uniq_lbls >= 0].tolist()
    if len(uniq_lbls) == 0:
        results.update({
            'overall_precision_weighted': 1.0, 'overall_recall_weighted': 1.0, 'overall_f1_weighted': 1.0,
            'overall_precision_macro': 1.0,    'overall_recall_macro': 1.0,    'overall_f1_macro': 1.0,
            'per_string': {}
        })
        logger.debug(f"[tab_metrics] Returning results (no labels fastpath): {describe(results)}")
        return results

    p_w = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)
    r_w = recall_score   (targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)
    f1_w= f1_score      (targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)

    p_m = precision_score(targets_flat, preds_flat, average='macro',    zero_division=0, labels=uniq_lbls)
    r_m = recall_score   (targets_flat, preds_flat, average='macro',    zero_division=0, labels=uniq_lbls)
    f1_m= f1_score      (targets_flat, preds_flat, average='macro',     zero_division=0, labels=uniq_lbls)

    results['overall_precision_weighted'] = float(p_w)
    results['overall_recall_weighted']    = float(r_w)
    results['overall_f1_weighted']        = float(f1_w)

    results['overall_precision_macro'] = float(p_m)
    results['overall_recall_macro']    = float(r_m)
    results['overall_f1_macro']        = float(f1_m)

    per_string_metrics = {}
    if preds_np.ndim == 2:
        num_strings = preds_np.shape[1]
        for s in range(num_strings):
            p_s = preds_np[:, s]
            t_s = targets_np[:, s]

            if not include_silence:
                m_s = (t_s != silence_class)
                if m_s.sum() == 0:
                    continue
                p_s = p_s[m_s]
                t_s = t_s[m_s]

            if len(t_s) > 0:
                uniq_s = np.unique(np.concatenate([p_s, t_s]))
                uniq_s = uniq_s[uniq_s >= 0].tolist()
                if len(uniq_s) == 0:
                    continue
                f1_s_w = f1_score(t_s, p_s, average='weighted', zero_division=0, labels=uniq_s)
                f1_s_m = f1_score(t_s, p_s, average='macro',    zero_division=0, labels=uniq_s)
                per_string_metrics[f'string_{s}_f1'] = float(f1_s_w)
                per_string_metrics[f'string_{s}_f1_macro'] = float(f1_s_m)

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
        return {
            'octave_f1': 1.0, 'octave_precision': 1.0, 'octave_recall': 1.0,
            'octave_f1_weighted': 1.0, 'octave_precision_weighted': 1.0, 'octave_recall_weighted': 1.0,
            'octave_f1_macro': 1.0, 'octave_precision_macro': 1.0, 'octave_recall_macro': 1.0
        }
    
    p_micro = precision_score(targets_flat, preds_flat, average='micro', zero_division=0)
    r_micro = recall_score(targets_flat, preds_flat, average='micro', zero_division=0)
    f1_micro = f1_score(targets_flat, preds_flat, average='micro', zero_division=0)

    p_w = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    r_w = recall_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    f1_w = f1_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    
    p_m = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    r_m = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    f1_m = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
    
    return {
        'octave_precision': p_micro, 
        'octave_recall': r_micro, 
        'octave_f1': f1_micro,

        'octave_precision_weighted': p_w, 
        'octave_recall_weighted': r_w, 
        'octave_f1_weighted': f1_w,

        'octave_precision_macro': p_m, 
        'octave_recall_macro': r_m, 
        'octave_f1_macro': f1_m
    }

def _tab_to_midi_sets(tab_array, tuning, silence_class, ignore_index=-1):
    num_frames, num_strings = tab_array.shape
    midi_sets = []
    tuning = np.array(tuning)
    for i in range(num_frames):
        frame_set = set()
        for s in range(num_strings):
            fret = tab_array[i, s]
            # Sessizlik veya yoksayma indeksi olmayan perdeleri işle
            if fret != silence_class and fret != ignore_index:
                midi_note = tuning[s] + fret
                frame_set.add(midi_note)
        midi_sets.append(frame_set)
    return midi_sets

def compute_tdr_old(preds_tab, targets_tab, tuning, silence_class):
    if isinstance(preds_tab, torch.Tensor):
        preds_tab = preds_tab.cpu().numpy()
    if isinstance(targets_tab, torch.Tensor):
        targets_tab = targets_tab.cpu().numpy()
    
    if preds_tab.shape[1] > preds_tab.shape[0]:
        preds_tab = preds_tab.T
    if targets_tab.shape[1] > targets_tab.shape[0]:
        targets_tab = targets_tab.T

    pred_midi_sets = _tab_to_midi_sets(preds_tab, tuning, silence_class)
    target_midi_sets = _tab_to_midi_sets(targets_tab, tuning, silence_class)

    n_correct_pitch_frames = 0
    n_correct_tablature_frames = 0
    num_frames = preds_tab.shape[0]

    for i in range(num_frames):
        if target_midi_sets[i] and pred_midi_sets[i] == target_midi_sets[i]:
            n_correct_pitch_frames += 1
            
            if np.array_equal(preds_tab[i], targets_tab[i]):
                n_correct_tablature_frames += 1

    if n_correct_pitch_frames == 0:
        return 1.0
    
    tdr = n_correct_tablature_frames / n_correct_pitch_frames
    
    return tdr

def compute_tdr(preds_tab, targets_tab, profile):
    if isinstance(preds_tab, torch.Tensor):
        preds_tab = preds_tab.cpu().numpy()
    if isinstance(targets_tab, torch.Tensor):
        targets_tab = targets_tab.cpu().numpy()

    tablature_est_logistic = tablature_to_stacked_multi_pitch(preds_tab, profile)
    tablature_ref_logistic = tablature_to_stacked_multi_pitch(targets_tab, profile)

    num_correct_tablature = np.sum(tablature_est_logistic * tablature_ref_logistic)

    multi_pitch_est = np.sum(tablature_est_logistic, axis=-3)
    multi_pitch_ref = np.sum(tablature_ref_logistic, axis=-3)

    num_correct_multi_pitch = np.sum(np.minimum(multi_pitch_est, multi_pitch_ref))

    tdr = num_correct_tablature / (num_correct_multi_pitch + EPSILON)

    return float(tdr)

def compute_substitution_error(preds_flat, targets_flat, n_ref, silence_class):
    """Substitution (E_sub) hatalarının sayısını ve oranını hesaplar."""
    count = np.sum(
        (targets_flat != silence_class) & 
        (preds_flat != silence_class) & 
        (targets_flat != preds_flat)
    )
    rate = count / (n_ref + 1e-9) if n_ref > 0 else 0.0
    return {'rate': rate, 'count': count}

def compute_miss_error(preds_flat, targets_flat, n_ref, silence_class):
    """Miss (E_miss) hatalarının sayısını ve oranını hesaplar."""
    count = np.sum((targets_flat != silence_class) & (preds_flat == silence_class))
    rate = count / (n_ref + 1e-9) if n_ref > 0 else 0.0
    return {'rate': rate, 'count': count}

def compute_false_alarm_error(preds_flat, targets_flat, n_ref, silence_class):
    """False Alarm (E_fa) hatalarının sayısını ve oranını hesaplar."""
    count = np.sum((targets_flat == silence_class) & (preds_flat != silence_class))
    if n_ref > 0:
        rate = count / (n_ref + 1e-9)
    else:
        n_est = np.sum(preds_flat != silence_class)
        rate = n_est / (n_est + 1e-9)
    return {'rate': rate, 'count': count}

def compute_duplicate_pitch_error(preds_np, targets_np, n_ref, tuning, silence_class):
    """Duplicate Pitch (E_d.p.) hatalarının sayısını ve oranını hesaplar."""
    tuning = np.array(tuning)
    num_frames, num_strings = preds_np.shape
    count = 0

    for n in range(num_frames):
        frame_preds = preds_np[n, :]
        frame_targets = targets_np[n, :]

        pred_pitches = [tuning[s] + frame_preds[s] for s in range(num_strings) if frame_preds[s] != silence_class]
        target_pitches = [tuning[s] + frame_targets[s] for s in range(num_strings) if frame_targets[s] != silence_class]

        pred_counts = Counter(pred_pitches)
        target_counts = Counter(target_pitches)

        frame_error = 0
        for pitch, pred_count in pred_counts.items():
            target_count = target_counts.get(pitch, 0)
            error = max(0, pred_count - target_count)
            frame_error += error
        
        count += frame_error
    
    rate = count / (n_ref + 1e-9) if n_ref > 0 else 0.0
    return {'rate': rate, 'count': count}

def compute_tablature_error_scores(preds, targets, silence_class, tuning):
    if isinstance(preds, torch.Tensor):
        preds_np = preds.cpu().numpy()
    else:
        preds_np = preds
        
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    else:
        targets_np = targets

    preds_flat = preds_np.flatten()
    targets_flat = targets_np.flatten()
    n_ref = np.sum(targets_flat != silence_class)

    sub_results = compute_substitution_error(preds_flat, targets_flat, n_ref, silence_class)
    miss_results = compute_miss_error(preds_flat, targets_flat, n_ref, silence_class)
    fa_results = compute_false_alarm_error(preds_flat, targets_flat, n_ref, silence_class)
    dup_results = compute_duplicate_pitch_error(preds_np, targets_np, n_ref, tuning, silence_class)

    final_results = {
        'tab_error_substitution_rate': sub_results['rate'],
        'tab_error_substitution_count': sub_results['count'],
        'tab_error_miss_rate': miss_results['rate'],
        'tab_error_miss_count': miss_results['count'],
        'tab_error_false_alarm_rate': fa_results['rate'],
        'tab_error_false_alarm_count': fa_results['count'],
        'tab_error_duplicate_pitch_rate': dup_results['rate'],
        'tab_error_duplicate_pitch_count': dup_results['count']
    }

    total_error_count = (sub_results['count'] + 
                         miss_results['count'] + 
                         fa_results['count'])
                         
    total_error_rate = (sub_results['rate'] + 
                        miss_results['rate'] + 
                        fa_results['rate'])

    final_results['tab_error_total_rate'] = total_error_rate
    final_results['tab_error_total_count'] = total_error_count

    return final_results

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
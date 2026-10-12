import torch
import numpy as np
import scipy.ndimage as ndimage
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
from tqdm import tqdm
from typing import Any, cast
import logging
from .logger import describe
from .agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch

EPSILON = 1e-9 

logger = logging.getLogger(__name__)

def finalize_output(logits, silence_class, return_shape="targets", mask_silence=False):
    """Converts logits to class predictions via argmax over the last dimension.

    If `mask_silence`, predicted silence-class entries are relabeled to -1.
    If `return_shape == "targets"`, permutes axes 1 and 2 of the resulting
    predictions (time-major to string-major layout) to match the target
    tensor's axis order.
    """
    logger.debug(f"[finalize_output] Input logits: {describe(logits)}")
    preds = logits.argmax(dim=-1)
    if mask_silence:
        preds[preds == silence_class] = -1
    if return_shape == "targets":
        preds = preds.permute(0, 2, 1)
    logger.debug(f"[finalize_output] Returning preds: {describe(preds)}")
    return preds

def accuracy(logits, targets, include_silence, silence_class):
    """Computes elementwise accuracy between predicted and target tablature classes.

    Predictions are obtained via argmax over `logits`, masking predicted
    silence-class frames to -1 when `include_silence` is False. If
    `include_silence` is False, only frames where the target is not -1 are
    counted toward accuracy (returns 1.0 if there are none); otherwise every
    frame is counted.
    """
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
    - overall_precision/recall/f1: LOGISTIC (binary activation) MICRO (evaluator logic)
      -> converts tablature labels to one-hot excluding silence, flattens across all axes, counts TP/pred/gt.
    - *_weighted and *_macro: the existing MULTICLASS method is preserved (for trainer compatibility).
    - per_string: preserved as-is (weighted + macro).
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

    results: dict[str, Any] = {
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

    p_w = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r_w = recall_score   (targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1_w= f1_score      (targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str

    p_m = precision_score(targets_flat, preds_flat, average='macro',    zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r_m = recall_score   (targets_flat, preds_flat, average='macro',    zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1_m= f1_score      (targets_flat, preds_flat, average='macro',     zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str

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
                f1_s_w = f1_score(t_s, p_s, average='weighted', zero_division=0, labels=uniq_s)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
                f1_s_m = f1_score(t_s, p_s, average='macro',    zero_division=0, labels=uniq_s)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
                per_string_metrics[f'string_{s}_f1'] = float(f1_s_w)
                per_string_metrics[f'string_{s}_f1_macro'] = float(f1_s_m)

    results['per_string'] = per_string_metrics

    logger.debug(f"[tab_metrics] Returning results: {describe(results)}")
    return results

def compute_multipitch_metrics(preds_tab, targets, profile, include_silence, silence_class):
    """Computes binary precision/recall/F1 for multi-pitch detection derived from tablature.

    Converts predicted and target tablature to stacked multi-pitch representations
    via `agt_tools`, collapses them to a single multi-pitch roll, and scores
    note-level presence/absence as a binary classification problem.
    """
    logger.debug(f"[mp_metrics] Inputs - preds_tab: {describe(preds_tab)}, targets: {describe(targets)}")
    
    preds_tab_transposed = torch.from_numpy(preds_tab.cpu().numpy().T)
    targets_transposed = torch.from_numpy(targets.cpu().numpy().T)
    
    preds_smp = tablature_to_stacked_multi_pitch(preds_tab_transposed, profile, include_silence, silence_class)
    targets_smp = tablature_to_stacked_multi_pitch(targets_transposed, profile, include_silence, silence_class)
    preds_mp = stacked_multi_pitch_to_multi_pitch(preds_smp)
    targets_mp = stacked_multi_pitch_to_multi_pitch(targets_smp)
    
    preds_flat = preds_mp.flatten()
    targets_flat = targets_mp.flatten()
    
    p = precision_score(targets_flat, preds_flat, average='binary', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r = recall_score(targets_flat, preds_flat, average='binary', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1 = f1_score(targets_flat, preds_flat, average='binary', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    
    results = {'multipitch_f1': f1, 'multipitch_precision': p, 'multipitch_recall': r}
    logger.debug(f"[mp_metrics] Returning results: {describe(results)}")
    return results

def compute_octave_tolerant_metrics(preds_tab, targets_tab, tuning, silence_class, num_pitch_classes=12):
    """Computes octave-invariant tablature metrics by reducing MIDI pitch to pitch class (mod 12).

    Converts each string/fret prediction and target to its MIDI note, then to a
    pitch class, and computes micro (logistic-style), weighted, and macro
    precision/recall/F1 the same way `compute_tablature_metrics` does —
    predictions differing only by octave from the reference now count as correct.
    """
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

    preds_np = octave_preds.cpu().numpy()
    targets_np = octave_targets.cpu().numpy()

    # --- logistic style micro ---
    def _to_logistic(tab_np: np.ndarray, silence: int, n_classes: int):
        T, S = tab_np.shape
        uniq = [u for u in np.unique(tab_np) if u not in (-1, silence)]
        if len(uniq) == 0:
            return np.zeros((T, S, 0), dtype=np.int8)
        mapper = {c:i for i,c in enumerate(sorted(uniq))}
        act = np.zeros((T, S, len(mapper)), dtype=np.int8)
        valid = (tab_np != silence) & (tab_np >= 0)
        rows, cols = np.where(valid)
        eff_idx = np.vectorize(lambda x: mapper[x])(tab_np[valid])
        act[rows, cols, eff_idx] = 1
        return act

    pred_act = _to_logistic(preds_np, num_pitch_classes, num_pitch_classes)
    ref_act  = _to_logistic(targets_np, num_pitch_classes, num_pitch_classes)

    C_eff = max(pred_act.shape[2], ref_act.shape[2]) if pred_act.ndim == 3 and ref_act.ndim == 3 else 0
    if C_eff == 0:
        return {
            'octave_precision': 1.0, 'octave_recall': 1.0, 'octave_f1': 1.0,
            'octave_precision_weighted': 1.0, 'octave_recall_weighted': 1.0, 'octave_f1_weighted': 1.0,
            'octave_precision_macro': 1.0, 'octave_recall_macro': 1.0, 'octave_f1_macro': 1.0
        }

    # padding
    if pred_act.shape[2] < C_eff:
        pred_act = np.pad(pred_act, ((0,0),(0,0),(0, C_eff - pred_act.shape[2])), mode='constant')
    if ref_act.shape[2] < C_eff:
        ref_act = np.pad(ref_act, ((0,0),(0,0),(0, C_eff - ref_act.shape[2])), mode='constant')

    est_f = pred_act.reshape(-1)
    ref_f = ref_act.reshape(-1)

    tp   = np.sum(est_f * ref_f)
    pred = np.sum(est_f)
    gt   = np.sum(ref_f)

    eps = 1e-9
    p_micro_log = float(tp / (pred + eps))
    r_micro_log = float(tp / (gt   + eps))
    f1_micro_log= float(0.0 if (p_micro_log + r_micro_log) == 0 else (2*p_micro_log*r_micro_log)/(p_micro_log+r_micro_log+eps))

    # --- sklearn weighted & macro ---
    preds_flat   = preds_np.flatten()
    targets_flat = targets_np.flatten()

    mask = (targets_flat != -1) & (targets_flat != num_pitch_classes)
    preds_flat   = preds_flat[mask]
    targets_flat = targets_flat[mask]

    if len(targets_flat) == 0:
        return {
            'octave_precision': 1.0, 'octave_recall': 1.0, 'octave_f1': 1.0,
            'octave_precision_weighted': 1.0, 'octave_recall_weighted': 1.0, 'octave_f1_weighted': 1.0,
            'octave_precision_macro': 1.0, 'octave_recall_macro': 1.0, 'octave_f1_macro': 1.0
        }

    uniq_lbls = np.unique(np.concatenate([preds_flat, targets_flat]))
    uniq_lbls = uniq_lbls[uniq_lbls >= 0].tolist()

    p_w = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r_w = recall_score   (targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1_w= f1_score      (targets_flat, preds_flat, average='weighted', zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str

    p_m = precision_score(targets_flat, preds_flat, average='macro',    zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r_m = recall_score   (targets_flat, preds_flat, average='macro',    zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1_m= f1_score      (targets_flat, preds_flat, average='macro',     zero_division=0, labels=uniq_lbls)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str

    return {
        'octave_precision': p_micro_log,
        'octave_recall': r_micro_log,
        'octave_f1': f1_micro_log,
        'octave_precision_weighted': float(p_w),
        'octave_recall_weighted': float(r_w),
        'octave_f1_weighted': float(f1_w),
        'octave_precision_macro': float(p_m),
        'octave_recall_macro': float(r_m),
        'octave_f1_macro': float(f1_m)
    }

def _tab_to_midi_sets(tab_array, tuning, silence_class, ignore_index=-1):
    """Converts a (frames, strings) tablature array into a list of per-frame sets of active MIDI notes."""
    num_frames, num_strings = tab_array.shape
    midi_sets = []
    tuning = np.array(tuning)
    for i in range(num_frames):
        frame_set = set()
        for s in range(num_strings):
            fret = tab_array[i, s]
            if fret != silence_class and fret != ignore_index:
                midi_note = tuning[s] + fret
                frame_set.add(midi_note)
        midi_sets.append(frame_set)
    return midi_sets

def compute_tdr(preds_tab, targets_tab, profile, include_silence, silence_class):
    """Computes the Tablature Disambiguation Rate (TDR).

    TDR is the fraction of correctly-detected multi-pitch notes (the minimum
    of predicted and reference activations per pitch) that are also assigned
    to the correct string/fret in the stacked (per-string) representation.
    """
    if isinstance(preds_tab, torch.Tensor):
        preds_tab = preds_tab.cpu().numpy()
    if isinstance(targets_tab, torch.Tensor):
        targets_tab = targets_tab.cpu().numpy()

    tablature_est_logistic = tablature_to_stacked_multi_pitch(preds_tab, profile, include_silence, silence_class)
    tablature_ref_logistic = tablature_to_stacked_multi_pitch(targets_tab, profile, include_silence, silence_class)

    num_correct_tablature = np.sum(tablature_est_logistic * tablature_ref_logistic)

    multi_pitch_est = np.sum(tablature_est_logistic, axis=-3)
    multi_pitch_ref = np.sum(tablature_ref_logistic, axis=-3)

    num_correct_multi_pitch = np.sum(np.minimum(multi_pitch_est, multi_pitch_ref))

    tdr = num_correct_tablature / (num_correct_multi_pitch + EPSILON)

    return float(tdr)

def compute_substitution_error(preds_flat, targets_flat, n_ref, silence_class):
    """Computes the count and rate of substitution (E_sub) errors."""
    count = np.sum(
        (targets_flat != silence_class) & 
        (preds_flat != silence_class) & 
        (targets_flat != preds_flat)
    )
    rate = count / (n_ref + 1e-9) if n_ref > 0 else 0.0
    return {'rate': rate, 'count': count}

def compute_miss_error(preds_flat, targets_flat, n_ref, silence_class):
    """Computes the count and rate of miss (E_miss) errors."""
    count = np.sum((targets_flat != silence_class) & (preds_flat == silence_class))
    rate = count / (n_ref + 1e-9) if n_ref > 0 else 0.0
    return {'rate': rate, 'count': count}

def compute_false_alarm_error(preds_flat, targets_flat, n_ref, silence_class):
    """Computes the count and rate of false alarm (E_fa) errors."""
    count = np.sum((targets_flat == silence_class) & (preds_flat != silence_class))
    if n_ref > 0:
        rate = count / (n_ref + 1e-9)
    else:
        n_est = np.sum(preds_flat != silence_class)
        rate = n_est / (n_est + 1e-9)
    return {'rate': rate, 'count': count}

def compute_duplicate_pitch_error(preds_np, targets_np, n_ref, tuning, silence_class):
    """Computes the count and rate of duplicate pitch (E_d.p.) errors."""
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
    """Computes the paper's error-rate breakdown: substitution, miss, false alarm,
    duplicate pitch, and total error rates/counts, normalized by the reference note count."""
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
    """Computes per-string, per-class loss weights from the tablature label distribution across a dataset.

    Scans every file's `tablature` array, counts occurrences of each fret class
    per string, and weights classes inversely by log-frequency (rarer classes
    get higher weight), scaling the silence class by `silence_factor` and
    normalizing each string's weights to sum to `num_classes`.

    Returns:
        A (num_strings, num_classes) weight tensor and a matching per-class count tensor.
    """
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
    """Suppresses predicted note segments shorter than `min_duration_frames` back to silence.

    For each string, finds contiguous non-silence prediction segments and
    discards any shorter than the threshold, tracking whether each discarded
    segment was mostly correct (`accidentally_discarded_segments`) or mostly
    wrong (`correctly_discarded_segments`) relative to the target.

    Returns:
        The filtered predictions and a dict of discard statistics.
    """
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
        labels, num_features = cast(tuple[np.ndarray, int], ndimage.label(string_preds != silence_class))
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

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_hand_position_metrics(preds, targets, include_silence=True, silence_class=0):
    """
    Hand Position Metrics.
    - silence_class: the code for silence/open string in the dataset (usually 0).
    - include_silence=False: only evaluates moments where the hand has taken a position (target != 0).
    """
    if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
    
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    if not include_silence:
        # Select non-silence frames (e.g., anything other than class 0)
        mask = (targets_flat != silence_class)
        if mask.sum() == 0:
            return {
                'hand_pos_acc': 1.0, 'hand_pos_precision': 1.0, 
                'hand_pos_recall': 1.0, 'hand_pos_f1': 1.0
            }
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]
    
    # 'macro': computes each class's score separately and averages them (less affected by class imbalance).
    p = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    acc = accuracy_score(targets_flat, preds_flat)
    
    return {
        'hand_pos_acc': float(acc),
        'hand_pos_precision': float(p),
        'hand_pos_recall': float(r),
        'hand_pos_f1': float(f1)
    }

def compute_string_activity_metrics(preds_logits, targets, include_silence=True, silence_class=None):
    """
    String Activity (Binary).
    Uses average='binary' to measure the performance of only the ACTIVE (1) strings.
    Knowing the silent (0) class does not affect the score.
    """
    if isinstance(preds_logits, torch.Tensor):
        preds_bin = (torch.sigmoid(preds_logits) > 0.5).int().cpu().numpy()
    else:
        preds_bin = (1 / (1 + np.exp(-preds_logits)) > 0.5).astype(int)
        
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
    
    preds_flat = preds_bin.flatten()
    targets_flat = targets.flatten()
    
    # average='binary': computes the metric only for the positive class (1).
    p = precision_score(targets_flat, preds_flat, average='binary', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r = recall_score(targets_flat, preds_flat, average='binary', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1 = f1_score(targets_flat, preds_flat, average='binary', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    
    return {
        'string_act_precision': float(p),
        'string_act_recall': float(r),
        'string_act_f1': float(f1)
    }

def compute_pitch_class_metrics(preds_logits, targets, include_silence=True, silence_class=None):
    """
    Pitch Class (Multi-Label Binary).
    Uses average='samples' to compute performance separately for each time frame.
    """
    if isinstance(preds_logits, torch.Tensor):
        preds_bin = (torch.sigmoid(preds_logits) > 0.5).int().cpu().numpy()
    else:
        preds_bin = (1 / (1 + np.exp(-preds_logits)) > 0.5).astype(int)
        
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
    
    # average='samples': computes F1 for each sample (frame) and averages them.
    # This is the most correct choice for musical analysis.
    p = precision_score(targets, preds_bin, average='samples', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    r = recall_score(targets, preds_bin, average='samples', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    f1 = f1_score(targets, preds_bin, average='samples', zero_division=0)  # type: ignore[reportArgumentType]  # sklearn accepts int zero_division at runtime; stub only types it as str
    
    return {
        'pitch_class_precision': float(p),
        'pitch_class_recall': float(r),
        'pitch_class_f1': float(f1)
    }

def compute_aux_multipitch_metrics(preds_logits, targets, include_silence=True, silence_class=None):
    """
    Multipitch Head (Binary).
    Uses the same 'binary' logic as String Activity.
    """
    res = compute_string_activity_metrics(preds_logits, targets)
    return {
        'mp_head_precision': res['string_act_precision'],
        'mp_head_recall': res['string_act_recall'],
        'mp_head_f1': res['string_act_f1']
    }

def string_assignment_breakdown(preds, targets, tuning, num_frets, silence_class):
    """Splits the pitch-correct note events behind TDR into correct and wrong-string assignments.

    Groups them by open vs. fretted string, fret region (1-5 vs. 6 and up), and
    whether the pitch has one or several playable positions; also counts how many
    wrong-string errors land on an adjacent string and which string pairs get
    confused. The overall error rate equals 1 - TDR.

    When a pitch sounds on several strings in the same frame, unmatched reference
    and estimated strings are paired in ascending string order.

    Args:
        preds, targets: (frames, strings) fret labels, silence as `silence_class`.
        tuning: Open-string MIDI pitches, low string first.
        num_frets: Highest fret label.
        silence_class: Label used for silence.

    Returns:
        A dict with {'events', 'errors', 'error_rate'} per category, the adjacent /
        non-adjacent error counts and share, and 'string_pairs' mapping
        "reference->estimated" (0-indexed from the low E) to an error count.
    """
    tuning = list(tuning)
    P = np.asarray(preds.cpu().numpy() if hasattr(preds, "cpu") else preds)
    R = np.asarray(targets.cpu().numpy() if hasattr(targets, "cpu") else targets)

    positions = {}

    def n_positions(pitch):
        if pitch not in positions:
            positions[pitch] = sum(1 for t in tuning if 0 <= pitch - t <= num_frets)
        return positions[pitch]

    categories = ["overall", "open_string", "fretted", "fret_1_5", "fret_6_19",
                  "single_position_pitch", "multi_position_pitch"]
    events = dict.fromkeys(categories, 0)
    errors = dict.fromkeys(categories, 0)
    adjacent = non_adjacent = 0
    pairs = {}

    def record(ref_string, pitch, is_error):
        fret = pitch - tuning[ref_string]
        keys = ["overall", "open_string" if fret == 0 else "fretted"]
        if 1 <= fret <= 5:
            keys.append("fret_1_5")
        elif fret >= 6:
            keys.append("fret_6_19")
        keys.append("single_position_pitch" if n_positions(pitch) == 1 else "multi_position_pitch")
        for k in keys:
            events[k] += 1
            errors[k] += int(is_error)

    sounding_ref = (R != silence_class) & (R >= 0)
    sounding_est = (P != silence_class) & (P >= 0)
    for i in np.where(sounding_ref.any(1) & sounding_est.any(1))[0]:
        ref, est = {}, {}
        for s in range(R.shape[1]):
            if sounding_ref[i, s]:
                ref.setdefault(tuning[s] + int(R[i, s]), []).append(s)
            if sounding_est[i, s]:
                est.setdefault(tuning[s] + int(P[i, s]), []).append(s)
        for pitch in ref.keys() & est.keys():
            ref_strings, est_strings = set(ref[pitch]), set(est[pitch])
            correct = ref_strings & est_strings
            n_wrong = min(len(ref_strings), len(est_strings)) - len(correct)
            for s in correct:
                record(s, pitch, False)
            ref_left, est_left = sorted(ref_strings - correct), sorted(est_strings - correct)
            for j in range(n_wrong):
                record(ref_left[j], pitch, True)
                key = f"{ref_left[j]}->{est_left[j]}"
                pairs[key] = pairs.get(key, 0) + 1
                if abs(ref_left[j] - est_left[j]) == 1:
                    adjacent += 1
                else:
                    non_adjacent += 1

    result: dict[str, Any] = {c: {"events": events[c], "errors": errors[c],
                                  "error_rate": errors[c] / events[c] if events[c] else float("nan")}
                              for c in categories}
    total = adjacent + non_adjacent
    result["adjacent_errors"] = adjacent
    result["nonadjacent_errors"] = non_adjacent
    result["adjacent_string_share_of_errors"] = adjacent / total if total else float("nan")
    result["string_pairs"] = pairs
    return result

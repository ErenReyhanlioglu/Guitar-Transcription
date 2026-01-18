import numpy as np
import torch
import librosa
from copy import deepcopy
import logging
from src.utils.logger import describe 

logger = logging.getLogger(__name__)

def framify_activations(activations, win_length, hop_length=1, pad=True):
    # This is a low-level helper, extensive logging might create too much noise.
    # We can add it later if needed.
    num_frames = activations.shape[-1]
    pad_length = (win_length // 2)
    if pad:
        num_frames_ = num_frames + 2 * pad_length
    else:
        num_frames_ = max(win_length, num_frames)
    activations = librosa.util.pad_center(activations, size=num_frames_)
    num_hops = (num_frames_ - 2 * pad_length) // hop_length
    chunk_idcs = np.arange(0, num_hops) * hop_length
    activations = [np.expand_dims(activations[..., i : i + win_length], axis=-2) for i in chunk_idcs]
    activations = np.concatenate(activations, axis=-2)
    return activations

def estimate_hop_length(times):
    if not len(times):
        raise ValueError('Cannot estimate hop length from an empty time array.')
    times = np.sort(times)
    non_gaps = np.append([False], np.isclose(np.diff(times, n=2), 0))
    if not np.sum(non_gaps):
        raise ValueError('Time observations are too irregular.')
    hop_length = np.median(np.diff(times)[non_gaps])
    return hop_length

def tablature_to_logistic(targets_tab, num_strings, num_classes):
    """
    Converts a single-note-per-string tablature representation to a multi-label logistic format
    with enhanced logging.
    """
    logger.debug(f"[tab_to_logistic] Inputs - targets_tab: {describe(targets_tab)}")
    
    B, T, S = targets_tab.shape
    num_frets = num_classes - 1
    device = targets_tab.device
    
    targets_logistic = torch.zeros(B * T, S * num_frets, device=device, dtype=torch.float32)
    
    targets_flat = targets_tab.reshape(B * T, S)
    logger.debug(f"[tab_to_logistic] Targets reshaped to flat format: {describe(targets_flat)}")

    total_targets = targets_flat.numel()
    silent_targets = torch.sum(targets_flat == -1).item()
    active_targets = total_targets - silent_targets
    logger.debug(
        f"[tab_to_logistic] Targets statistics: "
        f"Total frames: {targets_flat.shape[0]}, Total strings: {targets_flat.shape[1]}, "
        f"Silent predictions: {silent_targets}, Active predictions: {active_targets}"
    )
    
    for s in range(S):
        fret_values = targets_flat[:, s]
        
        active_mask = (fret_values >= 0) & (fret_values < num_frets)
        
        if torch.any(active_mask):
            active_frame_indices = torch.where(active_mask)[0]
            active_frets = fret_values[active_mask].long()
            
            logistic_indices = s * num_frets + active_frets
            
            logger.debug(
                f"[tab_to_logistic] String {s} conversion: "
                f"Found {len(active_frame_indices)} active frames. "
                f"Active frets: {active_frets.tolist()[:5]}..., " 
                f"Mapped logistic indices: {logistic_indices.tolist()[:5]}..." 
            )
            
            targets_logistic[active_frame_indices, logistic_indices] = 1.0
        else:
            logger.debug(f"[tab_to_logistic] String {s} conversion: No active frets found.")

    logger.debug(f"[tab_to_logistic] Returning targets_logistic: {describe(targets_logistic)}")
    
    total_positives = torch.sum(targets_logistic == 1.0).item()
    logger.debug(f"[tab_to_logistic] Final logistic tensor statistics: Total positive labels set: {total_positives}")
    
    return targets_logistic

def logistic_to_tablature(preds_logistic, num_strings, num_classes, threshold=0.1):
    """
    Converts logistic bank predictions to a tablature matrix, correctly handling multi-label output.
    
    This revised version avoids the torch.max operation to fully embrace the multi-label nature of
    the logistic bank approach, aligning with standard practices in literature.

    Parameters
    ----------
    preds_logistic : tensor (N x (S * F))
        Array of flattened predictions from a logistic model.
    num_strings : int
        Number of strings/degrees of freedom.
    num_classes : int
        Number of classes including silence.
    threshold : float
        Threshold for probability above which a prediction is considered active.

    Returns
    ----------
    fret_indices : tensor (N x S)
        Tablature matrix where each entry is a fret number or silence class.
    """
    logger.debug(f"[logistic_to_tab] Inputs - preds_logistic: {describe(preds_logistic)}, threshold: {threshold}")
    
    N, _ = preds_logistic.shape
    num_frets = num_classes - 1
    silence_class = num_frets
    device = preds_logistic.device
    
    probs = torch.sigmoid(preds_logistic)
    
    preds_reshaped = probs.view(N, num_strings, num_frets)
    
    active_mask = preds_reshaped >= threshold
    
    fret_indices = torch.full((N, num_strings), silence_class, dtype=torch.long, device=device)
    
    fret_probs, max_fret_indices = torch.max(preds_reshaped, dim=2)
    
    above_threshold_count = torch.sum(active_mask)
    total_preds = preds_reshaped.numel()
    logger.debug(
        f"[logistic_to_tab] Prediction statistics: "
        f"Total predictions: {total_preds}, "
        f"Above threshold ({threshold}): {above_threshold_count}, "
        f"Percentage: {100.0 * above_threshold_count / total_preds:.2f}%"
    )

    for s in range(num_strings):
        string_active_mask = active_mask[:, s, :]
        
        frames_with_active_preds = torch.any(string_active_mask, dim=1)
        
        if torch.any(frames_with_active_preds):
            fret_indices[frames_with_active_preds, s] = max_fret_indices[frames_with_active_preds, s]
            
            logger.debug(
                f"[logistic_to_tab] String {s} analysis: "
                f"Active frames: {torch.sum(frames_with_active_preds)}, "
                f"Max prob: {torch.max(fret_probs[frames_with_active_preds, s]):.4f}, "
                f"Min prob: {torch.min(fret_probs[frames_with_active_preds, s]):.4f}"
            )

    logger.debug(f"[logistic_to_tab] Returning fret_indices: {describe(fret_indices)}")
    return fret_indices

def tablature_to_stacked_multi_pitch(tablature, profile, include_silence, silence_class):
    if isinstance(tablature, torch.Tensor):
        tablature = tablature.long()
    else:
        tablature = tablature.astype(np.int64)
    
    logger.debug(f"[tab_to_smp] Tabşature: {describe(tablature)}")
    is_tensor = isinstance(tablature, torch.Tensor)

    if tablature.shape[-1] < tablature.shape[-2]:
        logger.debug(f"[tab_to_smp] Dims: (..., T, S) . Comverting: (..., S, T).")
        tablature = tablature.transpose(-1, -2) if is_tensor else np.swapaxes(tablature, -1, -2)

    original_dims = tablature.dim() if is_tensor else tablature.ndim
    if original_dims == 2:
        tablature = tablature.unsqueeze(0) if is_tensor else np.expand_dims(tablature, axis=0)
        logger.debug(f"[tab_to_smp] Batch dim implemented. New dim: {describe(tablature)}")

    num_dofs, num_frames = tablature.shape[-2:]
    num_pitches = profile.get_range_len()
    
    logger.debug(f"[tab_to_smp] Parameters: String={num_dofs}, Time={num_frames}, Pitch Count={num_pitches}")

    stacked_multi_pitch_shape = tablature.shape[:-2] + (num_dofs, num_pitches, num_frames)
    
    if is_tensor:
        stacked_multi_pitch = torch.zeros(stacked_multi_pitch_shape, device=tablature.device)
        dof_start = torch.tensor(profile.get_midi_tuning(), device=tablature.device) - profile.low
        dof_start = dof_start[None, :, None] 
    else:
        stacked_multi_pitch = np.zeros(stacked_multi_pitch_shape)
        tuning = np.array(profile.get_midi_tuning())
        dof_start = (tuning - profile.low)[np.newaxis, :, np.newaxis]

    logger.debug(f"[tab_to_smp] 'dof_start' (akort offset) tensor created: {describe(dof_start)}")

    if not include_silence:
      non_silent_frames = (tablature >= 0) & (tablature != silence_class)
    pitch_idcs = (tablature + dof_start)[non_silent_frames]
    
    non_silent_count = torch.sum(non_silent_frames).item() if is_tensor else np.sum(non_silent_frames)
    logger.debug(f"[tab_to_smp] Total active non-silent note count: {non_silent_count}")

    if is_tensor:
        pitch_idcs = pitch_idcs.long()
        batch_idcs, string_idcs, frame_idcs = non_silent_frames.nonzero(as_tuple=True)
        stacked_multi_pitch[batch_idcs, string_idcs, pitch_idcs, frame_idcs] = 1
    else:
        # for numpy
        non_silent_idcs = non_silent_frames.nonzero()
        pitch_idcs = pitch_idcs.astype(np.int64)
        # batch, string, frame index
        other_idcs = non_silent_idcs[:-1]
        frame_idcs = non_silent_idcs[-1]
        stacked_multi_pitch[other_idcs + (pitch_idcs, frame_idcs)] = 1
    
    if original_dims == 2:
        stacked_multi_pitch = stacked_multi_pitch.squeeze(0)

    logger.debug(f"[tab_to_smp] Fonksiyon tamamlandı. Dönen stacked_multi_pitch: {describe(stacked_multi_pitch)}")
    return stacked_multi_pitch

def stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch):
    logger.debug(f"[smp_to_mp] Input stacked_multi_pitch: {describe(stacked_multi_pitch)}")
    if isinstance(stacked_multi_pitch, torch.Tensor):
        multi_pitch = torch.max(stacked_multi_pitch, dim=-3)[0]
    else:
        multi_pitch = np.max(stacked_multi_pitch, axis=-3)
    logger.debug(f"[smp_to_mp] Returning multi_pitch: {describe(multi_pitch)}")
    return multi_pitch

def stacked_multi_pitch_to_logistic(stacked_multi_pitch, profile, silence=False):
    tuning = profile.get_midi_tuning()
    logistic = list()
    for dof in range(stacked_multi_pitch.shape[-3]):
        multi_pitch = stacked_multi_pitch[..., dof, :, :]
        lower_bound = tuning[dof] - profile.low
        upper_bound = lower_bound + profile.num_pitches
        multi_pitch = multi_pitch[..., lower_bound : upper_bound, :]
        if silence:
            if isinstance(multi_pitch, np.ndarray):
                silence_activations = np.sum(multi_pitch, axis=-2, keepdims=True) == 0
                multi_pitch = np.append(silence_activations, multi_pitch, axis=-2)
            else:
                silence_activations = torch.sum(multi_pitch, dim=-2, keepdims=True) == 0
                multi_pitch = torch.cat((silence_activations.to(multi_pitch.device), multi_pitch), dim=-2)
        logistic += [multi_pitch]
    if isinstance(stacked_multi_pitch, np.ndarray):
        logistic = np.concatenate(logistic, axis=-2)
    else:
        logistic = torch.cat(logistic, dim=-2)
    return logistic

def notes_to_multi_pitch(pitches, intervals, times, profile, include_offsets=True):
    num_pitches = profile.get_range_len()
    num_frames = len(times)
    multi_pitch = np.zeros((num_pitches, num_frames))
    _times = np.append(times, times[-1] + estimate_hop_length(times))
    pitches, intervals = filter_notes(pitches, intervals, profile, min_time=np.min(_times), max_time=np.max(_times))
    num_notes = len(pitches)
    if num_notes == 0:
        return multi_pitch
    pitches = np.round(pitches - profile.low).astype(int)
    times_broadcast = np.tile(_times, (num_notes, 1))
    onsets = np.argmin((times_broadcast <= intervals[..., :1]), axis=1) - 1
    offsets = np.argmin((times_broadcast <= intervals[..., 1:]), axis=1) - 1
    onsets[onsets == -1], offsets[offsets == -1] = 0, num_frames - 1
    for i in range(num_notes):
        multi_pitch[pitches[i], onsets[i] : offsets[i] + int(include_offsets)] = 1
    return multi_pitch

def multi_pitch_to_onsets(multi_pitch):
    first_frame = multi_pitch[..., :1]
    adjacent_diff = multi_pitch[..., 1:] - multi_pitch[..., :-1]
    onsets = np.concatenate([first_frame, adjacent_diff], axis=-1)
    onsets[onsets <= 0] = 0
    return onsets

def notes_to_onsets(pitches, intervals, times, profile, ambiguity=None):
    onset_times = np.copy(intervals[..., :1])
    offset_times = np.copy(intervals[..., 1:])
    if ambiguity is not None:
        durations = offset_times - onset_times
        durations = np.minimum(durations, ambiguity)
        offset_times = onset_times + durations
    else:
        offset_times = np.copy(onset_times)
    truncated_note_intervals = np.concatenate((onset_times, offset_times), axis=-1)
    onsets = notes_to_multi_pitch(pitches, truncated_note_intervals, times, profile)
    return onsets

def notes_to_batched_notes(pitches, intervals):
    batched_notes = np.empty([0, 3])
    if len(pitches) > 0:
        pitches = np.expand_dims(pitches, axis=-1)
        batched_notes = np.concatenate((intervals, pitches), axis=-1)
    return batched_notes

def batched_notes_to_notes(batched_notes):
    pitches, intervals = batched_notes[..., 2], batched_notes[:, :2]
    return pitches, intervals

def sort_batched_notes(batched_notes, by=0):
    sorted_idcs = np.argsort(batched_notes[..., by], kind='mergesort')
    batched_notes = batched_notes[sorted_idcs]
    return batched_notes

def sort_notes(pitches, intervals, by=0):
    batched_notes = notes_to_batched_notes(pitches, intervals)
    batched_notes = sort_batched_notes(batched_notes, by)
    pitches, intervals = batched_notes_to_notes(batched_notes)
    return pitches, intervals

def filter_notes(pitches, intervals, profile=None, min_time=-np.inf, max_time=np.inf, suppress_warnings=True):
    pitches_r = np.round(pitches)
    valid_idcs = np.ones(len(pitches), dtype=bool)
    if profile is not None:
        in_bounds_pitch = np.logical_and((pitches_r >= profile.low), (pitches_r <= profile.high))
        valid_idcs = np.logical_and(valid_idcs, in_bounds_pitch)
    in_bounds_interval_on = (intervals[:, 0] <= max_time)
    in_bounds_interval_off = (intervals[:, 1] >= min_time)
    valid_idcs = np.logical_and(valid_idcs, np.logical_and(in_bounds_interval_on, in_bounds_interval_off))
    pitches, intervals = pitches[valid_idcs], intervals[valid_idcs]
    return pitches, intervals

def threshold_activations(activations, threshold=0.5):
    if isinstance(activations, torch.Tensor):
        zeros = torch.zeros_like(activations)
        ones = torch.ones_like(activations)
        return torch.where(activations < threshold, zeros, ones)
    else:
        activations[activations < threshold] = 0
        activations[activations != 0] = 1
        return activations

def tensor_to_array(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    return data

def array_to_tensor(data, device=None):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if device is not None:
            data = data.to(device)
    return data

def dict_to_device(track, device):
    track = deepcopy(track)
    for key in track.keys():
        if isinstance(track[key], dict):
            track[key] = dict_to_device(track[key], device)
        elif isinstance(track[key], torch.Tensor):
            track[key] = track[key].to(device)
    return track

def dict_to_tensor(track):
    track = deepcopy(track)
    for key in track.keys():
        if isinstance(track[key], dict):
            track[key] = dict_to_tensor(track[key])
        elif isinstance(track[key], np.ndarray):
            track[key] = array_to_tensor(track[key])
    return track

def dict_to_array(track):
    for key in track.keys():
        if isinstance(track[key], dict):
            track[key] = dict_to_array(track[key])
        elif isinstance(track[key], torch.Tensor):
            track[key] = tensor_to_array(track[key])
    return track

def query_dict(dictionary, key):
    return key in dictionary.keys()

def unpack_dict(data, key):
    entry = None
    if isinstance(data, dict) and query_dict(data, key):
        entry = data[key]
    return entry
import numpy as np
import torch
import librosa
from copy import deepcopy

def framify_activations(activations, win_length, hop_length=1, pad=True):
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

def tablature_to_logistic(tablature, profile, silence=False):
    stacked_multi_pitch = tablature_to_stacked_multi_pitch(tablature, profile)
    logistic_activations = stacked_multi_pitch_to_logistic(stacked_multi_pitch, profile, silence)
    return logistic_activations

def logistic_to_tablature(logistic, profile, silence=False, silence_thr=0.5):
    """
    Converts a logistic activation tensor (sigmoid applied) into a tablature representation.
    This version expects the input tensor to be grouped by string.
    Expected input shape: (..., num_strings, num_classes_per_string)
    """
    if logistic.dim() < 2:
        raise ValueError("Input tensor for logistic_to_tablature must have at least 2 dimensions.")

    max_activations, highest_class = torch.max(logistic, dim=-1)

    silent_frames = max_activations < silence_thr
    highest_class[silent_frames] = -1
        
    return highest_class

def tablature_to_stacked_multi_pitch(tablature, profile):
    is_tensor = isinstance(tablature, torch.Tensor)
    
    if is_tensor:
        if tablature.dim() == 2:  
            tablature = tablature.unsqueeze(0)  
    else: 
        if tablature.ndim == 2:
            tablature = np.expand_dims(tablature, axis=0)

    num_dofs, num_frames = tablature.shape[-2:]
    num_pitches = profile.get_range_len()
    
    stacked_multi_pitch = np.zeros(tablature.shape[:-2] + (num_dofs, num_pitches, num_frames))
    
    tuning = np.array(profile.get_midi_tuning()) 
    
    dof_start = (tuning - profile.low)[np.newaxis, :, np.newaxis]
    
    if is_tensor:
        dof_start = torch.Tensor(dof_start).to(tablature.device)
    
    non_silent_frames = tablature >= 0
    
    pitch_idcs = (tablature + dof_start)[non_silent_frames]
    non_silent_idcs = non_silent_frames.nonzero()

    if is_tensor:
        pitch_idcs = pitch_idcs.long()
        non_silent_idcs = tuple(non_silent_idcs.transpose(-2, -1))
        stacked_multi_pitch = torch.from_numpy(stacked_multi_pitch).to(tablature.device)
        stacked_multi_pitch = stacked_multi_pitch.to(tablature.dtype)
    else:
        pitch_idcs = pitch_idcs.astype(np.int64)

    other_idcs, frame_idcs = non_silent_idcs[:-1], non_silent_idcs[-1]
    stacked_multi_pitch[other_idcs + (pitch_idcs, frame_idcs)] = 1
    
    return stacked_multi_pitch

def stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch):
    if isinstance(stacked_multi_pitch, torch.Tensor):
        multi_pitch = torch.max(stacked_multi_pitch, dim=-3)[0]
    else:
        multi_pitch = np.max(stacked_multi_pitch, axis=-3)
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
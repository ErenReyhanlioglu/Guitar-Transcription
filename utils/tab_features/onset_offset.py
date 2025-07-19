import numpy as np

def extract_onsets_offsets(notes, total_frames, fs=22050, hop=512):
    """
    Generate onset and offset matrices per string from note segments.

    Each note is marked with a binary value of 1.0 at its start (onset) and 
    end (offset) frame. This representation is useful for time-localized 
    training targets (e.g., onset detection, duration modeling).

    Parameters:
        notes (dict): 
            Dictionary of notes per string, in the form:
            {string_index: list of (frequency, (start_frame, end_frame)) tuples}.
        
        total_frames (int): 
            Total number of time frames (T).
        
        fs (int): 
            Sampling rate of the audio (default = 22050).
        
        hop (int): 
            Hop size used during feature extraction (default = 512).

    Returns:
        onsets (np.ndarray): 
            Binary matrix of shape (6, T), where 1 indicates an onset frame.
        
        offsets (np.ndarray): 
            Binary matrix of shape (6, T), where 1 indicates an offset frame.
    """
    num_strings = 6
    onsets = np.zeros((num_strings, total_frames), dtype=np.float32)
    offsets = np.zeros((num_strings, total_frames), dtype=np.float32)

    for s in range(num_strings):
        if s not in notes:
            continue
        for _, (start, end) in notes[s]:
            if 0 <= start < total_frames:
                onsets[s, start] = 1.0
            if 0 <= end < total_frames:
                offsets[s, end] = 1.0

    return onsets, offsets

def expand_onsets_offsets_harmonics(onsets, offsets, num_harmonics=44):
    """
    Expand (6, T) matrices to (6, H, T) along harmonic dimension.
    """
    onsets_h = np.repeat(onsets[:, np.newaxis, :], num_harmonics, axis=1)
    offsets_h = np.repeat(offsets[:, np.newaxis, :], num_harmonics, axis=1)
    return onsets_h, offsets_h

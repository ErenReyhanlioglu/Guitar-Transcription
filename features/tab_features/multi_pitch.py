import numpy as np

def extract_multi_pitch(notes, total_frames, num_harmonics=44):
    """
    Generate a binary multi-pitch matrix with (string, harmonic, time) shape.

    Each note activates one or more harmonics over its duration.

    Parameters:
        notes (dict): {s: [(freq, (start, end)), ...]}
        total_frames (int): total number of frames
        num_harmonics (int): number of harmonics (e.g., 44)

    Returns:
        np.ndarray: shape (6, num_harmonics, total_frames)
    """
    multi_pitch = np.zeros((6, num_harmonics, total_frames), dtype=np.float32)

    for s in range(6):
        if s not in notes:
            continue
        for h, (freq, (start, end)) in enumerate(notes[s]):
            for t in range(start, end):
                if 0 <= t < total_frames and h < num_harmonics:
                    multi_pitch[s, h, t] = 1.0

    return multi_pitch
import pickle
import numpy as np
import math

def freq_to_midi(freq: float) -> int | None:
    """
    Convert a frequency in Hz to the nearest MIDI pitch number.
    Returns None if frequency is zero or negative.

    Args:
        freq (float): Frequency in Hz.

    Returns:
        int or None: MIDI pitch number or None if invalid frequency.
    """
    if freq <= 0:
        return None
    return int(round(69 + 12 * math.log2(freq / 440.0)))


def extract_notes_and_pitch_list(
    pitch_files: dict,
    fs: int,
    hop: int,
    total_frames: int | None = None
) -> tuple[dict[int, list[tuple[float, tuple[int, int]]]], dict[int, list[float]]]:
    """
    Extract note segments and pitch lists per guitar string from SynthTab pitch .pkl files.

    Each .pkl file contains a list of frame-level pitch arrays from a pitch tracker.
    This function converts these frame-level pitch arrays into continuous note segments,
    each represented by its mean frequency and frame start/end indices.

    Args:
        pitch_files (dict): Mapping from string index (0-5) to .pkl file path.
        fs (int): Sampling rate of the original audio (default 22050 Hz).
        hop (int): Hop size used in STFT/feature extraction (default 512 samples).
        total_frames (int or None): Optional limit on total number of frames to consider.

    Returns:
        notes (dict): 
            {string_index: list of tuples (frequency_Hz, (start_frame, end_frame))}
            Represents continuous note segments per string.
        pitch_list (dict): 
            {string_index: list of frame-level pitch frequencies (Hz)}
            Flattened frame-level pitch values, useful for visualization or stats.

    Notes:
        - Frames with non-positive frequencies are discarded.
        - Frame indices are clipped by total_frames if provided.
        - Assumes one pitch array per frame.
    """
    notes = {}
    pitch_list = {}

    for s, path in pitch_files.items():
        with open(path, "rb") as f:
            data = pickle.load(f)

        segments = []
        freqs = []

        for i, pitch_array in enumerate(data):
            if not isinstance(pitch_array, np.ndarray) or pitch_array.size == 0:
                continue

            freq = float(np.mean(pitch_array))
            if freq <= 0:
                continue

            start_frame = i
            end_frame = i + 1

            if total_frames is not None:
                start_frame = min(start_frame, total_frames - 1)
                end_frame = min(end_frame, total_frames)

            segments.append((freq, (start_frame, end_frame)))
            freqs.append(freq)

        notes[s] = segments
        pitch_list[s] = freqs

    return notes, pitch_list


def extract_notes_framewise(
    pitch_files: dict,
    total_frames: int
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """
    Extract frame-level pitch notes and pitch lists per guitar string.

    This function produces a frame-level matrix of frequencies per string and
    a pitch_list dict containing arrays of frequencies and their frame times.

    Args:
        pitch_files (dict): Mapping from string index (0-5) to .pkl file path.
        total_frames (int): Number of frames to consider.

    Returns:
        notes (np.ndarray): shape (6, total_frames), frequency at each frame, 0 for silence.
        pitch_list (dict): {string_index: (freq_array, time_array)} frame-level pitches and times.
    """
    num_strings = 6
    notes = np.zeros((num_strings, total_frames), dtype=np.float32)
    pitch_list = {}

    for s in range(num_strings):
        path = pitch_files.get(s)
        if path is None:
            pitch_list[s] = (np.array([], dtype=np.float32), np.array([], dtype=np.int64))
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        freqs = []
        times = []

        for i, pitch_array in enumerate(data):
            if i >= total_frames:
                break
            if isinstance(pitch_array, np.ndarray) and pitch_array.size > 0:
                freq = float(np.mean(pitch_array))
                if freq > 0:
                    notes[s, i] = freq
                    freqs.append(freq)
                    times.append(i)

        pitch_list[s] = (np.array(freqs, dtype=np.float32), np.array(times, dtype=np.int64))

    return notes, pitch_list

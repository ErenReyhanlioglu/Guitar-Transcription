#utils.py
import numpy as np
import jams
from utils.tab_features.onset_offset import extract_onsets_offsets, expand_onsets_offsets_harmonics
from utils.tab_features.multi_pitch import extract_multi_pitch
from utils.tab_features.transforms import (
    compute_tablature_adj,
    compute_tablature_rel,
    compute_tablature_adj_spatial,
    compute_tablature_rel_spatial
)

def extract_open_string_midi_from_jams(jams_path: str, num_strings: int = 6) -> list:
    """
    Extract open string MIDI tuning values from a .jams file with 'note_tab' namespace.

    Assumes string indices are 1-based (i.e., 1–6), and maps to list index 0–5.
    Returns a list [high E, B, G, D, A, low E] with MIDI pitch values.

    Args:
        jams_path (str): Path to the .jams file.
        num_strings (int): Number of strings (default: 6).

    Returns:
        list[int]: Open string MIDI pitches in high E → low E order.
    """
    jam = jams.load(jams_path, validate=False)
    open_midi = [None] * num_strings

    for ann in jam.annotations:
        if ann.namespace != "note_tab":
            continue
        try:
            string_index = int(ann.sandbox.string_index)  # 1-based (1–6)
            tuning = int(ann.sandbox.open_tuning)
            idx = string_index - 1
            if 0 <= idx < num_strings:
                open_midi[idx] = tuning
        except Exception as e:
            print(f"⚠️ Sandbox parse error: {e}")
            continue

    return open_midi  # [string_1, string_2, ..., string_6]

def extract_tab_features(
    track_name: str,
    fs: int,
    audio: np.ndarray,
    tablature: np.ndarray,
    string_notes: dict,
    num_frames: int | None = None,
    hop_length: int = 512,
    pitch_bins: np.ndarray | None = None,
    num_harmonics: int = 44,
    max_fret: int = 20,
    silence_val: int = -1
) -> dict:
    """
    Generate a comprehensive set of frame-level guitar tablature features from SynthTab data.

    This function combines symbolic tablature information, audio metadata, and derived
    time-frequency features such as onset and offset maps and multi-pitch representations.
    All feature arrays are aligned along the time dimension and returned as a unified dictionary.

    Parameters:
        track_name (str): Identifier or filename of the track.
        fs (int): Audio sample rate in Hz.
        audio (np.ndarray): Raw mono audio waveform, shape (samples,).
        tablature (np.ndarray): Integer matrix (6 strings x num_frames) with fret numbers.
                                Silent frames are indicated with `silence_val`.
        string_notes (dict): Per-string note segments of form:
                             {string_index: [(onset_frame, offset_frame, midi_pitch), ...]}.
        num_frames (int or None): Number of frames to process; inferred from tablature if None.
        hop_length (int): Hop size in samples used for frame alignment.
        pitch_bins (np.ndarray or None): Frequency bin centers (Hz) for multi-pitch features (optional).
        num_harmonics (int): Number of harmonic bins for harmonic expansion features.
        max_fret (int): Maximum fret number to represent in spatial features.
        silence_val (int): Value indicating silence in tablature.

    Returns:
        dict: A dictionary containing extracted features including:
            - 'track', 'fs', 'audio'
            - 'tablature', 'tablature_adj', 'tablature_rel'
            - 'tablature_adj_spatial', 'tablature_rel_spatial'
            - 'onsets', 'offsets', 'onsets_h', 'offsets_h'
            - 'multi_pitch'
    """
    if num_frames is None:
        num_frames = tablature.shape[1]

    features = {}

    # Metadata
    features["track"] = track_name
    features["fs"] = fs
    features["audio"] = audio

    # Tablature symbolic and continuous pitch representations
    features["tablature"] = tablature
    features["tablature_adj"] = compute_tablature_adj(tablature, silence_val=silence_val)
    features["tablature_rel"] = compute_tablature_rel(tablature, silence_val=silence_val)

    # Spatial (string x fret x time) one-hot and continuous representations
    features["tablature_adj_spatial"] = compute_tablature_adj_spatial(
        tablature, max_fret=max_fret, silence_val=silence_val
    )
    features["tablature_rel_spatial"] = compute_tablature_rel_spatial(
        tablature, max_fret=max_fret, silence_val=silence_val
    )

    # Onset and offset binary maps per string
    onsets, offsets = extract_onsets_offsets(string_notes, num_frames, fs, hop_length)
    features["onsets"] = onsets
    features["offsets"] = offsets

    # Harmonic expansions of onsets and offsets
    onsets_h, offsets_h = expand_onsets_offsets_harmonics(onsets, offsets, num_harmonics=num_harmonics)
    features["onsets_h"] = onsets_h
    features["offsets_h"] = offsets_h

    # Multi-pitch harmonic representation per string
    features["multi_pitch"] = extract_multi_pitch(
        notes=string_notes,
        total_frames=num_frames,
        num_harmonics=num_harmonics
    )

    return features

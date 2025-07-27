import numpy as np
import soundfile as sf
import warnings

def check_time_alignment(data_dict, verbose=True):
    """
    Check that all time-varying arrays share the same time dimension T.

    This function inspects feature tensors like 'tablature', 'onsets',
    'multi_pitch', and confirms that their last dimension is consistent.

    Parameters:
        data_dict (dict): 
            Dictionary containing frame-level features.

        verbose (bool): 
            If True, prints detailed info (default = True).

    Returns:
        bool: True if all features have matching T dimension, False otherwise.
    """
    time_dims = {}

    for key in ['tablature', 'tablature_adj', 'tablature_rel',
                'onsets', 'offsets', 'multi_pitch', 'pitch_list']:
        value = data_dict.get(key)
        if value is None:
            continue
        try:
            time_dim = value.shape[-1]
            time_dims[key] = time_dim
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not extract time dimension from {key}: {e}")

    if verbose:
        print("\n🧪 Time dimensions (T):")
        for k, v in time_dims.items():
            print(f"  - {k}: {v}")

    if len(set(time_dims.values())) > 1:
        warnings.warn(f"Inconsistent time dimensions found: {time_dims}")
        return False
    return True


def check_empty_strings(data_dict, verbose=True):
    """
    Detect if any string (0–5) is completely empty in key features.

    This is helpful to catch data quality issues where some strings have 
    no annotations or pitch content at all.

    Parameters:
        data_dict (dict): Feature dictionary containing 'notes', 'multi_pitch', etc.
        verbose (bool): If True, prints info about empty strings.

    Returns:
        bool: True if all strings have data; False if any string is fully empty.
    """
    keys_to_check = ['notes', 'multi_pitch', 'onsets', 'offsets']
    empty_detected = False

    for key in keys_to_check:
        data = data_dict.get(key)
        if data is None:
            continue

        for s, value in enumerate(data):
            if isinstance(value, np.ndarray) and value.size == 0:
                if verbose:
                    print(f"[EMPTY] No data in string {s} for '{key}'")
                empty_detected = True

    return not empty_detected


def check_audio_properties(audio_path, expected_sr=22050, verbose=True):
    """
    Check if the audio file is mono and has the expected sample rate.

    Parameters:
        audio_path (str): Path to the audio file (e.g., .flac or .wav).
        expected_sr (int): Desired sampling rate (default = 22050).
        verbose (bool): If True, prints issues.

    Returns:
        bool: True if audio is valid; False otherwise.
    """
    try:
        audio, sr = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"[ERR] Failed to load audio: {audio_path} ({e})")

    issues = []

    if sr != expected_sr:
        issues.append(f"Sample rate mismatch: {sr} (expected: {expected_sr})")
    if audio.ndim != 1:
        issues.append(f"Audio is not mono (channels: {audio.ndim})")

    if verbose and issues:
        print(f"[AUDIO CHECK] {audio_path}")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def validate(data_dict, audio_path, expected_sr=22050):
    """
    Run a full suite of validation checks for SynthTab features and audio.

    Checks:
        1. Audio properties (mono, sample rate)
        2. Time dimension consistency
        3. Empty strings in annotations

    Parameters:
        data_dict (dict): Feature dictionary produced by extract_tab_features().
        audio_path (str): Path to associated audio file.
        expected_sr (int): Target sample rate (default = 22050).

    Returns:
        None
    """
    print("Running sanity checks...")
    check_audio_properties(audio_path, expected_sr=expected_sr)
    check_time_alignment(data_dict)
    check_empty_strings(data_dict)
    print("Sanity checks completed.\n")

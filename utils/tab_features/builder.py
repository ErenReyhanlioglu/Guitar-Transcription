import numpy as np
import soundfile as sf
import librosa
import jams
import os

from .transforms import (
    compute_tablature_adj,
    compute_tablature_rel,
    compute_tablature_adj_spatial,
    compute_tablature_rel_spatial
)
from .notes import extract_notes_and_pitch_list
from .onset_offset import extract_onsets_offsets
from .multi_pitch import extract_multi_pitch
from .utils import extract_tab_features
from .pitch import compute_tablature
import utils.tab_features.sanity as sanity
from librosa.core import cqt_frequencies

MIDI_OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E A D G B E

def freq_to_midi(freq: float) -> int | None:
    """
    Converts frequency (Hz) to nearest MIDI pitch value.
    Returns None for non-positive frequencies.
    """
    import math
    if freq <= 0:
        return None
    return int(round(69 + 12 * math.log2(freq / 440.0)))

def build_tab_npz(
    jams_path: str,
    audio_path: str,
    pitch_folder: str | None = None,
    extracted_features=None,
    sr: int = 22050,
    hop_length: int = 512,
    n_bins: int = 192,
    bins_per_octave: int = 24,
    fmin: float = 82.41,
    num_harmonics: int = 44,
    max_fret: int = 20,
    silence_val: int = -1,
    save_path: str | None = None,
    verbose: bool = False
) -> dict:
    """
    Builds a frame-level .npz feature dictionary from SynthTab annotations and audio.
    Includes symbolic tablature, pitch representations, onsets, multi-pitch, and
    optionally spectral features extracted via FeatureModule/FeatureCombo.

    Parameters:
        jams_path (str): Path to JAMS file.
        audio_path (str): Path to audio file (e.g., WAV or FLAC).
        pitch_folder (str): Path to pitch.pkl files per string.
        extracted_features (FeatureModule or FeatureCombo or None): Optional spectral features extractor.
        sr (int): Target sample rate.
        hop_length (int): Frame hop length.
        n_bins (int): Number of frequency bins for pitch.
        bins_per_octave (int): Bins per octave.
        fmin (float): Minimum frequency (Hz).
        num_harmonics (int): Number of harmonic bins.
        max_fret (int): Max fret value for spatial encoding.
        silence_val (int): Value used for silent frames.
        save_path (str): If given, saves the resulting npz.
        verbose (bool): If True, runs sanity checks.

    Returns:
        dict: Dictionary of all extracted frame-level features.
    """

    # Load annotation and audio
    jam = jams.load(jams_path, validate=False)
    audio, sr_orig = sf.read(audio_path)
    if sr_orig != sr:
        print(f"Resampling audio from {sr_orig} Hz to {sr} Hz.")
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr)

    # Replace original sample rate with resampled rate if changed
    effective_sr = sr

    track = os.path.basename(jams_path).replace(".jams", "")
    T = int(np.ceil(len(audio) / hop_length))

    if pitch_folder is None:
        raise ValueError("pitch_folder must be specified and contain 6 *_pitch.pkl files.")

    pitch_files = {
        s: os.path.join(pitch_folder, f"luthier_pick_nonoise_mono_body_string_{s+1}_pitch.pkl")
        for s in range(6)
    }

    notes, pitch_list = extract_notes_and_pitch_list(pitch_files, fs=effective_sr, hop=hop_length, total_frames=T)

    # Convert notes to pitch_dict for tablature computation
    pitch_dict = {}
    for s in range(6):
        pitch_dict[s] = []
        for item in notes[s]:
            if len(item) != 2:
                continue
            freq, (start_frame, end_frame) = item
            midi = freq_to_midi(freq)
            if midi is None:
                continue
            pitch_dict[s].append((start_frame, end_frame, midi))

    tablature = compute_tablature(pitch_dict=pitch_dict, open_string_midi=MIDI_OPEN_STRINGS, T=T)

    pitch_bins = cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    features = extract_tab_features(
        track_name=track,
        fs=effective_sr,
        audio=audio,
        tablature=tablature,
        string_notes=notes,
        num_frames=T,
        hop_length=hop_length,
        pitch_bins=pitch_bins,
        num_harmonics=num_harmonics,
        max_fret=max_fret,
        silence_val=silence_val
    )

    features["notes"] = notes
    features["pitch_list"] = pitch_list
    features["pitch_bins"] = pitch_bins

    # Extract spectral features if module provided
    if extracted_features and not isinstance(extracted_features, dict):
        try:
            spec_feats = extracted_features.process_audio(audio)
            if isinstance(spec_feats, dict):
                features.update(spec_feats)
            elif isinstance(spec_feats, list):
                for idx, f in enumerate(spec_feats):
                    features[f"{extracted_features.features_name()}_{idx}"] = f
            elif isinstance(spec_feats, np.ndarray):
                features[extracted_features.features_name()] = spec_feats
        except Exception as e:
            print(f"Feature extraction failed: {e}")

    if verbose:
        sanity.validate(features, audio_path, expected_sr=effective_sr)

    if "allow_pickle" in features:
        del features["allow_pickle"]

    if save_path:
        np.savez(save_path, **features, allow_pickle=True)

    return features
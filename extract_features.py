"""CLI entry point for turning raw GuitarSet audio into the feature-enriched
.npz files that train.py / evaluate.py expect as --data-root.

Three chained stages, each writable to its own subdirectory of --output-dir:

    1. ground_truth  - raw GuitarSet audio/annotations -> one .npz per track
                        with `audio`, `tablature`, `onsets`, `notes` (via
                        guitar_transcription_continuous.GuitarSetPlus).
    2. all_features   - adds `hcqt`, `cqt`, `mel`, `stft`, `power` to each
                        .npz (via amt_tools's feature extractors).
    3. enriched        - adds `hand_position`, `string_activity`, `pitch_class`
                        auxiliary-task targets derived from `tablature`
                        (paper section III-A-2, eq. (1)). This is the final
                        directory: point --data-root at it for train.py.

Needs its own Python 3.11 environment, separate from the project's main
(3.12) .venv — guitar-transcription-continuous depends on `muda`, an
unmaintained package that doesn't install under 3.12 (its setup.py imports
the `imp` stdlib module, removed in 3.12). Not needed for training/evaluation
once .npz files already exist. One-time setup:
    uv venv --python 3.11 .venv-extract
    uv pip install --python .venv-extract/Scripts/python.exe numpy librosa "setuptools<81" \\
        "amt-tools @ git+https://github.com/cwitkowitz/amt-tools" \\
        "guitar-transcription-continuous @ git+https://github.com/cwitkowitz/guitar-transcription-continuous"

Example:
    .venv-extract/Scripts/python.exe extract_features.py \\
        --guitarset-root /path/to/GuitarSet --output-dir /path/to/output
"""
import argparse
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

TUNING = np.array([40, 45, 50, 55, 59, 64])  # standard tuning, MIDI: E2 A2 D3 G3 B3 E4


# ─────────────────────────────────────────────────────────────────────────
# STAGE 1: raw audio -> ground-truth .npz (audio, tablature, onsets, notes)
# ─────────────────────────────────────────────────────────────────────────

def _build_ground_truth(guitarset_root: str, ground_truth_dir: str) -> None:
    """Stage 1: runs GuitarSetPlus over the raw dataset and copies its per-track ground-truth
    .npz files (audio, tablature, onsets, notes) into `ground_truth_dir`."""
    from amt_tools.tools.instrument import GuitarProfile
    from guitar_transcription_continuous.datasets.GuitarSet import GuitarSetPlus

    class SmartGuitarSet(GuitarSetPlus):
        """Fixes GuitarSetPlus's wav-file lookup across GuitarSet's inconsistent
        directory/naming conventions (mono_mic vs audio_mono-mic, comp vs solo)."""

        def __init__(self, *args, **kwargs):
            kwargs["profile"] = GuitarProfile()
            super().__init__(*args, **kwargs)

        def get_wav_path(self, track):
            base = self.base_dir
            if track.endswith("_comp") or track.endswith("_solo"):
                track = track[:-5]

            candidates = [
                os.path.join(base, "audio", "mono_mic", f"{track}_comp_mic.wav"),
                os.path.join(base, "audio", "mono_mic", f"{track}_solo_mic.wav"),
                os.path.join(base, "audio_mono-mic", f"{track}_comp_mic.wav"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            raise FileNotFoundError(f"Couldn't find any valid .wav file for track {track}")

    logger.info(f"Building ground-truth .npz files from: {guitarset_root}")
    dataset = SmartGuitarSet(
        base_dir=guitarset_root,
        hop_length=512,
        sample_rate=22050,
        evaluation_extras=True,
        save_data=True,
    )

    # GuitarSetPlus writes into its own package-relative "generated" directory;
    # copy the result into our own output tree so it isn't scattered inside
    # site-packages.
    source_dir = dataset.get_gt_dir()
    os.makedirs(ground_truth_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(source_dir):
        if fname.endswith(".npz"):
            import shutil
            shutil.copy2(os.path.join(source_dir, fname), os.path.join(ground_truth_dir, fname))
            copied += 1
    logger.info(f"Copied {copied} ground-truth .npz files -> {ground_truth_dir}")


# ─────────────────────────────────────────────────────────────────────────
# STAGE 2: + hcqt / cqt / mel / stft / power
# ─────────────────────────────────────────────────────────────────────────

def _extract_spectral_features(ground_truth_dir: str, all_features_dir: str, sample_rate: int = 22050) -> None:
    """Stage 2: adds HCQT, CQT, Mel, STFT, and power features to each ground-truth .npz,
    writing the enriched files to `all_features_dir`."""
    import librosa
    from amt_tools.features.cqt import CQT
    from amt_tools.features.hcqt import HCQT
    from amt_tools.features.mel import MelSpec
    from amt_tools.features.power import SignalPower
    from amt_tools.features.stft import STFT

    os.makedirs(all_features_dir, exist_ok=True)
    hop_length = 512

    hcqt_mod = HCQT(sample_rate=sample_rate, hop_length=hop_length, bins_per_octave=36, n_bins=144,
                     harmonics=[0.5, 1, 2, 3, 4, 5], fmin=librosa.note_to_hz("E2"), decibels=True)
    cqt_mod = CQT(sample_rate=sample_rate, hop_length=hop_length, fmin=librosa.note_to_hz("E2"),
                  n_bins=144, bins_per_octave=36, decibels=True)
    mel_mod = MelSpec(sample_rate=sample_rate, hop_length=hop_length, n_mels=256, n_fft=2048, decibels=True)
    power_mod = SignalPower(sample_rate=sample_rate, hop_length=hop_length, decibels=True)
    stft_mod = STFT(sample_rate=sample_rate, hop_length=hop_length, n_fft=2048, decibels=True)

    npz_files = sorted(f for f in os.listdir(ground_truth_dir) if f.endswith(".npz"))
    logger.info(f"Extracting HCQT/CQT/Mel/STFT/Power for {len(npz_files)} tracks...")

    for npz_file in npz_files:
        input_path = os.path.join(ground_truth_dir, npz_file)
        output_path = os.path.join(all_features_dir, npz_file)

        data = dict(np.load(input_path, allow_pickle=True))
        if "audio" not in data:
            logger.warning(f"'{npz_file}' has no 'audio' key, skipping.")
            continue

        audio = data["audio"].astype(np.float32)
        data["hcqt"] = hcqt_mod.process_audio(audio)
        data["cqt"] = cqt_mod.process_audio(audio)
        data["mel"] = mel_mod.process_audio(audio)
        data["stft"] = stft_mod.process_audio(audio)
        data["power"] = power_mod.process_audio(audio)

        np.savez_compressed(output_path, **data)

    logger.info(f"Spectral features written -> {all_features_dir}")


# ─────────────────────────────────────────────────────────────────────────
# STAGE 3: + hand_position / string_activity / pitch_class (paper eq. 1)
# ─────────────────────────────────────────────────────────────────────────

def generate_hand_position(tablature: np.ndarray) -> np.ndarray:
    """(6, T) fret-per-string tablature -> (T,) coarse fret-region label.

    0=no fretted note, 1: 0<mean_fret<=5, 2: <=10, 3: <=15, 4: >15.
    Matches the paper's eq. (1); open strings (fret 0) are excluded from the
    mean since they don't pin down a fretting-hand location.
    """
    num_frames = tablature.shape[1]
    hand_pos = np.zeros(num_frames, dtype=np.int8)
    for t in range(num_frames):
        active_frets = tablature[:, t][tablature[:, t] > 0]
        if len(active_frets) > 0:
            avg_fret = np.mean(active_frets)
            if avg_fret <= 5: hand_pos[t] = 1
            elif avg_fret <= 10: hand_pos[t] = 2
            elif avg_fret <= 15: hand_pos[t] = 3
            else: hand_pos[t] = 4
    return hand_pos


def generate_string_activity(tablature: np.ndarray) -> np.ndarray:
    """(6, T) tablature -> (6, T) binary: is each string sounding at frame t."""
    return (tablature >= 0).astype(np.float32)


def generate_pitch_class(tablature: np.ndarray, tuning: np.ndarray) -> np.ndarray:
    """(6, T) tablature -> (12, T) binary multi-hot chroma: (tuning + fret) % 12."""
    num_frames = tablature.shape[1]
    chroma = np.zeros((12, num_frames), dtype=np.float32)
    for t in range(num_frames):
        col = tablature[:, t]
        for s in np.where(col >= 0)[0]:
            pitch_class = int((tuning[s] + col[s]) % 12)
            chroma[pitch_class, t] = 1.0
    return chroma


def _add_auxiliary_labels(all_features_dir: str, enriched_dir: str) -> None:
    """Stage 3: derives hand_position/string_activity/pitch_class targets from each file's
    tablature and writes the final enriched .npz files to `enriched_dir`."""
    os.makedirs(enriched_dir, exist_ok=True)
    npz_files = sorted(f for f in os.listdir(all_features_dir) if f.endswith(".npz"))
    logger.info(f"Deriving hand_position/string_activity/pitch_class for {len(npz_files)} tracks...")

    written, skipped = 0, 0
    for fname in npz_files:
        src_path = os.path.join(all_features_dir, fname)
        dst_path = os.path.join(enriched_dir, fname)

        data = dict(np.load(src_path, allow_pickle=True))
        if "tablature" not in data:
            logger.warning(f"'{fname}' has no 'tablature' key, skipping.")
            skipped += 1
            continue

        tablature = data["tablature"]
        data["hand_position"] = generate_hand_position(tablature)
        data["string_activity"] = generate_string_activity(tablature)
        data["pitch_class"] = generate_pitch_class(tablature, TUNING)

        np.savez_compressed(dst_path, **data)
        written += 1

    logger.info(f"Enriched {written} tracks ({skipped} skipped) -> {enriched_dir}")


def verify_dataset(enriched_dir: str) -> bool:
    """Structural sanity check: every .npz has tablature-consistent aux-label shapes."""
    npz_files = sorted(f for f in os.listdir(enriched_dir) if f.endswith(".npz"))
    corrupt = []

    for fname in npz_files:
        with np.load(os.path.join(enriched_dir, fname), allow_pickle=True) as data:
            if "tablature" not in data:
                corrupt.append((fname, "missing tablature")); continue
            num_frames = data["tablature"].shape[1]

            if "hand_position" not in data or data["hand_position"].shape[0] != num_frames:
                corrupt.append((fname, "hand_position shape mismatch")); continue
            if "string_activity" not in data or data["string_activity"].shape != (6, num_frames):
                corrupt.append((fname, "string_activity shape mismatch")); continue
            if "pitch_class" not in data or data["pitch_class"].shape != (12, num_frames):
                corrupt.append((fname, "pitch_class shape mismatch")); continue

    logger.info(f"Verified {len(npz_files)} files, {len(corrupt)} corrupt.")
    for fname, reason in corrupt:
        logger.warning(f"  {fname}: {reason}")
    return not corrupt


def parse_args():
    """Parses CLI arguments for the feature-extraction pipeline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--guitarset-root", required=True,
                         help="Path to the raw GuitarSet dataset root (contains audio/, annotation/, etc.).")
    parser.add_argument("--output-dir", required=True,
                         help="Base output directory; creates ground_truth/, all_features/, enriched/ under it.")
    parser.add_argument(
        "--stages", default="1,2,3",
        help="Comma-separated stages to run: 1=ground truth, 2=spectral features, 3=auxiliary labels. Default: all.",
    )
    parser.add_argument("--verify", action="store_true", help="Run a structural integrity check after stage 3.")
    return parser.parse_args()


def main():
    """CLI entry point: runs the requested extraction stages in order and optionally verifies the result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    ground_truth_dir = os.path.join(args.output_dir, "ground_truth")
    all_features_dir = os.path.join(args.output_dir, "all_features")
    enriched_dir = os.path.join(args.output_dir, "enriched")

    stages = {int(s) for s in args.stages.split(",")}

    if 1 in stages:
        _build_ground_truth(args.guitarset_root, ground_truth_dir)
    if 2 in stages:
        _extract_spectral_features(ground_truth_dir, all_features_dir)
    if 3 in stages:
        _add_auxiliary_labels(all_features_dir, enriched_dir)

    if args.verify:
        ok = verify_dataset(enriched_dir)
        if not ok:
            raise SystemExit("Dataset verification found corrupt files — see warnings above.")

    logger.info(f"Done. Point train.py's --data-root at: {enriched_dir}")


if __name__ == "__main__":
    main()

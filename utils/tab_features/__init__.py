"""
This module defines all major feature extraction functions used to build
frame-level SynthTab-compatible datasets for guitar transcription.

Each function is responsible for computing a specific set of features such as:
- Fret-level tablature
- MIDI pitch maps
- Onset/offset detection
- Multi-pitch binary masks
- Audio-aligned symbolic annotations
"""

from utils.tab_features.utils import extract_tab_features
from utils.tab_features.onset_offset import extract_onsets_offsets
from utils.tab_features.notes import extract_notes_and_pitch_list
from utils.tab_features.multi_pitch import extract_multi_pitch
from utils.tab_features.transforms import compute_tablature_adj, compute_tablature_rel
from utils.tab_features.builder import build_tab_npz

__all__ = [
    "extract_tab_features",
    "extract_onsets_offsets",
    "extract_notes_and_pitch_list",
    "extract_multi_pitch",
    "compute_tablature_adj",
    "compute_tablature_rel",
    "build_tab_npz"
]

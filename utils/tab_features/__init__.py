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

# The main function to build the entire feature set
from .builder import build_tab_npz

# Core logic for note extraction and timing, now self-contained
from .notes import (
    create_all_note_data_from_jams,
    midi_to_freq
)

# Core logic for tablature matrix creation (now called from within notes.py)
from .pitch import compute_tablature

# Helper functions for specific feature types
from .utils import extract_tab_features, extract_open_string_midi_from_jams
from .onset_offset import extract_onsets_offsets
from .multi_pitch import extract_multi_pitch

# Functions for creating different tablature representations
from .transforms import (
    compute_tablature_adj,
    compute_tablature_rel,
    compute_tablature_adj_spatial,
    compute_tablature_rel_spatial
)


# List of functions to be exposed when the module is imported
__all__ = [
    # Main builder
    "build_tab_npz",

    # Core logic
    "create_all_note_data_from_jams",
    "compute_tablature",

    # Feature-specific extractors
    "extract_tab_features",
    "extract_onsets_offsets",
    "extract_multi_pitch",

    # Tablature transformations
    "compute_tablature_adj",
    "compute_tablature_rel",
    "compute_tablature_adj_spatial",
    "compute_tablature_rel_spatial",

    # Utility functions
    "midi_to_freq",
    "extract_open_string_midi_from_jams",
]

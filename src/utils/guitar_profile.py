# src/utils/guitar_profile.py

class GuitarProfile:
    """Holds the instrument's pitch range and tuning, derived from the instrument config."""
    def __init__(self, instrument_config: dict):
        """Reads the MIDI pitch range, per-string fret count, and string count from `instrument_config`."""
        self.config = instrument_config
        self.low = self.config['min_midi']
        self.high = self.config['max_midi']
        self.num_pitches = self.config['num_frets'] + 1
        self.num_strings = self.config['num_strings']

    def get_midi_tuning(self) -> list[int]:
        """Returns each string's open-string MIDI pitch."""
        return self.config['tuning']

    def get_range_len(self) -> int:
        """Returns the number of MIDI pitches in the instrument's full playable range."""
        return self.high - self.low + 1
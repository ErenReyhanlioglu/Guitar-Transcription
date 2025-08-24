# src/utils/guitar_profile.py

class GuitarProfile:
    def __init__(self, instrument_config: dict):
        self.config = instrument_config
        self.low = self.config['min_midi']
        self.high = self.config['max_midi']
        self.num_pitches = self.config['num_frets'] + 1
        self.num_strings = self.config['num_strings'] 

    def get_midi_tuning(self) -> list[int]:
        return self.config['tuning']

    def get_range_len(self) -> int:
        return self.high - self.low + 1
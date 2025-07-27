#combo.py
import numpy as np
from .common import FeatureModule

class FeatureCombo(FeatureModule):
    """
    Composite feature extractor that applies multiple FeatureModule instances
    and returns their outputs as a named dictionary.

    Example output:
        {
            'cqt': ndarray of shape [1, F, T],
            'melspec': ndarray of shape [1, F, T],
            ...
        }
    """

    def __init__(self, modules):
        """
        Parameters
        ----------
        modules : list of FeatureModule
            List of initialized feature extraction modules.
        """
        assert all(isinstance(m, FeatureModule) for m in modules), "All modules must inherit from FeatureModule."
        self.modules = modules

    def get_expected_frames(self, audio):
        """
        Ensures all modules agree on the number of frames for the given audio.
        """
        frame_counts = [mod.get_expected_frames(audio) for mod in self.modules]
        assert len(set(frame_counts)) == 1, f"Mismatch in expected frame counts: {frame_counts}"
        return frame_counts[0]

    def process_audio(self, audio, target_frames=None): 
        """
        Processes audio with all feature modules.
        """
        feats = {}
        for mod in self.modules:
            name = mod.__class__.__name__.lower()
            try:
                output = mod.process_audio(audio, target_frames=target_frames)
                if output is not None:
                    feats[name] = output
                else:
                    print(f"[WARN] {name} returned None.")
            except Exception as e:
                print(f"[ERROR] {name} failed: {e}")
        return feats

    def get_sample_rate(self):
        rates = [mod.get_sample_rate() for mod in self.modules]
        assert len(set(rates)) == 1, "Inconsistent sample rates across modules."
        return rates[0]

    def get_hop_length(self):
        hops = [mod.get_hop_length() for mod in self.modules]
        assert len(set(hops)) == 1, "Inconsistent hop lengths across modules."
        return hops[0]

    def get_num_channels(self):
        """
        Returns total number of channels across all modules.
        Only used for analysis, not for concatenation.
        """
        return sum(mod.get_num_channels() for mod in self.modules)

    def get_feature_size(self):
        """
        Returns a combined tuple of all feature sizes.
        Useful for summary or debugging.
        """
        sizes = [mod.get_feature_size() for mod in self.modules]
        return tuple(dim for size in sizes for dim in (size if isinstance(size, tuple) else (size,)))

    @classmethod
    def features_name(cls):
        return "featurecombo"

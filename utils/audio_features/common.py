from abc import ABC, abstractmethod
import librosa
import numpy as np

class FeatureModule(ABC):
    """
    Abstract base class for frame-aligned audio feature extraction modules.
    Handles sample rate, hop size, dB normalization, and padding.
    Subclasses must implement `process_audio`.
    """

    def __init__(self, sample_rate, hop_length, num_channels, decibels=True, center=False):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_channels = num_channels
        self.decibels = decibels
        self.center = center  

    def get_expected_frames(self, audio):
        """
        Calculates expected number of frames given audio length and hop size.
        """
        if audio.shape[-1] == 0:
            return 0
        if self.center:
            return 1 + int(np.ceil(len(audio) / self.hop_length))
        else:
            return 1 + len(audio) // self.hop_length

    def get_sample_range(self, num_frames):
        """
        Calculates sample indices needed for `num_frames` frames.
        """
        if num_frames <= 0:
            return np.array([0])
        max_samples = num_frames * self.hop_length - 1
        min_samples = max(1, max_samples - self.hop_length + 1)
        return np.arange(min_samples, max_samples + 1)

    def get_num_samples_required(self):
        """
        Returns number of samples required to produce at least one frame.
        """
        return self.get_sample_range(1)[-1]

    @staticmethod
    def divisor_pad(audio, divisor):
        """
        Zero-pads audio to be divisible by given `divisor`.
        """
        pad_amt = divisor - (audio.shape[-1] % divisor)
        if pad_amt > 0 and pad_amt != divisor:
            audio = np.append(audio, np.zeros(pad_amt, dtype=np.float32))
        return audio

    def frame_pad(self, audio):
        """
        Applies zero-padding based on mode.
        - If center=True, librosa will do center-padding (internal).
        - If center=False, we ensure length divisible by hop.
        """
        if self.center:
            return audio  # librosa handles centering internally
        else:
            divisor = self.get_num_samples_required()
            if audio.shape[-1] > divisor:
                divisor = self.hop_length
            return self.divisor_pad(audio, divisor)

    def trim_center(self, feats, original_len):
        """
        If center=True, trims the extra padding from the start and end.
        Ensures output has expected frame count.
        """
        expected_frames = 1 + len(original_len) // self.hop_length
        if feats.shape[-1] > expected_frames:
            extra = feats.shape[-1] - expected_frames
            trim_left = extra // 2
            trim_right = extra - trim_left
            return feats[..., trim_left : feats.shape[-1] - trim_right]
        return feats

    @abstractmethod
    def process_audio(self, audio):
        """
        Subclasses must implement this method to return [C, F, T] shaped features.
        """
        raise NotImplementedError

    def to_decibels(self, feats):
        """
        Converts magnitude features to dB scale and normalizes to [0, 1].
        """
        return librosa.amplitude_to_db(feats, ref=np.max)

    def post_proc(self, feats, original_audio=None):
        """
        Optional dB conversion + normalization and shape expansion to [1, ..., T].
        If center=True and original_audio is provided, trimming is applied.
        """
        if self.center and original_audio is not None:
            feats = self.trim_center(feats, original_audio)

        if self.decibels:
            feats = self.to_decibels(feats)
            feats = (feats + 80) / 80  # Normalize from [-80, 0] dB → [0, 1]

        return np.expand_dims(feats, axis=0)  # [1, F, T] or [1, H, F, T]

    def get_times(self, audio):
        """
        Returns frame-wise times (in seconds).
        """
        num_frames = self.get_expected_frames(audio)
        return librosa.frames_to_time(np.arange(num_frames),
                                      sr=self.sample_rate,
                                      hop_length=self.hop_length,
                                      center=self.center)

    def get_sample_rate(self):
        return self.sample_rate

    def get_hop_length(self):
        return self.hop_length

    def get_num_channels(self):
        return self.num_channels

    @classmethod
    def features_name(cls):
        """
        Returns a string name identifier for this feature type.
        """
        return cls.__name__

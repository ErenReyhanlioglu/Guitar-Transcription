#audio_features.py
import librosa
import numpy as np
from .common import FeatureModule

class CQT(FeatureModule):
    """
    Constant-Q Transform (CQT) feature extractor.
    """
    def __init__(self, sample_rate=22050, hop_length=512, num_channels=1,
                 bins_per_octave=36, n_octaves=4, fmin='E2', decibels=True, center=False):
        super().__init__(sample_rate, hop_length, num_channels, decibels, center)
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.fmin = librosa.note_to_hz(fmin)
        self.n_bins = bins_per_octave * n_octaves

    def process_audio(self, audio, target_frames=None): 
        audio = self.frame_pad(audio)
        cqt = librosa.cqt(y=audio, sr=self.sample_rate, hop_length=self.hop_length,
                          fmin=self.fmin, n_bins=self.n_bins,
                          bins_per_octave=self.bins_per_octave)
        magnitude = np.abs(cqt)

        return self.post_proc(magnitude, original_audio=audio, target_frames=target_frames)


class STFT(FeatureModule):
    """
    Short-Time Fourier Transform (STFT) feature extractor.
    """
    def __init__(self, sample_rate=22050, hop_length=512, n_fft=1024,
                 decibels=True, center=False):
        super().__init__(sample_rate, hop_length, n_fft // 2 + 1, decibels, center)
        self.n_fft = n_fft

    def process_audio(self, audio, target_frames=None): 
        audio = self.frame_pad(audio)
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=self.center)
        magnitude = np.abs(stft)
        return self.post_proc(magnitude, original_audio=audio, target_frames=target_frames)


class HCQT(FeatureModule):
    """
    Harmonic Constant-Q Transform (HCQT) feature extractor.
    """
    def __init__(self, sample_rate=22050, hop_length=512, bins_per_octave=36,
                 n_octaves=4, fmin='E2', harmonics=[0.5, 1, 2, 3, 4, 5],
                 decibels=True, center=False):
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.n_bins = bins_per_octave * n_octaves
        self.fmin = librosa.note_to_hz(fmin)
        self.harmonics = harmonics
        self.num_harmonics = len(harmonics)
        super().__init__(sample_rate, hop_length, self.num_harmonics, decibels, center)

    def process_audio(self, audio, target_frames=None):
        audio = self.frame_pad(audio)
        hcqt_list = []
        for h in self.harmonics:
            fmin_h = self.fmin * h
            cqt_h = librosa.cqt(y=audio, sr=self.sample_rate, hop_length=self.hop_length,
                                fmin=fmin_h, n_bins=self.n_bins,
                                bins_per_octave=self.bins_per_octave)
            hcqt_list.append(np.abs(cqt_h))
        hcqt = np.stack(hcqt_list, axis=0)
        return self.post_proc(hcqt, original_audio=audio, target_frames=target_frames)


class MelSpec(FeatureModule):
    """
    Mel spectrogram feature extractor.
    """
    def __init__(self, sample_rate=22050, hop_length=512, n_mels=128,
                 n_fft=2048,  
                 decibels=True, center=False):
        super().__init__(sample_rate, hop_length, n_mels, decibels, center)
        self.n_mels = n_mels
        self.n_fft = n_fft 

    def process_audio(self, audio, target_frames=None): 
        audio = self.frame_pad(audio)
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate,
                                             hop_length=self.hop_length,
                                             n_mels=self.n_mels,
                                             n_fft=self.n_fft,  
                                             center=self.center)
        return self.post_proc(mel, original_audio=audio, target_frames=target_frames)


class SignalPower(FeatureModule):
    """
    RMS energy feature extractor.
    """
    def __init__(self, sample_rate=22050, hop_length=512,
                 decibels=False, center=False):
        super().__init__(sample_rate, hop_length, 1, decibels, center)

    def process_audio(self, audio, target_frames=None): 
        audio = self.frame_pad(audio)
        power = librosa.feature.rms(y=audio, hop_length=self.hop_length, center=self.center)
        return self.post_proc(power, original_audio=audio, target_frames=target_frames)
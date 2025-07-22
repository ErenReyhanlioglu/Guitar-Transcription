"""
Purpose:
    This module defines the PyTorch Dataset class for the guitar transcription task.
    It is responsible for loading pre-processed .npz files, extracting the
    relevant features (like CQT) and targets (tablature), and preparing them
    for the model by creating fixed-size windows.

Dependencies:
    - torch
    - numpy

Current Status:
    - Loads .npz files containing CQT spectrograms and tablature data.
    - Implements a sliding window mechanism to create fixed-size samples.
    - Handles the silence class (`-1`) based on a global or config flag,
      either mapping it to a specific class index or leaving it as -1 to be ignored.

Future Plans:
    - [ ] Implement on-the-fly data augmentation (e.g., pitch shifting, time stretching)
          to increase dataset variety and model robustness.
    - [ ] Add support for dynamically selecting different input features (CQT, MelSpec, HCQT)
          based on the experiment's config file.
    - [ ] Explore more efficient data loading strategies for extremely large datasets,
          such as using memory-mapped files.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os

class TablatureDataset(Dataset):
    def __init__(self, npz_paths, config):
        self.samples = []
        self.config = config
        self.window_size = config['data']['window_size']
        self.hop_size = config['data'].get('hop_size', self.window_size)
        self.include_silence = config['data']['include_silence']
        self.silence_class = config['data']['silence_class']
        
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            cqt = data["cqt"]
            tab = data["tablature"]
            T = cqt.shape[-1]

            for start in range(0, T - self.window_size + 1, self.hop_size):
                end = start + self.window_size
                self.samples.append({
                    "cqt": cqt[:, :, start:end],
                    "tablature": tab[:, start:end]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        cqt = torch.tensor(item["cqt"], dtype=torch.float32)
        tab = torch.tensor(item["tablature"], dtype=torch.long)

        if self.include_silence:
            tab[tab == -1] = self.silence_class
        else:
            tab[tab == -1] = -1

        return {"cqt": cqt, "tablature": tab}
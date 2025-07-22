"""
Purpose:
    This module implements the TabCNN architecture, a Convolutional Neural Network
    designed for guitar tablature transcription from spectrogram inputs.

Dependencies:
    - torch.nn
    - .base_model.BaseTabModel

Current Status:
    - Implements a functional CNN with several convolutional blocks followed by
      fully connected layers.
    - Inherits from `BaseTabModel` to ensure a standard interface and use the
      shared `SoftmaxGroups` utility.
    - The architecture is fixed based on the initial implementation.

Future Plans:
    - [ ] Make the architecture more flexible by allowing the number of layers,
          kernel sizes, and channels to be specified in the config file.
    - [ ] Experiment with adding Batch Normalization or other normalization layers.
"""
import torch.nn as nn
from .base_model import BaseTabModel

class TabCNN(BaseTabModel):
    def __init__(self, in_channels, num_freq, num_strings, num_classes, **kwargs):
        super().__init__(num_strings, num_classes)
        self.total_output_size = self.num_strings * self.num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.25)
        )
        
        conv_out_freq = num_freq // 2
        embedding_dim = 64 * conv_out_freq

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.total_output_size)
        )

    def forward(self, x, apply_softmax=False):
        B, C, F, T = x.shape
        out = self.conv(x)
        out = out.permute(0, 3, 1, 2)
        out = out.reshape(B, T, -1)
        out = self.fc(out)
        logits = out.view(B, T, self.num_strings, self.num_classes)
        
        if apply_softmax:
            probs = self.softmax_groups(out)
            return probs.view(B, T, self.num_strings, self.num_classes)
            
        return logits
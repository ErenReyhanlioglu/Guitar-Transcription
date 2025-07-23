# src/models/fretnet.py

import torch
import torch.nn as nn
from .base_model import BaseTabModel

class FretNet(BaseTabModel):
    def __init__(self, in_channels, num_freq, num_strings, num_classes, 
                 rnn_hidden_size=128, rnn_num_layers=2, **kwargs):
        super().__init__(num_strings, num_classes)
        self.total_output_size = self.num_strings * self.num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.25)
        )

        conv_out_freq = num_freq // 2 
        conv_out_channels = 64
        rnn_input_size = conv_out_channels * conv_out_freq

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,  # (Batch, Time, Features) 
            bidirectional=True 
        )

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.total_output_size)
        )

    def forward(self, x, apply_softmax=False):
        # x boyutu: (B, C, F, T) -> (Batch, Channels, Freq, Time)
        B, C, F, T = x.shape

        out = self.conv(x) # -> (B, 64, F/2, T)

        out = out.permute(0, 3, 1, 2) # -> (B, T, 64, F/2)
        out = out.reshape(B, T, -1) # -> (B, T, 64 * F/2)

        out, _ = self.rnn(out) # -> (B, T, rnn_hidden_size * 2)

        out = self.fc(out) # -> (B, T, num_strings * num_classes)

        logits = out.view(B, T, self.num_strings, self.num_classes)

        if apply_softmax:
            probs = self.softmax_groups(out)
            return probs.view(B, T, self.num_strings, self.num_classes)

        return logits
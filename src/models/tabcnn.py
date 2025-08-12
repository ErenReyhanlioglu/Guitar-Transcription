import torch
import torch.nn as nn
from .base_model import BaseTabModel

class TabCNN(BaseTabModel):
    """
    A Convolutional Neural Network (CNN) for guitar tablature transcription,
    inspired by the original TabCNN paper.

    This implementation is designed to work with windowed input frames (`framify`),
    treating each window as a separate image-like patch for classification.
    """
    def __init__(self, in_channels: int, num_freq: int, num_strings: int, num_classes: int, config: dict, **kwargs):
        """
        Initializes the TabCNN model architecture.

        Args:
            in_channels (int): Number of input channels for the spectrogram (e.g., 1).
            num_freq (int): Number of frequency bins in the input spectrogram.
            num_strings (int): The number of strings for the instrument.
            num_classes (int): The number of classes per string.
            config (dict): A dictionary containing the model and data configuration.
        """
        super().__init__(num_strings, num_classes)
        
        self.output_mode = config['loss']['type']
        
        if self.output_mode == 'softmax_groups':
            self.total_output_size = self.num_strings * self.num_classes
        elif self.output_mode == 'logistic_bank':
            self.total_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25)
        )
        
        # This calculation assumes the input is windowed (framified)
        # The trainer will handle reshaping the input to (B*T, C, F, W)
        # For MaxPool2d(2,2), both freq and time dimensions are halved.
        conv_out_freq = num_freq // 2
        window_width = config['data'].get('framify_window_size', 9)
        conv_out_time = window_width // 2
        embedding_dim = 64 * conv_out_freq * conv_out_time

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.total_output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the TabCNN model.

        Args:
            x (torch.Tensor): Input tensor with shape (Batch, Channels, Freq, Window).
                              The trainer is responsible for reshaping the original
                              data to feed windowed frames to this model.

        Returns:
            torch.Tensor: The output logits from the model.
                          Shape for SoftmaxGroups: (Batch, num_strings * num_classes)
                          Shape for LogisticBank: (Batch, num_strings * (num_classes - 1))
        """
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        logits = self.fc(out)
        return logits
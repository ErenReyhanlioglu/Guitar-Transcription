import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class BaseTabModel(nn.Module):
    """
    Base class for all guitar transcription models.
    Provides common functionality for model initialization and summary.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.instrument_config = config['instrument']
        self.num_strings = self.instrument_config['num_strings']
        
        # Tablature class count (19 frets + 1 open + 1 silence = 21)
        self.num_classes_per_string = config['model']['params']['num_classes']
        
    def forward(self, x):
        """
        Forward pass logic should be implemented by subclasses.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Returns a string representation of the model architecture and parameter count.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return (
            f"\n--- Model Summary ---\n"
            f"Architecture: {self.__class__.__name__}\n"
            f"Total Parameters: {total_params:,}\n"
            f"Trainable Parameters: {trainable_params:,}\n"
            f"---------------------\n"
        )
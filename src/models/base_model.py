import torch
import torch.nn as nn

class BaseTabModel(nn.Module):
    """
    An abstract base class for tablature transcription models.
    
    This class ensures that all models share a common initialization
    for the number of strings and classes, but it does not enforce a
    specific output layer, allowing for flexibility (e.g., SoftmaxGroups or LogisticBank).
    """
    def __init__(self, num_strings: int, num_classes: int):
        """
        Initializes the base model with instrument properties.
        
        Args:
            num_strings (int): The number of strings for the instrument (e.g., 6 for a guitar).
            num_classes (int): The number of classes per string (e.g., number of frets + 1 for silence).
        """
        super().__init__()
        self.num_strings = num_strings
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Each model must implement its own forward method!")
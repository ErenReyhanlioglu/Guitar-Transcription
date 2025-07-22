"""
Purpose:
    This module defines the `BaseTabModel`, an abstract base class from which all
    other model architectures in this project will inherit. Its purpose is to
    enforce a standard interface and contain logic that is shared across all models.

Dependencies:
    - torch.nn
    - src.utils.losses.SoftmaxGroups

Current Status:
    - Inherits from `torch.nn.Module`.
    - Initializes the shared `SoftmaxGroups` layer, which is required by all models
      for the per-string classification output.
    - Defines a `forward` method that raises `NotImplementedError`, forcing any
      child class to implement its own specific forward pass.

Future Plans:
    - [ ] Add a standardized weight initialization method (e.g., `_init_weights`) that
          can be called by child classes.
    - [ ] Include other shared helper methods, such as calculating FLOPs or model size.
"""

import torch.nn as nn
from src.utils.losses import SoftmaxGroups

class BaseTabModel(nn.Module):
    def __init__(self, num_strings, num_classes):
        super().__init__()
        self.num_strings = num_strings
        self.num_classes = num_classes
        self.softmax_groups = SoftmaxGroups(num_groups=self.num_strings, group_size=self.num_classes)

    def forward(self, x):
        raise NotImplementedError("Each model must implement its own forward method!")
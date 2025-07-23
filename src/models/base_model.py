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
import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class TabCNN(BaseTabModel):
    def __init__(self, in_channels: int, num_freq: int, num_strings: int, num_classes: int, config: dict, **kwargs):
        super().__init__(num_strings, num_classes)
        
        self.config = config
        self.model_params = self.config['model']['params']
        self.active_loss = self.config['loss']['active_loss']
        self.predict_onsets = self.model_params.get('predict_onsets', False)

        logger.info("Initializing TabCNN model...")
        logger.info(f"  -> Active loss function for output shaping: '{self.active_loss}'")
        logger.info(f"  -> Onset prediction enabled: {self.predict_onsets}")

        if self.active_loss == 'softmax_groups':
            self.tab_output_size = self.num_strings * self.num_classes
        elif self.active_loss == 'logistic_bank':
            self.tab_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported loss: {self.active_loss}")

        self.onset_output_size = self.num_strings * (self.num_classes - 1)
        
        logger.debug(f"Calculated tablature head output size: {self.tab_output_size}")
        if self.predict_onsets:
            logger.debug(f"Calculated onset head output size: {self.onset_output_size}")

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
        
        conv_out_freq = num_freq // 2
        window_width = config['data'].get('framify_window_size', 9)
        conv_out_time = window_width // 2
        embedding_dim = 64 * conv_out_freq * conv_out_time
        logger.debug(f"Calculated embedding dimension for FC layers: {embedding_dim}")

        self.tablature_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.tab_output_size)
        )

        if self.predict_onsets:
            self.onset_head = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.onset_output_size)
            )

    def forward(self, x: torch.Tensor):
        logger.debug(f"[TabCNN] --- forward pass start ---")
        logger.debug(f"[TabCNN] Input: {describe(x)}")
        
        embedding = self.conv(x)
        logger.debug(f"[TabCNN] After CNN block: {describe(embedding)}")
        
        embedding = embedding.reshape(embedding.size(0), -1)
        logger.debug(f"[TabCNN] After reshape (flattened): {describe(embedding)}")
        
        tab_logits = self.tablature_head(embedding)
        logger.debug(f"[TabCNN] After tablature_head (raw logits): {describe(tab_logits)}")
        
        if self.predict_onsets:
            onset_logits = self.onset_head(embedding)
            logger.debug(f"[TabCNN] After onset_head (raw logits): {describe(onset_logits)}")
            
            final_output = {
                "tablature": tab_logits,
                "onsets": onset_logits
            }
            logger.debug(f"[TabCNN] Returning final output: {describe(final_output)}")
            return final_output
        else:
            logger.debug(f"[TabCNN] Returning final output: {describe(tab_logits)}")
            return tab_logits
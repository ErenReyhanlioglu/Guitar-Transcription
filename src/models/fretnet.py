import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class FretNet(BaseTabModel):
    def __init__(self, in_channels: int, num_freq: int, num_strings: int, num_classes: int, config: dict, **kwargs):
        super().__init__(num_strings, num_classes)
        
        self.config = config
        self.model_params = self.config['model']['params'] 
        self.active_loss = self.config['loss']['active_loss']
        self.predict_onsets = self.model_params.get('predict_onsets', False)
        
        logger.info("Initializing FretNet model...")
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
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.25)
        )

        conv_out_freq = num_freq // 2 
        rnn_input_size = 64 * conv_out_freq
        rnn_hidden_size = self.model_params.get('rnn_hidden_size', 128)
        
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=self.model_params.get('rnn_num_layers', 2),
            batch_first=True,
            bidirectional=True 
        )

        self.tablature_head = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.tab_output_size)
        )
        
        if self.predict_onsets:
            self.onset_head = nn.Sequential(
                nn.Linear(rnn_hidden_size * 2, 128), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.onset_output_size) 
            )

    def forward(self, x: torch.Tensor):
        logger.debug(f"[FretNet] --- forward pass start ---")
        logger.debug(f"[FretNet] Input: {describe(x)}")
        B, C, F, T = x.shape
        
        conv_out = self.conv(x)
        logger.debug(f"[FretNet] After CNN block: {describe(conv_out)}")
        
        # Reshape for RNN: (B, C_out, F_out, T) -> (B, T, C_out * F_out)
        rnn_in = conv_out.permute(0, 3, 1, 2).reshape(B, T, -1)
        logger.debug(f"[FretNet] Reshaped for RNN: {describe(rnn_in)}")
        
        rnn_out, _ = self.rnn(rnn_in)
        logger.debug(f"[FretNet] After RNN block: {describe(rnn_out)}")
        
        tab_logits = self.tablature_head(rnn_out)
        logger.debug(f"[FretNet] After tablature_head (raw logits): {describe(tab_logits)}")
        
        if self.predict_onsets:
            onset_logits = self.onset_head(rnn_out)
            logger.debug(f"[FretNet] After onset_head (raw logits): {describe(onset_logits)}")
            
            final_output = {
                "tablature": tab_logits.view(B, T, self.num_strings, -1),
                "onsets": onset_logits.view(B, T, self.num_strings, -1)
            }
            logger.debug(f"[FretNet] Returning final output: {describe(final_output)}")
            logger.debug(f"  -> 'tablature' tensor: {describe(final_output['tablature'])}")
            logger.debug(f"  -> 'onsets' tensor: {describe(final_output['onsets'])}")
            return final_output
        else:
            final_output = tab_logits.view(B, T, self.num_strings, -1)
            logger.debug(f"[FretNet] Returning final output: {describe(final_output)}")
            return final_output
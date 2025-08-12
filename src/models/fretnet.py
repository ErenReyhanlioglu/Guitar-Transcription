import torch
import torch.nn as nn
from .base_model import BaseTabModel

class FretNet(BaseTabModel):
    """
    A recurrent model for guitar tablature transcription, combining a CNN front-end
    with a bidirectional LSTM to capture temporal context.

    This version is enhanced with an optional multi-task learning head to predict
    note onsets simultaneously with tablature, which can improve temporal accuracy.
    """
    def __init__(self, in_channels: int, num_freq: int, num_strings: int, num_classes: int, config: dict, **kwargs):
        """
        Initializes the FretNet model architecture.
        """
        super().__init__(num_strings, num_classes)
        
        self.config = config
        # --- DÜZELTME BAŞLANGICI ---
        # Model parametrelerini doğru yoldan alıp bir değişkene atayalım
        self.model_params = self.config['model']['params'] 
        # --- DÜZELTME BİTİŞİ ---

        self.output_mode = self.config['loss']['type']
        
        # Artık doğru değişkeni kullanıyoruz
        self.predict_onsets = self.model_params.get('predict_onsets', False)
        
        if self.output_mode == 'softmax_groups':
            self.tab_output_size = self.num_strings * self.num_classes
        elif self.output_mode == 'logistic_bank':
            self.tab_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
            
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
        
        # Artık doğru değişkeni kullanıyoruz
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
                nn.Linear(128, self.tab_output_size)
            )

    def forward(self, x: torch.Tensor) -> dict | torch.Tensor:
        """
        Defines the forward pass for the FretNet model.

        Args:
            x (torch.Tensor): Input tensor with shape (Batch, Channels, Freq, Time).

        Returns:
            dict | torch.Tensor: If multi-task learning (predict_onsets) is enabled,
                                returns a dictionary {'tablature': tab_logits, 'onsets': onset_logits}.
                                Otherwise, returns a single tensor of tablature logits.
        """
        B, C, F, T = x.shape
        
        out = self.conv(x)
        out = out.permute(0, 3, 1, 2)
        out = out.reshape(B, T, -1)
        
        rnn_out, _ = self.rnn(out)
        
        tab_logits = self.tablature_head(rnn_out)
        
        if self.predict_onsets:
            onset_logits = self.onset_head(rnn_out)
            return {
                "tablature": tab_logits.view(B, T, self.num_strings, -1),
                "onsets": onset_logits.view(B, T, self.num_strings, -1)
            }
        else:
            return tab_logits.view(B, T, self.num_strings, -1)
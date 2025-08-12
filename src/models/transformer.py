import torch
import torch.nn as nn
import math
from .base_model import BaseTabModel

class PositionalEncoding(nn.Module):
    # Bu sınıf aynı kalacak, değişiklik yok.
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(BaseTabModel):
    def __init__(self, in_channels: int, num_freq: int, num_strings: int, num_classes: int, config: dict, **kwargs):
        super().__init__(num_strings, num_classes)
        
        self.config = config
        self.output_mode = self.config['loss']['type']
        
        if self.output_mode == 'softmax_groups':
            self.total_output_size = self.num_strings * self.num_classes
        elif self.output_mode == 'logistic_bank':
            self.total_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        d_model = kwargs.get('d_model', 256)
        n_head = kwargs.get('n_head', 8)
        num_encoder_layers = kwargs.get('num_encoder_layers', 6)
        dim_feedforward = kwargs.get('dim_feedforward', 1024)
        dropout = kwargs.get('dropout', 0.1)
        
        self.d_model = d_model
        self.input_proj = nn.Linear(in_channels * num_freq, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.fc_out = nn.Linear(d_model, self.total_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, -1)
        
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        out = self.transformer_encoder(x)
        
        logits = self.fc_out(out)
        
        return logits.view(B, T, self.num_strings, -1)
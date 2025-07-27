import torch
import torch.nn as nn
import math
from .base_model import BaseTabModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Konumsal kodlamayı (max_len, d_model) boyutunda oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Giriş x boyutu: (B, T, D) -> (Batch, Time, Dim)
        """
        # self.pe'nin boyutu (1, max_len, D), x'in boyutu (B, T, D)
        # Broadcasting sayesinde pe[: , :x.size(1), :] işlemi (1, T, D) boyutlu bir tensör verir
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer(BaseTabModel):
    def __init__(self, in_channels, num_freq, num_strings, num_classes, 
                 d_model=256, n_head=8, num_encoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, **kwargs):
        
        super().__init__(num_strings, num_classes)
        self.total_output_size = self.num_strings * self.num_classes

        self.input_proj = nn.Linear(in_channels * num_freq, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.d_model = d_model

        self.fc_out = nn.Linear(d_model, self.total_output_size)

    def forward(self, x, apply_softmax=False):
        # x boyutu: (B, C, F, T) -> (Batch, Channels, Freq, Time)
        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2) # -> (B, T, C, F)
        x = x.reshape(B, T, -1)   # -> (B, T, C*F)
        
        x = self.input_proj(x) * math.sqrt(self.d_model) # -> (B, T, D)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # Basit bir geçici çözüm
        
        out = self.transformer_encoder(x) # -> (B, T, D)
        
        logits = self.fc_out(out) # -> (B, T, num_strings * num_classes)
        
        logits = logits.view(B, T, self.num_strings, self.num_classes)
        
        if apply_softmax:
            probs = self.softmax_groups(logits.reshape(B * T, -1)) # SoftmaxGroups 2D 
            return probs.view(B, T, self.num_strings, self.num_classes)
            
        return logits
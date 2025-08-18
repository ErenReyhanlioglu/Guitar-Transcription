import torch
import torch.nn as nn
import math
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
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
        self.model_params = self.config['model']['params']
        self.active_loss = self.config['loss']['active_loss']
        self.predict_onsets = self.model_params.get('predict_onsets', False)
        
        logger.info("Initializing Transformer model...")
        logger.info(f"  -> Active loss function for output shaping: '{self.active_loss}'")
        logger.info(f"  -> Onset prediction enabled: {self.predict_onsets}")

        if self.active_loss == 'softmax_groups':
            self.tab_output_size = self.num_strings * self.num_classes
        elif self.active_loss == 'logistic_bank':
            self.tab_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported loss: {self.active_loss}")
        
        self.onset_output_size = self.num_strings * (self.num_classes - 1)

        d_model = self.model_params.get('d_model', 256)
        n_head = self.model_params.get('n_head', 8)
        num_encoder_layers = self.model_params.get('num_encoder_layers', 6)
        
        logger.debug(f"Model params: d_model={d_model}, n_head={n_head}, num_encoder_layers={num_encoder_layers}")
        logger.debug(f"Calculated tablature head output size: {self.tab_output_size}")
        
        self.d_model = d_model
        self.input_proj = nn.Linear(in_channels * num_freq, d_model)
        self.pos_encoder = PositionalEncoding(d_model, self.model_params.get('dropout', 0.1))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, self.model_params.get('dim_feedforward', 1024), 
            self.model_params.get('dropout', 0.1), batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.tablature_head = nn.Linear(d_model, self.tab_output_size)
        
        if self.predict_onsets:
            logger.debug(f"Calculated onset head output size: {self.onset_output_size}")
            self.onset_head = nn.Linear(d_model, self.onset_output_size)

    def forward(self, x: torch.Tensor):
        logger.debug(f"[Transformer] --- forward pass start ---")
        logger.debug(f"[Transformer] Input: {describe(x)}")
        B, C, F, T = x.shape
        
        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)
        logger.debug(f"[Transformer] Reshaped for projection: {describe(x)}")
        
        x = self.input_proj(x) * math.sqrt(self.d_model)
        logger.debug(f"[Transformer] After input projection: {describe(x)}")
        
        x = self.pos_encoder(x)
        logger.debug(f"[Transformer] After positional encoding: {describe(x)}")
        
        out = self.transformer_encoder(x)
        logger.debug(f"[Transformer] After TransformerEncoder block: {describe(out)}")
        
        tab_logits = self.tablature_head(out)
        logger.debug(f"[Transformer] After tablature_head (raw logits): {describe(tab_logits)}")
        
        if self.predict_onsets:
            onset_logits = self.onset_head(out)
            logger.debug(f"[Transformer] After onset_head (raw logits): {describe(onset_logits)}")
            
            final_output = {
                "tablature": tab_logits.view(B, T, self.num_strings, -1),
                "onsets": onset_logits.view(B, T, self.num_strings, -1)
            }
            logger.debug(f"[Transformer] Returning final output: {describe(final_output)}")
            logger.debug(f"  -> 'tablature' tensor: {describe(final_output['tablature'])}")
            logger.debug(f"  -> 'onsets' tensor: {describe(final_output['onsets'])}")
            return final_output
        else:
            final_output = tab_logits.view(B, T, self.num_strings, -1)
            logger.debug(f"[Transformer] Returning final output: {describe(final_output)}")
            return final_output
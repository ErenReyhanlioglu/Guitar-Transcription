# src/models/transformer.py
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
    def __init__(self, config: dict, **kwargs):
        num_strings = config['instrument']['num_strings']
        num_classes_per_string = config['model']['params']['num_classes']
        super().__init__(num_strings, num_classes_per_string)
        
        self.config = config
        self.model_params = self.config['model']['params']
        self.loss_config = self.config['loss']
        self.instrument_config = self.config['instrument']

        self.active_loss_strategy = self.loss_config['active_loss']
        self.use_auxiliary_head = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        self.active_features = self.config['data']['active_features']
        self.feature_definitions = self.config['feature_definitions']
        
        logger.info("Initializing DYNAMIC Transformer model...")
        logger.info(f" -> Active loss strategy: '{self.active_loss_strategy}'")
        logger.info(f" -> Using Auxiliary Head: {self.use_auxiliary_head}")

        self.cnn_branches = nn.ModuleDict()
        for feature_key in self.active_features:
            feature_def = self.feature_definitions[feature_key]
            in_channels = feature_def['in_channels']
            groups = in_channels if feature_key == 'hcqt' and in_channels > 1 else 1
            branch = nn.Sequential(
                nn.Conv2d(in_channels, 36, kernel_size=3, padding=1, groups=groups),
                nn.BatchNorm2d(36), nn.ReLU(),
                nn.Conv2d(36, 36, kernel_size=3, padding=1),
                nn.BatchNorm2d(36), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1))
            )
            self.cnn_branches[feature_key] = branch

        with torch.no_grad():
            total_cnn_output_size = 0
            dummy_time_dim = 200
            for feature_key in self.active_features:
                feature_def = self.feature_definitions[feature_key]
                dummy_input = torch.randn(1, feature_def['in_channels'], feature_def['num_freq'], dummy_time_dim)
                dummy_output = self.cnn_branches[feature_key](dummy_input)
                flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
                total_cnn_output_size += flattened_size
        
        d_model = self.model_params.get('d_model', 256)
        self.d_model = d_model
        self.input_proj = nn.Linear(total_cnn_output_size, d_model)
        
        self.backbone = nn.ModuleDict({
            "cnn_branches": self.cnn_branches,
            "input_proj": self.input_proj
        })

        n_head = self.model_params.get('n_head', 8)
        num_encoder_layers = self.model_params.get('num_encoder_layers', 6)
        
        self.pos_encoder = PositionalEncoding(d_model, self.model_params.get('dropout', 0.1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, self.model_params.get('dim_feedforward', 1024), 
            self.model_params.get('dropout', 0.1), batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        if self.active_loss_strategy == 'softmax_groups':
            tab_output_size = self.num_strings * self.num_classes
        else: # logistic_bank
            tab_output_size = self.num_strings * (self.num_classes - 1)
        self.tablature_head = nn.Linear(d_model, tab_output_size)
        
        self.auxiliary_head = None
        if self.use_auxiliary_head:
            num_pitches = self.instrument_config['max_midi'] - self.instrument_config['min_midi'] + 1
            self.auxiliary_head = nn.Linear(d_model, num_pitches)

        self.head = nn.ModuleDict({
            'pos_encoder': self.pos_encoder,
            'transformer_encoder': self.transformer_encoder,
            'tablature_head': self.tablature_head
        })
        if self.auxiliary_head:
            self.head.add_module('auxiliary_head', self.auxiliary_head)

    def forward(self, *args, **kwargs):
        features_dict = kwargs or (args[0] if args else {})
        if not features_dict:
            raise ValueError("Transformer forward pass received no input")
        
        B, T = next(iter(features_dict.values())).shape[0], next(iter(features_dict.values())).shape[3]

        branch_outputs = []
        for feature_key, input_tensor in features_dict.items():
            conv_out = self.cnn_branches[feature_key](input_tensor)
            reshaped_out = conv_out.permute(0, 3, 1, 2).reshape(B, T, -1)
            branch_outputs.append(reshaped_out)
            
        x = torch.cat(branch_outputs, dim=2)
        
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        
        tab_logits = self.tablature_head(transformer_out)
        
        if self.active_loss_strategy == 'softmax_groups':
            final_tab_output = tab_logits.view(B, T, self.num_strings, self.num_classes)
        else: # logistic_bank
            final_tab_output = tab_logits.view(B, T, self.num_strings, self.num_classes - 1)
            
        if self.use_auxiliary_head:
            multipitch_logits = self.auxiliary_head(transformer_out)
            return final_tab_output, multipitch_logits
        else:
            return final_tab_output
import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class FretNet(BaseTabModel):
    """
    A CRNN (CNN-RNN) based model for sequence-to-sequence tablature transcription.
    
    It uses dynamic CNN branches for feature extraction, followed by an LSTM layer
    to capture temporal dependencies. Multiple heads predict tablature, multi-pitch,
    onsets, and offsets from the shared RNN features.
    """
    def __init__(self, config: dict, **kwargs):
        """Initializes the FretNet model architecture."""
        num_strings = config['instrument']['num_strings']
        num_classes_per_string = config['model']['params']['num_classes']
        super().__init__(num_strings, num_classes_per_string)
        
        self.config = config
        self.model_params = self.config['model']['params']
        self.loss_config = self.config['loss']
        self.instrument_config = self.config['instrument']
        
        self.active_loss_strategy = self.loss_config['active_loss']
        self.use_projection = self.model_params.get('use_projection_layer', False)
        
        # --- DEĞİŞİKLİK BAŞLANGICI ---
        self.use_auxiliary_head = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        self.use_onset_head = self.loss_config.get('onset_loss', {}).get('enabled', False)
        self.use_offset_head = self.loss_config.get('offset_loss', {}).get('enabled', False)
        # --- DEĞİŞİKLİK SONU ---

        self.active_features = self.config['data']['active_features']
        self.feature_definitions = self.config['feature_definitions']
        
        logger.info("Initializing DYNAMIC FretNet model...")
        logger.info(f" -> Active loss strategy: '{self.active_loss_strategy}'")
        logger.info(f" -> Using Projection Layers: {self.use_projection}")
        # --- DEĞİŞİKLİK BAŞLANGICI ---
        logger.info(f" -> Using Auxiliary Head: {self.use_auxiliary_head}")
        logger.info(f" -> Using Onset Head: {self.use_onset_head}")
        logger.info(f" -> Using Offset Head: {self.use_offset_head}")
        # --- DEĞİŞİKLİK SONU ---
        logger.info(f" -> Model will be built for features: {self.active_features}")

        self.cnn_branches = nn.ModuleDict()
        for feature_key in self.active_features:
            feature_def = self.feature_definitions[feature_key]
            in_channels = feature_def['in_channels']
            groups = in_channels if feature_key == 'hcqt' and in_channels > 1 else 1
            branch = nn.Sequential(
                nn.Conv2d(in_channels, 36, kernel_size=3, padding=1, groups=groups), nn.BatchNorm2d(36), nn.ReLU(),
                nn.Conv2d(36, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)), nn.Dropout(0.25)
            )
            self.cnn_branches[feature_key] = branch

        self.projection_layers = nn.ModuleDict() if self.use_projection else None
        total_rnn_input_size = 0
        projection_output_size = self.model_params.get('projection_output_size', 256)

        with torch.no_grad():
            dummy_time_dim = 200 
            for feature_key in self.active_features:
                feature_def = self.feature_definitions[feature_key]
                dummy_input = torch.randn(1, feature_def['in_channels'], feature_def['num_freq'], dummy_time_dim)
                dummy_output = self.cnn_branches[feature_key](dummy_input)
                flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
                if self.use_projection:
                    self.projection_layers[feature_key] = nn.Linear(flattened_size, projection_output_size)
                    total_rnn_input_size += projection_output_size
                else:
                    total_rnn_input_size += flattened_size
        
        self.backbone = self.cnn_branches
        if self.use_projection:
            self.backbone.add_module("projection_layers", self.projection_layers)

        rnn_hidden_size = self.model_params.get('rnn_hidden_size', 128)
        self.rnn = nn.LSTM(
            input_size=total_rnn_input_size, hidden_size=rnn_hidden_size,
            num_layers=self.model_params.get('rnn_num_layers', 2),
            batch_first=True, bidirectional=True 
        )

        head_input_size = rnn_hidden_size * 2 
        
        if self.active_loss_strategy == 'softmax_groups':
            tab_output_size = self.num_strings * self.num_classes
        else: # logistic_bank
            tab_output_size = self.num_strings * (self.num_classes - 1)
        
        self.tablature_head = nn.Sequential(
            nn.Linear(head_input_size, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, tab_output_size)
        )
        
        self.auxiliary_head = None
        if self.use_auxiliary_head:
            num_pitches = self.instrument_config['max_midi'] - self.instrument_config['min_midi'] + 1
            self.auxiliary_head = nn.Sequential(
                nn.Linear(head_input_size, 128), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(128, num_pitches)
            )

        # --- DEĞİŞİKLİK BAŞLANGICI ---
        self.onset_head = None
        if self.use_onset_head:
            self.onset_head = nn.Sequential(
                nn.Linear(head_input_size, 64), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(64, self.num_strings)
            )
            logger.info(f" -> Onset head created for {self.num_strings} strings.")

        self.offset_head = None
        if self.use_offset_head:
            self.offset_head = nn.Sequential(
                nn.Linear(head_input_size, 64), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(64, self.num_strings)
            )
            logger.info(f" -> Offset head created for {self.num_strings} strings.")

        self.head = nn.ModuleDict({
            'rnn': self.rnn,
            'tablature_head': self.tablature_head
        })
        if self.auxiliary_head: self.head.add_module('auxiliary_head', self.auxiliary_head)
        if self.onset_head: self.head.add_module('onset_head', self.onset_head)
        if self.offset_head: self.head.add_module('offset_head', self.offset_head)
        # --- DEĞİŞİKLİK SONU ---

    def forward(self, *args, **kwargs):
        """Performs a forward pass through the FretNet model."""
        features_dict = kwargs or (args[0] if args else {})
        if not features_dict:
            raise ValueError("Model forward pass received no input")

        batch_size = next(iter(features_dict.values())).shape[0]
        time_steps = next(iter(features_dict.values())).shape[3]

        branch_outputs = []
        for feature_key, input_tensor in features_dict.items():
            conv_out = self.cnn_branches[feature_key](input_tensor)
            flattened_out = conv_out.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)
            if self.use_projection:
                branch_outputs.append(self.projection_layers[feature_key](flattened_out))
            else:
                branch_outputs.append(flattened_out)

        rnn_in = torch.cat(branch_outputs, dim=2)
        rnn_out, _ = self.rnn(rnn_in)
        
        # --- DEĞİŞİKLİK BAŞLANGICI: Çıktıları sözlük olarak döndür ---
        outputs = {}
        tab_logits = self.tablature_head(rnn_out)
        if self.active_loss_strategy == 'softmax_groups':
            outputs['tab_logits'] = tab_logits.view(batch_size, time_steps, self.num_strings, self.num_classes)
        else: # logistic_bank
            outputs['tab_logits'] = tab_logits.view(batch_size, time_steps, self.num_strings, self.num_classes - 1)
        
        if self.use_auxiliary_head:
            outputs['multipitch_logits'] = self.auxiliary_head(rnn_out)
        if self.use_onset_head:
            outputs['onset_logits'] = self.onset_head(rnn_out)
        if self.use_offset_head:
            outputs['offset_logits'] = self.offset_head(rnn_out)
        
        log_str = ", ".join([f"{k}={v.shape}" for k, v in outputs.items()])
        logger.debug(f"[FretNet] Output shapes: {log_str}")
        
        return outputs
        # --- DEĞİŞİKLİK SONU ---
import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class TabCNN(BaseTabModel):
    def __init__(self, config: dict, **kwargs):
        num_strings = config['instrument']['num_strings']
        num_classes_per_string = config['model']['params']['num_classes']
        super().__init__(num_strings, num_classes_per_string)
        
        self.config = config
        self.model_params = self.config['model']['params']
        self.loss_config = self.config['loss']
        self.instrument_config = self.config['instrument']

        self.active_loss_strategy = self.loss_config['active_loss']
        self.use_projection = self.model_params.get('use_projection_layer', False)
        self.use_auxiliary_head = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        self.active_features = self.config['data']['active_features']
        self.feature_definitions = self.config['feature_definitions']

        logger.info("Initializing DYNAMIC TabCNN model...")
        logger.info(f" -> Active loss strategy: '{self.active_loss_strategy}'")
        logger.info(f" -> Using Projection Layers: {self.use_projection}")
        logger.info(f" -> Using Auxiliary Head: {self.use_auxiliary_head}")
        logger.info(f" -> Model will be built for features: {self.active_features}")
        
        self.cnn_branches = nn.ModuleDict()
        for feature_key in self.active_features:
            feature_def = self.feature_definitions[feature_key]
            in_channels = feature_def['in_channels']
            groups = in_channels if feature_key == 'hcqt' and in_channels > 1 else 1
            
            branch = nn.Sequential(
                nn.Conv2d(in_channels, 36, kernel_size=3, padding=1, groups=groups),
                nn.BatchNorm2d(36), nn.ReLU(),
                nn.Conv2d(36, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(0.25)
            )
            self.cnn_branches[feature_key] = branch
        
        self.projection_layers = nn.ModuleDict() if self.use_projection else None
        total_head_input_size = 0
        projection_output_size = self.model_params.get('projection_output_size', 256)
        window_width = self.config['data'].get('framify_window_size', 9)

        # Geçici bir girdi ile CNN çıktısının boyutunu dinamik olarak hesapla
        with torch.no_grad():
            for feature_key in self.active_features:
                feature_def = self.feature_definitions[feature_key]
                dummy_input = torch.randn(1, feature_def['in_channels'], feature_def['num_freq'], window_width)
                dummy_output = self.cnn_branches[feature_key](dummy_input)
                flattened_size = dummy_output.flatten(1).shape[1]

                if self.use_projection:
                    self.projection_layers[feature_key] = nn.Linear(flattened_size, projection_output_size)
                    total_head_input_size += projection_output_size
                    logger.info(f" -> '{feature_key}' branch: CNN out({flattened_size}) -> Projection out({projection_output_size})")
                else:
                    total_head_input_size += flattened_size
                    logger.info(f" -> '{feature_key}' branch: CNN out({flattened_size})")
        
        logger.info(f" -> Total dynamic input size for head: {total_head_input_size}")

        # --- FretNet ile tutarlılık için backbone ve head tanımlaması ---
        self.backbone = self.cnn_branches
        if self.use_projection:
            self.backbone.add_module("projection_layers", self.projection_layers)

        if self.active_loss_strategy == 'softmax_groups':
            tab_output_size = self.num_strings * self.num_classes
        else: # logistic_bank
            tab_output_size = self.num_strings * (self.num_classes - 1)

        self.tablature_head = nn.Sequential(
            nn.Linear(total_head_input_size, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, tab_output_size)
        )
        
        self.auxiliary_head = None
        if self.use_auxiliary_head:
            num_pitches = self.instrument_config['max_midi'] - self.instrument_config['min_midi'] + 1
            self.auxiliary_head = nn.Sequential(
                nn.Linear(total_head_input_size, 128), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(128, num_pitches)
            )
            logger.info(f" -> Auxiliary head created for {num_pitches} pitches.")

        self.head = nn.ModuleDict({'tablature_head': self.tablature_head})
        if self.auxiliary_head:
            self.head.add_module('auxiliary_head', self.auxiliary_head)

    def forward(self, *args, **kwargs):
        features_dict = kwargs or (args[0] if args else {})
        if not features_dict:
            raise ValueError("TabCNN forward pass received no input")

        logger.debug(f"[TabCNN] Input shapes: " + ", ".join([f"{k}: {v.shape}" for k, v in features_dict.items()]))

        branch_outputs = []
        for feature_key, input_tensor in features_dict.items():
            embedding = self.backbone[feature_key](input_tensor)
            flattened_embedding = embedding.reshape(embedding.size(0), -1)
            
            if self.use_projection:
                projected_embedding = self.backbone.projection_layers[feature_key](flattened_embedding)
                branch_outputs.append(projected_embedding)
            else:
                branch_outputs.append(flattened_embedding)

        combined_embedding = torch.cat(branch_outputs, dim=1)
        
        tab_logits = self.head.tablature_head(combined_embedding)
        
        if self.use_auxiliary_head:
            multipitch_logits = self.head.auxiliary_head(combined_embedding)
            logger.debug(f"[TabCNN] Output shapes: tab_logits={tab_logits.shape}, multipitch_logits={multipitch_logits.shape}")
            return tab_logits, multipitch_logits
        else:
            logger.debug(f"[TabCNN] Output shape: tab_logits={tab_logits.shape}")
            return tab_logits
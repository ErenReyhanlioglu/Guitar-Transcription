# src/models/tabcnn.py

import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class TabCNN(BaseTabModel):
    def __init__(self, config: dict, **kwargs):
        num_strings = config['instrument']['num_strings']
        num_classes = config['model']['params']['num_classes']
        super().__init__(num_strings, num_classes)
        
        self.config = config
        self.model_params = self.config['model']['params']
        self.active_loss = self.config['loss']['active_loss']
        self.predict_onsets = self.model_params.get('predict_onsets', False)

        logger.info("Initializing DYNAMIC TabCNN model...")
        
        self.active_features = self.config['data']['active_features']
        self.feature_definitions = self.config['feature_definitions']
        logger.info(f" -> Model will be built for features: {self.active_features}")
        
        self.cnn_branches = nn.ModuleDict()
        for feature_key in self.active_features:
            feature_def = self.feature_definitions[feature_key]
            in_channels = feature_def['in_channels']
            groups = in_channels if feature_key == 'hcqt' and in_channels > 1 else 1
            
            branch = nn.Sequential(
                nn.Conv2d(in_channels, 36, kernel_size=3, padding=1, groups=groups),
                nn.BatchNorm2d(36),
                nn.ReLU(),
                nn.Conv2d(36, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(0.25)
            )
            self.cnn_branches[feature_key] = branch
            logger.info(f"  -> Created CNN branch for '{feature_key}' with in_channels={in_channels}")
            
        with torch.no_grad():
            total_embedding_dim = 0
            window_width = self.config['data'].get('framify_window_size', 9)

            for feature_key in self.active_features:
                feature_def = self.feature_definitions[feature_key]
                dummy_input = torch.randn(1, feature_def['in_channels'], feature_def['num_freq'], window_width)
                dummy_output = self.cnn_branches[feature_key](dummy_input)
                flattened_size = dummy_output.flatten(1).shape[1]
                logger.info(f"  -> Calculated flattened output size for '{feature_key}': {flattened_size}")
                total_embedding_dim += flattened_size
        
        logger.info(f" -> Total dynamic embedding dimension: {total_embedding_dim}")

        if self.active_loss == 'softmax_groups':
            self.tab_output_size = self.num_strings * self.num_classes
        elif self.active_loss == 'logistic_bank':
            self.tab_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported loss: {self.active_loss}")

        self.tablature_head = nn.Sequential(
            nn.Linear(total_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.tab_output_size)
        )

        if self.predict_onsets:
            self.onset_output_size = self.num_strings * (self.num_classes - 1)
            self.onset_head = nn.Sequential(
                nn.Linear(total_embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.onset_output_size)
            )

    def forward(self, *args, **kwargs):
        if kwargs:
            features_dict = kwargs
        elif args and isinstance(args[0], dict):
            features_dict = args[0]
        else:
            raise ValueError("TabCNN forward pass received unexpected input format")

        branch_outputs = []
        for feature_key, input_tensor in features_dict.items():
            logger.debug(f"[TabCNN] Processing feature '{feature_key}': {describe(input_tensor)}")
            embedding = self.cnn_branches[feature_key](input_tensor)
            branch_outputs.append(embedding.reshape(embedding.size(0), -1))

        combined_embedding = torch.cat(branch_outputs, dim=1)
        logger.debug(f"[TabCNN] Combined & flattened embedding: {describe(combined_embedding)}")
        
        tab_logits = self.tablature_head(combined_embedding)
        logger.debug(f"[TabCNN] After tablature_head (raw logits): {describe(tab_logits)}")
        
        if self.predict_onsets:
            onset_logits = self.onset_head(combined_embedding)
            return {"tablature": tab_logits, "onsets": onset_logits}
        else:
            return tab_logits
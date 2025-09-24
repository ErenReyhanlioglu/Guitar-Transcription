import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class TabCNN(BaseTabModel):
    """
    A CNN-based model for frame-wise tablature transcription.

    This model uses separate CNN branches to process different input features 
    (e.g., HCQT, Mel spectrograms). The outputs of these branches are concatenated, 
    optionally projected to a common dimension, and then fed into multiple
    prediction heads for tablature, multi-pitch, onsets, and offsets.
    """
    def __init__(self, config: dict, **kwargs):
        """
        Initializes the TabCNN model architecture based on the provided config.

        Args:
            config (dict): The experiment configuration dictionary.
        """
        num_strings = config['instrument']['num_strings']
        num_classes_per_string = config['model']['params']['num_classes']
        super().__init__(num_strings, num_classes_per_string)
         
        self.config = config
        self.model_params = self.config['model']['params']
        self.loss_config = self.config['loss']
        self.instrument_config = self.config['instrument']

        self.active_loss_strategy = self.loss_config['active_loss']
        self.use_projection = self.model_params.get('use_projection_layer', False)
         
        self.use_activity_head = self.loss_config.get('activity_loss', {}).get('enabled', False)
        self.use_auxiliary_head = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        self.use_onset_head = self.loss_config.get('onset_loss', {}).get('enabled', False)
        self.use_offset_head = self.loss_config.get('offset_loss', {}).get('enabled', False)

        self.active_features = self.config['data']['active_features']
        self.feature_definitions = self.config['feature_definitions']

        logger.info("Initializing DYNAMIC TabCNN model...")
        logger.info(f" -> Active loss strategy: '{self.active_loss_strategy}'")
        logger.info(f" -> Using Projection Layers: {self.use_projection}")
        logger.info(f" -> Using Auxiliary Head: {self.use_auxiliary_head}")
        logger.info(f" -> Using Activity Head: {self.use_activity_head}")
        logger.info(f" -> Using Onset Head: {self.use_onset_head}")
        logger.info(f" -> Using Offset Head: {self.use_offset_head}")
        logger.info(f" -> Model will be built for features: {self.active_features}")
         
        self.cnn_branches = nn.ModuleDict()
        for feature_key in self.active_features:
            feature_def = self.feature_definitions[feature_key]
            in_channels = feature_def['in_channels']
             
            if feature_key == 'power':
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16), nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16), nn.ReLU(),
                    nn.Dropout(0.25)
                )
            else: 
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

        with torch.no_grad():
            for feature_key in self.active_features:
                feature_def = self.feature_definitions[feature_key]
                dummy_input = torch.randn(1, feature_def['in_channels'], feature_def['num_freq'], window_width)
                dummy_output = self.cnn_branches[feature_key](dummy_input)
                flattened_size = dummy_output.flatten(1).shape[1]

                if self.use_projection and flattened_size > projection_output_size:
                    self.projection_layers[feature_key] = nn.Linear(flattened_size, projection_output_size)
                    total_head_input_size += projection_output_size
                    logger.info(f" -> '{feature_key}' branch: CNN out({flattened_size}) -> Projection out({projection_output_size})")
                else:
                    total_head_input_size += flattened_size
                    if self.use_projection: 
                        logger.info(f" -> '{feature_key}' branch: CNN out({flattened_size}) (Projection SKIPPED, size <= target)")
                    else:
                        logger.info(f" -> '{feature_key}' branch: CNN out({flattened_size})")

        logger.info(f" -> Total dynamic input size for all heads: {total_head_input_size}")

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

        self.activity_head = None
        if self.use_activity_head:
            self.activity_head = nn.Sequential(
                nn.Linear(total_head_input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 1) 
            )
            logger.info(f" -> Activity head created for binary activity detection.")

        self.onset_head = None
        if self.use_onset_head:
            self.onset_head = nn.Sequential(
                nn.Linear(total_head_input_size, 64), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(64, self.num_strings)
            )
            logger.info(f" -> Onset head created for {self.num_strings} strings.")

        self.offset_head = None
        if self.use_offset_head:
            self.offset_head = nn.Sequential(
                nn.Linear(total_head_input_size, 64), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(64, self.num_strings)
            )
            logger.info(f" -> Offset head created for {self.num_strings} strings.")

        self.head = nn.ModuleDict({'tablature_head': self.tablature_head})
        if self.auxiliary_head:
            self.head.add_module('auxiliary_head', self.auxiliary_head)
        if self.activity_head:
            self.head.add_module('activity_head', self.activity_head)
        if self.onset_head:
            self.head.add_module('onset_head', self.onset_head)
        if self.offset_head:
            self.head.add_module('offset_head', self.offset_head)

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass through the model.

        Args:
            features_dict (dict): A dictionary where keys are feature names 
                                  (e.g., 'hcqt') and values are input tensors.

        Returns:
            dict: A dictionary of output logits from all active heads.
                  Keys can include 'tab_logits', 'multipitch_logits', 
                  'onset_logits', 'offset_logits'.
        """
        features_dict = kwargs or (args[0] if args else {})
        if not features_dict:
            raise ValueError("TabCNN forward pass received no input")

        logger.debug(f"[TabCNN] Input shapes: " + ", ".join([f"{k}: {v.shape}" for k, v in features_dict.items()]))

        branch_outputs = []
        for feature_key, input_tensor in features_dict.items():
            embedding = self.backbone[feature_key](input_tensor)
            flattened_embedding = embedding.reshape(embedding.size(0), -1)

            if self.projection_layers and feature_key in self.projection_layers:
                projected_embedding = self.backbone.projection_layers[feature_key](flattened_embedding)
                branch_outputs.append(projected_embedding)
            else:
                branch_outputs.append(flattened_embedding)

        combined_embedding = torch.cat(branch_outputs, dim=1)
         
        outputs = {}
         
        outputs['tab_logits'] = self.head.tablature_head(combined_embedding)
         
        if self.use_auxiliary_head:
            outputs['multipitch_logits'] = self.head.auxiliary_head(combined_embedding)
         
        if self.use_activity_head:
            outputs['activity_logits'] = self.head.activity_head(combined_embedding)

        if self.use_onset_head:
            outputs['onset_logits'] = self.head.onset_head(combined_embedding)
             
        if self.use_offset_head:
            outputs['offset_logits'] = self.head.offset_head(combined_embedding)
         
        log_str = ", ".join([f"{k}={v.shape}" for k, v in outputs.items()])
        logger.debug(f"[TabCNN] Output shapes: {log_str}")
         
        return outputs
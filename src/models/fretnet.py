import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel
from src.utils.logger import describe

logger = logging.getLogger(__name__)

class FretNet(BaseTabModel):
    def __init__(self, config: dict, **kwargs):
        
        num_strings = config['instrument']['num_strings']
        num_classes = config['model']['params']['num_classes']
        
        super().__init__(num_strings, num_classes)
        
        self.config = config
        self.model_params = self.config['model']['params'] 
        self.active_loss = self.config['loss']['active_loss']
        self.predict_onsets = self.model_params.get('predict_onsets', False)
        
        logger.info("Initializing DYNAMIC FretNet model...")
        
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
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout(0.25)
            )
            self.cnn_branches[feature_key] = branch
            logger.info(f"  -> Created CNN branch for '{feature_key}' with in_channels={in_channels} and groups={groups}")

        with torch.no_grad():
            total_rnn_input_size = 0
            dummy_time_dim = 200 
            
            for feature_key in self.active_features:
                feature_def = self.feature_definitions[feature_key]
                dummy_input = torch.randn(1, feature_def['in_channels'], feature_def['num_freq'], dummy_time_dim)
                dummy_output = self.cnn_branches[feature_key](dummy_input)
                
                flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
                logger.info(f"  -> Calculated flattened output size for '{feature_key}': {flattened_size}")
                total_rnn_input_size += flattened_size
        
        logger.info(f" -> Total dynamic RNN input size: {total_rnn_input_size}")
        
        rnn_hidden_size = self.model_params.get('rnn_hidden_size', 128)
        self.rnn = nn.LSTM(
            input_size=total_rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=self.model_params.get('rnn_num_layers', 2),
            batch_first=True,
            bidirectional=True 
        )

        if self.active_loss == 'softmax_groups':
            self.tab_output_size = self.num_strings * self.num_classes
        elif self.active_loss == 'logistic_bank':
            self.tab_output_size = self.num_strings * (self.num_classes - 1)
        else:
            raise ValueError(f"Unsupported loss: {self.active_loss}")

        self.tablature_head = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.tab_output_size)
        )
        
        if self.predict_onsets:
            self.onset_output_size = self.num_strings * (self.num_classes - 1)
            self.onset_head = nn.Sequential(
                nn.Linear(rnn_hidden_size * 2, 128), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.onset_output_size) 
            )
    def forward(self, *args, **kwargs):
        if kwargs:
            features_dict = kwargs
        elif args:
            features_dict = args[0]
        else:
            raise ValueError("Model forward pass received no input")
        
        branch_outputs = []
        for feature_key, input_tensor in features_dict.items():
            logger.debug(f"[FretNet] Processing feature '{feature_key}': {describe(input_tensor)}")
            # Input tensor (B, C, H, T)
            B, C, F, T = input_tensor.shape

            conv_out = self.cnn_branches[feature_key](input_tensor)
            logger.debug(f"[FretNet]  -> After '{feature_key}' CNN branch: {describe(conv_out)}")
            
            # Reshape for RNN: (B, C_out, F_out, T) -> (B, T, C_out * F_out)
            rnn_in_branch = conv_out.permute(0, 3, 1, 2).reshape(B, T, -1)
            branch_outputs.append(rnn_in_branch)

        rnn_in = torch.cat(branch_outputs, dim=2)
        logger.debug(f"[FretNet] Combined & Reshaped for RNN: {describe(rnn_in)}")
        
        rnn_out, _ = self.rnn(rnn_in)
        logger.debug(f"[FretNet] After RNN block: {describe(rnn_out)}")
        
        tab_logits = self.tablature_head(rnn_out)
        logger.debug(f"[FretNet] After tablature_head (raw logits): {describe(tab_logits)}")
        
        if self.active_loss == 'softmax_groups':
            num_output_classes_per_string = self.num_classes
        else: # logistic_bank
            num_output_classes_per_string = self.num_classes - 1
            
        B, T, _ = tab_logits.shape 

        if self.predict_onsets:
            onset_logits = self.onset_head(rnn_out)
            logger.debug(f"[FretNet] After onset_head (raw logits): {describe(onset_logits)}")
            
            final_output = {
                "tablature": tab_logits.view(B, T, self.num_strings, num_output_classes_per_string),
                "onsets": onset_logits.view(B, T, self.num_strings, num_output_classes_per_string)
            }
            logger.debug(f" -> 'tablature' tensor: {describe(final_output['tablature'])}")
            logger.debug(f" -> 'onsets' tensor: {describe(final_output['onsets'])}")
            return final_output
        else:
            final_output = tab_logits.view(B, T, self.num_strings, num_output_classes_per_string)
            logger.debug(f"[FretNet] Returning final output: {describe(final_output)}")
            return final_output
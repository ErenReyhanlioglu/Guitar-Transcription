import torch
import torch.nn as nn
import logging
from .base_model import BaseTabModel

logger = logging.getLogger(__name__)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: recalibrates channel-wise feature responses.

    Squeezes each channel to a scalar via global average pooling, learns a
    channel-wise gating vector through a small bottleneck MLP, and rescales
    the input channels by it.
    """
    def __init__(self, channel, reduction=16):
        """Args:
            channel: number of input/output channels.
            reduction: bottleneck reduction factor for the gating MLP's hidden layer.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Applies channel-wise gating: pools x to (B, C), computes gates via the FC bottleneck,
        and scales x by them."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiScaleConvBlock(nn.Module):
    """Multi-Scale Convolutional Block: parallel conv branches at different kernel sizes, fused,
    optionally recalibrated by an SE block, then pooled and dropped out.

    Each branch is a two-layer Conv-BN-ReLU stack at one of the given kernel
    sizes (e.g. 3x3, 5x5, an asymmetric 1x9 for time-only context); branch
    outputs are concatenated along channels and reduced back to
    `out_channels` via a 1x1 conv + BN + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernels, use_se=False, dropout=0.25):
        """Args:
            in_channels: input channel count.
            out_channels: output channel count (per branch, and after fusion).
            kernels: list of (h, w) kernel sizes, one parallel branch per entry.
            use_se: whether to apply an SEBlock after fusion.
            dropout: dropout rate applied after pooling.
        """
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            p_h = (k[0] - 1) // 2
            p_w = (k[1] - 1) // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(p_h, p_w)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=(p_h, p_w)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.branches.append(branch)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(kernels), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Runs all kernel branches on x, concatenates and fuses them, optionally applies SE
        recalibration, then 2x2 max-pools and applies dropout."""
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.fusion(out)
        if self.use_se:
            out = self.se(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out

class cnn_mtl(BaseTabModel):
    """The cascaded multi-task guitar transcription model.

    Encodes each active input representation (e.g. HCQT, Mel) through its own
    convolutional branch, concatenates them into a shared bottleneck, runs the
    enabled auxiliary heads (multipitch, fret region, string activity, pitch
    class, onset) on the bottleneck, and feeds their soft (sigmoid/softmax)
    outputs alongside the bottleneck into the tablature head.
    """
    def __init__(self, config: dict, **kwargs):
        """Builds the per-feature backbone branches, optional projection layers, the enabled
        auxiliary heads, and the tablature head sized to take the bottleneck plus every enabled
        auxiliary head's output.

        Args:
            config: Full experiment config (uses `model.params`, `data`, `feature_definitions`, `instrument`).
        """
        super().__init__(config)

        model_params = config['model']['params']
        backbone_conf = model_params['backbone']

        # Safe access to the heads config
        heads_conf = model_params.get('heads', config['model'].get('heads', {}))
        
        self.use_projection = model_params.get('use_projection_layer', False)
        self.projection_dim = model_params.get('projection_output_size', 256)
        
        # 1. INPUT & FEATURE DEFINITIONS
        data_conf = config['data']
        if isinstance(data_conf['active_features'], dict):
             self.active_features = [k for k, v in data_conf['active_features'].items() if v]
        else:
             self.active_features = data_conf['active_features']

        self.feature_definitions = config['feature_definitions']
        
        logger.info(f"Building CNN_MTL Model (Cascaded Architecture)...")
        
        # 2. BUILD BACKBONE
        self.feature_branches = nn.ModuleDict()
        for feat_key in self.active_features:
            feat_def = self.feature_definitions[feat_key]
            in_ch = feat_def['in_channels']
            layers = []
            current_ch = in_ch
            filters = backbone_conf['filters']
            for out_ch in filters:
                block = MultiScaleConvBlock(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    kernels=backbone_conf['kernels'],
                    use_se=backbone_conf['use_se_block'],
                    dropout=backbone_conf['dropout']
                )
                layers.append(block)
                current_ch = out_ch
            self.feature_branches[feat_key] = nn.Sequential(*layers)
                
        # 3. PROJECTION / BOTTLENECK CALCULATION
        if self.use_projection:
            self.projections = nn.ModuleDict()
            for feat_key in self.active_features:
                self.projections[feat_key] = nn.LazyLinear(self.projection_dim)
            self.bottleneck_dim = self.projection_dim * len(self.active_features)
        else:
            total_flat_dim = 0
            window_size = config['data'].get('framify_window_size', 9)
            with torch.no_grad():
                for feat_key in self.active_features:
                    feat_def = self.feature_definitions[feat_key]
                    dummy = torch.randn(1, feat_def['in_channels'], feat_def['num_freq'], window_size)
                    out = self.feature_branches[feat_key](dummy)
                    total_flat_dim += out.flatten(1).shape[1]
            self.bottleneck_dim = total_flat_dim

        # 4. HEADS DEFINITION
        self.heads = nn.ModuleDict()
        self.aux_input_dim = 0 # Extra dimension to be added to the tablature head

        # --- Aux Heads Definition First (To calculate dimension) ---
        
        # Multipitch
        if heads_conf.get('multipitch', {}).get('enabled', False):
            num_pitches = config['instrument']['max_midi'] - config['instrument']['min_midi'] + 1
            self.heads['multipitch'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, num_pitches)
            )
            self.aux_input_dim += num_pitches # Sigmoid Outputs

        # Fret region (stored as 'hand_position')
        if heads_conf.get('hand_position', {}).get('enabled', False):
            hp_classes = heads_conf['hand_position']['num_classes']
            self.heads['hand_position'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, hp_classes)
            )
            self.aux_input_dim += hp_classes # Softmax Probabilities

        # String Activity
        if heads_conf.get('string_activity', {}).get('enabled', False):
            self.heads['string_activity'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, self.num_strings) 
            )
            self.aux_input_dim += self.num_strings # Sigmoid Outputs

        # Pitch Class
        if heads_conf.get('pitch_class', {}).get('enabled', False):
            pc_classes = heads_conf['pitch_class']['num_classes']
            self.heads['pitch_class'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, pc_classes)
            )
            self.aux_input_dim += pc_classes # Sigmoid Outputs

        # Onset
        if heads_conf.get('onset', {}).get('enabled', False):
            self.heads['onset'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, self.num_strings)
            )
            self.aux_input_dim += self.num_strings # Sigmoid Outputs

        # --- Tablature Head (Main) ---
        # Input size: bottleneck + outputs of all active aux heads (soft fusion).
        # use_cascade_fusion=False keeps the aux heads and their losses but feeds the
        # tablature head the bottleneck alone (component ablation "auxiliary loss, no
        # fusion"); tab_head_hidden_size widens the head for the parameter-matched
        # single-task baseline (771 units). The defaults reproduce the proposed model.
        self.use_cascade_fusion = bool(model_params.get('use_cascade_fusion', True))
        self.tab_hidden_dim = int(model_params.get('tab_head_hidden_size', 256))
        total_tab_input_dim = self.bottleneck_dim + (self.aux_input_dim if self.use_cascade_fusion else 0)

        tab_out = self.num_strings * self.num_classes_per_string
        self.heads['tablature'] = nn.Sequential(
            nn.Linear(total_tab_input_dim, self.tab_hidden_dim), # <-- Expanded input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.tab_hidden_dim, tab_out)
        )

        logger.info(f"Cascaded Input Size: {self.bottleneck_dim} (Bottleneck) + {self.aux_input_dim} (Aux) = {total_tab_input_dim}"
                    f" | fusion={self.use_cascade_fusion}, tab hidden={self.tab_hidden_dim}")

    def forward(self, inputs):
        """Runs the full forward pass: per-feature backbone -> bottleneck -> auxiliary heads -> tablature head.

        Each active feature is encoded and (optionally) projected, then
        concatenated into the bottleneck. Enabled auxiliary heads run on the
        bottleneck; their logits are stored in `outputs`, and (unless
        `use_cascade_fusion` is off) their sigmoid/softmax probabilities are
        concatenated with the bottleneck to form the tablature head's input.

        Returns:
            A dict with 'tab_logits' and, for each enabled auxiliary head, its '{name}_logits'.
        """
        # 1. Backbone Extraction
        embeddings = []
        for feat_key in self.active_features:
            x = inputs[feat_key]
            out = self.feature_branches[feat_key](x)
            out = out.flatten(1)
            if self.use_projection:
                out = self.projections[feat_key](out)
            embeddings.append(out)
            
        bottleneck = torch.cat(embeddings, dim=1)
        outputs = {}

        # List to hold the auxiliary task outputs to be concatenated
        aux_features_list = []

        # 2. Run Aux Heads FIRST

        if 'multipitch' in self.heads:
            mp_logits = self.heads['multipitch'](bottleneck)
            outputs['multipitch_logits'] = mp_logits # Keep logits for the loss
            aux_features_list.append(torch.sigmoid(mp_logits)) # Use probabilities for tablature

        if 'hand_position' in self.heads:
            hp_logits = self.heads['hand_position'](bottleneck)
            outputs['hand_pos_logits'] = hp_logits
            aux_features_list.append(torch.softmax(hp_logits, dim=1)) # Softmax for multi-class

        if 'string_activity' in self.heads:
            sa_logits = self.heads['string_activity'](bottleneck)
            outputs['activity_logits'] = sa_logits
            aux_features_list.append(torch.sigmoid(sa_logits)) # Multi-label

        if 'pitch_class' in self.heads:
            pc_logits = self.heads['pitch_class'](bottleneck)
            outputs['pitch_class_logits'] = pc_logits
            aux_features_list.append(torch.sigmoid(pc_logits)) # Multi-label

        if 'onset' in self.heads:
            on_logits = self.heads['onset'](bottleneck)
            outputs['onset_logits'] = on_logits
            aux_features_list.append(torch.sigmoid(on_logits))

        # 3. Concatenation (Soft-Fusion)
        if aux_features_list and self.use_cascade_fusion:
            # [Batch, Aux_Total_Dim]
            aux_concat = torch.cat(aux_features_list, dim=1)
            # [Batch, Bottleneck + Aux_Total_Dim]
            tab_input = torch.cat([bottleneck, aux_concat], dim=1)
        else:
            tab_input = bottleneck

        # 4. Run Main Task (Tablature)
        outputs['tab_logits'] = self.heads['tablature'](tab_input)
            
        return outputs
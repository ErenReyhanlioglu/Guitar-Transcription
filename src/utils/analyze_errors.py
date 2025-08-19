import torch
import yaml
import numpy as np
import os
import argparse
import torch.nn.functional as F
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging 

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import TabCNN, FretNet, Transformer
from src.data_loader import TablatureDataset, get_dataloaders
from torch.utils.data import DataLoader
from src.utils.metrics import finalize_output
from src.utils.plotting import plot_confusion_matrix
from src.utils.config_helpers import process_config
from src.utils.agt_tools import logistic_to_tablature
from src.utils.guitar_profile import GuitarProfile
from src.utils.logger import setup_logger, describe 

def analyze(experiment_path: str, val_loader: DataLoader = None):
    """
    Analyzes the errors of a trained model by generating confusion matrices.
    """
    try:
        EXP_CONFIG_PATH = os.path.join(experiment_path, "config.yaml")
        with open(EXP_CONFIG_PATH, 'r') as f:
            raw_exp_config = yaml.safe_load(f)
        config = process_config(raw_exp_config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {EXP_CONFIG_PATH}. Cannot proceed.")
        return
    
    stateful_filter = setup_logger(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting error analysis script.")
    logger.info(f"Loading configuration from: {EXP_CONFIG_PATH}")
    
    model_name = config['model']['name']
    model_params = config['model']['params'].copy()
    
    MODEL_PATH = os.path.join(experiment_path, "model_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_map = {'TabCNN': TabCNN, 'FretNet': FretNet, 'Transformer': Transformer}
    
    feature_config = config['feature_definitions'][config['data']['active_feature']]
    in_channels = feature_config['in_channels']
    num_freq = feature_config['num_freq']
    num_strings = config['instrument']['num_strings']
    num_classes = config['data']['num_classes']

    model_params.pop('in_channels', None)
    model_params.pop('num_freq', None)
    model_params.pop('num_strings', None)
    model_params.pop('num_classes', None)
    model_params.pop('config', None)
    
    logger.info(f"Loading '{model_name}' model from: {MODEL_PATH}")
    
    model = model_map[model_name](
        in_channels=in_channels,
        num_freq=num_freq,
        num_strings=num_strings,
        num_classes=num_classes,
        config=config,
        **model_params
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"'{model_name}' model loaded successfully.")

    if val_loader is None:
        logger.info("Validation dataloader not provided. Creating one from config...")
        _, val_loader, _ = get_dataloaders(config)
    
    profile = GuitarProfile(config['instrument'])
    active_feature_name = config['data']['active_feature']
    feature_key = config['feature_definitions'][active_feature_name]['key']
    silence_class = config['data']['silence_class']
    
    preparation_mode = config['data'].get('active_preparation_mode', 'windowing')
    num_strings = config['instrument']['num_strings']
    num_classes = config['data']['num_classes']

    all_preds_tab_list, all_targets_tab_list = [], []
    logger.info("Making predictions on the validation set for analysis...")
    
    stateful_filter.reset() 
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch[feature_key].to(device)
            targets_full = batch['tablature']
            
            if i == 0:
                logger.debug(f"[Batch {i}] Inputs: {describe(inputs)}, Targets: {describe(targets_full)}")
            
            if preparation_mode == 'framify':
                framify_win_size = config['data'].get('framify_window_size', 9)
                pad_amount = framify_win_size // 2
                inputs_padded = F.pad(inputs, (0, 0, pad_amount, pad_amount), 'constant', 0)
                unfolded = inputs_padded.unfold(2, framify_win_size, 1).permute(0, 2, 1, 3, 4)
                B, T, C, F_in, W = unfolded.shape
                input_frames = unfolded.reshape(B * T, C, F_in, W)
                
                logits_flat = model(input_frames)
                
                if config['loss']['active_loss'] == 'logistic_bank':
                    num_output_classes = num_classes - 1
                else:
                    num_output_classes = num_classes
                
                logits = logits_flat.view(B, T, num_strings, num_output_classes)
            else: # 'windowing'
                logits = model(inputs)

            if isinstance(logits, dict):
                logits = logits['tablature']

            targets = targets_full.permute(0, 2, 1)

            logger.debug(f"[Batch {i}] Raw logits: {describe(logits)}, Targets: {describe(targets)}")
            
            if config['loss']['active_loss'] == 'logistic_bank':
                probs = torch.sigmoid(logits.reshape(logits.shape[0], logits.shape[1], -1)).cpu()
                pred_threshold = config['post_processing'].get('prediction_threshold', 0.5)
                preds = logistic_to_tablature(
                    probs.view(-1, probs.shape[-1]), 
                    num_strings=num_strings, 
                    num_classes=num_classes, 
                    threshold=pred_threshold
                ).view(targets.shape[0], targets.shape[1], -1)

            else: 
                preds = finalize_output(
                    logits.cpu(),
                    silence_class=silence_class,
                    return_shape="tablature"
                )
            
            logger.debug(f"[Batch {i}] Final Predictions: {describe(preds)}")
            
            all_preds_tab_list.append(preds.reshape(-1, num_strings))
            all_targets_tab_list.append(targets.reshape(-1, num_strings))

    logger.info("Prediction loop completed. Concatenating tensors for analysis...")
    all_preds_tab_flat = torch.cat(all_preds_tab_list, dim=0)
    all_targets_tab_flat = torch.cat(all_targets_tab_list, dim=0)
    
    logger.debug(f"Concatenated Predictions: {describe(all_preds_tab_flat)}")
    logger.debug(f"Concatenated Targets: {describe(all_targets_tab_flat)}")
    
    all_targets_tab_flat[all_targets_tab_flat == -1] = silence_class
    
    eval_dir = os.path.join(experiment_path, 'analysis_plots')
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Analysis plots will be saved to: {eval_dir}")

    for s in range(num_strings):
        preds_s = all_preds_tab_flat[:, s].flatten().numpy()
        targets_s = all_targets_tab_flat[:, s].flatten().numpy()
        
        true_labels = targets_s
        pred_labels = preds_s
        
        logger.info(f"Generating confusion matrix for String {s+1}...")
        cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(num_classes))
        target_names = [f'F{i}' for i in range(num_classes - 1)] + ['Silence']
        
        plot_confusion_matrix(cm, target_names=target_names, title=f'String {s+1} Confusion Matrix - {model_name}')
        save_path = os.path.join(eval_dir, f'confusion_matrix_string_{s+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot for String {s+1} saved to {save_path}.")
    
    logger.info("Analysis complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Error Analysis Script for Guitar Transcription Models")
    parser.add_argument('--experiment_path', type=str, required=True, help="Full path to the experiment to be analyzed.")
    args = parser.parse_args()
    analyze(args.experiment_path)
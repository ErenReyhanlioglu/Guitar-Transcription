# src/utils/analyze_errors.py
import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging 
from pathlib import Path
import sys
import yaml

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_PATH))

from src.utils.plotting import plot_confusion_matrix
from src.utils.logger import setup_logger
from src.utils.config_helpers import process_config 
from src.models import get_model                   
from src.data_loader import get_dataloaders        

logger = logging.getLogger(__name__)

def analyze(experiment_path: str, val_loader: DataLoader = None):
    try:
        EXP_CONFIG_PATH = os.path.join(experiment_path, "config.yaml")
        with open(EXP_CONFIG_PATH, 'r') as f:
            raw_exp_config = yaml.safe_load(f)
        config = process_config(raw_exp_config) 
    except FileNotFoundError:
        logger.error(f"Config file not found at {EXP_CONFIG_PATH}. Cannot proceed.")
        return
    
    setup_logger(config)
    
    logger.info("--- Starting Error Analysis ---")
    logger.info(f"Analyzing experiment: {experiment_path}")
    
    MODEL_PATH = os.path.join(experiment_path, "model_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if val_loader is None:
        logger.info("Validation dataloader not provided, creating one from config...")
        _, val_loader, _ = get_dataloaders(config)
        
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"'{config['model']['name']}' model loaded successfully in eval mode.")
    num_frets = config['instrument']['num_frets'] 
    silence_class = config['instrument']['num_frets'] + 1
    num_strings = config['instrument']['num_strings']
    num_classes = config['model']['params']['num_classes']

    all_preds_tab_list, all_targets_tab_list = [], []
    logger.info("Making predictions on the validation set...")
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: tensor.to(device) for key, tensor in batch['features'].items()}
            targets = batch['tablature']
            
            model_output = model(**inputs)
            tab_logits = model_output[0] if isinstance(model_output, tuple) else model_output
            
            # --- FIX 2: Use the robust logic from Trainer._calculate_all_metrics ---
            preds_tab = None
            if config['loss']['active_loss'] == 'softmax_groups':
                if tab_logits.dim() == 4: # FretNet-like output (B, T, S, C)
                    preds_tab = torch.argmax(tab_logits.permute(0, 2, 1, 3), dim=-1)
                elif tab_logits.dim() == 2: # TabCNN-like output (B*T, S*C)
                    preds_tab = torch.argmax(tab_logits.view(-1, num_strings, num_classes), dim=-1)
            else:
                # Placeholder for logistic_bank logic if you implement it later
                raise NotImplementedError("Analysis for logistic_bank is not implemented yet.")
            
            # Unify both preds and targets to a common flat format (Total_Frames, S)
            # This makes the logic universal for any model.
            
            # Flatten predictions
            if preds_tab.dim() == 3: # FretNet preds (B, S, T) -> flatten
                preds_flat = preds_tab.permute(0, 2, 1).reshape(-1, num_strings)
            else: # TabCNN preds are already (B*T, S)
                preds_flat = preds_tab
            
            # Flatten targets
            if targets.dim() == 3: # FretNet targets (B, S, T) -> flatten
                targets_flat = targets.permute(0, 2, 1).reshape(-1, num_strings)
            else: # TabCNN targets are already (B*T, S)
                targets_flat = targets
            # --- FIX 2 END ---

            all_preds_tab_list.append(preds_flat.cpu())
            all_targets_tab_list.append(targets_flat.cpu())

    logger.info("Concatenating all batches for final analysis...")
    all_preds_tab_flat = torch.cat(all_preds_tab_list, dim=0)
    all_targets_tab_flat = torch.cat(all_targets_tab_list, dim=0)
    
    # Replace -1 padding with the actual silence class index
    all_targets_tab_flat[all_targets_tab_flat == -1] = silence_class
    
    eval_dir = os.path.join(experiment_path, 'analysis_plots')
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Analysis plots will be saved to: {eval_dir}")

    class_names = [f'F{i}' for i in range(num_frets + 1)] + ["Silence"]
    
    for s in range(num_strings):
        preds_s = all_preds_tab_flat[:, s].numpy()
        targets_s = all_targets_tab_flat[:, s].numpy()
        
        logger.info(f"Generating confusion matrix for String {s+1}...")
        cm = confusion_matrix(targets_s, preds_s, labels=np.arange(num_classes))
        
        plot_confusion_matrix(
            cm, 
            target_names=class_names, 
            title=f'String {s+1} Confusion Matrix',
            normalize=True # It's often better to see percentages
        )
        save_path = os.path.join(eval_dir, f'confusion_matrix_string_{s+1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Plot for String {s+1} saved.")
    
    logger.info("--- Error Analysis Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Error Analysis Script for Guitar Transcription Models")
    parser.add_argument('experiment_path', type=str, help="Full path to the experiment directory to be analyzed.")
    args = parser.parse_args()
    
    # This script is now robust enough to be called from the command line
    analyze(args.experiment_path)
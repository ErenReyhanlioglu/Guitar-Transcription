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

logger = logging.getLogger(__name__)

def analyze(experiment_path: str, main_exp_path: str, val_loader: DataLoader = None):
    """
    Loads a trained model from an experiment, runs it on the validation set,
    and generates detailed error analysis plots (confusion matrices) for
    tablature, onsets, and offsets on the raw, unfiltered data.
    """
    try:
        EXP_CONFIG_PATH = os.path.join(main_exp_path, "config.yaml")
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
        from src.data_loader import get_dataloaders_for_fold
        _, val_loader, _ = get_dataloaders_for_fold(config)
        
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"'{config['model']['name']}' model loaded successfully in eval mode.")
    
    num_strings = config['instrument']['num_strings']
    num_classes = config['model']['params']['num_classes']
    silence_class = config['instrument']['silence_class']

    all_preds_tab, all_targets_tab = [], []
    all_preds_onset, all_targets_onset = [], []
    all_preds_offset, all_targets_offset = [], []
    
    logger.info("Making predictions on the validation set...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: tensor.to(device) for key, tensor in batch['features'].items()}
            
            model_output = model(**inputs)
            
            if 'tablature' in batch and 'tab_logits' in model_output:
                targets_tab = batch['tablature']
                tab_logits = model_output['tab_logits']
                
                preds_tab = None
                if tab_logits.dim() == 4: # FretNet/Transformer (B, T, S, C)
                    preds_tab = torch.argmax(tab_logits.permute(0, 2, 1, 3), dim=-1)
                elif tab_logits.dim() == 2: # TabCNN (B*T, S*C)
                    preds_tab = torch.argmax(tab_logits.view(-1, num_strings, num_classes), dim=-1)

                preds_flat = preds_tab.permute(0, 2, 1).reshape(-1, num_strings) if preds_tab.dim() == 3 else preds_tab
                targets_flat = targets_tab.permute(0, 2, 1).reshape(-1, num_strings) if targets_tab.dim() == 3 else targets_tab
                
                all_preds_tab.append(preds_flat.cpu())
                all_targets_tab.append(targets_flat.cpu())

            if 'onset_target' in batch and 'onset_logits' in model_output:
                onset_logits = model_output['onset_logits']
                preds_onset = (torch.sigmoid(onset_logits) > 0.5).int()
                onset_targets = batch['onset_target']
                if onset_logits.shape != onset_targets.shape and onset_targets.dim() == 3:
                         onset_targets = onset_targets.permute(0,2,1).reshape(-1, num_strings)
                all_preds_onset.append(preds_onset.cpu())
                all_targets_onset.append(onset_targets.cpu())

            if 'offset_target' in batch and 'offset_logits' in model_output:
                offset_logits = model_output['offset_logits']
                preds_offset = (torch.sigmoid(offset_logits) > 0.5).int()
                offset_targets = batch['offset_target']
                if offset_logits.shape != offset_targets.shape and offset_targets.dim() == 3:
                       offset_targets = offset_targets.permute(0,2,1).reshape(-1, num_strings)
                all_preds_offset.append(preds_offset.cpu())
                all_targets_offset.append(offset_targets.cpu())

    eval_dir = os.path.join(experiment_path, 'analysis_plots')
    tab_analysis_path = os.path.join(eval_dir, 'tablature')
    onset_analysis_path = os.path.join(eval_dir, 'onset')
    offset_analysis_path = os.path.join(eval_dir, 'offset')

    for path in [tab_analysis_path, onset_analysis_path, offset_analysis_path]:
        os.makedirs(path, exist_ok=True)
    logger.info(f"Analysis plots will be saved to subdirectories in: {eval_dir}")

    if all_preds_tab:
        logger.info("Generating Tablature Confusion Matrices...")
        all_preds_tab_flat = torch.cat(all_preds_tab, dim=0)
        all_targets_tab_flat = torch.cat(all_targets_tab, dim=0)
        all_targets_tab_flat[all_targets_tab_flat == -1] = silence_class
        
        class_names = [f'F{i}' for i in range(silence_class)] + ["Silence", "Out"]
        class_names = class_names[:num_classes]

        for s in range(num_strings):
            preds_s = all_preds_tab_flat[:, s]
            targets_s = all_targets_tab_flat[:, s]
            
            if len(targets_s) == 0:
                continue

            cm = confusion_matrix(targets_s.numpy(), preds_s.numpy(), labels=np.arange(num_classes))
            save_path = os.path.join(tab_analysis_path, f'cm_tablature_string_{s+1}.png')
            plot_confusion_matrix(cm, target_names=class_names, title=f'String {s+1} Tablature CM', save_path=save_path)
        logger.info(f"Tablature analysis plots saved to: {tab_analysis_path}")

    for task in ['onset', 'offset']:
        preds_list = all_preds_onset if task == 'onset' else all_preds_offset
        targets_list = all_targets_onset if task == 'onset' else all_targets_offset

        if preds_list:
            logger.info(f"Generating {task.capitalize()} Confusion Matrices...")
            all_preds_flat = torch.cat(preds_list, dim=0)
            all_targets_flat = torch.cat(targets_list, dim=0)
            
            output_folder = onset_analysis_path if task == 'onset' else offset_analysis_path

            for s in range(num_strings):
                cm = confusion_matrix(all_targets_flat[:, s].numpy(), all_preds_flat[:, s].numpy(), labels=[0, 1])
                save_path = os.path.join(output_folder, f'cm_{task}_string_{s+1}.png')
                plot_confusion_matrix(cm, target_names=[f'No {task}', task.capitalize()], title=f'String {s+1} {task.capitalize()} CM', save_path=save_path)
            logger.info(f"{task.capitalize()} analysis plots saved to: {output_folder}")
    
    logger.info("--- Error Analysis Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Error Analysis Script for Guitar Transcription Models")
    parser.add_argument('experiment_path', type=str, help="Full path to the experiment directory to be analyzed.")
    args = parser.parse_args()
    
    analyze(args.experiment_path)
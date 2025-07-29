# src/utils/analyze_errors.py
import torch
import yaml
import numpy as np
import os
import sys
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import TabCNN, FretNet, Transformer
from src.data_loader import TablatureDataset 
from torch.utils.data import DataLoader
from src.utils.metrics import finalize_output
from src.utils.plotting import plot_confusion_matrix
from src.utils.config_helpers import process_config


def analyze(experiment_path, val_loader=None):
    EXP_CONFIG_PATH = os.path.join(experiment_path, "config.yaml")
    with open(EXP_CONFIG_PATH, 'r') as f:
        raw_exp_config = yaml.safe_load(f)
    config = process_config(raw_exp_config)
    
    model_name = config['model']['name']
    model_params = config['model']['params']

    MODEL_PATH = os.path.join(experiment_path, "model_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_map = {'TabCNN': TabCNN, 'FretNet': FretNet, 'Transformer': Transformer}
    model = model_map[model_name](**model_params).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"'{model_name}' model loaded successfully from {experiment_path}")

    if val_loader:
        print("\nUsing the provided validation dataloader. Skipping creation.")
    else:
        print("\nValidation dataloader not provided. Creating one from scratch...")
        original_config_name = f"{model_name.lower()}_config.yaml"
        BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', original_config_name)
        with open(BASE_CONFIG_PATH, 'r') as f:
            base_config = yaml.safe_load(f)
            
        data_config = base_config['data']
        data_root_path = data_config.get('local_data_path')
        if not data_root_path or not os.path.exists(data_root_path):
            data_root_path = os.path.join(PROJECT_ROOT, data_config.get('drive_data_path'))
        
        all_files = sorted(glob(os.path.join(data_root_path, '*.npz')))
        _, validation_paths = train_test_split(
            all_files,
            test_size=data_config['validation_split_size'],
            random_state=data_config['random_state']
        )
        val_dataset = TablatureDataset(validation_paths, base_config)
        val_loader = DataLoader(
            val_dataset, batch_size=data_config['batch_size'], shuffle=False,
            num_workers=data_config.get('num_workers', 0),
            pin_memory=True if data_config.get('num_workers', 0) > 0 else False
        )
        print(f"Validation loader created with {len(val_dataset)} samples.")

    all_preds, all_targets = [], []
    print("\nMaking predictions on the validation set...")
    for batch in val_loader:
        inputs = batch['cqt'].to(device) 
        targets = batch['tablature']
        logits = model(inputs)
        preds = finalize_output(
            logits.cpu(),
            silence_class=model_params.get('num_classes', 22) - 1,
            return_shape="targets"
        )
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    num_strings = all_preds.shape[1]
    all_preds_flat = all_preds.permute(1, 0, 2).reshape(num_strings, -1)
    all_targets_flat = all_targets.permute(1, 0, 2).reshape(num_strings, -1)

    eval_dir = os.path.join(experiment_path, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {eval_dir}")

    for s in range(num_strings):
        mask = all_targets_flat[s] != -1
        true_labels = all_targets_flat[s][mask].numpy()
        pred_labels = all_preds_flat[s][mask].numpy()
        num_classes = model_params.get('num_classes', 22)
        cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(num_classes))
        target_names = [f'Fret {i}' for i in range(num_classes - 1)] + ['Silence']
        
        plot_confusion_matrix(cm, target_names=target_names, title=f'String {s+1} Confusion Matrix - {model_name}')
        save_path = os.path.join(eval_dir, f'confusion_matrix_string_{s+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Plot for String {s+1} saved.")
    
    print("\nAnalysis complete. All plots have been saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Error Analysis Script for Guitar Transcription Models")
    parser.add_argument('--experiment_path', type=str, required=True, help="Full path to the experiment to be analyzed.")
    args = parser.parse_args()
    analyze(args.experiment_path)

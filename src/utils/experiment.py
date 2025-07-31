# src/utils/experiment.py

import os
import re
import shutil
import subprocess
import sys
import torch
import numpy as np
import io
from contextlib import redirect_stdout
from torchsummary import summary
from pathlib import Path

from .plotting import (
    plot_loss_curves,
    plot_metrics,
    plot_spectrogram,
    plot_guitar_tablature,
    plot_pianoroll,
    plot_notes
)

def create_experiment_directory(base_output_path: str, model_name: str, config_path: str):
    model_path = os.path.join(base_output_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    existing_versions = [int(d[1:]) for d in os.listdir(model_path) if d.startswith('V') and d[1:].isdigit()]
    next_version = max(existing_versions) + 1 if existing_versions else 0
    exp_dir_name = f"V{next_version}"
    experiment_path = os.path.join(model_path, exp_dir_name)
    os.makedirs(experiment_path, exist_ok=True)
    checkpoints_path = os.path.join(experiment_path, "model_checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    print(f"Experiment directory created: {experiment_path}")
    shutil.copy(config_path, os.path.join(experiment_path, 'config.yaml'))
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).strip().decode('utf-8')
        with open(os.path.join(experiment_path, 'environment.txt'), 'w') as f:
            f.write(pip_freeze)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not run 'pip freeze'. Library versions were not saved.")
    return experiment_path

def save_model_summary(model, config, experiment_path):
    input_size = (
        config['model']['params']['in_channels'],
        config['model']['params']['num_freq'],
        config['data']['window_size']
    )
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        summary(model.to(device), input_size=input_size, device=str(device))
    model_summary_str = summary_buffer.getvalue()
    summary_file_path = os.path.join(experiment_path, "model_summary.txt")
    with open(summary_file_path, 'w') as f:
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Input Size: {input_size}\n\n")
        f.write(model_summary_str)
    print(f"Model summary saved: {summary_file_path}")

def generate_experiment_report(model, history, val_loader, config, experiment_path, device):
    print("\n" + "="*50)
    print("Generating Final Experiment Report...")
    print("="*50)

    charts_path = os.path.join(experiment_path, "charts")
    os.makedirs(charts_path, exist_ok=True)
    
    plot_loss_curves(history, os.path.join(charts_path, "loss_curve.png"))
    plot_metrics(history, 'f1', os.path.join(charts_path, "f1_score_curve.png"))
    plot_metrics(history, 'acc', os.path.join(charts_path, "accuracy_curve.png"))

    print("Sampling a batch for visual comparison...")
    try:
        sample_batch = next(iter(val_loader))
    except StopIteration:
        print("⚠️ Validation loader is empty. Cannot generate sample plots.")
        return

    active_feature_name = config['data']['active_feature']
    feature_key = config['data']['features'][active_feature_name]['key']

    model.eval()
    with torch.no_grad():
        feature_tensor = sample_batch[feature_key].to(device)
        logits = model(feature_tensor, apply_softmax=False)
        
        predicted_tab_tensor = torch.argmax(logits, dim=-1)
        predicted_tab_tensor[predicted_tab_tensor == config['data'].get('silence_class', 20)] = -1

    feature_np = feature_tensor[0].cpu().numpy().squeeze()

    gt_tab_np = sample_batch["tablature"][0].cpu().numpy()
    pred_tab_np = predicted_tab_tensor[0].cpu().numpy().T
    hop_seconds = config['data']['hop_length'] / config['data']['sample_rate']

    plot_spectrogram(feature_np, hop_seconds, os.path.join(charts_path, "sample_spectrogram.png"))
    
    plot_guitar_tablature(
        gt_tab_np, hop_seconds, 
        os.path.join(charts_path, "sample_tablature_ground_truth.png"),
        title="Ground Truth Tablature"
    )
    plot_guitar_tablature(
        pred_tab_np, hop_seconds, 
        os.path.join(charts_path, "sample_tablature_prediction.png"),
        title="Predicted Tablature"
    )

    def _tab_to_pianoroll(tab, tuning):
        pianoroll = np.zeros((128, tab.shape[1]))
        for string, string_tuning in enumerate(tuning):
            for time, fret in enumerate(tab[string]):
                if fret != -1:
                    midi_note = string_tuning + fret
                    if 0 <= midi_note < 128:
                        pianoroll[int(midi_note), time] = 1
        return pianoroll[21:109, :] 
    
    tuning = config['data'].get('tuning', (40, 45, 50, 55, 59, 64))
    gt_pianoroll = _tab_to_pianoroll(gt_tab_np, tuning)
    pred_pianoroll = _tab_to_pianoroll(pred_tab_np, tuning)
    
    plot_pianoroll(gt_pianoroll, hop_seconds, os.path.join(charts_path, "sample_pianoroll_ground_truth.png"), title="Ground Truth (Pianoroll View)")
    plot_pianoroll(pred_pianoroll, hop_seconds, os.path.join(charts_path, "sample_pianoroll_prediction.png"), title="Prediction (Pianoroll View)")

    print("="*50)
    print("Experiment report generation complete.")
    print("="*50)
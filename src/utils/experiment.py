import os
import shutil
import subprocess
import sys
import torch
import numpy as np
import io
import torch.nn.functional as F
from contextlib import redirect_stdout
from torchsummary import summary
from pathlib import Path

from .plotting import (
    plot_loss_curves,
    plot_metrics_custom, 
    plot_spectrogram,
    plot_guitar_tablature,
    plot_pianoroll
)
from .agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch, logistic_to_tablature
from .metrics import finalize_output
from .guitar_profile import GuitarProfile

def create_experiment_directory(base_output_path: str, model_name: str, config: dict, config_path: str) -> str:
    """
    Creates an experiment output directory, handling regular and fine-tuning runs.
    """
    pretrained_path = config.get('training', {}).get('pretrained_model_path')

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Fine-tuning run detected. Pre-trained model path: {pretrained_path}")
        pretrain_exp_path = Path(pretrained_path).resolve().parent
        if pretrain_exp_path.name.endswith("checkpoints"):
             pretrain_exp_path = pretrain_exp.parent
        base_path_for_versioning = os.path.join(pretrain_exp_path, "finetune")
    else:
        base_path_for_versioning = os.path.join(base_output_path, model_name)
    
    os.makedirs(base_path_for_versioning, exist_ok=True)
    
    existing_versions = [int(d[1:]) for d in os.listdir(base_path_for_versioning) if d.startswith('V') and d[1:].isdigit()]
    next_version = max(existing_versions) + 1 if existing_versions else 0
    exp_dir_name = f"V{next_version}"
    experiment_path = os.path.join(base_path_for_versioning, exp_dir_name)
    
    os.makedirs(os.path.join(experiment_path, "model_checkpoints"), exist_ok=True)
    print(f"Experiment directory created: {experiment_path}")
    
    shutil.copy(config_path, os.path.join(experiment_path, 'config.yaml'))
    
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).strip().decode('utf-8')
        with open(os.path.join(experiment_path, 'environment.txt'), 'w') as f:
            f.write(pip_freeze)
    except Exception as e:
        print(f"Warning: Could not save environment.txt. Reason: {e}")
            
    return experiment_path

def save_model_summary(model, config, experiment_path):
    """
    Saves a summary of the model architecture and parameters to a text file.
    This version correctly handles both 'windowing' and 'framify' preparation modes.
    """
    try:
        from torchinfo import summary
    except ImportError:
        print("Warning: 'torchinfo' not found. Model summary will be skipped.")
        return

    summary_path = os.path.join(experiment_path, "model_summary.txt")
    
    data_config = config['data']
    feature_config = config['feature_definitions'][data_config['active_feature']]
    
    in_channels = feature_config['in_channels']
    num_freq = feature_config['num_freq']
    
    preparation_mode = data_config.get('preparation_mode', 'windowing')
    if preparation_mode == 'framify':
        time_dim = data_config.get('framify_window_size', 9) 
    else:
        time_dim = data_config.get('window_size', 200)

    input_size_for_summary = (1, in_channels, num_freq, time_dim)

    try:
        model_summary = summary(model, input_size=input_size_for_summary, verbose=0)
        with open(summary_path, "w") as f:
            f.write(str(model_summary))
        print(f"Model summary saved to {summary_path}")
    except Exception as e:
        print(f"Could not generate or save model summary. Error: {e}")
        with open(summary_path, "w") as f:
            f.write("Could not generate model summary.\n")
            f.write(f"Error: {e}\n")

def generate_experiment_report(model: torch.nn.Module, history: dict, val_loader: torch.utils.data.DataLoader, config: dict, experiment_path: str, device: torch.device):
    """Generates visual reports (plots, sample predictions) at the end of training."""
    print("\n" + "="*50)
    print("Generating Final Experiment Report...")
    print("="*50)

    charts_path = os.path.join(experiment_path, "charts")
    os.makedirs(charts_path, exist_ok=True)
    
    plot_loss_curves(history, os.path.join(charts_path, "loss_curve.png"))
    plot_metrics_custom(history, 'val_tab_f1', 'train_tab_f1', 'Tablature F1 Score', os.path.join(charts_path, "tab_f1_curve.png"))
    plot_metrics_custom(history, 'val_mp_f1', 'train_mp_f1', 'Multi-pitch F1 Score', os.path.join(charts_path, "mp_f1_curve.png"))
    plot_metrics_custom(history, 'val_tab_precision', 'train_tab_precision', 'Tablature Precision', os.path.join(charts_path, "tab_precision_curve.png"))
    plot_metrics_custom(history, 'val_tab_recall', 'train_tab_recall', 'Tablature Recall', os.path.join(charts_path, "tab_recall_curve.png"))
    plot_metrics_custom(history, 'val_mp_precision', 'train_mp_precision', 'Multi-pitch Precision', os.path.join(charts_path, "mp_precision_curve.png"))
    plot_metrics_custom(history, 'val_mp_recall', 'train_mp_recall', 'Multi-pitch Recall', os.path.join(charts_path, "mp_recall_curve.png"))
    
    print("History plots saved.")

    print("Sampling a batch for visual comparison...")
    try:
        sample_batch = next(iter(val_loader))
    except StopIteration:
        print("⚠️ Warning: Validation loader is empty. Cannot generate sample plots.")
        return

    profile = GuitarProfile(config['instrument'])
    feature_key = config['feature_definitions'][config['data']['active_feature']]['key']
    preparation_mode = config['data'].get('preparation_mode', 'windowing')
    
    model.eval()
    with torch.no_grad():
        features = sample_batch[feature_key].to(device)
        
        if preparation_mode == 'framify':
            framify_win_size = config['data'].get('framify_window_size', 9)
            pad_amount = framify_win_size // 2
            
            inputs_padded = F.pad(features, (pad_amount, pad_amount), 'constant', 0)
            
            unfolded = inputs_padded.unfold(3, framify_win_size, 1).permute(0, 3, 1, 2, 4)
            
            B, T, C, n_freqs, W = unfolded.shape
            inputs = unfolded.reshape(B * T, C, n_freqs, W)
            
            logits_flat = model(inputs)
            
            if config['loss']['type'] == 'logistic_bank':
                num_output_classes = config['data']['num_classes'] - 1
            else:
                num_output_classes = config['data']['num_classes']
            
            logits = logits_flat.reshape(B, T, config['instrument']['num_strings'], num_output_classes)
        
        else: 
            inputs = features
            logits = model(inputs)

        if isinstance(logits, dict):
            logits = logits['tablature']
        
        # Output'u finalize etme
        if config['loss']['type'] == 'logistic_bank':
            pred_tab_tensor = logistic_to_tablature(
                torch.sigmoid(logits).cpu(), profile, silence=False
            ).permute(0, 2, 1)
        else: 
            pred_tab_tensor = finalize_output(
                logits.cpu(),
                silence_class=config['data']['silence_class'],
                return_shape="logits", 
                mask_silence=True
            )

    if preparation_mode == 'framify':
        # framify için features'ı (B, C, F, T) şeklinden (C, F, T) şekline getiriyoruz
        unfolded_np = unfolded.cpu().numpy()
        feature_np = np.transpose(unfolded_np[0], (1, 2, 0, 3)) # (T, C, F, W) -> (C, F, W, T)
        center_frame_idx = feature_np.shape[2] // 2
        feature_np = feature_np[:, :, center_frame_idx, :] # (C, F, T)
    else:
        feature_np = features[0].cpu().numpy()

    if feature_np.ndim == 3 and feature_np.shape[0] > 1:
        feature_np = np.mean(feature_np, axis=0)

    gt_tab_np = sample_batch["tablature"][0].cpu().numpy()
    pred_tab_np = pred_tab_tensor[0].cpu().numpy().T
    
    hop_length = config['data'].get('hop_length', 512)
    sample_rate = config['data'].get('sample_rate', 22050)
    hop_seconds = hop_length / sample_rate

    plot_spectrogram(feature_np, hop_seconds, os.path.join(charts_path, "sample_spectrogram.png"))
    plot_guitar_tablature(gt_tab_np, hop_seconds, os.path.join(charts_path, "sample_tablature_ground_truth.png"), title="Ground Truth Tablature")
    plot_guitar_tablature(pred_tab_np, hop_seconds, os.path.join(charts_path, "sample_tablature_prediction.png"), title="Predicted Tablature")

    gt_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(gt_tab_np).unsqueeze(0), profile)
    gt_pianoroll = stacked_multi_pitch_to_multi_pitch(gt_smp).squeeze(0).numpy()
    
    pred_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(pred_tab_np).unsqueeze(0), profile)
    pred_pianoroll = stacked_multi_pitch_to_multi_pitch(pred_smp).squeeze(0).numpy()

    plot_pianoroll(gt_pianoroll, hop_seconds, os.path.join(charts_path, "sample_pianoroll_ground_truth.png"), title="Ground Truth (Pianoroll View)")
    plot_pianoroll(pred_pianoroll, hop_seconds, os.path.join(charts_path, "sample_pianoroll_prediction.png"), title="Prediction (Pianoroll View)")

    print("="*50)
    print("Experiment report generation complete.")
    print("="*50)
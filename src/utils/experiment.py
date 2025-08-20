import os
import shutil
import subprocess
import sys
import torch
import numpy as np
import io
import torch.nn.functional as F
from contextlib import redirect_stdout
from pathlib import Path
import logging
from src.utils.logger import describe

from .plotting import (
    plot_loss_curves, plot_metrics_custom, plot_spectrogram,
    plot_guitar_tablature, plot_pianoroll
)
from .agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch, logistic_to_tablature
from .metrics import finalize_output
from .guitar_profile import GuitarProfile

logger = logging.getLogger(__name__)

def create_experiment_directory(base_output_path: str, model_name: str, config: dict, config_path: str) -> str:
    pretrained_path = config.get('training', {}).get('pretrained_model_path')
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Fine-tuning run detected. Pre-trained model path: {pretrained_path}")
        pretrain_exp_path = Path(pretrained_path).resolve().parent
        if pretrain_exp_path.name.endswith("checkpoints"):
            pretrain_exp_path = pretrain_exp_path.parent 
        base_path_for_versioning = os.path.join(pretrain_exp_path, "finetune")
    else:
        base_path_for_versioning = os.path.join(base_output_path, model_name)
    
    os.makedirs(base_path_for_versioning, exist_ok=True)
    
    existing_versions = [int(d[1:]) for d in os.listdir(base_path_for_versioning) if d.startswith('V') and d[1:].isdigit()]
    next_version = max(existing_versions) + 1 if existing_versions else 0
    exp_dir_name = f"V{next_version}"
    experiment_path = os.path.join(base_path_for_versioning, exp_dir_name)
    
    os.makedirs(os.path.join(experiment_path, "model_checkpoints"), exist_ok=True)
    logger.info(f"Experiment directory created: {experiment_path}")
    
    shutil.copy(config_path, os.path.join(experiment_path, 'config.yaml'))
    
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).strip().decode('utf-8')
        with open(os.path.join(experiment_path, 'environment.txt'), 'w') as f:
            f.write(pip_freeze)
    except Exception as e:
        logger.warning(f"Could not save environment.txt. Reason: {e}")
            
    return experiment_path

def save_model_summary(model, config, experiment_path):
    """
    Saves a simple summary of the model (architecture printout and parameter count)
    to a text file, avoiding torchinfo for stability.
    """
    summary_path = os.path.join(experiment_path, "model_summary.txt")
    logger.info("Generating simple model summary...")
    
    try:
        s = io.StringIO()
        with redirect_stdout(s):
            print(model)
        model_architecture = s.getvalue()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        with open(summary_path, "w") as f:
            f.write("MODEL ARCHITECTURE:\n")
            f.write("="*20 + "\n")
            f.write(model_architecture)
            f.write("\n\n")
            f.write("PARAMETER COUNT:\n")
            f.write("="*20 + "\n")
            f.write(f"Total params: {total_params:,}\n")
            f.write(f"Trainable params: {trainable_params:,}\n")

        logger.info(f"Simple model summary saved to {summary_path}")

    except Exception as e:
        logger.error(f"Could not generate or save simple model summary. Error: {e}", exc_info=True)
        with open(summary_path, "w") as f:
            f.write(f"Could not generate model summary.\nError: {e}\n")


def generate_experiment_report(model: torch.nn.Module, history: dict, val_loader: torch.utils.data.DataLoader, config: dict, experiment_path: str, device: torch.device):
    logger.info("--- Generating Final Experiment Report ---")

    charts_path = os.path.join(experiment_path, "charts")
    sample_path = os.path.join(charts_path, "sample_outputs")
    os.makedirs(sample_path, exist_ok=True)
    
    tab_metrics_path = os.path.join(charts_path, "tablature")
    mp_metrics_path = os.path.join(charts_path, "multi_pitch")
    octave_metrics_path = os.path.join(charts_path, "octave_tolerated")
    os.makedirs(tab_metrics_path, exist_ok=True)
    os.makedirs(mp_metrics_path, exist_ok=True)
    os.makedirs(octave_metrics_path, exist_ok=True)
    
    plot_loss_curves(history, os.path.join(charts_path, "loss_curve.png"))
    logger.info(f"Loss curve plot saved to {os.path.join(charts_path, 'loss_curve.png')}")

    plot_metrics_custom(history, 'val_tab_f1', 'train_tab_f1', 'Tablature F1 Score', os.path.join(tab_metrics_path, "tab_f1_curve.png"))
    plot_metrics_custom(history, 'val_tab_precision', 'train_tab_precision', 'Tablature Precision', os.path.join(tab_metrics_path, "tab_precision_curve.png"))
    plot_metrics_custom(history, 'val_tab_recall', 'train_tab_recall', 'Tablature Recall', os.path.join(tab_metrics_path, "tab_recall_curve.png"))
    logger.info(f"Tablature F1, Precision, and Recall plots saved to {tab_metrics_path}.")

    plot_metrics_custom(history, 'val_mp_f1', 'train_mp_f1', 'Multi-pitch F1 Score', os.path.join(mp_metrics_path, "mp_f1_curve.png"))
    plot_metrics_custom(history, 'val_mp_precision', 'train_mp_precision', 'Multi-pitch Precision', os.path.join(mp_metrics_path, "mp_precision_curve.png"))
    plot_metrics_custom(history, 'val_mp_recall', 'train_mp_recall', 'Multi-pitch Recall', os.path.join(mp_metrics_path, "mp_recall_curve.png"))
    logger.info(f"Multi-pitch F1, Precision, and Recall plots saved to {mp_metrics_path}.")

    if 'val_octave_f1' in history:
        plot_metrics_custom(history, 'val_octave_f1', 'train_octave_f1', 'Octave Tolerant F1 Score', os.path.join(octave_metrics_path, "octave_f1_curve.png"))
        plot_metrics_custom(history, 'val_octave_precision', 'train_octave_precision', 'Octave Tolerant Precision', os.path.join(octave_metrics_path, "octave_precision_curve.png"))
        plot_metrics_custom(history, 'val_octave_recall', 'train_octave_recall', 'Octave Tolerant Recall', os.path.join(octave_metrics_path, "octave_recall_curve.png"))
        logger.info(f"Octave Tolerant F1, Precision, and Recall plots saved to {octave_metrics_path}.")
    
    logger.info("All history plots have been saved.")

    logger.info("Generating a sample batch for visual comparison...")
    try:
        sample_batch = next(iter(val_loader))
    except StopIteration:
        logger.warning("Validation loader is empty. Cannot generate sample plots.")
        return

    profile = GuitarProfile(config['instrument'])
    active_features = config['data']['active_features']
    preparation_mode = config['data'].get('active_preparation_mode', 'windowing')

    hop_length = config['data']['hop_length']
    sample_rate = config['data']['sample_rate']
    hop_seconds = hop_length / sample_rate
    
    model.eval()
    with torch.no_grad():
        features_dict = {key: tensor.to(device) for key, tensor in sample_batch['features'].items()}
        logger.debug(f"Sample report generation - Input features: {list(features_dict.keys())}")
        
        if preparation_mode == 'framify':
            input_frames_dict = {}
            B, T = -1, -1
            for key, tensor in features_dict.items():
                framify_win_size = config['data'].get('framify_window_size', 9)
                pad_amount = framify_win_size // 2
                inputs_padded = F.pad(tensor, (pad_amount, pad_amount, 0, 0), 'constant', 0)
                unfolded = inputs_padded.unfold(3, framify_win_size, 1)
                permuted = unfolded.permute(0, 3, 1, 2, 4)
                _B, _T, C, H, W_win = permuted.shape
                if B == -1: B, T = _B, _T
                input_frames_dict[key] = permuted.reshape(B * T, C, H, W_win)
            
            logits_flat = model(input_frames_dict)
            
            if config['loss']['active_loss'] == 'logistic_bank':
                num_output_classes = config['data']['num_classes'] - 1
            else:
                num_output_classes = config['data']['num_classes']
            logits = logits_flat.view(B, T, config['instrument']['num_strings'], num_output_classes)
        else: 
            logits = model(features_dict)

        if isinstance(logits, dict):
            logits = logits['tablature']
        
        logger.debug(f"Sample report generation - Raw logits from model: {describe(logits)}")
        
        if config['loss']['active_loss'] == 'logistic_bank':
            preds_flat = logits.reshape(-1, logits.shape[-1] * logits.shape[-2])
            pred_tab_tensor = logistic_to_tablature(
                torch.sigmoid(preds_flat).cpu(), model.num_strings, model.num_classes,
                threshold=config.get('post_processing', {}).get('prediction_threshold', 0.5)
            ).view(logits.shape[0], logits.shape[1], logits.shape[2])
        else: 
            pred_tab_tensor = finalize_output(
                logits.cpu(), silence_class=config['data']['silence_class'],
                return_shape="tablature"
            )
        
        pred_tab_tensor = pred_tab_tensor.permute(0, 2, 1) 
        logger.debug(f"Sample report generation - Final predicted tablature tensor: {describe(pred_tab_tensor)}")

    logger.info(f"Visualizing spectrograms for all active features: {active_features}")
    for feature_key in active_features:
        feature_np = sample_batch['features'][feature_key][0].cpu().numpy()
        
        if feature_np.ndim == 3 and feature_np.shape[0] > 1:
            feature_np = np.mean(feature_np, axis=0)
        elif feature_np.ndim == 3 and feature_np.shape[0] == 1:
            feature_np = feature_np.squeeze(0)
            
        save_path = os.path.join(sample_path, f"sample_spectrogram_{feature_key}.png")
        plot_spectrogram(feature_np, hop_seconds, save_path=save_path, title=f"Spectrogram ({feature_key})")

    gt_tab_np = sample_batch["tablature"][0].cpu().numpy()
    pred_tab_np = pred_tab_tensor[0].cpu().numpy()
    
    plot_guitar_tablature(gt_tab_np, hop_seconds, save_path=os.path.join(sample_path, "sample_tablature_ground_truth.png"), title="Ground Truth Tablature")
    plot_guitar_tablature(pred_tab_np, hop_seconds, save_path=os.path.join(sample_path, "sample_tablature_prediction.png"), title="Predicted Tablature")
    
    gt_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(gt_tab_np).unsqueeze(0), profile)
    gt_pianoroll = stacked_multi_pitch_to_multi_pitch(gt_smp).squeeze(0).numpy()
    pred_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(pred_tab_np).unsqueeze(0), profile)
    pred_pianoroll = stacked_multi_pitch_to_multi_pitch(pred_smp).squeeze(0).numpy()
    
    plot_pianoroll(gt_pianoroll, hop_seconds, save_path=os.path.join(sample_path, "sample_pianoroll_ground_truth.png"), title="Ground Truth (Pianoroll View)")
    plot_pianoroll(pred_pianoroll, hop_seconds, save_path=os.path.join(sample_path, "sample_pianoroll_prediction.png"), title="Prediction (Pianoroll View)")

    logger.info("Experiment report generation complete.")
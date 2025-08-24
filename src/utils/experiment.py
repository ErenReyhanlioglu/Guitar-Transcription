import torch
import yaml
import numpy as np
import os
import io
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
import logging

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_PATH))

from src.utils.logger import describe
from src.utils.plotting import (
    plot_loss_curves, plot_metrics_custom, plot_spectrogram,
    plot_guitar_tablature, plot_pianoroll
)
from src.utils.agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch, logistic_to_tablature
from src.utils.guitar_profile import GuitarProfile

logger = logging.getLogger(__name__)

def create_experiment_directory(base_output_path: str, model_name: str, config: dict, config_path: str) -> str:
    pretrained_path = config.get('training', {}).get('pretrained_model_path')
    if pretrained_path and os.path.exists(pretrained_path):
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
    
    os.makedirs(experiment_path, exist_ok=True)
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
    summary_path = os.path.join(experiment_path, "model_summary.txt")
    try:
        s = io.StringIO()
        with redirect_stdout(s):
            print(model)
        model_architecture = s.getvalue()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        with open(summary_path, "w") as f:
            f.write("MODEL ARCHITECTURE:\n" + "="*20 + "\n" + model_architecture + "\n\n")
            f.write("PARAMETER COUNT:\n" + "="*20 + "\n")
            f.write(f"Total params: {total_params:,}\n")
            f.write(f"Trainable params: {trainable_params:,}\n")
        logger.info(f"Simple model summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"Could not generate model summary. Error: {e}", exc_info=True)

import torch
import os
import logging
from src.utils.plotting import (
    plot_metrics_custom, plot_spectrogram,
    plot_guitar_tablature, plot_pianoroll
)
from src.utils.agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch, logistic_to_tablature
from src.utils.guitar_profile import GuitarProfile
from src.utils.logger import describe

logger = logging.getLogger(__name__)

def generate_experiment_report(model: torch.nn.Module, history: dict, val_loader: torch.utils.data.DataLoader, 
                              config: dict, experiment_path: str, device: torch.device, profile: GuitarProfile):
    logger.info("--- Generating Final Experiment Report ---")

    charts_path = os.path.join(experiment_path, "charts")
    sample_path = os.path.join(charts_path, "sample_outputs")
    loss_charts_path = os.path.join(charts_path, "loss")
    tab_charts_path = os.path.join(charts_path, "tablature")
    mp_charts_path = os.path.join(charts_path, "multi_pitch")
    octave_charts_path = os.path.join(charts_path, "octave_tolerated")
    error_charts_path = os.path.join(charts_path, "errors")
    
    for path in [charts_path, sample_path, loss_charts_path, tab_charts_path, mp_charts_path, octave_charts_path, error_charts_path]:
        os.makedirs(path, exist_ok=True)
    
    aux_used = 'train_loss_aux' in history and any(val > 1e-9 for val in history['train_loss_aux'])

    if aux_used:
        logger.info("Plotting all loss components...")
        plot_metrics_custom(history, 'val_loss_total', 'train_loss_total', 'Total Loss', 
                            os.path.join(loss_charts_path, "total_loss_curve.png"))
        plot_metrics_custom(history, 'val_loss_primary', 'train_loss_primary', 'Primary (Tablature) Loss', 
                            os.path.join(loss_charts_path, "primary_loss_curve.png"))
        plot_metrics_custom(history, 'val_loss_aux', 'train_loss_aux', 'Auxiliary (Multipitch) Loss', 
                            os.path.join(loss_charts_path, "auxiliary_loss_curve.png"))
    else:
        logger.info("Plotting total loss...")
        plot_metrics_custom(history, 'val_loss_total', 'train_loss_total', 'Total Loss', 
                            os.path.join(loss_charts_path, "loss_curve.png"))

    plot_jobs = [
        # Tablature Weighted
        {'key': 'tab_f1', 'title': 'Tablature F1 Score (Weighted)', 'dir': tab_charts_path, 'file': 'f1_score_curve_weighted.png'},
        {'key': 'tab_precision', 'title': 'Tablature Precision (Weighted)', 'dir': tab_charts_path, 'file': 'precision_curve_weighted.png'},
        {'key': 'tab_recall', 'title': 'Tablature Recall (Weighted)', 'dir': tab_charts_path, 'file': 'recall_curve_weighted.png'},
        # Tablature Macro
        {'key': 'tab_f1_macro', 'title': 'Tablature F1 Score (Macro)', 'dir': tab_charts_path, 'file': 'f1_score_curve_macro.png'},
        {'key': 'tab_precision_macro', 'title': 'Tablature Precision (Macro)', 'dir': tab_charts_path, 'file': 'precision_curve_macro.png'},
        {'key': 'tab_recall_macro', 'title': 'Tablature Recall (Macro)', 'dir': tab_charts_path, 'file': 'recall_curve_macro.png'},
        
        # Multi-pitch
        {'key': 'mp_f1', 'title': 'Multi-pitch F1 Score', 'dir': mp_charts_path, 'file': 'f1_score_curve.png'},
        {'key': 'mp_precision', 'title': 'Multi-pitch Precision', 'dir': mp_charts_path, 'file': 'precision_curve.png'},
        {'key': 'mp_recall', 'title': 'Multi-pitch Recall', 'dir': mp_charts_path, 'file': 'recall_curve.png'},

        # Octave Tolerant Weighted
        {'key': 'octave_f1', 'title': 'Octave Tolerant F1 (Weighted)', 'dir': octave_charts_path, 'file': 'f1_score_curve_weighted.png'},
        {'key': 'octave_precision', 'title': 'Octave Tolerant Precision (Weighted)', 'dir': octave_charts_path, 'file': 'precision_curve_weighted.png'},
        {'key': 'octave_recall', 'title': 'Octave Tolerant Recall (Weighted)', 'dir': octave_charts_path, 'file': 'recall_curve_weighted.png'},
        # Octave Tolerant Macro
        {'key': 'octave_f1_macro', 'title': 'Octave Tolerant F1 (Macro)', 'dir': octave_charts_path, 'file': 'f1_score_curve_macro.png'},
        {'key': 'octave_precision_macro', 'title': 'Octave Tolerant Precision (Macro)', 'dir': octave_charts_path, 'file': 'precision_curve_macro.png'},
        {'key': 'octave_recall_macro', 'title': 'Octave Tolerant Recall (Macro)', 'dir': octave_charts_path, 'file': 'recall_curve_macro.png'},

        # Error Scores
        {'key': 'tab_error_total', 'title': 'Total Error Rate', 'dir': error_charts_path, 'file': 'error_total_curve.png'},
        {'key': 'tab_error_substitution', 'title': 'Substitution Error Rate', 'dir': error_charts_path, 'file': 'error_substitution_curve.png'},
        {'key': 'tab_error_miss', 'title': 'Miss Error Rate', 'dir': error_charts_path, 'file': 'error_miss_curve.png'},
        {'key': 'tab_error_false_alarm', 'title': 'False Alarm Error Rate', 'dir': error_charts_path, 'file': 'error_false_alarm_curve.png'},
    ]

    logger.info("Generating all metric plots...")
    for job in plot_jobs:
        val_key = f"val_{job['key']}"
        train_key = f"train_{job['key']}"
        
        if val_key in history and train_key in history and any(history[val_key]):
            plot_metrics_custom(
                history=history,
                val_metric_key=val_key,
                train_metric_key=train_key,
                plot_title=job['title'],
                save_path=os.path.join(job['dir'], job['file'])
            )

    logger.info("All history plots have been saved.")

    logger.info("Generating a sample batch for visual comparison...")
    try:
        sample_batch = next(iter(val_loader))
    except StopIteration:
        logger.warning("Validation loader is empty. Cannot generate sample plots.")
        return

    data_config = config['data']
    hop_seconds = data_config['hop_length'] / data_config['sample_rate']
    
    for feature_key, tensor in sample_batch['features'].items():
        spec_to_plot = tensor[0][0].cpu().numpy()
        save_path = os.path.join(sample_path, f"sample_spectrogram_{feature_key}.png")
        plot_spectrogram(spec_to_plot, hop_seconds=hop_seconds, save_path=save_path, title=f"Sample Spectrogram ({feature_key.upper()})")
        logger.info(f"Spectrogram for '{feature_key}' saved to {save_path}")

    model.eval()
    with torch.no_grad():
        features_dict = {key: tensor.to(device) for key, tensor in sample_batch['features'].items()}
        model_output = model(**features_dict)
        
        aux_enabled = config.get('loss', {}).get('auxiliary_loss', {}).get('enabled', False)
        tab_logits = model_output[0] if aux_enabled else model_output
        
        S = config['instrument']['num_strings']
        C = config['model']['params']['num_classes']

        if config['loss']['active_loss'] == 'softmax_groups':
            if tab_logits.dim() == 4: # FretNet (B, T, S, C)
                preds_tab = torch.argmax(tab_logits.permute(0, 2, 1, 3), dim=-1) # (B, S, T)
            elif tab_logits.dim() == 2: # TabCNN (B_flat, S*C)
                preds_tab = torch.argmax(tab_logits.view(-1, S, C), dim=-1) # (B_flat, S)
            else:
                raise ValueError(f"Unsupported tab_logits dimension for report: {tab_logits.dim()}")
        else: # logistic_bank
            # ... (logistic bank kodu) ...
            pass
    
    # 5. Görselleri oluşturmak için veriyi, hazırlama moduna göre doğru formatlarda al
    preparation_mode = config['data']['active_preparation_mode']

    if preparation_mode == 'framify':
        batch_size = config['data']['batch_size']
        time_steps = sample_batch['tablature'].shape[0] // batch_size
        
        gt_tab_np_raw = sample_batch['tablature'][:time_steps].cpu().numpy().T
        pred_tab_np_raw = preds_tab[:time_steps].cpu().numpy().T
        logger.info(f"Framify mode: sample extracted with T={time_steps}. Final shape: {gt_tab_np_raw.shape}")
    else: # 'windowing' for FretNet
        gt_tab_np_raw = sample_batch["tablature"][0].cpu().numpy()
        pred_tab_np_raw = preds_tab[0].cpu().numpy()
        logger.info(f"Windowing mode: sample extracted. Final shape: {gt_tab_np_raw.shape}")

    # 6. Tüm görselleri çizdir ve kaydet
    save_path = os.path.join(sample_path, "sample_tablature_ground_truth.png")
    plot_guitar_tablature(gt_tab_np_raw.T, hop_seconds, save_path=save_path, title="Ground Truth Tablature")
    logger.info(f"Ground Truth Tablature plot saved to: {save_path}")

    save_path = os.path.join(sample_path, "sample_tablature_prediction.png")
    plot_guitar_tablature(pred_tab_np_raw.T, hop_seconds, save_path=save_path, title="Predicted Tablature")
    logger.info(f"Predicted Tablature plot saved to: {save_path}")

    gt_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(gt_tab_np_raw), profile)
    gt_pianoroll = stacked_multi_pitch_to_multi_pitch(gt_smp).numpy()
    
    pred_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(pred_tab_np_raw), profile)
    pred_pianoroll = stacked_multi_pitch_to_multi_pitch(pred_smp).numpy()
    
    save_path = os.path.join(sample_path, "sample_pianoroll_ground_truth.png")
    plot_pianoroll(gt_pianoroll, hop_seconds, save_path=save_path, title="Ground Truth (Pianoroll View)")
    logger.info(f"Ground Truth Pianoroll plot saved to: {save_path}")

    save_path = os.path.join(sample_path, "sample_pianoroll_prediction.png")
    plot_pianoroll(pred_pianoroll, hop_seconds, save_path=save_path, title="Prediction (Pianoroll View)")
    logger.info(f"Predicted Pianoroll plot saved to: {save_path}")

    logger.info("Experiment report generation complete.")
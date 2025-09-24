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
    plot_guitar_tablature, plot_pianoroll, plot_binary_activation,
    plot_tablature_errors, plot_pianoroll_errors  
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
    """Saves a summary of the model architecture and parameter counts."""
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
            f.write("OVERALL PARAMETER COUNT:\n" + "="*20 + "\n")
            f.write(f"Total params: {total_params:,}\n")
            f.write(f"Trainable params: {trainable_params:,}\n")

            if hasattr(model, 'backbone') and hasattr(model, 'head'):
                f.write("\nPARAMETER COUNT BY COMPONENT:\n" + "="*30 + "\n")
                
                backbone_params = sum(p.numel() for p in model.backbone.parameters())
                head_params = sum(p.numel() for p in model.head.parameters())
                
                f.write(f"Backbone params: {backbone_params:,} ({backbone_params/total_params:.2%})\n")
                f.write(f"Head params: {head_params:,} ({head_params/total_params:.2%})\n")
                
        logger.info(f"Model summary with component breakdown saved to {summary_path}")

    except Exception as e:
        logger.error(f"Could not generate model summary. Error: {e}", exc_info=True)

def generate_experiment_report(model: torch.nn.Module, history: dict, val_loader: torch.utils.data.DataLoader, 
                             config: dict, experiment_path: str, device: torch.device, profile: GuitarProfile):
    """The main function that orchestrates the report generation."""
    logger.info("--- Generating Final Experiment Report ---")
    paths = _setup_report_directories(experiment_path)
    _plot_history_curves(history, paths)
    sample_data = _get_sample_data_and_predictions(model, val_loader, config, device, profile)
    _plot_sample_visualizations(sample_data, paths, config)
    logger.info("Experiment report generation complete.")

def _setup_report_directories(experiment_path: str) -> dict:
    """Rapor için gerekli tüm klasörleri oluşturur ve yollarını döndürür."""
    logger.info("Setting up report directories...")
    paths = {
        "charts": os.path.join(experiment_path, "charts"),
        "samples": os.path.join(experiment_path, "charts", "sample_outputs"),
        "loss": os.path.join(experiment_path, "charts", "loss"),
        "tablature": os.path.join(experiment_path, "charts", "tablature"),
        "multi_pitch": os.path.join(experiment_path, "charts", "multi_pitch"),
        "octave": os.path.join(experiment_path, "charts", "octave_tolerated"),
        "errors": os.path.join(experiment_path, "charts", "errors"),
        "tdr": os.path.join(experiment_path, "charts", "tdr")  
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def _plot_history_curves(history: dict, paths: dict):
    """Eğitim geçmişindeki tüm metrikler için kayıp/performans eğrilerini çizer."""
    logger.info("Plotting all history curves...")
    
    plot_jobs = [
        # --- Tablature (Micro) ---
        {'key': 'tab_f1', 'title': 'Tablature F1 Score (Micro)', 'dir': paths['tablature'], 'file': 'f1_score_curve_micro.png'},
        {'key': 'tab_precision', 'title': 'Tablature Precision (Micro)', 'dir': paths['tablature'], 'file': 'precision_curve_micro.png'},
        {'key': 'tab_recall', 'title': 'Tablature Recall (Micro)', 'dir': paths['tablature'], 'file': 'recall_curve_micro.png'},
        
        # --- Tablature (Weighted) ---
        {'key': 'tab_f1_weighted', 'title': 'Tablature F1 Score (Weighted)', 'dir': paths['tablature'], 'file': 'f1_score_curve_weighted.png'},
        {'key': 'tab_precision_weighted', 'title': 'Tablature Precision (Weighted)', 'dir': paths['tablature'], 'file': 'precision_curve_weighted.png'},
        {'key': 'tab_recall_weighted', 'title': 'Tablature Recall (Weighted)', 'dir': paths['tablature'], 'file': 'recall_curve_weighted.png'},
        
        # --- Tablature (Macro) ---
        {'key': 'tab_f1_macro', 'title': 'Tablature F1 Score (Macro)', 'dir': paths['tablature'], 'file': 'f1_score_curve_macro.png'},
        {'key': 'tab_precision_macro', 'title': 'Tablature Precision (Macro)', 'dir': paths['tablature'], 'file': 'precision_curve_macro.png'},
        {'key': 'tab_recall_macro', 'title': 'Tablature Recall (Macro)', 'dir': paths['tablature'], 'file': 'recall_curve_macro.png'},
        
        # --- Multi-pitch ---
        {'key': 'mp_f1', 'title': 'Multi-pitch F1 Score', 'dir': paths['multi_pitch'], 'file': 'f1_score_curve.png'},
        {'key': 'mp_precision', 'title': 'Multi-pitch Precision', 'dir': paths['multi_pitch'], 'file': 'precision_curve.png'},
        {'key': 'mp_recall', 'title': 'Multi-pitch Recall', 'dir': paths['multi_pitch'], 'file': 'recall_curve.png'},

        # --- TDR ---
        {'key': 'tdr', 'title': 'Tablature Disambiguation Rate (TDR)', 'dir': paths['tdr'], 'file': 'tdr_curve.png'},

        # --- Octave Tolerant (Micro) ---
        {'key': 'octave_f1', 'title': 'Octave Tolerant F1 (Micro)', 'dir': paths['octave'], 'file': 'f1_score_curve_micro.png'},
        {'key': 'octave_precision', 'title': 'Octave Tolerant Precision (Micro)', 'dir': paths['octave'], 'file': 'precision_curve_micro.png'},
        {'key': 'octave_recall', 'title': 'Octave Tolerant Recall (Micro)', 'dir': paths['octave'], 'file': 'recall_curve_micro.png'},
        
        # --- Octave Tolerant (Weighted) ---
        {'key': 'octave_f1_weighted', 'title': 'Octave Tolerant F1 (Weighted)', 'dir': paths['octave'], 'file': 'f1_score_curve_weighted.png'},
        {'key': 'octave_precision_weighted', 'title': 'Octave Tolerant Precision (Weighted)', 'dir': paths['octave'], 'file': 'precision_curve_weighted.png'},
        {'key': 'octave_recall_weighted', 'title': 'Octave Tolerant Recall (Weighted)', 'dir': paths['octave'], 'file': 'recall_curve_weighted.png'},
        
        # --- Octave Tolerant (Macro) ---
        {'key': 'octave_f1_macro', 'title': 'Octave Tolerant F1 (Macro)', 'dir': paths['octave'], 'file': 'f1_score_curve_macro.png'},
        {'key': 'octave_precision_macro', 'title': 'Octave Tolerant Precision (Macro)', 'dir': paths['octave'], 'file': 'precision_curve_macro.png'},
        {'key': 'octave_recall_macro', 'title': 'Octave Tolerant Recall (Macro)', 'dir': paths['octave'], 'file': 'recall_curve_macro.png'},

        # --- Errors ---
        {'key': 'tab_error_total', 'title': 'Total Error Rate', 'dir': paths['errors'], 'file': 'error_total_curve.png'},
        {'key': 'tab_error_substitution', 'title': 'Substitution Error Rate', 'dir': paths['errors'], 'file': 'error_substitution_curve.png'},
        {'key': 'tab_error_miss', 'title': 'Miss Error Rate', 'dir': paths['errors'], 'file': 'error_miss_curve.png'},
        {'key': 'tab_error_false_alarm', 'title': 'False Alarm Error Rate', 'dir': paths['errors'], 'file': 'error_false_alarm_curve.png'},
        {'key': 'tab_error_duplicate_pitch', 'title': 'Duplicate Pitch Error Rate', 'dir': paths['errors'], 'file': 'error_duplicate_pitch_curve.png'}
    ]
    
    loss_keys = ["total", "primary", "aux", "activity", "onset", "offset"]
    for loss_key in loss_keys:
        history_key = f'train_loss_{loss_key}'
        if any(val > 1e-9 for val in history.get(history_key, [])):
            plot_jobs.append({
                'key': f'loss_{loss_key}', 
                'title': f'{loss_key.capitalize()} Loss', 
                'dir': paths['loss'], 
                'file': f'{loss_key}_loss_curve.png'
            })
    
    for job in plot_jobs:
        val_key, train_key = f"val_{job['key']}", f"train_{job['key']}"
        if val_key in history and train_key in history and any(history[val_key]):
            plot_metrics_custom(history, val_key, train_key, job['title'], os.path.join(job['dir'], job['file']))
            
    logger.info("All history plots have been saved.")

def _get_sample_data_and_predictions(model, val_loader, config, device, profile):
    """Bir örneklem alır, modelin tüm kafaları için tahmin yapar ve verileri görselleştirme için hazırlar."""
    logger.info("Generating a sample batch for visual comparison...")
    try:
        sample_batch = next(iter(val_loader))
    except StopIteration:
        return None

    model.eval()
    with torch.no_grad():
        features_dict = {key: tensor.to(device) for key, tensor in sample_batch['features'].items()}
        model_output = model(**features_dict)
        
        tab_logits = model_output.get('tab_logits')
        activity_logits = model_output.get('activity_logits')
        onset_logits = model_output.get('onset_logits')
        offset_logits = model_output.get('offset_logits')
        
        S, C = config['instrument']['num_strings'], config['model']['params']['num_classes']
        preds_tab, preds_activity, preds_onset, preds_offset = (None,) * 4

        if tab_logits is not None:
            if tab_logits.dim() == 4: preds_tab = torch.argmax(tab_logits.permute(0, 2, 1, 3), dim=-1)
            elif tab_logits.dim() == 2: preds_tab = torch.argmax(tab_logits.view(-1, S, C), dim=-1)
        if activity_logits is not None: preds_activity = (torch.sigmoid(activity_logits) > 0.5).int()
        if onset_logits is not None: preds_onset = (torch.sigmoid(onset_logits) > 0.5).int()
        if offset_logits is not None: preds_offset = (torch.sigmoid(offset_logits) > 0.5).int()
        
    gt_tab_np, pred_tab_np, gt_activity_np, pred_activity_np, gt_onset_np, pred_onset_np, gt_offset_np, pred_offset_np = (None,) * 8

    preparation_mode = config['data']['active_preparation_mode']
    if preparation_mode == 'framify':
        chunk_size = config['data']['framify_chunk_size'] 
        
        gt_tab_np = sample_batch['tablature'][:chunk_size].cpu().numpy().T
        
        if 'activity_target' in sample_batch:
            gt_activity_np = sample_batch['activity_target'][:chunk_size].cpu().numpy()[np.newaxis, :]
        if 'onset_target' in sample_batch:
            gt_onset_np = sample_batch['onset_target'][:chunk_size].cpu().numpy().T
        if 'offset_target' in sample_batch:
            gt_offset_np = sample_batch['offset_target'][:chunk_size].cpu().numpy().T

        if preds_tab is not None: 
            pred_tab_np = preds_tab[:chunk_size].cpu().numpy().T
        if preds_activity is not None: 
            pred_activity_np = preds_activity[:chunk_size].cpu().numpy()[np.newaxis, :]
        if preds_onset is not None:
            pred_onset_np = preds_onset[:chunk_size].cpu().numpy().T
        if preds_offset is not None:
            pred_offset_np = preds_offset[:chunk_size].cpu().numpy().T

    else: # windowing mode
        gt_tab_np = sample_batch["tablature"][0].cpu().numpy()
        pred_tab_np = preds_tab[0].cpu().numpy()
        if 'activity_target' in sample_batch and preds_activity is not None:
              gt_activity_np = sample_batch['activity_target'][0].cpu().numpy()[np.newaxis, :]
              pred_activity_np = preds_activity[0].cpu().numpy()[np.newaxis, :]
        # ... (add similar logic for onset/offset if needed for windowing)
        pass

    gt_pianoroll, pred_pianoroll = None, None
    if gt_tab_np is not None:
        # Note: tablature_to_stacked_multi_pitch expects (S, T) shape
        gt_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(gt_tab_np), profile)
        gt_pianoroll = stacked_multi_pitch_to_multi_pitch(gt_smp).numpy()
    if pred_tab_np is not None:
        pred_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(pred_tab_np), profile)
        pred_pianoroll = stacked_multi_pitch_to_multi_pitch(pred_smp).numpy()

    return {
        "sample_batch": sample_batch, "gt_tab_np": gt_tab_np, "pred_tab_np": pred_tab_np,
        "gt_activity_np": gt_activity_np, "pred_activity_np": pred_activity_np,
        "gt_onset_np": gt_onset_np, "pred_onset_np": pred_onset_np,
        "gt_offset_np": gt_offset_np, "pred_offset_np": pred_offset_np,
        "gt_pianoroll": gt_pianoroll, "pred_pianoroll": pred_pianoroll,
    }

def _plot_sample_visualizations(sample_data: dict, paths: dict, config: dict):
    """Uses the prepared data to plot all visualizations."""
    if not sample_data: return
    logger.info("Plotting sample visualizations...")
    sample_path = paths['samples']
    hop_seconds = config['data']['hop_length'] / config['data']['sample_rate']
    silence_class = config['instrument']['silence_class']

    for key, tensor in sample_data['sample_batch']['features'].items():
        spec_to_plot = tensor[0][0].cpu().numpy()
        plot_spectrogram(spec_to_plot, hop_seconds, save_path=os.path.join(sample_path, f"sample_spec_{key}.png"), title=f"Sample Spectrogram ({key.upper()})")

    if sample_data['gt_tab_np'] is not None:
        plot_guitar_tablature(sample_data['gt_tab_np'].T, hop_seconds, os.path.join(sample_path, "sample_tab_truth.png"), "Ground Truth Tablature")
    if sample_data['pred_tab_np'] is not None:
        plot_guitar_tablature(sample_data['pred_tab_np'].T, hop_seconds, os.path.join(sample_path, "sample_tab_pred.png"), "Predicted Tablature")
    if sample_data['gt_pianoroll'] is not None:
        plot_pianoroll(sample_data['gt_pianoroll'], hop_seconds, os.path.join(sample_path, "sample_pianoroll_truth.png"), "Ground Truth (Pianoroll)")
    if sample_data['pred_pianoroll'] is not None:
        plot_pianoroll(sample_data['pred_pianoroll'], hop_seconds, os.path.join(sample_path, "sample_pianoroll_pred.png"), "Prediction (Pianoroll)")
    
    if sample_data.get('gt_tab_np') is not None and sample_data.get('pred_tab_np') is not None:
        logger.info("Plotting tablature error visualization...")
        plot_tablature_errors(
            preds_np=sample_data['pred_tab_np'].T, 
            targets_np=sample_data['gt_tab_np'].T,
            hop_seconds=hop_seconds,
            silence_class=silence_class,
            save_path=os.path.join(sample_path, "sample_tab_ERRORS.png")
        )

    if sample_data.get('gt_pianoroll') is not None and sample_data.get('pred_pianoroll') is not None:
        logger.info("Plotting pianoroll error visualization...")
        plot_pianoroll_errors(
            pred_pianoroll=sample_data['pred_pianoroll'],
            target_pianoroll=sample_data['gt_pianoroll'],
            hop_seconds=hop_seconds,
            save_path=os.path.join(sample_path, "sample_pianoroll_ERRORS.png")
        )

    plot_tasks = [
        {'name': 'Activity', 'gt': sample_data.get('gt_activity_np'), 'pred': sample_data.get('pred_activity_np')},
        {'name': 'Onsets', 'gt': sample_data.get('gt_onset_np'), 'pred': sample_data.get('pred_onset_np')},
        {'name': 'Offsets', 'gt': sample_data.get('gt_offset_np'), 'pred': sample_data.get('pred_offset_np')},
    ]

    for task in plot_tasks:
        if task['gt'] is not None:
            plot_binary_activation(task['gt'].T, hop_seconds, os.path.join(sample_path, f"sample_{task['name'].lower()}_truth.png"), f"Ground Truth {task['name']}")
        if task['pred'] is not None:
            plot_binary_activation(task['pred'].T, hop_seconds, os.path.join(sample_path, f"sample_{task['name'].lower()}_pred.png"), f"Predicted {task['name']}")

    logger.info(f"Sample visualizations saved to: {sample_path}")
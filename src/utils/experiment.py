import torch
import yaml
import numpy as np
import os
import io
import shutil
import subprocess
import sys
import datetime 
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
from src.utils.agt_tools import tablature_to_stacked_multi_pitch, stacked_multi_pitch_to_multi_pitch
from src.utils.guitar_profile import GuitarProfile

logger = logging.getLogger(__name__)

def create_experiment_directory(base_output_path: str, model_name: str, config: dict, config_path: str) -> str:
    """
    Creates an experiment directory named with the format: ModelName_YYYYMMDD_HHMMSS
    Example: outputs/CNN_MTL_20231027_143000
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_name}_{timestamp}"
    
    pretrained_path = config.get('training', {}).get('pretrained_model_path')
    if pretrained_path and os.path.exists(pretrained_path):
        experiment_path = os.path.join(os.path.dirname(base_output_path), "finetune", dir_name)
    else:
        experiment_path = os.path.join(base_output_path, dir_name)
    
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
            f.write("OVERALL PARAMETER COUNT:\n" + "="*20 + "\n")
            f.write(f"Total params: {total_params:,}\n")
            f.write(f"Trainable params: {trainable_params:,}\n")
            
            # Bileşen bazlı parametre dökümü (CNN_MTL için)
            if hasattr(model, 'feature_branches') and hasattr(model, 'heads'):
                f.write("\nPARAMETER COUNT BY COMPONENT:\n" + "="*30 + "\n")
                backbone_params = sum(p.numel() for p in model.feature_branches.parameters())
                f.write(f"Backbone (Feature Branches): {backbone_params:,} ({backbone_params/total_params:.2%})\n")
                if hasattr(model, 'projections'):
                    proj_params = sum(p.numel() for p in model.projections.parameters())
                    f.write(f"Projections: {proj_params:,} ({proj_params/total_params:.2%})\n")
                for name, head in model.heads.items():
                    h_params = sum(p.numel() for p in head.parameters())
                    f.write(f"Head '{name}': {h_params:,} ({h_params/total_params:.2%})\n")
                    
    except Exception as e:
        logger.error(f"Could not generate model summary. Error: {e}")

def generate_experiment_report(model: torch.nn.Module, history: dict, val_loader: torch.utils.data.DataLoader, 
                             config: dict, experiment_path: str, device: torch.device, profile: GuitarProfile, include_silence, silence_class):
    logger.info("--- Generating Final Experiment Report ---")
    paths = _setup_report_directories(experiment_path)
    
    # 1. Geçmiş (History) Grafiklerini Çiz (DETAYLI)
    _plot_history_curves(history, paths)
    
    # 2. Örnek Veri ve Tahminleri Al (ESKİ GÜVENİLİR MANTIK)
    sample_data = _get_sample_data_and_predictions(model, val_loader, config, device, profile, include_silence, silence_class)
    
    # 3. Görselleştirmeleri Çiz (DETAYLI AUX VISUALS DAHİL)
    _plot_sample_visualizations(sample_data, paths, config)
    
    logger.info("Experiment report generation complete.")

def _setup_report_directories(experiment_path: str) -> dict:
    logger.info("Setting up report directories...")
    paths = {
        "charts": os.path.join(experiment_path, "charts"),
        "samples": os.path.join(experiment_path, "charts", "sample_outputs"),
        "loss": os.path.join(experiment_path, "charts", "loss"),
        
        # Main Task
        "tablature": os.path.join(experiment_path, "charts", "tablature"),
        
        # Aux Tasks (Ayrı klasörler)
        "hand_position": os.path.join(experiment_path, "charts", "hand_position"),
        "string_activity": os.path.join(experiment_path, "charts", "string_activity"),
        "pitch_class": os.path.join(experiment_path, "charts", "pitch_class"),
        
        # Extras
        "multi_pitch": os.path.join(experiment_path, "charts", "multi_pitch"),
        "octave": os.path.join(experiment_path, "charts", "octave_tolerated"),
        
        # Analysis
        "errors": os.path.join(experiment_path, "charts", "errors"),
        "tdr": os.path.join(experiment_path, "charts", "tdr")  
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def _plot_history_curves(history: dict, paths: dict):
    logger.info("Plotting all history curves...")
    
    plot_jobs = [
        # --- 1. Tablature (Main) ---
        {'key': 'tab_f1',        'title': 'Tablature F1 Score',     'dir': paths['tablature'], 'file': 'f1_score.png'},
        {'key': 'tab_precision', 'title': 'Tablature Precision',    'dir': paths['tablature'], 'file': 'precision.png'},
        {'key': 'tab_recall',    'title': 'Tablature Recall',       'dir': paths['tablature'], 'file': 'recall.png'},
        {'key': 'tdr',           'title': 'Tablature Disambiguation Rate (TDR)', 'dir': paths['tdr'], 'file': 'tdr_curve.png'},

        # --- 2. Hand Position (GÜNCELLENDİ) ---
        {'key': 'hand_pos_acc',       'title': 'Hand Position Accuracy',  'dir': paths['hand_position'], 'file': 'accuracy.png'},
        {'key': 'hand_pos_f1',        'title': 'Hand Position F1',        'dir': paths['hand_position'], 'file': 'f1_score.png'},
        {'key': 'hand_pos_precision', 'title': 'Hand Position Precision', 'dir': paths['hand_position'], 'file': 'precision.png'},
        {'key': 'hand_pos_recall',    'title': 'Hand Position Recall',    'dir': paths['hand_position'], 'file': 'recall.png'},
        
        # --- 3. String Activity (GÜNCELLENDİ) ---
        {'key': 'string_act_f1',        'title': 'String Activity F1',        'dir': paths['string_activity'], 'file': 'f1_score.png'},
        {'key': 'string_act_precision', 'title': 'String Activity Precision', 'dir': paths['string_activity'], 'file': 'precision.png'},
        {'key': 'string_act_recall',    'title': 'String Activity Recall',    'dir': paths['string_activity'], 'file': 'recall.png'},
        
        # --- 4. Pitch Class (GÜNCELLENDİ) ---
        {'key': 'pitch_class_f1',        'title': 'Pitch Class F1',        'dir': paths['pitch_class'], 'file': 'f1_score.png'},
        {'key': 'pitch_class_precision', 'title': 'Pitch Class Precision', 'dir': paths['pitch_class'], 'file': 'precision.png'},
        {'key': 'pitch_class_recall',    'title': 'Pitch Class Recall',    'dir': paths['pitch_class'], 'file': 'recall.png'},
        
        # --- 5. Multi-pitch (Extra) ---
        {'key': 'mp_f1',        'title': 'Multi-pitch F1 Score',  'dir': paths['multi_pitch'], 'file': 'f1_score.png'},
        {'key': 'mp_precision', 'title': 'Multi-pitch Precision', 'dir': paths['multi_pitch'], 'file': 'precision.png'},
        {'key': 'mp_recall',    'title': 'Multi-pitch Recall',    'dir': paths['multi_pitch'], 'file': 'recall.png'},

        # --- 6. Octave Tolerant (Extra) ---
        {'key': 'octave_f1',        'title': 'Octave Tolerant F1',        'dir': paths['octave'], 'file': 'f1_score.png'},
        {'key': 'octave_precision', 'title': 'Octave Tolerant Precision', 'dir': paths['octave'], 'file': 'precision.png'},
        {'key': 'octave_recall',    'title': 'Octave Tolerant Recall',    'dir': paths['octave'], 'file': 'recall.png'},
        
        # --- 7. Errors (Rates) ---
        {'key': 'tab_error_total',           'title': 'Total Error Rate',           'dir': paths['errors'], 'file': 'error_total.png'},
        {'key': 'tab_error_substitution',    'title': 'Substitution Error Rate',    'dir': paths['errors'], 'file': 'error_substitution.png'},
        {'key': 'tab_error_miss',            'title': 'Miss Error Rate (FN)',       'dir': paths['errors'], 'file': 'error_miss.png'},
        {'key': 'tab_error_false_alarm',     'title': 'False Alarm Error Rate (FP)','dir': paths['errors'], 'file': 'error_false_alarm.png'},
        {'key': 'tab_error_duplicate_pitch', 'title': 'Duplicate Pitch Error Rate', 'dir': paths['errors'], 'file': 'error_duplicate_pitch.png'}
    ]
    
    # Loss Plotting
    all_loss_keys = [k.replace('train_', '') for k in history.keys() if k.startswith('train_loss_')]
    for loss_key in all_loss_keys:
        friendly_name = loss_key.replace('loss_', '').replace('_', ' ').title()
        plot_jobs.append({
            'key': loss_key, 
            'title': f'{friendly_name} Loss', 
            'dir': paths['loss'], 
            'file': f'{loss_key}.png'
        })
    
    for job in plot_jobs:
        val_key, train_key = f"val_{job['key']}", f"train_{job['key']}"
        if (val_key in history and any(history[val_key])) or (train_key in history and any(history[train_key])):
            try:
                plot_metrics_custom(history, val_key, train_key, job['title'], os.path.join(job['dir'], job['file']))
            except Exception as e:
                logger.warning(f"Could not plot {job['key']}: {e}")

def _get_sample_data_and_predictions(model, val_loader, config, device, profile, include_silence, silence_class):
    logger.info("Generating a sample batch for visual comparison...")
    try:
        sample_batch = next(iter(val_loader))
    except StopIteration:
        return None

    model.eval()
    with torch.no_grad():
        features_dict = {key: tensor.to(device) for key, tensor in sample_batch['features'].items()}
        model_output = model(features_dict)
        
        # 1. Tablature Tahmini
        tab_logits = model_output.get('tab_logits')
        S, C = config['instrument']['num_strings'], config['model']['params']['num_classes']
        preds_tab = None
        if tab_logits is not None:
            if tab_logits.dim() == 3: 
                 preds_tab = torch.argmax(tab_logits, dim=-1)
            elif tab_logits.dim() == 2:
                 preds_tab = torch.argmax(tab_logits.view(-1, S, C), dim=-1)

        # 2. Aux Tahminleri
        preds_hp = torch.argmax(model_output.get('hand_pos_logits'), dim=-1) if 'hand_pos_logits' in model_output else None
        preds_act = (torch.sigmoid(model_output.get('activity_logits')) > 0.5).int() if 'activity_logits' in model_output else None
        preds_pc = (torch.sigmoid(model_output.get('pitch_class_logits')) > 0.5).int() if 'pitch_class_logits' in model_output else None

    # --- ESKİ VE GÜVENİLİR VERİ SEÇİMİ ---
    chunk_size = config['data']['framify_chunk_size']
    limit = min(chunk_size, sample_batch['tablature'].shape[0])
    
    def to_np(t): 
        if t is None: return None
        return t[:limit].cpu().numpy()

    gt_tab_np = to_np(sample_batch['tablature']).T 
    pred_tab_np = to_np(preds_tab).T if preds_tab is not None else None
    
    gt_act_np = to_np(sample_batch.get('activity_target')).T if 'activity_target' in sample_batch else None
    pred_act_np = to_np(preds_act).T if preds_act is not None else None
    
    gt_pc_np = to_np(sample_batch.get('pitch_class_target')).T if 'pitch_class_target' in sample_batch else None
    pred_pc_np = to_np(preds_pc).T if preds_pc is not None else None
    
    gt_hp_np = to_np(sample_batch.get('hand_pos_target')) # (Time,)
    pred_hp_np = to_np(preds_hp)

    # --- PIANO ROLL DÖNÜŞÜMÜ (agt_tools kullanır - GÜVENİLİR) ---
    gt_pianoroll, pred_pianoroll = None, None
    
    if gt_tab_np is not None:
        gt_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(gt_tab_np), profile, include_silence, silence_class)
        gt_pianoroll = stacked_multi_pitch_to_multi_pitch(gt_smp).numpy()
        
    if pred_tab_np is not None:
        pred_smp = tablature_to_stacked_multi_pitch(torch.from_numpy(pred_tab_np), profile, include_silence, silence_class)
        pred_pianoroll = stacked_multi_pitch_to_multi_pitch(pred_smp).numpy()

    return {
        "sample_batch": sample_batch, 
        "gt_tab_np": gt_tab_np, "pred_tab_np": pred_tab_np,
        "gt_act_np": gt_act_np, "pred_act_np": pred_act_np,
        "gt_pc_np": gt_pc_np, "pred_pc_np": pred_pc_np,
        "gt_hp_np": gt_hp_np, "pred_hp_np": pred_hp_np,
        "gt_pianoroll": gt_pianoroll, "pred_pianoroll": pred_pianoroll,
    }

def _plot_sample_visualizations(sample_data: dict, paths: dict, config: dict):
    if not sample_data: return
    logger.info("Plotting sample visualizations...")
    sample_path = paths['samples']
    hop_seconds = config['data']['hop_length'] / config['data']['sample_rate']
    silence_class = config['instrument']['silence_class']
    min_midi_val = config['instrument']['min_midi']

    # 1. Tablature
    if sample_data['gt_tab_np'] is not None:
        plot_guitar_tablature(sample_data['gt_tab_np'], hop_seconds, os.path.join(sample_path, "sample_tab_truth.png"), "Ground Truth Tablature")
    if sample_data['pred_tab_np'] is not None:
        plot_guitar_tablature(sample_data['pred_tab_np'], hop_seconds, os.path.join(sample_path, "sample_tab_pred.png"), "Predicted Tablature")
    
    # 2. Pianoroll 
    if sample_data['gt_pianoroll'] is not None:
        plot_pianoroll(sample_data['gt_pianoroll'], hop_seconds, os.path.join(sample_path, "sample_pianoroll_truth.png"), "Ground Truth Pianoroll", min_midi_val)
    if sample_data['pred_pianoroll'] is not None:
        plot_pianoroll(sample_data['pred_pianoroll'], hop_seconds, os.path.join(sample_path, "sample_pianoroll_pred.png"), "Prediction Pianoroll", min_midi_val)
    
    # 3. Tablature Errors 
    if sample_data.get('gt_tab_np') is not None and sample_data.get('pred_tab_np') is not None:
        plot_tablature_errors(
            preds_np=sample_data['pred_tab_np'], 
            targets_np=sample_data['gt_tab_np'],
            hop_seconds=hop_seconds,
            silence_class=silence_class,
            save_path=os.path.join(sample_path, "sample_tab_ERRORS.png")
        )

    # 4. Aux Visuals (YENİ VE KAPSAMLI)
    if sample_data.get('gt_act_np') is not None:
        plot_binary_activation(sample_data['gt_act_np'], hop_seconds, os.path.join(sample_path, "sample_string_act_truth.png"), "GT String Activity")
    if sample_data.get('pred_act_np') is not None:
        plot_binary_activation(sample_data['pred_act_np'], hop_seconds, os.path.join(sample_path, "sample_string_act_pred.png"), "Pred String Activity")
        
    if sample_data.get('gt_pc_np') is not None:
        plot_binary_activation(sample_data['gt_pc_np'], hop_seconds, os.path.join(sample_path, "sample_chroma_truth.png"), "GT Pitch Class")
    if sample_data.get('pred_pc_np') is not None:
        plot_binary_activation(sample_data['pred_pc_np'], hop_seconds, os.path.join(sample_path, "sample_chroma_pred.png"), "Pred Pitch Class")

    # Hand Position (Isı Haritası Mantığıyla Çizim)
    num_hp_classes = 5
    def hp_to_matrix(hp_arr):
        if hp_arr is None: return None
        T = hp_arr.shape[0]
        mat = np.zeros((num_hp_classes, T))
        for t in range(T):
            c = hp_arr[t]
            if 0 <= c < num_hp_classes:
                mat[c, t] = 1
        return mat

    gt_hp_mat = hp_to_matrix(sample_data.get('gt_hp_np'))
    pred_hp_mat = hp_to_matrix(sample_data.get('pred_hp_np'))
    
    if gt_hp_mat is not None:
        plot_binary_activation(gt_hp_mat, hop_seconds, os.path.join(sample_path, "sample_hand_pos_truth.png"), "GT Hand Position (Regions)")
    if pred_hp_mat is not None:
        plot_binary_activation(pred_hp_mat, hop_seconds, os.path.join(sample_path, "sample_hand_pos_pred.png"), "Pred Hand Position (Regions)")

    logger.info(f"Sample visualizations saved to: {sample_path}")
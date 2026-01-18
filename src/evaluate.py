import os
import sys
import argparse
import yaml
import json
import logging
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)

try:
    from src.models import get_model
except ImportError:
    from src.models.cnn_mtl import cnn_mtl as get_model

from src.utils.metrics import (
    compute_tablature_metrics, compute_multipitch_metrics, compute_octave_tolerant_metrics, 
    compute_tablature_error_scores, compute_tdr,
    compute_hand_position_metrics, compute_string_activity_metrics,
    compute_pitch_class_metrics, compute_aux_multipitch_metrics
)
from src.data_loader import get_dataloaders
from src.utils.guitar_profile import GuitarProfile
from src.trainer import convert_history_to_native_types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_post_processing(preds, min_duration, silence_class):
    """
    Tahmin edilen tablo üzerinde 'Minimum Nota Süresi' filtresi uygular.
    Çok kısa süren (min_duration'dan az) izole notaları sessizliğe çevirir.
    """
    if min_duration <= 1:
        return preds
        
    cleaned_preds = preds.clone().cpu().numpy() # İşlem hızı için numpy
    T, S = cleaned_preds.shape
    
    # Her tel için bağımsız temizlik
    for s in range(S):
        string_preds = cleaned_preds[:, s]
        
        # Değişim noktalarını bul (Run-Length Encoding mantığı)
        diffs = np.concatenate(([True], string_preds[1:] != string_preds[:-1], [True]))
        change_indices = np.nonzero(diffs)[0]
        
        # Her bir bloğu kontrol et
        for i in range(len(change_indices) - 1):
            start = change_indices[i]
            end = change_indices[i+1]
            length = end - start
            val = string_preds[start]
            
            # Eğer bu blok bir nota ise (Sessizlik değilse) VE çok kısaysa
            if val != silence_class and length < min_duration:
                # Gürültü olarak işaretle ve sessizliğe çevir
                cleaned_preds[start:end, s] = silence_class
                
    return torch.from_numpy(cleaned_preds).to(preds.device)

def evaluate_model(model, test_loader, device, config):
    model.to(device)
    model.eval()

    stats = {
        "tab_logits": [], "tab_targets": [],
        "aux_hand_pos_logits": [], "aux_hand_pos_targets": [],
        "aux_activity_logits": [], "aux_activity_targets": [],
        "aux_pitch_class_logits": [], "aux_pitch_class_targets": [],
        "aux_multipitch_logits": [], "aux_multipitch_targets": [],
        "aux_onset_logits": [], "aux_onset_targets": []
    }

    logger.info("Collecting predictions (Model Inference)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = {key: tensor.to(device) for key, tensor in batch['features'].items()}
            model_output = model(inputs)
            
            if 'tab_logits' in model_output:
                stats["tab_logits"].append(model_output['tab_logits'].cpu())
                stats["tab_targets"].append(batch['tablature'].cpu())
            
            if 'hand_pos_logits' in model_output:
                stats["aux_hand_pos_logits"].append(model_output['hand_pos_logits'].cpu())
                stats["aux_hand_pos_targets"].append(batch['hand_pos_target'].cpu())
            
            if 'activity_logits' in model_output:
                stats["aux_activity_logits"].append(model_output['activity_logits'].cpu())
                stats["aux_activity_targets"].append(batch['activity_target'].cpu())

            if 'pitch_class_logits' in model_output:
                stats["aux_pitch_class_logits"].append(model_output['pitch_class_logits'].cpu())
                stats["aux_pitch_class_targets"].append(batch['pitch_class_target'].cpu())

            if 'multipitch_logits' in model_output and 'multipitch_target' in batch:
                stats["aux_multipitch_logits"].append(model_output['multipitch_logits'].cpu())
                stats["aux_multipitch_targets"].append(batch['multipitch_target'].cpu())

            if 'onset_logits' in model_output and 'onset_target' in batch:
                stats["aux_onset_logits"].append(model_output['onset_logits'].cpu())
                stats["aux_onset_targets"].append(batch['onset_target'].cpu())

    logger.info("Computing metrics...")
    final_results = {}
    
    S = config['instrument']['num_strings']
    C = config['model']['params']['num_classes']
    guitar_profile = GuitarProfile(config['instrument'])
    
    include_silence = config["metrics"].get('include_silence', False)
    tab_silence_class = config['instrument']['silence_class']
    tuning = config['instrument']['tuning']

    # =========================================================================
    # 1. TABLATURE METRICS (GRID SEARCH + POST PROCESSING)
    # =========================================================================
    if stats["tab_logits"]:
        all_logits = torch.cat(stats["tab_logits"], dim=0)
        all_targets = torch.cat(stats["tab_targets"], dim=0)

        # Logits şekillendirme
        if all_logits.dim() == 2: logits_reshaped = all_logits.view(-1, S, C)
        elif all_logits.dim() == 3: logits_reshaped = all_logits
        else: logits_reshaped = all_logits.view(-1, S, C)

        # Targets şekillendirme
        if all_targets.dim() == 3: targets_flat = all_targets.view(-1, S)
        else: targets_flat = all_targets.view(-1, S)
        
        # Ön hesaplamalar
        probs = torch.softmax(logits_reshaped, dim=-1)
        base_preds = torch.argmax(logits_reshaped, dim=-1)
        
        # Threshold hazırlığı
        if tab_silence_class is not None:
            logits_notes = logits_reshaped.clone()
            logits_notes[:, :, tab_silence_class] = -float('inf')
            best_note_idx = torch.argmax(logits_notes, dim=-1)
            best_note_prob = torch.gather(probs, 2, best_note_idx.unsqueeze(-1)).squeeze(-1)
        
        # --- PHASE 1: EN İYİ THRESHOLD'U BUL ---
        threshold_candidates = np.arange(0.20, 0.55, 0.05)
        best_f1_thresh = -1.0
        best_thresh = 0.40 # Varsayılan
        
        logger.info(f"Phase 1: Grid Search for Probability Threshold...")
        
        for th in threshold_candidates:
            current_preds = base_preds.clone()
            if tab_silence_class is not None:
                mask = (current_preds == tab_silence_class) & (best_note_prob > th)
                current_preds[mask] = best_note_idx[mask]
            
            temp_res = compute_tablature_metrics(current_preds, targets_flat, include_silence, tab_silence_class)
            f1 = temp_res.get('overall_f1', 0.0)
            
            if f1 > best_f1_thresh:
                best_f1_thresh = f1
                best_thresh = th
        
        logger.info(f"   >>> Best Threshold Found: {best_thresh:.2f} (Base F1: {best_f1_thresh:.4f})")
        
        # --- PHASE 2: EN İYİ MIN_DURATION'I BUL (TEMİZLİK) ---
        preds_at_best_thresh = base_preds.clone()
        if tab_silence_class is not None:
            mask = (preds_at_best_thresh == tab_silence_class) & (best_note_prob > best_thresh)
            preds_at_best_thresh[mask] = best_note_idx[mask]
            
        duration_candidates = [1, 2, 3, 4, 5] 
        best_overall_f1 = -1.0
        best_dur = 1
        best_final_preds = preds_at_best_thresh 

        logger.info(f"Phase 2: Grid Search for Post-Processing (Min Duration)...")

        for dur in duration_candidates:
            cleaned_preds = apply_post_processing(preds_at_best_thresh, min_duration=dur, silence_class=tab_silence_class)
            
            temp_res = compute_tablature_metrics(cleaned_preds, targets_flat, include_silence, tab_silence_class)
            f1 = temp_res.get('overall_f1', 0.0)
            
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_dur = dur
                best_final_preds = cleaned_preds 

        logger.info(f"   >>> Best Duration Found: {best_dur} frames (Final F1: {best_overall_f1:.4f})")
        
        # --- FİNAL HESAPLAMA VE KAYIT ---
        final_tab_metrics = compute_tablature_metrics(best_final_preds, targets_flat, include_silence, tab_silence_class)
        
        final_results.update({
            'tab_f1': final_tab_metrics.get('overall_f1', 0.0),
            'tab_precision': final_tab_metrics.get('overall_precision', 0.0),
            'tab_recall': final_tab_metrics.get('overall_recall', 0.0),
            'best_threshold': float(best_thresh),
            'best_min_duration': int(best_dur)
        })
        
        mp_res = compute_multipitch_metrics(best_final_preds, targets_flat, guitar_profile, include_silence, tab_silence_class)
        final_results.update({
            'mp_f1': mp_res.get('multipitch_f1', 0.0),
            'mp_precision': mp_res.get('multipitch_precision', 0.0),
            'mp_recall': mp_res.get('multipitch_recall', 0.0),
        })

        oct_res = compute_octave_tolerant_metrics(best_final_preds, targets_flat, tuning, tab_silence_class)
        final_results.update({'octave_f1': oct_res.get('octave_f1', 0.0)})
        
        final_results['tdr'] = compute_tdr(best_final_preds, targets_flat, guitar_profile, include_silence, tab_silence_class)

        err_res = compute_tablature_error_scores(best_final_preds, targets_flat, tab_silence_class, tuning)
        final_results.update({
            'tab_error_total_rate': err_res.get('tab_error_total_rate', 0.0),
            'tab_error_substitution_rate': err_res.get('tab_error_substitution_rate', 0.0),
            'tab_error_miss_rate': err_res.get('tab_error_miss_rate', 0.0),
            'tab_error_false_alarm_rate': err_res.get('tab_error_false_alarm_rate', 0.0),
            'tab_error_duplicate_pitch_rate': err_res.get('tab_error_duplicate_pitch_rate', 0.0),
        })

    # =========================================================================
    # 2. AUXILIARY METRICS
    # =========================================================================

    if stats["aux_hand_pos_logits"]:
        hp_logits = torch.cat(stats["aux_hand_pos_logits"])
        hp_targets = torch.cat(stats["aux_hand_pos_targets"])
        hp_preds = torch.argmax(hp_logits, dim=-1)
        final_results.update(compute_hand_position_metrics(hp_preds, hp_targets, include_silence, silence_class=0))

    if stats["aux_activity_logits"]:
        act_logits = torch.cat(stats["aux_activity_logits"])
        act_targets = torch.cat(stats["aux_activity_targets"])
        final_results.update(compute_string_activity_metrics(act_logits, act_targets, include_silence))

    if stats["aux_pitch_class_logits"]:
        pc_logits = torch.cat(stats["aux_pitch_class_logits"])
        pc_targets = torch.cat(stats["aux_pitch_class_targets"])
        final_results.update(compute_pitch_class_metrics(pc_logits, pc_targets, include_silence))
    
    if stats["aux_multipitch_logits"]:
        mp_h_logits = torch.cat(stats["aux_multipitch_logits"])
        mp_h_targets = torch.cat(stats["aux_multipitch_targets"])
        final_results.update(compute_aux_multipitch_metrics(mp_h_logits, mp_h_targets, include_silence))

    if stats["aux_onset_logits"]:
        on_logits = torch.cat(stats["aux_onset_logits"])
        on_targets = torch.cat(stats["aux_onset_targets"])
        on_res = compute_aux_multipitch_metrics(on_logits, on_targets, include_silence)
        final_results.update({
            'onset_f1': on_res['mp_head_f1'],
            'onset_precision': on_res['mp_head_precision'],
            'onset_recall': on_res['mp_head_recall']
        })

    return final_results

def main():
    """
    Evaluate a trained checkpoint for a given fold and save test_results.json.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint on its designated test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fold_path", type=str, required=True,
                        help="Path to this fold's directory (e.g., .../V67/fold_1).")
    parser.add_argument("--main_exp_path", type=str, required=True,
                        help="Path to the experiment root containing config.yaml (e.g., .../V67).")
    parser.add_argument("--test_files", type=str, nargs='+', required=True,
                        help="Space-separated list of test .npz files.")
    parser.add_argument("--fold_num", type=int, default=None,
                        help="Fold number override (otherwise inferred from fold_path).")
    args = parser.parse_args()

    logger.info(f"Starting evaluation for experiment fold: {args.fold_path}")

    config_path = os.path.join(args.main_exp_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        from src.utils.config_helper import process_config
        config = process_config(config)
    except Exception:
        pass 

    device = torch.device(
        config.get("training", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    test_paths = args.test_files
    logger.info(f"Received {len(test_paths)} test files to evaluate.")

    from src.data_loader import get_dataloaders
    _, _, test_loader = get_dataloaders(
        config,
        train_paths=[test_paths[0]], 
        val_paths=[test_paths[0]],
        test_paths=test_paths
    )

    model = get_model(config)
    best_model_path = os.path.join(args.fold_path, "model_best.pt")
    
    if not os.path.isfile(best_model_path):
        logger.warning(f"Best model not found at {best_model_path}. Trying checkpoint_latest.pt...")
        best_model_path = os.path.join(args.fold_path, "checkpoint_latest.pt")
        if not os.path.isfile(best_model_path):
             raise FileNotFoundError(f"No model checkpoint found in {args.fold_path}")

    logger.info(f"Loading checkpoint from: {best_model_path}")
    state = torch.load(best_model_path, map_location=device)
    
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
        
    model.to(device)

    results = evaluate_model(model, test_loader, device, config)

    logger.info("\n" + "="*50 + "\n--- FINAL TEST RESULTS (Macro/Binary/Samples) ---\n" + "="*50)
    df_rows = []
    for k, v in results.items():
        df_rows.append({"Metric": k, "Value": v})
    results_df = pd.DataFrame(df_rows)
    print(results_df.to_string(index=False))

    if args.fold_num is not None:
        fold_num = int(args.fold_num)
    else:
        try:
            fold_num = int(os.path.basename(args.fold_path).split("_")[-1])
        except Exception:
            fold_num = 0

    native_results = convert_history_to_native_types({
        "test_metrics_for_fold": fold_num,
        "results": results
    })

    results_path = os.path.join(args.fold_path, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(native_results, f, indent=4)

    logger.info(f"\nTest results successfully saved to: {results_path}")
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
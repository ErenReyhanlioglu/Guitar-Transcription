import os
import sys
import argparse
import yaml
import json
import logging
import torch
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)
    print(f"Added project root to path: {PROJECT_ROOT_PATH}")

from src.models import get_model

from src.utils.metrics import (
    compute_tablature_metrics, compute_multipitch_metrics,
    compute_octave_tolerant_metrics, compute_tablature_error_scores, compute_tdr
)

from src.data_loader import prepare_dataset_files, get_dataloaders
from src.utils.guitar_profile import GuitarProfile
from src.trainer import convert_history_to_native_types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device, config):
    """
    Evaluates a given model on a test data loader and computes all relevant metrics.

    This function contains the core evaluation logic, mirroring the `_calculate_all_metrics`
    method in the Trainer class to ensure consistency.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (DataLoader): The DataLoader for the test set.
        device (torch.device): The device (CPU or CUDA) to run evaluation on.
        config (dict): The experiment configuration dictionary.

    Returns:
        dict: A dictionary containing all computed test metrics.
    """
    model.to(device)
    model.eval()

    all_logits_list = []
    all_targets_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Model on Test Set", leave=False):
            inputs = {key: tensor.to(device) for key, tensor in batch['features'].items()}
            targets = batch['tablature'].to(device)
            
            model_output = model(**inputs)
            logits = model_output['tab_logits']
            
            all_logits_list.append(logits.cpu())
            all_targets_list.append(targets.cpu())

    all_logits = torch.cat(all_logits_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)
    
    S = config['instrument']['num_strings']
    C = config['model']['params']['num_classes']
    
    if config['loss']['active_loss'] == 'softmax_groups':
        if all_logits.dim() == 4:
            preds_tab = torch.argmax(all_logits.permute(0, 2, 1, 3), dim=-1)
        elif all_logits.dim() == 2:
            preds_tab = torch.argmax(all_logits.view(-1, S, C), dim=-1)
        else:
                raise ValueError(f"Unsupported logits dimension for metrics: {all_logits.dim()}")
    else:
        raise NotImplementedError(f"Prediction logic for loss type '{config['loss']['active_loss']}' is not implemented.")

    if all_targets.dim() == 3:
        targets_flat = all_targets.permute(0, 2, 1).reshape(-1, S)
    else:
        targets_flat = all_targets
        
    if preds_tab.dim() == 3:
        preds_tab_flat = preds_tab.permute(0, 2, 1).reshape(-1, S)
    else:
        preds_tab_flat = preds_tab

    guitar_profile = GuitarProfile(config['instrument'])
    silence_class = config['instrument']['silence_class']
    tuning = config['instrument']['tuning']
    
    tab_metrics_raw = compute_tablature_metrics(preds_tab_flat, targets_flat, False, silence_class)
    mp_metrics_raw = compute_multipitch_metrics(preds_tab_flat, targets_flat, guitar_profile)
    octave_metrics_raw = compute_octave_tolerant_metrics(preds_tab_flat, targets_flat, tuning, silence_class)
    error_scores_raw = compute_tablature_error_scores(preds_tab_flat, targets_flat, silence_class, tuning)
    
    tdr_score = compute_tdr(preds_tab_flat, targets_flat, guitar_profile)

    test_results = {
        # Tablature Metrics (Micro, Weighted, Macro)
        'tab_f1': tab_metrics_raw.get('overall_f1', 0.0),
        'tab_precision': tab_metrics_raw.get('overall_precision', 0.0),
        'tab_recall': tab_metrics_raw.get('overall_recall', 0.0),
        'tab_f1_weighted': tab_metrics_raw.get('overall_f1_weighted', 0.0),
        'tab_precision_weighted': tab_metrics_raw.get('overall_precision_weighted', 0.0),
        'tab_recall_weighted': tab_metrics_raw.get('overall_recall_weighted', 0.0),
        'tab_f1_macro': tab_metrics_raw.get('overall_f1_macro', 0.0),
        'tab_precision_macro': tab_metrics_raw.get('overall_precision_macro', 0.0),
        'tab_recall_macro': tab_metrics_raw.get('overall_recall_macro', 0.0),
        
        # Multi-pitch Metrics
        'mp_f1': mp_metrics_raw.get('multipitch_f1', 0.0),
        'mp_precision': mp_metrics_raw.get('multipitch_precision', 0.0),
        'mp_recall': mp_metrics_raw.get('multipitch_recall', 0.0),
        
        # Octave Tolerant Metrics (Micro, Weighted, Macro)
        'octave_f1': octave_metrics_raw.get('octave_f1', 0.0),
        'octave_precision': octave_metrics_raw.get('octave_precision', 0.0),
        'octave_recall': octave_metrics_raw.get('octave_recall', 0.0),
        'octave_f1_weighted': octave_metrics_raw.get('octave_f1_weighted', 0.0),
        'octave_precision_weighted': octave_metrics_raw.get('octave_precision_weighted', 0.0),
        'octave_recall_weighted': octave_metrics_raw.get('octave_recall_weighted', 0.0),
        'octave_f1_macro': octave_metrics_raw.get('octave_f1_macro', 0.0),
        'octave_precision_macro': octave_metrics_raw.get('octave_precision_macro', 0.0),
        'octave_recall_macro': octave_metrics_raw.get('octave_recall_macro', 0.0),

        # Tablature Error Scores
        'tab_error_total_rate': error_scores_raw.get('tab_error_total_rate', 0.0),
        'tab_error_substitution_rate': error_scores_raw.get('tab_error_substitution_rate', 0.0),
        'tab_error_miss_rate': error_scores_raw.get('tab_error_miss_rate', 0.0),
        'tab_error_false_alarm_rate': error_scores_raw.get('tab_error_false_alarm_rate', 0.0),
        'tab_error_duplicate_pitch_rate': error_scores_raw.get('tab_error_duplicate_pitch_rate', 0.0),
        'tab_error_total_count': error_scores_raw.get('tab_error_total_count', 0),
        'tab_error_substitution_count': error_scores_raw.get('tab_error_substitution_count', 0),
        'tab_error_miss_count': error_scores_raw.get('tab_error_miss_count', 0),
        'tab_error_false_alarm_count': error_scores_raw.get('tab_error_false_alarm_count', 0),
        'tab_error_duplicate_pitch_count': error_scores_raw.get('tab_error_duplicate_pitch_count', 0),

        # TDR
        'tdr': tdr_score,
    }
    
    return test_results

def main():
    """
    Evaluate a trained checkpoint for a given fold and save test_results.json.
    """
    import argparse, os, json
    from pathlib import Path
    import torch
    import pandas as pd

    # ---- CLI ----
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

    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        from src.utils.config_helper import process_config
    except Exception:
        from src.utils.config_helpers import process_config  # fallback
    config = process_config(config)

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

    from src.models import get_model
    model = get_model(config)
    best_model_path = os.path.join(args.fold_path, "model_best.pt")
    logger.info(f"Loading best model checkpoint from: {best_model_path}")
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best model checkpoint not found at: {best_model_path}")

    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    try:
        results = evaluate_model(model, test_loader, device, config)
    except NameError:
        from src.evaluate import evaluate_model as _eval_fn
        results = _eval_fn(model, test_loader, device, config)

    logger.info("\n" + "="*50 + "\n--- FINAL TEST RESULTS ---\n" + "="*50)
    results_df = pd.DataFrame([results])
    print(results_df.to_string())

    if args.fold_num is not None:
        fold_num = int(args.fold_num)
    else:
        try:
            fold_num = int(Path(args.fold_path).name.split("_")[-1])
        except Exception:
            fold_num = None  

    from src.trainer import convert_history_to_native_types
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
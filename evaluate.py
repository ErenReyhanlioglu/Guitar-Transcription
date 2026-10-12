"""CLI entry point for evaluating a trained k-fold experiment.

This is the paper's actual reported-results pipeline. Per fold it loads
fold_<n>/model_best.pt, runs whole-track inference (its own manual windowing,
not the training DataLoader), grid-searches the silence-recovery probability
threshold on the *validation* player only, then applies the winning threshold
to the held-out *test* player and computes the paper's metrics. Aggregates all
folds into k-fold_test_results.txt inside the experiment directory.

Ported from the Colab notebook's post-training re-evaluation pass, which
replaced an earlier evaluation path that picked the threshold directly against
the test set (a train/test leak) instead of the validation player.

Example:
    python evaluate.py --experiment-path outputs/cnn_mtl_20251210_113013 \\
        --data-root /path/to/guitarset_enriched_npz
"""
import argparse
import json
import logging
import math
import os

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from typing import cast

from src.data_loader import fold_split, prepare_dataset_files
from src.models import get_model
from src.trainer import convert_history_to_native_types
from src.utils.config_helpers import process_config, resolve_device
from src.utils.guitar_profile import GuitarProfile
from src.utils.logger import setup_logger
from src.utils.metrics import (
    compute_aux_multipitch_metrics,
    compute_hand_position_metrics,
    compute_multipitch_metrics,
    compute_pitch_class_metrics,
    compute_string_activity_metrics,
    compute_tablature_error_scores,
    compute_tablature_metrics,
    compute_tdr,
    string_assignment_breakdown,
)

logger = logging.getLogger(__name__)


# (model output key, short name, array in the .npz file) for the auxiliary heads scored on test
AUX_OUTPUTS = [("hand_pos_logits", "hp", "hand_position"),
               ("activity_logits", "sa", "string_activity"),
               ("pitch_class_logits", "pc", "pitch_class")]


def full_track_inference(paths: list[str], model, config: dict, device: torch.device) -> dict:
    """Runs inference over whole tracks (not fixed-size training chunks).

    Slides the model's context window across each full track in CHUNK-sized
    pieces, unfolding each piece into the model's expected window batches.
    Padded frames at the end of a track's last chunk are dropped.

    Returns:
        A dict of frame-aligned tensors: 'tab' (tablature logits, frames x strings
        x classes) and 'tgt' (reference frets), plus for each auxiliary head the
        model has, its logits and reference: 'mp'/'mp_tgt' (multi-pitch),
        'hp'/'hp_tgt' (fret region), 'sa'/'sa_tgt' (string activity),
        'pc'/'pc_tgt' (pitch class). Missing heads map to None.
    """
    chunk_size = config["data"].get("framify_chunk_size", 500)
    window_size = config["data"].get("framify_window_size", 19)
    pad_amount = window_size // 2
    silence_class = config["instrument"]["silence_class"]
    feature_to_file_map = config["data"].get("feature_to_file_map", {})
    raw_features = config["data"]["active_features"]
    active_features = [k for k, v in raw_features.items() if v] if isinstance(raw_features, dict) else raw_features
    min_midi = config["instrument"]["min_midi"]
    max_midi = config["instrument"]["max_midi"]
    num_pitches = max_midi - min_midi + 1
    tuning = np.array(config["instrument"]["tuning"])
    num_strings = config["instrument"]["num_strings"]
    num_classes = config["model"]["params"]["num_classes"]

    store = {k: [] for k in ("tab", "tgt", "mp", "mp_tgt", "hp", "hp_tgt", "sa", "sa_tgt", "pc", "pc_tgt")}

    model.eval()
    with torch.no_grad():
        for path in paths:
            data = np.load(path, allow_pickle=True)
            first_key = feature_to_file_map.get(active_features[0], active_features[0])
            total_frames = data[first_key].shape[-1]
            tablature = data["tablature"].astype(np.int64)
            tablature[tablature == -1] = silence_class
            aux_full = {short: np.asarray(data[key]) for _, short, key in AUX_OUTPUTS if key in data}

            num_chunks = math.ceil(total_frames / chunk_size)
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total_frames)
                pad_len = chunk_size - (end - start)

                features = {}
                for feature_key in active_features:
                    npz_key = feature_to_file_map.get(feature_key, feature_key)
                    feature_chunk = data[npz_key].astype(np.float32)[..., start:end]
                    if pad_len > 0:
                        feature_chunk = np.pad(feature_chunk, [(0, 0)] * (feature_chunk.ndim - 1) + [(0, pad_len)])
                    tensor = torch.from_numpy(feature_chunk).unsqueeze(0)
                    tensor = F.pad(tensor, (pad_amount, pad_amount), "constant", 0)
                    tensor = tensor.unfold(3, window_size, 1).permute(0, 3, 1, 2, 4)
                    b, t, ch, freq, w = tensor.shape
                    features[feature_key] = tensor.reshape(b * t, ch, freq, w).to(device)

                output = model(features)
                valid = end - start
                store["tab"].append(output["tab_logits"].view(-1, num_strings, num_classes).float().cpu()[:valid])
                tab_chunk = tablature[:, start:end]
                store["tgt"].append(torch.from_numpy(np.ascontiguousarray(tab_chunk.T)))

                if "multipitch_logits" in output:
                    store["mp"].append(output["multipitch_logits"].float().cpu()[:valid])
                    mp_target = np.zeros((valid, num_pitches), dtype=np.float32)
                    for s in range(tab_chunk.shape[0]):
                        frets = tab_chunk[s]
                        midi = tuning[s] + frets
                        on = (frets != silence_class) & (midi >= min_midi) & (midi <= max_midi)
                        mp_target[np.where(on)[0], midi[on] - min_midi] = 1.0
                    store["mp_tgt"].append(torch.from_numpy(mp_target))

                for out_key, short, _ in AUX_OUTPUTS:
                    if out_key in output and short in aux_full:
                        y = aux_full[short][..., start:end]
                        store[short].append(output[out_key].float().cpu()[:valid])
                        store[short + "_tgt"].append(torch.from_numpy(np.ascontiguousarray(y.T if y.ndim == 2 else y)))

            data.close()

    return {k: (torch.cat(v, 0) if v else None) for k, v in store.items()}


def find_best_postproc(logits: torch.Tensor, targets: torch.Tensor, config: dict) -> tuple[float, int]:
    """Grid-searches the silence-recovery probability threshold (0.20-0.50, step 0.05).

    The inner `for dur in ...` loop is carried over unchanged from the
    original notebook: it does not actually apply any duration-based
    filtering (that only happens in evaluate_fold), so every dur value
    scores identically per threshold and best_dur always resolves to 1.
    Left exactly as-is intentionally — this is the code that produced the
    paper's published numbers, and "fixing" it would change them.
    """
    silence_class = config["instrument"]["silence_class"]

    probs = torch.softmax(logits, dim=-1)
    base_preds = torch.argmax(logits, dim=-1)

    logits_notes = logits.clone()
    logits_notes[:, :, silence_class] = -float("inf")
    best_note_idx = torch.argmax(logits_notes, dim=-1)
    best_note_prob = torch.gather(probs, 2, best_note_idx.unsqueeze(-1)).squeeze(-1)

    best_thresh, best_dur, best_f1 = 0.40, 1, -1.0

    for threshold in np.arange(0.20, 0.55, 0.05):
        preds = base_preds.clone()
        mask = (preds == silence_class) & (best_note_prob > threshold)
        preds[mask] = best_note_idx[mask]
        for duration in (1, 2, 3, 4, 5):
            result = compute_tablature_metrics(preds, targets, False, silence_class)
            f1 = result.get("overall_f1", 0.0)
            if f1 > best_f1:
                best_f1, best_thresh, best_dur = f1, float(threshold), duration

    return best_thresh, best_dur


def apply_silence_threshold(tab_logits: torch.Tensor, threshold: float, silence_class: int) -> torch.Tensor:
    """Argmax per string, then reassigns silence to the most probable fret when that fret's
    probability exceeds `threshold`."""
    probs = torch.softmax(tab_logits, dim=-1)
    preds = torch.argmax(tab_logits, dim=-1)
    logits_notes = tab_logits.clone()
    logits_notes[:, :, silence_class] = -float("inf")
    best_note_idx = torch.argmax(logits_notes, dim=-1)
    best_note_prob = torch.gather(probs, 2, best_note_idx.unsqueeze(-1)).squeeze(-1)
    mask = (preds == silence_class) & (best_note_prob > threshold)
    preds[mask] = best_note_idx[mask]
    return preds


def evaluate_fold(inference: dict, threshold: float, min_duration: int, config: dict) -> dict:
    """Applies the silence-recovery threshold and minimum-duration filter, then scores predictions.

    Recovers low-confidence silence predictions above `threshold`, drops
    predicted note segments shorter than `min_duration` frames back to
    silence, and computes the paper's full metric set: tablature, TDR,
    multi-pitch, the error-rate breakdown, the wrong-string breakdown, and the
    accuracy of every auxiliary head present in `inference` (see
    `full_track_inference`).
    """
    tab_logits, targets = inference["tab"], inference["tgt"]
    silence_class = config["instrument"]["silence_class"]
    tuning = config["instrument"]["tuning"]
    profile = GuitarProfile(config["instrument"])

    preds = apply_silence_threshold(tab_logits, threshold, silence_class)

    preds_np = preds.numpy()
    for s in range(preds_np.shape[1]):
        col = preds_np[:, s]
        labels, num_labels = cast(tuple[np.ndarray, int], ndimage.label(col != silence_class))
        for label in range(1, num_labels + 1):
            idx = np.where(labels == label)[0]
            if len(idx) < min_duration:
                preds_np[idx, s] = silence_class
    preds = torch.from_numpy(preds_np)

    tab_res = compute_tablature_metrics(preds, targets, False, silence_class)
    mp_res = compute_multipitch_metrics(preds, targets, profile, False, silence_class)
    tdr_val = compute_tdr(preds, targets, profile, False, silence_class)
    err_res = compute_tablature_error_scores(preds, targets, silence_class, tuning)

    result = {
        "tab_f1": tab_res.get("overall_f1", 0.0),
        "tab_precision": tab_res.get("overall_precision", 0.0),
        "tab_recall": tab_res.get("overall_recall", 0.0),
        "tdr": tdr_val,
        "tab_mp_f1": mp_res.get("multipitch_f1", 0.0),
        "tab_mp_precision": mp_res.get("multipitch_precision", 0.0),
        "tab_mp_recall": mp_res.get("multipitch_recall", 0.0),
        "tab_error_total_rate": err_res.get("tab_error_total_rate", 0.0),
        "tab_error_substitution_rate": err_res.get("tab_error_substitution_rate", 0.0),
        "tab_error_miss_rate": err_res.get("tab_error_miss_rate", 0.0),
        "tab_error_false_alarm_rate": err_res.get("tab_error_false_alarm_rate", 0.0),
        "tab_error_duplicate_pitch_rate": err_res.get("tab_error_duplicate_pitch_rate", 0.0),
        "best_threshold": threshold,
        "best_min_duration": min_duration,
        "n_test_frames": int(targets.shape[0]),
    }

    if inference["mp"] is not None:
        aux_res = compute_aux_multipitch_metrics(inference["mp"], inference["mp_tgt"], False)
        result.update({
            "aux_mp_f1": aux_res.get("mp_head_f1", 0.0),
            "aux_mp_precision": aux_res.get("mp_head_precision", 0.0),
            "aux_mp_recall": aux_res.get("mp_head_recall", 0.0),
        })
    if inference["hp"] is not None:
        # Fret-region accuracy: over all frames, and over frames with a fretted note (the paper's Table 7).
        hp_pred = torch.argmax(inference["hp"], dim=-1)
        hp_true = inference["hp_tgt"]
        hp_true = (torch.argmax(hp_true, dim=-1) if hp_true.dim() == 2 else hp_true).long().reshape(-1)
        result["hand_pos_acc_all_frames"] = float((hp_pred == hp_true).float().mean())
        result.update(compute_hand_position_metrics(hp_pred, hp_true, False, silence_class=0))
    if inference["sa"] is not None:
        result.update(compute_string_activity_metrics(inference["sa"], inference["sa_tgt"].float(), False))
    if inference["pc"] is not None:
        result.update(compute_pitch_class_metrics(inference["pc"], inference["pc_tgt"].float(), False))
        pc_pred = (torch.sigmoid(inference["pc"]) > 0.5).int().numpy().reshape(-1)
        result["pitch_class_f1_micro_frame"] = float(
            f1_score(inference["pc_tgt"].int().numpy().reshape(-1), pc_pred, average="binary", zero_division=0))  # type: ignore[reportArgumentType]

    result["string_assignment"] = string_assignment_breakdown(
        preds, targets, tuning, config["instrument"]["num_frets"], silence_class)
    result["one_minus_tdr"] = 1.0 - tdr_val
    return result


def write_summary(fold_results: list[dict], experiment_path: str) -> str:
    """Aggregates mean/std across folds for tablature/TDR, auxiliary multi-pitch head, and
    error-rate metrics, appends a per-fold detail line, and writes it to k-fold_test_results.txt.

    Returns:
        The path to the written summary file.
    """
    metrics_tab = ["tab_f1", "tab_precision", "tab_recall", "tdr", "tab_mp_f1", "tab_mp_precision", "tab_mp_recall"]
    metrics_aux = ["aux_mp_f1", "aux_mp_precision", "aux_mp_recall", "hand_pos_acc",
                   "hand_pos_acc_all_frames", "string_act_f1", "pitch_class_f1"]
    metrics_err = ["tab_error_total_rate", "tab_error_substitution_rate",
                   "tab_error_miss_rate", "tab_error_false_alarm_rate"]

    lines = ["K-FOLD TEST RESULTS\n", "=" * 50 + "\n"]

    lines.append("\n[Tablature & TDR]\n")
    for key in metrics_tab:
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            lines.append(f"  {key:<30} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}\n")

    if any(k in r for r in fold_results for k in metrics_aux):
        lines.append("\n[Auxiliary Heads]\n")
        for key in metrics_aux:
            vals = [r[key] for r in fold_results if key in r]
            if vals:
                lines.append(f"  {key:<30} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}\n")

    lines.append("\n[Error Rates]\n")
    for key in metrics_err:
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            lines.append(f"  {key:<35} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}\n")

    pooled = [r["string_assignment"] for r in fold_results if "string_assignment" in r]
    if pooled:
        lines.append("\n[Wrong-string rate among pitch-correct notes, pooled over folds]\n")
        for cat in ("overall", "open_string", "fretted", "fret_1_5", "fret_6_19",
                    "single_position_pitch", "multi_position_pitch"):
            events = sum(sa[cat]["events"] for sa in pooled)
            errors = sum(sa[cat]["errors"] for sa in pooled)
            if events:
                lines.append(f"  {cat:<35} {errors / events:.4f}  ({errors}/{events})\n")
        adjacent = sum(sa["adjacent_errors"] for sa in pooled)
        total = adjacent + sum(sa["nonadjacent_errors"] for sa in pooled)
        if total:
            lines.append(f"  {'adjacent-string share of errors':<35} {adjacent / total:.4f}\n")

    lines.append("\n[Fold Details]\n")
    for r in fold_results:
        lines.append(
            f"  Fold {r['fold']} (test={r['test_group']}) | "
            f"tab_f1={r['tab_f1']:.4f} | tdr={r['tdr']:.4f} | "
            f"thresh={r['best_threshold']:.2f} | dur={r['best_min_duration']}\n"
        )

    txt_path = os.path.join(experiment_path, "k-fold_test_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return txt_path


def parse_args():
    """Parses CLI arguments for evaluating a trained k-fold experiment."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-path", required=True,
                         help="Path to an outputs/<model>_<timestamp> directory containing fold_*/model_best.pt.")
    parser.add_argument("--config", default=None,
                         help="Path to a config YAML; defaults to <experiment-path>/config.yaml.")
    parser.add_argument("--data-root", default=None, help="Override the dataset's data_root from the config.")
    parser.add_argument(
        "--folds", default=None,
        help="Comma-separated 1-based fold numbers to evaluate, e.g. '1,2'. Default: every fold_* found.",
    )
    return parser.parse_args()


def run_evaluation(
    experiment_path: str,
    config: dict,
    device: torch.device | None = None,
    fold_numbers: list[int] | None = None,
) -> str:
    """Runs the leak-free evaluation pipeline against an already-trained experiment.

    Shared by evaluate.py's CLI (manual re-evaluation) and train.py's automatic
    post-training evaluation. Loads each requested fold's fold_<n>/model_best.pt,
    grid-searches the silence-recovery threshold on the validation player, scores
    the held-out test player, and writes k-fold_test_results.txt + per-fold
    test_results.json into experiment_path. Returns the summary file's path.
    """
    if device is None:
        device = resolve_device(config)
    logger.info(f"Using device: {device}")

    all_npz_paths, groups = prepare_dataset_files(config)
    unique_groups = sorted(set(groups))
    group_to_paths = {g: [] for g in unique_groups}
    for path, group in zip(all_npz_paths, groups):
        group_to_paths[group].append(path)

    if fold_numbers is None:
        fold_numbers = sorted(
            int(name.split("_")[1])
            for name in os.listdir(experiment_path)
            if name.startswith("fold_") and os.path.isdir(os.path.join(experiment_path, name))
        )

    fold_results = []
    for fold_num in fold_numbers:
        fold_index = fold_num - 1
        _, val_paths, test_paths, test_group, val_group = fold_split(unique_groups, group_to_paths, fold_index)

        logger.info(f"Fold {fold_num} | Test: {test_group} | Val: {val_group}")

        checkpoint_path = os.path.join(experiment_path, f"fold_{fold_num}", "model_best.pt")
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        logger.info("Validation inference (threshold/duration search)...")
        val = full_track_inference(val_paths, model, config, device)
        best_thresh, best_dur = find_best_postproc(val["tab"], val["tgt"], config)
        logger.info(f"  Best threshold: {best_thresh:.2f} | Best min_dur: {best_dur}")
        del val

        logger.info("Test inference...")
        test = full_track_inference(test_paths, model, config, device)
        results = evaluate_fold(test, best_thresh, best_dur, config)
        results["fold"] = fold_num
        results["test_group"] = test_group
        results["val_group"] = val_group
        fold_results.append(results)

        logger.info(
            f"  tab_f1={results['tab_f1']:.4f} | tdr={results['tdr']:.4f} | tab_mp_f1={results['tab_mp_f1']:.4f}"
        )

        fold_path = os.path.join(experiment_path, f"fold_{fold_num}")
        with open(os.path.join(fold_path, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(convert_history_to_native_types({"fold": fold_num, "results": results}), f, indent=4)

        del model

    summary_path = write_summary(fold_results, experiment_path)
    logger.info(f"Summary written to: {summary_path}")
    return summary_path


def main():
    """CLI entry point: loads the experiment's config, applies a --data-root override if given,
    and runs the leak-free evaluation pipeline over the requested folds."""
    args = parse_args()
    config_path = args.config or os.path.join(args.experiment_path, "config.yaml")
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    if args.data_root:
        raw_config[raw_config["dataset"]]["data_root"] = args.data_root
    config = process_config(raw_config)
    setup_logger(config)

    fold_numbers = [int(n) for n in args.folds.split(",")] if args.folds else None
    run_evaluation(args.experiment_path, config, fold_numbers=fold_numbers)


if __name__ == "__main__":
    main()

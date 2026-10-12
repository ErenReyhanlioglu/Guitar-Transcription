"""Checks that run before any training, so that a broken split, an unreadable
feature file, or a misconfigured objective stops the run instead of silently
changing what gets trained.

Ported from the revision campaign's runner (run_reviewer_experiments_v2.py).
"""
import hashlib
import json
import logging
import math
import os
import zipfile

import numpy as np
import torch

from src.data_loader import fold_split, get_dataloaders
from src.utils.losses import CombinedLoss

logger = logging.getLogger(__name__)

# data.target_keys entry -> array the target is read from in each .npz file
TARGET_TO_NPZ_KEY = {
    'tablature': 'tablature',
    'hand_pos_target': 'hand_position',
    'activity_target': 'string_activity',
    'pitch_class_target': 'pitch_class',
}


class PreflightError(RuntimeError):
    """Raised when a preflight check fails."""


def _file_sha256(path: str, block_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file's bytes, read in blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def _npz_shapes(path: str) -> dict:
    """Reads every array's shape from an .npz file's .npy headers, without loading the data."""
    shapes = {}
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            with zf.open(member) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
            key = member[:-len('.npy')] if member.endswith('.npy') else member
            shapes[key] = tuple(int(x) for x in shape)
    return shapes


def _required_npz_keys(config: dict) -> list[str]:
    """The .npz arrays this config reads: its active features plus its enabled targets."""
    data_conf = config['data']
    features = data_conf['active_features']
    features = [k for k, v in features.items() if v] if isinstance(features, dict) else list(features)
    feature_map = data_conf.get('feature_to_file_map', {})
    targets = data_conf.get('target_keys', {})
    targets = [k for k, v in targets.items() if v] if isinstance(targets, dict) else list(targets)

    required = [feature_map.get(k, k) for k in features]
    required += [TARGET_TO_NPZ_KEY[t] for t in targets if t in TARGET_TO_NPZ_KEY]
    # Config file maps use the archive member name ("hcqt.npy"); compare on the logical key.
    return sorted({k[:-len('.npy')] if k.endswith('.npy') else k for k in required})


def audit_dataset(unique_groups: list[str], group_to_paths: dict, config: dict,
                  output_dir: str | None = None) -> dict:
    """Verifies the dataset and every fold's split before training.

    Checks that every player shares the same repertoire (GuitarSet: six players,
    the same 60 excerpts each), that no file is duplicated, that every fold's
    train/validation/test file sets are pairwise disjoint, and that every file
    holds the arrays this config reads with a consistent frame count. Writes
    the result to `<output_dir>/dataset_split_manifest.json` if given.

    Raises:
        PreflightError: On the first failed check.
    """
    all_paths = [p for g in unique_groups for p in group_to_paths[g]]
    counts = {g: len(group_to_paths[g]) for g in unique_groups}
    basenames = [os.path.basename(p) for p in all_paths]
    if len(set(basenames)) != len(basenames):
        raise PreflightError("Duplicate .npz file names in the data root.")

    per_item = {}
    for group in unique_groups:
        for path in group_to_paths[group]:
            prefix, sep, item = os.path.basename(path).partition('_')
            if not sep or prefix != group:
                raise PreflightError(f"Player prefix mismatch: group {group}, file {os.path.basename(path)}.")
            per_item.setdefault(item, set()).add(group)
    incomplete = {item: sorted(gs) for item, gs in per_item.items() if gs != set(unique_groups)}
    if incomplete:
        raise PreflightError(f"Players do not share one complete repertoire: {incomplete}")

    folds = []
    for i in range(len(unique_groups)):
        train, val, test, test_group, val_group = fold_split(unique_groups, group_to_paths, i)
        train_set, val_set, test_set = (set(map(os.path.realpath, s)) for s in (train, val, test))
        overlaps = {'train_val': len(train_set & val_set), 'train_test': len(train_set & test_set),
                    'val_test': len(val_set & test_set)}
        if any(overlaps.values()):
            raise PreflightError(f"Path overlap between splits in fold {i + 1}: {overlaps}")
        folds.append({'fold': i + 1, 'test_player': test_group, 'validation_player': val_group,
                      'train_files': len(train), 'validation_files': len(val), 'test_files': len(test),
                      'path_overlaps': overlaps})

    required = _required_npz_keys(config)
    num_strings = config['instrument']['num_strings']
    errors, records = [], []
    for path in all_paths:
        name = os.path.basename(path)
        try:
            shapes = _npz_shapes(path)
        except Exception as exc:
            errors.append(f"{name} unreadable: {type(exc).__name__}: {exc}")
            continue
        missing = [k for k in required if k not in shapes]
        if missing:
            errors.append(f"{name} missing {missing}")
            continue
        lengths = {k: shapes[k][-1] for k in required}
        if len(set(lengths.values())) != 1:
            errors.append(f"{name} inconsistent frame counts {lengths}")
        if 'tablature' in shapes and (len(shapes['tablature']) != 2 or shapes['tablature'][0] != num_strings):
            errors.append(f"{name} invalid tablature shape {shapes['tablature']}")
        records.append({'name': name, 'bytes': os.path.getsize(path), 'frames': lengths[required[0]]})
    if errors:
        raise PreflightError("Feature file check failed:\n" + "\n".join(errors[:20]))

    # Identical files must have identical sizes, so only hash files that share a size.
    by_size = {}
    for path in all_paths:
        by_size.setdefault(os.path.getsize(path), []).append(path)
    duplicates = []
    for candidates in (c for c in by_size.values() if len(c) > 1):
        by_hash = {}
        for path in candidates:
            by_hash.setdefault(_file_sha256(path), []).append(os.path.basename(path))
        duplicates += [names for names in by_hash.values() if len(names) > 1]
    if duplicates:
        raise PreflightError(f"Byte-identical .npz files: {duplicates}")

    manifest = {
        'protocol': 'leave-one-player-out; the next player (cyclically) is used for validation',
        'scope': 'unseen performer on a shared repertoire; not unseen composition',
        'players': list(unique_groups),
        'files_per_player': counts,
        'total_files': len(all_paths),
        'shared_items': len(per_item),
        'required_npz_keys': required,
        'folds': folds,
        'files': sorted(records, key=lambda r: r['name']),
    }
    if output_dir:
        with open(os.path.join(output_dir, 'dataset_split_manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
    logger.info(f"Preflight: {len(all_paths)} files, {counts}, {len(per_item)} shared items, "
                f"no overlap between splits in any of {len(folds)} folds.")
    return manifest


def expected_total_loss(losses: dict, config: dict, log_vars: dict | None = None) -> float:
    """Recomputes the composite loss from the per-task losses, independently of CombinedLoss.

    Static: tablature + sum(weight * auxiliary). Uncertainty: sum over tasks of
    0.5 * exp(-s) * L + 0.5 * s, with s the task's current log-variance.
    """
    loss_conf = config['loss']
    if loss_conf.get('weighting_strategy', 'static') == 'uncertainty':
        total = 0.0
        for task, value in losses.items():
            s = float(log_vars[task]) if log_vars and task in log_vars else None
            total += value if s is None else 0.5 * math.exp(-s) * value + 0.5 * s
        return total
    total = losses.get('tablature', 0.0)
    for task, value in losses.items():
        if task != 'tablature':
            total += loss_conf.get(f"{task}_loss", {}).get('weight', 1.0) * value
    return total


def verify_loss_objective(model_factory, config: dict, train_paths: list[str], val_paths: list[str],
                          test_paths: list[str], device: torch.device) -> dict:
    """Checks on one real training batch that the composite loss is the objective the config asks for.

    In particular, a static run with every auxiliary weight at 0.0 must train on
    the tablature loss alone.

    Raises:
        PreflightError: If the composite loss differs from the recomputed one.
    """
    cfg = {**config, 'data': {**config['data'], 'batch_size': 1, 'num_workers': 0}}
    train_loader, _, _ = get_dataloaders(cfg, train_paths, val_paths, test_paths)
    batch = next(iter(train_loader))
    features = {k: v.to(device) for k, v in batch['features'].items()}
    targets = {k: v.to(device) for k, v in batch.items() if k != 'features'}

    model = model_factory(cfg).to(device).eval()
    loss_fn = CombinedLoss(config=cfg).to(device)
    with torch.no_grad():
        out = loss_fn(model(features), targets)

    per_task = {k[:-len('_loss')]: float(v) for k, v in out.items() if k != 'total_loss'}
    log_vars = ({k: v.item() for k, v in loss_fn.uncertainty_wrapper.log_vars.items()}
                if hasattr(loss_fn, 'uncertainty_wrapper') else None)
    total = float(out['total_loss'])
    expected = expected_total_loss(per_task, cfg, log_vars)
    if not math.isclose(total, expected, rel_tol=1e-5, abs_tol=1e-6):
        raise PreflightError(f"Composite loss {total:.6f} differs from the configured objective {expected:.6f}.")

    aux_weights = {t: cfg['loss'].get(f"{t}_loss", {}).get('weight', 1.0) for t in per_task if t != 'tablature'}
    tab_only = (cfg['loss'].get('weighting_strategy', 'static') != 'uncertainty'
                and all(w == 0.0 for w in aux_weights.values()))
    if tab_only and not math.isclose(total, per_task['tablature'], rel_tol=1e-6, abs_tol=1e-6):
        raise PreflightError(f"Tablature-only objective failed: total={total:.6f}, tablature={per_task['tablature']:.6f}.")

    logger.info(f"Preflight: composite loss matches the configured objective "
                f"(total={total:.6f}, tablature={per_task['tablature']:.6f}"
                f"{', tablature-only' if tab_only else ''}).")
    del model, loss_fn, train_loader
    return {'total': total, 'tablature': per_task['tablature'], 'tablature_only': tab_only}

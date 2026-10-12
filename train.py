"""CLI entry point for training the cascaded multi-task guitar transcription model.

Runs the paper's six-fold leave-one-player-out cross-validation (fold i holds
out player i for testing and uses player (i+1) mod n for validation) over a
GuitarSet feature directory. Ported from the original Colab training notebook;
all actual training logic lives in src/trainer.py and is unchanged here.

By default, every fold just trained is automatically evaluated afterwards via
evaluate.py's leak-free pipeline (--skip-eval to disable). Re-run evaluate.py
manually any time afterwards to redo evaluation without retraining.

Example:
    python train.py --config configs/cnn_mtl_config.yaml \\
        --data-root /path/to/guitarset_enriched_npz \\
        --output-root /path/to/outputs
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from evaluate import run_evaluation
from src.data_loader import fold_split, get_dataloaders, prepare_dataset_files
from src.models import get_model
from src.trainer import Trainer
from src.utils.config_helpers import get_optimizer, get_scheduler, process_config, resolve_device
from src.utils.experiment import create_experiment_directory
from src.utils.logger import setup_logger
from src.utils.preflight import audit_dataset, verify_loss_objective

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seeds torch/numpy/random for reproducibility and forces deterministic, non-benchmarked cuDNN."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parses CLI arguments for training the cascaded multi-task model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a config YAML (e.g. configs/cnn_mtl_config.yaml).")
    parser.add_argument("--data-root", default=None, help="Override the dataset's data_root from the config.")
    parser.add_argument("--output-root", default=None, help="Override paths.output_root from the config.")
    parser.add_argument(
        "--folds", default=None,
        help="Comma-separated 0-based fold indices to run, e.g. '0,1'. Default: all folds.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override data.num_workers from the config (e.g. 0 to avoid multiprocessing "
             "DataLoader workers, which can crash on some Windows setups). Purely an I/O "
             "parallelism knob — never changes results.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override data.batch_size from the config. Each item is a "
             "framify_chunk_size-frame chunk, so the paper's batch_size=4 means "
             "~2000 windowed samples per step — this needs datacenter-class VRAM "
             "(the paper used an A100). Lower this for small local GPUs. NOTE: unlike "
             "--num-workers, this can shift BatchNorm statistics slightly, so it's meant "
             "for local smoke-testing, not for runs claiming to reproduce the paper.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the automatic evaluate.py run after training finishes. By default, "
             "every fold just trained is immediately evaluated (leak-free threshold "
             "search on validation, scored on test) and written to "
             "k-fold_test_results.txt. Re-run evaluate.py manually any time afterwards "
             "to redo evaluation without retraining.",
    )
    return parser.parse_args()


def load_config(config_path: str, data_root_override: str | None) -> dict:
    """Loads a config YAML, applies an optional --data-root override, and runs it through process_config."""
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if data_root_override:
        raw_config[raw_config["dataset"]]["data_root"] = data_root_override

    return process_config(raw_config)


def run_fold(fold_index: int, n_splits: int, unique_groups: list[str], group_to_paths: dict,
             config: dict, experiment_path_main: str, device: torch.device) -> None:
    """Trains one leave-one-player-out fold: builds its split, data loaders, model, optimizer,
    and scheduler, then runs the training loop via Trainer.

    Fold `fold_index` holds out player `fold_index` for testing and player
    `(fold_index + 1) % n_splits` for validation; all other players are used
    for training. Frees the model/loaders and clears the CUDA cache afterward.
    """
    fold_num = fold_index + 1
    train_paths, val_paths, test_paths, test_group, val_group = fold_split(unique_groups, group_to_paths, fold_index)
    train_groups = [g for g in unique_groups if g not in (test_group, val_group)]

    logger.info(f"##### FOLD {fold_num}/{n_splits} STARTING #####")
    logger.info(f"Train groups: {train_groups} ({len(train_paths)} files)")
    logger.info(f"Validation group: {val_group} ({len(val_paths)} files)")
    logger.info(f"Test group: {test_group} ({len(test_paths)} files)")

    fold_path = os.path.join(experiment_path_main, f"fold_{fold_num}")
    os.makedirs(fold_path, exist_ok=True)

    # The six input/decoding runs (Table 1) seeded once at start-up; the component
    # campaign (Table 3) reseeded at the start of every fold, so any fold can be
    # rerun on its own. Its configs set training.reseed_each_fold: true.
    if config["training"].get("reseed_each_fold", False):
        set_seed(config["data"]["random_state"])

    train_loader, val_loader, test_loader = get_dataloaders(config, train_paths, val_paths, test_paths)

    model = get_model(config).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    use_tensorboard = config.get("logging_and_checkpointing", {}).get("use_tensorboard", False)
    writer = SummaryWriter(log_dir=fold_path) if use_tensorboard else None

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device, config=config,
        experiment_path=fold_path, main_exp_path=experiment_path_main,
        class_weights=None, writer=writer,
    )

    try:
        trainer.train()
    finally:
        if writer:
            writer.close()
        del model, optimizer, scheduler, trainer, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def main():
    """CLI entry point: loads the config, resolves the output root/device, seeds RNGs, creates
    the experiment directory, trains every requested fold, and (unless --skip-eval) runs the
    leak-free evaluation pipeline on the folds just trained."""
    args = parse_args()
    config = load_config(args.config, args.data_root)

    output_root = args.output_root or config.get("paths", {}).get("output_root")
    if not output_root:
        raise ValueError(
            "No output root set. Pass --output-root or set paths.output_root in the config."
        )

    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    setup_logger(config)
    if args.batch_size is not None:
        logger.warning(
            f"Overriding batch_size to {args.batch_size} via --batch-size — this can "
            f"shift BatchNorm statistics slightly. Fine for local smoke-testing, but "
            f"don't use this for a run meant to reproduce the paper's numbers."
        )
    set_seed(config["data"]["random_state"])

    experiment_path_main = create_experiment_directory(
        base_output_path=output_root,
        model_name=config["model"]["name"],
        config=config,
        config_path=args.config,
    )

    all_npz_paths, groups = prepare_dataset_files(config)
    unique_groups = sorted(set(groups))
    n_splits = len(unique_groups)
    group_to_paths = {g: [] for g in unique_groups}
    for path, group in zip(all_npz_paths, groups):
        group_to_paths[group].append(path)

    device = resolve_device(config)
    logger.info(f"Using device: {device}")

    # Stop before any training if the split, the feature files, or the objective are wrong.
    audit_dataset(unique_groups, group_to_paths, config, output_dir=experiment_path_main)
    verify_loss_objective(get_model, config, *fold_split(unique_groups, group_to_paths, 0)[:3], device=device)
    set_seed(config["data"]["random_state"])

    fold_indices = [int(i) for i in args.folds.split(",")] if args.folds else list(range(n_splits))

    for fold_index in fold_indices:
        run_fold(fold_index, n_splits, unique_groups, group_to_paths, config, experiment_path_main, device)

    logger.info("All requested folds trained.")

    if args.skip_eval:
        logger.info("--skip-eval set: not running automatic evaluation.")
    else:
        logger.info("Running automatic evaluation on the fold(s) just trained...")
        fold_numbers = [i + 1 for i in fold_indices]
        run_evaluation(experiment_path_main, config, device=device, fold_numbers=fold_numbers)


if __name__ == "__main__":
    main()

import torch
import numpy as np
import os
import json
import logging
from typing import Any
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from src.utils.logger import describe, log_epoch_summary
from src.utils.metrics import (
    compute_tablature_metrics, compute_multipitch_metrics, compute_octave_tolerant_metrics,
    compute_tablature_error_scores, compute_tdr,
    compute_hand_position_metrics, compute_string_activity_metrics,
    compute_pitch_class_metrics, compute_aux_multipitch_metrics
)
from src.utils.losses import CombinedLoss
from src.utils.guitar_profile import GuitarProfile
from src.utils.experiment import save_model_summary, generate_experiment_report
from src.utils.callbacks import EarlyStopping
from src.utils.analyze_errors import analyze as run_error_analysis

logger = logging.getLogger(__name__)


def convert_history_to_native_types(history) -> Any:
    """Recursively converts numpy scalars/arrays inside a nested dict/list into native Python types, for JSON serialization."""
    if isinstance(history, dict):
        return {k: convert_history_to_native_types(v) for k, v in history.items()}
    elif isinstance(history, list):
        return [convert_history_to_native_types(i) for i in history]
    elif isinstance(history, np.integer):
        return int(history)
    elif isinstance(history, np.floating):
        return float(history)
    elif isinstance(history, np.ndarray):
        return history.tolist()
    return history


class Trainer:
    """Runs the training loop for one cross-validation fold.

    Handles the train/validation epoch loop, the Gradient Cosine Similarity
    diagnostic, checkpointing, early stopping, and post-training reporting.
    """
    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler,
        device, config, experiment_path, main_exp_path,
        class_weights=None, writer=None, initial_history=None
    ):
        """Wires up the model, data loaders, optimizer/scheduler, loss function, and training state.

        Args:
            model: The model to train.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Optimizer already constructed for `model`'s parameters.
            scheduler: LR scheduler stepped on the monitored validation metric, or None.
            device: Device to move the model and loss function to.
            config: Full experiment config.
            experiment_path: Directory for this fold's checkpoints/history/reports.
            main_exp_path: Root experiment directory (shared across folds).
            class_weights: Optional class weights forwarded to the loss function.
            writer: Optional SummaryWriter-like object, closed at the end of training.
            initial_history: Existing history dict to resume from, or None to start fresh.
        """
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.config         = config
        self.experiment_path = experiment_path
        self.main_exp_path  = main_exp_path
        self.writer         = writer

        self.loss_config       = self.config['loss']
        self.training_config   = self.config['training']
        self.instrument_config = self.config['instrument']
        self.metrics_config    = self.config['metrics']
        self.guitar_profile    = GuitarProfile(self.instrument_config)

        self.loss_fn = CombinedLoss(
            config=self.config, class_weights=class_weights
        ).to(self.device)

        self.history = (
            initial_history if initial_history else self._create_empty_history()
        )

        # ── GCS hook: capture the bottleneck tensor ──
        self.bottleneck = None
        self._register_bottleneck_hook()

        # ── Add uncertainty-weighting parameters to the optimizer ──
        if self.loss_config.get('weighting_strategy') == 'uncertainty':
            loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
            if loss_params:
                active_opt = self.training_config['optimizer']['active_optimizer']
                opt_conf   = self.training_config['optimizer']['configurations'].get(active_opt, {})
                loss_lr    = opt_conf.get('params', {}).get('loss_lr', None)
                param_group = {'params': loss_params}
                if loss_lr is not None:
                    param_group['lr'] = loss_lr
                    logger.info(f"Uncertainty params added with custom LR: {loss_lr}")
                else:
                    logger.info("Uncertainty params added with default optimizer LR.")
                self.optimizer.add_param_group(param_group)

        self.use_amp      = self.training_config.get('use_mixed_precision', False)
        self.scaler       = GradScaler(enabled=self.use_amp)
        self.early_stopper = self._setup_early_stopping()

        logger.info("Trainer initialized.")

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def _register_bottleneck_hook(self):
        """
        Registers a forward_pre_hook on one of the auxiliary heads.
        The hook captures the tensor entering that head (= the pure bottleneck z).
        The tablature head receives bottleneck + aux_concat, so an auxiliary
        head is preferred in order to capture the pure z.
        """
        def hook_fn(module, inp):
            self.bottleneck = inp[0]
            if self.bottleneck.requires_grad:
                self.bottleneck.retain_grad()

        model_ref   = self.model.module if hasattr(self.model, 'module') else self.model
        target_head = None
        hook_name   = ""

        if hasattr(model_ref, 'heads'):
            for candidate in ('multipitch', 'hand_position', 'string_activity',
                              'pitch_class', 'tablature'):
                if candidate in model_ref.heads:
                    target_head = model_ref.heads[candidate]
                    hook_name   = candidate
                    break

        if target_head is not None:
            target_head.register_forward_pre_hook(hook_fn)
            logger.info(
                f"GCS hook registered on 'heads.{hook_name}' "
                f"({'fallback' if hook_name == 'tablature' else 'pure bottleneck'})."
            )
        else:
            logger.warning("No suitable head found for GCS hook — GCS disabled.")

    def _setup_early_stopping(self):
        """Builds the `EarlyStopping` callback from `training.early_stopping` config, or returns None if disabled."""
        es_cfg = self.training_config.get('early_stopping', {})
        if es_cfg.get('enabled', False):
            return EarlyStopping(
                patience  = es_cfg.get('patience', 10),
                min_delta = es_cfg.get('min_delta', 0),
                monitor   = es_cfg.get('monitor', 'val_tab_f1'),
                mode      = es_cfg.get('mode', 'max'),
            )
        return None

    def _create_empty_history(self):
        """Builds an empty history dict with train_/val_-prefixed lists for every tracked metric,
        plus GCS, learning-rate, and (if enabled) uncertainty sigma-weight entries."""
        loss_keys     = ["loss_total", "loss_tablature", "loss_hand_position",
                         "loss_string_activity", "loss_pitch_class",
                         "loss_multipitch", "loss_onset"]
        tab_keys      = ["tab_f1", "tab_precision", "tab_recall", "tdr"]
        hand_keys     = ["hand_pos_acc", "hand_pos_f1",
                         "hand_pos_precision", "hand_pos_recall"]
        string_keys   = ["string_act_f1", "string_act_precision", "string_act_recall"]
        pitch_keys    = ["pitch_class_f1", "pitch_class_precision", "pitch_class_recall"]
        mp_head_keys  = ["mp_head_f1", "mp_head_precision", "mp_head_recall"]
        onset_keys    = ["onset_f1", "onset_precision", "onset_recall"]
        mp_deriv_keys = ["mp_f1", "mp_precision", "mp_recall"]
        octave_keys   = ["octave_f1", "octave_precision", "octave_recall"]
        error_keys    = [
            "tab_error_total_rate", "tab_error_substitution_rate",
            "tab_error_miss_rate", "tab_error_false_alarm_rate",
            "tab_error_duplicate_pitch_rate", "tab_error_total_count",
            "tab_error_substitution_count", "tab_error_miss_count",
            "tab_error_false_alarm_count", "tab_error_duplicate_pitch_count",
        ]

        all_keys = (loss_keys + tab_keys + hand_keys + string_keys + pitch_keys
                    + mp_head_keys + onset_keys + mp_deriv_keys
                    + octave_keys + error_keys)

        history = {
            f"{phase}_{key}": []
            for phase in ("train", "val")
            for key in all_keys
        }

        # GCS keys — no train/val prefix, stored as-is
        heads = self.config['model'].get('params', {}).get('heads', {})
        for task, cfg in heads.items():
            if task != 'tablature' and cfg.get('enabled', False):
                history[f"gcs_tab_{task}"] = []

        history["lr"] = []
        if self.loss_config.get('weighting_strategy') == 'uncertainty':
            history['sigma_weights'] = []

        return history

    # ─────────────────────────────────────────────────────────────────────────
    # GCS — END OF EPOCH, MULTI-BATCH AVERAGE
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_gcs_averaged(self, num_batches: int = 15) -> dict:
        """
        After the epoch ends, takes the first `num_batches` batches of the
        (shuffled, randomly cropped) training loader with the model in eval
        mode, computes GCS for each, and averages the results. The paper uses
        15 batches (`loss.grad_cos_sim.num_batches`), about 30,000 frames.

        Why this approach:
          - Relies on an average over multiple samples, not just the last batch.
          - Robust to batch-to-batch noise.
          - Defensible for the paper: "GCS averaged over N batches."
          - Does not touch model weights; only a forward pass + gradient computation.
        """
        if self.bottleneck is None:
            logger.warning("GCS: hook not registered or not triggered yet.")
            return {}

        gcs_conf    = self.loss_config.get('grad_cos_sim', {})
        gcs_enabled = gcs_conf.get('enabled', True)
        if not gcs_enabled:
            return {}

        self.model.eval()
        accumulator  = {}
        valid_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= num_batches:
                break

            inputs = self._prepare_model_input(batch)
            for key in batch:
                if key != 'features':
                    batch[key] = batch[key].to(self.device)

            self.bottleneck = None   # the hook refills this on every forward pass

            try:
                with torch.enable_grad():
                    with autocast(enabled=self.use_amp):
                        model_output = self.model(inputs)

                # Did the hook populate the bottleneck?
                if self.bottleneck is None or not self.bottleneck.requires_grad:
                    continue

                self.bottleneck.retain_grad()

                with autocast(enabled=self.use_amp):
                    loss_dict = self.loss_fn(model_output, batch)

                # Find the tablature loss key
                tab_key = next(
                    (k for k in ('tablature_loss', 'tablature') if k in loss_dict),
                    None
                )
                if tab_key is None:
                    continue

                # Tablature gradient
                grad_tab = torch.autograd.grad(
                    loss_dict[tab_key],
                    self.bottleneck,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                if grad_tab is None:
                    continue

                grad_tab_flat = grad_tab.detach().flatten()

                # GCS for each auxiliary task
                aux_keys = [
                    k for k in loss_dict
                    if k not in ('total_loss', tab_key)
                    and isinstance(loss_dict[k], torch.Tensor)
                ]

                for idx, aux_key in enumerate(aux_keys):
                    keep_graph = (idx < len(aux_keys) - 1)
                    grad_aux   = torch.autograd.grad(
                        loss_dict[aux_key],
                        self.bottleneck,
                        retain_graph=keep_graph,
                        allow_unused=True,
                    )[0]

                    if grad_aux is not None:
                        grad_aux_flat = grad_aux.detach().flatten()
                        sim = torch.nn.functional.cosine_similarity(
                            grad_tab_flat.unsqueeze(0),
                            grad_aux_flat.unsqueeze(0),
                            eps=1e-8,
                        ).item()
                        gcs_key = f"gcs_tab_{aux_key.replace('_loss', '')}"
                        accumulator[gcs_key] = accumulator.get(gcs_key, 0.0) + sim

                valid_batches += 1

            except Exception as exc:
                logger.debug(f"GCS batch {batch_idx} skipped: {exc}")
                continue
            finally:
                self.optimizer.zero_grad(set_to_none=True)

        self.model.train()

        if valid_batches == 0:
            logger.warning("GCS: no valid batches found, result is empty.")
            return {}

        return {k: v / valid_batches for k, v in accumulator.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # EPOCH
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_model_input(self, batch):
        """Moves a batch's feature tensors to the training device."""
        return {key: tensor.to(self.device)
                for key, tensor in batch['features'].items()}

    def _run_epoch(self, data_loader, is_training=True):
        """Runs one training or validation epoch over `data_loader`.

        Per batch: runs the forward pass, computes the combined loss, and
        backpropagates only if `is_training`. Accumulates the loss breakdown
        plus per-task logits/targets, then reduces them into epoch-level
        metrics via `_calculate_all_metrics`.
        """
        self.model.train(is_training)

        epoch_stats = {
            "loss_total": 0.0,
            "loss_breakdown": {},
            "tab_logits": [],            "tab_targets": [],
            "aux_hand_pos_logits": [],   "aux_hand_pos_targets": [],
            "aux_activity_logits": [],   "aux_activity_targets": [],
            "aux_pitch_class_logits": [], "aux_pitch_class_targets": [],
            "aux_multipitch_logits": [],  "aux_multipitch_targets": [],
            "aux_onset_logits": [],       "aux_onset_targets": [],
        }

        desc = "Training" if is_training else "Validating"

        for batch in tqdm(data_loader, desc=desc, leave=False):
            inputs = self._prepare_model_input(batch)
            for key in batch:
                if key != 'features':
                    batch[key] = batch[key].to(self.device)

            with torch.set_grad_enabled(is_training):
                self.optimizer.zero_grad(set_to_none=True)
                self.bottleneck = None

                with autocast(enabled=self.use_amp):
                    model_output    = self.model(inputs)
                    loss_dict       = self.loss_fn(model_output, batch)
                    loss_to_backprop = loss_dict['total_loss']

                if is_training and torch.isfinite(loss_to_backprop):
                    self.scaler.scale(loss_to_backprop).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if torch.isfinite(loss_to_backprop):
                epoch_stats["loss_total"] += loss_to_backprop.item()
                for k, v in loss_dict.items():
                    if k == 'total_loss':
                        continue
                    epoch_stats["loss_breakdown"][k] = (
                        epoch_stats["loss_breakdown"].get(k, 0.0) + v.item()
                    )

            # Collect outputs
            if 'tab_logits' in model_output:
                epoch_stats["tab_logits"].append(
                    model_output['tab_logits'].cpu().detach())
                epoch_stats["tab_targets"].append(
                    batch['tablature'].cpu().detach())

            if 'hand_pos_logits' in model_output:
                epoch_stats["aux_hand_pos_logits"].append(
                    model_output['hand_pos_logits'].cpu().detach())
                epoch_stats["aux_hand_pos_targets"].append(
                    batch['hand_pos_target'].cpu().detach())

            if 'activity_logits' in model_output:
                epoch_stats["aux_activity_logits"].append(
                    model_output['activity_logits'].cpu().detach())
                epoch_stats["aux_activity_targets"].append(
                    batch['activity_target'].cpu().detach())

            if 'pitch_class_logits' in model_output:
                epoch_stats["aux_pitch_class_logits"].append(
                    model_output['pitch_class_logits'].cpu().detach())
                epoch_stats["aux_pitch_class_targets"].append(
                    batch['pitch_class_target'].cpu().detach())

            if 'multipitch_logits' in model_output and 'multipitch_target' in batch:
                epoch_stats["aux_multipitch_logits"].append(
                    model_output['multipitch_logits'].cpu().detach())
                epoch_stats["aux_multipitch_targets"].append(
                    batch['multipitch_target'].cpu().detach())

            if 'onset_logits' in model_output and 'onset_target' in batch:
                epoch_stats["aux_onset_logits"].append(
                    model_output['onset_logits'].cpu().detach())
                epoch_stats["aux_onset_targets"].append(
                    batch['onset_target'].cpu().detach())

        epoch_metrics = self._calculate_all_metrics(epoch_stats)

        n = max(1, len(data_loader))
        epoch_metrics['loss_total'] = epoch_stats["loss_total"] / n
        for k, v in epoch_stats["loss_breakdown"].items():
            clean = k.replace("loss_", "").replace("_loss", "")
            epoch_metrics[f"loss_{clean}"] = v / n

        return epoch_metrics

    def _calculate_all_metrics(self, stats):
        """Computes tablature, multi-pitch, octave-tolerant, TDR, error-rate, and auxiliary-head
        metrics from the logits/targets accumulated over an epoch in `stats`."""
        final_metrics   = {}
        include_silence = self.metrics_config.get('include_silence', False)
        silence_class   = self.instrument_config['silence_class']
        S = self.instrument_config['num_strings']
        C = self.config['model']['params']['num_classes']

        if stats["tab_logits"]:
            all_logits  = torch.cat(stats["tab_logits"],  dim=0)
            all_targets = torch.cat(stats["tab_targets"], dim=0)

            if all_logits.dim() == 2:
                preds_tab = torch.argmax(all_logits.view(-1, S, C), dim=-1)
            elif all_logits.dim() == 3:
                preds_tab = torch.argmax(all_logits, dim=-1)
            else:
                preds_tab = torch.argmax(all_logits, dim=1)

            targets_flat = all_targets.view(-1, S)

            tab_res = compute_tablature_metrics(
                preds_tab, targets_flat, include_silence, silence_class)
            final_metrics.update({
                'tab_f1':        tab_res.get('overall_f1'),
                'tab_precision': tab_res.get('overall_precision'),
                'tab_recall':    tab_res.get('overall_recall'),
            })

            mp_res = compute_multipitch_metrics(
                preds_tab, targets_flat, self.guitar_profile,
                include_silence, silence_class)
            final_metrics.update({
                'mp_f1':        mp_res.get('multipitch_f1'),
                'mp_precision': mp_res.get('multipitch_precision'),
                'mp_recall':    mp_res.get('multipitch_recall'),
            })

            oct_res = compute_octave_tolerant_metrics(
                preds_tab, targets_flat,
                self.instrument_config['tuning'], silence_class)
            final_metrics.update({
                'octave_f1':        oct_res.get('octave_f1'),
                'octave_precision': oct_res.get('octave_precision'),
                'octave_recall':    oct_res.get('octave_recall'),
            })

            final_metrics['tdr'] = compute_tdr(
                preds_tab, targets_flat, self.guitar_profile,
                include_silence, silence_class)

            err_res = compute_tablature_error_scores(
                preds_tab, targets_flat, silence_class,
                self.instrument_config['tuning'])
            final_metrics.update(err_res)

        if stats["aux_hand_pos_logits"]:
            hp_logits  = torch.cat(stats["aux_hand_pos_logits"])
            hp_targets = torch.cat(stats["aux_hand_pos_targets"])
            hp_preds   = torch.argmax(hp_logits, dim=-1)
            final_metrics.update(
                compute_hand_position_metrics(
                    hp_preds, hp_targets, include_silence, silence_class=0))

        if stats["aux_activity_logits"]:
            act_logits  = torch.cat(stats["aux_activity_logits"])
            act_targets = torch.cat(stats["aux_activity_targets"])
            final_metrics.update(
                compute_string_activity_metrics(
                    act_logits, act_targets, include_silence))

        if stats["aux_pitch_class_logits"]:
            pc_logits  = torch.cat(stats["aux_pitch_class_logits"])
            pc_targets = torch.cat(stats["aux_pitch_class_targets"])
            final_metrics.update(
                compute_pitch_class_metrics(pc_logits, pc_targets, include_silence))

        if stats.get("aux_multipitch_logits"):
            mp_h_logits  = torch.cat(stats["aux_multipitch_logits"])
            mp_h_targets = torch.cat(stats["aux_multipitch_targets"])
            final_metrics.update(
                compute_aux_multipitch_metrics(
                    mp_h_logits, mp_h_targets, include_silence))

        if stats.get("aux_onset_logits"):
            on_logits  = torch.cat(stats["aux_onset_logits"])
            on_targets = torch.cat(stats["aux_onset_targets"])
            on_res     = compute_aux_multipitch_metrics(
                on_logits, on_targets, include_silence)
            final_metrics.update({
                'onset_f1':        on_res['mp_head_f1'],
                'onset_precision': on_res['mp_head_precision'],
                'onset_recall':    on_res['mp_head_recall'],
            })

        return final_metrics

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, start_epoch=0, best_val_metric=-1):
        """Runs the full training loop from `start_epoch` through `training.epochs`.

        Each epoch: runs training and validation passes, optionally computes the
        averaged GCS diagnostic, updates and logs history, steps the LR scheduler
        on the monitored validation metric, saves a checkpoint (and `model_best.pt`
        if it's the best so far), and checks early stopping. Finalizes training
        (saving history, running error analysis, generating reports) once the
        loop ends.

        Returns:
            The trained model and its history dict.
        """
        self.model.to(self.device)

        monitor_key = (
            self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
        )
        if best_val_metric == -1:
            best_val_metric = -np.inf

        epochs = self.training_config['epochs']

        # GCS configuration
        gcs_conf        = self.loss_config.get('grad_cos_sim', {})
        gcs_enabled     = gcs_conf.get('enabled', True)
        gcs_num_batches = gcs_conf.get('num_batches', 15)

        label_map = {
            'gcs_tab_multipitch':      'Multi-Pitch',
            'gcs_tab_hand_position':   'Fret Region',
            'gcs_tab_string_activity': 'String Activity',
            'gcs_tab_pitch_class':     'Pitch Class',
        }

        logger.info("--- Multi-Task Training starting ---")
        if gcs_enabled:
            logger.info(
                f"GCS active - averaging over {gcs_num_batches} batches "
                f"at the end of each epoch."
            )

        for epoch in range(start_epoch, epochs):

            # 1. Training & validation epochs
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics   = self._run_epoch(self.val_loader,   is_training=False)

            # 2. End-of-epoch GCS computation (fully independent of the training step)
            gcs_results = {}
            if gcs_enabled:
                gcs_results = self._compute_gcs_averaged(num_batches=gcs_num_batches)
                # add to train_metrics → flows into history inside _update_and_log_history
                train_metrics.update(gcs_results)

            # 3. Update & log history
            self._update_and_log_history(epoch, epochs, train_metrics, val_metrics)

            # 4. Print the GCS line for Colab / terminal output
            if gcs_enabled and gcs_results:
                parts = [
                    f"{label_map.get(k, k.replace('gcs_tab_','').replace('_',' ').title())}: "
                    f"{v:+.4f}"
                    for k, v in sorted(gcs_results.items())
                ]
                gcs_line = "  |  ".join(parts)
                # print() shows up in Colab cell output
                print(f"  [GCS Ep {epoch+1:03d}]  {gcs_line}")
                logger.info(f"GCS Ep {epoch+1}: {gcs_line}")

            # 5. Scheduler & checkpoint
            current_val = val_metrics.get(
                monitor_key.replace('val_', ''), 0.0
            )
            if self.scheduler:
                self.scheduler.step(current_val)

            is_best = current_val > best_val_metric
            if is_best:
                best_val_metric = current_val
            self._save_checkpoint(epoch, best_val_metric, is_best)

            # 6. Early stopping
            if self.early_stopper:
                self.early_stopper(current_val)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break

        self._finalize_training()
        return self.model, self.history

    # ─────────────────────────────────────────────────────────────────────────
    # HISTORY & CHECKPOINT
    # ─────────────────────────────────────────────────────────────────────────

    def _update_and_log_history(self, epoch, total_epochs, train_metrics, val_metrics):
        """Appends this epoch's values to `self.history` for every tracked key (train_/val_-prefixed
        metrics, GCS, learning rate, uncertainty sigma weights) and logs the epoch summary."""
        for key in self.history.keys():
            if key == 'lr':
                self.history[key].append(
                    self.optimizer.param_groups[0]['lr'])

            elif key == 'sigma_weights':
                if hasattr(self.loss_fn, 'uncertainty_wrapper'):
                    sigmas = {
                        k: v.item()
                        for k, v in self.loss_fn.uncertainty_wrapper.log_vars.items()
                    }
                    self.history[key].append(sigmas)

            elif key.startswith('gcs_'):
                # GCS values arrive in train_metrics as 'gcs_tab_...';
                # history has no prefix, so they match directly.
                val = train_metrics.get(key, None)
                self.history[key].append(float(val) if val is not None else 0.0)

            elif key.startswith('train_'):
                clean = key[len('train_'):]
                self.history[key].append(train_metrics.get(clean, 0.0))

            elif key.startswith('val_'):
                clean = key[len('val_'):]
                self.history[key].append(val_metrics.get(clean, 0.0))

        lr = self.history['lr'][-1]
        log_epoch_summary(logger, epoch, total_epochs, lr, train_metrics, val_metrics)

    def _save_checkpoint(self, epoch, best_val_metric, is_best):
        """Saves the full training state to checkpoint_latest.pt, and the model weights alone
        to model_best.pt if `is_best`."""
        state = {
            'epoch':               epoch,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            'best_val_metric': best_val_metric,
            'history':         self.history,
        }
        torch.save(state, os.path.join(self.experiment_path, 'checkpoint_latest.pt'))
        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.experiment_path, 'model_best.pt')
            )

    def _finalize_training(self):
        """Closes the writer, saves history.json (logging a GCS summary if present), runs the
        final error analysis, and writes the model summary and experiment report."""
        logger.info("--- Training complete ---")
        if self.writer:
            self.writer.close()

        # Save history.json
        history_path = os.path.join(self.experiment_path, "history.json")
        try:
            native_history = convert_history_to_native_types(self.history)
            with open(history_path, 'w') as f:
                json.dump(native_history, f, indent=4)
            logger.info(f"History saved: {history_path}")

            # Confirm the GCS values were actually written
            gcs_keys = [k for k in native_history if k.startswith('gcs_')]
            if gcs_keys:
                logger.info("--- GCS Summary (across all epochs) ---")
                for k in sorted(gcs_keys):
                    vals = [v for v in native_history[k] if v != 0.0]
                    if vals:
                        mean_v = float(np.mean(vals))
                        std_v  = float(np.std(vals))
                        min_v  = float(np.min(vals))
                        max_v  = float(np.max(vals))
                        logger.info(
                            f"  {k:<35} mean={mean_v:+.4f}  "
                            f"std={std_v:.4f}  "
                            f"min={min_v:+.4f}  max={max_v:+.4f}"
                        )
                        # Also print for Colab cell output
                        print(
                            f"  {k:<35} mean={mean_v:+.4f}  "
                            f"std={std_v:.4f}  "
                            f"[{min_v:+.4f}, {max_v:+.4f}]"
                        )
                    else:
                        logger.warning(f"  {k}: no non-zero values.")
            else:
                logger.warning(
                    "No GCS key found in history.json. "
                    "Make sure grad_cos_sim.enabled: true in the config."
                )
        except Exception as e:
            logger.error(f"Could not save history: {e}")

        # Error analysis
        try:
            logger.info("Running final error analysis...")
            run_error_analysis(
                experiment_path=self.experiment_path,
                main_exp_path=self.main_exp_path,
                val_loader=self.val_loader,
            )
        except Exception as e:
            logger.warning(f"Error analysis failed: {e}")

        # Model summary & experiment report
        save_model_summary(self.model, self.config, self.experiment_path)
        try:
            generate_experiment_report(
                model          = self.model,
                history        = self.history,
                val_loader     = self.val_loader,
                config         = self.config,
                experiment_path = self.experiment_path,
                device         = self.device,
                profile        = self.guitar_profile,
                include_silence = self.metrics_config.get('include_silence', False),
                silence_class  = self.instrument_config['silence_class'],
            )
        except Exception as e:
            logger.warning(f"Could not generate report: {e}")
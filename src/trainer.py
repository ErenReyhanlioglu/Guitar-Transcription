import torch
import numpy as np
import os
import json
import logging
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

def convert_history_to_native_types(history):
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
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, experiment_path, main_exp_path, class_weights=None, writer=None, initial_history=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.experiment_path = experiment_path
        self.main_exp_path = main_exp_path
        self.writer = writer
        
        self.loss_config = self.config['loss']
        self.training_config = self.config['training']
        self.instrument_config = self.config['instrument']
        self.metrics_config = self.config['metrics']
        self.guitar_profile = GuitarProfile(self.instrument_config)
        
        self.loss_fn = CombinedLoss(config=self.config, class_weights=class_weights).to(self.device)
        self.history = initial_history if initial_history else self._create_empty_history()
        
        # --- GCS HOOK ---
        self.bottleneck = None
        self._register_bottleneck_hook()
        
        # --- UNCERTAINTY WEIGHTING ---
        if self.loss_config.get('weighting_strategy') == 'uncertainty':
            loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
            if loss_params:
                active_opt = self.training_config['optimizer']['active_optimizer']
                opt_conf = self.training_config['optimizer']['configurations'].get(active_opt, {})
                loss_lr = opt_conf.get('params', {}).get('loss_lr', None)

                param_group = {'params': loss_params}
                if loss_lr is not None:
                    param_group['lr'] = loss_lr
                    logger.info(f"Uncertainty params added with custom LR: {loss_lr}")
                else:
                    logger.info("Uncertainty params added with default Optimizer LR.")

                self.optimizer.add_param_group(param_group)
        
        self.use_amp = self.training_config.get('use_mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = self._setup_early_stopping()
        
        logger.info("Trainer initialized.")

    def _register_bottleneck_hook(self):
        """
        Modelin bottleneck (ortak özellik) vektörünü yakalar.
        CASCADED GÜNCELLEMESİ: Tablature kafasının girişi artık saf bottleneck değil (concat edilmiş).
        Bu yüzden saf bottleneck kullanan bir 'Auxiliary Head'e kanca atıyoruz.
        """
        def hook_fn(module, input):
            # Input bir tuple'dır (tensor, )
            self.bottleneck = input[0]
            if self.bottleneck.requires_grad:
                self.bottleneck.retain_grad()
        
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # Hangi kafa saf bottleneck kullanıyor? Öncelik sırasına göre kontrol et.
        target_head = None
        hook_name = ""

        if hasattr(model_ref, 'heads'):
            if 'multipitch' in model_ref.heads:
                target_head = model_ref.heads['multipitch']
                hook_name = "multipitch"
            elif 'hand_position' in model_ref.heads:
                target_head = model_ref.heads['hand_position']
                hook_name = "hand_position"
            elif 'string_activity' in model_ref.heads:
                target_head = model_ref.heads['string_activity']
                hook_name = "string_activity"
            elif 'pitch_class' in model_ref.heads:
                target_head = model_ref.heads['pitch_class']
                hook_name = "pitch_class"
            elif 'tablature' in model_ref.heads:
                # Eğer hiçbir aux head yoksa mecburen tab'a takarız 
                target_head = model_ref.heads['tablature']
                hook_name = "tablature (Fallback)"

        if target_head is not None:
            target_head.register_forward_pre_hook(hook_fn)
            logger.info(f"GCS Hook registered successfully on 'heads.{hook_name}' (Pure Bottleneck Source).")
        else:
            logger.warning("No suitable head found for GCS Hook! GCS analysis will be DISABLED.")

    def _setup_early_stopping(self):
        es_config = self.training_config.get('early_stopping', {})
        if es_config.get('enabled', False):
            monitor = es_config.get('monitor', 'val_tab_f1')
            return EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0),
                monitor=monitor,
                mode=es_config.get('mode', 'max')
            )
        return None

    def _create_empty_history(self):
        loss_keys = [
            "loss_total", "loss_tablature", "loss_hand_position", 
            "loss_string_activity", "loss_pitch_class", "loss_multipitch", "loss_onset"
        ]
        tab_keys = ["tab_f1", "tab_precision", "tab_recall", "tdr"]
        
        hand_keys = ["hand_pos_acc", "hand_pos_f1", "hand_pos_precision", "hand_pos_recall"]
        string_keys = ["string_act_f1", "string_act_precision", "string_act_recall"]
        pitch_keys = ["pitch_class_f1", "pitch_class_precision", "pitch_class_recall"]
        mp_head_keys = ["mp_head_f1", "mp_head_precision", "mp_head_recall"]
        onset_keys = ["onset_f1", "onset_precision", "onset_recall"]
        
        mp_derived_keys = ["mp_f1", "mp_precision", "mp_recall"] 
        octave_keys = ["octave_f1", "octave_precision", "octave_recall"]
        
        error_keys = [
            "tab_error_total_rate", "tab_error_substitution_rate", "tab_error_miss_rate", 
            "tab_error_false_alarm_rate", "tab_error_duplicate_pitch_rate",
            "tab_error_total_count", "tab_error_substitution_count", "tab_error_miss_count", 
            "tab_error_false_alarm_count", "tab_error_duplicate_pitch_count"
        ]

        # GCS keys ayrı oluşturuluyor
        gcs_keys = []
        heads = self.config['model'].get('params', {}).get('heads', {})
        for task in heads:
            if task != 'tablature' and heads[task].get('enabled', False):
                gcs_keys.append(f"gcs_tab_{task}")

        # GCS keys hariç diğerleri train/val prefix alacak
        all_keys = loss_keys + tab_keys + hand_keys + string_keys + pitch_keys + mp_head_keys + onset_keys + mp_derived_keys + octave_keys + error_keys
        
        history = {f"{phase}_{key}": [] for phase in ["train", "val"] for key in all_keys}
        
        # GCS değerlerini prefixsiz (yalın) olarak ekle
        for k in gcs_keys:
            history[k] = []

        history["lr"] = [] 
        if self.loss_config.get('weighting_strategy') == 'uncertainty':
            history['sigma_weights'] = []
        return history
    
    def _prepare_model_input(self, batch):
        return {key: tensor.to(self.device) for key, tensor in batch['features'].items()}
    
    def _compute_epoch_gcs(self, model_output, batch):
        gcs_results = {}
        if self.bottleneck is None: 
            return gcs_results

        try:
            with autocast(enabled=self.use_amp):
                loss_dict = self.loss_fn(model_output, batch)
            
            tab_key = 'tablature_loss' if 'tablature_loss' in loss_dict else 'tablature'
            
            if tab_key not in loss_dict:
                return gcs_results
                
            tab_loss = loss_dict[tab_key]
            
            # 1. Ana Görev (Tablature) Gradyanı
            grad_tab = torch.autograd.grad(
                tab_loss, 
                self.bottleneck, 
                retain_graph=True, 
                allow_unused=True
            )[0]
            
            if grad_tab is None: 
                return gcs_results
            
            grad_tab_flat = grad_tab.flatten()

            # 2. Yardımcı Görevler
            for key, val in loss_dict.items():
                if key == 'total_loss' or key == tab_key: continue
                
                task_name = key.replace('_loss', '')

                if not isinstance(val, torch.Tensor): continue

                grad_aux = torch.autograd.grad(
                    val, 
                    self.bottleneck, 
                    retain_graph=True, 
                    allow_unused=True
                )[0]
                
                if grad_aux is not None:
                    grad_aux_flat = grad_aux.flatten()
                    
                    sim = torch.nn.functional.cosine_similarity(
                        grad_tab_flat.unsqueeze(0), 
                        grad_aux_flat.unsqueeze(0), 
                        eps=1e-8
                    ).item()
                    
                    # Key formatı yalın: 'gcs_tab_hand_position'
                    gcs_results[f"gcs_tab_{task_name}"] = sim
                    
        except Exception as e:
            logger.warning(f"GCS calculation ERROR: {e}")
            
        return gcs_results

    def _run_epoch(self, data_loader, is_training=True):
        self.model.train(is_training)
        
        epoch_stats = {
            "loss_total": 0.0,
            "loss_breakdown": {},
            "tab_logits": [], "tab_targets": [],
            "aux_hand_pos_logits": [], "aux_hand_pos_targets": [],
            "aux_activity_logits": [], "aux_activity_targets": [],
            "aux_pitch_class_logits": [], "aux_pitch_class_targets": [],
            "aux_multipitch_logits": [], "aux_multipitch_targets": [],
            "aux_onset_logits": [], "aux_onset_targets": [],
            "gcs_values": {} 
        }
        
        desc = "Training" if is_training else "Validating"
        num_batches = len(data_loader)
        
        for i, batch in enumerate(tqdm(data_loader, desc=desc, leave=False)):
            inputs = self._prepare_model_input(batch)
            for key in batch:
                if key != 'features':
                    batch[key] = batch[key].to(self.device)

            with torch.set_grad_enabled(is_training):
                self.optimizer.zero_grad(set_to_none=True)
                self.bottleneck = None 
                
                with autocast(enabled=self.use_amp):                    
                    model_output = self.model(inputs)
                    loss_dict = self.loss_fn(model_output, batch)
                    loss_to_backprop = loss_dict['total_loss']
                
                if is_training and i == (num_batches - 1):
                    gcs_res = self._compute_epoch_gcs(model_output, batch)
                    epoch_stats["gcs_values"] = gcs_res

                if is_training and torch.isfinite(loss_to_backprop):
                    self.scaler.scale(loss_to_backprop).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if torch.isfinite(loss_to_backprop):
                epoch_stats["loss_total"] += loss_to_backprop.item()
                for k, v in loss_dict.items():
                    if k == 'total_loss': continue
                    current_val = epoch_stats["loss_breakdown"].get(k, 0.0)
                    epoch_stats["loss_breakdown"][k] = current_val + v.item()
            
            # --- OUTPUT COLLECTION ---
            if 'tab_logits' in model_output:
                epoch_stats["tab_logits"].append(model_output['tab_logits'].cpu().detach())
                epoch_stats["tab_targets"].append(batch['tablature'].cpu().detach())
            
            if 'hand_pos_logits' in model_output:
                epoch_stats["aux_hand_pos_logits"].append(model_output['hand_pos_logits'].cpu().detach())
                epoch_stats["aux_hand_pos_targets"].append(batch['hand_pos_target'].cpu().detach())
            
            if 'activity_logits' in model_output:
                epoch_stats["aux_activity_logits"].append(model_output['activity_logits'].cpu().detach())
                epoch_stats["aux_activity_targets"].append(batch['activity_target'].cpu().detach())

            if 'pitch_class_logits' in model_output:
                epoch_stats["aux_pitch_class_logits"].append(model_output['pitch_class_logits'].cpu().detach())
                epoch_stats["aux_pitch_class_targets"].append(batch['pitch_class_target'].cpu().detach())

            if 'multipitch_logits' in model_output and 'multipitch_target' in batch:
                epoch_stats["aux_multipitch_logits"].append(model_output['multipitch_logits'].cpu().detach())
                epoch_stats["aux_multipitch_targets"].append(batch['multipitch_target'].cpu().detach())

            if 'onset_logits' in model_output and 'onset_target' in batch:
                epoch_stats["aux_onset_logits"].append(model_output['onset_logits'].cpu().detach())
                epoch_stats["aux_onset_targets"].append(batch['onset_target'].cpu().detach())

        # Metrics
        epoch_metrics = self._calculate_all_metrics(epoch_stats)
        
        num_batches = max(1, len(data_loader))
        epoch_metrics['loss_total'] = epoch_stats["loss_total"] / num_batches
        
        for k, v in epoch_stats["loss_breakdown"].items():
            clean_name = k.replace("loss_", "").replace("_loss", "")
            metric_key = f"loss_{clean_name}"
            epoch_metrics[metric_key] = v / num_batches
            
        if epoch_stats["gcs_values"]:
            for k, v in epoch_stats["gcs_values"].items():
                epoch_metrics[k] = v

        return epoch_metrics

    def _calculate_all_metrics(self, stats):
        final_metrics = {}
        include_silence = self.metrics_config.get('include_silence', False)
        tab_silence_class = self.instrument_config['silence_class']
        
        if stats["tab_logits"]:
            all_logits = torch.cat(stats["tab_logits"], dim=0)
            all_targets = torch.cat(stats["tab_targets"], dim=0)
            S = self.instrument_config['num_strings']
            C = self.config['model']['params']['num_classes']

            if all_logits.dim() == 2: 
                preds_tab = torch.argmax(all_logits.view(-1, S, C), dim=-1)
            elif all_logits.dim() == 3: 
                preds_tab = torch.argmax(all_logits, dim=-1)
            else: 
                preds_tab = torch.argmax(all_logits, dim=1) 

            targets_flat = all_targets.view(-1, S)
            
            tab_res = compute_tablature_metrics(preds_tab, targets_flat, include_silence, tab_silence_class)
            final_metrics.update({
                'tab_f1': tab_res.get('overall_f1'), 
                'tab_precision': tab_res.get('overall_precision'), 
                'tab_recall': tab_res.get('overall_recall')
            })
            
            mp_res = compute_multipitch_metrics(preds_tab, targets_flat, self.guitar_profile, include_silence, tab_silence_class)
            final_metrics.update({
                'mp_f1': mp_res.get('multipitch_f1'), 
                'mp_precision': mp_res.get('multipitch_precision'), 
                'mp_recall': mp_res.get('multipitch_recall')
            })

            oct_res = compute_octave_tolerant_metrics(preds_tab, targets_flat, self.instrument_config['tuning'], tab_silence_class)
            final_metrics.update({
                'octave_f1': oct_res.get('octave_f1'), 
                'octave_precision': oct_res.get('octave_precision'), 
                'octave_recall': oct_res.get('octave_recall')
            })
            
            final_metrics['tdr'] = compute_tdr(preds_tab, targets_flat, self.guitar_profile, include_silence, tab_silence_class)
            
            err_res = compute_tablature_error_scores(preds_tab, targets_flat, tab_silence_class, self.instrument_config['tuning'])
            for k, v in err_res.items(): 
                final_metrics[k] = v
        
        if stats["aux_hand_pos_logits"]:
            hp_logits = torch.cat(stats["aux_hand_pos_logits"])
            hp_targets = torch.cat(stats["aux_hand_pos_targets"])
            hp_preds = torch.argmax(hp_logits, dim=-1)
            final_metrics.update(compute_hand_position_metrics(hp_preds, hp_targets, include_silence, silence_class=0))

        if stats["aux_activity_logits"]:
            act_logits = torch.cat(stats["aux_activity_logits"])
            act_targets = torch.cat(stats["aux_activity_targets"])
            final_metrics.update(compute_string_activity_metrics(act_logits, act_targets, include_silence))

        if stats["aux_pitch_class_logits"]:
            pc_logits = torch.cat(stats["aux_pitch_class_logits"])
            pc_targets = torch.cat(stats["aux_pitch_class_targets"])
            final_metrics.update(compute_pitch_class_metrics(pc_logits, pc_targets, include_silence))

        if stats.get("aux_multipitch_logits"):
            mp_h_logits = torch.cat(stats["aux_multipitch_logits"])
            mp_h_targets = torch.cat(stats["aux_multipitch_targets"])
            final_metrics.update(compute_aux_multipitch_metrics(mp_h_logits, mp_h_targets, include_silence))

        if stats.get("aux_onset_logits"):
            on_logits = torch.cat(stats["aux_onset_logits"])
            on_targets = torch.cat(stats["aux_onset_targets"])
            on_res = compute_aux_multipitch_metrics(on_logits, on_targets, include_silence)
            final_metrics.update({
                'onset_f1': on_res['mp_head_f1'],
                'onset_precision': on_res['mp_head_precision'],
                'onset_recall': on_res['mp_head_recall']
            })

        return final_metrics
        
    def train(self, start_epoch=0, best_val_metric=-1):
        self.model.to(self.device)
        monitor_metric_key = self.early_stopper.monitor if self.early_stopper else 'val_tab_f1'
        if best_val_metric == -1: best_val_metric = -np.inf
        
        epochs = self.training_config['epochs']
        logger.info(f"--- Starting Multi-Task Training with GCS Analysis ---")

        for epoch in range(start_epoch, epochs):
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics = self._run_epoch(self.val_loader, is_training=False)
            self._update_and_log_history(epoch, epochs, train_metrics, val_metrics)
            
            current_metric_val = val_metrics.get(monitor_metric_key.replace('val_',''), 0.0)
            if self.scheduler: self.scheduler.step(current_metric_val)
            
            is_best = current_metric_val > best_val_metric
            if is_best: best_val_metric = current_metric_val
            self._save_checkpoint(epoch, best_val_metric, is_best)
            if self.early_stopper:
                self.early_stopper(current_metric_val)
                if self.early_stopper.early_stop: break
        
        self._finalize_training()
        return self.model, self.history

    def _update_and_log_history(self, epoch, total_epochs, train_metrics, val_metrics):
        for key in self.history.keys():
            if key == 'lr':
                self.history[key].append(self.optimizer.param_groups[0]['lr'])
            elif key == 'sigma_weights':
                if hasattr(self.loss_fn, 'uncertainty_wrapper'):
                    sigmas = {k: v.item() for k, v in self.loss_fn.uncertainty_wrapper.log_vars.items()}
                    self.history[key].append(sigmas)
            elif key.startswith('gcs_'):
                 # GCS değerleri sadece train_metrics içinde 'gcs_tab_...' olarak bulunur ve prefix almaz.
                 self.history[key].append(train_metrics.get(key, 0.0))
            elif key.startswith('train_'):
                clean = key.replace('train_', '')
                self.history[key].append(train_metrics.get(clean, 0.0))
            elif key.startswith('val_'):
                clean = key.replace('val_', '')
                self.history[key].append(val_metrics.get(clean, 0.0))

        lr = self.history['lr'][-1]
        log_epoch_summary(logger, epoch, total_epochs, lr, train_metrics, val_metrics)

    def _save_checkpoint(self, epoch, best_val_metric, is_best):
        state = {
            'epoch': epoch, 
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': best_val_metric, 
            'history': self.history
        }
        latest_path = os.path.join(self.experiment_path, 'checkpoint_latest.pt')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.experiment_path, 'model_best.pt')
            torch.save(self.model.state_dict(), best_path)
    
    def _finalize_training(self):
        logger.info("--- Training finished ---")
        if self.writer: self.writer.close()
        
        history_path = os.path.join(self.experiment_path, "history.json")
        try:
            native_history = convert_history_to_native_types(self.history)
            with open(history_path, 'w') as f:
                json.dump(native_history, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            
        try:
            logger.info("Running Final Error Analysis...")
            run_error_analysis(experiment_path=self.experiment_path, main_exp_path=self.main_exp_path, val_loader=self.val_loader)
        except Exception as e:
             logger.warning(f"Error analysis failed: {e}")

        save_model_summary(self.model, self.config, self.experiment_path)
        try:
            generate_experiment_report(
                model=self.model, history=self.history, val_loader=self.val_loader,
                config=self.config, experiment_path=self.experiment_path, device=self.device,
                profile=self.guitar_profile, include_silence=self.metrics_config.get('include_silence', False),
                silence_class=self.instrument_config['silence_class']
            )
        except Exception as e:
             logger.warning(f"Report generation failed: {e}")
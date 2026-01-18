import logging
import sys
import torch
import numpy as np

def setup_logger(config):
    """
    Sets up a logger with a specified name, log file, and logging level.
    Clears existing handlers to prevent duplicate logs.
    """
    log_config = config.get('logging_and_checkpointing', {})
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    log_levels = log_config.get('log_levels', {})
    for module, level_str in log_levels.items():
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger(module).setLevel(level)
        
    logging.info(f"Logger initialized (Handlers reset).")
    return root_logger

def describe(data):
    """
    Helper to describe data stats for debugging.
    Handles Tensors, Arrays, Dicts, and other types safely.
    """
    if data is None:
        return "None"
    
    if isinstance(data, (torch.Tensor, np.ndarray)):
        return f"Shape: {data.shape} | Type: {data.dtype} | Min: {data.min()} | Max: {data.max()}"
    
    if isinstance(data, (dict, list, tuple)):
        return f"{str(data)[:200]}..." 
        
    return str(data)

def log_epoch_summary(logger, epoch, total_epochs, lr, train_metrics, val_metrics):
    """
    Eğitim ve Validasyon metriklerini EKSİKSİZ, detaylı bir tablo olarak basar.
    GCS (Task Affinity) bölümü eklendi.
    """
    def g(d, k): return d.get(k, 0.0)
    
    # Yardımcı: Satır Formatlayıcı (Label | Train | Val)
    def row(label, key, fmt=".4f", suffix=""):
        tr_val = g(train_metrics, key)
        va_val = g(val_metrics, key)
        
        if "count" in key:
            tr_str = f"{int(tr_val)}"
            va_str = f"{int(va_val)}"
        else:
            tr_str = f"{tr_val:{fmt}}"
            va_str = f"{va_val:{fmt}}"
            
        return f"   {label:<25} | {tr_str:<12} | {va_str:<12} {suffix}"

    width = 85
    sep = "-" * width
    double_sep = "=" * width
    
    lines = []
    lines.append("\n" + double_sep)
    lines.append(f" EPOCH {epoch+1:02d}/{total_epochs} | LR: {lr:.2e} ".center(width))
    lines.append(double_sep)
    lines.append(f"   {'METRIC':<25} | {'TRAIN':<12} | {'VALIDATION':<12}")
    lines.append(sep)

    # --- 1. LOSSES (HEPSİ) ---
    lines.append(f"  [LOSS BREAKDOWN]")
    all_loss_keys = sorted(list(set([k for k in train_metrics.keys() if k.startswith('loss_')] + 
                                    [k for k in val_metrics.keys() if k.startswith('loss_')])))
    
    if 'loss_total' in all_loss_keys:
        all_loss_keys.remove('loss_total')
        lines.append(row("TOTAL LOSS", "loss_total", fmt=".5f"))
    
    for k in all_loss_keys:
        label = k.replace('loss_', '').replace('_', ' ').title()
        lines.append(row(label, k, fmt=".5f"))
    lines.append(sep)

    # --- 2. MAIN TASK (TABLATURE) ---
    lines.append(f"  [TABLATURE (Main)]")
    lines.append(row("F1 Score", "tab_f1"))
    lines.append(row("Precision", "tab_precision"))
    lines.append(row("Recall", "tab_recall"))
    lines.append(row("TDR", "tdr"))
    lines.append(sep)

    # --- 3. ERROR ANALYSIS (RATES & COUNTS) ---
    lines.append(f"  [ERRORS (Rates & Counts)]")
    def err_row(label, base_key):
        r_key, c_key = f"{base_key}_rate", f"{base_key}_count"
        tr_r, tr_c = g(train_metrics, r_key), int(g(train_metrics, c_key))
        va_r, va_c = g(val_metrics, r_key), int(g(val_metrics, c_key))
        return f"   {label:<25} | {tr_r:.4f} ({tr_c})  | {va_r:.4f} ({va_c})"

    lines.append(err_row("Total Error", "tab_error_total"))
    lines.append(err_row(" > Substitution", "tab_error_substitution"))
    lines.append(err_row(" > Miss (FN)", "tab_error_miss"))
    lines.append(err_row(" > False Alarm (FP)", "tab_error_false_alarm"))
    lines.append(err_row(" > Duplicate Pitch", "tab_error_duplicate_pitch"))
    lines.append(sep)

    # --- 4. AUX TASKS & EXTRAS ---
    task_groups = [
        ("HAND POSITION", "hand_pos_", ["acc", "f1", "precision", "recall"]),
        ("STRING ACTIVITY", "string_act_", ["f1", "precision", "recall"]),
        ("PITCH CLASS", "pitch_class_", ["f1", "precision", "recall"]),
        ("MULTIPITCH (Head)", "mp_head_", ["f1", "precision", "recall"]),
        ("ONSET (Head)", "onset_", ["f1", "precision", "recall"]),
        ("MULTIPITCH (Derived)", "mp_", ["f1", "precision", "recall"]),
        ("OCTAVE TOLERANT", "octave_", ["f1", "precision", "recall"]),
    ]

    for title, prefix, metrics in task_groups:
        # Validasyon metriklerinde bu task var mı kontrol et
        # (Eğitimde batch atlandığı için bazen olmayabilir ama val kesin referanstır)
        check_key = f"{prefix}f1" if "f1" in metrics else f"{prefix}acc"
        if check_key in val_metrics:
            lines.append(f"  [{title}]")
            for m in metrics:
                full_key = f"{prefix}{m}"
                label = m.title() if m != "acc" else "Accuracy"
                lines.append(row(label, full_key))
            lines.append(sep)

    # --- 5. GCS (TASK AFFINITY) ---
    # Sadece Training metrics içinde 'gcs_tab_' ile başlayan keyler aranır.
    gcs_keys = [k for k in train_metrics.keys() if k.startswith("gcs_tab_")]
    if gcs_keys:
        lines.append(f"  [TASK AFFINITY (Grad Cosine Sim vs Tablature)]")
        for k in sorted(gcs_keys):
            # Örn: gcs_tab_string_activity -> String Activity
            label = k.replace("gcs_tab_", "").replace("_", " ").title()
            val = train_metrics.get(k, 0.0)
            
            # Görsel ipucu (Pozitif/Negatif/Nötr)
            symbol = "(+)" if val > 0.05 else ("(-)" if val < -0.05 else "(o)")
            
            # Val sütunu boş bırakılır çünkü validation sırasında hesaplanmaz
            val_str = f"{val:+.4f} {symbol}"
            
            lines.append(f"   {label:<25} | {val_str:<12} | {'-':<12}")
        lines.append(sep)

    lines.append(double_sep)

    for line in lines:
        logger.info(line)
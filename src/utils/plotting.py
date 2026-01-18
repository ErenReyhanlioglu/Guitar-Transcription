import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from pathlib import Path
import matplotlib.patches as mpatches 
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage 

# Sunucu taraflı (Colab/Headless) çalışma için Agg backend
matplotlib.use('Agg')

def plot_loss_curves(history: dict, save_path: str or Path):
    """
    Plots and saves training and validation loss curves.
    Dynamically finds all loss keys (e.g., loss_tablature, loss_hand_position).
    """
    plt.figure(figsize=(12, 7))
    
    # 1. Ana Loss (Total)
    if 'train_loss_total' in history:
        plt.plot(history['train_loss_total'], label='Train Total', color='black', linewidth=2, linestyle='--')
    elif 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Total', color='black', linewidth=2, linestyle='--')
        
    if 'val_loss_total' in history:
        plt.plot(history['val_loss_total'], label='Val Total', color='black', linewidth=2)
    elif 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Total', color='black', linewidth=2)

    # 2. Alt Görev Lossları (Opsiyonel - Kalabalık olmasın diye sadece Val çizdirilebilir)
    # Renk paleti oluştur
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    val_sub_keys = [k for k in history.keys() if k.startswith('val_loss_') and k != 'val_loss_total']
    
    for i, key in enumerate(val_sub_keys):
        label_name = key.replace('val_loss_', '').replace('_', ' ').title()
        color = colors[i % len(colors)]
        plt.plot(history[key], label=f'{label_name} (Val)', color=color, linewidth=1.5, alpha=0.8)

    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metrics_custom(history: dict, val_metric_key: str, train_metric_key: str, plot_title: str, save_path: str or Path):
    """
    Plots comparison of a specific metric between Train and Val.
    """
    if val_metric_key not in history and train_metric_key not in history:
        return

    plt.figure(figsize=(12, 7))
    
    if train_metric_key in history and len(history[train_metric_key]) > 0:
        plt.plot(history[train_metric_key], label=f'Training', color='royalblue', marker='o', markersize=4, linestyle='--')
    
    if val_metric_key in history and len(history[val_metric_key]) > 0:
        plt.plot(history[val_metric_key], label=f'Validation', color='orangered', marker='x', markersize=5)
    
    plt.title(f'{plot_title} Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(plot_title, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_spectrogram(spec_data, hop_seconds, save_path, title='Spectrogram', y_axis='Frequency Bin'):
    """Plots and saves a Spectrogram."""
    plt.figure(figsize=(15, 6))
    # Eksen kontrolü: (Freq, Time) olmalı
    if spec_data.shape[0] > spec_data.shape[1]: 
        # Eğer Freq > Time ise muhtemelen doğrudur, ama kareye yakınsa karışabilir.
        # Genelde (F, T) gelir.
        pass
        
    duration_seconds = spec_data.shape[1] * hop_seconds
    img = plt.imshow(spec_data, aspect='auto', origin='lower', cmap='magma', 
                     extent=[0, duration_seconds, 0, spec_data.shape[0]])
    plt.title(title, fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(y_axis, fontsize=12)
    plt.colorbar(img, label='Magnitude')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pianoroll(pianoroll, hop_seconds, save_path, title='Piano Roll', low_midi=21):
    """
    Plots a pianoroll heatmap. Expects (Pitch, Time).
    """
    plt.figure(figsize=(15, 6))
    if pianoroll.shape[0] > pianoroll.shape[1]: # Muhtemelen (Time, Pitch) gelmiş, çevir
         pianoroll = pianoroll.T
         
    num_pitches, num_frames = pianoroll.shape
    duration_seconds = num_frames * hop_seconds
    extent = [0, duration_seconds, low_midi - 0.5, low_midi + num_pitches - 0.5]
    
    plt.imshow(pianoroll, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    plt.title(title, fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('MIDI Pitch', fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_guitar_tablature(tab_data, hop_seconds, save_path, title='Guitar Tablature'):
    """
    Plots guitar tablature.
    Auto-corrects shape: Expects (Strings, Time), fixes if (Time, Strings).
    """
    # Düzeltme: (Time, Strings) -> (Strings, Time)
    if tab_data.shape[0] > tab_data.shape[1] and tab_data.shape[1] == 6:
        tab_data = tab_data.T
    elif tab_data.shape[0] == 6:
        pass # Zaten (6, T)
        
    fig, ax = plt.subplots(figsize=(15, 6))
    num_strings, num_frames = tab_data.shape # Artık (6, T) olduğundan emin
    duration_seconds = num_frames * hop_seconds
    
    # Tel Çizgileri
    for i in range(num_strings):
        ax.plot([0, duration_seconds], [i, i], color='grey', linewidth=0.75)
        
    for string_idx in range(num_strings):
        is_note_active = False
        onset_frame = 0
        active_fret = -1
        
        string_data = tab_data[string_idx, :] 
        
        for frame_idx, fret in enumerate(string_data):
            current_fret = int(fret)
            # Not: -1 veya 20 (silence class) sessiz kabul edilir
            is_silence = (current_fret == -1) or (current_fret >= 20) 
            
            if not is_silence and not is_note_active:
                is_note_active = True
                onset_frame = frame_idx
                active_fret = current_fret
            elif (is_silence or current_fret != active_fret or frame_idx == num_frames - 1) and is_note_active:
                is_note_active = False
                offset_frame = frame_idx if frame_idx == num_frames - 1 and not is_silence else frame_idx - 1
                
                onset_time = onset_frame * hop_seconds
                offset_time = (offset_frame + 1) * hop_seconds
                
                # Notaları Çiz
                ax.text(onset_time, string_idx, str(active_fret), 
                        va='center', ha='center', fontsize=9, 
                        bbox=dict(boxstyle='circle', facecolor='lightblue', ec='black', lw=0.5))
                ax.plot([onset_time, offset_time], [string_idx, string_idx], color='blue', linewidth=2.5)
                
                if not is_silence:
                    is_note_active = True
                    onset_frame = frame_idx
                    active_fret = current_fret

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('String', fontsize=12)
    ax.set_yticks(range(num_strings))
    ax.set_yticklabels([f'EADGBe'[i] for i in range(num_strings-1, -1, -1)])
    ax.set_ylim(-0.5, num_strings - 0.5)
    ax.set_xlim(0, duration_seconds)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tablature_errors(preds_np, targets_np, hop_seconds, silence_class, save_path, title="Tablature Error Visualization"):
    """
    Plots detailed error map for tablature (TP, FN, FP, Sub).
    Auto-corrects shape: Expects (Strings, Time).
    """
    # Düzeltme: (Time, Strings) -> (Strings, Time)
    if preds_np.shape[0] > preds_np.shape[1] and preds_np.shape[1] == 6:
        preds_np = preds_np.T
    if targets_np.shape[0] > targets_np.shape[1] and targets_np.shape[1] == 6:
        targets_np = targets_np.T

    fig, ax = plt.subplots(figsize=(20, 8))
    num_strings, num_frames = targets_np.shape
    duration_seconds = num_frames * hop_seconds

    colors = {"TP": "green", "FN": "red", "FP": "blue", "SUB": "purple"}

    for s in range(num_strings):
        preds_s, targets_s = preds_np[s, :], targets_np[s, :]

        is_tp = (preds_s == targets_s) & (targets_s != silence_class)
        is_fn = (preds_s == silence_class) & (targets_s != silence_class)
        is_fp = (preds_s != silence_class) & (targets_s == silence_class)
        is_sub = (preds_s != targets_s) & (preds_s != silence_class) & (targets_s != silence_class)

        for error_type, mask in [("TP", is_tp), ("FN", is_fn), ("FP", is_fp), ("SUB", is_sub)]:
            if not np.any(mask): continue
            
            labels, num_labels = ndimage.label(mask)
            for i in range(1, num_labels + 1):
                segment_indices = np.where(labels == i)[0]
                if len(segment_indices) == 0: continue

                start_frame, end_frame = segment_indices[0], segment_indices[-1]
                start_time, end_time = start_frame * hop_seconds, (end_frame + 1) * hop_seconds
                text_time_pos = (start_time + end_time) / 2
                
                ax.hlines(s, start_time, end_time, color=colors[error_type], linewidth=5, capstyle='butt') 

                if error_type == "TP":
                    fret = targets_s[start_frame]
                    ax.text(text_time_pos, s, str(fret), color='white', ha='center', va='center', fontsize=7, weight='bold')
                elif error_type == "FN":
                    fret = targets_s[start_frame]
                    ax.text(text_time_pos, s - 0.1, str(fret), color=colors[error_type], ha='center', va='top', fontsize=8)
                elif error_type == "FP":
                    fret = preds_s[start_frame]
                    ax.text(text_time_pos, s + 0.1, str(fret), color=colors[error_type], ha='center', va='bottom', fontsize=8)
                elif error_type == "SUB":
                    pred_fret = preds_s[start_frame]
                    target_fret = targets_s[start_frame]
                    ax.text(text_time_pos, s + 0.1, str(pred_fret), color=colors[error_type], ha='center', va='bottom', fontsize=8)
                    ax.text(text_time_pos, s - 0.1, f"({target_fret})", color='orange', ha='center', va='top', fontsize=8)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Strings', fontsize=12)
    ax.set_yticks(range(num_strings))
    ax.set_yticklabels([f'String {num_strings-i}' for i in range(num_strings)]) 
    ax.set_ylim(-0.5, num_strings - 0.5)
    ax.set_xlim(0, duration_seconds)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    legend_patches = [
        mpatches.Patch(color=colors['TP'], label='TP (Correct)'),
        mpatches.Patch(color=colors['FN'], label='FN (Miss)'),
        mpatches.Patch(color=colors['FP'], label='FP (False Alarm)'),
        mpatches.Patch(color=colors['SUB'], label='SUB (Wrong Fret)')
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_pianoroll_errors(pred_pianoroll, target_pianoroll, hop_seconds, save_path, low_midi=21, title="Pianoroll Error Visualization"):
    """Plots Pianoroll errors (Green=TP, Red=FN, Blue=FP)."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Ensure (Pitch, Time)
    if pred_pianoroll.shape[0] > pred_pianoroll.shape[1]: pred_pianoroll = pred_pianoroll.T
    if target_pianoroll.shape[0] > target_pianoroll.shape[1]: target_pianoroll = target_pianoroll.T

    error_matrix = np.zeros_like(target_pianoroll, dtype=int)
    error_matrix[ (target_pianoroll == 1) & (pred_pianoroll == 1) ] = 1 # TP
    error_matrix[ (target_pianoroll == 1) & (pred_pianoroll == 0) ] = 2 # FN
    error_matrix[ (target_pianoroll == 0) & (pred_pianoroll == 1) ] = 3 # FP
    
    cmap = ListedColormap(['#FFFFFF', 'green', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    num_pitches, num_frames = target_pianoroll.shape
    duration_seconds = num_frames * hop_seconds
    extent = [0, duration_seconds, low_midi - 0.5, low_midi + num_pitches - 0.5]
    
    ax.imshow(error_matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('MIDI Pitch', fontsize=12)
    
    legend_patches = [mpatches.Patch(color='green', label='TP'), mpatches.Patch(color='red', label='FN'), mpatches.Patch(color='blue', label='FP')]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_binary_activation(activation_matrix, hop_seconds, save_path=None, title="Binary Activation"):
    """
    Plots binary activation heatmaps (e.g., String Activity, Pitch Class, Hand Pos).
    Auto-detects labels based on channel count.
    Ensures (Channels, Time) orientation.
    """
    if activation_matrix is None or activation_matrix.size == 0:
        return

    # Otomatik Dönüş (Time > Channels varsayımıyla)
    # Örn: (500, 6) gelirse (6, 500) yap
    if activation_matrix.shape[0] > activation_matrix.shape[1] and activation_matrix.shape[1] <= 12:
        activation_matrix = activation_matrix.T

    num_channels = activation_matrix.shape[0]
    num_frames = activation_matrix.shape[1]

    fig, ax = plt.subplots(figsize=(15, 5))
    
    img = ax.imshow(
        activation_matrix, 
        aspect='auto', 
        origin='lower', 
        interpolation='nearest',
        cmap='Greys', 
        extent=[0, num_frames * hop_seconds, -0.5, num_channels - 0.5]
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Class")
    
    # Dinamik Etiketleme
    ax.set_yticks(np.arange(num_channels))
    if num_channels == 6:
        ax.set_yticklabels([f'Str {i+1}' for i in range(num_channels)])
    elif num_channels == 12: 
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        ax.set_yticklabels(chroma_names)
    elif num_channels == 5: 
        hp_names = ['Open', 'Low', 'Mid', 'High', 'V.High']
        ax.set_yticklabels(hp_names)
    else:
        ax.set_yticklabels([f'Ch {i}' for i in range(num_channels)])

    plt.tight_layout()

    if save_path:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        except Exception as e:
            print(f"WARNING: {save_path}. Cause of: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True, save_path=None):
    """Plots and saves confusion matrix."""
    accuracy = np.trace(cm) / float(np.sum(cm)) if np.sum(cm) > 0 else 0
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(10, 8)) 
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=90)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

    cm_for_text = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) if normalize else cm

    thresh = cm_for_text.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm_for_text[i, j]
        format_str = "{:0.2f}" if normalize else "{:,}"
        ax.text(j, i, format_str.format(value),
                 horizontalalignment="center",
                 color="white" if value > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True Label')
    ax.set_xlabel(f'Predicted Label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
    
    if save_path:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
        except Exception as e:
            print(f"WARNING: {save_path}. Cause of: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()
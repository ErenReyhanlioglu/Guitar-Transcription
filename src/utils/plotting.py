import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from pathlib import Path
import matplotlib.patches as mpatches 
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage 

matplotlib.use('Agg')

def plot_loss_curves(history: dict, save_path: str or Path):
    """Plots and saves the training and validation loss curves from the history dictionary."""
    plt.figure(figsize=(12, 7))
    if 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='Training Loss', color='royalblue', marker='o', markersize=4, linestyle='--')
        plt.plot(history['val_loss'], label='Validation Loss', color='orangered', marker='x', markersize=5)
        plt.title('Training and Validation Loss Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        print("Warning: 'train_loss' or 'val_loss' not found in history. Skipping loss plot.")
    plt.close()

def plot_metrics_custom(history: dict, val_metric_key: str, train_metric_key: str, plot_title: str, save_path: str or Path):
    """
    Plots and saves a specific training and validation metric curve from the history dictionary.
    
    Args:
        history (dict): The training history dictionary.
        val_metric_key (str): The key for the validation metric in the history dict (e.g., 'val_tab_f1').
        train_metric_key (str): The key for the training metric in the history dict (e.g., 'train_tab_f1').
        plot_title (str): The title for the plot.
        save_path (str or Path): The path to save the plot image.
    """
    if val_metric_key not in history:
        print(f"Warning: Validation metric '{val_metric_key}' not found in history. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    
    if train_metric_key in history:
        plt.plot(history[train_metric_key], label=f'Training {plot_title}', color='royalblue', marker='o', markersize=4, linestyle='--')
    
    plt.plot(history[val_metric_key], label=f'Validation {plot_title}', color='orangered', marker='x', markersize=5)
    
    plt.title(f'Training and Validation {plot_title} Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(plot_title, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def _find_notes(note_grid, hop_seconds):
    """
    Finds note events (pitch, onset, offset) from a multi-pitch grid.
    The grid can be pianoroll (pitch x time) or tablature (string x time).
    """
    notes = []
    num_pitches_or_strings, num_frames = note_grid.shape
    
    for i in range(num_pitches_or_strings):
        is_note_active = False
        onset_frame = 0
        for t in range(num_frames):
            activation = note_grid[i, t]
            if activation > 0 and not is_note_active:
                is_note_active = True
                onset_frame = t
            elif (activation == 0 or t == num_frames - 1) and is_note_active:
                is_note_active = False
                offset_frame = t if t == num_frames - 1 and activation > 0 else t - 1
                
                onset_time = onset_frame * hop_seconds
                offset_time = (offset_frame + 1) * hop_seconds
                notes.append({'pitch': i, 'onset': onset_time, 'offset': offset_time})
    return notes
    
def plot_spectrogram(spec_data, hop_seconds, save_path, title='Spectrogram', y_axis='Frequency Bin'):
    """Plots and saves a 2D time-frequency representation (e.g., CQT, STFT)."""
    plt.figure(figsize=(15, 6))
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
    Plots a pianoroll (pitch-vs-time activation map) as a heatmap.
    `pianoroll` is a 2D numpy array (num_pitches x num_frames).
    """
    plt.figure(figsize=(15, 6))
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
    Plots guitar tablature, showing fret numbers on string lines.
    `tab_data` is a 2D numpy array (num_frames x 6) where values are fret numbers (-1 for silence).
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    num_frames, num_strings = tab_data.shape
    duration_seconds = num_frames * hop_seconds
    
    for i in range(num_strings):
        ax.plot([0, duration_seconds], [i, i], color='grey', linewidth=0.75)
        
    for string_idx in range(num_strings):
        is_note_active = False
        onset_frame = 0
        active_fret = -1
        
        string_data = tab_data[:, string_idx] 
        
        for frame_idx, fret in enumerate(string_data):
            current_fret = int(fret)
            if current_fret != -1 and not is_note_active:
                is_note_active = True
                onset_frame = frame_idx
                active_fret = current_fret
            elif (current_fret == -1 or current_fret != active_fret or frame_idx == num_frames - 1) and is_note_active:
                is_note_active = False
                offset_frame = frame_idx if frame_idx == num_frames - 1 and current_fret != -1 else frame_idx - 1
                
                onset_time = onset_frame * hop_seconds
                offset_time = (offset_frame + 1) * hop_seconds
                
                ax.text(onset_time, string_idx, str(active_fret), 
                        va='center', ha='center', fontsize=9, 
                        bbox=dict(boxstyle='circle', facecolor='lightblue', ec='black', lw=0.5))
                ax.plot([onset_time, offset_time], [string_idx, string_idx], color='blue', linewidth=2.5)
                
                if current_fret != -1:
                    is_note_active = True
                    onset_frame = frame_idx
                    active_fret = current_fret

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Guitar String', fontsize=12)
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
    Tablature hatalarını (Miss, False Alarm, Substitution) nota bazlı olarak gruplayarak görselleştirir.
    Art arda gelen aynı perde numaraları tek bir etiketle gösterilir.
    """
    if preds_np.shape[1] > preds_np.shape[0]:
        preds_np = preds_np.T
    if targets_np.shape[1] > targets_np.shape[0]:
        targets_np = targets_np.T

    fig, ax = plt.subplots(figsize=(20, 8))
    num_frames, num_strings = targets_np.shape
    duration_seconds = num_frames * hop_seconds

    colors = {
        "TP": "green",
        "FN": "red",
        "FP": "blue",
        "SUB": "purple"
    }

    for s in range(num_strings):
        preds_s, targets_s = preds_np[:, s], targets_np[:, s]

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
                
                #ax.hlines(s, start_time, end_time, color=colors[error_type], linewidth=5, solid_capstyle='butt')
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
    ax.set_xlabel('Zaman (s)', fontsize=12)
    ax.set_ylabel('Gitar Teli', fontsize=12)
    ax.set_yticks(range(num_strings))
    ax.set_yticklabels([f'Tel {num_strings-i}' for i in range(num_strings)]) 
    ax.set_ylim(-0.5, num_strings - 0.5)
    ax.set_xlim(0, duration_seconds)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    legend_patches = [
        mpatches.Patch(color=colors['TP'], label='TP'),
        mpatches.Patch(color=colors['FN'], label='FN/Miss'),
        mpatches.Patch(color=colors['FP'], label='FP/FA'),
        mpatches.Patch(color=colors['SUB'], label='SUB')
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Lejant için yer aç
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_pianoroll_errors(pred_pianoroll, target_pianoroll, hop_seconds, save_path, low_midi=21, title="Pianoroll Error Visualization"):
    """
    Pianoroll hatalarını (Miss, False Alarm) bir heatmap olarak görselleştirir.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Hata matrisini oluştur:
    # 0: True Negative (Doğru sessizlik) - Beyaz
    # 1: True Positive (Doğru nota) - Yeşil
    # 2: False Negative (Miss - Kaçırılan) - Kırmızı
    # 3: False Positive (False Alarm - Yanlış Alarm) - Mavi
    error_matrix = np.zeros_like(target_pianoroll, dtype=int)
    error_matrix[ (target_pianoroll == 1) & (pred_pianoroll == 1) ] = 1 # TP
    error_matrix[ (target_pianoroll == 1) & (pred_pianoroll == 0) ] = 2 # FN (Miss)
    error_matrix[ (target_pianoroll == 0) & (pred_pianoroll == 1) ] = 3 # FP (False Alarm)
    
    # Renk haritası ve sınırlar
    cmap = ListedColormap(['#FFFFFF', 'green', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Çizim
    num_pitches, num_frames = target_pianoroll.shape
    duration_seconds = num_frames * hop_seconds
    extent = [0, duration_seconds, low_midi - 0.5, low_midi + num_pitches - 0.5]
    
    ax.imshow(error_matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Zaman (s)', fontsize=12)
    ax.set_ylabel('MIDI Notası', fontsize=12)
    
    legend_patches = [
        mpatches.Patch(color='green', label='TP'),
        mpatches.Patch(color='red', label='FN/Miss'),
        mpatches.Patch(color='blue', label='FP/FA')
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_binary_activation(activation_matrix, hop_seconds, save_path=None, title="Binary Activation"):
    """
    Generates and saves a pianoroll-style plot for a binary activation matrix.
    If save_path is provided, it saves the figure and closes it. Otherwise, it shows the plot.
    """
    if activation_matrix is None or activation_matrix.size == 0:
        return

    fig, ax = plt.subplots(figsize=(20, 4))
    
    img = ax.imshow(
        activation_matrix.T, 
        aspect='auto', 
        origin='lower', 
        interpolation='nearest',
        cmap='viridis' 
    )

    ax.set_title(title, fontsize=16)
    
    num_strings = activation_matrix.shape[1]
    ax.set_ylabel("String")
    ax.set_yticks(np.arange(num_strings))
    ax.set_yticklabels([f'Str {i+1}' for i in range(num_strings)])

    num_frames = activation_matrix.shape[0]
    time_axis_ticks = np.arange(0, num_frames, max(1, num_frames // 10))
    ax.set_xticks(time_axis_ticks)
    ax.set_xticklabels([f"{tick * hop_seconds:.2f}" for tick in time_axis_ticks])
    ax.set_xlabel("Time (s)")

    fig.colorbar(img, ax=ax, label="Activation")
    plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        except Exception as e:
            print(f"WARNING: {save_path}. Cause of: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True, save_path=None):
    """
    Generates and saves a plot for a given sklearn confusion matrix.
    If save_path is provided, it saves the figure and closes it. Otherwise, it shows the plot.
    """
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
                 color="white")

    fig.tight_layout()
    ax.set_ylabel('True Label')
    ax.set_xlabel(f'Predicted Label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
    
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
        except Exception as e:
            print(f"WARNING: {save_path}. Cause of: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()
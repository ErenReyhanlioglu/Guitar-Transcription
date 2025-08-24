import matplotlib.pyplot as plt
import numpy as np
import itertools
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib

# Use a non-interactive backend to prevent issues in environments without a GUI
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

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(14, 12)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=9)
        plt.yticks(tick_marks, target_names, fontsize=9)

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(cm.astype('float'), row_sums, 
                                  out=np.zeros_like(cm, dtype=float), 
                                  where=row_sums!=0)

    thresh = cm_normalized.max() / 2. if normalize else cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black",
                     fontsize=7) 
        else:
            plt.text(j, i, f"{cm[i, j]:,}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=7)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel(f'Predicted Label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}', fontsize=12)
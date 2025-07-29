# src/utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

def _estimate_hop_length(times):
    """Estimate the hop length in seconds from a series of time points."""
    return np.mean(np.diff(times)) if len(times) > 1 else 0.02 

def _unroll_pitch_list(times, pitch_list):
    """Converts a list of pitch arrays into plottable coordinates."""
    unrolled_times, unrolled_pitches = [], []
    for t, p in zip(times, pitch_list):
        if p.size > 0:
            unrolled_times.extend([t] * len(p))
            unrolled_pitches.extend(p)
    return np.array(unrolled_times), np.array(unrolled_pitches)

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

def plot_waveform(samples, sample_rate, save_path, title='Audio Waveform'):
    """Plots and saves a 1D audio waveform."""
    plt.figure(figsize=(15, 5))
    times = np.arange(len(samples)) / sample_rate
    plt.plot(times, samples, color='black', linewidth=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(0, times[-1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Waveform plot saved: {save_path}")

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
    print(f"✅ Spectrogram plot saved: {save_path}")

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
    print(f"✅ Piano roll plot saved: {save_path}")

def plot_notes(note_grid, hop_seconds, save_path, title='Notes Visualization', y_axis='MIDI Pitch', low_val=21):
    """
    Plots notes from a grid as discrete rectangles.
    `note_grid` is a 2D binary numpy array (num_pitches/strings x num_frames).
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    notes = _find_notes(note_grid, hop_seconds)
    
    for note in notes:
        pitch = note['pitch'] + low_val
        onset = note['onset']
        duration = note['offset'] - note['onset']
        rect = Rectangle((onset, pitch - 0.5), duration, 1, 
                         linewidth=1, edgecolor='darkblue', facecolor='skyblue')
        ax.add_patch(rect)
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(y_axis, fontsize=12)
    if notes:
        pitches = [n['pitch'] for n in notes]
        ax.set_ylim(min(pitches) + low_val - 2, max(pitches) + low_val + 2)
        ax.set_xlim(0, max(n['offset'] for n in notes))
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Notes plot saved: {save_path}")

def plot_pitch_list(times, pitch_data, save_path, title='Pitch Contour'):
    """
    Plots a list of active pitches over time as a scatter plot.
    `pitch_data` is a list of numpy arrays, where each array holds pitches for a time frame.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    unrolled_times, unrolled_pitches = _unroll_pitch_list(times, pitch_data)
    
    ax.scatter(unrolled_times, unrolled_pitches, s=5, color='red')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('MIDI Pitch', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Pitch list plot saved: {save_path}")

def plot_guitar_tablature(tab_data, hop_seconds, save_path, title='Guitar Tablature', string_tuning=(40, 45, 50, 55, 59, 64)):
    """
    Plots guitar tablature, showing fret numbers on string lines.
    `tab_data` is a 2D numpy array (6 x num_frames) where values are fret numbers (-1 for silence).
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    num_strings, num_frames = tab_data.shape
    duration_seconds = num_frames * hop_seconds
    
    for i in range(num_strings):
        ax.plot([0, duration_seconds], [i, i], color='grey', linewidth=0.75)
        
    for string_idx in range(num_strings):
        is_note_active = False
        onset_frame = 0
        active_fret = -1
        for frame_idx, fret in enumerate(tab_data[string_idx]):
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
    print(f"Guitar tablature plot saved: {save_path}")

def plot_loss_curves(history: dict, save_path: str or Path):
    """Plots and saves the training and validation loss curves."""
    plt.figure(figsize=(12, 7))
    plt.plot(history['train_loss'], label='Training Loss', color='royalblue', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', color='orangered', marker='x')
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Loss curve saved: {save_path}")

def plot_metrics(history: dict, metric_name: str, save_path: str or Path):
    """Plots and saves training and validation metrics (e.g., F1, accuracy)."""
    train_key = f'train_{metric_name}'
    val_key = f'val_{metric_name}'
    if val_key not in history: return
    plt.figure(figsize=(12, 7))
    if train_key in history:
        plt.plot(history[train_key], label=f'Training {metric_name.upper()}', color='royalblue', marker='o')
    plt.plot(history[val_key], label=f'Validation {metric_name.upper()}', color='orangered', marker='x')
    plt.title(f'Training and Validation {metric_name.upper()} Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ {metric_name.upper()} curve saved: {save_path}")

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
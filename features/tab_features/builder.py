# builder.py
import numpy as np
import soundfile as sf
import librosa
import jams
import os

from .notes import create_all_note_data_from_jams
from .utils import extract_tab_features
from .transforms import compute_tablature_adj, compute_tablature_rel
from librosa.core import cqt_frequencies

def build_tab_npz(
    jams_path: str,
    audio_path: str,
    extracted_features=None,
    sr: int = 22050,
    hop_length: int = 512,
    trim_mode: str = 'first', 
    max_frames: int = 500,
    n_bins: int = 192,
    bins_per_octave: int = 24,
    fmin: float = 82.41,
    num_harmonics: int = 44,
    max_fret: int = 20,
    silence_val: int = -1,
    save_path: str | None = None,
    verbose: bool = False
) -> dict:
    
    jam = jams.load(jams_path, validate=False)
    audio, sr_orig = sf.read(audio_path, dtype='float32')
    if sr_orig != sr:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr, dtype=np.float32)

    tempo_annotations = [ann for ann in jam.annotations if ann.namespace == 'tempo']
    bpm = 120.0
    if tempo_annotations and tempo_annotations[0].data:
        bpm = tempo_annotations[0].data[0].value
    ticks_per_quarter = 960
    seconds_per_tick = 60.0 / (bpm * ticks_per_quarter)
    
    last_event_time_sec = 0
    tab_annotations = [ann for ann in jam.annotations if ann.namespace == 'note_tab']
    for annotation in tab_annotations:
        if annotation.data:
            current_last_tick = max([note.time + note.duration for note in annotation.data], default=0)
            current_last_sec = current_last_tick * seconds_per_tick
            if current_last_sec > last_event_time_sec:
                last_event_time_sec = current_last_sec
    
    total_frames = int(np.ceil(last_event_time_sec * sr / hop_length))

    start_frame = 0
    if trim_mode == 'active' and total_frames > max_frames:
        if verbose: print("En aktif aralık aranıyor...")
        activity_matrix = np.zeros((6, total_frames))
        for string_index, annotation in enumerate(tab_annotations):
            for note in annotation.data:
                start_f = int(round((note.time * seconds_per_tick) * sr / hop_length))
                end_f = start_f + int(round((note.duration * seconds_per_tick) * sr / hop_length))
                activity_matrix[string_index, start_f:min(end_f, total_frames)] = 1
        
        total_activity_per_frame = np.sum(activity_matrix, axis=0)
        max_score = -1
        best_start = 0
        for start_f_win in range(total_frames - max_frames):
            current_score = np.sum(total_activity_per_frame[start_f_win : start_f_win + max_frames])
            if current_score > max_score:
                max_score = current_score
                best_start = start_f_win
        
        start_frame = best_start
        if verbose: print(f"En aktif aralık bulundu: {start_frame}-{start_frame + max_frames}")

    T_final = min(max_frames, total_frames - start_frame)
    end_frame = start_frame + T_final
    audio = audio[start_frame * hop_length : end_frame * hop_length]

    if verbose: print(f"Nihai uzunluk {T_final} frame olarak belirlendi. Veriler {start_frame}-{end_frame} aralığından alındı.")
    
    effective_sr = sr
    track = os.path.basename(jams_path).replace(".jams", "")
    
    tablature, notes, pitch_list = create_all_note_data_from_jams(
        jam=jam,
        jams_path=jams_path,
        sr=effective_sr,
        hop_length=hop_length,
        time_offset_frames=start_frame, 
        total_frames=T_final,
        max_fret=max_fret
    )

    pitch_bins = cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    features = extract_tab_features(
        track_name=track,
        fs=effective_sr,
        audio=audio,
        tablature=tablature,
        string_notes=notes, 
        num_frames=T_final,
        hop_length=hop_length,
        pitch_bins=pitch_bins,
        num_harmonics=num_harmonics,
        max_fret=max_fret,
        silence_val=silence_val
    )

    features["notes"] = notes
    features["pitch_list"] = pitch_list
    features["pitch_bins"] = pitch_bins

    if extracted_features:
        try:
            spec_feats = extracted_features.process_audio(audio, target_frames=T_final)
            
            if isinstance(spec_feats, dict):
                features.update(spec_feats)
            elif isinstance(spec_feats, np.ndarray):
                features[extracted_features.features_name()] = spec_feats
        except Exception as e:
            print(f"Feature extraction failed: {e}")

    if save_path:
        np.savez(save_path, **features, allow_pickle=True)

    return features
# notes.py
import numpy as np
import jams
import math
from .pitch import compute_tablature
from .utils import extract_open_string_midi_from_jams

def midi_to_freq(midi: int) -> float:
    """MIDI notasını frekansa çevirir."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def create_all_note_data_from_jams(
    jam: jams.JAMS,
    jams_path: str,
    sr: int,
    hop_length: int,
    time_offset_frames: int, # Başlangıç frame'i ofseti
    total_frames: int, # Bu artık pencere boyutu (örn: 500)
    max_fret: int
) -> tuple[np.ndarray, dict, dict]:
    """
    Verilen JAMS objesindeki zamanlama ve perde bilgisini kullanarak nihai
    tablatür matrisini ve diğer nota formatlarını oluşturan ana fonksiyondur.
    """
    open_string_midi = extract_open_string_midi_from_jams(jams_path)
    
    tempo_annotations = [ann for ann in jam.annotations if ann.namespace == 'tempo']
    bpm = 120.0
    if tempo_annotations and tempo_annotations[0].data:
        bpm = tempo_annotations[0].data[0].value
    
    ticks_per_quarter = 960
    seconds_per_tick = 60.0 / (bpm * ticks_per_quarter)

    tab_annotations = [ann for ann in jam.annotations if ann.namespace == 'note_tab']
    if len(tab_annotations) != 6:
        raise ValueError(f"JAMS'te 6 'note_tab' katmanı bekleniyordu, {len(tab_annotations)} bulundu.")

    pitch_dict = {s: [] for s in range(6)}
    
    for s, annotation in enumerate(tab_annotations):
        for note in annotation.data:
            start_time_sec = note.time * seconds_per_tick
            duration_sec = note.duration * seconds_per_tick
            
            # Notanın orijinal zaman çizelgesindeki mutlak konumunu hesapla
            original_start_frame = int(round(start_time_sec * sr / hop_length))
            num_frames = int(round(duration_sec * sr / hop_length))
            original_end_frame = original_start_frame + num_frames

            # Notanın bizim yeni penceremize göre göreceli konumunu hesapla
            relative_start_frame = original_start_frame - time_offset_frames
            relative_end_frame = original_end_frame - time_offset_frames
            
            # Sadece pencerenin içine düşen notaları işle
            if relative_end_frame > 0 and relative_start_frame < total_frames:
                # Pencere sınırları dışına taşan kısımları kırp
                final_start_frame = max(0, relative_start_frame)
                final_end_frame = min(total_frames, relative_end_frame)

                fret = note.value.get('fret')
                if fret is not None and open_string_midi[s] != 0:
                    midi_pitch = open_string_midi[s] + fret
                    pitch_dict[s].append((final_start_frame, final_end_frame, midi_pitch))

    # Tablatürü hesapla
    tablature = compute_tablature(pitch_dict, open_string_midi, total_frames, max_fret)

    # Diğer formatları üret
    notes = {s: [] for s in range(6)}
    pitch_list = {s: [] for s in range(6)}
    for s, note_events in pitch_dict.items():
        for start_frame, end_frame, midi_pitch in note_events:
            freq = midi_to_freq(midi_pitch)
            notes[s].append((freq, (start_frame, end_frame)))
            pitch_list[s].extend([freq] * (end_frame - start_frame))

    return tablature, notes, pitch_list

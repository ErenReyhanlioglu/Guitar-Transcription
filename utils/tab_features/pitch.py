#pitch.py
import numpy as np

def compute_tablature(pitch_dict, open_string_midi, T, max_fret=19, silence_val=-1, verbose=False):

    num_strings = 6
    tablature = np.full((num_strings, T), fill_value=silence_val, dtype=np.int32)

    for s in pitch_dict:
        for t_start, t_end, pitch in pitch_dict[s]:
            fret = int(round(pitch - open_string_midi[s]))
            if fret < 0 or fret > max_fret:
                if verbose:
                    print(f"⚠️ Ignored fret {fret} (pitch={pitch}) on string {s} — outside 0–{max_fret}")
                continue
            tablature[s, t_start:t_end] = fret

    return tablature

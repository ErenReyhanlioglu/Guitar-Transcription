import numpy as np

def compute_tablature(pitch_dict, open_string_midi, T):
    """
    Construct a frame-level tablature matrix (fret numbers) from pitch annotations.

    This function processes a dictionary of pitch segments per string and converts 
    MIDI pitch values into fret numbers, taking into account the tuning of each string.

    Parameters:
        pitch_dict (dict): 
            Dictionary mapping each string index (0–5) to a list of tuples:
            (start_frame, end_frame, midi_pitch).

        open_string_midi (list of int): 
            MIDI pitch values for each open string (e.g., [40, 45, 50, 55, 59, 64] for EADGBE tuning).

        T (int): 
            Total number of frames in the output tablature matrix.

    Returns:
        np.ndarray: 
            Tablature matrix of shape (6, T) where each value is the fret number
            or -1 if silent.
    """
    num_strings = 6
    tablature = np.full((num_strings, T), fill_value=-1, dtype=np.int32)

    for s in pitch_dict:
        for t_start, t_end, pitch in pitch_dict[s]:
            fret = int(round(pitch - open_string_midi[s]))
            if fret < 0 or fret > 24:  # ignore invalid fret numbers
                continue
            tablature[s, t_start:t_end] = fret

    return tablature

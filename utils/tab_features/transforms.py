import numpy as np

# MIDI pitch values of open strings in standard tuning (E A D G B E)
MIDI_OPEN_STRINGS = [40, 45, 50, 55, 59, 64]

def compute_tablature_adj(
    tablature: np.ndarray,
    silence_val: int = -1,
    max_midi: int = 119
) -> np.ndarray:
    """
    Convert tablature matrix to a one-hot encoded MIDI pitch matrix.

    For each string and time frame, the played fret is mapped to a MIDI pitch
    using that string's open MIDI value. Silent frames are skipped. The output
    is a binary activation matrix over pitch and time.

    Parameters:
        tablature (np.ndarray): Shape (6, T), integer fret numbers.
        silence_val (int): Value used to indicate silence (default = -1).
        max_midi (int): Maximum allowed MIDI pitch index (default = 119).

    Returns:
        np.ndarray: Shape (max_midi+1, T), one-hot MIDI pitch activations.
    """
    num_strings, T = tablature.shape
    output = np.zeros((max_midi + 1, T), dtype=np.float32)

    for s in range(num_strings):
        open_midi = MIDI_OPEN_STRINGS[s]
        for t in range(T):
            fret = tablature[s, t]
            if fret == silence_val:
                continue
            midi = open_midi + fret
            if 0 <= midi <= max_midi:
                output[midi, t] = 1.0

    return output


def compute_tablature_rel(
    tablature: np.ndarray,
    silence_val: int = -1,
    max_midi: int = 119
) -> np.ndarray:
    """
    Generate a log-scale pitch representation relative to each string's open frequency.

    For each frame and string, computes the log2 ratio between played note frequency
    and the open-string frequency. Outputs a sparse matrix indexed by absolute MIDI pitch.

    Parameters:
        tablature (np.ndarray): Shape (6, T), integer fret values.
        silence_val (int): Value used to indicate silence (default = -1).
        max_midi (int): Maximum MIDI pitch to include in the output (default = 119).

    Returns:
        np.ndarray: Shape (max_midi+1, T), continuous log2(f/f_open) values.
    """
    num_strings, T = tablature.shape
    output = np.zeros((max_midi + 1, T), dtype=np.float32)

    for s in range(num_strings):
        open_midi = MIDI_OPEN_STRINGS[s]
        open_freq = 440.0 * 2 ** ((open_midi - 69) / 12.0)
        for t in range(T):
            fret = tablature[s, t]
            if fret == silence_val:
                continue
            midi = open_midi + fret
            if 0 <= midi <= max_midi:
                target_freq = 440.0 * 2 ** ((midi - 69) / 12.0)
                log_ratio = np.log2(target_freq / open_freq)
                output[midi, t] = log_ratio

    return output


def compute_tablature_adj_spatial(
    tablature: np.ndarray,
    max_fret: int = 20,
    silence_val: int = -1
) -> np.ndarray:
    """
    Convert tablature matrix to one-hot encoded spatial format (string x fret x time).

    Parameters:
        tablature (np.ndarray): Shape (6, T), integer fret numbers.
        max_fret (int): Maximum fret number to represent (default = 20).
        silence_val (int): Value indicating silence (default = -1).

    Returns:
        np.ndarray: Shape (6, max_fret+1, T), one-hot spatial fret activations.
                    Channel 0 corresponds to silence.
    """
    num_strings, T = tablature.shape
    output = np.zeros((num_strings, max_fret + 1, T), dtype=np.float32)

    for s in range(num_strings):
        for t in range(T):
            fret = tablature[s, t]
            if fret == silence_val or fret < 0 or fret > max_fret:
                output[s, 0, t] = 1.0  # silence channel
            else:
                output[s, fret, t] = 1.0

    return output


def compute_tablature_rel_spatial(
    tablature: np.ndarray,
    max_fret: int = 20,
    silence_val: int = -1
) -> np.ndarray:
    """
    Generate a continuous log-scaled pitch representation in spatial format (string x fret x time).

    Parameters:
        tablature (np.ndarray): Shape (6, T), integer fret values.
        max_fret (int): Maximum fret number to represent (default = 20).
        silence_val (int): Value indicating silence (default = -1).

    Returns:
        np.ndarray: Shape (6, max_fret+1, T), continuous log2(f/f_open) values.
                    Silence frames are zero.
    """
    num_strings, T = tablature.shape
    output = np.zeros((num_strings, max_fret + 1, T), dtype=np.float32)

    for s in range(num_strings):
        open_midi = MIDI_OPEN_STRINGS[s]
        open_freq = 440.0 * 2 ** ((open_midi - 69) / 12.0)
        for t in range(T):
            fret = tablature[s, t]
            if fret == silence_val or fret < 0 or fret > max_fret:
                continue  # silence, leave zero
            target_midi = open_midi + fret
            target_freq = 440.0 * 2 ** ((target_midi - 69) / 12.0)
            log_ratio = np.log2(target_freq / open_freq)
            output[s, fret, t] = log_ratio

    return output

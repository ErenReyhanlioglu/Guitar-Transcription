# src/utils/config_helpers.py
def process_config(config):
    """
    Loads the raw config dictionary, derives necessary parameters,
    and returns the fully processed config.
    """
    max_fret = config['data']['max_fret']
    include_silence = config['data']['include_silence']

    num_fret_classes = max_fret + 1
    silence_class_idx = num_fret_classes
    total_model_classes = num_fret_classes + 1 if include_silence else num_fret_classes

    config['data']['silence_class'] = silence_class_idx
    config['model']['params']['num_classes'] = total_model_classes

    return config
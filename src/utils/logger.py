import logging
import sys
import torch
import numpy as np

class StatefulLogFilter(logging.Filter):
    """
    Only allows a log record if its message is different from the last message
    logged from the same source (module and line number).
    This is used to suppress repetitive DEBUG logs within a loop.
    """
    def __init__(self):
        super().__init__()
        self.last_logs = {}

    def filter(self, record):
        # Apply this filter only to DEBUG messages
        if record.levelno != logging.DEBUG:
            return True

        log_key = f"{record.name}:{record.lineno}"
        current_message = record.getMessage()
        last_message = self.last_logs.get(log_key)

        if current_message == last_message:
            return False  # Suppress the log
        else:
            self.last_logs[log_key] = current_message
            return True  # Allow the log

    def reset(self):
        """Clears the stored log history."""
        self.last_logs.clear()

def setup_logger(config):
    log_config = config.get('logging_and_checkpointing', {})
    is_verbose = log_config.get('verbose_shape_logging', False)
    root_log_level = logging.DEBUG if is_verbose else logging.INFO
    
    root_logger = logging.getLogger()
    root_logger.setLevel(root_log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    stateful_filter = StatefulLogFilter()
    handler.addFilter(stateful_filter)
    
    root_logger.addHandler(handler)
    
    specific_levels = log_config.get('log_levels', {})
    for name, level_str in specific_levels.items():
        numeric_level = logging.getLevelName(level_str.upper())
        if isinstance(numeric_level, int):
            logging.getLogger(name).setLevel(numeric_level)
        else:
            logging.warning(f"Invalid log level '{level_str}' for logger '{name}'. Skipping.")

    logging.info(f"Logger initialized. Root level: {logging.getLevelName(root_log_level)}. Specific overrides: {specific_levels or 'None'}")
    
    return stateful_filter

def describe(data):
    if data is None: return "<None>"
    if isinstance(data, (torch.Tensor, np.ndarray)):
        return f"<{type(data).__name__}, shape={data.shape}, dtype={data.dtype}>"
    if isinstance(data, (list, tuple)):
        return f"<{type(data).__name__}, len={len(data)}>"
    if isinstance(data, dict):
        return f"<{type(data).__name__}, keys={list(data.keys())}>"
    return f"<{type(data).__name__}>"
import numpy as np

class EarlyStopping:
    """
    Stops training when a monitored metric has stopped improving.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            monitor (str): Quantity to be monitored (e.g., 'val_loss', 'val_tab_f1').
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will stop
                        when it has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.wait_count = 0
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.early_stop = False

        if self.mode not in {'min', 'max'}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")

    def __call__(self, current_score: float):
        """
        Checks if training should be stopped based on the current score.
        """
        score_improved = False
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                score_improved = True
        else: # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                score_improved = True

        if score_improved:
            self.best_score = current_score
            self.wait_count = 0
            print(f"EarlyStopping: {self.monitor} improved to {self.best_score:.6f}. Resetting counter.")
        else:
            self.wait_count += 1
            print(f"EarlyStopping: No improvement in {self.monitor} for {self.wait_count}/{self.patience} epochs.")
            if self.wait_count >= self.patience:
                self.early_stop = True
                print(f"--- Early stopping triggered after {self.patience} epochs of no improvement. ---")
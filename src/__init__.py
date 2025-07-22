"""
Purpose:
    This __init__.py file serves two primary purposes:
    1.  It officially marks the 'src' directory as a Python package, which is
        essential for enabling relative imports between its submodules (e.g.,
        `from .models import ...` within trainer.py).
    2.  (Optional) It can expose key classes and functions from submodules at the
        top level of the 'src' package. This allows for cleaner and shorter import
        statements in the experiment notebooks.

Dependencies:
    - This file imports from the submodules within the 'src' package itself,
      such as `data_loader`, `trainer`, `models`, and `utils`.

Current Status:
    - Its main role is to enable the package structure. Key components are
      currently imported directly from their respective submodules in the notebooks.

Future Plans:
    - [ ] Consider adding top-level imports for the most frequently used classes
          to simplify the workflow. For example:
          `from .trainer import Trainer`
          This would allow `from src import Trainer` instead of
          `from src.trainer import Trainer` in the notebook.
"""

# Example of how you might expose key components (currently commented out):
# from .data_loader import TabCNNDataset
# from .trainer import Trainer
# from .models import get_model

from .trainer import Trainer
from .data_loader import TablatureDataset
from .models import get_model
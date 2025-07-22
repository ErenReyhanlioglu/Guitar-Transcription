"""
Purpose:
    This module provides utility functions for managing the experiment lifecycle.
    Its primary role is to create a unique, versioned, and timestamped directory
    for each training run, ensuring that results are never overwritten and are
    always reproducible.

Dependencies:
    - os, re, datetime, shutil

Current Status:
    - Contains the `create_experiment_directory` function.
    - This function automatically determines the next version number (V0, V1, etc.)
      for a given model, creates a new directory with a timestamp, and copies the
      run's config file into it for perfect reproducibility.

Future Plans:
    - [ ] Integrate with Git to automatically save the current commit hash into the
          experiment directory.
    - [ ] Add a function to load results/models from a specified experiment path.
"""
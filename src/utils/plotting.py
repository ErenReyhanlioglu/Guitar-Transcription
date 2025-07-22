"""
Purpose:
    This module contains utility functions for creating visualizations of
    experiment results, such as training curves and confusion matrices.

Dependencies:
    - matplotlib
    - seaborn
    - numpy

Current Status:
    - A function to plot training and validation loss/accuracy curves from a
      `history.json` file.
    - A function to generate and save per-string confusion matrices to visualize
      which frets are being confused.

Future Plans:
    - [ ] Add a function to plot an input spectrogram alongside the ground truth
          and predicted tablature for qualitative analysis.
    - [ ] For Transformer models, create visualizations for attention maps to
          understand what the model is focusing on.
"""
#src/utils/experiment.py
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from torchsummary import summary
import io
from contextlib import redirect_stdout
import torch  

def create_experiment_directory(base_output_path: str, model_name: str, config_path: str):
    model_path = os.path.join(base_output_path, model_name)
    os.makedirs(model_path, exist_ok=True)

    existing_versions = []
    for d in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, d)) and d.startswith('V'):
            version_num = re.search(r'V(\d+)', d)
            if version_num:
                existing_versions.append(int(version_num.group(1)))

    next_version = max(existing_versions) + 1 if existing_versions else 0
    
    exp_dir_name = f"V{next_version}"
    experiment_path = os.path.join(model_path, exp_dir_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    checkpoints_path = os.path.join(experiment_path, "model_checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    print(f"Experiment directory created: {experiment_path}")

    shutil.copy(config_path, os.path.join(experiment_path, 'config.yaml'))
    
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).strip().decode('utf-8')
        with open(os.path.join(experiment_path, 'environment.txt'), 'w') as f:
            f.write(pip_freeze)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not run 'pip freeze'. Library versions were not saved.")

    return experiment_path

def save_model_summary(model, config, experiment_path):
    """
    Generates a model summary using torch-summary and saves it to a text file.

    Args:
        model (torch.nn.Module): The model to be summarized.
        config (dict): The experiment config dictionary.
        experiment_path (str): The path to the experiment output directory.
    """
    input_size = (
        config['model']['params']['in_channels'],
        config['model']['params']['num_freq'],
        config['data']['window_size']
    )
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        summary(model.to(device), input_size=input_size, device=device.type)
    model_summary_str = summary_buffer.getvalue()

    summary_file_path = os.path.join(experiment_path, "model_summary.txt")
    with open(summary_file_path, 'w') as f:
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Input Size: {input_size}\n\n")
        f.write(model_summary_str)

    print(f"📄 Model summary saved: {summary_file_path}")

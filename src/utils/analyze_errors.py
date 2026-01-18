import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import logging 
from pathlib import Path
import sys
import yaml

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_PATH))

from src.utils.plotting import plot_confusion_matrix
from src.utils.logger import setup_logger
from src.utils.config_helpers import process_config 
from src.models import get_model

logger = logging.getLogger(__name__)

def analyze(experiment_path: str, main_exp_path: str = None, val_loader: DataLoader = None):
    """
    Loads a trained model, runs validation, and generates error analysis plots.
    Smartly looks for config.yaml in the experiment_path OR main_exp_path.
    """
    # 1. Config Loading Logic (DÜZELTİLDİ)
    if main_exp_path is None:
        # Eğer main path verilmemişse, bir üst klasör varsayılır
        main_exp_path = str(Path(experiment_path).parent)

    # Önce Fold içine bak
    config_path = os.path.join(experiment_path, "config.yaml")
    
    # Bulamazsa Ana Klasöre bak
    if not os.path.exists(config_path):
        logger.info(f"Config not found in {experiment_path}, checking {main_exp_path}...")
        config_path = os.path.join(main_exp_path, "config.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path} or within experiment dir.")
        return

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    config = raw_config 
    
    logger.info("--- Starting Error Analysis ---")
    logger.info(f"Analyzing experiment: {experiment_path}")
    logger.info(f"Using config from: {config_path}")
    
    MODEL_PATH = os.path.join(experiment_path, "model_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Model Loading
    # Modeli config ile başlat
    try:
        model = get_model(config).to(device)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model state from: {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=device)
        # Checkpoint bazen 'model_state_dict' anahtarı içinde olabilir
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
    else:
        logger.error(f"Model file not found: {MODEL_PATH}")
        return

    model.eval()
    logger.info("Model loaded successfully.")
    
    # 3. Data Loader
    if val_loader is None:
        logger.info("Validation dataloader not provided. Creating from config...")
        from src.data_loader import prepare_dataset_files, get_dataloaders
        
        # Bu kısım biraz maliyetli olabilir, eğer val_loader dışarıdan geliyorsa burası çalışmaz.
        all_paths, groups = prepare_dataset_files(config)
        
        # Hızlı bir validation seti (Eğer dışarıdan verilmediyse mecburen rastgele)
        np.random.seed(42)
        indices = np.random.permutation(len(all_paths))
        val_size = int(len(all_paths) * 0.2)
        val_paths = all_paths[indices[:val_size]].tolist()
        
        # Sadece val_loader lazım
        _, val_loader, _ = get_dataloaders(config, [], val_paths, [])
    
    # 4. Prediction Loop
    all_preds = { 'tablature': [], 'hand_pos': [], 'string_act': [], 'pitch_class': [] }
    all_targets = { 'tablature': [], 'hand_pos': [], 'string_act': [], 'pitch_class': [] }
    
    logger.info("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: tensor.to(device) for key, tensor in batch['features'].items()}
            outputs = model(inputs) 
            
            # --- Tablature ---
            if 'tab_logits' in outputs and 'tablature' in batch:
                tab_logits = outputs['tab_logits']
                targets = batch['tablature']
                
                S = config['instrument']['num_strings']
                C = config['model']['params']['num_classes']
                
                if tab_logits.dim() == 2: tab_logits = tab_logits.view(-1, S, C)
                preds = torch.argmax(tab_logits, dim=-1)
                if targets.dim() == 3: targets = targets.reshape(-1, S)
                if targets.dim() == 1: targets = targets.reshape(-1, S)
                
                all_preds['tablature'].append(preds.cpu())
                all_targets['tablature'].append(targets.cpu())

            # --- Hand Position ---
            if 'hand_pos_logits' in outputs and 'hand_pos_target' in batch:
                preds = torch.argmax(outputs['hand_pos_logits'], dim=-1)
                all_preds['hand_pos'].append(preds.cpu())
                all_targets['hand_pos'].append(batch['hand_pos_target'].cpu())

            # --- String Activity ---
            if 'activity_logits' in outputs and 'activity_target' in batch:
                preds = (torch.sigmoid(outputs['activity_logits']) > 0.5).int()
                all_preds['string_act'].append(preds.cpu())
                all_targets['string_act'].append(batch['activity_target'].cpu())

            # --- Pitch Class ---
            if 'pitch_class_logits' in outputs and 'pitch_class_target' in batch:
                preds = (torch.sigmoid(outputs['pitch_class_logits']) > 0.5).int()
                all_preds['pitch_class'].append(preds.cpu())
                all_targets['pitch_class'].append(batch['pitch_class_target'].cpu())

    # 5. Plotting
    analysis_dir = os.path.join(experiment_path, 'analysis_plots')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # --- A. Tablature CM ---
    if all_preds['tablature']:
        logger.info("Generating Tablature Confusion Matrices...")
        preds_flat = torch.cat(all_preds['tablature'], dim=0) 
        targets_flat = torch.cat(all_targets['tablature'], dim=0)
        
        silence_class = config['instrument']['silence_class']
        num_classes = config['model']['params']['num_classes']
        labels = [str(i) for i in range(silence_class)] + ['Sil']
        labels = labels[:num_classes]
        
        tab_dir = os.path.join(analysis_dir, 'tablature')
        os.makedirs(tab_dir, exist_ok=True)

        for s in range(config['instrument']['num_strings']):
            p_s = preds_flat[:, s].numpy()
            t_s = targets_flat[:, s].numpy()
            cm = confusion_matrix(t_s, p_s, labels=np.arange(num_classes))
            save_path = os.path.join(tab_dir, f'cm_string_{s+1}.png')
            plot_confusion_matrix(cm, target_names=labels, title=f'String {s+1} Confusion Matrix', save_path=save_path)

    # --- B. Hand Position CM ---
    if all_preds['hand_pos']:
        logger.info("Generating Hand Position Confusion Matrix...")
        p_hp = torch.cat(all_preds['hand_pos'], dim=0).numpy()
        t_hp = torch.cat(all_targets['hand_pos'], dim=0).numpy()
        
        hp_labels = ['Open', 'Low', 'Mid', 'High', 'V.High'] 
        cm = confusion_matrix(t_hp, p_hp, labels=np.arange(len(hp_labels)))
        save_path = os.path.join(analysis_dir, 'cm_hand_position.png')
        plot_confusion_matrix(cm, target_names=hp_labels, title='Hand Position Confusion Matrix', save_path=save_path)

    # --- C. String Activity CM ---
    if all_preds['string_act']:
        logger.info("Generating String Activity Confusion Matrices...")
        p_act = torch.cat(all_preds['string_act'], dim=0).numpy()
        t_act = torch.cat(all_targets['string_act'], dim=0).numpy()
        
        act_dir = os.path.join(analysis_dir, 'string_activity')
        os.makedirs(act_dir, exist_ok=True)
        
        for s in range(6):
            cm = confusion_matrix(t_act[:, s], p_act[:, s], labels=[0, 1])
            save_path = os.path.join(act_dir, f'cm_activity_string_{s+1}.png')
            plot_confusion_matrix(cm, target_names=['Silent', 'Active'], title=f'Activity String {s+1}', save_path=save_path)

    # --- D. Pitch Class (PERFORMANCE SUMMARY CHART) ---
    if all_preds['pitch_class']:
        logger.info("Generating Pitch Class Performance Summary...")
        p_pc = torch.cat(all_preds['pitch_class'], dim=0).numpy() # (N, 12)
        t_pc = torch.cat(all_targets['pitch_class'], dim=0).numpy()
        
        pc_dir = os.path.join(analysis_dir, 'pitch_class')
        os.makedirs(pc_dir, exist_ok=True)
        
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        accuracies = []
        f1_scores = []
        
        for i in range(12):
            acc = accuracy_score(t_pc[:, i], p_pc[:, i])
            f1 = f1_score(t_pc[:, i], p_pc[:, i], zero_division=0)
            accuracies.append(acc)
            f1_scores.append(f1)
            
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(chroma_names))
        width = 0.35 
        
        rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue', edgecolor='black', alpha=0.8)
        rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='salmon', edgecolor='black', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Pitch Class Performance by Note (Accuracy vs F1)')
        ax.set_xticks(x)
        ax.set_xticklabels(chroma_names)
        ax.set_ylim(0, 1.1) 
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

        autolabel(rects1)
        autolabel(rects2)
        
        save_path = os.path.join(pc_dir, 'pitch_class_performance_summary.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        logger.info(f"Pitch class summary saved to {save_path}")

    logger.info("--- Error Analysis Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Error Analysis Script for Multi-Task Guitar Model")
    parser.add_argument('experiment_path', type=str, help="Full path to the experiment directory")
    parser.add_argument('--main_exp_path', type=str, default=None, help="Root path of experiments")
    
    args = parser.parse_args()
    analyze(args.experiment_path, args.main_exp_path)
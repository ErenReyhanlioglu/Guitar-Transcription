import os
import time
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from src.utils.logger import describe
from functools import partial
import logging

logger = logging.getLogger(__name__)

class TablatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-computed features and tablature targets.
    This version is updated to handle multiple input features dynamically and
    generate the auxiliary multipitch target on-the-fly.
    """
    def __init__(self, npz_paths: list[str], config: dict):
        self.npz_paths = npz_paths
        self.config = config
        self.data_config = config['data']
        self.instrument_config = config['instrument']
        self.loss_config = config['loss']
        
        self.active_features = self.data_config['active_features']
        self.target_keys = self.data_config.get('target_keys', ['tablature'])
        
        self.aux_loss_enabled = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        if self.aux_loss_enabled:
            logger.info("Auxiliary loss is enabled. Multipitch target will be generated.")
            self.tuning = torch.tensor(self.instrument_config['tuning'], dtype=torch.long)
            self.min_midi = self.instrument_config['min_midi']
            self.max_midi = self.instrument_config['max_midi']
            self.num_pitches = self.max_midi - self.min_midi + 1
        
        self.samples_metadata = [{'file_path': path} for path in self.npz_paths]
        # YENİ LOG EKLENDİ - Sadece başlangıçta bilgi vermek için
        if len(self.samples_metadata) > 0:
             logger.debug(f"Dataset initialized for {len(self.samples_metadata)} files. Features: {self.active_features}, Targets: {self.target_keys}.")

    def __len__(self) -> int:
        return len(self.samples_metadata)

    def __getitem__(self, idx: int) -> dict:
        path = self.samples_metadata[idx]['file_path']
        sample = {}
        features = {}

        with np.load(path, allow_pickle=True) as data:
            for feature_key in self.active_features:
                if feature_key not in data:
                    logger.warning(f"Feature key '{feature_key}' not found in {path}. Skipping.")
                    continue

                feature_data_full = data[feature_key].astype(np.float32)
                
                if feature_data_full.ndim == 4 and feature_data_full.shape[0] == 1:
                    feature_data_full = feature_data_full.squeeze(0)
                
                # NPZ'den gelen boyut: (C, H, T)
                feature_tensor = torch.from_numpy(feature_data_full.copy()).permute(0, 2, 1)
                # Permute sonrası boyut: (C, T, H)
                
                # YENİ LOG EKLENDİ - __getitem__ içindeki tensör boyutunu görmek için
                logger.debug(f"Loaded feature '{feature_key}' from {os.path.basename(path)} with shape: {feature_tensor.shape} (C, T, H)")
                
                features[feature_key] = feature_tensor

            sample['features'] = features

            tablature_tensor = None
            if 'tablature' in self.target_keys and 'tablature' in data:
                tablature_data = data['tablature'].astype(np.int64)
                tablature_tensor = torch.from_numpy(tablature_data.copy())
                
                silence_class = self.instrument_config.get('num_frets', 19) + 1
                tablature_tensor[tablature_tensor == -1] = silence_class
                sample['tablature'] = tablature_tensor

            for key in self.target_keys:
                if key == 'tablature' or key not in data:
                    continue
                target_data = data[key]
                target_tensor = torch.from_numpy(target_data.copy().astype(np.float32))
                sample[key] = target_tensor

            if self.aux_loss_enabled and tablature_tensor is not None:
                num_strings, num_steps = tablature_tensor.shape
                y_multipitch = torch.zeros((self.num_pitches, num_steps), dtype=torch.float32)
                
                silence_class = self.instrument_config.get('num_frets', 19) + 1
                played_frets_mask = (tablature_tensor < silence_class) & (tablature_tensor >= 0)

                for s in range(num_strings):
                    for t in range(num_steps):
                        if played_frets_mask[s, t]:
                            fret = tablature_tensor[s, t]
                            midi_note = self.tuning[s] + fret
                            if self.min_midi <= midi_note <= self.max_midi:
                                pitch_index = midi_note - self.min_midi
                                y_multipitch[pitch_index, t] = 1.0
                
                sample['multipitch_target'] = y_multipitch
            
            # YENİ LOG EKLENDİ - Dönen sample'daki tüm anahtarları ve boyutlarını loglayalım
            log_str = f"Sample {idx} final shapes: "
            for k, v in sample.items():
                if k == 'features':
                    for fk, fv in v.items():
                        log_str += f"features['{fk}']: {fv.shape}, "
                else:
                    log_str += f"'{k}': {v.shape}, "
            logger.debug(log_str)

            return sample

def collate_fn(batch: list[dict], config: dict) -> dict:
    preparation_mode = config['data']['active_preparation_mode']
    logger.debug(f"\n--- Collate Function Start (Mode: {preparation_mode}, Batch Size: {len(batch)}) ---") # YENİ LOG EKLENDİ
    collated_batch = {}
    
    feature_keys = list(batch[0]['features'].keys())
    collated_features = {}
    
    if preparation_mode == 'windowing':
        for key in feature_keys:
            tensors_to_pad = [sample['features'][key] for sample in batch] # Liste içinde (C, T, H)
            permuted = [t.permute(1, 0, 2) for t in tensors_to_pad] # Liste içinde (T, C, H)
            padded = pad_sequence(permuted, batch_first=True, padding_value=0) # (B, T_padded, C, H)
            final_tensor = padded.permute(0, 2, 3, 1) # (B, C, H, T_padded)
            collated_features[key] = final_tensor
            # YENİ LOG EKLENDİ
            logger.debug(f"[Windowing] Feature '{key}' final collated shape: {final_tensor.shape} (B, C, H, T_padded)")
    
    elif preparation_mode == 'framify':
        padded_features = {}
        max_len = max(sample['features'][feature_keys[0]].shape[1] for sample in batch)
        for key in feature_keys:
            padded_list = []
            for sample in batch:
                tensor = sample['features'][key] # (C, T, H)
                pad_len = max_len - tensor.shape[1]
                padded_tensor = F.pad(tensor, (0, 0, 0, pad_len), 'constant', 0) # (C, T_padded, H)
                padded_list.append(padded_tensor)
            padded_features[key] = torch.stack(padded_list) # (B, C, T_padded, H)
            # YENİ LOG EKLENDİ
            logger.debug(f"[Framify] Feature '{key}' after padding to max_len={max_len}: {padded_features[key].shape} (B, C, T_padded, H)")

        framify_win_size = config['data'].get('framify_window_size', 9)
        pad_amount = framify_win_size // 2
        for key, tensor in padded_features.items():
            tensor_permuted = tensor.permute(0, 1, 3, 2) # (B, C, H, T_padded)
            inputs_padded = F.pad(tensor_permuted, (pad_amount, pad_amount), 'constant', 0)
            unfolded = inputs_padded.unfold(3, framify_win_size, 1) # (B, C, H, T_out, W)
            permuted_unfolded = unfolded.permute(0, 3, 1, 2, 4) # (B, T_out, C, H, W)
            B, T, C, H, W = permuted_unfolded.shape
            final_tensor = permuted_unfolded.reshape(B * T, C, H, W)
            collated_features[key] = final_tensor
            # YENİ LOG EKLENDİ
            logger.debug(f"[Framify] Feature '{key}' after unfold(W={W}) & reshape: {final_tensor.shape} (B*T, C, H, W)")

    collated_batch['features'] = collated_features

    target_keys = [k for k in batch[0].keys() if k != 'features']
    for key in target_keys:
        tensors_to_pad = [sample[key] for sample in batch]
        padding_value = -1 if key == 'tablature' else 0.0
        
        if tensors_to_pad[0].ndim == 2: # (S, T) or (P, T)
            permuted = [t.permute(1, 0) for t in tensors_to_pad] # (T, S) or (T, P)
            padded = pad_sequence(permuted, batch_first=True, padding_value=padding_value) # (B, T_padded, S)
            final_tensor = padded.permute(0, 2, 1) # (B, S, T_padded)
        else:
            final_tensor = pad_sequence(tensors_to_pad, batch_first=True, padding_value=padding_value)

        if preparation_mode == 'framify' and key in ['tablature', 'multipitch_target']:
            S_or_P = final_tensor.shape[1]
            reshaped_target = final_tensor.permute(0, 2, 1).reshape(-1, S_or_P) # (B*T, S) or (B*T, P)
            collated_batch[key] = reshaped_target
            # YENİ LOG EKLENDİ
            logger.debug(f"Target '{key}' reshaped for framify: {reshaped_target.shape}")
        else:
            collated_batch[key] = final_tensor
            # YENİ LOG EKLENDİ
            logger.debug(f"Target '{key}' final collated shape: {final_tensor.shape}")

    logger.debug(f"--- Collate Function End ---") # YENİ LOG EKLENDİ
    return collated_batch

def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    dataset_name = config['dataset']
    drive_path = config['dataset_configs'][dataset_name]['drive_data_path']
    local_path = config['dataset_configs'][dataset_name]['local_data_path']
    
    if not os.path.exists(local_path) or not os.listdir(local_path):
        logger.info("Data not found locally. Starting to copy from source...")
        logger.info(f"Source: {drive_path}")
        logger.info(f"Target: {local_path}")
        os.makedirs(local_path, exist_ok=True)
        start_time = time.time()
        try:
            subprocess.run(['rsync', '-aP', drive_path, local_path], check=True)
            end_time = time.time()
            logger.info(f"Data copy finished in {int(end_time - start_time)} seconds.")
        except Exception as e:
            logger.warning(f"rsync failed: {e}. Falling back to 'cp'...")
            try:
                subprocess.run(['cp', '-rv', os.path.join(drive_path, '.'), local_path], check=True)
                end_time = time.time()
                logger.info(f"Data copy with 'cp' finished in {int(end_time - start_time)} seconds.")
            except Exception as cp_e:
                logger.critical(f"Both rsync and cp failed. Could not copy data. Error: {cp_e}")
                raise
    else:
        logger.info("Data found locally. Skipping copy step.")
    logger.info("-" * 30)
    
    npz_dir = local_path
    file_list = sorted(os.listdir(npz_dir))
    all_npz_paths = [
        os.path.join(npz_dir, fname)
        for fname in tqdm(file_list, desc="Scanning .npz files")
        if fname.endswith(".npz")
    ]
    val_size = config['data']['validation_split_size']
    random_state = config['data']['random_state']
    train_paths, val_paths = train_test_split(all_npz_paths, test_size=val_size, random_state=random_state)
    
    logger.info("Creating Train and Validation datasets...")
    train_dataset = TablatureDataset(train_paths, config)
    val_dataset = TablatureDataset(val_paths, config)
    

    collate_with_config = partial(collate_fn, config=config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True if config['data'].get('num_workers', 0) > 0 else False,
        drop_last=True,
        collate_fn=collate_with_config 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True if config['data'].get('num_workers', 0) > 0 else False,
        drop_last=True,
        collate_fn=collate_with_config  
    )
    
    logger.info(f"Training data: {len(train_dataset)} files, {len(train_loader)} batches.")
    logger.info(f"Validation data: {len(val_dataset)} files, {len(val_loader)} batches.")
    return train_loader, val_loader, train_paths
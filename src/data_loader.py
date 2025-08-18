import os
import time
import subprocess
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from src.utils.logger import describe
import logging

logger = logging.getLogger(__name__)

class TablatureDataset(Dataset):
    def __init__(self, npz_paths: list[str], config: dict):
        self.npz_paths = npz_paths
        self.config = config
        self.data_config = config['data']
        self.loss_config = config['loss']
        active_feature_name = self.data_config['active_feature']
        self.feature_key = self.config['feature_definitions'][active_feature_name]['key']
        self.target_keys = self.data_config.get('target_keys', ['tablature'])
        self.samples_metadata = [{'file_path': path} for path in self.npz_paths]
        logger.info(f"Dataset initialized for feature '{active_feature_name}' and targets {self.target_keys}.")

    def __len__(self) -> int:
        return len(self.samples_metadata)

    def __getitem__(self, idx: int) -> dict:
        path = self.samples_metadata[idx]['file_path']
        sample = {}
        with np.load(path, allow_pickle=True) as data:
            feature_data_full = data[self.feature_key].astype(np.float32)
            if feature_data_full.ndim == 4:
                if feature_data_full.shape[0] == 1:
                    feature_data_full = feature_data_full.squeeze(0)
                else:
                    error_msg = f"Feature data at {path} has 4 dims, but first is not 1: {feature_data_full.shape}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            feature_tensor = torch.from_numpy(feature_data_full.copy()).permute(0, 2, 1)
            sample[self.feature_key] = feature_tensor

            for key in self.target_keys:
                if key not in data:
                    logger.warning(f"Target key '{key}' not found in {path}. Skipping.")
                    continue
                
                target_data = data[key]
                
                if key == 'tablature':
                    target_data = target_data.astype(np.int64)
                    target_tensor = torch.from_numpy(target_data.copy())
                    silence_class = self.config['data']['silence_class']
                    target_tensor[target_tensor == -1] = silence_class
                else:
                    target_tensor = torch.from_numpy(target_data.copy().astype(np.float32))

                sample[key] = target_tensor
        return sample

def collate_fn(batch: list[dict]) -> dict:
    all_keys = list(batch[0].keys())
    collated_batch = {}

    logger.debug(f"Collate_fn processing a batch of size: {len(batch)} with keys: {all_keys}")

    for key in all_keys:
        tensors_to_pad = [sample[key] for sample in batch]

        if tensors_to_pad[0].ndim == 3: # Feature
            permuted = [t.permute(1, 0, 2) for t in tensors_to_pad]
            padded = pad_sequence(permuted, batch_first=True, padding_value=0)
            final_tensor = padded.permute(0, 2, 3, 1) 
        elif tensors_to_pad[0].ndim == 2: # Target
            permuted = [t.permute(1, 0) for t in tensors_to_pad]
            padding_value = -1 if key == 'tablature' else 0
            padded = pad_sequence(permuted, batch_first=True, padding_value=padding_value)
            final_tensor = padded.permute(0, 2, 1)
        else:
            final_tensor = pad_sequence(tensors_to_pad, batch_first=True, padding_value=0)

        collated_batch[key] = final_tensor
        logger.debug(f"  -> Final collated '{key}' batch shape: {describe(collated_batch[key])}")

    return collated_batch

def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    drive_path = config['data']['drive_data_path']
    local_path = config['data']['local_data_path']
    
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
            logger.warning(f"rsync failed. Falling back to 'cp'...")
            try:
                subprocess.run(['cp', '-rv', os.path.join(drive_path, '.'), local_path], check=True)
                end_time = time.time()
                logger.info(f"Data copy with 'cp' finished in {int(end_time - start_time)} seconds.")
            except Exception as cp_e:
                logger.critical(f"Both rsync and cp failed. Could not copy data.")
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True if config['data'].get('num_workers', 0) > 0 else False,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True if config['data'].get('num_workers', 0) > 0 else False,
        drop_last=True,
        collate_fn=collate_fn
    )
    logger.info(f"Training data: {len(train_dataset)} files, {len(train_loader)} batches.")
    logger.info(f"Validation data: {len(val_dataset)} files, {len(val_loader)} batches.")
    return train_loader, val_loader, train_paths
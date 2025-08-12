import os
import time
import subprocess
import numpy as np
import torch
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

from .utils.agt_tools import tablature_to_logistic
from .utils.guitar_profile import GuitarProfile

class TablatureDataset(Dataset):
    def __init__(self, npz_paths: list[str], config: dict):
        self.npz_paths = npz_paths
        self.config = config
        self.data_config = config['data']
        self.loss_config = config['loss']

        active_feature_name = self.data_config['active_feature']
        self.feature_key = self.config['feature_definitions'][active_feature_name]['key']
        
        self.samples_metadata = [{'file_path': path} for path in self.npz_paths]
        print(f"Dataset initialized with {len(self.samples_metadata)} audio files.")

    def __len__(self) -> int:
        return len(self.samples_metadata)

    def __getitem__(self, idx: int) -> dict:
        path = self.samples_metadata[idx]['file_path']
        
        with np.load(path, allow_pickle=True) as data:
            feature_data_full = data[self.feature_key].astype(np.float32)
            tab_full = data["tablature"].astype(np.int64)

        if feature_data_full.ndim == 4:
            if feature_data_full.shape[0] == 1:
                feature_data_full = feature_data_full.squeeze(0)
            else:
                raise ValueError(
                    f"Feature data at {path} has 4 dimensions, but the first dimension is not 1: {feature_data_full.shape}"
                )
        
        feature_tensor = torch.from_numpy(feature_data_full.copy()).permute(0, 2, 1)
        tab_tensor = torch.from_numpy(tab_full.copy())

        silence_class = self.config['data']['silence_class']
        tab_tensor[tab_tensor == -1] = silence_class
        
        sample = {
            self.feature_key: feature_tensor,
            "tablature": tab_tensor
        }

        return sample

def collate_fn(batch: list[dict]) -> dict:
    feature_key = list(batch[0].keys())[0]

    features = [s[feature_key] for s in batch]
    tablatures = [s['tablature'] for s in batch]

    features_permuted = [f.permute(1, 0, 2) for f in features]
    tablatures_permuted = [t.permute(1, 0) for t in tablatures]

    features_padded = pad_sequence(features_permuted, batch_first=True, padding_value=0)
    tablatures_padded = pad_sequence(tablatures_permuted, batch_first=True, padding_value=-1)

    features_padded = features_padded.permute(0, 2, 3, 1)
    tablatures_padded = tablatures_padded.permute(0, 2, 1)

    collated_batch = {
        feature_key: features_padded,
        'tablature': tablatures_padded
    }
    
    return collated_batch


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    active_dataset = config['dataset']
    drive_path = config['dataset_configs'][active_dataset]['drive_data_path']
    local_path = config['dataset_configs'][active_dataset]['local_data_path']
    
    if not os.path.exists(local_path) or not os.listdir(local_path):
        print("Data not found locally. Starting to copy from source...")
        print(f"Source: {drive_path}")
        print(f"Target: {local_path}")
        os.makedirs(local_path, exist_ok=True)
        start_time = time.time()
        
        try:
            subprocess.run(['rsync', '-aP', drive_path, local_path], check=True)
            end_time = time.time()
            print(f"Data copy finished in {int(end_time - start_time)} seconds.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: rsync failed: {e}. Falling back to 'cp'...")
            try:
                subprocess.run(['cp', '-r', os.path.join(drive_path, '.'), local_path], check=True)
                end_time = time.time()
                print(f"Data copy with 'cp' finished in {int(end_time - start_time)} seconds.")
            except Exception as cp_e:
                print(f"CRITICAL ERROR: Both rsync and cp failed. Could not copy data. Error: {cp_e}")
                raise
    else:
        print("Data found locally. Skipping copy step.")
    print("-" * 30)

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
    
    print(f"\nTraining data: {len(train_dataset)} files, {len(train_loader)} batches.")
    print(f"Validation data: {len(val_dataset)} files, {len(val_loader)} batches.")
    
    return train_loader, val_loader, train_paths
import os
import time
import subprocess # Shell komutlarını çalıştırmak için
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TablatureDataset(Dataset):
    def __init__(self, npz_paths, config):
        self.npz_paths = npz_paths
        self.config = config
        self.window_size = config['data']['window_size']
        self.hop_size = config['data'].get('hop_size', self.window_size // 2)
        
        self.max_fret = config['data'].get('max_fret', 19)
        self.include_silence = config['data'].get('include_silence', False)

        self.samples_metadata = []
        print("Creating dataset map (not loading data into RAM)...")
        for i, path in enumerate(tqdm(self.npz_paths, desc="Mapping samples")):
            with np.load(path) as data:
                T = data["cqt"].shape[-1]
            
            num_windows_in_file = (T - self.window_size) // self.hop_size + 1
            for j in range(num_windows_in_file):
                self.samples_metadata.append({'file_idx': i, 'start_frame': j * self.hop_size})

    def __len__(self):
        return len(self.samples_metadata)

    def __getitem__(self, idx):
        metadata = self.samples_metadata[idx]
        file_idx = metadata['file_idx']
        start_frame = metadata['start_frame']
        
        path = self.npz_paths[file_idx]
        with np.load(path) as data:
            cqt_full = data["cqt"]
            tab_full = data["tablature"]

        end_frame = start_frame + self.window_size
        cqt_window = cqt_full[:, :, start_frame:end_frame]
        tab_window = tab_full[:, start_frame:end_frame]
        
        cqt = torch.tensor(cqt_window, dtype=torch.float32)
        tab = torch.tensor(tab_window, dtype=torch.long)
        
        tab[tab > self.max_fret] = -1
        
        if self.include_silence:
            silence_class = self.config['data'].get('silence_class', 20)
            tab[tab == -1] = silence_class
        
        return {"cqt": cqt, "tablature": tab}

def get_dataloaders(config):
    """
    Config dosyasını kullanarak:
    1. Verinin yerel kopyasının olup olmadığını kontrol eder, yoksa Drive'dan kopyalar.
    2. Train ve validation DataLoader'larını oluşturur ve döndürür.
    """
    drive_path = config['data']['drive_data_path']
    local_path = config['data']['local_data_path']
    
    if not os.path.exists(local_path):
        print(f"Data not found on local disk. Starting copy process...")
        print(f"Source: {drive_path}")
        print(f"Target: {local_path}")
        
        os.makedirs(local_path, exist_ok=True)
        start_time = time.time()
        
        try:
            subprocess.run(['rsync', '-aP', drive_path, local_path], check=True)
            end_time = time.time()
            print(f"Data copying completed in {int(end_time - start_time)} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to copy data. rsync command failed: {e}")
            raise
    else:
        print(f"Data already exists on the local disk. Copy step skipped.")
    print("------------------------------------")

    npz_dir = local_path
    all_npz_paths = [os.path.join(npz_dir, fname) for fname in sorted(os.listdir(npz_dir)) if fname.endswith(".npz")]

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
        pin_memory=True if config['data'].get('num_workers', 0) > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True if config['data'].get('num_workers', 0) > 0 else False
    )
    
    print(f"\nTrain data: {len(train_dataset)} örnek, {len(train_loader)} batch.")
    print(f"Validation data: {len(val_dataset)} örnek, {len(val_loader)} batch.")
    
    return train_loader, val_loader, train_paths
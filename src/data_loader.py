import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import logging
import torchaudio.transforms as T
import random
import shutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from functools import partial

logger = logging.getLogger(__name__)

class TablatureDataset(Dataset):
    """
    A PyTorch Dataset for CNN_MTL (Context Window) model.
    """
    def __init__(self, npz_paths: list[str], config: dict, augment: bool = False, zero_shot_map: dict = None):
        self.npz_paths = npz_paths
        self.config = config
        self.data_config = config['data']
        self.instrument_config = config['instrument']
        self.loss_config = config['loss']
        
        self.zero_shot_map = zero_shot_map
        self.silence_class = self.instrument_config.get('silence_class', self.instrument_config.get('num_frets', 19) + 1)
        
        # --- KRİTİK DÜZELTME: Active Features List Conversion ---
        # Config'den dict veya list gelebilir. Garantili listeye çeviriyoruz.
        raw_features = self.data_config['active_features']
        if isinstance(raw_features, dict):
            self.active_features = [k for k, v in raw_features.items() if v]
        elif isinstance(raw_features, list):
            self.active_features = raw_features
        else:
            # Fallback
            self.active_features = ['hcqt']
            logger.warning("Active features format not recognized, defaulting to ['hcqt']")
            
        self.target_keys = self.data_config.get('target_keys', {})
        if isinstance(self.target_keys, dict):
            self.target_keys = [k for k, v in self.target_keys.items() if v]
        
        self.feature_map = self.data_config.get('feature_to_file_map', {})
        self.chunk_size = self.data_config.get('framify_chunk_size', 500)
        
        # Multipitch Generation
        self.generate_multipitch = 'multipitch_target' in self.target_keys
        
        if self.generate_multipitch:
            self.tuning = torch.tensor(self.instrument_config['tuning'], dtype=torch.long)
            self.min_midi = self.instrument_config['min_midi']
            self.max_midi = self.instrument_config['max_midi']
            self.num_pitches = self.max_midi - self.min_midi + 1
        
        self.augment = augment
        self.augmentation_transforms = None
        if self.augment:
            aug_config = self.data_config.get('augmentation', {})
            time_mask = aug_config.get('time_masking_param', 30)
            freq_mask = aug_config.get('freq_masking_param', 24)
            self.augmentation_transforms = torch.nn.Sequential(
                T.FrequencyMasking(freq_mask_param=freq_mask),
                T.TimeMasking(time_mask_param=time_mask)
            )
            
        self.samples_metadata = [{'file_path': path} for path in self.npz_paths]

    def __len__(self) -> int:
        return len(self.samples_metadata)

    def __getitem__(self, idx: int) -> dict:
        path = self.samples_metadata[idx]['file_path']
        if path is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        try:            
            data = np.load(path, allow_pickle=True)
            features_dict = {}
            targets_dict = {}
            
            # --- 1. Load Features ---
            for feature_key in self.active_features:
                npz_key = self.feature_map.get(feature_key, feature_key)
                if npz_key not in data:
                    raise KeyError(f"Feature '{npz_key}' not found in {path}.")
                
                feature_data = data[npz_key].astype(np.float32)
                if feature_key == 'power' and feature_data.ndim == 1:
                    feature_data = feature_data[np.newaxis, np.newaxis, :] 
                elif feature_data.ndim == 4 and feature_data.shape[0] == 1:
                    feature_data = feature_data.squeeze(0)
                features_dict[feature_key] = torch.from_numpy(feature_data.copy())
            
            # --- 2. Load Targets ---
            if 'tablature' in self.target_keys and 'tablature' in data:
                tab_data = data['tablature'].astype(np.int64)
                tab_tensor = torch.from_numpy(tab_data.copy())
                tab_tensor[tab_tensor == -1] = self.silence_class
                targets_dict['tablature'] = tab_tensor

            if 'hand_pos_target' in self.target_keys and 'hand_position' in data:
                hp_data = data['hand_position']
                targets_dict['hand_pos_target'] = torch.from_numpy(hp_data.copy()).long()

            if 'activity_target' in self.target_keys and 'string_activity' in data:
                act_data = data['string_activity'].astype(np.float32)
                targets_dict['activity_target'] = torch.from_numpy(act_data.copy())

            if 'pitch_class_target' in self.target_keys and 'pitch_class' in data:
                pc_data = data['pitch_class'].astype(np.float32)
                targets_dict['pitch_class_target'] = torch.from_numpy(pc_data.copy())

            if 'onset_target' in self.target_keys:
                 if 'onsets' in data:
                     onset_raw = data['onsets']
                     if onset_raw.ndim == 3: 
                         onset_str = np.any(onset_raw, axis=1).astype(np.float32)
                     else:
                         onset_str = onset_raw.astype(np.float32)
                     targets_dict['onset_target'] = torch.from_numpy(onset_str.copy())

            data.close()

            # --- 3. Slicing ---
            # Burada artık active_features kesinlikle liste olduğu için [0] hata vermez.
            first_feature_key = self.active_features[0]
            full_length = features_dict[first_feature_key].shape[-1]
            target_length = self.chunk_size

            if full_length < target_length:
                pad_amount = target_length - full_length
                for key in features_dict:
                    features_dict[key] = F.pad(features_dict[key], (0, pad_amount), 'constant', 0)
                for key in targets_dict:
                    pad_val = self.silence_class if key == 'tablature' else 0
                    targets_dict[key] = F.pad(targets_dict[key], (0, pad_amount), 'constant', pad_val)
            else:
                start_idx = random.randint(0, full_length - target_length)
                end_idx = start_idx + target_length
                for key in features_dict:
                    features_dict[key] = features_dict[key][..., start_idx:end_idx]
                for key in targets_dict:
                    targets_dict[key] = targets_dict[key][..., start_idx:end_idx]

            # --- 4. Multipitch Generation ---
            tab_slice = targets_dict.get('tablature')
            if self.generate_multipitch and tab_slice is not None:
                num_strings, num_steps = tab_slice.shape
                y_multipitch = torch.zeros((self.num_pitches, num_steps), dtype=torch.float32)
                mask = (tab_slice < self.silence_class) & (tab_slice >= 0)
                for s in range(num_strings):
                    string_notes = self.tuning[s] + tab_slice[s, :]
                    valid_notes_mask = mask[s, :]
                    active_indices = torch.where(valid_notes_mask)[0]
                    for t in active_indices:
                        midi_note = string_notes[t]
                        if self.min_midi <= midi_note <= self.max_midi:
                            pitch_index = midi_note - self.min_midi
                            y_multipitch[pitch_index, t] = 1.0
                targets_dict['multipitch_target'] = y_multipitch

            # --- 5. Augmentation ---
            if self.augment and self.augmentation_transforms:
                for key in features_dict:
                    if key != 'power':
                        features_dict[key] = self.augmentation_transforms(features_dict[key])
            
            if self.zero_shot_map and tab_slice is not None:
                mask = torch.zeros(tab_slice.shape[1], dtype=torch.bool)
                for s_idx, frets in self.zero_shot_map.items():
                    s = int(s_idx)
                    if s < tab_slice.shape[0]:
                        for fret in frets:
                            mask.logical_or_(tab_slice[s, :] == fret)
                if torch.any(mask):
                    tab_slice[:, mask] = self.silence_class
                    targets_dict['tablature'] = tab_slice

            sample = {'features': features_dict}
            sample.update(targets_dict)
            return sample

        except Exception as e:
            # Hata durumunda log basıp başka dosya deniyoruz
            # recursion limit'e takılmamak için bu logu görmek önemli
            logger.error(f"Error processing file '{path}': {e}. Skipping.")
            return self.__getitem__(random.randint(0, len(self) - 1))

def collate_fn(batch: list[dict], config: dict) -> dict:
    if not batch: return {}
    collated_batch = {}
    
    feature_keys = list(batch[0]['features'].keys())
    collated_features = {}
    framify_win_size = config['data'].get('framify_window_size', 9)
    pad_amount = framify_win_size // 2

    for key in feature_keys:
        tensor_batch = torch.stack([sample['features'][key] for sample in batch])            
        tensor_padded = F.pad(tensor_batch, (pad_amount, pad_amount), 'constant', 0)
        unfolded = tensor_padded.unfold(3, framify_win_size, 1)
        unfolded = unfolded.permute(0, 3, 1, 2, 4)
        B, T_chunk, C, F_bins, W = unfolded.shape
        collated_features[key] = unfolded.reshape(B * T_chunk, C, F_bins, W)

    collated_batch['features'] = collated_features

    target_keys_present = [k for k in batch[0].keys() if k != 'features']
    framify_keys = ['tablature', 'hand_pos_target', 'activity_target', 'pitch_class_target', 'multipitch_target', 'onset_target']

    for key in target_keys_present:
        final_tensor = torch.stack([sample[key] for sample in batch])
        if key in framify_keys:
            if final_tensor.ndim == 3: 
                reshaped_target = final_tensor.permute(0, 2, 1).reshape(-1, final_tensor.shape[1])
            else: 
                reshaped_target = final_tensor.reshape(-1)
            collated_batch[key] = reshaped_target
        else:
            collated_batch[key] = final_tensor

    return collated_batch

def prepare_dataset_files(config: dict) -> tuple[np.ndarray, list[str]]:
    dataset_name = config['dataset']
    if dataset_name in config: dataset_conf = config[dataset_name]
    else: dataset_conf = config['dataset_configs'][dataset_name]

    drive_path = dataset_conf['drive_data_path']
    local_path = dataset_conf['local_data_path']
    
    if not os.path.exists(local_path) or not os.listdir(local_path):
        logger.info(f"Copying data to local path: {local_path}")
        os.makedirs(local_path, exist_ok=True)
        
        src_files = [os.path.join(drive_path, f) for f in os.listdir(drive_path) if f.endswith('.npz')]
        if not src_files:
            logger.warning(f"No .npz files found in {drive_path}")
        else:
            logger.info(f"Transferring {len(src_files)} files...")
            for src_file in tqdm(src_files, desc="Copying Dataset", unit="file"):
                shutil.copy(src_file, local_path)
    
    all_files = sorted(os.listdir(local_path))
    all_npz_paths = []
    
    for fname in tqdm(all_files, desc="Scanning Local Files", unit="file"):
        if fname.endswith(".npz"):
            all_npz_paths.append(os.path.join(local_path, fname))
            
    all_npz_paths = np.array(all_npz_paths)
    groups = [os.path.basename(path).split('_')[0] for path in all_npz_paths]
    
    logger.info(f"Total files: {len(all_npz_paths)}, Unique groups: {len(np.unique(groups))}")
    return all_npz_paths, groups

def get_dataloaders(config: dict, train_paths: list[str], val_paths: list[str], test_paths: list[str], zero_shot_map: dict = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    logger.info("Initializing DataLoaders...")
    data_conf = config['data']
    is_augment = data_conf.get('augmentation', {}).get('enabled', False)
    
    train_dataset = TablatureDataset(train_paths, config, augment=is_augment, zero_shot_map=None)
    val_dataset = TablatureDataset(val_paths, config, augment=False, zero_shot_map=zero_shot_map) 
    
    collate_with_config = partial(collate_fn, config=config)
    
    train_loader = DataLoader(train_dataset, batch_size=data_conf['batch_size'], shuffle=True, num_workers=data_conf.get('num_workers', 0), pin_memory=True, drop_last=True, collate_fn=collate_with_config)
    val_loader = DataLoader(val_dataset, batch_size=data_conf['batch_size'], shuffle=False, num_workers=data_conf.get('num_workers', 0), pin_memory=True, drop_last=True, collate_fn=collate_with_config)
    
    test_dataset = TablatureDataset(test_paths, config, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=data_conf['batch_size'], shuffle=False, num_workers=data_conf.get('num_workers', 0), pin_memory=True, drop_last=False, collate_fn=collate_with_config)
    
    return train_loader, val_loader, test_loader
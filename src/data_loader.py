import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import logging
import torchaudio.transforms as T
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

logger = logging.getLogger(__name__)

class TablatureDataset(Dataset):
    """
    A PyTorch Dataset for tablature data.
    
    This Dataset is designed to work with full-track NPZ files. On each __getitem__ call,
    it loads a full track and returns a random slice of a predefined length. This approach
    ensures that the model sees a wide variety of data from all parts of the tracks
    across epochs, improving generalization.
    
    It supports two main preparation modes, configured via the 'active_preparation_mode' key:
    1. 'windowing': Returns a slice of 'window_size' length, intended as direct input for
       sequence models like CRNNs or Transformers.
    2. 'framify': Returns a larger slice of 'framify_chunk_size' length, intended as
       "raw material" for a collate_fn that will further process it into smaller
       frame-wise windows for CNNs.
    """
    def __init__(self, npz_paths: list[str], config: dict, augment: bool = False, zero_shot_map: dict = None):
        """
        Initializes the Dataset.
        
        Args:
            npz_paths (list[str]): List of file paths to the NPZ data files.
            config (dict): The main configuration dictionary.
            augment (bool): Whether to apply data augmentation.
            zero_shot_map (dict, optional): A map for zero-shot class handling. Defaults to None.
        """
        self.npz_paths = npz_paths
        self.config = config
        self.data_config = config['data']
        self.instrument_config = config['instrument']
        self.loss_config = config['loss']
        
        self.zero_shot_map = zero_shot_map
        self.silence_class = self.instrument_config.get('silence_class', self.instrument_config.get('num_frets', 19) + 1)
        
        self.active_features = self.data_config['active_features']
        self.target_keys = self.data_config.get('target_keys', ['tablature'])
        
        self.feature_map = self.data_config.get('feature_to_file_map', {})
        if not self.feature_map:
            logger.warning("`feature_to_file_map` not found in config. Assuming feature keys match NPZ keys.")

        # Get preparation mode and parameters from the config
        self.active_preparation_mode = self.data_config.get('active_preparation_mode', 'windowing')
        self.window_size = self.data_config.get('window_size')
        self.framify_chunk_size = self.data_config.get('framify_chunk_size')
        
        logger.info(f"Dataset mode: '{self.active_preparation_mode}'. Window size: {self.window_size}, Framify chunk size: {self.framify_chunk_size}")

        # Auxiliary loss setup
        self.aux_loss_enabled = self.loss_config.get('auxiliary_loss', {}).get('enabled', False)
        if self.aux_loss_enabled:
            logger.info("Auxiliary loss is enabled. Multipitch target will be generated on-the-fly.")
            self.tuning = torch.tensor(self.instrument_config['tuning'], dtype=torch.long)
            self.min_midi = self.instrument_config['min_midi']
            self.max_midi = self.instrument_config['max_midi']
            self.num_pitches = self.max_midi - self.min_midi + 1
        
        # Augmentation setup
        self.augment = augment
        self.augmentation_transforms = None
        if self.augment:
            aug_config = self.data_config.get('augmentation', {})
            time_mask_param = aug_config.get('time_masking_param', 30)
            freq_mask_param = aug_config.get('freq_masking_param', 24)
            self.augmentation_transforms = torch.nn.Sequential(
                T.FrequencyMasking(freq_mask_param=freq_mask_param),
                T.TimeMasking(time_mask_param=time_mask_param)
            )
            logger.info(f"On-the-fly data augmentation ENABLED. FreqMask: {freq_mask_param}, TimeMask: {time_mask_param}")
            
        self.samples_metadata = [{'file_path': path} for path in self.npz_paths]
        if len(self.samples_metadata) > 0:
            logger.debug(f"Dataset initialized for {len(self.samples_metadata)} files. Features: {self.active_features}, Targets: {self.target_keys}.")

    def __len__(self) -> int:
        return len(self.samples_metadata)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single data item (a random slice from a track).
        Includes robust error handling to prevent training crashes.
        """
        path = self.samples_metadata[idx]['file_path']
        
        if path is None:
            logger.warning(f"Received a None path for idx={idx}. Skipping.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        try:            
            data = np.load(path, allow_pickle=True)
            
            features_dict = {}
            targets_dict = {}
            
            for feature_key in self.active_features:
                npz_key = self.feature_map.get(feature_key, feature_key)
                if npz_key not in data:
                    raise KeyError(f"Feature key '{npz_key}' (from '{feature_key}') not found in {path}.")
                feature_data = data[npz_key].astype(np.float32)
                if feature_key == 'power' and feature_data.ndim == 1:
                    feature_data = feature_data[np.newaxis, np.newaxis, :] # (T,) -> (1, 1, T)
                elif feature_data.ndim == 4 and feature_data.shape[0] == 1:
                    feature_data = feature_data.squeeze(0)
                features_dict[feature_key] = torch.from_numpy(feature_data.copy())
            
            for key in self.target_keys:
                if key not in data: continue
                if key == 'tablature':
                    tab_data = data['tablature'].astype(np.int64)
                    tab_tensor = torch.from_numpy(tab_data.copy())
                    tab_tensor[tab_tensor == -1] = self.silence_class
                    targets_dict['tablature'] = tab_tensor
                
                elif key == 'onsets':                
                    onset_data_pitch_wise = data[key].astype(np.float32) # pitch-wise                    
                    onset_data_per_string = np.any(onset_data_pitch_wise, axis=1).astype(np.float32) # (S, T)                    
                    activity_target_np = np.any(onset_data_per_string, axis=0).astype(np.float32) # (T,)
                    targets_dict['activity_target'] = torch.from_numpy(activity_target_np.copy())
                    
                    if 'onset_loss' in self.loss_config and self.loss_config['onset_loss'].get('enabled', False):
                        targets_dict['onset_target'] = torch.from_numpy(onset_data_per_string.copy())

                elif key == 'offsets':
                    if 'offset_loss' in self.loss_config and self.loss_config['offset_loss'].get('enabled', False):
                        offset_data_pitch_wise = data[key].astype(np.float32)
                        offset_data_per_string = np.any(offset_data_pitch_wise, axis=1).astype(np.float32)
                        targets_dict['offset_target'] = torch.from_numpy(offset_data_per_string.copy())

            data.close()

            first_feature_key = self.active_features[0]
            full_length = features_dict[first_feature_key].shape[-1]
            
            if self.active_preparation_mode == 'windowing':
                target_length = self.window_size
            elif self.active_preparation_mode == 'framify':
                target_length = self.framify_chunk_size
            else:
                raise ValueError(f"Unknown preparation mode: '{self.active_preparation_mode}'")

            if not isinstance(target_length, int) or target_length <= 0:
                raise ValueError(f"Invalid target_length '{target_length}' for mode '{self.active_preparation_mode}'")
            
            # --- 3. Perform random slicing or padding ---
            if full_length < target_length:
                pad_amount = target_length - full_length
                for key in features_dict:
                    features_dict[key] = F.pad(features_dict[key], (0, pad_amount), 'constant', 0)
                for key in targets_dict:
                    pad_value = self.silence_class if key == 'tablature' else 0.0
                    targets_dict[key] = F.pad(targets_dict[key], (0, pad_amount), 'constant', pad_value)
                logger.debug(f"Padded '{os.path.basename(path)}' from {full_length} to {target_length} frames.")
            else:
                start_idx = random.randint(0, full_length - target_length)
                end_idx = start_idx + target_length
                for key in features_dict:
                    features_dict[key] = features_dict[key][..., start_idx:end_idx]
                for key in targets_dict:
                    targets_dict[key] = targets_dict[key][..., start_idx:end_idx]

            # --- 4. Apply post-slicing operations ---
            if self.augment and self.augmentation_transforms:
                for key in features_dict:
                    if key != 'power':
                        features_dict[key] = self.augmentation_transforms(features_dict[key])
            
            tablature_tensor = targets_dict.get('tablature')

            if self.zero_shot_map and tablature_tensor is not None:
                mask = torch.zeros(tablature_tensor.shape[1], dtype=torch.bool)
                for s_idx, frets in self.zero_shot_map.items():
                    s = int(s_idx) # Ensure string index is integer
                    if s < tablature_tensor.shape[0]:
                        for fret in frets:
                            mask.logical_or_(tablature_tensor[s, :] == fret)
                if torch.any(mask):
                    tablature_tensor[:, mask] = self.silence_class
                    targets_dict['tablature'] = tablature_tensor

            if self.aux_loss_enabled and tablature_tensor is not None:
                num_strings, num_steps = tablature_tensor.shape
                y_multipitch = torch.zeros((self.num_pitches, num_steps), dtype=torch.float32)
                mask = (tablature_tensor < self.silence_class) & (tablature_tensor >= 0)
                for s in range(num_strings):
                    string_notes = self.tuning[s] + tablature_tensor[s, :]
                    valid_notes_mask = mask[s, :]
                    for t in torch.where(valid_notes_mask)[0]:
                        midi_note = string_notes[t]
                        if self.min_midi <= midi_note <= self.max_midi:
                            pitch_index = midi_note - self.min_midi
                            y_multipitch[pitch_index, t] = 1.0
                targets_dict['multipitch_target'] = y_multipitch

            # --- 5. Assemble final sample and return ---
            sample = {'features': features_dict}
            sample.update(targets_dict)
            return sample

        except Exception as e:
            logger.error(f"Error processing file '{path}': {e}. Skipping and trying another random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

def collate_fn(batch: list[dict], config: dict) -> dict:
    """
    Custom collate function to handle batching of data samples.
    It prepares the data according to the 'active_preparation_mode'.
    """
    preparation_mode = config['data']['active_preparation_mode']
    if not batch:
        return {}
    
    collated_batch = {}
    
    feature_keys = list(batch[0]['features'].keys())
    collated_features = {}

    if preparation_mode == 'windowing':
        # For CRNNs, all samples are already sliced to the same length.
        # We just stack them into a batch.
        for key in feature_keys:
            # Stack tensors along a new batch dimension
            collated_features[key] = torch.stack([sample['features'][key] for sample in batch])
    
    elif preparation_mode == 'framify':
        framify_win_size = config['data'].get('framify_window_size', 9)
        pad_amount = framify_win_size // 2

        for key in feature_keys:
            tensor_batch = torch.stack([sample['features'][key] for sample in batch])            
            tensor_padded_window = F.pad(tensor_batch, (pad_amount, pad_amount), 'constant', 0)
            unfolded = tensor_padded_window.unfold(3, framify_win_size, 1)
            unfolded = unfolded.permute(0, 3, 1, 2, 4)
            B, T, C, F_bins, W = unfolded.shape
            collated_features[key] = unfolded.reshape(B * T, C, F_bins, W)

    collated_batch['features'] = collated_features

    target_keys = [k for k in batch[0].keys() if k != 'features']
    for key in target_keys:
        tensors_to_stack = [sample[key] for sample in batch]
        
        final_tensor = torch.stack(tensors_to_stack)

        if preparation_mode == 'framify' and key in ['tablature', 'multipitch_target', 'onset_target', 'offset_target', 'activity_target']:
            if final_tensor.ndim == 3: # (B, Channels, Time) -> (B*T, Channels)
                reshaped_target = final_tensor.permute(0, 2, 1).reshape(-1, final_tensor.shape[1])
            else: # (B, Time) -> (B*T)
                reshaped_target = final_tensor.reshape(-1)
            collated_batch[key] = reshaped_target
        else:
            collated_batch[key] = final_tensor

    return collated_batch

def prepare_dataset_files(config: dict) -> tuple[np.ndarray, list[str]]:
    """
    Prepares dataset files for K-Fold cross-validation.
    Copies data to a local directory if not present and returns file paths and group IDs.
    """
    dataset_name = config['dataset']
    drive_path = config['dataset_configs'][dataset_name]['drive_data_path']
    local_path = config['dataset_configs'][dataset_name]['local_data_path']
    
    if not os.path.exists(local_path) or not os.listdir(local_path):
        logger.info(f"Data not found locally. Copying from '{drive_path}' to '{local_path}'...")
        os.makedirs(local_path, exist_ok=True)
        try:
            subprocess.run(['rsync', '-aP', drive_path.rstrip('/') + '/', local_path], check=True)
        except Exception:
            logger.warning("rsync failed. Falling back to 'cp'...")
            subprocess.run(['cp', '-rv', os.path.join(drive_path, '.'), local_path], check=True)
        logger.info("Data copy finished.")
    else:
        logger.info("Data found locally, skipping copy.")
    
    all_npz_paths = np.array([os.path.join(local_path, fname) for fname in sorted(os.listdir(local_path)) if fname.endswith(".npz")])
    groups = [os.path.basename(path).split('_')[0] for path in all_npz_paths]
    
    logger.info(f"Total of {len(all_npz_paths)} files and {len(np.unique(groups))} unique groups (guitarists) found.")
    
    return all_npz_paths, groups


def get_dataloaders(config: dict, train_paths: list[str], val_paths: list[str], test_paths: list[str], zero_shot_map: dict = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns the training, validation, AND test DataLoaders for a specific split.
    """
    logger.info("Initializing train, validation, and test datasets for the current fold...")
    
    data_conf = config['data']
    is_augment_enabled = data_conf.get('augmentation', {}).get('enabled', False)
    
    train_dataset = TablatureDataset(train_paths, config, augment=is_augment_enabled, zero_shot_map=None)
    val_dataset = TablatureDataset(val_paths, config, augment=False, zero_shot_map=zero_shot_map) 
    
    collate_with_config = partial(collate_fn, config=config)
    
    train_loader = DataLoader(
        train_dataset, batch_size=data_conf['batch_size'], shuffle=True,
        num_workers=data_conf.get('num_workers', 0),
        pin_memory=data_conf.get('num_workers', 0) > 0,
        drop_last=True, collate_fn=collate_with_config 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_conf['batch_size'], shuffle=False,
        num_workers=data_conf.get('num_workers', 0),
        pin_memory=data_conf.get('num_workers', 0) > 0,
        drop_last=True, collate_fn=collate_with_config   
    )
    
    test_dataset = TablatureDataset(test_paths, config, augment=False, zero_shot_map=None)
    test_loader = DataLoader(
        test_dataset, batch_size=data_conf['batch_size'], shuffle=False,
        num_workers=data_conf.get('num_workers', 0),
        pin_memory=data_conf.get('num_workers', 0) > 0,
        drop_last=False, 
        collate_fn=collate_with_config
    )
    
    logger.info(f"DataLoaders created. Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}.")
    
    return train_loader, val_loader, test_loader
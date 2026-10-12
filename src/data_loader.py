import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
import torchaudio.transforms as T
import random
from torch.utils.data import Dataset, DataLoader
from functools import partial

logger = logging.getLogger(__name__)

class TablatureDataset(Dataset):
    """
    A PyTorch Dataset for CNN_MTL (Context Window) model.
    """
    def __init__(self, npz_paths: list[str], config: dict, augment: bool = False,
                 zero_shot_map: dict | None = None):
        """Configures feature/target keys, silence class, chunk size, and augmentation from `config`.

        Args:
            npz_paths: Paths to feature-enriched .npz files backing this split.
            config: Full experiment config (uses the `data`, `instrument`, and `loss` sections).
            augment: Whether to apply frequency/time masking augmentation in `__getitem__`.
            zero_shot_map: Optional per-string set of frets to mask out to the silence class,
                used for zero-shot fret evaluation.
        """
        self.npz_paths = npz_paths
        self.config = config
        self.data_config = config['data']
        self.instrument_config = config['instrument']
        self.loss_config = config['loss']

        self.zero_shot_map = zero_shot_map
        self.silence_class = self.instrument_config.get('silence_class', self.instrument_config.get('num_frets', 19) + 1)
        
        # --- Active features list conversion ---
        # The config may provide active_features as a dict or a list; normalize to a list either way.
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
        """Returns the number of feature files in the dataset."""
        return len(self.samples_metadata)

    def __getitem__(self, idx: int) -> dict:
        """Loads one file's features/targets and returns a fixed-length training chunk.

        Reads the active feature arrays and any configured target arrays from the
        .npz file at `idx`, randomly crops or pads them to `chunk_size` frames
        (every split, including validation, as in the paper's runs), derives the
        multi-pitch target from the tablature slice if enabled, applies
        augmentation, and applies zero-shot fret masking if configured.

        Raises:
            RuntimeError: If the file cannot be read or is missing a feature.
                Substituting a different random file instead would silently
                change the split, so loading fails loudly.
        """
        path = self.samples_metadata[idx]['file_path']

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
            # active_features is guaranteed to be a list at this point, so indexing [0] is safe.
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
            raise RuntimeError(f"Error processing file '{path}': {e}") from e

def collate_fn(batch: list[dict], config: dict) -> dict:
    """Batches dataset samples and unfolds each frame into a context window.

    Each feature tensor is stacked into a batch, padded along the time axis by
    `framify_window_size // 2` on each side, then unfolded so every original
    time step becomes its own `(channels, freq, framify_window_size)` sample,
    yielding shape `(batch * chunk_size, channels, freq, framify_window_size)`.
    Targets listed in `framify_keys` are reshaped to line up one row per
    unfolded frame; other targets are simply stacked per-sample.
    """
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
    """Lists every .npz feature file under the configured data root and derives its player group.

    Each file's group is the prefix before the first underscore in its
    filename (the GuitarSet player id), which the caller uses to build the
    leave-one-player-out cross-validation folds.

    Raises:
        FileNotFoundError: If the configured `data_root` directory does not exist.
    """
    dataset_name = config['dataset']
    if dataset_name in config: dataset_conf = config[dataset_name]
    else: dataset_conf = config['dataset_configs'][dataset_name]

    data_root = dataset_conf['data_root']
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"'{dataset_name}.data_root' does not exist: {data_root}\n"
            f"Point it at a directory of feature-enriched .npz files "
            f"(see extract_features.py) via the config or --data-root."
        )

    all_files = sorted(f for f in os.listdir(data_root) if f.endswith(".npz"))
    all_npz_paths = np.array([os.path.join(data_root, f) for f in all_files])
    groups = [os.path.basename(path).split('_')[0] for path in all_npz_paths]

    logger.info(f"Total files: {len(all_npz_paths)}, Unique groups: {len(np.unique(groups))}")
    return all_npz_paths, groups

def fold_split(unique_groups: list[str], group_to_paths: dict, fold_index: int):
    """Leave-one-player-out split for one fold.

    Fold `fold_index` holds out player `fold_index` for testing and uses player
    `(fold_index + 1) % n` for validation; all other players are used for training.

    Returns:
        (train_paths, val_paths, test_paths, test_group, val_group)
    """
    n_splits = len(unique_groups)
    test_group = unique_groups[fold_index]
    val_group = unique_groups[(fold_index + 1) % n_splits]
    train_groups = [g for g in unique_groups if g not in (test_group, val_group)]
    train_paths = [p for g in train_groups for p in group_to_paths[g]]
    return train_paths, list(group_to_paths[val_group]), list(group_to_paths[test_group]), test_group, val_group

def get_dataloaders(config: dict, train_paths: list[str], val_paths: list[str], test_paths: list[str], zero_shot_map: dict | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Builds the train/validation/test DataLoaders for one cross-validation fold.

    Every split draws a random `framify_chunk_size` crop per item, so the
    validation metrics that drive checkpoint selection and early stopping come
    from randomly cropped segments, as in the paper's runs. Only training data
    is shuffled and (per `data.augmentation.enabled`) augmented. The validation
    dataset accepts an optional `zero_shot_map` for masking specific frets. All
    three loaders share `collate_fn` for context-window unfolding. Reported test
    metrics come from evaluate.py's full-track inference, not from these loaders.
    """
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
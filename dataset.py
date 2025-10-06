import os
import json
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Any
import math
import torch.nn.functional as F
import random
from collections import defaultdict


def load_vocabs(vocab_path: str) -> Dict[str, Any]:
    """Loads vocabularies and augments with per-component PAD/NONE indices.

    For 3-part prediction, only `root`, `quality`, and `bass` are loaded.
    """
    with open(vocab_path, 'rb') as f:
        data = pickle.load(f)

    root_map = data['root_to_idx']
    pad_token = 'PAD'
    none_tokens = ['N', 'None']  # allow either spelling in source vocabs
    bass_map = root_map

    # Only keep the three chord parts for prediction
    vocabs = {
        'root': root_map,
        'quality': data['quality_to_idx'],
        'bass': bass_map,
        'key': data['key_to_idx'],
    }

    # Global root PAD index (back-compat)
    vocabs['pad_idx'] = root_map[pad_token]

    # Add per-component PAD and NONE indices
    for comp, comp_map in list(vocabs.items()):
        if comp == 'pad_idx':
            continue
        # per-component PAD index (must exist)
        comp_pad_idx = comp_map.get(pad_token)
        if comp_pad_idx is None:
            raise ValueError(f"Component '{comp}' vocab lacks PAD token")
        vocabs[f'{comp}_pad_idx'] = comp_pad_idx

        # NONE index preference: N > None > PAD
        none_idx = None
        for tok in none_tokens:
            if tok in comp_map:
                none_idx = comp_map[tok]
                break
        if none_idx is None:
            none_idx = comp_pad_idx
        vocabs[f'{comp}_none_idx'] = none_idx

    return vocabs


class PianoRollDataset(Dataset):
    """Dataset for piano roll representation."""
    pad_idx = -1 # Will be updated in __init__
    def __init__(
        self,
        data_root: str,
        config: dict,
        vocabs: Dict[str, Any],
        split: str = 'train',
        use_augmentation: bool = False,
        use_key: bool = False,
    ):
        self.data_root = data_root
        self.config = config
        self.n_beats = self.config['n_beats']
        self.split = split
        self.use_augmentation = use_augmentation
        self.use_key = use_key
        self.beat_resolution = self.config['beat_resolution']
        self.label_resolution = self.config['label_resolution']
        self.pr_to_label_ratio = self.beat_resolution // self.label_resolution

        self.vocabs = vocabs
        self.pad_idx = self.vocabs['pad_idx']

        self.chord_components = ['root', 'quality', 'bass']
        self.label_indices_map = {'root': 0, 'quality': 1, 'bass': 2}
        if self.use_key:
            self.chord_components.append('key')
            self.label_indices_map['key'] = 3

        # --- Lengths in pianoroll-frame resolution ---
        self.max_len = self.n_beats * self.beat_resolution

        for comp in self.chord_components:
            setattr(self, f'{comp}_vocab', self.vocabs[comp])
            setattr(self, f'{comp}_none_idx', self.vocabs[f'{comp}_none_idx'])

        suffix = 'shift0.npz' if not self.use_augmentation else '.npz'
        # print(f"Loading {suffix} files from {data_root}")
        self.file_list = sorted([
            os.path.join(data_root, f) 
            for f in os.listdir(data_root) if f.endswith(suffix)
        ])

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        filepath = self.file_list[idx]
        with np.load(filepath, allow_pickle=True) as data:
            pianoroll_full = torch.from_numpy(data['pianoroll'].T).float()
            labels_full = data['labels']
            boundaries_full = data['boundaries']
 
        pianoroll = pianoroll_full
        labels = labels_full

        # --- Create ground truth chord tensor from labels (map to per-component vocab indices) ---
        target_indices = {}

        for comp in self.chord_components:
            vocab = getattr(self, f'{comp}_vocab')
            none_idx = getattr(self, f'{comp}_none_idx')
            label_col_idx = self.label_indices_map[comp]
            col = labels[:, label_col_idx]
            mapped_tensor = None
            # If labels are already integer indices within range, accept directly
            try:
                if np.issubdtype(col.dtype, np.integer):
                    col_int = col.astype(np.int64)
                    if col_int.min(initial=0) >= 0 and col_int.max(initial=0) < len(vocab):
                        mapped_tensor = torch.from_numpy(col_int)
            except Exception:
                mapped_tensor = None
            # Otherwise map string/mixed labels through vocab with fallback to none_idx
            if mapped_tensor is None:
                try:
                    col_list = col.astype(str).tolist()
                except Exception:
                    col_list = [str(x) for x in col.tolist()]
                mapped = [vocab.get(lbl, none_idx) for lbl in col_list]
                mapped_tensor = torch.tensor(mapped, dtype=torch.long)
            target_indices[comp] = mapped_tensor.long()

        # --- Load pre-computed boundary flag ---
        boundary_flag = torch.from_numpy(boundaries_full.astype(np.float32))
        
        if self.split == 'train':
            return self._get_train_item(pianoroll, target_indices, boundary_flag)
        else: # 'val' or 'test'
            piece_name = _get_piece_name(filepath)
            # Build accurate targets from labels for evaluation
            return self._get_eval_item(pianoroll, labels, boundary_flag, piece_name)
    
    def _sample_stratified_start(self, X: int) -> int:
        """
        Sample s ∈ {0..X} with P(s) ∝ 1 + beta * (s/X).
        Implemented as a mixture of Uniform and 'linear-in-s' discrete law.
        Exact, O(1), numerically stable.

        beta ∈ [0,2]. beta=0 -> uniform; beta=1 -> mild late tilt (good default).
        """
        if X <= 0:
            return 0

        beta = float(getattr(self, 'stratify_beta', 1.0))

        # Mixture weights: P = a * Uniform + (1-a) * Linear(s)
        a = 1.0 - beta / 2.0  # ∈ [0,1]
        if np.random.rand() < a:
            # Uniform over 0..X
            return int(np.random.randint(0, X + 1))
        else:
            # Sample from Q(s) ∝ s over {0..X} (i.e., s=0 has weight 0).
            # Do it by inverting triangular numbers over 1..X.
            M = X * (X + 1) // 2  # sum_{s=1}^X s
            r = np.random.randint(1, M + 1)  # 1..M inclusive
            s = int((math.isqrt(1 + 8 * r) - 1) // 2)  # floor((sqrt(1+8r)-1)/2)
            # Numerical guard (rare when r hits exact triangle): clamp
            if s > X:
                s = X
            return s

    def _get_train_item(self, pianoroll, target_indices, boundary_flag):
        n_pr_frames = pianoroll.shape[0]
        # start with at least half of window size and convert to label frames
        max_start_label_frame = (n_pr_frames - self.max_len // 2) // self.pr_to_label_ratio
        target_max_len = self.max_len // self.pr_to_label_ratio
        
        # Stratified start over 0..max_start_label_frame (tilt to late positions)
        start_label_frame = self._sample_stratified_start(max_start_label_frame)
        start_pr_frame = start_label_frame * self.pr_to_label_ratio
        
        # --- slice & pad encoder input ---
        pr_segment = pianoroll[start_pr_frame : start_pr_frame + self.max_len]
        pr_pad_amount = self.max_len - pr_segment.shape[0]
        if pr_pad_amount > 0:
            # keep dtype/device consistent with pr_segment
            pr_pad = pr_segment.new_zeros((pr_pad_amount, pr_segment.shape[1]))
            pr_segment = torch.cat([pr_segment, pr_pad], dim=0)

        # --- slice targets at label resolution ---
        target_start = start_label_frame
        target_segs = {}
        for comp in self.chord_components:
            target_segs[comp] = target_indices[comp][target_start : target_start + target_max_len]
        boundary_seg = boundary_flag[target_start : target_start + target_max_len]

        # --- masks & padding for targets ---
        current_target_len = target_segs[self.chord_components[0]].shape[0]
        target_mask = torch.zeros(target_max_len, dtype=torch.bool)
        target_mask[:current_target_len] = True

        # expand target mask to encoder (frame) mask
        encoder_mask = target_mask.repeat_interleave(self.pr_to_label_ratio)
        if pr_pad_amount > 0:
            encoder_mask[-pr_pad_amount:] = False

        target_pad_amount = target_max_len - current_target_len
        if target_pad_amount > 0:
            for comp in self.chord_components:
                comp_none_idx = getattr(self, f'{comp}_none_idx')
                pad_tensor = torch.full((target_pad_amount,), comp_none_idx, dtype=torch.long)
                target_segs[comp] = torch.cat([target_segs[comp], pad_tensor])

            boundary_pad = torch.zeros(target_pad_amount, dtype=boundary_seg.dtype)
            boundary_seg = torch.cat([boundary_seg, boundary_pad])

        item = {
            'encoder_input': pr_segment,
            'target_boundary': boundary_seg,
            'mask': target_mask,
            'encoder_mask': encoder_mask,
        }
        for comp in self.chord_components:
            item[f'target_{comp}'] = target_segs[comp]

        return item

    def _get_eval_item(self, pianoroll, labels, boundary_flag, piece_name):
        # Reconstruct per-component target indices directly from the label matrix
        n_label_frames = labels.shape[0]
        target_indices = {}
        for comp in self.chord_components:
            vocab = getattr(self, f'{comp}_vocab')
            none_idx = getattr(self, f'{comp}_none_idx')
            label_col_idx = self.label_indices_map[comp]
            # Extract the column for this component; handle types robustly
            col = labels[:, label_col_idx]
            mapped_tensor = None
            # Case 1: already integer indices
            try:
                if np.issubdtype(col.dtype, np.integer):
                    col_int = col.astype(np.int64)
                    # If values look like valid indices, accept directly; otherwise fallback to mapping
                    if col_int.min(initial=0) >= 0 and col_int.max(initial=0) < len(vocab):
                        mapped_tensor = torch.from_numpy(col_int)
            except Exception:
                mapped_tensor = None
            # Case 2: map from labels (strings or mixed types) to indices
            if mapped_tensor is None:
                try:
                    col_list = col.astype(str).tolist()
                except Exception:
                    col_list = [str(x) for x in col.tolist()]
                mapped = [vocab.get(lbl, none_idx) for lbl in col_list]
                mapped_tensor = torch.tensor(mapped, dtype=torch.long)
            target_indices[comp] = mapped_tensor.long()

        mask = torch.ones(n_label_frames, dtype=torch.bool)
        encoder_mask = torch.ones(pianoroll.shape[0], dtype=torch.bool)
        item = {
            'piece_name': piece_name,
            'encoder_input': pianoroll,
            'target_boundary': boundary_flag,
            'mask': mask,
            'encoder_mask': encoder_mask,
        }
        for comp in self.chord_components:
            item[f'target_{comp}'] = target_indices[comp]
        return item

    def get_vocab_sizes(self) -> Dict[str, int]:
        sizes = {comp: len(self.vocabs[comp]) for comp in self.chord_components}
        return sizes

    def get_pad_idx(self) -> int:
        return self.pad_idx


def _get_piece_name(filename: str) -> str:
    """Extracts the base piece name from a filename by splitting on '_shift'."""
    base_filename = os.path.basename(filename)
    if '_shift' in base_filename:
        piece_name = base_filename.split('_shift')[0]
    else:
        piece_name = base_filename
    return piece_name


def create_datasets(
    data_root: str,
    config: dict,
    vocabs: Dict[str, Any],
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets with group-based splitting.
    This ensures that all augmentations of a piece belong to the same split.
    """
    full_dataset = PianoRollDataset(
        data_root=data_root,
        config=config,
        vocabs=vocabs,
        split='train',  # split does not matter here
        use_augmentation=config['use_augmentation'],
        use_key=config['use_key'],
    )

    # Group files by piece name
    piece_files = defaultdict(list)
    for f in full_dataset.file_list:
        piece_name = _get_piece_name(f)
        piece_files[piece_name].append(f)

    unique_pieces = sorted(list(piece_files.keys()))

    # Shuffle for random split
    random.seed(seed)
    random.shuffle(unique_pieces)

    # Split unique pieces (90% train, 10% validation)
    train_size = int(0.9 * len(unique_pieces))
    train_pieces = unique_pieces[:train_size]
    val_pieces = unique_pieces[train_size:]

    # Get file lists for each split, only use shift0.npz for validation
    train_files = [file for piece in train_pieces for file in piece_files[piece]]
    if config['use_augmentation']:
        val_files = [file for piece in val_pieces for file in piece_files[piece] if file.endswith('shift0.npz')]
    else:
        val_files = [file for piece in val_pieces for file in piece_files[piece]]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create datasets for each split with the correct file list
    train_dataset = PianoRollDataset(data_root, config, vocabs, 'train', use_key=config['use_key'])
    train_dataset.file_list = train_files

    val_dataset = PianoRollDataset(data_root, config, vocabs, 'val', use_key=config['use_key'])
    val_dataset.file_list = val_files

    json.dump(sorted([_get_piece_name(file) for file in val_files]), 
              open('val_files_unique.json', 'w'), indent=2)

    return train_dataset, val_dataset


def collate_fn(batch):
    """
    Collate function that filters out empty or invalid samples.
    For training, it uses default collate.
    For evaluation (variable length), it handles padding if needed, but typically used with batch_size=1.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    # If batch contains only a single sample, simply return that sample's dict.
    # This is handy for evaluation where we usually set batch_size = 1 and do
    # not need the extra list wrapper.
    if len(batch) == 1 and 'piece_name' in batch[0]:
        return batch[0]

    # For training batches (fixed-length segments) every sample has the same
    # sequence length, so the default PyTorch collate works fine.
    if 'encoder_input' in batch[0] and batch[0]['encoder_input'].shape[0] == batch[-1]['encoder_input'].shape[0]:
        return torch.utils.data.dataloader.default_collate(batch)

    # Otherwise we have variable-length sequences – fall back to returning the
    # list so the caller can deal with padding/iteration manually.
    return batch

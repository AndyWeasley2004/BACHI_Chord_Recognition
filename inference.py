import os
import sys
import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import multiprocessing

import yaml
import torch
import numpy as np
import miditoolkit
from music21 import converter, note as m21_note, chord as m21_chord
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import load_vocabs
from models.variants import build_model


def extract_pianoroll(score_path: Path, resolution: int = 12) -> Optional[np.ndarray]:
    """Extract piano roll from a music score file, excluding drum/percussion tracks.
    
    Args:
        score_path: Path to the music score file
        resolution: Frames per beat (default: 12)
    
    Returns:
        Piano roll array of shape (88, T) or None if extraction fails
    """
    try:
        notes_data = []  # (midi, onset_qb, offset_qb)
        
        suffix = score_path.suffix.lower()
        if suffix in {".mid", ".midi"}:
            # MIDI file processing
            midi = miditoolkit.MidiFile(str(score_path))
            tpb = midi.ticks_per_beat or 480
            for inst in midi.instruments:
                # Exclude drum tracks
                if inst.is_drum:
                    continue
                    
                for n in inst.notes:
                    start_qb = n.start / tpb
                    end_qb = n.end / tpb
                    notes_data.append((n.pitch, start_qb, end_qb))
        else:
            # MusicXML/MXL file processing
            sc = converter.parse(str(score_path))
            
            # Handle parts to detect instruments
            if sc.hasPartLikeStreams():
                parts = list(sc.parts)
            else:
                parts = [sc]
                
            for part in parts:
                # Check instrument for percussion
                inst = part.getInstrument()
                if inst:
                    # Check classes
                    classes = inst.classes if hasattr(inst, 'classes') else []
                    if 'Percussion' in classes or 'Unpitched' in classes:
                        continue
                    # Check name as fallback
                    name = inst.bestName if hasattr(inst, 'bestName') else ''
                    if name and ('drum' in name.lower() or 'percussion' in name.lower()):
                        continue
                
                # Extract notes from this part
                for el in part.flat.notes:
                    dur = float(el.quarterLength)
                    start_qb = float(el.offset)
                    end_qb = start_qb + dur
                    if isinstance(el, m21_note.Note):
                        notes_data.append((el.pitch.midi, start_qb, end_qb))
                    elif isinstance(el, m21_chord.Chord):
                        for p in el.pitches:
                            notes_data.append((p.midi, start_qb, end_qb))
        
        if not notes_data:
            return None
        
        # Sort notes by onset time
        notes_data.sort(key=lambda x: x[1])
        
        # Calculate total duration from last note
        last_note_end = max(nd[2] for nd in notes_data)
        total_frames = math.ceil(last_note_end * resolution)
        
        # Create pianoroll (binary 88 × T)
        pianoroll = np.zeros((88, total_frames), dtype=np.int8)
        for midi_pitch, start_b, end_b in notes_data:
            row = midi_pitch - 21  # A0 == 21
            if not (0 <= row < 88):
                continue
            s_f = max(0, math.floor(start_b * resolution))
            e_f = min(total_frames, math.ceil(end_b * resolution))
            pianoroll[row, s_f:e_f] = 1
        
        return pianoroll
    
    except Exception as e:
        # We print error here but returning None is handled by caller
        print(f"Error extracting pianoroll from {score_path.name}: {e}")
        return None


class MusicScoreDataset(Dataset):
    """Dataset for parallel loading and extraction of pianorolls."""
    def __init__(self, file_paths: List[Path], resolution: int):
        self.file_paths = file_paths
        self.resolution = resolution

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # Returns (path_str, pianoroll_or_None)
        # We return path as string because Path objects pickle fine but string is safer/simpler across processes
        pr = extract_pianoroll(path, self.resolution)
        return str(path), pr


def collate_scores(batch):
    """Simple collate function that returns the list of items."""
    return batch


def predict_piece(
    pianoroll: torch.Tensor,
    model,
    config: Dict[str, Any],
    vocabs: Dict[str, Any],
    device: torch.device,
    use_key: bool,
) -> Optional[str]:
    """Run inference on a single pianoroll tensor and return prediction string."""
    
    n_frames = pianoroll.shape[0]
    
    # Inference parameters
    beat_resolution = config['model']['beat_resolution']
    label_resolution = config['model']['label_resolution']
    segment_len = config['model']['n_beats'] * beat_resolution
    pr_to_label_ratio = beat_resolution // label_resolution
    
    comps_eval = ['root', 'quality', 'bass'] + (['key'] if use_key else [])
    
    # Segment the piece
    segments, masks = [], []
    for i in range(0, n_frames, segment_len):
        end_i = i + segment_len
        segment = pianoroll[i:end_i, :]
        orig_len = segment.shape[0]
        if orig_len < segment_len:
            pad_amount = segment_len - orig_len
            segment = torch.cat([segment, torch.zeros(pad_amount, segment.shape[1])], dim=0)
        encoder_mask_seg = torch.ones(segment_len, dtype=torch.bool)
        if orig_len < segment_len:
            encoder_mask_seg[orig_len:] = False
        segments.append(segment)
        masks.append(encoder_mask_seg)
    
    # Run inference on each segment
    piece_preds = {k: [] for k in comps_eval + ['boundary']}
    
    # Batch segments for GPU efficiency?
    # The original code processed segments one by one in a loop, but added unsqueeze(0).
    # We can stack them if there are many segments, but for now let's keep logic simple
    # or stack them into a batch.
    
    # Stack segments to run in one go (or batches of segments)
    # Depending on piece length, this could be large. Let's use a batch size of 16 segments.
    segment_batch_size = 16
    
    for i in range(0, len(segments), segment_batch_size):
        batch_segs = torch.stack(segments[i : i + segment_batch_size]).to(device)
        batch_masks = torch.stack(masks[i : i + segment_batch_size]).to(device)
        
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            out = model.forward_infer(batch_segs, src_key_padding_mask=~batch_masks)
            
        for k in comps_eval + ['boundary']:
            if k in out:
                # Move to CPU immediately
                piece_preds[k].append(out[k].detach().cpu())

    # Aggregate predictions
    n_target_frames = int(math.ceil(n_frames / pr_to_label_ratio))
    piece_pred_ids = {}
    for k, parts_list in piece_preds.items():
        if not parts_list:
            continue
        # parts_list is a list of tensors (B, Len) or (B, Len, ...)
        # Flatten the batch dimension
        cat = torch.cat([p.reshape(-1) for p in parts_list], dim=0)
        if k == 'boundary':
            piece_pred_ids[k] = cat[:n_target_frames]
        else:
            piece_pred_ids[k] = cat[:n_target_frames].long()
    
    # Build inverse vocabs for pretty printing
    inv_root = {v: k for k, v in vocabs['root'].items()}
    inv_qual = {v: k for k, v in vocabs['quality'].items()}
    inv_bass = {v: k for k, v in vocabs['bass'].items()}
    
    # Extract valid predictions
    valid_len = len(piece_pred_ids.get('root', []))
    if valid_len == 0:
        return None
    
    r_seq = piece_pred_ids['root'][:valid_len].tolist()
    q_seq = piece_pred_ids['quality'][:valid_len].tolist()
    b_seq = piece_pred_ids['bass'][:valid_len].tolist()
    
    # Merge identical consecutive chords
    lr = config['model']['label_resolution']
    time_per_token = 1.0 / max(1, lr)
    merged_lines = []
    
    if valid_len > 0:
        cur_r, cur_q, cur_b = r_seq[0], q_seq[0], b_seq[0]
        cur_start = 0
        for t in range(1, valid_len):
            if (r_seq[t] != cur_r) or (q_seq[t] != cur_q) or (b_seq[t] != cur_b):
                start_beat = cur_start * time_per_token
                label = f"{inv_root.get(cur_r, str(cur_r))}_{inv_qual.get(cur_q, str(cur_q))}_{inv_bass.get(cur_b, str(cur_b))}"
                merged_lines.append((start_beat, label))
                cur_r, cur_q, cur_b = r_seq[t], q_seq[t], b_seq[t]
                cur_start = t
        # Flush last segment
        start_beat = cur_start * time_per_token
        label = f"{inv_root.get(cur_r, str(cur_r))}_{inv_qual.get(cur_q, str(cur_q))}_{inv_bass.get(cur_b, str(cur_b))}"
        merged_lines.append((start_beat, label))
    
    # Format output
    output_lines = []
    for start_beat, label in merged_lines:
        output_lines.append(f"{start_beat:.2f} {label}")
    
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description='BACHI Chord Recognition Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input music score file or directory of score files (Pitch range: 21-108)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory for predictions')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory containing best_model.pt, config.yaml, and vocab')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading (default: auto-detect)')
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist.")
        sys.exit(1)

    # Load config from checkpoint directory
    config_path = checkpoint_dir / 'config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at '{config_path}'.")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabs
    vocab_path = os.path.join(checkpoint_dir, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' not found.")
        sys.exit(1)
    
    vocabs = load_vocabs(str(vocab_path))
    
    # Determine experiment and key usage
    experiment = config['experiment']
    use_key = bool(config.get('use_key', False)) or bool(config['training'].get('use_key', False)) or bool(config['model'].get('use_key', False))

    # Build and load model
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'.")
        sys.exit(1)
    
    print(f"Loading model: {experiment}")
    model = build_model(experiment, config['model'], vocabs, use_key=use_key).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded successfully (params: {sum(p.numel() for p in model.parameters())})")
    
    # Collect input files
    supported_extensions = {'.musicxml', '.mxl', '.xml', '.mid', '.midi'}
    
    if input_path.is_file():
        if input_path.suffix.lower() not in supported_extensions:
            print(f"Error: Unsupported file format '{input_path.suffix}'. Supported formats: {supported_extensions}")
            sys.exit(1)
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        if not input_files:
            print(f"Error: No supported music files found in '{input_path}'.")
            sys.exit(1)
        input_files = sorted(input_files)
    else:
        print(f"Error: Input path '{input_path}' is neither a file nor a directory.")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup DataLoader
    num_workers = args.num_workers if args.num_workers is not None else min(8, multiprocessing.cpu_count())
    print(f"Processing {len(input_files)} file(s) using {num_workers} workers...")
    
    dataset = MusicScoreDataset(input_files, config['model']['beat_resolution'])
    loader = DataLoader(
        dataset, 
        batch_size=4,  # Load 4 files at a time (collated as list)
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_scores
    )
    
    successful = 0
    failed = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            for path_str, pianoroll_np in batch:
                score_file = Path(path_str)
                
                if pianoroll_np is None:
                    print(f"\nWarning: Failed to extract pianoroll from '{score_file.name}'")
                    failed += 1
                    continue
                
                try:
                    # Convert to tensor
                    pianoroll = torch.from_numpy(pianoroll_np.T).float()
                    
                    # Run inference
                    prediction_text = predict_piece(
                        pianoroll, model, config, vocabs, device, use_key
                    )
                    
                    if prediction_text is None:
                        print(f"\nWarning: Failed to predict for '{score_file.name}' (empty result)")
                        failed += 1
                        continue
                    
                    # Write output
                    output_file = output_path / f"{score_file.stem}.txt"
                    with open(output_file, 'w') as f:
                        f.write(prediction_text + '\n')
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"\nError processing '{score_file.name}': {e}")
                    failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"Successfully processed: {successful}/{len(input_files)} files")
    if failed > 0:
        print(f"Failed: {failed}/{len(input_files)} files")
    print(f"Predictions saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    main()

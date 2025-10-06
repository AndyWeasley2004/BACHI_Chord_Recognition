import os
import sys
import math
from typing import Dict, Any

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from dataset import create_datasets, load_vocabs, collate_fn
from models.variants import build_model


def compute_transition_errors(pred_indices: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], chord_components, pad_idx: int) -> Dict[str, int]:
    # T alignment
    T = targets['boundary'].shape[0]
    chord_mask = targets['root'] != pad_idx
    valid_pair = torch.zeros(T, dtype=torch.bool, device=targets['root'].device)
    if T > 1:
        valid_pair[1:] = chord_mask[1:] & chord_mask[:-1]
    true_boundary = (targets['boundary'][:T] > 0).bool() & valid_pair

    pred_change = torch.zeros(T, dtype=torch.bool, device=true_boundary.device)
    if T > 1:
        for comp in chord_components:
            comp_change = torch.zeros(T, dtype=torch.bool, device=true_boundary.device)
            comp_change[1:] = pred_indices[comp][1:T] != pred_indices[comp][0:T-1]
            pred_change |= comp_change
    pred_change &= valid_pair

    correct_chord = torch.ones(T, dtype=torch.bool, device=true_boundary.device)
    for comp in chord_components:
        correct_chord &= (pred_indices[comp] == targets[comp][:T])

    nonboundary_mask = (~true_boundary) & valid_pair
    false_alarm = (pred_change & (~true_boundary)).sum().item()
    missed = (true_boundary & (~pred_change)).sum().item()
    confusion = (true_boundary & pred_change & (~correct_chord)).sum().item()

    return {
        'false_alarm': false_alarm,
        'missed': missed,
        'confusion': confusion,
        'nb_denom': nonboundary_mask.sum().item(),
        'tb_denom': true_boundary.sum().item(),
    }


def main():
    checkpoint_dir = sys.argv[1]
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    model_path = os.path.join(checkpoint_dir, 'best_model.pt')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(config['training']['device'])
    torch.manual_seed(config['training']['seed'])
    import numpy as np
    np.random.seed(config['training']['seed'])

    experiment = config['experiment']
    use_key = bool(config.get('use_key', False)) or bool(config['training'].get('use_key', False)) or bool(config['model'].get('use_key', False))
    chord_components = ['root','quality','bass'] + (['key'] if use_key else [])
    comps_eval = ['root','quality','bass'] + (['key'] if use_key else [])

    vocabs = load_vocabs(config['training']['vocab_path'])
    pad_idx = vocabs['pad_idx']

    dataset_config = config['model']
    dataset_config = {**config['model']}
    dataset_config['use_key'] = use_key
    dataset_config['use_augmentation'] = dataset_config.get('use_augmentation', True)
    _, val_dataset = create_datasets(
        config['training']['data_root'], dataset_config, vocabs, config['training']['seed']
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=config['training']['num_workers'], pin_memory=True)

    model = build_model(experiment, config['model'], vocabs, use_key=use_key).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Metric accumulators (micro)
    total_correct = {comp: 0 for comp in chord_components + ['chord']}
    total_tokens_per_comp = {comp: 0 for comp in chord_components}
    total_chord_tokens = 0
    total_error_counts = {'false_alarm': 0, 'missed': 0, 'confusion': 0}
    total_error_denoms = {'false_alarm': 0, 'missed': 0, 'confusion': 0}
    # Macro accumulators (per piece)
    per_piece_acc = []
    per_piece_decode_orders = [] if experiment == 'film_kdec' else None

    beat_resolution = config['model']['beat_resolution']
    segment_len = config['model']['n_beats'] * beat_resolution
    pr_to_label_ratio = config['model']['beat_resolution'] // config['model']['label_resolution']

    # Prepare inverse vocabs for pretty printing
    inv_root = {v: k for k, v in vocabs['root'].items()}
    inv_qual = {v: k for k, v in vocabs['quality'].items()}
    inv_bass = {v: k for k, v in vocabs['bass'].items()}
    # Output directory for per-piece predictions
    pred_dir = os.path.join(checkpoint_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for piece_batch in tqdm(val_loader, desc='Evaluating'):
            piece = piece_batch if isinstance(piece_batch, dict) else piece_batch[0]
            pianoroll = piece['encoder_input']
            n_frames = pianoroll.shape[0]

            segments, masks, target_segments = [], [], {k: [] for k in comps_eval + ['boundary']}
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

                target_start = i // pr_to_label_ratio
                target_end = int(math.ceil(end_i / pr_to_label_ratio))
                for key in comps_eval + ['boundary']:
                    target = piece[f'target_{key}']
                    seg_len = segment_len // pr_to_label_ratio
                    target_segment = target[target_start:target_end]
                    if target_segment.shape[0] < seg_len:
                        if key == 'boundary':
                            pad_val = 0.
                            dtype = torch.float
                        else:
                            comp_pad = vocabs.get(f"{key}_pad_idx", pad_idx)
                            pad_val = comp_pad
                            dtype = torch.long
                        pad_amount = seg_len - target_segment.shape[0]
                        target_pad = torch.full((pad_amount,), pad_val, dtype=dtype)
                        target_segment = torch.cat([target_segment, target_pad])
                    target_segments[key].append(target_segment)

            piece_preds = {k: [] for k in comps_eval + ['boundary']}
            decode_orders_piece = [] if experiment == 'film_kdec' else None
            for i in range(0, len(segments)):
                segment_batch = segments[i].unsqueeze(0).to(device)
                mask_batch = masks[i].unsqueeze(0).to(device)
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                    out = model.forward_infer(segment_batch, src_key_padding_mask=~mask_batch)
                for k in comps_eval + ['boundary']:
                    if k in out:
                        piece_preds[k].append(out[k].detach().cpu())
                if experiment == 'film_kdec' and ('decode_order' in out):
                    # out['decode_order']: (B=1, T, 3) with slots in {0,1,2} for [q,r,b]
                    decode_orders_piece.append(out['decode_order'].detach().cpu())

            n_target_frames = int(math.ceil(n_frames / pr_to_label_ratio))
            piece_pred_ids = {}
            for k, parts in piece_preds.items():
                if not parts:
                    continue
                cat = torch.cat([p.reshape(-1) for p in parts], dim=0)
                if k == 'boundary':
                    piece_pred_ids[k] = cat[:n_target_frames].to(device)
                else:
                    piece_pred_ids[k] = cat[:n_target_frames].to(device).long()
            # Aggregate decode orders per piece
            if experiment == 'film_kdec' and decode_orders_piece:
                do_cat = torch.cat([d.squeeze(0) for d in decode_orders_piece], dim=0)  # (T_piece, 3)
                if per_piece_decode_orders is not None:
                    per_piece_decode_orders.append(do_cat)

            piece_targets = {k: torch.cat(target_segments[k])[:n_target_frames].to(device) for k in target_segments}
            chord_mask = piece_targets['root'] != pad_idx
            n_chord_tokens_piece = chord_mask.sum().item()
            total_chord_tokens += n_chord_tokens_piece

            # Component accuracies (match train.py): core comps for chord; count 'key' only if predicted
            comps_eval = ['root','quality','bass'] + (['key'] if use_key else [])
            core_comps = ['root','quality','bass']
            if n_chord_tokens_piece > 0:
                correct_chord = torch.ones_like(chord_mask)
                for comp in core_comps:
                    comp_pad = vocabs.get(f"{comp}_pad_idx", pad_idx)
                    comp_mask = piece_targets[comp] != comp_pad
                    total_tokens_per_comp[comp] += comp_mask.sum().item()
                    total_correct[comp] += (piece_pred_ids[comp][comp_mask] == piece_targets[comp][comp_mask]).sum().item()
                    correct_chord &= (piece_pred_ids[comp] == piece_targets[comp])
                if use_key and ('key' in piece_pred_ids):
                    comp = 'key'
                    comp_pad = vocabs.get(f"{comp}_pad_idx", pad_idx)
                    comp_mask = piece_targets[comp] != comp_pad
                    total_tokens_per_comp[comp] += comp_mask.sum().item()
                    total_correct[comp] += (piece_pred_ids[comp][comp_mask] == piece_targets[comp][comp_mask]).sum().item()
                total_correct['chord'] += correct_chord[chord_mask].sum().item()

                # Per-piece macro stats
                piece_stats = {k: 0.0 for k in comps_eval + ['chord']}
                for comp in comps_eval:
                    comp_pad = vocabs.get(f"{comp}_pad_idx", pad_idx)
                    comp_mask = piece_targets[comp] != comp_pad
                    if comp in piece_pred_ids and comp_mask.sum() > 0:
                        piece_stats[comp] = (piece_pred_ids[comp][comp_mask] == piece_targets[comp][comp_mask]).float().mean().item()
                    else:
                        piece_stats[comp] = 0.0
                piece_stats['chord'] = correct_chord[chord_mask].float().mean().item() if chord_mask.sum() > 0 else 0.0
                per_piece_acc.append(piece_stats)

                # --- Write per-piece predictions file ---
                try:
                    piece_name = piece.get('piece_name', 'unknown')
                except Exception:
                    piece_name = 'unknown'
                base_name = os.path.splitext(piece_name)[0]
                out_path_piece = os.path.join(pred_dir, f"{base_name}.txt")

                # Compute per-piece boundary acc
                boundary_acc = 0.0
                if 'boundary' in piece_pred_ids:
                    pred_b = (torch.sigmoid(piece_pred_ids['boundary']) > 0.5)
                    boundary_acc = (pred_b[:chord_mask.shape[0]][chord_mask] == piece_targets['boundary'][:chord_mask.shape[0]][chord_mask]).float().mean().item() if chord_mask.sum() > 0 else 0.0

                # Optional key accuracy
                key_acc = None
                if ('key' in piece_pred_ids) and ('key' in piece_targets):
                    comp = 'key'
                    comp_pad = vocabs.get(f"{comp}_pad_idx", pad_idx)
                    comp_mask = piece_targets[comp] != comp_pad
                    if comp_mask.sum() > 0:
                        key_acc = (piece_pred_ids[comp][comp_mask] == piece_targets[comp][comp_mask]).float().mean().item()
                    else:
                        key_acc = 0.0

                # Build merged chord timeline from predictions, only valid (non-pad) tokens
                valid_len = int(chord_mask.sum().item())
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
                    # flush last segment
                    start_beat = cur_start * time_per_token
                    label = f"{inv_root.get(cur_r, str(cur_r))}_{inv_qual.get(cur_q, str(cur_q))}_{inv_bass.get(cur_b, str(cur_b))}"
                    merged_lines.append((start_beat, label))

                # Write file
                with open(out_path_piece, 'w') as pf:
                    # Header
                    hdr_parts = [
                        f"Root: {piece_stats['root']:.4f}",
                        f"Quality: {piece_stats['quality']:.4f}",
                        f"Bass: {piece_stats['bass']:.4f}",
                        f"Boundary: {boundary_acc:.4f}",
                        f"Chord: {piece_stats['chord']:.4f}",
                    ]
                    if key_acc is not None:
                        hdr_parts.insert(3, f"Key: {key_acc:.4f}")
                    pf.write(" | ".join(hdr_parts) + "\n")
                    # Body
                    for start_beat, label in merged_lines:
                        pf.write(f"{start_beat:.2f} {label}\n")

            # Transition metrics
            # Guard when 'key' might not be present in outputs
            comps_present = [c for c in chord_components if c in piece_pred_ids]
            errs = compute_transition_errors(piece_pred_ids, piece_targets, comps_present, pad_idx)
            total_error_counts['false_alarm'] += errs['false_alarm']
            total_error_denoms['false_alarm'] += errs['nb_denom']
            total_error_counts['missed'] += errs['missed']
            total_error_denoms['missed'] += errs['tb_denom']
            total_error_counts['confusion'] += errs['confusion']
            total_error_denoms['confusion'] += errs['tb_denom']

            # For FiLM+KToken model, optionally apply pre-trained boundary model constraint to suppress false alarms
            if experiment == 'film_kdec' and config['training'].get('boundary_checkpoint_dir'):
                # Load boundary logits via boundary model for this piece and re-evaluate transition metrics with suppression.
                # To keep this script simple per request, we assume boundary logits are provided externally.
                pass

    avg_acc = {comp: (total_correct[comp] / max(1, total_tokens_per_comp[comp])) for comp in chord_components}
    avg_acc['chord'] = total_correct['chord'] / max(1, total_chord_tokens)

    # Macro-average across pieces
    if per_piece_acc:
        macro_acc = {k: sum(p[k] for p in per_piece_acc) / len(per_piece_acc) for k in per_piece_acc[0].keys()}
    else:
        macro_acc = {k: 0.0 for k in chord_components + ['chord']}

    error_token = {}
    for k in ['false_alarm', 'missed', 'confusion']:
        denom = total_error_denoms[k]
        error_token[k] = (total_error_counts[k] / denom) if denom > 0 else 0.0

    # Decode order frequencies for film_kdec
    decode_order_stats = None
    if experiment == 'film_kdec' and per_piece_decode_orders:
        # Map slot index to name in fixed order [quality, root, bass]
        slot_names = ['quality', 'root', 'bass']
        # Count most frequent order strings (e.g., root->bass->quality)
        from collections import Counter
        counter = Counter()
        first_counter = Counter()
        total_steps = 0
        total_first = 0
        for do in per_piece_decode_orders:
            # do: (T, 3) with ints in {0,1,2}
            orders = [tuple(row.tolist()) for row in do]
            counter.update(orders)
            first_counter.update([row[0].item() for row in do])
            total_steps += len(orders)
            total_first += len(orders)
        # Build readable frequencies
        def order_to_str(order_tuple):
            return '->'.join(slot_names[idx] for idx in order_tuple)
        top_orders = counter.most_common(5)
        order_freq = [(order_to_str(o), c / max(1, total_steps)) for o, c in top_orders]
        first_freq = {slot_names[k]: v / max(1, total_first) for k, v in first_counter.items()}
        decode_order_stats = {
            'top_orders': order_freq,
            'first_slot_freq': first_freq,
        }

    out_path = os.path.join(checkpoint_dir, 'evaluation.txt')
    with open(out_path, 'w') as f:
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write("--- Evaluation Results ---\n")
        for k in ['root','quality','bass','chord']:
            f.write(f"{k.capitalize()} Accuracy: {avg_acc[k]:.4f}\n")
        f.write("--- Macro (per piece) ---\n")
        for k in ['root','quality','bass','chord']:
            f.write(f"{k.capitalize()} Macro-Acc: {macro_acc[k]:.4f}\n")
        for name, disp in [('false_alarm','False Alarm'),('missed','Missed'),('confusion','Confusion')]:
            f.write(f"{disp}: {error_token[name]:.4f}\n")
        if decode_order_stats is not None:
            f.write("--- Decode Order (film_kdec) ---\n")
            for s, p in decode_order_stats['top_orders']:
                f.write(f"Order {s}: {p*100:.2f}%\n")
            f.write("First-slot frequencies:\n")
            for name, p in decode_order_stats['first_slot_freq'].items():
                f.write(f"  {name}: {p*100:.2f}%\n")
    print(f"Saved metrics to {out_path}")


if __name__ == '__main__':
    main()



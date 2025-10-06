import os
import sys
import argparse
import math
import shutil
from typing import Dict, Any

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import create_datasets, load_vocabs, collate_fn
from models.variants import build_model


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
    initial_lr = optimizer.param_groups[0]['lr']
    min_lr_ratio = 0.0 if initial_lr == 0 else (min_lr / initial_lr)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_progress = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_progress

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pad_idx: int) -> float:
    mask = mask & (targets != pad_idx)
    if mask.sum() == 0:
        return 0.0
    preds = torch.argmax(logits, dim=-1)
    correct = (preds[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def train_epoch(model, dataloader, optimizer, scheduler, device, pad_idx, scaler, chord_components, vocabs):
    model.train()
    total_loss = 0.0
    total_acc = {comp: 0.0 for comp in chord_components}
    total_acc.update({'boundary': 0.0, 'chord': 0.0})
    num_batches = len(dataloader)

    for batch in dataloader:
        if not batch:
            continue
        optimizer.zero_grad()

        encoder_input = batch['encoder_input'].to(device)
        target_mask = batch['mask'].to(device)
        encoder_mask = batch.get('encoder_mask', target_mask).to(device)
        targets = {comp: batch[f'target_{comp}'].to(device) for comp in chord_components}
        targets['boundary'] = batch['target_boundary'].to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            out = model.forward_train(
                encoder_input,
                targets,
                src_key_padding_mask=~encoder_mask,
                target_mask=target_mask,
                vocabs=vocabs,
            )
            loss = out['loss']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss.item())

        logits = out.get('logits', {})
        for comp in chord_components:
            if comp in logits:
                total_acc[comp] += calculate_accuracy(logits[comp], targets[comp], target_mask, pad_idx)

        if set(['root','quality','bass']).issubset(logits.keys()):
            component_preds = {comp: torch.argmax(logits[comp], dim=-1) for comp in ['root','quality','bass']}
            correct_chord = torch.ones_like(target_mask, dtype=torch.bool)
            for comp in ['root','quality','bass']:
                correct_chord &= (component_preds[comp] == targets[comp])
            valid_mask = target_mask & (targets['root'] != pad_idx)
            total_acc['chord'] += (correct_chord[valid_mask].sum().item() / valid_mask.sum().item()) if valid_mask.sum() > 0 else 0.0

        if 'boundary_logits' in out:
            preds_b = (torch.sigmoid(out['boundary_logits']) > 0.5).float()
            correct = (preds_b[target_mask] == targets['boundary'][target_mask]).sum().item()
            total = target_mask.sum().item()
            total_acc['boundary'] += (correct / total) if total > 0 else 0.0

    avg_loss = total_loss / max(1, num_batches)
    avg_acc = {k: v / max(1, num_batches) for k, v in total_acc.items()}
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Train chord recognition models')
    parser.add_argument('config_path', help='Path to YAML config')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(config['training']['device'])
    torch.manual_seed(config['training']['seed'])
    import numpy as np
    np.random.seed(config['training']['seed'])

    experiment = config['experiment']  # 'baseline' | 'film_ctx' | 'film_kdec' | 'film_kdec_key' | 'ht'
    use_key = bool(config.get('use_key', False)) or bool(config['training'].get('use_key', False)) or bool(config['model'].get('use_key', False))
    chord_components = ['root','quality','bass'] + (['key'] if use_key else [])

    vocabs = load_vocabs(config['training']['vocab_path'])
    pad_idx = vocabs['pad_idx']

    dataset_config = {**config['model']}
    dataset_config['use_key'] = use_key
    dataset_config['use_augmentation'] = dataset_config.get('use_augmentation', True)
    train_dataset, val_dataset = create_datasets(
        config['training']['data_root'],
        dataset_config,
        vocabs,
        config['training']['seed'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )

    model = build_model(experiment, config['model'], vocabs, use_key=use_key).to(device)
    print(f"Using model: {experiment} | Params: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config['training']['weight_decay'],
    )

    scheduler_cfg = config['training']['scheduler']
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_cfg['warmup_steps'],
        num_training_steps=num_training_steps,
        min_lr=config['training']['min_learning_rate'],
    )
    scaler = GradScaler(enabled=True)

    # Derive data_name from CLI or data_root, and save under checkpoints/{data_name}_{experiment}
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_dir, 'config.yaml'))
    log_file_path = os.path.join(save_dir, 'log.txt')
    best_val_acc = 0.0

    for epoch in range(config['training']['num_epochs']):
        log_lines = []
        log_lines.append(f"--- Epoch {epoch+1}/{config['training']['num_epochs']} ---\n")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, pad_idx, scaler, chord_components, vocabs)
        log_lines.append(f"Train Loss: {train_loss:.4f}\n")
        train_acc_str = " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in train_acc.items()])
        log_lines.append(f"Train Acc -> {train_acc_str} | LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Simple val run with loss proxy (boundary loss) and accuracies
        beat_resolution = config['model']['beat_resolution']
        segment_len = config['model']['n_beats'] * beat_resolution
        eval_batch_size = config['training']['batch_size']

        # Inline quick evaluation similar to old evaluate() without plots
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        comps_eval = ['root','quality','bass'] + (['key'] if use_key else [])
        total_correct = {k: 0 for k in comps_eval + ['boundary','chord']}
        total_tokens_per_comp = {k: 0 for k in comps_eval}
        total_chord_tokens = 0
        with torch.no_grad():
            for piece_batch in val_loader:
                if not piece_batch:
                    continue
                piece = piece_batch if isinstance(piece_batch, dict) else piece_batch[0]
                pianoroll = piece['encoder_input']
                n_frames = pianoroll.shape[0]
                pr_to_label_ratio = config['model']['beat_resolution'] // config['model']['label_resolution']

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
                for i in range(0, len(segments), eval_batch_size):
                    segment_batch = torch.stack(segments[i:i+eval_batch_size]).to(device)
                    mask_batch = torch.stack(masks[i:i+eval_batch_size]).to(device)
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                        out = model.forward_infer(segment_batch, src_key_padding_mask=~mask_batch)
                    for k in comps_eval + ['boundary']:
                        if k in out:
                            piece_preds[k].append(out[k].detach().cpu())

                n_target_frames = int(math.ceil(n_frames / pr_to_label_ratio))
                piece_preds_flat = {}
                for key, parts in piece_preds.items():
                    if not parts:
                        continue
                    if key == 'boundary':
                        cat = torch.cat([p.reshape(-1) for p in parts], dim=0)
                        piece_preds_flat[key] = cat[:n_target_frames].to(device)
                    else:
                        cat = torch.cat([p.reshape(-1) for p in parts], dim=0)
                        piece_preds_flat[key] = cat[:n_target_frames].long().to(device)

                piece_targets = {key: torch.cat(target_segments[key])[:n_target_frames].to(device) for key in target_segments}
                chord_mask = piece_targets['root'] != pad_idx
                n_chord_tokens_piece = chord_mask.sum().item()
                total_chord_tokens += n_chord_tokens_piece
                total_boundary_tokens = piece_targets['boundary'].shape[0]

                # boundary-only loss proxy
                bce = F.binary_cross_entropy_with_logits(piece_preds_flat['boundary'], piece_targets['boundary'].float(), reduction='sum')
                total_loss += float(bce.item())
                total_tokens += total_boundary_tokens

                # accuracies
                comps_present = [comp for comp in comps_eval if comp in piece_preds_flat]
                pred_indices = {comp: piece_preds_flat[comp] for comp in comps_present}
                pred_indices['boundary'] = (torch.sigmoid(piece_preds_flat['boundary']) > 0.5).long()

                if n_chord_tokens_piece > 0:
                    correct_chord = torch.ones_like(chord_mask)
                    for comp in ['root','quality','bass']:
                        comp_pad = vocabs.get(f"{comp}_pad_idx", pad_idx)
                        comp_mask = piece_targets[comp] != comp_pad
                        total_tokens_per_comp[comp] += comp_mask.sum().item()
                        total_correct[comp] += (pred_indices[comp][comp_mask] == piece_targets[comp][comp_mask]).sum().item()
                        correct_chord &= (pred_indices[comp] == piece_targets[comp])
                    # key accuracy (auxiliary)
                    if use_key and ('key' in pred_indices):
                        comp = 'key'
                        comp_pad = vocabs.get(f"{comp}_pad_idx", pad_idx)
                        comp_mask = piece_targets[comp] != comp_pad
                        total_tokens_per_comp[comp] += comp_mask.sum().item()
                        total_correct[comp] += (pred_indices[comp][comp_mask] == piece_targets[comp][comp_mask]).sum().item()
                    total_correct['chord'] += correct_chord[chord_mask].sum().item()

                total_correct['boundary'] += (pred_indices['boundary'] == piece_targets['boundary']).sum().item()

        val_loss = total_loss / max(1, total_tokens)
        val_acc = {comp: (total_correct[comp] / max(1, total_tokens_per_comp[comp])) for comp in comps_eval}
        val_acc['chord'] = total_correct['chord'] / max(1, total_chord_tokens)
        val_acc['boundary'] = total_correct['boundary'] / max(1, total_tokens)

        log_lines.append(f"Validation Loss: {val_loss:.4f}\n")
        val_acc_str = " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in val_acc.items()])
        log_lines.append(f"Validation Acc -> {val_acc_str}\n")

        if val_acc['chord'] > best_val_acc:
            best_val_acc = val_acc['chord']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            log_lines.append(f"Saved new best model with validation chord accuracy: {best_val_acc:.4f}\n")

        log_str = ''.join(log_lines)
        print(log_str, end='')
        with open(log_file_path, 'a') as f:
            f.write(log_str)


if __name__ == '__main__':
    main()



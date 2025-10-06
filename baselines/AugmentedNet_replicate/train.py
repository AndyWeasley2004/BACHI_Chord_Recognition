import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import AugmentedNetPT


class PackedDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        self.X_list = X_list
        self.y_list = y_list
        self.length = X_list[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        Xs = [torch.tensor(x[idx], dtype=torch.float32) for x in self.X_list]
        ys = []
        for y in self.y_list:
            t = torch.tensor(y[idx], dtype=torch.long)
            if t.ndim == 3 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            ys.append(t)
        return Xs, ys


def load_npz(npz_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[str]]:
    data = np.load(npz_path, allow_pickle=True)
    X_train, y_train, X_val, y_val = [], [], [], []
    output_dims = []
    task_names = []
    for name in sorted(data.files):
        arr = data[name]
        if name.startswith("training_X_"):
            X_train.append(arr)
        elif name.startswith("training_y_"):
            y_train.append(arr)
            task = name.split("_")[-1]
            task_names.append(task)
            max_idx = int(arr.max())
            val_name = name.replace("training_y_", "validation_y_")
            if val_name in data:
                max_idx = max(max_idx, int(data[val_name].max()))
            output_dims.append(max_idx + 1)
        elif name.startswith("validation_X_"):
            X_val.append(arr)
        elif name.startswith("validation_y_"):
            y_val.append(arr)
    return X_train, y_train, X_val, y_val, output_dims, task_names


def compute_monitored_metrics(logits_list: List[torch.Tensor], targets_list: List[torch.Tensor], task_names: List[str]):
    with torch.no_grad():
        accs = []
        losses = []
        ce = nn.CrossEntropyLoss(reduction="mean")
        for logit, target, name in zip(logits_list, targets_list, task_names):
            # Flatten to (N, C) and (N,)
            flat_logits = logit.reshape(-1, logit.shape[-1])
            flat_targets = target.reshape(-1)
            loss = ce(flat_logits, flat_targets)
            pred = logit.argmax(dim=-1)
            # Mask accuracy by ignore_index (-100)
            mask = (target != -100)
            denom = mask.float().sum().item()
            if denom > 0:
                acc = (pred[mask] == target[mask]).float().mean().item()
            else:
                acc = 0.0
            if name not in ["ChordQuality11", "ChordRoot35", "Inversion4", "PrimaryDegree22", "SecondaryDegree22"]:
                accs.append(acc)
                losses.append(loss.item())
        monitored_acc = float(np.mean(accs)) if accs else 0.0
        monitored_loss = float(np.sum(losses)) if losses else 0.0
    return monitored_acc, monitored_loss


def train_loop(model: AugmentedNetPT, train_loader: DataLoader, val_loader: DataLoader, task_names: List[str], epochs: int, device: str, lr_boundaries: List[int], lr_values: List[float], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr_values[0])
    ce = nn.CrossEntropyLoss()
    best_score = -1.0
    best_path = out_dir / "best.pth"

    global_step = 0
    boundary_steps = [b * len(train_loader) for b in lr_boundaries]

    model.to(device)
    # Indices for joint chord metric (root + quality + bass)
    try:
        idx_root = task_names.index("ChordRoot12")
        idx_qual = task_names.index("ChordQuality")
        idx_bass = task_names.index("Bass12")
    except ValueError:
        idx_root = idx_qual = idx_bass = None
    for epoch in range(epochs):
        model.train()
        print(f"Training Epoch {epoch}")
        # Running metrics for training
        train_correct = [0 for _ in task_names]
        train_total = [0 for _ in task_names]
        chord_train_correct = 0
        chord_train_total = 0
        for Xs, ys in tqdm(train_loader):
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]
            logits = model(Xs)
            loss = 0.0
            for logit, y in zip(logits, ys):
                loss = loss + ce(logit.reshape(-1, logit.shape[-1]), y.reshape(-1))
            optim.zero_grad()
            # debug prints removed for training
            loss.backward()
            optim.step()

            # Update training metrics
            with torch.no_grad():
                for i, (logit, target) in enumerate(zip(logits, ys)):
                    pred = logit.argmax(dim=-1)  # (B, T)
                    target_s = target.squeeze(-1) if target.dim() == 3 and target.size(-1) == 1 else target  # (B, T)
                    mask = (target_s != -100)
                    train_correct[i] += int((pred[mask] == target_s[mask]).sum().item())
                    train_total[i] += int(mask.sum().item())
                if idx_root is not None and idx_qual is not None and idx_bass is not None:
                    pr = logits[idx_root].argmax(dim=-1)
                    pq = logits[idx_qual].argmax(dim=-1)
                    pb = logits[idx_bass].argmax(dim=-1)
                    tr = ys[idx_root].squeeze(-1)
                    tq = ys[idx_qual].squeeze(-1)
                    tb = ys[idx_bass].squeeze(-1)
                    mask = (tr != -100) & (tq != -100) & (tb != -100)
                    chord_train_correct += int(((pr == tr) & (pq == tq) & (pb == tb) & mask).sum().item())
                    chord_train_total += int(mask.sum().item())

            global_step += 1
            # if global_step == 10: break
            if boundary_steps and global_step in boundary_steps and len(lr_values) > 1:
                new_lr = lr_values[min(boundary_steps.index(global_step) + 1, len(lr_values) - 1)]
                for g in optim.param_groups:
                    g["lr"] = new_lr

        # Report training metrics
        train_task_acc = {name: (train_correct[i] / max(train_total[i], 1)) for i, name in enumerate(task_names)}
        chord_train_acc = chord_train_correct / max(chord_train_total, 1)

        # Validation metrics
        model.eval()
        val_correct = [0 for _ in task_names]
        val_total = [0 for _ in task_names]
        chord_val_correct = 0
        chord_val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]
                logits = model(Xs)
                # loss
                batch_loss = 0.0
                for logit, target in zip(logits, ys):
                    batch_loss = batch_loss + ce(logit.reshape(-1, logit.shape[-1]), target.reshape(-1))
                val_loss_sum += float(batch_loss.item())
                # per-task accs
                for i, (logit, target) in enumerate(zip(logits, ys)):
                    pred = logit.argmax(dim=-1)
                    target_s = target.squeeze(-1) if target.dim() == 3 and target.size(-1) == 1 else target
                    mask = (target_s != -100)
                    val_correct[i] += int((pred[mask] == target_s[mask]).sum().item())
                    val_total[i] += int(mask.sum().item())
                # chord acc
                if idx_root is not None and idx_qual is not None and idx_bass is not None:
                    pr = logits[idx_root].argmax(dim=-1)
                    pq = logits[idx_qual].argmax(dim=-1)
                    pb = logits[idx_bass].argmax(dim=-1)
                    tr = ys[idx_root].squeeze(-1)
                    tq = ys[idx_qual].squeeze(-1)
                    tb = ys[idx_bass].squeeze(-1)
                    mask = (tr != -100) & (tq != -100) & (tb != -100)
                    chord_val_correct += int(((pr == tr) & (pq == tq) & (pb == tb) & mask).sum().item())
                    chord_val_total += int(mask.sum().item())

        val_task_acc = {name: (val_correct[i] / max(val_total[i], 1)) for i, name in enumerate(task_names)}
        chord_val_acc = chord_val_correct / max(chord_val_total, 1)

        # Save checkpoints, select best by whole-chord accuracy
        ckpt_name = f"{epoch:02d}-valLoss{val_loss_sum:.3f}-valChordAcc{chord_val_acc:.4f}.pth"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "val_chord_acc": chord_val_acc,
            "val_loss": val_loss_sum,
            "val_task_acc": val_task_acc,
            "train_task_acc": train_task_acc,
            "train_chord_acc": chord_train_acc,
        }, out_dir / ckpt_name)
        if chord_val_acc > best_score:
            best_score = chord_val_acc
            torch.save(model.state_dict(), best_path)

        # Logging
        print(f"Epoch {epoch}")
        print(" Train task accs:" )
        for n in task_names:
            print(f"  {n}: {train_task_acc[n]:.4f}")
        print(f" Train chord acc: {chord_train_acc:.4f}")
        print(" Val task accs:")
        for n in task_names:
            print(f"  {n}: {val_task_acc[n]:.4f}")
        print(f" Val chord acc: {chord_val_acc:.4f}")
    return str(best_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr_boundaries", nargs="+", type=int, default=[40])
    ap.add_argument("--lr_values", nargs="+", type=float, default=[0.001, 0.001])
    ap.add_argument("--out_dir", type=str, default="augnet_pt/checkpoints")
    args = ap.parse_args()

    Xtr, ytr, Xv, yv, out_dims, task_names = load_npz(Path(args.npz))
    # Restrict and order tasks to the required subset
    required_order = [
        "ChordRoot12", "ChordQuality", "Bass12",
    ]
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(task_names)}
    present = [n for n in required_order if n in name_to_idx]
    if len(present) == 0:
        raise RuntimeError("No required tasks found in NPZ. Re-run encode_npz to include ChordRoot12/ChordQuality/Bass12.")
    sel_indices = [name_to_idx[n] for n in present]
    # Reorder / select
    ytr = [ytr[i] for i in sel_indices]
    yv = [yv[i] for i in sel_indices]
    out_dims = [out_dims[i] for i in sel_indices]
    task_names = present
    print(f"Training tasks: {task_names}")
    # _ = input()
    input_dims = [x.shape[-1] for x in Xtr]

    train_ds = PackedDataset(Xtr, ytr)
    val_ds = PackedDataset(Xv, yv)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = AugmentedNetPT(input_feature_dims=input_dims, output_class_dims=out_dims)
    print("Model Parameters: ", sum(p.numel() for p in model.parameters()))
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # device = "mps"
    best = train_loop(model, train_loader, val_loader, task_names, args.epochs, device, args.lr_boundaries, args.lr_values, Path(args.out_dir))
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
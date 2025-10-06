import argparse
import os
from pathlib import Path

import numpy as np
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .model import AugmentedNetPT
from .train import PackedDataset, load_npz


def evaluate(model_path: Path, npz_path: Path):
    # Load data and model
    Xtr, ytr, Xv, yv, out_dims, task_names = load_npz(npz_path)

    # Restrict and order tasks to the required subset (match training script)
    required_order = [
        "ChordRoot12", "ChordQuality", "Bass12",
    ]
    name_to_idx = {n: i for i, n in enumerate(task_names)}
    present = [n for n in required_order if n in name_to_idx]
    if len(present) == 0:
        raise RuntimeError("No required tasks found in NPZ. Re-run encode_npz to include outputs.")
    sel_indices = [name_to_idx[n] for n in present]
    yv = [yv[i] for i in sel_indices]
    out_dims = [out_dims[i] for i in sel_indices]
    task_names = present

    input_dims = [x.shape[-1] for x in Xtr]

    ds = PackedDataset(Xv, yv)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = AugmentedNetPT(input_feature_dims=input_dims, output_class_dims=out_dims)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])  # from full checkpoint
    else:
        model.load_state_dict(state)  # from best weights
    model.eval()

    # Resolve task indices on filtered task_names

    def try_index(names, keys):
        for k in keys:
            if k in names:
                return names.index(k)
        return None

    idx_root = try_index(task_names, ["ChordRoot12"])
    idx_qual = try_index(task_names, ["ChordQuality"])
    idx_bass = try_index(task_names, ["Bass12"])

    # Aggregators
    num_tasks = len(out_dims)
    total_tokens = [0 for _ in range(num_tasks)]
    total_correct = [0 for _ in range(num_tasks)]
    total_loss_sum = [0.0 for _ in range(num_tasks)]
    piece_acc_sum = [0.0 for _ in range(num_tasks)]
    piece_count = [0 for _ in range(num_tasks)]
    # Per-piece running accumulators (across potentially multiple segments)
    piece_curr_correct = [0 for _ in range(num_tasks)]
    piece_curr_total = [0 for _ in range(num_tasks)]

    # Whole-chord metrics
    rqi_correct_total = 0
    rqi_token_total = 0
    rqi_piece_acc_sum = 0.0
    rqi_piece_count = 0

    # Per-piece running accumulators for whole-chord metrics
    rqi_piece_curr_correct = 0
    rqi_piece_curr_total = 0

    # Confusion matrix buffers
    cm_buffers = {
        "ChordQuality": {"true": [], "pred": [], "C": out_dims[idx_qual] if idx_qual is not None else 0, "idx": idx_qual},
        "ChordRoot12": {"true": [], "pred": [], "C": out_dims[idx_root] if idx_root is not None else 0, "idx": idx_root},
        "Bass12": {"true": [], "pred": [], "C": out_dims[idx_bass] if idx_bass is not None else 0, "idx": idx_bass},
    }

    with torch.no_grad():
        for Xs, ys in tqdm(loader):
            # Ensure label tensors are (B, T)
            ys_clean = []
            for y in ys:
                if y.dim() == 3 and y.size(-1) == 1:
                    ys_clean.append(y.squeeze(-1))
                else:
                    ys_clean.append(y)

            logits_list = model([x for x in Xs])  # list of (B,T,C)

            # Padding mask from first input branch: padded frames are -1 per feature
            x0 = Xs[0]
            pad_mask = (x0.sum(dim=2) == -x0.shape[2])  # (B, T) True if pad
            valid_mask = ~pad_mask  # (B, T)

            B, T = valid_mask.shape

            # Per-task metrics
            for i, (logits, y) in enumerate(zip(logits_list, ys_clean)):
                preds = logits.argmax(dim=-1).cpu()  # (B, T)
                y_cpu = y.cpu()
                # Combine input-based valid mask with label padding (-100)
                vm = valid_mask & (y_cpu != -100)

                # Absolute accuracy and loss
                flat_vm = vm.reshape(-1)
                flat_preds = preds.reshape(-1)
                flat_targets = y_cpu.reshape(-1)
                total_tokens[i] += int(flat_vm.sum().item())
                total_correct[i] += int((flat_preds[flat_vm] == flat_targets[flat_vm]).sum().item())

                flat_logits = logits.reshape(-1, logits.shape[-1])
                if int(flat_vm.sum().item()) > 0:
                    loss_sum = F.cross_entropy(flat_logits[flat_vm], flat_targets[flat_vm], reduction="sum")
                    total_loss_sum[i] += float(loss_sum.item())

                # Accumulate per-piece counts (across segments)
                denom_seg = int(flat_vm.sum().item())
                if denom_seg > 0:
                    correct_seg = int((flat_preds[flat_vm] == flat_targets[flat_vm]).sum().item())
                    piece_curr_correct[i] += correct_seg
                    piece_curr_total[i] += denom_seg

                # Confusion buffers for selected tasks
                for task_key, cfg in cm_buffers.items():
                    if cfg["idx"] == i:
                        if int(flat_vm.sum().item()) > 0:
                            cm_buffers[task_key]["true"].extend(flat_targets[flat_vm].tolist())
                            cm_buffers[task_key]["pred"].extend(flat_preds[flat_vm].tolist())

            # Whole-chord: Root + Quality + Bass
            if idx_root is not None and idx_qual is not None and idx_bass is not None:
                pr = logits_list[idx_root].argmax(dim=-1).cpu()
                pq = logits_list[idx_qual].argmax(dim=-1).cpu()
                pb = logits_list[idx_bass].argmax(dim=-1).cpu()
                tr = ys_clean[idx_root].cpu()
                tq = ys_clean[idx_qual].cpu()
                tb = ys_clean[idx_bass].cpu()
                vm_rqi = valid_mask & (tr != -100) & (tq != -100) & (tb != -100)
                corr = (pr == tr) & (pq == tq) & (pb == tb) & vm_rqi
                rqi_correct_total += int(corr.sum().item())
                rqi_token_total += int(vm_rqi.sum().item())
                # Accumulate per-piece
                rqi_piece_curr_correct += int(corr.sum().item())
                rqi_piece_curr_total += int(vm_rqi.sum().item())

            # Detect end of piece via padding presence in inputs (rare edge case: exact multiple of sequence_length)
            end_flags = pad_mask.any(dim=1)  # (B,) True if this segment has any padding at end
            for b in range(B):
                if end_flags[b].item():
                    # Finalize per-task piece accuracies
                    for i in range(num_tasks):
                        if piece_curr_total[i] > 0:
                            piece_acc_sum[i] += piece_curr_correct[i] / piece_curr_total[i]
                            piece_count[i] += 1
                            piece_curr_correct[i] = 0
                            piece_curr_total[i] = 0
                    # Finalize whole-chord piece accuracies
                    if rqi_piece_curr_total > 0:
                        rqi_piece_acc_sum += rqi_piece_curr_correct / rqi_piece_curr_total
                        rqi_piece_count += 1
                        rqi_piece_curr_correct = 0
                        rqi_piece_curr_total = 0
                    

    # Flush last piece if it ended exactly on a segment boundary (no padding)
    for i in range(num_tasks):
        if piece_curr_total[i] > 0:
            piece_acc_sum[i] += piece_curr_correct[i] / piece_curr_total[i]
            piece_count[i] += 1
            piece_curr_correct[i] = 0
            piece_curr_total[i] = 0
    if rqi_piece_curr_total > 0:
        rqi_piece_acc_sum += rqi_piece_curr_correct / rqi_piece_curr_total
        rqi_piece_count += 1
        rqi_piece_curr_correct = 0
        rqi_piece_curr_total = 0
    

    # Compute aggregates
    abs_acc = [ (total_correct[i] / max(total_tokens[i], 1)) for i in range(num_tasks) ]
    abs_loss = [ (total_loss_sum[i] / max(total_tokens[i], 1)) for i in range(num_tasks) ]
    macro_acc = [ (piece_acc_sum[i] / max(piece_count[i], 1)) for i in range(num_tasks) ]

    rqi_abs_acc = (rqi_correct_total / max(rqi_token_total, 1)) if (idx_root is not None and idx_qual is not None and idx_bass is not None) else None
    rqi_macro_acc = (rqi_piece_acc_sum / max(rqi_piece_count, 1)) if (idx_root is not None and idx_qual is not None and idx_bass is not None) else None

    # Write metrics to file in checkpoint directory
    ckpt_dir = model_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_path = ckpt_dir / "evaluation_augnetPT.txt"
    with open(report_path, "w") as f:
        f.write("--- Component Metrics (Absolute) ---\n")
        for i, name in enumerate(task_names):
            f.write(f"{name}: acc={abs_acc[i]:.4f}, loss={abs_loss[i]:.6f}\n")
        f.write("\n--- Component Metrics (Macro-average by piece) ---\n")
        for i, name in enumerate(task_names):
            f.write(f"{name}: macro_acc={macro_acc[i]:.4f}\n")
        f.write("\n--- Whole-Chord Accuracies ---\n")
        if rqi_abs_acc is not None:
            f.write(f"Root+Quality+Bass: abs_acc={rqi_abs_acc:.4f}, macro_acc={rqi_macro_acc:.4f}\n")
        else:
            f.write("Root+Quality+Bass: N/A (tasks missing)\n")

    # Also print brief summary to stdout
    for i, t in enumerate(task_names):
        print(f"{t}: abs_acc={abs_acc[i]:.3f} macro_acc={macro_acc[i]:.3f} loss={abs_loss[i]:.4f}")
    if rqi_abs_acc is not None:
        print(f"WholeChord (Root+Quality+Bass): abs={rqi_abs_acc:.3f} macro={rqi_macro_acc:.3f}")

    # Load vocab for human-readable labels
    vocab_labels = None
    vocab_path = Path(npz_path).with_suffix('.vocab.json')
    if vocab_path.exists():
        try:
            with open(vocab_path, 'r') as vf:
                vocab_labels = json.load(vf)
        except Exception:
            vocab_labels = None

    def get_label_names(task_key: str, C: int):
        if vocab_labels is None:
            return [str(i) for i in range(C)]
        if task_key == "ChordQuality" and isinstance(vocab_labels.get("QUALITIES"), list):
            names = vocab_labels["QUALITIES"]
            return [str(n) for n in names[:C]] if len(names) >= C else [str(i) for i in range(C)]
        if task_key == "ChordRoot12" and isinstance(vocab_labels.get("ROOTS"), list):
            names = vocab_labels["ROOTS"]
            return [str(n) for n in names[:C]] if len(names) >= C else [str(i) for i in range(C)]
        if task_key == "Bass12" and isinstance(vocab_labels.get("ROOTS"), list):
            names = vocab_labels["ROOTS"]
            return [str(n) for n in names[:C]] if len(names) >= C else [str(i) for i in range(C)]
        return [str(i) for i in range(C)]

    # Confusion matrices for Quality, Root, Bass
    for key, buf in cm_buffers.items():
        idx = buf["idx"]
        if idx is None or len(buf["true"]) == 0:
            continue
        C = out_dims[idx]
        true_vals = np.array(buf["true"], dtype=int)
        pred_vals = np.array(buf["pred"], dtype=int)
        labels_idx = list(range(C))
        cm = confusion_matrix(true_vals, pred_vals, labels=labels_idx)
        total_cm = cm.sum() if cm.sum() > 0 else 1
        ratios = cm / total_cm

        # Create annotated heatmap with counts and ratios
        plt.figure(figsize=(max(6, C * 0.5), max(5, C * 0.5)))
        label_names = get_label_names(key, C)
        ax = sns.heatmap(cm, annot=False, cmap='Blues', cbar=False, xticklabels=label_names, yticklabels=label_names)
        # Add custom annotations with both count and ratio in [0,1]
        for i in range(C):
            for j in range(C):
                ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n{ratios[i, j]:.2f}",
                        ha='center', va='center', fontsize=8, color='black')
        ax.set_title(f'Confusion Matrix – {key}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(np.arange(C) + 0.5)
        ax.set_yticks(np.arange(C) + 0.5)
        ax.set_xticklabels(label_names, rotation=90)
        ax.set_yticklabels(label_names, rotation=0)
        plt.tight_layout()
        out_png = ckpt_dir / f'confusion_matrix_{key}.png'
        plt.savefig(out_png)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--npz", required=True, type=str)
    args = ap.parse_args()
    evaluate(Path(args.model), Path(args.npz))


if __name__ == "__main__":
    main()
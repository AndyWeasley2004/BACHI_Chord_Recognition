import os
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

import chordgnn as st
from chordgnn.models.chord import PostChordPrediction, ChordPredictionModel, unique_onsets


def _prepare_device(gpus: str) -> Tuple[torch.device, List[int]]:
    if isinstance(eval(gpus), int):
        if eval(gpus) >= 0:
            devices = [eval(gpus)]
        else:
            devices = []
    else:
        devices = [eval(gpu) for gpu in gpus.split(",")]
    device = torch.device("cuda:0" if (len(devices) > 0 and torch.cuda.is_available()) else "cpu")
    return device, devices


def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    yt = y_true[valid]
    yp = y_pred[valid]
    if yt.size == 0:
        return cm
    idx = yt * num_classes + yp
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    cm = binc.reshape((num_classes, num_classes))
    return cm


def _normalize_cm(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    return norm


def _plot_cm(cm_norm: np.ndarray, out_path: str, title: str, class_names: List[str]):
    import matplotlib.pyplot as plt

    n = cm_norm.shape[0]
    # Heuristic sizing: scale with number of classes
    size = max(6.0, min(0.18 * n, 28.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Ticks and labels (very large class sets will be unreadable; keep ticks but skip long texts)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    if n <= 40 and class_names is not None and len(class_names) == n:
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
    else:
        ax.set_xticklabels([str(i) for i in range(n)], rotation=90, fontsize=6)
        ax.set_yticklabels([str(i) for i in range(n)], fontsize=6)

    # Annotate all cells with two decimals (percentages between 0 and 1)
    thresh = 0.5
    fontsize = 6 if n > 60 else (7 if n > 30 else 8)
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=fontsize)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def evaluate(args):
    device, devices = _prepare_device(args.gpus)

    # Data
    datamodule = st.data.AugmentedGraphDatamodule(
        num_workers=args.num_workers,
        include_synth=args.include_synth,
        num_tasks=args.num_tasks,
        collection=args.collection,
        batch_size=1,
        version=args.data_version,
        skip_processing=args.skip_processing,
    )
    datamodule.setup("test")

    tasks: Dict[str, int] = datamodule.tasks
    in_feats: int = datamodule.features

    # Labels per task (for axis tick labels)
    if args.data_version in ("v1.0.0", "pop909"):
        from chordgnn.utils.chord_representations import available_representations
    else:
        from chordgnn.utils.chord_representations_latest import available_representations
    class_lists = {t: available_representations[t].classList for t in tasks.keys()}

    # Build a skeleton frozen model to satisfy PostChordPrediction init
    frozen_skeleton = ChordPredictionModel(in_feats, 256, tasks, 2, dropout=0.0)

    # Load post-processed model checkpoint (Stage 2)
    map_loc = device
    model: PostChordPrediction = PostChordPrediction.load_from_checkpoint(
        args.use_ckpt,
        frozen_model=frozen_skeleton,
        map_location=map_loc,
    )
    model.eval()
    model.to(device)

    # Accumulators
    totals = {t: 0 for t in tasks.keys()}
    corrects = {t: 0 for t in tasks.keys()}
    piece_accs: Dict[str, List[float]] = {t: [] for t in tasks.keys()}
    cms: Dict[str, np.ndarray] = {t: np.zeros((tasks[t], tasks[t]), dtype=np.int64) for t in tasks.keys()}
    # Full-chord (root+quality+bass) metrics
    compute_full = all(k in tasks for k in ("root", "quality", "bass"))
    full_totals: int = 0
    full_corrects: int = 0
    full_piece_accs: List[float] = []

    # Output directory
    ckpt_base = os.path.splitext(os.path.basename(args.use_ckpt.rstrip('/')))[0]
    out_dir = os.path.join(args.out_dir, ckpt_base)
    os.makedirs(out_dir, exist_ok=True)

    loader = datamodule.test_dataloader()
    for idx, batch in enumerate(tqdm(loader, desc="Evaluating (per piece)")):
        if args.limit_pieces and idx >= args.limit_pieces:
            break
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch

        # Move tensors
        batch_inputs = batch_inputs.to(device)
        edges = edges.to(device)
        edge_type = edge_type.to(device)
        onset_divs = onset_divs.to(device)

        # Prepare onset grouping
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)

        with torch.no_grad():
            base_out = model.frozen_model((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
            pred_out = model.module(base_out)

        # Per-task metrics for this piece
        for task, n_classes in tasks.items():
            y_true = batch_labels[task].detach().cpu().numpy().astype(np.int64)
            y_pred = pred_out[task].detach().cpu().argmax(dim=1).numpy().astype(np.int64)
            mask = y_true >= 0
            if mask.sum() == 0:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]

            # Micro accumulators
            totals[task] += int(mask.sum())
            corrects[task] += int((yt == yp).sum())

            # Per-piece accuracy for macro over pieces
            piece_acc = float((yt == yp).mean())
            piece_accs[task].append(piece_acc)

            # Confusion matrix accumulation
            cms[task] += _compute_confusion_matrix(yt, yp, n_classes)

        # Full-chord metrics (simultaneous correctness on root, quality, bass)
        if compute_full:
            ytr = batch_labels["root"].detach().cpu().numpy().astype(np.int64)
            ytq = batch_labels["quality"].detach().cpu().numpy().astype(np.int64)
            yti = batch_labels["bass"].detach().cpu().numpy().astype(np.int64)
            pr = pred_out["root"].detach().cpu().argmax(dim=1).numpy().astype(np.int64)
            pq = pred_out["quality"].detach().cpu().argmax(dim=1).numpy().astype(np.int64)
            pi = pred_out["bass"].detach().cpu().argmax(dim=1).numpy().astype(np.int64)
            mask_all = (ytr >= 0) & (ytq >= 0) & (yti >= 0)
            if mask_all.sum() > 0:
                corr = (pr == ytr) & (pq == ytq) & (pi == yti)
                full_totals += int(mask_all.sum())
                full_corrects += int(corr[mask_all].sum())
                full_piece_accs.append(float(corr[mask_all].mean()))

    # Save metrics (per-task micro and macro-by-piece)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("task,micro_acc,macro_piece_acc,total_support\n")
        for task in tasks.keys():
            micro = (corrects[task] / totals[task]) if totals[task] > 0 else 0.0
            macro = (float(np.mean(piece_accs[task])) if len(piece_accs[task]) > 0 else 0.0)
            f.write(f"{task},{micro:.6f},{macro:.6f},{totals[task]}\n")

    # Save per-piece accuracies
    per_piece_path = os.path.join(out_dir, "per_piece_metrics.csv")
    # Re-run to write per-piece lines (task-wise lists are already stored; we will write only averages per task here)
    with open(per_piece_path, "w") as f:
        f.write("task,macro_piece_acc,num_pieces\n")
        for task in tasks.keys():
            macro = (float(np.mean(piece_accs[task])) if len(piece_accs[task]) > 0 else 0.0)
            f.write(f"{task},{macro:.6f},{len(piece_accs[task])}\n")

    # Plot confusion matrices (normalized 0..1 with two decimals) only for root, bass, quality
    for task in [t for t in ("root", "bass", "quality") if t in cms]:
        cm_counts = cms[task]
        cm_norm = _normalize_cm(cm_counts)
        class_names = class_lists.get(task, None)
        out_png = os.path.join(out_dir, f"confmat_{task}.png")
        _plot_cm(cm_norm, out_png, title=f"Confusion Matrix (normalized) - {task}", class_names=class_names)

    # Write human-readable summary including full-chord accuracies
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_base}\n")
        f.write(f"Collection: {args.collection}\n")
        f.write(f"Data version: {args.data_version}\n")
        f.write(f"Include synth: {args.include_synth}\n")
        f.write("\nPer-task accuracies:\n")
        for task in tasks.keys():
            micro = (corrects[task] / totals[task]) if totals[task] > 0 else 0.0
            macro = (float(np.mean(piece_accs[task])) if len(piece_accs[task]) > 0 else 0.0)
            f.write(f"- {task}: micro={micro:.4f}, macro_by_piece={macro:.4f}, support={totals[task]}\n")
        if compute_full:
            full_micro = (full_corrects / full_totals) if full_totals > 0 else 0.0
            full_macro = float(np.mean(full_piece_accs)) if len(full_piece_accs) > 0 else 0.0
            f.write("\nFull-chord accuracy (root+quality+bass):\n")
            f.write(f"- token_micro={full_micro:.4f} (support={full_totals})\n")
            f.write(f"- macro_by_piece={full_macro:.4f} (num_pieces={len(full_piece_accs)})\n")
        f.write("\nConfusion matrices saved for: root, bass, quality.\n")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrices to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--include_synth', action='store_true')
    parser.add_argument('--collection', type=str, default="all",
                        choices=["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"]) 
    parser.add_argument('--num_tasks', type=int, default=11)
    parser.add_argument('--data_version', type=str, default="v1.0.0", choices=["v1.0.0", "latest", "pop909"]) 
    parser.add_argument('--skip_processing', action='store_true')
    parser.add_argument('--use_ckpt', type=str, required=True, help='Path to Stage 2 (post-processing) checkpoint (.ckpt).')
    parser.add_argument('--out_dir', type=str, default=os.path.join('logs', 'eval'))
    parser.add_argument('--limit_pieces', type=int, default=0, help='Optional limit on number of pieces to evaluate (0=all).')

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()


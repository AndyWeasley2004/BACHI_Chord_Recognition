import argparse
from pathlib import Path

import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .augnet_utils import (
    parse_score,
    encode_bass19,
    encode_chromagram19,
    encode_measure_note_onset14,
    pad_to_sequence_length,
    load_vocabs,
)
from .model import AugmentedNetPT
from .train import load_npz


def select_heads_from_npz(npz_path: Path):
    # Mirror train/evaluate task selection
    _, _, _, _, out_dims, task_names = load_npz(npz_path)
    required_order = [
        "LocalKey38", "PrimaryDegree22", "SecondaryDegree22",
        "ChordQuality11", "Inversion4", "ChordRoot35",
        "RomanNumeral31", "HarmonicRhythm7", "Bass35",
        "TonicizedKey38", "PitchClassSet121",
    ]
    name_to_idx = {n: i for i, n in enumerate(task_names)}
    present = [n for n in required_order if n in name_to_idx]
    sel_indices = [name_to_idx[n] for n in present]
    selected_dims = [out_dims[i] for i in sel_indices]
    return present, selected_dims


def build_decoders(vocabs):
    return {
        "LocalKey38": list(vocabs.get("KEYS", [])),
        "PrimaryDegree22": list(vocabs.get("DEGREES", [])),
        "SecondaryDegree22": list(vocabs.get("DEGREES", [])),
        "ChordQuality11": list(vocabs.get("QUALITIES", [])),
        "Inversion4": list(vocabs.get("INVERSIONS", [])),
        "ChordRoot35": list(vocabs.get("ROOTS", [])),
        "RomanNumeral31": list(vocabs.get("COMMON_RN", [])),
        "HarmonicRhythm7": [str(i) for i in range(7)],
        "Bass35": list(vocabs.get("SPELLINGS", [])),
        "TonicizedKey38": list(vocabs.get("KEYS", [])),
        "PitchClassSet121": [str(tuple(x)) for x in vocabs.get("PCSETS", [])],
    }


def run_inference_for_piece(model, task_names, decoders, score_path: Path, out_dir: Path, device: torch.device):
    df = parse_score(score_path)
    # Encode inputs
    X_bass19 = encode_bass19(df)
    X_chroma19 = encode_chromagram19(df)
    X_mn14 = encode_measure_note_onset14(df)
    seqlen = 640
    enc_inputs = [
        pad_to_sequence_length(X_bass19, seqlen, value=-1),
        pad_to_sequence_length(X_chroma19, seqlen, value=-1),
        pad_to_sequence_length(X_mn14, seqlen, value=0),
    ]
    X = [torch.tensor(x, dtype=torch.float32) for x in enc_inputs]
    X = [x.to(device) for x in X]

    with torch.no_grad():
        logits = model([x for x in X])  # list of (N, T, C)
        preds_idx = [logit.argmax(dim=-1).cpu() for logit in logits]  # list of (N, T)

    # Compute valid mask from first input branch (pad=-1)
    x0 = X[0]
    pad_mask = (x0.sum(dim=2) == -x0.shape[2])  # (N, T)
    valid_mask = ~pad_mask

    # Flatten to full-length, then mask padded frames
    flat_valid = valid_mask.reshape(-1).cpu()
    decoded = {}
    for t, p in zip(task_names, preds_idx):
        flat_idx = p.reshape(-1)
        idx_valid = flat_idx[flat_valid].numpy().tolist()
        vocab = decoders.get(t, None)
        if vocab is None or len(vocab) == 0:
            decoded[t] = idx_valid
        else:
            # Guard against OOB indices
            vocab_len = len(vocab)
            decoded[t] = [vocab[i] if 0 <= int(i) < vocab_len else str(i) for i in idx_valid]

    dfout = pd.DataFrame(decoded)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / (score_path.stem + "_annotated_pt.csv")
    dfout.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--npz", required=True, type=str, help="Dataset npz used for training to infer head dims and order")
    ap.add_argument("--data_root", type=str, default="unique_data_collection")
    ap.add_argument("--test_list", type=str, default="test_files.json")
    args = ap.parse_args()

    model_path = Path(args.model)
    npz_path = Path(args.npz)
    data_root = Path(args.data_root)
    test_list_path = Path(args.test_list)

    # Select heads as in training/evaluation
    task_names, class_dims = select_heads_from_npz(npz_path)
    vocabs = load_vocabs(npz_path.with_suffix('.vocab.json'))
    decoders = build_decoders(vocabs)

    # Build model aligned with heads and input feature dims (fixed encoders)
    input_feature_dims = [19, 19, 14]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AugmentedNetPT(input_feature_dims=input_feature_dims, output_class_dims=class_dims).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])  # full checkpoint
    else:
        model.load_state_dict(state)
    model.eval()

    # Output directory inside checkpoint directory
    out_dir = model_path.parent / "test_predictions_pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read test file ids
    with open(test_list_path, "r") as f:
        file_ids = json.load(f)

    saved = 0
    for fid in tqdm(file_ids):
        piece_dir = data_root / fid
        if not piece_dir.exists():
            # Skip missing entries silently
            continue
        # Find a score file (prefer .musicxml)
        score_path = None
        for ext in [".musicxml", ".xml", ".mxl"]:
            cand = list(piece_dir.glob(f"*{ext}"))
            if cand:
                score_path = cand[0]
                break
        if score_path is None:
            continue
        try:
            out_csv = run_inference_for_piece(model, task_names, decoders, score_path, out_dir, device)
            saved += 1
        except Exception as e:
            print(f"Error processing {score_path}: {e}")
            continue

    print(f"Saved {saved} prediction files to {out_dir}")


if __name__ == "__main__":
    main()
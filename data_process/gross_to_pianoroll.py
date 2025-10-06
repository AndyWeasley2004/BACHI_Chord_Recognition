import csv
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils import get_pianoroll_and_labels, shift_pianoroll, transpose_label
import warnings
warnings.filterwarnings("ignore")

def _load_events_csv(csv_path: Path) -> list:
    """Read the *decomposed.csv* file and return a list of harmony events.
    The current (7-field) format contains the columns
        offset_qb, root, triad, bass, seventh, ninth, eleventh, thirteenth
    whereas the legacy format contains only
        offset_qb, root, quality, bass.
    This loader handles **both** versions: missing columns are replaced with
    the neutral symbol ``"N"`` so that the downstream pipeline always
    receives a 7-element label list.
    """
    events: list = []
    with csv_path.open() as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if header is None:
            return events

        # Build a mapping header->index for quick look-ups (case insensitive)
        col_idx = {h.lower(): i for i, h in enumerate(header)}

        # Mandatory columns
        idx_offset = col_idx["offset_qb"]
        idx_root = col_idx["root"]
        idx_bass = col_idx["bass"]
        idx_key = col_idx["local_key"]
        idx_quality = col_idx["quality"]

        for row in rdr:
            off = float(row[idx_offset])
            root = row[idx_root]
            bass = row[idx_bass]
            key = row[idx_key]
            quality = row[idx_quality]
            label = [root, quality, bass, key]
            events.append([off, label])

    if not events:
        return events

    # Shift everything so the first event starts at beat 0
    events.sort(key=lambda e: e[0])
    start_beat = events[0][0]
    if start_beat != 0:
        events = [[e[0] - start_beat, e[1]] for e in events]
    return events


def _find_score_file(piece_dir: Path) -> Path:
    """Return the first recognised score file within *piece_dir*."""
    exts = (".musicxml", ".mxl", ".xml", ".mid", ".midi")
    for p in piece_dir.iterdir():
        if p.suffix.lower() in exts:
            return p
    return None


def process_piece(piece_dir: Path, out_root: Path, resolution: int = 12, label_resolution: int = 2) -> bool:
    """Convert a single *piece_dir* → multiple NPZ files under *out_root*.

    Returns ``True`` on success, ``False`` otherwise.
    """
    # print(piece_dir)
    score_path = _find_score_file(piece_dir)
    csv_path = piece_dir / "chord_symbol.csv"
    if score_path is None or not csv_path.exists():
        return False

    events = _load_events_csv(csv_path)
    pianoroll, unaligned_labels, boundaries = get_pianoroll_and_labels(score_path, events, resolution, label_resolution)
    if pianoroll is None or unaligned_labels is None or boundaries is None:
        print(f"Error processing {piece_dir.stem}")
        return False

    piece_name = piece_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    for shift in range(-6, 6):
        shifted_roll = shift_pianoroll(pianoroll, shift)
        labels = np.array([transpose_label(lbl, shift) for lbl in unaligned_labels])
        out_name = f"{piece_name}_shift{shift}.npz"
        np.savez_compressed(out_root / out_name, 
                          pianoroll=shifted_roll, 
                          labels=labels, 
                          boundaries=boundaries)
    return True

def main():
    import argparse

    p = argparse.ArgumentParser(description="Convert gross dataset to pianoroll+labels")
    p.add_argument("--gross-root", type=Path, default=Path("data_root/gross_dataset"))
    p.add_argument("--out", type=Path, default=Path("data_root/final_data"))
    p.add_argument("--resolution", type=int, default=12, help="frames per beat")
    p.add_argument("--label-resolution", type=int, default=2, help="labels per beat")
    args = p.parse_args()

    piece_dirs = [d for d in args.gross_root.iterdir() if d.is_dir()]
    successes = 0
    for pd in tqdm(piece_dirs, desc="Gross → NPZ"):
        if process_piece(pd, args.out, args.resolution, args.label_resolution):
            successes += 1
    print(f"Converted {successes} / {len(piece_dirs)} piece(s).")


if __name__ == "__main__":
    main()

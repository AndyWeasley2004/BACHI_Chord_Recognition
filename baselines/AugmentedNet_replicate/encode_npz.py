import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .augnet_utils import (
    DATASETSUMMARYFILE,
    build_vocabs_from_dataset,
    save_vocabs,
    encode_measure_note_onset14,
    pad_to_sequence_length,
    encode_output_series,
    normalize_pitch_name_to_keyboard,
    PREFERRED_PITCH_CLASSES,
    NOTENAMES,
)


def collect_joint_paths(tsv_dir: Path, splits: List[str]) -> List[Path]:
    paths: List[Path] = []
    for split in splits:
        split_dir = tsv_dir / split
        if not split_dir.exists():
            continue
        for p in split_dir.glob("*.tsv"):
            paths.append(p)
    return paths


def _ensure_four_and_normalize(pitch_names_row):
    try:
        vals = list(pitch_names_row)
    except Exception:
        vals = []
    while len(vals) < 4:
        vals.append(vals[0] if vals else "C")
    vals = vals[:4]
    return [normalize_pitch_name_to_keyboard(v) for v in vals]


def _normalize_to_keyboard_12(token: str) -> str:
    """Map any textual note to a 12-name keyboard spelling (C..B) without music21.

    Handles bb/-- and ##; ignores octave numbers. Examples: 'Abb4' -> 'G', 'C##' -> 'D'.
    """
    import re
    s = str(token).strip()
    m = re.match(r"^\s*([A-Ga-g])([#b\-]{0,2}).*$", s)
    if not m:
        return "C"
    letter = m.group(1).upper()
    acc = (m.group(2) or "").lower()
    base_pc_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = base_pc_map.get(letter, 0)
    # Count sharps and flats; treat '-' as flat
    n_sharp = acc.count('#')
    n_flat = acc.count('b') + acc.count('-')
    offset = max(min(n_sharp - n_flat, 2), -2)
    pc = (base + offset) % 12
    return PREFERRED_PITCH_CLASSES[pc]


def _sanitize_note_token(n: str) -> str:
    # Pure-text normalization to 12-name set
    return _normalize_to_keyboard_12(n)


def _sanitize_score_notes_column(series: pd.Series) -> pd.Series:
    def _sanitize_list(lst):
        try:
            return [_sanitize_note_token(x) for x in list(lst)]
        except Exception:
            return ["C"]
    return series.apply(_sanitize_list)


# ---- Local, music21-free encoders to avoid accidental parsing issues ----

def _safe_transpose_keyboard_name(name: str, semitones: int) -> str:
    base = _normalize_to_keyboard_12(name)
    try:
        idx = PREFERRED_PITCH_CLASSES.index(base)
    except ValueError:
        idx = 0
    return PREFERRED_PITCH_CLASSES[(idx + semitones) % 12]


def _encode_bass12_safe(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(PREFERRED_PITCH_CLASSES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        bass = "C"
        try:
            if notes and len(notes) > 0:
                bass = str(notes[0])
        except Exception:
            pass
        name_t = _safe_transpose_keyboard_name(bass, semitones)
        pc_idx = PREFERRED_PITCH_CLASSES.index(name_t)
        arr[i, pc_idx] = 1
    return arr


def _encode_bass7_safe(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(NOTENAMES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        bass = "C"
        try:
            if notes and len(notes) > 0:
                bass = str(notes[0])
        except Exception:
            pass
        name_t = _safe_transpose_keyboard_name(bass, semitones)
        letter = name_t[0]
        idx = NOTENAMES.index(letter)
        arr[i, idx] = 1
    return arr


def _encode_chromagram12_safe(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(PREFERRED_PITCH_CLASSES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        iterable = []
        try:
            iterable = list(notes)
        except Exception:
            pass
        for n in iterable:
            name_t = _safe_transpose_keyboard_name(str(n), semitones)
            pc_idx = PREFERRED_PITCH_CLASSES.index(name_t)
            arr[i, pc_idx] = 1
    return arr


def _encode_chromagram7_safe(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(NOTENAMES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        iterable = []
        try:
            iterable = list(notes)
        except Exception:
            pass
        for n in iterable:
            name_t = _safe_transpose_keyboard_name(str(n), semitones)
            letter = name_t[0]
            idx = NOTENAMES.index(letter)
            arr[i, idx] = 1
    return arr


def _encode_bass19_safe(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    return np.concatenate([_encode_bass7_safe(df, semitones), _encode_bass12_safe(df, semitones)], axis=1)


def _encode_chromagram19_safe(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    return np.concatenate([_encode_chromagram7_safe(df, semitones), _encode_chromagram12_safe(df, semitones)], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv_dir", required=True, type=str)
    p.add_argument("--npz_out", required=True, type=str)
    p.add_argument("--sequence_length", type=int, default=640)
    p.add_argument("--transpose_training", action="store_true", help="Add 11 extra transpositions for training examples (originals only).")
    args = p.parse_args()

    tsv_dir = Path(args.tsv_dir)
    npz_out = Path(args.npz_out)
    npz_out.parent.mkdir(parents=True, exist_ok=True)

    summary_file = tsv_dir / DATASETSUMMARYFILE
    if not summary_file.exists():
        raise FileNotFoundError("dataset_summary.tsv not found in tsv_dir; run preprocess first.")

    joint_paths = collect_joint_paths(tsv_dir, ["training", "validation"]) \
        or collect_joint_paths(tsv_dir, ["training"])  # tolerate missing validation
    
    print(f"Building vocabs from {len(joint_paths)} joint paths")
    vocabs = build_vocabs_from_dataset(joint_paths)
    save_vocabs(vocabs, npz_out.with_suffix(".vocab.json"))

    arrays = {}
    for split in ["training", "validation"]:
        split_dir = tsv_dir / split
        if not split_dir.exists():
            continue
        X_bass19_list = []
        X_chroma19_list = []
        X_mn14_list = []
        # Targets restricted to POP909 tasks
        y_dict = {k: [] for k in [
            "ChordRoot12", "ChordQuality", "Bass12",
        ]}
        for path in tqdm(split_dir.glob("*.tsv")):
            df = pd.read_csv(path, sep="\t")
            df.set_index("j_offset", inplace=True)
            for col in ["s_notes", "s_intervals", "s_isOnset", "a_pitchNames", "a_pcset", "qualityScoreNotes"]:
                if col in df.columns:
                    df[col] = df[col].apply(eval)
            # Sanitize score notes to avoid unsupported accidental shapes during encoding
            # Sanitize score notes to avoid unsupported accidental shapes during encoding
            if "s_notes" in df.columns:
                df["s_notes"] = _sanitize_score_notes_column(df["s_notes"])

            # Derive HarmonicRhythm7 from a_isOnset boolean flags
            template = [1, 2, 2, 3, 3, 3, 3] + ([4] * 8) + ([5] * 16) + ([6] * 32)
            a_is_onset_series = df["a_isOnset"].fillna(False).astype(bool) if "a_isOnset" in df.columns else pd.Series([False] * len(df), index=df.index)
            hr_vals = []
            t = 62
            for onset in a_is_onset_series.tolist():
                if onset:
                    hr_vals.append(0)
                    t = 0
                else:
                    hr_vals.append(template[min(t, 62)])
                    t = min(t + 1, 62)

            is_synth = str(path.name).endswith("_synth.tsv")
            # Only transpose original training files when requested
            transpositions = range(12) if (args.transpose_training and split == "training" and not is_synth) else [0]
            for s in transpositions:
                try:
                    X_bass19 = _encode_bass19_safe(df, semitones=s)
                    X_chroma19 = _encode_chromagram19_safe(df, semitones=s)
                    X_mn14 = encode_measure_note_onset14(df)
                    X_bass19_list.append(pad_to_sequence_length(X_bass19, args.sequence_length, value=-1))
                    X_chroma19_list.append(pad_to_sequence_length(X_chroma19, args.sequence_length, value=-1))
                    X_mn14_list.append(pad_to_sequence_length(X_mn14, args.sequence_length, value=0))
                    from .augnet_utils import transpose_spelling
                    # Root transposes with example
                    y_dict["ChordRoot12"].append(pad_to_sequence_length(
                        encode_output_series([transpose_spelling(normalize_pitch_name_to_keyboard(r), s) for r in df["a_root"].astype(str).tolist()], vocabs["SPELLINGS"]).reshape(-1, 1),
                        args.sequence_length, value=-100
                    ))
                    # Bass transposes with example
                    y_dict["Bass12"].append(pad_to_sequence_length(
                        encode_output_series([transpose_spelling(normalize_pitch_name_to_keyboard(b), s) for b in df["a_bass"].astype(str).tolist()], vocabs["SPELLINGS"]).reshape(-1, 1),
                        args.sequence_length, value=-100
                    ))
                    # Quality does not change with global transposition
                    y_dict["ChordQuality"].append(pad_to_sequence_length(
                        encode_output_series(df["a_quality"].astype(str).tolist(), vocabs["QUALITIES"]).reshape(-1, 1),
                        args.sequence_length, value=-100
                    ))
                except Exception as e:
                    print(f"Skip {path.name} (transpose={s}) due to encoding error: {e}")
                    continue

        if X_bass19_list:
            arrays[f"{split}_X_Bass19"] = np.concatenate(X_bass19_list, axis=0)
            arrays[f"{split}_X_Chromagram19"] = np.concatenate(X_chroma19_list, axis=0)
            arrays[f"{split}_X_MeasureNoteOnset14"] = np.concatenate(X_mn14_list, axis=0)
            for k in y_dict:
                arrays[f"{split}_y_{k}"] = np.concatenate(y_dict[k], axis=0)

    np.savez_compressed(npz_out, **arrays)


if __name__ == "__main__":
    main()
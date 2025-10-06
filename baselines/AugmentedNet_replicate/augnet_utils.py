from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import math
import numpy as np
import pandas as pd
from music21 import converter, roman as m21_roman, note as m21_note, chord as m21_chord, pitch as m21_pitch, interval as m21_interval

# Constants mirroring AugmentedNet defaults
FLOATSCALE = 4
FRAMEBASENOTE = 32
FIXEDOFFSET = round(4.0 / FRAMEBASENOTE, FLOATSCALE)  # sixteenth note
DATASETSUMMARYFILE = "dataset_summary.tsv"

NOTENAMES = ("C", "D", "E", "F", "G", "A", "B")
PITCHCLASSES = list(range(12))
ACCIDENTALS = ("--", "-", "", "#", "##")
# Restrict spellings to keyboard-friendly names only (12 classes)
SPELLINGS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]
NOTEDURATIONS = [0, 1, 2, 3, 4, 5, 6]
PREFERRED_PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


# ----------------------- Padding -----------------------

def pad_to_sequence_length(arr: np.ndarray, sequence_length: int, value: int = 0) -> np.ndarray:
    frames, features = arr.shape
    features_per_sequence = sequence_length * features
    features_in_example = frames * features
    remainder = features_in_example % features_per_sequence
    pad = (features_per_sequence - remainder) if remainder != 0 else 0
    pad_timesteps = int(pad / features)
    if pad_timesteps > 0:
        arr = np.pad(arr, ((0, pad_timesteps), (0, 0)), constant_values=value)
    arr = arr.reshape(-1, sequence_length, features)
    return arr


# ----------------------- Score parsing -----------------------

S_COLUMNS = [
    "s_offset",
    "s_duration",
    "s_measure",
    "s_notes",
    "s_intervals",
    "s_isOnset",
]
S_LISTTYPE_COLUMNS = ["s_notes", "s_intervals", "s_isOnset"]


def parse_score(score_path: Path, fixed_offset: float = FIXEDOFFSET) -> pd.DataFrame:
    s = converter.parse(str(score_path), forceSource=True)
    # Remove percussion parts if present
    perc = [p for p in s.parts if list(p.recurse().getElementsByClass("PercussionClef"))]
    s.remove(perc, recurse=True)

    dfdict: Dict[str, List] = {col: [] for col in S_COLUMNS}

    # Measure number shift for pickup measures
    first_measure = s.parts[0].measure(0) or s.parts[0].measure(1)
    is_anacrusis = True if first_measure and first_measure.paddingLeft > 0.0 else False
    measure_shift = -1 if (is_anacrusis and getattr(first_measure, "number", 0) == 1) else 0

    for c in s.chordify().flat.notesAndRests:
        dfdict["s_offset"].append(round(float(c.offset), FLOATSCALE))
        dfdict["s_duration"].append(round(float(c.quarterLength), FLOATSCALE))
        dfdict["s_measure"].append(int(getattr(c, "measureNumber", 0)) + measure_shift)
        if isinstance(c, m21_note.Rest):
            dfdict["s_notes"].append(np.nan)
            dfdict["s_intervals"].append(np.nan)
            dfdict["s_isOnset"].append(np.nan)
            continue
        note_names = [n.pitch.nameWithOctave for n in c]
        dfdict["s_notes"].append(note_names)
        intervs = []
        if len(note_names) > 1:
            try:
                rootP = m21_pitch.Pitch(note_names[0])
                for pstr in note_names[1:]:
                    intervs.append(m21_interval.Interval(rootP, m21_pitch.Pitch(pstr)).simpleName)
            except Exception:
                intervs = []
        dfdict["s_intervals"].append(intervs)
        onsets = [(not n.tie or n.tie.type == "start") for n in c]
        dfdict["s_isOnset"].append(onsets)

    df = pd.DataFrame(dfdict)
    # Extend last duration to match the score's true end
    try:
        current_last_offset = float(df.tail(1)["s_offset"].iloc[0]) + float(df.tail(1)["s_duration"].iloc[0])
    except Exception:
        current_last_offset = 0.0
    try:
        last_measure = s.parts[0].measure(-1)
        filled = last_measure.duration.quarterLength / float(last_measure.barDurationProportion())
        true_end = float(last_measure.offset + filled)
        delta = true_end - current_last_offset
        df.loc[len(df) - 1, "s_duration"] += round(float(delta), FLOATSCALE)
    except Exception:
        pass

    df.set_index("s_offset", inplace=True)
    df = df[~df.index.duplicated()]

    # Reindex to fixed grid
    first_row = df.head(1)
    last_row = df.tail(1)
    min_offset = float(first_row.index.to_numpy()[0]) if len(first_row) else 0.0
    max_offset = float((last_row.index + last_row.s_duration).to_numpy()[0]) if len(last_row) else 0.0
    new_index = np.arange(min_offset, max_offset, fixed_offset)
    df = df.reindex(index=df.index.union(new_index))
    df.s_notes = df.s_notes.fillna(method="ffill").fillna(method="bfill")
    new_col = pd.Series([[False] * n for n in df.s_notes.str.len().to_list()], index=df.index)
    df.s_isOnset.fillna(value=new_col, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df = df.reindex(index=new_index)
    return df


# ----------------------- Annotation parsing (RomanText/DCML) -----------------------

A_COLUMNS = [
    "a_offset",
    "a_measure",
    "a_duration",
    "a_annotationNumber",
    "a_romanNumeral",
    "a_isOnset",
    "a_pitchNames",
    "a_bass",
    "a_tenor",
    "a_alto",
    "a_soprano",
    "a_root",
    "a_inversion",
    "a_quality",
    "a_pcset",
    "a_localKey",
    "a_tonicizedKey",
    "a_degree1",
    "a_degree2",
]
A_LISTTYPE_COLUMNS = ["a_pitchNames", "a_pcset"]


def simplify_rn_figure(fig: str) -> str:
    """Simplify a RomanNumeral figure to a common-vocabulary token.

    Operations:
    - Collapse inversion figures (6, 64, 65, 43, 42 -> handled as separate task)
    - Strip all bracketed modifiers like [add#11], [nob3], etc.
    - Remove secondary-slash parts (e.g., V/ii -> V). Tonicization is stored separately.
    - Map Neapolitan (N) to bII.
    """
    ret = str(fig)
    # Normalize common inversion encodings first
    ret = ret.replace("6/4", "64").replace("6/5", "65").replace("4/3", "43").replace("4/2", "42")
    # Remove bracketed additions/omissions of any content
    import re as _re
    ret = _re.sub(r"\[[^\]]*\]", "", ret)
    # Remove secondary relations: keep only the primary RN before first '/'
    if "/" in ret:
        ret = ret.split("/")[0]
    # Handle N -> bII
    ret = ret.replace("N", "bII")
    # Remove inversion numerals; keep seventh quality if present
    ret = ret.replace("65", "7").replace("43", "7").replace("64", "").replace("6", "").replace("42", "7")
    return ret


def normalize_pitch_name_to_keyboard(name: str) -> str:
    try:
        p = m21_pitch.Pitch(str(name).replace("-", "b"))
        pc = p.pitchClass
        return PREFERRED_PITCH_CLASSES[pc]
    except Exception:
        return str(name).strip().replace("-", "b").upper()


def transpose_spelling(token: str, semitones: int) -> str:
    base = normalize_pitch_name_to_keyboard(token)
    try:
        idx = PREFERRED_PITCH_CLASSES.index(base)
        return PREFERRED_PITCH_CLASSES[(idx + semitones) % 12]
    except ValueError:
        return base


def transpose_key_token(key_token: str, semitones: int) -> str:
    if not key_token:
        return key_token
    try:
        # Preserve major/minor via case
        is_minor = str(key_token)[0].islower()
        base = normalize_key_token(str(key_token))
        if semitones % 12 == 0:
            return base
        # Work with uppercase for indexing into preferred pitch classes
        letter = base[0].upper()
        # Normalize accidental spelling to keyboard set (sharps only)
        if len(base) > 1 and base[1] in ['#', 'b', '-']:
            # normalize b/- to sharp-based spelling via pitch class index
            from music21 import pitch as _m21_pitch
            try:
                p = _m21_pitch.Pitch(base.replace('-', 'b').upper())
                idx = p.pitchClass
            except Exception:
                idx = PREFERRED_PITCH_CLASSES.index(letter)
        else:
            try:
                idx = PREFERRED_PITCH_CLASSES.index(letter)
            except ValueError:
                idx = 0
        new_idx = (idx + semitones) % 12
        name = PREFERRED_PITCH_CLASSES[new_idx]
        return name.lower() if is_minor else name
    except Exception:
        return normalize_key_token(str(key_token))


def normalize_key_token(key_token: str) -> str:
    """Normalize a key token to keyboard spelling while preserving case (major/minor)."""
    if not key_token:
        return key_token
    is_minor = key_token[0].islower()
    try:
        p = m21_pitch.Pitch(str(key_token).replace("-", "b"))
        name = normalize_pitch_name_to_keyboard(p.name)
        return name.lower() if is_minor else name
    except Exception:
        # Fallback: normalize textual sharp/flat and case
        name = normalize_pitch_name_to_keyboard(str(key_token))
        return name.lower() if is_minor else name


def rotate_pcset(pcset: Sequence[int], semitones: int) -> Tuple[int, ...]:
    try:
        return tuple(sorted(((int(pc) + semitones) % 12 for pc in pcset)))
    except Exception:
        return tuple(pcset)


def canonicalize_pcset(pcset: Sequence[int]) -> Tuple[int, ...]:
    """Return a transposition-invariant canonical representative for a pitch-class set.

    We map the minimum element to 0 and wrap modulo 12.
    This is simple, stable, and sufficient for reducing vocabulary size.
    """
    try:
        if not pcset:
            return tuple()
        base = min(int(x) % 12 for x in pcset)
        return tuple(sorted(((int(x) - base) % 12 for x in pcset)))
    except Exception:
        try:
            return tuple(sorted(set(int(x) % 12 for x in pcset)))
        except Exception:
            return tuple()


def annotation_from_rntxt(rntxt_path: Path, fixed_offset: float = FIXEDOFFSET) -> pd.DataFrame:
    """Parse RomanText and build A_COLUMNS using HarmonyEvent for unified logic.

    Quality, inversion, keys, root, bass, and pcsets are computed via HarmonyEvent
    so DCML and RNText paths are consistent.
    """
    s = converter.parse(str(rntxt_path), format="romantext", forceSource=True)
    dfdict: Dict[str, List] = {col: [] for col in A_COLUMNS}
    from .harmony_utils import HarmonyEvent
    for idx, rn in enumerate(s.flat.getElementsByClass("RomanNumeral")):
        he = HarmonyEvent(float(rn.offset), rn, measure=int(getattr(rn, "measureNumber", 0)), duration=float(getattr(rn, "quarterLength", 0.0)))
        dfdict["a_offset"].append(round(float(he.abs_beat), FLOATSCALE))
        dfdict["a_measure"].append(int(he.measure))
        dfdict["a_duration"].append(round(float(he.duration or 0.0), FLOATSCALE))
        dfdict["a_annotationNumber"].append(idx)
        dfdict["a_romanNumeral"].append(simplify_rn_figure(he.figure_string))
        dfdict["a_isOnset"].append(True)
        pitchNames = [he.bass, he.root, he.root, he.root]
        pitchNames = [normalize_pitch_name_to_keyboard(n) for n in pitchNames[:4]]
        dfdict["a_pitchNames"].append(tuple(pitchNames))
        dfdict["a_bass"].append(pitchNames[0])
        dfdict["a_tenor"].append(pitchNames[1])
        dfdict["a_alto"].append(pitchNames[2])
        dfdict["a_soprano"].append(pitchNames[3])
        dfdict["a_root"].append(he.root)
        try:
            inv = int(he.rn.inversion())
        except Exception:
            inv = 0
        dfdict["a_inversion"].append(inv)
        dfdict["a_quality"].append(he.quality)
        try:
            pcs = he._strip_nonfunctional_additions(set(he.degrees))
            dfdict["a_pcset"].append(tuple(sorted(int(x) % 12 for x in pcs)))
        except Exception:
            dfdict["a_pcset"].append(tuple(sorted(set(he.degrees))))
        dfdict["a_localKey"].append(he.local_key)
        dfdict["a_tonicizedKey"].append(he.tonicized_key)
        try:
            sd, alt = he.rn.scaleDegreeWithAlteration
            dfdict["a_degree1"].append(f"{alt.modifier}{sd}" if alt else f"{sd}")
        except Exception:
            dfdict["a_degree1"].append("1")
        try:
            sec = he.rn.secondaryRomanNumeral
            if sec:
                sd2, alt2 = sec.scaleDegreeWithAlteration
                dfdict["a_degree2"].append(f"{alt2.modifier}{sd2}" if alt2 else f"{sd2}")
            else:
                dfdict["a_degree2"].append("None")
        except Exception:
            dfdict["a_degree2"].append("None")
    df = pd.DataFrame(dfdict)
    df.set_index("a_offset", inplace=True)

    firstRow = df.head(1)
    lastRow = df.tail(1)
    minOffset = float(firstRow.index.to_numpy()[0]) if len(firstRow) else 0.0
    maxOffset = float((lastRow.index + lastRow.a_duration).to_numpy()[0]) if len(lastRow) else 0.0
    newIndex = np.arange(minOffset, maxOffset, fixed_offset)
    df = df.reindex(index=df.index.union(newIndex))
    # Onset flags: True at annotation boundaries, False elsewhere
    df["a_isOnset"] = df["a_isOnset"].fillna(False)
    df.fillna(method="ffill", inplace=True)
    df = df.reindex(index=newIndex)
    return df


# ----------------------- Joint parsing helpers -----------------------

def measure_alignment_score(df: pd.DataFrame) -> pd.DataFrame:
    df["measureMisalignment"] = df.s_measure != df.a_measure
    return df


def quality_metric(df: pd.DataFrame) -> pd.DataFrame:
    df["qualityScoreNotes"] = np.nan
    df["qualityNonChordTones"] = np.nan
    df["qualityMissingChordTones"] = np.nan
    df["qualitySquaredSum"] = np.nan
    notesdf = df.explode("s_notes")
    annotations = df.a_annotationNumber.unique()
    for n in annotations:
        rows = notesdf[notesdf.a_annotationNumber == n]
        import re as _re
        scoreNotes = [_re.sub(r"\d", "", str(s)) for s in rows.s_notes]
        annotationNotes = rows.iloc[0].a_pitchNames
        missingChordTones = set(annotationNotes) - set(scoreNotes)
        nonChordTones = [sn for sn in scoreNotes if sn not in annotationNotes]
        missingScore = len(missingChordTones) / max(len(set(annotationNotes)), 1)
        nonChordScore = len(nonChordTones) / max(len(scoreNotes), 1)
        squared = (missingScore + nonChordScore) ** 2
        mask = (df.a_annotationNumber == n)
        group_len = int(mask.sum())
        if group_len > 0:
            df.loc[mask, "qualityScoreNotes"] = pd.Series([scoreNotes] * group_len, index=df.index[mask], dtype=object)
        df.loc[mask, "qualityNonChordTones"] = round(nonChordScore, 2)
        df.loc[mask, "qualityMissingChordTones"] = round(missingScore, 2)
        df.loc[mask, "qualitySquaredSum"] = round(squared, 2)
    return df


def inversion_metric(df: pd.DataFrame) -> pd.DataFrame:
    df["incongruentBass"] = np.nan
    annotationIndexes = df[df.a_isOnset].a_pitchNames.index.to_list()
    annotationBasses = df[df.a_isOnset].a_bass.to_list()
    annotationIndexes.append("end")
    ranges = [(annotationIndexes[i], annotationIndexes[i + 1], annotationBasses[i]) for i in range(len(annotationBasses))]
    import re as _re
    for start, end, annotationBass in ranges:
        slices = df[start:] if end == "end" else df[start:end].iloc[:-1]
        scoreBasses = [_re.sub(r"\d", "", c[0]) for c in slices.s_notes]
        counts = scoreBasses.count(annotationBass)
        invScore = 1.0 - counts / max(len(scoreBasses), 1)
        df.loc[slices.index, "incongruentBass"] = round(invScore, 2)
    return df


# ----------------------- Input representation encoders -----------------------

def _transpose_pitch_name(name: str, semitones: int) -> str:
    """Transpose a note name and return a keyboard-normalized spelling.

    This avoids accidentals like 'bb' or '--' that may not be parsed
    reliably across environments by music21 when re-instantiating Pitch.
    """
    try:
        p = m21_pitch.Pitch(str(name).replace('-', 'b'))
        if semitones != 0:
            p.transpose(semitones, inPlace=True)
        return normalize_pitch_name_to_keyboard(p.name)
    except Exception:
        return normalize_pitch_name_to_keyboard(str(name))


def encode_bass12(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(PITCHCLASSES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        bass = str(notes[0])
        name = _transpose_pitch_name(bass, semitones)
        pc = m21_pitch.Pitch(name).pitchClass
        arr[i, pc] = 1
    return arr


def encode_bass7(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(NOTENAMES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        bass = str(notes[0])
        name = _transpose_pitch_name(bass, semitones)
        letter = m21_pitch.Pitch(name).step
        idx = NOTENAMES.index(letter)
        arr[i, idx] = 1
    return arr


def encode_chromagram12(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(PITCHCLASSES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        for n in notes:
            name = _transpose_pitch_name(str(n), semitones)
            pc = m21_pitch.Pitch(name).pitchClass
            arr[i, pc] = 1
    return arr


def encode_chromagram7(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    arr = np.zeros((len(df.index), len(NOTENAMES)), dtype=np.int8)
    for i, notes in enumerate(df.s_notes):
        for n in notes:
            name = _transpose_pitch_name(str(n), semitones)
            letter = m21_pitch.Pitch(name).step
            idx = NOTENAMES.index(letter)
            arr[i, idx] = 1
    return arr


def encode_bass19(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    return np.concatenate([encode_bass7(df, semitones), encode_bass12(df, semitones)], axis=1)


def encode_chromagram19(df: pd.DataFrame, semitones: int = 0) -> np.ndarray:
    return np.concatenate([encode_chromagram7(df, semitones), encode_chromagram12(df, semitones)], axis=1)


_pattern_64 = [list(reversed(f"{x:06b}0")) for x in range(64)]
_pattern_64 = [[int(n) for n in row] for row in _pattern_64]
_pattern_64[0][0] = 1


def encode_measure_onset7(df: pd.DataFrame) -> np.ndarray:
    arr = np.zeros((len(df.index), len(NOTEDURATIONS)), dtype=np.int8)
    prev_measure = -1
    idx = 0
    for i, m in enumerate(df.s_measure):
        if m != prev_measure:
            idx = 0
            prev_measure = m
        pattern = _pattern_64[min(idx, len(_pattern_64) - 1)]
        arr[i] = pattern
        idx = min(idx + 1, len(_pattern_64) - 1)
    return arr


def encode_note_onset7(df: pd.DataFrame) -> np.ndarray:
    arr = np.zeros((len(df.index), len(NOTEDURATIONS)), dtype=np.int8)
    idx = 0
    for i, onset_flags in enumerate(df.s_isOnset):
        if sum(onset_flags) > 0:
            idx = 0
        pattern = _pattern_64[min(idx, len(_pattern_64) - 1)]
        arr[i] = pattern
        idx = min(idx + 1, len(_pattern_64) - 1)
    return arr


def encode_measure_note_onset14(df: pd.DataFrame) -> np.ndarray:
    return np.concatenate([encode_measure_onset7(df), encode_note_onset7(df)], axis=1)


# ----------------------- Output encoders and vocab -----------------------

def build_vocabs_from_dataset(joint_tsv_paths: List[Path], max_common_rn: int = 75, max_pcsets: int = 121) -> Dict[str, List]:
    # Collect and normalize tokens
    key_tokens: List[str] = []
    pcset_tokens: List[Tuple[int, ...]] = []
    rn_tokens: List[str] = []
    degree_tokens: List[str] = []
    quality_tokens: List[str] = []
    root_tokens: List[str] = []
    for p in joint_tsv_paths:
        df = pd.read_csv(p, sep="\t")
        if "a_pcset" in df:
            df["a_pcset"] = df["a_pcset"].apply(eval)
        # Keys: normalize to 24 (12 major, 12 minor) keyboard spellings
        if "a_localKey" in df.columns:
            key_tokens.extend([normalize_key_token(k) for k in df["a_localKey"].astype(str).tolist()])
        if "a_tonicizedKey" in df.columns:
            key_tokens.extend([normalize_key_token(k) for k in df["a_tonicizedKey"].astype(str).tolist()])
        # PC sets: canonicalize to be transposition-invariant
        if "a_pcset" in df.columns:
            pcset_tokens.extend([canonicalize_pcset(tuple(x)) for x in df["a_pcset"].tolist()])
        # Roman numerals: simplify aggressively (no brackets, no slashes, no inversions)
        if "a_romanNumeral" in df.columns:
            rn_tokens.extend([simplify_rn_figure(r) for r in df["a_romanNumeral"].astype(str).tolist()])
        # Degrees and qualities
        if "a_degree1" in df.columns:
            degree_tokens.extend([str(x) for x in df["a_degree1"].tolist()])
        if "a_degree2" in df.columns:
            degree_tokens.extend([str(x) for x in df["a_degree2"].tolist()])
        if "a_quality" in df.columns:
            quality_tokens.extend([str(x) for x in df["a_quality"].tolist()])
        if "a_root" in df.columns:
            root_tokens.extend([normalize_pitch_name_to_keyboard(x) for x in df["a_root"].astype(str).tolist()])

    # Keys vocabulary: fixed 24 keyboard spellings (major upper, minor lower)
    majors = [k for k in PREFERRED_PITCH_CLASSES]
    minors = [k.lower() for k in PREFERRED_PITCH_CLASSES]
    keys_vocab = majors + minors

    # PCSETS: take most frequent canonical sets up to max_pcsets
    from collections import Counter
    pc_counter = Counter(pcset_tokens)
    most_common_pc = [pcs for pcs, _ in pc_counter.most_common(max_pcsets)]

    # COMMON_RN: take most frequent up to max_common_rn
    rn_counter = Counter(rn_tokens)
    most_common_rn = [rn for rn, _ in rn_counter.most_common(max_common_rn)]

    vocabs = {
        "SPELLINGS": SPELLINGS,  # 12 keyboard spellings
        "KEYS": keys_vocab,      # 24 keys
        "PCSETS": most_common_pc,
        "COMMON_RN": most_common_rn,
        "NOTEDURATIONS": NOTEDURATIONS,
        "NOTENAMES": list(NOTENAMES),
        "PITCHCLASSES": PITCHCLASSES,
        # Additional label vocabularies
        "DEGREES": sorted(list(set(degree_tokens))),
        "QUALITIES": sorted(list(set(quality_tokens))),
        "INVERSIONS": [0, 1, 2, 3],
        "ROOTS": sorted(list(set(root_tokens))) or SPELLINGS,
    }
    return vocabs


def save_vocabs(vocabs: Dict[str, List], path: Path):
    with open(path, "w") as f:
        json.dump(vocabs, f)


def load_vocabs(path: Path) -> Dict[str, List]:
    with open(path, "r") as f:
        return json.load(f)


def encode_output_series(series: Sequence, class_list: List) -> np.ndarray:
    idxs = np.array([class_list.index(x) if x in class_list else 0 for x in series], dtype=np.int64)
    return idxs
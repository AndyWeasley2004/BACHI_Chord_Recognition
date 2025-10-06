import os
import csv
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import math

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

import matplotlib.pyplot as plt
import numpy as np


# --------------------------
# Data structures
# --------------------------


@dataclass
class ChordEvent:
    start: float
    end: float
    root: str
    quality: str
    bass: str


# --------------------------
# Pitch and quality utilities
# --------------------------


PITCH_TO_INT: Dict[str, int] = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

INT_TO_PITCH: Dict[int, str] = {v: k for k, v in PITCH_TO_INT.items()}


def normalize_note_to_sharp(note: str) -> str:
    """Normalize a pitch name to sharp-only spelling used by labels.

    - Handles flats, double-accidentals where musically sensible, and enharmonics like E# -> F.
    - Returns 'N' unchanged for no-chord.
    """
    if not note:
        return note
    note = note.strip()
    if note.upper() == "N":
        return "N"

    # Basic letter and optional accidentals
    base = note[0].upper()
    acc = note[1:]

    # Map some common enharmonics explicitly first
    enharmonic_map = {
        "Cb": "B",
        "B#": "C",
        "E#": "F",
        "Fb": "E",
    }
    if note in enharmonic_map:
        return enharmonic_map[note]

    # Convert flats and sharps to semitone offset
    semitone = PITCH_TO_INT.get(base)
    if semitone is None:
        return note  # Unknown token; return as-is

    i = 0
    while i < len(acc):
        if acc[i] == 'b':
            semitone -= 1
            i += 1
        elif acc[i] == '#':
            semitone += 1
            i += 1
        else:
            # Unexpected extra (e.g., numbers); stop
            break
    semitone %= 12
    return INT_TO_PITCH[semitone]


def parse_pred_symbol(symbol: str) -> Tuple[str, str, Optional[str]]:
    """Parse predicted chord symbol like 'E:maj/3', 'B:sus4(b7)', 'N'.

    Returns (root, quality_raw, inversion_token or None).
    inversion token can be like '3', 'b3', '5', '#5', 'B' (absolute), etc.
    """
    symbol = symbol.strip()
    if symbol.upper() == 'N':
        return 'N', 'N', None

    # Split by ':' into root and rest
    if ':' in symbol:
        root, rest = symbol.split(':', 1)
    else:
        # If no ':', treat entire as root
        root, rest = symbol, ''

    inversion = None
    if '/' in rest:
        qual, inversion = rest.split('/', 1)
    else:
        qual = rest
    return root.strip(), qual.strip(), (inversion.strip() if inversion else None)


def quality_to_triad_base(quality_raw: str) -> str:
    """Infer underlying triad quality for inversion computation: one of {'M','m','o','+'} or 'other'."""
    if not quality_raw:
        return 'M'
    q = quality_raw.lower()
    if q == 'n':
        return 'other'

    # Identify half-diminished and diminished
    if 'm7b5' in q or 'ø' in q or 'hdim' in q or '/o' in q:
        return 'o'
    if 'dim' in q or 'o7' in q or q == 'o':
        return 'o'

    if 'aug' in q or '+' in q:
        return '+'

    if 'min' in q or q.startswith('m'):
        return 'm'

    if 'maj' in q or q.startswith('m7') is False:
        # Default to major triad for ambiguous dominant types
        if '7' in q or '9' in q or '11' in q or '13' in q:
            return 'M'
        if 'maj' in q:
            return 'M'
    return 'M'


def map_pred_quality_to_label(quality_raw: str) -> str:
    """Map diverse predicted quality tokens to label vocabulary.

    Returns one of label qualities like 'M','m','o','o7','/o7','D7','M7','m7','+','+7','other','N'.
    """
    if not quality_raw:
        return 'other'
    q = quality_raw.strip()
    if q.upper() == 'N':
        return 'N'
    ql = q.lower()

    # Remove content in parentheses for base-class checks (keep info for 7/9 decisions before removing)
    ql_no_paren = ql.split('(')[0]

    # Half-diminished
    if 'm7b5' in ql or 'ø' in ql or 'hdim' in ql or '/o7' in ql:
        return '/o7'

    # Diminished
    if 'dim7' in ql:
        return 'o7'
    if 'dim' in ql :
        return 'o'

    # Augmented
    if ('aug' in ql) or ('+' in ql and '+7' not in ql):
        # Check augmented seventh
        if '7' in ql:
            return '+7'
        return '+'

    # Minor-major seventh (mM7) must be checked BEFORE generic maj7 contains check
    if 'minmaj7' in ql:
        return 'mM7'

    # Major family
    if 'maj7' in ql:
        return 'M7'
    # Any maj with extensions beyond 7 -> other (e.g., maj9)
    if 'maj9' in ql or 'maj11' in ql or 'maj13' in ql or '6' in ql:
        return 'other'
    if 'maj' in ql:
        return 'M'

    # Minor family
    if 'min7' in ql or ql == 'm7':
        return 'm7'
    if 'min' in ql or (ql.startswith('m') and ql not in {'m7', 'm7b5'}):
        # e.g., min(9) -> main class 'min'
        if any(ext in ql for ext in ['9', '11', '13', '6']):
            return 'm'  # map to main class per instruction
        return 'm'

    # Dominant family: '7', '9', '11', '13' without maj/min modifiers -> D7
    if ql in {'7', 'dom7'} or any(ext in ql for ext in ['9', '11', '13']):
        return 'D7'

    # Suspended -> map to other
    if 'sus' in ql:
        return 'other'

    # Fallbacks
    return 'other'


def add_semitones_to_root(root: str, semitones: int) -> str:
    if root.upper() == 'N':
        return 'N'
    r = normalize_note_to_sharp(root)
    pc = PITCH_TO_INT[r]
    return INT_TO_PITCH[(pc + semitones) % 12]


def parse_inversion_to_bass(root: str, quality_raw: str, inversion: Optional[str]) -> str:
    """Compute absolute bass note from inversion token.

    - Accepts explicit absolute note in inversion (e.g., '/B').
    - Relative degrees like '3','b3','#5','5','7','b7' are supported.
    - Uses inferred triad base ('M','m','o','+') for generic '3'/'5'.
    - If inversion is None, bass is root.
    """
    if root.upper() == 'N':
        return 'N'
    if not inversion or inversion.upper() == 'N':
        return normalize_note_to_sharp(root)

    inv = inversion.strip()
    # Absolute pitch provided
    if len(inv) >= 1 and inv[0].upper() in 'ABCDEFG' and (len(inv) == 1 or inv[1] in {'#', 'b'}):
        return normalize_note_to_sharp(inv)

    base = quality_to_triad_base(quality_raw)
    # Defaults for 3rd and 5th based on triad base
    third = 4
    fifth = 7
    if base == 'm' or base == 'o':
        third = 3
    if base == 'o':
        fifth = 6
    if base == '+':
        third = 4
        fifth = 8

    # Parse degree with accidentals
    accidental = 0
    degree_str = inv
    while degree_str and degree_str[0] in {'b', '#'}:
        if degree_str[0] == 'b':
            accidental -= 1
        else:
            accidental += 1
        degree_str = degree_str[1:]

    # Map degree to semitones
    if degree_str == '3':
        semi = third + accidental
    elif degree_str == '5':
        semi = fifth + accidental
    elif degree_str == '7':
        # Generic 7th: dominant/minor 7th relative to major triad baseline
        semi = 10 + accidental
    else:
        # Unrecognized token; default to root
        return normalize_note_to_sharp(root)
    return add_semitones_to_root(root, semi)


# --------------------------
# Parsing prediction and label files
# --------------------------


def parse_predictions_file(path: str) -> List[Tuple[str, float]]:
    """Read finalized_chord.txt and return list of (symbol, duration_qb).

    The format varies; take the first whitespace-separated token as symbol and the last as duration.
    Skip empty or malformed lines.
    """
    results: List[Tuple[str, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            symbol = parts[0]
            # Find last numeric token
            dur_token = None
            for token in reversed(parts[1:]):
                try:
                    float(token)
                    dur_token = token
                    break
                except Exception:
                    continue
            if dur_token is None:
                continue
            try:
                duration = float(dur_token)
            except Exception:
                continue
            if duration <= 0:
                continue
            results.append((symbol, duration))
    return results


def parse_labels_csv(path: str) -> List[Tuple[float, str, str, str]]:
    """Read chord_symbol.csv and return list of (offset_qb, root, quality, bass)."""
    rows: List[Tuple[float, str, str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                off = float(r['offset_qb'])
            except Exception:
                continue
            root = (r.get('root') or '').strip()
            quality = (r.get('quality') or '').strip()
            bass = (r.get('bass') or '').strip()
            rows.append((off, root, quality, bass))
    rows.sort(key=lambda x: x[0])
    return rows


# --------------------------
# Alignment and evaluation
# --------------------------


def build_pred_events(pred_pairs: List[Tuple[str, float]]) -> List[ChordEvent]:
    events: List[ChordEvent] = []
    t = 0.0
    for symbol, dur in pred_pairs:
        root_raw, qual_raw, inversion = parse_pred_symbol(symbol)
        root_norm = normalize_note_to_sharp(root_raw)
        qual_label = map_pred_quality_to_label(qual_raw)
        bass_norm = parse_inversion_to_bass(root_raw, qual_raw, inversion)
        bass_norm = normalize_note_to_sharp(bass_norm)
        if root_norm == 'N':
            qual_label = 'N'
            bass_norm = 'N'
        events.append(ChordEvent(start=t, end=t + dur, root=root_norm, quality=qual_label, bass=bass_norm))
        t += dur
    return events


def build_label_events(label_rows: List[Tuple[float, str, str, str]], end_time: float) -> List[ChordEvent]:
    events: List[ChordEvent] = []
    n = len(label_rows)
    for i, (off, root, quality, bass) in enumerate(label_rows):
        start = off
        if i < n - 1:
            end = label_rows[i + 1][0]
        else:
            end = end_time
        if end <= start:
            continue
        root_norm = normalize_note_to_sharp(root)
        bass_norm = normalize_note_to_sharp(bass)
        qual = quality.strip() if quality else ''
        if root_norm == 'N':
            qual = 'N'
            bass_norm = 'N'
        events.append(ChordEvent(start=start, end=end, root=root_norm, quality=qual, bass=bass_norm))
    return events


def align_and_score(pred_events: List[ChordEvent], label_events: List[ChordEvent]) -> Dict[str, Any]:
    i = 0
    j = 0
    total = 0.0
    correct_root = 0.0
    correct_quality = 0.0
    correct_bass = 0.0
    correct_all = 0.0

    root_confusion: Dict[str, Dict[str, float]] = {}
    qual_confusion: Dict[str, Dict[str, float]] = {}

    while i < len(pred_events) and j < len(label_events):
        p = pred_events[i]
        g = label_events[j]
        overlap_start = max(p.start, g.start)
        overlap_end = min(p.end, g.end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            total += overlap
            # Accuracies
            if p.root == g.root:
                correct_root += overlap
            if p.quality == g.quality:
                correct_quality += overlap
            if p.bass == g.bass:
                correct_bass += overlap
            if p.root == g.root and p.quality == g.quality and p.bass == g.bass:
                correct_all += overlap

            # Confusions (time-weighted)
            root_confusion.setdefault(g.root, {}).setdefault(p.root, 0.0)
            root_confusion[g.root][p.root] += overlap

            qual_confusion.setdefault(g.quality, {}).setdefault(p.quality, 0.0)
            qual_confusion[g.quality][p.quality] += overlap

        # Advance pointer that ends first
        if p.end <= g.end + 1e-9:
            i += 1
        else:
            j += 1

    return {
        'total': total,
        'correct_root': correct_root,
        'correct_quality': correct_quality,
        'correct_bass': correct_bass,
        'correct_all': correct_all,
        'root_confusion': root_confusion,
        'qual_confusion': qual_confusion,
    }


def merge_confusions(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {k: v.copy() for k, v in a.items()}
    for true_cls, mp in b.items():
        if true_cls not in out:
            out[true_cls] = mp.copy()
        else:
            for pred_cls, dur in mp.items():
                out[true_cls][pred_cls] = out[true_cls].get(pred_cls, 0.0) + dur
    return out


def confusion_to_matrix(confusion: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str], List[str]]:
    true_classes = sorted(confusion.keys())
    pred_classes = sorted({p for m in confusion.values() for p in m.keys()})
    mat = np.zeros((len(true_classes), len(pred_classes)), dtype=float)
    for i, t in enumerate(true_classes):
        row = confusion.get(t, {})
        for j, p in enumerate(pred_classes):
            mat[i, j] = row.get(p, 0.0)
    return mat, true_classes, pred_classes


def plot_confusion_percentage(mat: np.ndarray, true_labels: List[str], pred_labels: List[str], title: str, out_path: str) -> None:
    # Row-normalize to percentages
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(row_sums > 0, mat / row_sums * 100.0, 0.0)
        ratio = np.where(row_sums > 0, mat / row_sums, 0.0)

    # Build annotation strings as 0.xx with two decimals
    annot = np.empty_like(ratio, dtype=object)
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            annot[i, j] = f"{ratio[i, j]:.2f}"

    plt.figure(figsize=(max(8, 0.5 * len(pred_labels)), max(6, 0.5 * len(true_labels))))
    if sns is not None:
        ax = sns.heatmap(pct, annot=annot, fmt='s', cmap='Blues', cbar_kws={'label': 'Percentage'})
    else:  # pragma: no cover
        ax = plt.imshow(pct, aspect='auto', cmap='Blues')  # type: ignore
        plt.colorbar(label='Percentage')
        # Manual annotations
        for i in range(ratio.shape[0]):
            for j in range(ratio.shape[1]):
                plt.text(j, i, annot[i, j], ha='center', va='center', color='black')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=np.arange(len(pred_labels)), labels=pred_labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(true_labels)), labels=true_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_all(test_set_dir: str, out_dir: str) -> None:
    piece_metrics: List[Dict[str, float]] = []
    global_total = 0.0
    global_correct_all = 0.0
    global_root_conf: Dict[str, Dict[str, float]] = {}
    global_qual_conf: Dict[str, Dict[str, float]] = {}

    # Traverse piece directories
    for entry in sorted(os.listdir(test_set_dir)):
        piece_dir = os.path.join(test_set_dir, entry)
        if not os.path.isdir(piece_dir):
            continue

        pred_path = os.path.join(piece_dir, 'finalized_chord.txt')
        label_path = os.path.join(piece_dir, 'chord_symbol.csv')
        if not (os.path.exists(pred_path) and os.path.exists(label_path)):
            continue

        pred_pairs = parse_predictions_file(pred_path)
        if not pred_pairs:
            continue
        pred_events = build_pred_events(pred_pairs)
        pred_end = pred_events[-1].end if pred_events else 0.0

        label_rows = parse_labels_csv(label_path)
        if not label_rows:
            continue
        # Build label events up to prediction end time to keep a consistent evaluation window
        label_events = build_label_events(label_rows, end_time=pred_end)
        if not label_events:
            continue

        stats = align_and_score(pred_events, label_events)

        total = stats['total']
        if total <= 0:
            continue
        piece_result = {
            'piece': entry,
            'acc_root': stats['correct_root'] / total,
            'acc_quality': stats['correct_quality'] / total,
            'acc_bass': stats['correct_bass'] / total,
            'acc_all': stats['correct_all'] / total,
            'total': total,
        }
        piece_metrics.append(piece_result)

        global_total += total
        global_correct_all += stats['correct_all']
        global_root_conf = merge_confusions(global_root_conf, stats['root_confusion'])
        global_qual_conf = merge_confusions(global_qual_conf, stats['qual_confusion'])

    # Aggregate
    macro_root = float(np.mean([m['acc_root'] for m in piece_metrics])) if piece_metrics else 0.0
    macro_quality = float(np.mean([m['acc_quality'] for m in piece_metrics])) if piece_metrics else 0.0
    macro_bass = float(np.mean([m['acc_bass'] for m in piece_metrics])) if piece_metrics else 0.0
    macro_all = float(np.mean([m['acc_all'] for m in piece_metrics])) if piece_metrics else 0.0
    absolute_duration_acc = (global_correct_all / global_total) if global_total > 0 else 0.0

    os.makedirs(out_dir, exist_ok=True)

    # Write summary
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Rule-based chord evaluation\n')
        f.write('===========================\n')
        f.write(f'Pieces evaluated: {len(piece_metrics)}\n')
        f.write(f'Absolute duration accuracy (overall): {absolute_duration_acc:.4f}\n')
        f.write('\n')
        f.write('Macro-averages (per-piece, then averaged):\n')
        f.write(f'- Root: {macro_root:.4f}\n')
        f.write(f'- Quality: {macro_quality:.4f}\n')
        f.write(f'- Bass/Inversion: {macro_bass:.4f}\n')
        f.write(f'- Whole chord: {macro_all:.4f}\n')
        f.write('\n')
        f.write('Per-piece accuracies:\n')
        for m in piece_metrics:
            f.write(f"{m['piece']}: root={m['acc_root']:.4f}, quality={m['acc_quality']:.4f}, bass={m['acc_bass']:.4f}, chord={m['acc_all']:.4f}, duration={m['total']:.2f}\n")

    # Confusion matrices
    if global_root_conf:
        mat, t_labels, p_labels = confusion_to_matrix(global_root_conf)
        plot_confusion_percentage(
            mat,
            t_labels,
            p_labels,
            title='Root confusion (time-weighted %)',
            out_path=os.path.join(out_dir, 'confusion_root.png'),
        )

    if global_qual_conf:
        mat, t_labels, p_labels = confusion_to_matrix(global_qual_conf)
        plot_confusion_percentage(
            mat,
            t_labels,
            p_labels,
            title='Quality confusion (time-weighted %)',
            out_path=os.path.join(out_dir, 'confusion_quality.png'),
        )


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_dir = os.path.join(base_dir, 'pop909_test')
    out_dir = os.path.join(base_dir, 'pop909 rule-based eval')
    evaluate_all(test_set_dir, out_dir)


if __name__ == '__main__':
    main()



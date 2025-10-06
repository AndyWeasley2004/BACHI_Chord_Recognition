import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import miditoolkit

from .harmony_utils import parse_rntxt_m21, parse_dcml
from .data_utils import PREFERRED_PITCH_CLASSES
from .augnet_utils import (
    annotation_from_rntxt,
    parse_score,
    measure_alignment_score,
    quality_metric,
    inversion_metric,
    FIXEDOFFSET,
    FLOATSCALE,
    DATASETSUMMARYFILE,
    simplify_rn_figure,
    normalize_pitch_name_to_keyboard,
)
from .texturizers import apply_texture_template, available_durations, available_number_of_notes


def load_test_list(test_json: Path) -> List[str]:
    with open(test_json, "r") as f:
        return json.load(f)


def guess_pair_for(piece_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    score = None
    ann = None
    for f in piece_dir.iterdir():
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf in [".mxl", ".musicxml", ".xml", ".mid", ".midi"] and score is None:
            score = f
        if suf in [".txt", ".tsv"] and ann is None:
            ann = f
    return score, ann


def events_to_aug_df(events) -> pd.DataFrame:
    # Build a minimal AugNet-like annotation DataFrame from HarmonyEvent events
    from .augnet_utils import A_COLUMNS
    from .augnet_utils import simplify_rn_figure, normalize_pitch_name_to_keyboard, FIXEDOFFSET, FLOATSCALE

    d = {col: [] for col in A_COLUMNS}
    if not events:
        return pd.DataFrame(columns=A_COLUMNS)
    for idx, he in enumerate(events):
        beat = getattr(he, "abs_beat", None)
        if beat is None and isinstance(he, (list, tuple)):
            beat = he[0]
        d["a_offset"].append(round(float(beat), FLOATSCALE))
        d["a_annotationNumber"].append(idx)
        d["a_isOnset"].append(True)
        rn_fig = getattr(he, "figure_string", "N")
        rn_simple = simplify_rn_figure(rn_fig)
        root = normalize_pitch_name_to_keyboard(he.root)
        bass = normalize_pitch_name_to_keyboard(he.bass)
        pitch_names = (bass, root, root, root)
        d["a_romanNumeral"].append(rn_simple)
        d["a_pitchNames"].append(pitch_names)
        d["a_bass"].append(pitch_names[0])
        d["a_tenor"].append(pitch_names[1])
        d["a_alto"].append(pitch_names[2])
        d["a_soprano"].append(pitch_names[3])
        d["a_root"].append(root)
        # Inversion from HarmonyEvent's rn; 0 if unavailable
        try:
            inv = int(getattr(he, "rn", None).inversion()) if getattr(he, "rn", None) else 0
        except Exception:
            inv = 0
        d["a_inversion"].append(inv)
        # Use HarmonyEvent quality logic
        d["a_quality"].append(he.quality)
        # PC set: non-functional stripped degrees if available
        try:
            pcs = he._strip_nonfunctional_additions(set(getattr(he, "degrees", [])))
            d["a_pcset"].append(tuple(sorted(int(x) % 12 for x in pcs)))
        except Exception:
            d["a_pcset"].append(tuple(sorted(getattr(he, "degrees", [])[:4])))
        d["a_localKey"].append(getattr(he, "local_key", "C"))
        d["a_tonicizedKey"].append(getattr(he, "tonicized_key", getattr(he, "local_key", "C")))
        # Degrees
        try:
            sd, alt = he.rn.scaleDegreeWithAlteration
            d["a_degree1"].append(f"{alt.modifier}{sd}" if alt else f"{sd}")
        except Exception:
            d["a_degree1"].append("1")
        try:
            sec = he.rn.secondaryRomanNumeral
            if sec:
                sd2, alt2 = sec.scaleDegreeWithAlteration
                d["a_degree2"].append(f"{alt2.modifier}{sd2}" if alt2 else f"{sd2}")
            else:
                d["a_degree2"].append("None")
        except Exception:
            d["a_degree2"].append("None")
        measure = getattr(he, "measure", 0)
        duration = getattr(he, "duration", 0.0) or 0.0
        d["a_measure"].append(int(measure))
        d["a_duration"].append(round(float(duration), FLOATSCALE))
    df = pd.DataFrame(d)
    df.set_index("a_offset", inplace=True)
    # Reindex to fixed grid
    firstRow = df.head(1)
    lastRow = df.tail(1)
    minOffset = float(firstRow.index.to_numpy()[0]) if len(firstRow) else 0.0
    maxOffset = float((lastRow.index + lastRow.a_duration).to_numpy()[0]) if len(lastRow) else 0.0
    newIndex = np.arange(minOffset, maxOffset, FIXEDOFFSET)
    df = df.reindex(index=df.index.union(newIndex))
    # Onset flags: True at annotation boundaries, False elsewhere
    df["a_isOnset"] = df["a_isOnset"].fillna(False)
    df.fillna(method="ffill", inplace=True)
    df = df.reindex(index=newIndex)
    return df


def synthesize_score_from_annotation(adf: pd.DataFrame, texturize: bool = True) -> pd.DataFrame:
    """Synthesize a score DataFrame from annotation DataFrame using texturizers (optional)."""
    # Build a simple block-chord stream then enrich with textures
    rows = []
    def _assign_octaves(base_notes: List[str], target_n: int) -> List[str]:
        try:
            sel = list(base_notes)[:max(1, int(target_n))]
        except Exception:
            sel = list(base_notes)
        n = len(sel)
        # Default octave templates around lower=3 and upper=4
        templates = {
            1: [4],
            2: [3, 4],
            3: [3, 3, 4],
            4: [3, 3, 4, 4],
        }
        tmpl = templates.get(n, [3] * (n - 1) + [4])
        try:
            return [f"{note}{octv}" for note, octv in zip(sel, tmpl)]
        except Exception:
            return sel
    for offset, row in adf.iterrows():
        duration = float(row["a_duration"]) if "a_duration" in adf.columns else FIXEDOFFSET
        notes = list(row["a_pitchNames"]) if "a_pitchNames" in adf.columns else []
        if not notes or len(notes) < 3:
            notes = [row["a_root"], row["a_root"], row["a_root"]]
        # choose duration and templating
        if texturize:
            # pick first supported options that match duration and note count when possible
            d = duration if duration in available_durations else 1.0
            n = len(notes) if len(notes) in available_number_of_notes else 3
            notes = notes[:n]
            notes_oct = _assign_octaves(notes, n)
            # intervals for patterns
            intervs = []
            try:
                from music21 import pitch as m21_pitch, interval as m21_interval
                root = m21_pitch.Pitch(notes_oct[0])
                for nn in notes_oct[1:]:
                    intervs.append(m21_interval.Interval(root, m21_pitch.Pitch(nn)).simpleName)
            except Exception:
                intervs = []
            tdf = apply_texture_template(d, notes_oct, intervs, template_name=None)
            tdf["s_offset"] += float(offset)
            tdf["s_measure"] = int(row.get("a_measure", 1))
            rows.append(tdf)
        else:
            notes_oct = _assign_octaves(notes, len(notes))
            rows.append(pd.DataFrame({
                "s_offset": [float(offset)],
                "s_duration": [duration],
                "s_notes": [notes_oct],
                "s_intervals": [[]],
                "s_isOnset": [[True for _ in notes_oct]],
                "s_measure": [int(row.get("a_measure", 1))],
            }))
    sdf = pd.concat(rows, ignore_index=True)
    sdf.set_index("s_offset", inplace=True)
    sdf.sort_index(inplace=True)
    # Reindex to fixed grid
    firstRow = sdf.head(1)
    lastRow = sdf.tail(1)
    minOffset = float(firstRow.index.to_numpy()[0]) if len(firstRow) else 0.0
    maxOffset = float((lastRow.index + lastRow.s_duration).to_numpy()[0]) if len(lastRow) else 0.0
    newIndex = np.arange(minOffset, maxOffset, FIXEDOFFSET)
    sdf = sdf.reindex(index=sdf.index.union(newIndex))
    sdf.s_notes = sdf.s_notes.fillna(method="ffill").fillna(method="bfill")
    newCol = pd.Series([[False] * len(n) for n in sdf.s_notes.to_list()], index=sdf.index)
    sdf.s_isOnset.fillna(value=newCol, inplace=True)
    sdf.fillna(method="ffill", inplace=True)
    sdf.fillna(method="bfill", inplace=True)
    sdf = sdf.reindex(index=newIndex)
    return sdf


def build_joint_tsv(score_path: Path, ann_path: Path, synthesize_from_annotation: bool = False, texturize: bool = False) -> pd.DataFrame:
    if ann_path.suffix.lower() == ".tsv":
        events = parse_dcml(ann_path)
        adf = events_to_aug_df(events)
    else:
        # Parse RN, then re-normalize using HarmonyEvent logic to match DCML path
        adf = annotation_from_rntxt(ann_path, fixed_offset=FIXEDOFFSET)

    if synthesize_from_annotation:
        # Ignore original score and synthesize a texturized one from annotation
        sdf = synthesize_score_from_annotation(adf, texturize=texturize)
    else:
        sdf = parse_score(score_path, fixed_offset=FIXEDOFFSET)

    # Ensure common simplifications across both sources
    jointdf = pd.concat([sdf, adf], axis=1)
    jointdf.index.name = "j_offset"
    jointdf["a_isOnset"].fillna(False, inplace=True)
    jointdf.fillna(method="ffill", inplace=True)
    jointdf.fillna(method="bfill", inplace=True)
    jointdf = measure_alignment_score(jointdf)
    jointdf = quality_metric(jointdf)
    jointdf = inversion_metric(jointdf)
    return jointdf


# ----------------------- POP909 preprocessing -----------------------
POP909_TEST_JSON = "pop909_test.json"

# (major, minor, diminished, augmented, sus2, sus4)
TRIAD_NAMES = ["M", "m", "o", "+", "sus2", "sus4"]
TRIAD_DEGREES = [
    {0, 4, 7},
    {0, 3, 7},
    {0, 3, 6},
    {0, 4, 8},
    {0, 2, 7},
    {0, 5, 7},
]

# (dominant 7th, major 7th, minor 7th, half-diminished 7th, diminished 7th, minor-major 7th, augmented 7th)
SEVENTH_NAMES = ["D7", "M7", "m7", "/o7", "o7", "mM7", "+7"]
SEVENTH_DEGREES = [
    {0, 4, 7, 10},
    {0, 4, 7, 11},
    {0, 3, 7, 10},
    {0, 3, 6, 10},
    {0, 3, 6, 9},
    {0, 3, 7, 11},
    {0, 4, 8, 10},
]


def _quality_from_pitch_classes(pitch_classes) -> Tuple[int, str]:
    if not pitch_classes:
        return -1, "N"
    pcs = sorted(list(pitch_classes))
    for root_pc in pcs:
        degrees = {(pc - root_pc) % 12 for pc in pcs}
        for i, sev in enumerate(SEVENTH_DEGREES):
            if degrees == sev:
                return root_pc, SEVENTH_NAMES[i]
        for i, tri in enumerate(TRIAD_DEGREES):
            if degrees == tri:
                return root_pc, TRIAD_NAMES[i]
    return -1, "other"


def _build_score_only_midi(midi_path: Path, dest_path: Path) -> None:
    midi = miditoolkit.MidiFile(str(midi_path))
    score_track = midi.instruments[0]
    # Shift so that first note onset becomes 0
    time_shift = min(n.start for n in score_track.notes) if score_track.notes else 0
    if time_shift > 0:
        for n in score_track.notes:
            n.start -= time_shift
            n.end -= time_shift
    out = miditoolkit.MidiFile()
    out.ticks_per_beat = midi.ticks_per_beat
    out.instruments.append(score_track)
    out.dump(str(dest_path))


def _annotation_from_pop909_midi(midi_path: Path) -> pd.DataFrame:
    midi = miditoolkit.MidiFile(str(midi_path))
    tpb = midi.ticks_per_beat or 480
    # By spec: chord notes track; use last track per dataset convention
    chord_track = midi.instruments[-1]
    # Align chord times by shifting so that the score's first onset is at 0
    score_track = midi.instruments[0]
    time_shift = min(n.start for n in score_track.notes) if score_track.notes else 0
    notes_by_time: Dict[int, List[miditoolkit.Note]] = {}
    for note in chord_track.notes:
        start_adj = note.start - time_shift
        end_adj = note.end - time_shift
        # Clamp to non-negative to keep labels on/after score start
        start_adj = max(0, start_adj)
        end_adj = max(0, end_adj)
        if start_adj not in notes_by_time:
            notes_by_time[start_adj] = []
        # Create a lightweight proxy with adjusted times
        proxy = miditoolkit.Note(velocity=note.velocity, pitch=note.pitch, start=start_adj, end=end_adj)
        notes_by_time[start_adj].append(proxy)

    chord_blocks = []
    for start, notes in sorted(notes_by_time.items()):
        if not notes:
            continue
        end = max(n.end for n in notes)
        pitch_classes = {n.pitch % 12 for n in notes}
        bass_pc = min(notes, key=lambda n: n.pitch).pitch % 12
        root_pc, qual = _quality_from_pitch_classes(pitch_classes)
        if root_pc != -1:
            chord_blocks.append({
                "start": start,
                "end": end,
                "root": PREFERRED_PITCH_CLASSES[root_pc],
                "quality": qual,
                "bass": PREFERRED_PITCH_CLASSES[bass_pc],
            })

    events = []
    # Insert gap at 0 if needed
    if not chord_blocks or chord_blocks[0]["start"] > 0:
        events.append({"time": 0.0, "is_n": True})
    for i, blk in enumerate(chord_blocks):
        events.append({
            "time": blk["start"] / tpb,
            "is_n": False,
            "root": blk["root"],
            "quality": blk["quality"],
            "bass": blk["bass"],
        })
        end_qb = blk["end"] / tpb
        is_last = i == len(chord_blocks) - 1
        if not is_last:
            next_start_qb = chord_blocks[i + 1]["start"] / tpb
            if next_start_qb > end_qb:
                events.append({"time": end_qb, "is_n": True})

    events.sort(key=lambda x: x["time"])

    # Build minimal annotation DataFrame on fixed grid
    d = {"a_offset": [], "a_annotationNumber": [], "a_isOnset": [], "a_root": [], "a_quality": [], "a_bass": [], "a_duration": []}
    last_time = None
    for idx, ev in enumerate(events):
        t = float(ev["time"])
        if last_time is not None:
            d["a_duration"].append(round(t - float(d["a_offset"][-1]), FLOATSCALE))
        if ev.get("is_n", False):
            root = "N"
            quality = "N"
            bass = "N"
        else:
            root = normalize_pitch_name_to_keyboard(ev["root"])  # sharp-only
            quality = str(ev["quality"])  # keep as token
            bass = normalize_pitch_name_to_keyboard(ev["bass"])  # sharp-only
        d["a_offset"].append(round(t, FLOATSCALE))
        d["a_annotationNumber"].append(idx)
        d["a_isOnset"].append(True)
        d["a_root"].append(root)
        d["a_quality"].append(quality)
        d["a_bass"].append(bass)
        last_time = t
    # Close duration for last event with a minimal epsilon to allow grid creation
    if d["a_offset"]:
        d["a_duration"].append(round(FIXEDOFFSET, FLOATSCALE))

    adf = pd.DataFrame(d)
    adf.set_index("a_offset", inplace=True)
    # Reindex to fixed grid
    firstRow = adf.head(1)
    lastRow = adf.tail(1)
    minOffset = float(firstRow.index.to_numpy()[0]) if len(firstRow) else 0.0
    maxOffset = float((lastRow.index + lastRow.a_duration).to_numpy()[0]) if len(lastRow) else 0.0
    newIndex = np.arange(minOffset, maxOffset, FIXEDOFFSET)
    adf = adf.reindex(index=adf.index.union(newIndex))
    adf["a_isOnset"] = adf["a_isOnset"].fillna(False)
    adf.fillna(method="ffill", inplace=True)
    adf = adf.reindex(index=newIndex)
    return adf


def build_joint_tsv_pop909(midi_path: Path, tmp_dir: Path) -> pd.DataFrame:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_score = tmp_dir / f"{midi_path.stem}.score.mid"
    _build_score_only_midi(midi_path, tmp_score)
    sdf = parse_score(tmp_score, fixed_offset=FIXEDOFFSET)
    adf = _annotation_from_pop909_midi(midi_path)
    jointdf = pd.concat([sdf, adf], axis=1)
    jointdf.index.name = "j_offset"
    jointdf.fillna(method="ffill", inplace=True)
    jointdf.fillna(method="bfill", inplace=True)
    return jointdf


def _process_pop909_one(midi_path: str, out_dir: str, tmp_dir: str, split: str) -> List[dict]:
    midi_p = Path(midi_path)
    out_p = Path(out_dir)
    tmp_p = Path(tmp_dir)
    name = midi_p.stem
    joint = build_joint_tsv_pop909(midi_p, tmp_p)
    out_tsv = out_p / split / f"{name}.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    joint.to_csv(out_tsv, sep="\t")
    summary_rows = [{"collection": "pop909", "split": split, "file": name}]
    # Always add synthesized texturized variant for training, per POP909 assumption
    if split == "training":
        adf = _annotation_from_pop909_midi(midi_p)
        sdf_syn = synthesize_score_from_annotation(adf, texturize=True)
        joint_syn = pd.concat([sdf_syn, adf], axis=1)
        joint_syn.index.name = "j_offset"
        joint_syn.fillna(method="ffill", inplace=True)
        joint_syn.fillna(method="bfill", inplace=True)
        out_tsv_s = out_p / split / f"{name}_synth.tsv"
        joint_syn.to_csv(out_tsv_s, sep="\t")
        summary_rows.append({"collection": "pop909", "split": split, "file": f"{name}_synth"})
    return summary_rows


def _process_one(name: str, score: str, ann: str, split: str, out_dir: str, 
                 synth: bool, texturize: bool) -> List[dict]:
    summary_rows: List[dict] = []
    try:
        # Always original
        joint_orig = build_joint_tsv(Path(score), Path(ann), synthesize_from_annotation=False, texturize=False)
        out_tsv = Path(out_dir) / split / f"{name}.tsv"
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        joint_orig.to_csv(out_tsv, sep="\t")
        summary_rows.append({"collection": "custom", "split": split, "file": name})
        if split == "training" and synth:
            joint_synth = build_joint_tsv(Path(score), Path(ann), synthesize_from_annotation=synth, texturize=texturize)
            out_tsv_s = Path(out_dir) / split / f"{name}_synth.tsv"
            joint_synth.to_csv(out_tsv_s, sep="\t")
            summary_rows.append({"collection": "custom", "split": split, "file": f"{name}_synth"})
    except Exception as e:
        print(f"Error building joint tsv for {name}: {e}")
        return []
    return summary_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--pop909_dir", type=str, help="If provided, process POP909 MIDIs in this directory (all go to training split).")
    p.add_argument("--use_unique_collection", action="store_true")
    p.add_argument("--synthesize", action="store_true", help="Synthesize score from annotations (adds a _synth TSV in training)")
    p.add_argument("--texturize", action="store_true", help="If synthesizing, apply rhythmic/melodic textures")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = p.parse_args()

    # Conditional requirement: data_root only needed for non-POP909 mode
    if not args.pop909_dir and not args.data_root:
        raise ValueError("--data_root is required when --pop909_dir is not provided")

    data_root = Path(args.data_root) if args.data_root else Path(".")
    out_dir = Path(args.out_dir)
    (out_dir / "training").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)

    # If POP909 directory specified, process those MIDIs into training split
    if args.pop909_dir:
        pop909_dir = Path(args.pop909_dir)
        tmp_dir = out_dir / "_tmp_score"
        summary_rows: List[dict] = []
        midi_paths = sorted(list(pop909_dir.glob("*.mid"))) + sorted(list(pop909_dir.glob("*.midi")))
        # Test split from constant JSON file
        test_json = POP909_TEST_JSON
        with open(test_json, "r") as f:
            test_name_stems = set(json.load(f))
        test_name_files = {f"{stem}.mid" for stem in test_name_stems}

        futures = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for midi_path in midi_paths:
                name = midi_path.stem
                split = "validation" if (midi_path.name in test_name_files or name in test_name_stems) else "training"
                futures.append(ex.submit(_process_pop909_one, str(midi_path), str(out_dir), str(tmp_dir), split))
            for fut in tqdm(as_completed(futures), total=len(futures)):
                rows = fut.result()
                if rows:
                    summary_rows.extend(rows)
        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / DATASETSUMMARYFILE
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"Summary saved to {summary_path}")
        return

    # Default behavior: process paired score/annotation collections
    candidates: List[Tuple[str, Path, Path]] = []
    unique_root = data_root / "unique_data_collection"
    if unique_root.exists():
        for piece in unique_root.iterdir():
            if not piece.is_dir():
                continue
            score, ann = guess_pair_for(piece)
            if score and ann:
                candidates.append((piece.name, score, ann))

    test_list_path = data_root / "test_files.json"
    test_names = set(load_test_list(test_list_path))

    summary_rows: List[dict] = []
    print("Synthesize:", args.synthesize, "Texturize:", args.texturize, "Workers:", args.workers)

    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, score, ann in candidates:
            split = "validation" if name in test_names else "training"
            synth_flag = args.synthesize if split == "training" else False
            futures.append(
                ex.submit(_process_one, name, str(score), str(ann), split, str(out_dir), synth_flag, args.texturize)
            )
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                rows = fut.result()
                if rows:
                    summary_rows.extend(rows)
            except Exception as e:
                print(f"Worker failed: {e}")
                continue

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / DATASETSUMMARYFILE
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
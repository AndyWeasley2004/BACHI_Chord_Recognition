import csv
import shutil
from pathlib import Path
from typing import List, Set, Tuple
from tqdm import tqdm
from harmony_utils import parse_rntxt_m21, parse_dcml, HarmonyEvent


def _write_events_csv(
    events: List[HarmonyEvent], out_csv: Path, harmony_type: str = "decomposed"
) -> None:
    """Write *events* to *out_csv* in the simple 5-column format described above."""
    with out_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        if harmony_type == "decomposed":
            wr.writerow(
                [
                    "offset_qb",
                    "local_key",
                    "roman",
                    "root",
                    "triad",
                    "bass",
                    "seventh",
                    "ninth",
                    "eleventh",
                    "thirteenth",
                ]
            )
        else:
            wr.writerow(["offset_qb", "local_key", "roman", "root", "quality", "bass", "third"])

        prev = None
        for ev in events:
            if prev is None or prev.label != ev.label:
                prev = ev
                wr.writerow([f"{ev.abs_beat:.4f}", ev.local_key, ev.figure_string] + ev.label)


def _ensure_copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* if *dst* does not already exist."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _collect_piece_dirs(root: Path) -> Set[Path]:
    """Return a set of *relative* directories (as ``Path`` instances) that
    live under *root* and represent individual pieces in the flattened Rome
    corpus.

    The Rome corpora we work with have already been *flattened*, meaning that
    every immediate child of *root* corresponds to **one** musical piece.
    Therefore, we only need to list the direct sub-directories of *root*.

    The returned paths **must be relative** so they can be re-combined with
    different corpus roots (XML vs. MIDI) further down the pipeline.  _process_rome
    expects exactly this behaviour when it subsequently does

        xml_dir  = xml_root  / rel_dir
        midi_dir = midi_root / rel_dir

    If *root* does not exist (for instance the caller might point to a corpus
    modality that is missing on disk), we simply return an empty set so that
    the calling code can continue gracefully.
    """

    piece_dirs: Set[Path] = set()

    # Gracefully handle non-existing roots – just return empty set.
    if not root.exists():
        return piece_dirs

    # Enumerate direct children. Every directory is considered one piece.
    for child in root.iterdir():
        if child.is_dir():
            # Store the path *relative* to the corpus root so that it can be
            # reused with other modality roots (XML/MIDI).
            piece_dirs.add(Path(child.name))

    return piece_dirs


def _process_rome(
    xml_root: Path, midi_root: Path, out_root: Path, harmony_type: str = "decomposed"
) -> Tuple[int, List[str]]:
    piece_dirs: Set[Path] = _collect_piece_dirs(xml_root).union(
        _collect_piece_dirs(midi_root)
    )
    # print(list(piece_dirs)[:20])

    failures: List[str] = []
    processed = 0

    for rel_dir in tqdm(sorted(piece_dirs), desc="Rome → gross dataset"):
        piece_name = rel_dir.as_posix()  # already unique & flat in flattened corpus
        piece_out_dir = out_root / piece_name
        if not piece_out_dir.exists():
            print(f"Piece {piece_name} is not in unique collection")
            continue
        piece_out_dir.mkdir(parents=True, exist_ok=True)

        # Locate source score & analysis – prefer XML over MIDI.
        score_path: Path | None = None
        analysis_path: Path | None = None

        xml_dir = xml_root / rel_dir
        if xml_dir.exists():
            xml_scores = [
                p
                for p in xml_dir.iterdir()
                if p.suffix.lower() in (".mxl", ".xml", ".musicxml")
            ]
            if xml_scores:
                score_path = xml_scores[0]
            ana_path = xml_dir / "analysis.txt"
            if ana_path.exists():
                analysis_path = ana_path

        midi_dir = midi_root / rel_dir
        if score_path is None and midi_dir.exists():
            midi_scores = [
                p for p in midi_dir.iterdir() if p.suffix.lower() in (".mid", ".midi")
            ]
            if midi_scores:
                score_path = midi_scores[0]

        if score_path is None:
            failures.append(str(rel_dir))
            continue

        # Copy original files (score + raw harmony) to destination once.
        _ensure_copy(score_path, piece_out_dir / score_path.name)
        _ensure_copy(analysis_path, piece_out_dir / "analysis_raw.txt")

        # Parse harmony → events and dump CSV.
        events = parse_rntxt_m21(analysis_path)
        if not events:
            failures.append(str(rel_dir))
            continue
        annotation_name = "chord_symbol.csv" if harmony_type == "quality" else "chord_decompose.csv"
        _write_events_csv(
            events, piece_out_dir / annotation_name, harmony_type=harmony_type
        )
        processed += 1

    return processed, failures


def _process_dcml(
    dcml_root: Path, out_root: Path, harmony_type: str = "decomposed"
) -> Tuple[int, List[str]]:
    """Process the unified DCML corpus and export pieces into *out_root*.

    The expected directory structure is the same that we have in the repository
    (``harmonies/`` and ``musicxml/`` living under *dcml_root*).
    """
    harm_dir = dcml_root / "harmonies"
    xml_dir = dcml_root / "musicxml"
    midi_dir = dcml_root / "midi"  # fall-back

    failures: List[str] = []
    processed = 0

    for harm_path in tqdm(sorted(harm_dir.glob("*.tsv")), desc="DCML → gross dataset"):
        stem = harm_path.stem.replace(".harmonies", "")
        piece_name = stem  # unique across DCML already
        piece_out_dir = out_root / piece_name
        if not piece_out_dir.exists():
            print(f"Piece {piece_name} is not in unique collection")
            continue
        piece_out_dir.mkdir(parents=True, exist_ok=True)

        # Locate corresponding score – MusicXML preferred.
        score_path: Path | None = None
        for candidate in (
            xml_dir / f"{stem}.musicxml",
            xml_dir / f"{stem}.mxl",
        ):
            if candidate.exists():
                score_path = candidate
                break
        if score_path is None:
            midi_cand = midi_dir / f"{stem}.mid"
            if midi_cand.exists():
                score_path = midi_cand

        if score_path is None:
            failures.append(str(harm_path))
            continue

        # Copy source files.
        _ensure_copy(score_path, piece_out_dir / score_path.name)
        _ensure_copy(harm_path, piece_out_dir / "analysis_raw.tsv")

        # Parse harmony tsv.
        events = parse_dcml(harm_path)
        if not events:
            failures.append(str(harm_path))
            continue
        
        annotation_name = "chord_symbol.csv" if harmony_type == "quality" else "chord_decompose.csv"
        _write_events_csv(
            events, piece_out_dir / annotation_name, harmony_type=harmony_type
        )
        processed += 1

    return processed, failures


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the stage-1 gross dataset.")
    p.add_argument(
        "--rome-xml", type=Path, default=Path("data_root/rome_flattened_mxl")
    )
    p.add_argument(
        "--rome-midi", type=Path, default=Path("data_root/rome_flattened_mid")
    )
    p.add_argument("--dcml-root", type=Path, default=Path("data_root/dcml_unified"))
    p.add_argument("--out", type=Path, default=Path("data_root/all_data_collection"))
    p.add_argument(
        "--harmony-type",
        type=str,
        default="quality",
        choices=["decomposed", "quality"],
        help="type of harmony analysis to use",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rome_ok, rome_fail = _process_rome(
        args.rome_xml, args.rome_midi, args.out, args.harmony_type
    )
    dcml_ok, dcml_fail = _process_dcml(args.dcml_root, args.out, args.harmony_type)

    print("\nSUMMARY")
    print("Rome  :", rome_ok, "piece(s) processed;", len(rome_fail), "failure(s)")
    if rome_fail:
        print("  Failed:", ", ".join(rome_fail))
    print("DCML  :", dcml_ok, "piece(s) processed;", len(dcml_fail), "failure(s)")
    if dcml_fail:
        print("  Failed:", ", ".join(dcml_fail))

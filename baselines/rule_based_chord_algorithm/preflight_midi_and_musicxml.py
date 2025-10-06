"""
Preflight MIDI loader and MusicXML exporter.

- Safely load and normalize each MIDI file found under ../data_root/test_set/<piece>/<piece>.mid
  and overwrite the original MIDI with a cleaned version to prevent downstream loading issues.
- If a subdirectory lacks an existing music score file (*.musicxml, *.xml, *.mxl),
  export a MusicXML file named exactly <piece>.musicxml using music21.

Dependencies:
  pip install pretty_midi mido music21

Run:
  python preflight_midi_and_musicxml.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional


def find_workspace_root(current_file: str) -> Path:
    return Path(current_file).resolve().parent.parent


def get_test_set_root(workspace_root: Path) -> Path:
    return (workspace_root / "data_root" / "test_set").resolve()


def safe_load_and_normalize_midi(midi_path: Path):
    """Load a MIDI file robustly and return a PrettyMIDI instance.

    Strategy:
      1) Try pretty_midi.PrettyMIDI directly (most robust musically)
      2) If that fails, round-trip through mido (clip=True) to sanitize bytes/events,
         then parse again with pretty_midi.
    """
    try:
        import pretty_midi  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "pretty_midi is required. Please install with: pip install pretty_midi"
        ) from exc

    # First attempt: direct PrettyMIDI parse
    try:
        return pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        # Fallback: re-encode using mido with clipping, then re-parse
        try:
            from mido import MidiFile  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "mido is required for fallback parsing. Please install with: pip install mido"
            ) from exc

        try:
            midi_file = MidiFile(filename=str(midi_path), clip=True)
        except Exception as parse_exc:
            raise RuntimeError(
                f"Unable to parse MIDI even with mido: {midi_path}\n{parse_exc}"
            ) from parse_exc

        # Write to a temporary in-memory bytes buffer, then re-parse with PrettyMIDI
        temp_path = midi_path.with_suffix(".preflight.tmp.mid")
        try:
            midi_file.save(str(temp_path))
            pm = pretty_midi.PrettyMIDI(str(temp_path))
            return pm
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                # Non-fatal cleanup failure
                pass


def overwrite_midi_from_prettymidi(midi_path: Path, pm) -> None:
    """Persist the PrettyMIDI object back to the original path, normalizing structure."""
    # pretty_midi writes a clean, normalized MIDI file structure
    pm.write(str(midi_path))


def has_any_score_file(directory: Path) -> bool:
    score_suffixes = {".musicxml", ".xml", ".mxl"}
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() in score_suffixes:
            return True
    return False


def export_musicxml_if_absent(directory: Path, piece_name: str, midi_path: Path) -> Optional[Path]:
    """Export MusicXML to <directory>/<piece_name>.musicxml if no other score files exist.

    Returns the path to the written file if created, otherwise None.
    """
    if has_any_score_file(directory):
        return None

    output_path = (directory / f"{piece_name}.musicxml").resolve()

    # Enforce .musicxml extension by construction; just ensure parent exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from music21 import converter  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "music21 is required to write MusicXML. Please install with: pip install music21"
        ) from exc

    try:
        score = converter.parse(str(midi_path))
        # Write explicitly as 'musicxml' to guarantee the correct format/extension
        score.write('musicxml', fp=str(output_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to export MusicXML for {piece_name} from {midi_path}: {exc}"
        ) from exc

    return output_path


def process_test_set(root: Path) -> int:
    """Process all subdirectories in the test set.

    Returns the count of processed pieces.
    """
    processed_count = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        piece_name = child.name
        midi_path = (child / f"{piece_name}.mid").resolve()
        if not midi_path.exists():
            print(f"[skip] MIDI not found: {midi_path}")
            continue

        print(f"[process] {piece_name}")

        # Step 1: safe load and normalize MIDI, then overwrite original
        try:
            pm = safe_load_and_normalize_midi(midi_path)
            overwrite_midi_from_prettymidi(midi_path, pm)
            print(f"  - normalized MIDI: {midi_path}")
        except Exception:
            print(f"  ! failed to normalize MIDI: {midi_path}")
            traceback.print_exc()
            # Continue to next piece; XML export depends on a readable MIDI
            continue

        # Step 2: export MusicXML if none exists
        try:
            written = export_musicxml_if_absent(child, piece_name, midi_path)
            if written is not None:
                print(f"  - wrote MusicXML: {written}")
            else:
                print("  - score exists; skipping MusicXML export")
        except Exception:
            print(f"  ! failed to export MusicXML for: {piece_name}")
            traceback.print_exc()

        processed_count += 1

    return processed_count


def main(argv: list[str]) -> int:
    workspace_root = find_workspace_root(__file__)
    test_set_root = get_test_set_root(workspace_root)

    if not test_set_root.exists():
        print(f"Test set root not found: {test_set_root}")
        return 1

    print(f"Scanning: {test_set_root}")
    count = process_test_set(test_set_root)
    print(f"Done. Processed pieces: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


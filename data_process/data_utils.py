from __future__ import annotations
import pandas as pd
from pathlib import Path
import math
from typing import List, Tuple
import numpy as np
import miditoolkit
from music21 import converter, note as m21_note, chord as m21_chord, pitch, meter


PREFERRED_PITCH_CLASSES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

PITCH_CLASS_TO_INT = {pc: i for i, pc in enumerate(PREFERRED_PITCH_CLASSES)}

KEY_PITCH_DICT = {
    0: "C", 1: "C#", 2: "D", 3: "E-", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "A-", 9: "A", 10: "B-", 11: "B"
}

def transpose_label(label: List[str], shift: int) -> List[str]:
    """Transpose the *root* and *bass* of a chord label by *shift* semitones.

    The label is expected to be produced by :pyclass:`HarmonyEvent` and thus
    to contain at least the first three fields::

        [root, triad, bass, ...]

    The remaining quality/extension fields describe *intervallic* information
    and therefore remain unchanged under global transposition.
    """

    # Early exit for trivial cases ------------------------------------------------
    if shift == 0:
        return label

    if label[1] in {"N", "None", "PAD"}:
        return label

    new_parts = list(label)  # Make a mutable copy

    def _shift_pc(token: str) -> str:
        """Utility: shift a single pitch-class token; keep unknown tokens intact."""
        if token not in PITCH_CLASS_TO_INT:
            # print(f"Unknown pitch class: {token}")
            return token  # e.g. 'N'
        old_pc = PITCH_CLASS_TO_INT[token]
        new_pc = (old_pc + shift) % 12
        return PREFERRED_PITCH_CLASSES[new_pc]

    # Apply shift to root (index 0) and bass (index 2)
    new_parts[0] = _shift_pc(new_parts[0])
    new_parts[2] = _shift_pc(new_parts[2])
    if len(new_parts) > 3:
        new_parts[3] = transpose_key(new_parts[3], shift)
    # new_parts[4] = _shift_pc(new_parts[4]) # third

    return new_parts


def transpose_key(key: str, shift: int) -> str:
    """Transpose a local key string by *shift* semitones.

    The key is represented by its root's pitch class, with case indicating
    the mode: uppercase for major, lowercase for minor (e.g., 'C' -> C major,
    'c' or 'c#' -> C or C# minor).
    """
    # Early exit for trivial cases or special tokens
    if key in {"N", "None", "PAD"}:
        return key

    is_major = key[0].isupper() # major key is uppercase, minor key is lowercase
    old_pc = pitch.Pitch(key).pitchClass # get the pitch class of the key
    new_pc = (old_pc + shift) % 12 # transpose the pitch class
    new_root_str = KEY_PITCH_DICT[new_pc] # get the new pitch class

    # add case to the new pitch class -> key
    if is_major:
        return new_root_str
    else:
        return new_root_str.lower()


def resample_events(
    events: List, grid_step: float, end_time: float = None
) -> Tuple[List[List[str]], np.ndarray]:
    """Resamples HarmonyEvents to a constant grid resolution and generates chord change boundaries."""
    if not events:
        return [], np.array([])

    expanded_labels: List[List[str]] = []
    boundaries: List[int] = []
    event_idx = 0
    num_steps = math.ceil(end_time / grid_step)

    for i in range(num_steps):
        current_beat = i * grid_step

        # Check if we need to advance to the next event, 1e-6 to avoid floating point inaccuracies
        while (
            event_idx + 1 < len(events)
            and events[event_idx + 1][0] <= current_beat + 1e-6
        ):
            event_idx += 1

        current_event = events[event_idx]
        label = current_event[1]
        expanded_labels.append(label)

        # Determine if this is a chord change boundary
        if i == 0:
            boundaries.append(1)  # First frame is always a boundary
        else:
            # Check if chord changed from previous frame
            prev_label = expanded_labels[i-1]
            if label != prev_label:
                boundaries.append(1)
            else:
                boundaries.append(0)

    return expanded_labels, np.array(boundaries, dtype=np.int8)


def get_pianoroll_and_labels(
    score_path: Path,
    events: List,
    resolution: int = 12,
    label_resolution: int = 2,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    try:
        # Parse the score – support MusicXML/score files **or** plain MIDI
        notes_data: List[Tuple[int, float, float]] = []  # (midi, onset_qb, offset_qb)

        suffix = score_path.suffix.lower()
        if suffix in {".mid", ".midi"}:
            midi = miditoolkit.MidiFile(str(score_path))
            tpb = midi.ticks_per_beat or 480
            for inst in midi.instruments:
                for n in inst.notes:
                    start_qb = n.start / tpb
                    end_qb = n.end / tpb
                    notes_data.append((n.pitch, start_qb, end_qb))
        else:  # assume MusicXML / MXL / XML
            sc = converter.parse(str(score_path))
            sc.makeVoices(inPlace=True)
            for el in sc.flat.notes:
                dur = float(el.quarterLength)
                start_qb = float(el.offset)
                end_qb = start_qb + dur
                if isinstance(el, m21_note.Note):
                    notes_data.append((el.pitch.midi, start_qb, end_qb))
                elif isinstance(el, m21_chord.Chord):
                    for p in el.pitches:
                        notes_data.append((p.midi, start_qb, end_qb))

        # Sort notes by onset time
        notes_data.sort(key=lambda x: x[1])

        # use the last note's offset as the end time
        last_note_end = max(nd[2] for nd in notes_data)
        total_beats = last_note_end  # Use only the last note's end time
        total_frames = math.ceil(total_beats * resolution)

        # Pianoroll (binary 88 × T)
        pianoroll = np.zeros((88, total_frames), dtype=np.int8)
        for midi_pitch, start_b, end_b in notes_data:
            row = midi_pitch - 21  # A0 == 21
            if not (0 <= row < 88):
                continue
            s_f = max(0, math.floor(start_b * resolution))
            e_f = min(total_frames, math.ceil(end_b * resolution))
            pianoroll[row, s_f:e_f] = 1

        # Resample harmony events to fixed grid (step = 1/2 beat)
        res_labels, boundaries = resample_events(
            events, grid_step=1 / label_resolution, end_time=total_beats
        )
        labels = np.array(res_labels, dtype=object)

        # Align lengths and extend last chord label to the end
        num_label_frames = math.ceil(total_beats * label_resolution)

        if len(labels) > num_label_frames:
            labels = labels[:num_label_frames]
            boundaries = boundaries[:num_label_frames]
        elif len(labels) < num_label_frames:
            pad_len = num_label_frames - len(labels)
            if len(labels) > 0:
                # Extend the last chord label to the end
                lbl_pad = np.array([labels[-1]] * pad_len, dtype=object)
                boundary_pad = np.zeros(pad_len, dtype=np.int8)
                labels = np.vstack([labels, lbl_pad])
                boundaries = np.concatenate([boundaries, boundary_pad])
            else:
                # If there are no labels, pad with "None" and set boundaries
                print("Incomplete labels, padding with None")
                labels = np.array([["None", "None", "None", "None"]] * pad_len, dtype=object)
                boundaries = np.ones(pad_len, dtype=np.int8)  # First frame is boundary if no labels

        return pianoroll, labels, boundaries

    except Exception as e:
        print(f"Error processing {score_path.name}: {e}")
        return None, None, None


def shift_pianoroll(pianoroll: np.ndarray, shift: int) -> np.ndarray:
    """Return a semitone-shifted copy of *pianoroll* without cyclic wrap-around.

    Notes that would move below A0 or above C8 are dropped (filled with zeros).
    The input array is expected to have shape (88, T) where axis-0 index 0
    corresponds to MIDI pitch 21 (A0).
    """
    if shift == 0:
        return pianoroll.copy()

    out = np.zeros_like(pianoroll)
    if shift > 0:
        # Shift upward: rows lose the top *shift* pitches.
        out[shift:, :] = pianoroll[:-shift, :]
    else:  # shift < 0
        out[:shift, :] = pianoroll[-shift:, :]
    return out

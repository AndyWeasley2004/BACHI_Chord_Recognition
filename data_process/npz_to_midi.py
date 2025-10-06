import os
import numpy as np
import glob
from tqdm import tqdm
from music21 import stream, note
from fractions import Fraction

# --- Configuration ---
NPZ_DIR = "data_root/symbol_no_shift_npz"
OUTPUT_DIR = "data_root/symbol_no_shift_recon"
PIANO_START_MIDI_NOTE = 21
FRAMES_PER_QUARTER = 12


def npz_to_midi_and_txt(npz_path, output_path_midi, output_path_txt):
    """
    Converts a .npz file containing a pianoroll, labels, and boundaries into
    a MIDI file, a text file with chord annotations, and a text file with
    boundary annotations.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        pianoroll = data["pianoroll"]
        labels = data["labels"]
    except Exception as e:
        print(f"Skipping {os.path.basename(npz_path)} due to loading error: {e}")
        return

    # --- Create MIDI File ---
    midi_part = stream.Part()

    # Convert pianoroll to notes/chords. Collect simultaneous onsets first.
    events = {}
    for pitch_idx in range(pianoroll.shape[0]):
        midi_pitch = pitch_idx + PIANO_START_MIDI_NOTE
        binary_pianoroll_pitch = (pianoroll[pitch_idx] > 0).astype(int)
        frames = np.pad(binary_pianoroll_pitch, (1, 1), "constant")
        diffs = np.diff(frames)
        onsets = np.where(diffs == 1)[0]
        offsets = np.where(diffs == -1)[0]

        for i, onset_frame in enumerate(onsets):
            if i >= len(offsets):
                continue  # safety guard
            offset_frame = offsets[i]
            duration_frames = offset_frame - onset_frame
            if duration_frames == 0:
                continue

            events.setdefault(onset_frame, []).append((midi_pitch, duration_frames))

    # Insert events sorted by onset.
    for onset_frame in sorted(events.keys()):
        # Use exact rational representation to avoid floating-point round-off.
        offset_quarters = Fraction(int(onset_frame), FRAMES_PER_QUARTER)
        for midi_pitch, duration_frames in events[onset_frame]:
            n = note.Note(midi_pitch)
            n.duration.quarterLength = Fraction(int(duration_frames), FRAMES_PER_QUARTER)
            midi_part.insert(offset_quarters, n)

    # Write to MIDI file
    try:
        # stripTies is good practice before MIDI export
        midi_stream = midi_part.stripTies()
        midi_stream.write("midi", fp=output_path_midi)
    except Exception as e:
        print(f"Failed to write MIDI for {os.path.basename(npz_path)}: {e}")

    # --- Create Chord Annotation TXT File ---
    chord_annotations = []
    last_chord_label = None

    for frame_idx in range(len(labels)):
        offset_quarters = Fraction(int(frame_idx), 2)

        current_chord_label = tuple(labels[frame_idx])
        if current_chord_label != last_chord_label and current_chord_label[0] != "None":
            root, qual, bass = current_chord_label
            chord_text = f"{root}_{qual}_{bass}"
            chord_annotations.append(f"{float(offset_quarters)}\t{chord_text}")
            last_chord_label = current_chord_label

    # Write to TXT file
    try:
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(chord_annotations))
    except Exception as e:
        print(f"Failed to write TXT for {os.path.basename(npz_path)}: {e}")

if __name__ == "__main__":
    print("Starting conversion from NPZ to MIDI and TXT chord annotations...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    npz_file_names = [
        'Piano_Sonatas_Mozart_K311_3_shift0.npz',
        ]
    # npz_file_names = os.listdir(NPZ_DIR)
    npz_files = [os.path.join(NPZ_DIR, f) for f in npz_file_names]
    print(f"Found {len(npz_files)} files to process.")

    for npz_path in tqdm(npz_files, desc="Converting files", total=len(npz_files)):
        file_name = os.path.basename(npz_path)
        base_name = os.path.splitext(file_name)[0].replace('shift0', '')
        output_path_midi = os.path.join(OUTPUT_DIR, base_name + ".mid")
        output_path_txt = os.path.join(OUTPUT_DIR, base_name + ".txt")
        npz_to_midi_and_txt(npz_path, output_path_midi, output_path_txt)

    print("Conversion finished.")

#! /bin/bash

input_root="../rule_based_evaluation/pop909_test"
find "$input_root" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' sub; do
    base=$(basename "$sub")
    mid_file="$sub/${base}.mid"

    if [ -f "$mid_file" ]; then
            echo "Processing $base"

            # Step 1: tempo estimation
            python raw_tempo.py "$mid_file" || { echo "raw_tempo failed for $base; skipping."; continue; }

            # Step 2: melody track shifting
            python shift_melody_track.py "$mid_file" || { echo "shift_melody_track failed for $base; skipping."; continue; }

            # Step 3: timestamps
            serpent64 serpent_get_timestamp.srp "$mid_file" || { echo "serpent timestamp extraction failed for $base; skipping."; continue; }

            # Ensure timestamps were actually created
            if [ ! -f "$sub/timestamps.txt" ]; then
                echo "timestamps.txt missing for $base; skipping."
                continue
            fi

            # Step 4: chord recognition
            python exported_midi_chord_recognition/main.py "$mid_file" || { echo "chord recognition failed for $base; skipping."; continue; }

            # Step 5: finalize format
            python finalize_chord_midi_only.py "$mid_file" || { echo "finalize failed for $base; skipping."; continue; }
    else
            echo "Warning: $mid_file not found."
    fi
done
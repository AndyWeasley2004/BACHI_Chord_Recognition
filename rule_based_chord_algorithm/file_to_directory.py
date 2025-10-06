import os
import shutil
import miditoolkit

# Root folder containing composer subfolders.
source_root = '../top4_real'
# Target folder for flat output.
target_dir = '../top4_real_chords'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

index = 1

# Iterate over each composer folder in the source root.
for composer_folder in os.listdir(source_root):
    composer_path = os.path.join(source_root, composer_folder)
    if os.path.isdir(composer_path):
        # Iterate over each file in the composer's folder.
        for filename in os.listdir(composer_path):
            file_path = os.path.join(composer_path, filename)
            if os.path.isfile(file_path):
                # Create a folder with an index and composer name signature.
                folder_name = f'{index}_{composer_folder}'
                new_folder_path = os.path.join(target_dir, folder_name)
                os.makedirs(new_folder_path, exist_ok=True)
                new_filename = folder_name + '.mid'
                new_file_path = os.path.join(new_folder_path, new_filename)
                try:
                    # Load and re-save the MIDI file using miditoolkit.
                    midi_obj = miditoolkit.MidiFile(file_path)
                    midi_obj.dump(new_file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    # Fallback: copy the file as-is.
                    shutil.copy(file_path, new_file_path)
                index += 1

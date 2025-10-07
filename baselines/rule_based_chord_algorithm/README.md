
# Rule-based Preprocessing

## Dependencies
python dependencies will be checked and installed at the beginning of the preprocessing script. 

### Serpent
Our preprocessing steps make use of [Serpent](https://www.cs.cmu.edu/~music/aura/serpent-info.htm), a language for audio processing. Please follow the [installation guide](https://www.cs.cmu.edu/~music/serpent/doc/installation.htm) and install serpent64. 

## Usage

Please organize MIDI file in a sub-folder with the same name, like {filename}/{filename.mid} and then update the root folder with your own in `preprocess.sh` and run the following:

```bash
bash ./preprocessing.sh
```

### Output
Under the directory of song.mid, you'll find:
- song_shift.mid
- analyzed_key.txt
- beging_time.txt
- chord_midi.txt
- chords_summary.txt
- finalized_chord.txt
- melody.txt
- tempo.txt
- time_signature.txt
- timestamps.txt


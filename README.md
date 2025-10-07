# BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding

Official implementation of **BACHI** (Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding), a state-of-the-art model for automatic chord recognition from symbolic music scores.

[Paper]() | [Demo Page](https://andyweasley2004.github.io/BACHI.github.io/) | [POP909-CL Dataset](https://github.com/AndyWeasley2004/POP909-CL-Dataset) | [Model Weight & Data](https://huggingface.co/datasets/Itsuki-music/BACHI_Chord_Recognition)

## Environment

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

```bash
pip install -r requirements.txt
```


## Quick Start

Use the provided inference script to predict chords for new music scores with pre-trained checkpoints below:

- **Classical Model**: Trained on When-in-Rome + DCML corpus; [checkpoint](https://huggingface.co/datasets/Itsuki-music/BACHI_Chord_Recognition/tree/main/classical_film_kdec)
- **Pop Model**: Trained on POP909-CL; [checkpoint](https://huggingface.co/datasets/Itsuki-music/BACHI_Chord_Recognition/tree/main/pop909_film_kdec)

### Single File Inference

```bash
python inference.py \
    --input /path/to/score \
    --output predictions/ \
    --checkpoint_dir /path/to/checkpoint_directory
```

### Group Inference by Folder

```bash
python inference.py \
    --input /path/to/scores_directory/ \
    --output predictions/ \
    --checkpoint_dir /path/to/checkpoint_directory
```

The `--checkpoint_dir` should point to a directory containing:
- `best_model.pt` (model checkpoint)
- `config.yaml` (model configuration)
- Vocabulary file (path specified in config)

**Supported formats**: `.musicxml`, `.mxl`, `.xml`, `.mid`, `.midi`

### Output Format

Predictions are saved as text files with the format.
```
0.00 C_M_C
2.50 F_M_F
4.00 G_M_G
6.50 C_M_C
```

Each line represents a chord change with:
- Beat position (in quarter notes)
- Root note
- Quality (M=major, m=minor, D7=dominant 7th, etc.)
- Bass note (for inversions)


## Train your own model
### Data Process

BACHI requires symbolic music scores to be converted into piano roll representations with aligned chord labels. The data processing pipeline consists of several stages:

### 1. Dataset Download and Organization

#### Classical Music (When-in-Rome & DCML)

Note that `build_gross_dataset.py` will prioritize music score formats like mxl, musicxml, etc, to keep the music sheet integrity better. When it's not available in "rome_flattened_mxl", it will look for "rome_flattened_mid" for MIDI in classical. 

```bash
# Download When-in-Rome corpus
# Visit: https://github.com/MarkGotham/When-in-Rome

# Process When-in-Rome to organized format
python data_process/rome_download.py

# Download DCML corpus
# Visit: https://github.com/orgs/DCMLab/repositories

# Build unified classical dataset
python data_process/build_gross_dataset.py \
    --rome-xml data_root/rome_flattened_mxl \
    --rome-midi data_root/rome_flattened_mid \
    --dcml-root data_root/dcml_unified \
    --out data_root/all_data_collection \
    --harmony-type quality
```

For quick download our processed data, please visit [here](https://huggingface.co/datasets/Itsuki-music/BACHI_Chord_Recognition/tree/main/unique_data_collection)

#### Pop Music (POP909-CL)

Please refer to the release repository of [POP909-CL](https://github.com/AndyWeasley2004/POP909-CL-Dataset) for dataset details. For quick start, you can use the following script for quick extraction of chord labels.

```bash
# Process POP909 to organized format
python data_process/process_pop909.py \
    --pop909-root POP909_processed \
    --out data_root/all_data_collection
```

### 2. Convert to Piano Roll Format

```bash
# Convert processed dataset to piano roll representations with labels
python data_process/gross_to_pianoroll.py \
    --gross-root data_root/all_data_collection \
    --out data_root/final_data \
    --resolution 12 \
    --label-resolution 2
```

This generates `.npz` files containing:
- Piano roll representations (88 × T frames)
- Chord labels (root, quality, bass, key)
- Boundary markers

### 3. Dataset Statistics (Optional)

```bash
# View dataset statistics
python data_process/data_summary.py
```

## Training

### Configuration

Training configurations are stored in the `config/` directory. The main configuration for our best model is `config/film_kdec.yaml`:

### Train Single Model

You can adjust the dataset and vocab to use inside `config/film_kdec.yaml`

```bash
python train.py config/film_kdec.yaml 
```

### Train All Ablation Models

```bash
# Run all ablation experiments
python run_all_ablation_train.py --data-name [classical/pop909]
```

## Evaluation

### Evaluate Trained Model

```bash
# Evaluate a trained model
python eval.py /path/to/checkpoint_dir
```

This will:
- Load the model from `checkpoint_dir/best_model.pt`
- Generate predictions on the validation set
- Save detailed metrics to `checkpoint_dir/evaluation.txt`
- Save per-piece predictions to `checkpoint_dir/predictions/`

### Evaluate All Ablation Models

```bash
# Run evaluation on all ablation experiments
bash run_all_ablation_eval.sh {data-name}
```

## POP909-CL Dataset

POP909-CL is an enhanced version of POP909 with human-corrected annotations:

[Dataset download link will be provided upon publication]

## Baselines

The `baselines/` directory contains implementations and evaluation scripts for comparison methods. Detailed documentation lies inside each model inside `baselines/`

## Citation

If you use BACHI or POP909-CL in your research, please cite:

```bibtex
@inproceedings{yao2026bachi,
  title={BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding on Pop and Classical Music},
  author={Mingyang Yao and Ke Chen and Shlomo Dubnov and Taylor Berg-Kirkpatrick},
  booktitle={arxiv},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
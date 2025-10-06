# ChordGNN 

## Environment

You can config the environment for ChordGNN via

```bash
pip install -r requirements.txt
```

## Datasets

This sample project uses the `POP909` dataset and can also be configured to use datasets like `AugmentedNet`. The datasets are expected to be present locally in a specific directory structure in format of data used in `AugmentedNet`. You can use `AugmentedNet_replicate` directory for processing the data. 

### `POP909` Dataset Setup

The `POP909` dataset is not downloaded automatically. You need to obtain it and place it in the expected directory.

1.  **Download `POP909`:** Obtain the `POP909` dataset and prepare it in `.tsv` format.

2.  **Directory Structure:** The code expects the data to be in the following structure:
    ```
    <data_root>/Pop909ChordDataset/dataset/
    ├── training/
    │   ├── 1.tsv
    │   └── ...
    ├── validation/
    │   ├── 2.tsv
    │   └── ...
    └── test/
        ├── 3.tsv
        └── ...
    ```

3.  **Set Data Root:** The `<data_root>` directory is determined by the `chordgnn_DOWNLOAD_DIR` environment variable. If this variable is not set, it defaults to `/graft1/datasets/`. You can set the environment variable to your preferred location:
    ```bash
    export chordgnn_DOWNLOAD_DIR=/path/to/your/datasets
    ```

Once the data is in the correct location, you can train a model on `POP909` by specifying it as the data version:

```bash
python chordgnn/train/chord_prediction.py --data_version pop909 --gpus 0 --batch_size 64 --include_synth [--skip_processing (when you run the script not in the first time)]
```

### Other Collections

The codebase also supports other collections for the `AugmentedNet` dataset. You should be able to replace `POP909` with any other dataset

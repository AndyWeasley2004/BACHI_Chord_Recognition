# AugmentedNet Replication (PyTorch)

This folder contains a reimplementation of AugmentedNet using PyTorch, with a generalized preprocessing pipeline that supports multiple score and annotation formats and produces inspectable intermediate files.

## Key points
- Input/output representations, targets, and texturization follow the original AugmentedNet paper and code.
- Supports scores: .mxl, .musicxml, .xml, .mid.
- Supports annotations: RomanText .txt and DCML-like .tsv.
- Intermediate artifacts: per-piece joint TSVs and a dataset summary TSV.
- Train/test split uses `data_root/test_files.json`.
- Synthetic texturization can be generated similarly to the original.
- The prediction field naming could be a little confusing, for cleaning usage, you can remove the number after the text (like "root35" to "root" for clarity)

## Quickstart

You can use the same environment as our BACHI.

1) Preprocess pairs into inspectable TSVs:

```bash
python -m AugmentedNet_replicate.preprocess   --pop909_dir "Path of dataset"   --out_dir  "AugmentedNet_replicate/{output_tsv_path}"  --workers 8 --synthesize --texturize
```

2) Encode to NPZ (inputs/outputs):

```bash
python -m AugmentedNet_replicate.encode_npz \
  --tsv_dir "Path of output tsv in last step" \
  --npz_out "AugmentedNet_replicate/{dataset}_npz/dataset.npz" \
  --with_synth  # optional, adds synthetic texturizations
```

3) Train:

```bash
python -m AugmentedNet_replicate.train  --npz "AugmentedNet_replicate/{dataset}_npz/dataset.npz"  --out_dir "AugmentedNet_replicate/{dataset}_ckpt"  --epochs 50 --batch_size 16 --lr_boundaries 40 --lr_values 0.001 0.001
```

4) Evaluate and Inference:


```bash
python -m AugmentedNet_replicate.evaluate --npz "/.../dataset.npz" --model "/.../best.pth" # For evaluation
python -m AugmentedNet_replicate.infer --model "/.../best.pth" --input "/path/to/score.musicxml" # For inference
```
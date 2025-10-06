import os
from pathlib import Path
import numpy as np
import pandas as pd


def get_shift0_npz_files(npz_root: str, suffix: str = "shift0.npz"):
    """Yield absolute paths to *shift0* npz files inside *npz_root*."""

    for filename in os.listdir(npz_root):
        if filename.endswith(suffix):
            yield os.path.join(npz_root, filename)


def count_dataset_distributions(npz_root: str):
    from collections import Counter

    root_counter = Counter()
    quality_counter = Counter()
    bass_counter = Counter()
    key_counter = Counter()
    
    for npz_path in get_shift0_npz_files(npz_root, suffix=".npz"):
        labels = np.load(npz_path, allow_pickle=True)["labels"].tolist()

        for chord_list in labels:
            root, triad, bass, key = chord_list
            root_counter[root] += 1
            quality_counter[triad] += 1
            bass_counter[bass] += 1
            key_counter[key] += 1

    return root_counter, quality_counter, bass_counter, key_counter


if __name__ == "__main__":
    npz_path = "data_root/symbol_with_key"
    root_cnt, quality_cnt, bass_cnt, key_cnt = count_dataset_distributions(npz_path)


    print("\n--- Root distribution (top 20) ---")
    cnt_dict = dict(root_cnt.most_common(20))
    print(cnt_dict)

    print("\n--- Quality distribution (top 20) ---")
    cnt_dict = dict(quality_cnt.most_common(20))
    print(cnt_dict)

    print("\n--- Bass distribution (top 20) ---")
    cnt_dict = dict(bass_cnt.most_common(20))
    print(cnt_dict)

    print("\n--- Key distribution (top 30) ---")
    cnt_dict = dict(key_cnt.most_common(30))
    print(cnt_dict)
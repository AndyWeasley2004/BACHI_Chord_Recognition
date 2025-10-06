import pickle

CHORD_QUALITIES = {
    "M",
    "m",
    "o",
    "+",
    "D7",
    "M7",
    "m7",
    "o7",
    "/o7",
    "mM7",
    "+7",
    "sus2",
    "sus4",
    "other",
    "N",
    "PAD",
}

ROOT_PITCHES = {
    "A",
    "A#",
    "B",
    "B#",
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "N",
    "PAD",
}

KEY2SYMBOL = {
    "C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B",
    "c", "c#", "d", "e-", "e", "f", "f#", "g", "a-", "a", "b-", "b",
    "N",
    "PAD",
}


def build_vocab():
    root_to_idx = {r: i for i, r in enumerate(sorted(list(ROOT_PITCHES)))}
    quality_to_idx = {q: i for i, q in enumerate(sorted(list(CHORD_QUALITIES)))}
    key_to_idx = {q: i for i, q in enumerate(sorted(list(KEY2SYMBOL)))}
    return root_to_idx, quality_to_idx, key_to_idx


if __name__ == "__main__":
    root_to_idx, quality_to_idx, key_to_idx = \
                        build_vocab()
    vocab_data = {"root_to_idx": root_to_idx, 
                  "quality_to_idx": quality_to_idx, 
                  "key_to_idx": key_to_idx,
                  }
    vocab_path = "data_root/pop909_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab_data, f)

    print(f"Saved vocabulary to {vocab_path}")
    print("--- Vocabulary Summary ---")
    print(f"Root pitches: {root_to_idx}")
    print(f"Chord qualities: {quality_to_idx}")
    print(f"Keys: {key_to_idx}")
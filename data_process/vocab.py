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
# INTERVAL2SYMBOL = {'433': '7', '434': 'M7', '343': 'm7', '333': 'o7', '334': '/o7',
#                    '43': 'M', '34': 'm', '33': 'o', '44': '+',
#                    '25': 'sus2', '52': 'sus4', 'temp': 'N'}
# CHORD_QUALITIES = {
#     "maj",
#     "min",
#     "min7",
#     "7",
#     "maj7",
#     "sus4",
#     "maj6",
#     "other",
#     "dim",
#     "sus2",
#     "maj9",
#     "min9",
#     "aug",
#     "9",
#     "hdim7",
#     "min6",
#     "dim7",
#     "None",
#     "PAD",
# }
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

SEVENTH = {
    "7",
    "b7",
    "bb7",
    "N",
    "PAD",
}

NINTH = {
    "9",
    "b9",
    "#9",
    "N",
    "PAD",
}

ELEVENTH = {
    "11",
    "#11",
    "N",
    "PAD",
}

THIRTEENTH = {
    "13",
    "b13",
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
                #   "seventh_to_idx": seventh_to_idx,
                #   "ninth_to_idx": ninth_to_idx, 
                #   "eleventh_to_idx": eleventh_to_idx,
                #   "thirteenth_to_idx": thirteenth_to_idx
                  }
    vocab_path = "data_root/pop909_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab_data, f)

    print(f"Saved vocabulary to {vocab_path}")
    print("--- Vocabulary Summary ---")
    print(f"Root pitches: {root_to_idx}")
    print(f"Chord qualities: {quality_to_idx}")
    print(f"Keys: {key_to_idx}")
    # print(f"Sevenths: {seventh_to_idx}")
    # print(f"Ninth: {ninth_to_idx}")
    # print(f"Eleventh: {eleventh_to_idx}")
    # print(f"Thirteenth: {thirteenth_to_idx}")
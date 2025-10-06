import numpy as np
from chordgnn.utils import load_score_hgraph, hetero_graph_from_note_array, select_features, HeteroScoreGraph
from chordgnn.utils import time_divided_tsv_to_part
from chordgnn.models.core import positional_encoding
from chordgnn.utils.chord_representations import available_representations
import torch
import os
from chordgnn.data.dataset import BuiltinDataset, chordgnnDataset
from joblib import Parallel, delayed
from tqdm import tqdm
import random


class AugmentedNetChordDataset(BuiltinDataset):
    r"""The AugmentedNet Chord Dataset.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if AugmentedNet Chord Dataset scores are already available otherwise it will download it.
        Default: ~/.chordgnn/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, is_zip=True):
        url = "https://github.com/napulen/AugmentedNet/releases/download/v1.0.0/dataset.zip"
        super(AugmentedNetChordDataset, self).__init__(
            name="AugmentedNetChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        """Override to avoid remote download when local dataset is present.

        If a local dataset exists at ``<raw_dir>/LocalChordDataset/dataset``, we
        simply ensure ``raw_path`` exists and skip downloading.
        """
        local_dataset_root = os.path.join(self.raw_dir, "LocalChordDataset", "dataset")
        if os.path.isdir(local_dataset_root):
            if not os.path.exists(self.raw_path):
                os.makedirs(self.raw_path, exist_ok=True)
            return
        # Fallback to original download behavior
        return super().download()

    def process(self, subset=""):
        self.scores = list()
        # Prefer locally provided dataset structure if available
        # Expected: <raw_dir>/LocalChordDataset/dataset/{training,validation}
        local_candidate = os.path.join(self.raw_dir, "LocalChordDataset", "dataset")
        print("Dataset path: ", local_candidate)
        alt_candidate = os.path.join(self.raw_dir, "dataset")
        base_dir = None
        if os.path.isdir(local_candidate):
            base_dir = local_candidate
        elif os.path.isdir(alt_candidate):
            base_dir = alt_candidate
        else:
            base_dir = self.raw_path

        for root, dirs, files in os.walk(base_dir):
            if subset and not os.path.basename(root).endswith(subset):
                continue
            for file in files:
                if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                    self.scores.append(os.path.join(root, file))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False



class AugmentedNetLatestChordDataset(BuiltinDataset):
    r"""The AugmentedNet Chord Dataset.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if AugmentedNet Chord Dataset scores are already available otherwise it will download it.
        Default: ~/.chordgnn/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, is_zip=True):
        url = "https://github.com/napulen/AugmentedNet/releases/latest/download/dataset.zip"
        super(AugmentedNetLatestChordDataset, self).__init__(
            name="AugmentedNetLatestChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        """Override to avoid remote download when local dataset is present.

        If a local dataset exists at ``<raw_dir>/LocalChordDataset/dataset``, we
        simply ensure ``raw_path`` exists and skip downloading.
        """
        local_dataset_root = os.path.join(self.raw_dir, "LocalChordDataset", "dataset")
        if os.path.isdir(local_dataset_root):
            if not os.path.exists(self.raw_path):
                os.makedirs(self.raw_path, exist_ok=True)
            return
        # Fallback to original download behavior
        return super().download()

    def process(self, subset=""):
        self.scores = list()
        # Prefer locally provided dataset structure if available
        # Expected: <raw_dir>/LocalChordDataset/dataset/{training,validation}
        local_candidate = os.path.join(self.raw_dir, "LocalChordDataset", "dataset")
        alt_candidate = os.path.join(self.raw_dir, "dataset")
        base_dir = None
        if os.path.isdir(local_candidate):
            base_dir = local_candidate
        elif os.path.isdir(alt_candidate):
            base_dir = alt_candidate
        else:
            base_dir = self.raw_path

        for root, dirs, files in os.walk(base_dir):
            if subset and not os.path.basename(root).endswith(subset):
                continue
            for file in files:
                if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                    self.scores.append(os.path.join(root, file))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class ChordGraphDataset(chordgnnDataset):
    def __init__(self, dataset_base, max_size=None, verbose=True, nprocs=1, name=None, raw_dir=None, force_reload=False, prob_pieces=[], skip_processing=False):
        self.dataset_base = dataset_base
        self.prob_pieces = prob_pieces
        self.dataset_base.process()
        self.max_size = max_size
        if verbose:
            print("Loaded AugmentedNetChordDataset Successfully, now processing...")
        self.graph_dicts = list()
        self.n_jobs = nprocs
        super(ChordGraphDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            skip_processing=skip_processing)

    def process(self):
        Parallel(self.n_jobs)(delayed(self._process_score)(fn) for fn in
                              tqdm(self.dataset_base.scores, desc="Processing AugmentedNetChordGraphDataset"))
        self.load()

    def _process_score(self, score_fn):
        pass
    def has_cache(self):
        # return True
        if all([
            os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0])) for path
            in
            self.dataset_base.scores]):
            return True
        return False

    def save(self):
        """save the graph list and the labels"""
        pass

    def load(self):
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path, fn)
            if not self.include_synth and (graph.name.endswith("-synth") or graph.name.endswith("synth")):
                continue
            if self.collection != "all" and not graph.name.startswith(
                    self.collection) and graph.collection == "test":
                continue
            if graph.name in self.prob_pieces:
                continue
            self.graphs.append(graph)

    @property
    def features(self):
        return self.graphs[0].x.shape[-1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return [
            self.get_graph_attr(i)
            for i in idx
        ]

    def get_graph_attr(self, idx):
        if self.graphs[idx].x.size(0) > self.max_size and self.graphs[idx].collection != "test":
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            # Select labels based on onset beats overlap with the cropped notes
            # Compute labels exactly at the unique onset_divs present in this segment
            segment_onset_div = torch.tensor(self.graphs[idx].note_array["onset_div"][random_idx:random_idx + self.max_size])
            segment_unique_divs = torch.unique(segment_onset_div, sorted=True)
            # Build LUT from unique onset_div across the piece to onset_beat (use numpy for return_index compatibility)
            full_divs_np = self.graphs[idx].note_array["onset_div"].astype(int)
            full_beats_np = self.graphs[idx].note_array["onset_beat"].astype(float)
            unique_divs_np, first_idx_np = np.unique(full_divs_np, return_index=True)
            unique_divs = torch.from_numpy(unique_divs_np).long()
            lut_beats = torch.from_numpy(full_beats_np[first_idx_np]).float()
            # Map segment unique divs to beats via searchsorted
            idxs = torch.searchsorted(unique_divs, segment_unique_divs.long())
            idxs = torch.clamp(idxs, max=unique_divs.shape[0]-1)
            matched = unique_divs[idxs] == segment_unique_divs.long()
            seg_beats = lut_beats[idxs[matched]]
            labels_onset_beats = torch.as_tensor(self.graphs[idx].y[:, -1]).squeeze().float()
            # torch.isin compatibility: use broadcasting equality then any()
            label_idx = labels_onset_beats.unsqueeze(1).eq(seg_beats.unsqueeze(0)).any(dim=1)
            return [
                self.graphs[idx].x[indices],
                self.graphs[idx].edge_index[:, edge_indices] - random_idx,
                self.graphs[idx].edge_type[edge_indices],
                self.graphs[idx].y[label_idx],
                segment_onset_div,
                self.graphs[idx].name
            ]

        else:
            return [
                self.graphs[idx].x,
                self.graphs[idx].edge_index,
                self.graphs[idx].edge_type,
                self.graphs[idx].y,
                torch.tensor(self.graphs[idx].note_array["onset_div"]),
                self.graphs[idx].name
            ]


class AugmentedNetChordGraphDataset(ChordGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11, 
                 collection="all", max_size=512, skip_processing=False):
        dataset_base = AugmentedNetChordDataset(raw_dir=raw_dir)
        self.collection = collection
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        self.prob_pieces = []# ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            # Use representation-defined vocabulary sizes for robustness (e.g., quality from CHORD_QUALITIES)
            from chordgnn.utils.chord_representations import available_representations as _avail_repr
            quality_dim = len(_avail_repr["quality"].classList)
            root_dim = len(_avail_repr["root"].classList)
            bass_dim = len(_avail_repr["bass"].classList)
            key_dim = len(_avail_repr["localkey"].classList)
            tonkey_dim = len(_avail_repr["tonkey"].classList)
            degree1_dim = len(_avail_repr["degree1"].classList)
            degree2_dim = len(_avail_repr["degree2"].classList)
            inversion_dim = len(_avail_repr["inversion"].classList)
            roman_dim = len(_avail_repr["romanNumeral"].classList)
            hrhythm_dim = len(_avail_repr["hrhythm"].classList)
            pcset_dim = len(_avail_repr["pcset"].classList)

            if num_tasks <= 6:
                self.tasks = {
                    "localkey": key_dim, "tonkey": tonkey_dim, "degree1": degree1_dim, 
                    "degree2": degree2_dim, "quality": quality_dim, "inversion": inversion_dim,
                    "root": root_dim, "romanNumeral": roman_dim, "hrhythm": hrhythm_dim,
                    "pcset": pcset_dim, "bass": bass_dim,
                }
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": key_dim, "tonkey": tonkey_dim, "degree1": degree1_dim, 
                    "degree2": degree2_dim, "quality": quality_dim, "inversion": inversion_dim,
                    "root": root_dim, "romanNumeral": roman_dim, "hrhythm": hrhythm_dim,
                    "pcset": pcset_dim, "bass": bass_dim,
                }
        else:
            from chordgnn.utils.chord_representations import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        super(AugmentedNetChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="AugmentedNetChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            skip_processing=skip_processing)

    def _process_score(self, score_fn):
        try:
            base_name = os.path.splitext(os.path.basename(score_fn))[0]
            is_synth = ("synth" in base_name.lower()) or ("synth" in score_fn.lower())
            name = base_name + "synth" if (is_synth and not base_name.endswith("synth")) else base_name
            # Determine collection: map validation to test (placeholder), keep training as training
            # Determine collection by any path segment
            path_parts = [p.lower() for p in os.path.normpath(score_fn).split(os.sep)]
            collection = "test" if any(p in ["validation", "val", "test"] for p in path_parts) else "training"
            # Skip synthetic in test/eval
            if is_synth and collection == "test":
                return
            if collection == "test":
                # No transposition for validation/test
                note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False)
                data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
            else:
                # Training: transpose originals; keep synth without transposition
                if is_synth:
                    note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False)
                    data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
                else:
                    x = time_divided_tsv_to_part(score_fn, transpose=True)
                    for i, (note_array, labels) in enumerate(x):
                        data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name), save_path=self.save_path)
        except Exception as e:
            print(f"[WARN] Skipping problematic piece due to error: {os.path.basename(score_fn)} | {e}")
        return


class Augmented2022ChordGraphDataset(ChordGraphDataset):

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11, collection="all", max_size=512, skip_processing=False):
        dataset_base = AugmentedNetLatestChordDataset(raw_dir=raw_dir)
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        self.collection = collection
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "mps", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        prob_pieces = [
            'keymodt-reger-96A',
            'keymodt-rimsky-korsakov-3-23c',
            'keymodt-reger-88A',
            'keymodt-reger-68',
            'keymodt-reger-84',
            'keymodt-rimsky-korsakov-3-24a',
            'keymodt-rimsky-korsakov-3-23b',
            'keymodt-aldwell-ex27-4b',
            'keymodt-reger-42a',
            'keymodt-reger-08',
            'mps-k545-1',
            'keymodt-tchaikovsky-173b',
            'keymodt-reger-73',
            'mps-k282-3',
            'keymodt-tchaikovsky-173j',
            'keymodt-kostka-payne-ex19-5',
            'mps-k457-3',
            'keymodt-reger-59',
            'keymodt-reger-82',
            'keymodt-rimsky-korsakov-3-5h',
            'mps-k332-2',
            'mps-k310-1',
            'mps-k457-2',
            'keymodt-rimsky-korsakov-3-7',
            'mps-k576-1',
            'keymodt-kostka-payne-ex18-4',
            'keymodt-reger-81',
            'keymodt-reger-45a',
            'keymodt-rimsky-korsakov-3-14g',
            'keymodt-reger-64',
            'keymodt-tchaikovsky-193b',
            'keymodt-reger-86A',
            'keymodt-reger-15',
            'keymodt-reger-28',
            'mps-k309-1',
            'keymodt-reger-99A',
            'keymodt-reger-55',
            'keymodt-tchaikovsky-189']

        # ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35}
            elif num_tasks == 14:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
                    "alto": 35, "soprano": 35}
        else:
            from chordgnn.utils.chord_representations_latest import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        super(Augmented2022ChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="Augmented2022ChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            prob_pieces=prob_pieces,
            skip_processing=skip_processing)

    def _process_score(self, score_fn):
        try:
            base_name = os.path.splitext(os.path.basename(score_fn))[0]
            is_synth = ("synth" in base_name.lower()) or ("synth" in score_fn.lower())
            name = base_name + "-synth" if (is_synth and not base_name.endswith("-synth")) else base_name
            path_parts = [p.lower() for p in os.path.normpath(score_fn).split(os.sep)]
            collection = "test" if any(p in ["validation", "val", "test"] for p in path_parts) else "training"
            if is_synth and collection == "test":
                return
            if collection == "test":
                note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False, version="latest")
                data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
            else:
                if is_synth:
                    note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False, version="latest")
                    data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
                else:
                    x = time_divided_tsv_to_part(score_fn, transpose=True, version="latest")
                    for i, (note_array, labels) in enumerate(x):
                        data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name),
                                      save_path=self.save_path)
        except Exception as e:
            print(f"[WARN] Skipping problematic piece due to error: {os.path.basename(score_fn)} | {e}")
        return




class Pop909ChordDataset(BuiltinDataset):
    r"""Local Pop909-like Chord Dataset organized as TSVs.

    Expects folder structure under raw_dir:
        Pop909ChordDataset/dataset/{training,validation}/.../*.tsv
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, is_zip=False):
        # url not used; kept for interface compatibility
        url = "local://Pop909ChordDataset"
        super(Pop909ChordDataset, self).__init__(
            name="Pop909ChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        # Local dataset; ensure raw_path exists; no download
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path, exist_ok=True)
        return

    def process(self, subset=""):
        self.scores = list()
        base_dir = os.path.join(self.raw_dir, "Pop909ChordDataset", "dataset")
        if not os.path.isdir(base_dir):
            # fallback to raw_path/dataset
            base_dir = os.path.join(self.raw_path, "dataset")
        print("Dataset path: ", base_dir)
        for root, dirs, files in os.walk(base_dir):
            if subset and not os.path.basename(root).endswith(subset):
                continue
            for file in files:
                if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                    self.scores.append(os.path.join(root, file))

    def has_cache(self):
        return os.path.exists(self.save_path)


class Pop909ChordGraphDataset(ChordGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, include_synth=False, num_tasks=3, collection="all", max_size=512, skip_processing=False):
        dataset_base = Pop909ChordDataset(raw_dir=raw_dir)
        self.collection = collection
        assert self.collection in ["training", "validation", "test", "all"] or self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"]
        self.include_synth = include_synth
        # Only three tasks: root, quality, bass
        from chordgnn.utils.chord_representations import available_representations as _avail_repr
        quality_dim = len(_avail_repr["quality"].classList)
        root_dim = len(_avail_repr["root"].classList)
        bass_dim = len(_avail_repr["bass"].classList)
        self.tasks = {"root": root_dim, "quality": quality_dim, "bass": bass_dim}
        super(Pop909ChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="Pop909ChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            skip_processing=skip_processing)

    def _process_score(self, score_fn):
        try:
            base_name = os.path.splitext(os.path.basename(score_fn))[0]
            name = base_name
            path_parts = [p.lower() for p in os.path.normpath(score_fn).split(os.sep)]
            collection = "test" if any(p in ["validation", "val", "test"] for p in path_parts) else "training"
            if collection == "test":
                note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False, version="pop909")
                data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
            else:
                # Augment originals by transposition; synth (if any) stays without augmentation
                is_synth = ("synth" in base_name.lower()) or ("synth" in score_fn.lower())
                if is_synth:
                    note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False, version="pop909")
                    data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
                else:
                    x = time_divided_tsv_to_part(score_fn, transpose=True, version="pop909")
                    for i, (note_array, labels) in enumerate(x):
                        data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name), save_path=self.save_path)
        except Exception as e:
            print(f"[WARN] Skipping problematic piece due to error: {os.path.basename(score_fn)} | {e}")
        return

def data_to_graph(note_array, labels, collection, name, save_path):
    nodes, edges = hetero_graph_from_note_array(note_array=note_array)
    note_features = select_features(note_array, "chord")
    hg = HeteroScoreGraph(
        note_features,
        edges,
        name=name,
        labels=labels,
        note_array=note_array,
    )
    setattr(hg, "collection", collection)
    # pos_enc = positional_encoding(hg.edge_index, len(hg.x), 20)
    # hg.x = torch.cat((hg.x, pos_enc), dim=1)
    hg.save(save_path)
    del hg, note_array, nodes, edges, note_features
    return
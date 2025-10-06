import chordgnn as st
import torch
import random
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument("--collection", type=str, default="all",
                choices=["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"],  help="Collection to test on.")
parser.add_argument("--predict", action="store_true", help="Obtain Predictions using wandb cloud stored artifact.")
parser.add_argument('--use_nade', action="store_true", help="Use NADE instead of MLP classifier.")
parser.add_argument('--use_jk', action="store_true", help="Use Jumping Knowledge In graph Encoder.")
parser.add_argument('--use_rotograd', action="store_true", help="Use Rotograd for MTL training.")
parser.add_argument("--include_synth", action="store_true", help="Include synthetic data.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--use_ckpt", type=str, default=None, help="Use checkpoint for prediction.")
parser.add_argument("--num_tasks", type=int, default=11, help="Number of tasks to train on.")
parser.add_argument("--data_version", type=str, default="v1.0.0", choices=["v1.0.0", "latest", "pop909"], help="Version of the dataset to use.")
parser.add_argument("--skip_processing", action="store_true", help="Skip preprocessing and load existing graphs only.")

# for reproducibility
torch.manual_seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)


args = parser.parse_args()
if isinstance(eval(args.gpus), int):
    if eval(args.gpus) >= 0:
        devices = [eval(args.gpus)]
        dev = devices[0]
    else:
        devices = None
        dev = "cpu"
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    dev = None
n_layers = args.n_layers
n_hidden = args.n_hidden
force_reload = False
num_workers = args.num_workers


name = "Post_pop909"

csv_logger = CSVLogger(save_dir="./logs", name="post_process", version=name)

datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=16, include_synth=args.include_synth, num_tasks=args.num_tasks,
    collection=args.collection, batch_size=args.batch_size, version=args.data_version, skip_processing=args.skip_processing)

if not args.skip_processing:
    if args.data_version == "v1.0.0":
        from chordgnn.utils.chord_representations import available_representations as _avail_repr
        from chordgnn.utils.chord_representations import normalize_pitch_to_sharp, normalize_key_to_sharp
    elif args.data_version == "latest":
        from chordgnn.utils.chord_representations_latest import available_representations as _avail_repr
        from chordgnn.utils.chord_representations_latest import normalize_pitch_to_sharp, normalize_key_to_sharp
    else:
        from chordgnn.utils.chord_representations import available_representations as _avail_repr
        from chordgnn.utils.chord_representations import normalize_pitch_to_sharp, normalize_key_to_sharp
    quality_tokens = list(_avail_repr["quality"].classList)
    root_tokens = list(_avail_repr["root"].classList)
    print("Vocab - quality (raw):", quality_tokens)
    print("Vocab - root (raw):", root_tokens)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="Val Chord Acc", mode="max")
early_stop_callback = EarlyStopping(monitor="Val Chord Acc", min_delta=0.001, patience=5, verbose=False, mode="max")
use_ddp = len(devices) > 1 if isinstance(devices, list) else False
trainer = Trainer(
    max_epochs=31,
    accelerator="auto", devices=devices,
    num_sanity_val_steps=1,
    logger=csv_logger,
    plugins=DDPPlugin(find_unused_parameters=False) if use_ddp else None,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5
    )

import os
encoder = st.models.chord.ChordPrediction(
    datamodule.features, 256, datamodule.tasks, 1, lr=0.001, dropout=0.0,
    weight_decay=0.0, use_nade=False, use_jk=False, use_rotograd=False,
    device=dev)
if args.use_ckpt is None:
    raise ValueError("Please provide --use_ckpt pointing to the Stage 1 checkpoint (.ckpt file or directory).")
ckpt_path = args.use_ckpt
frozen_model = encoder.load_from_checkpoint(ckpt_path).module
model = st.models.chord.PostChordPrediction(
    datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers, lr=args.lr, dropout=args.dropout,
    weight_decay=args.weight_decay, use_nade=args.use_nade, use_jk=args.use_jk, use_rotograd=args.use_rotograd,
    device=dev, frozen_model=frozen_model
    )
trainer.fit(model, datamodule)
trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)


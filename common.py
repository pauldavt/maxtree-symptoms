"""Command-line options and bookkeeping shared by the scripts."""

import argparse
import datetime
import os

import torch

from config import DEFAULT_DATA_DIR


def make_parser(doc):
    return argparse.ArgumentParser(description=doc,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)


def add_data_args(p):
    p.add_argument("--cultivar", default="Rudolph",
                   help="cultivar to use (default: %(default)s)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="folder containing one sub-folder per cultivar (default: %(default)s)")


def add_split_args(p):
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of tiles used for validation (default: %(default)s)")
    p.add_argument("--seed", type=int, default=1,
                   help="seed for the train/validation split (default: %(default)s)")


def add_runtime_args(p, output=True):
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    if output:
        p.add_argument("--output-dir", default=".",
                       help="where to save the model (default: current folder)")
        p.add_argument("--no-timestamp", action="store_true",
                       help="leave the time out of the file name, e.g. when --output-dir "
                            "is already unique per run (an existing file is overwritten)")


def timestamped_path(output_dir, prefix, cultivar, ext, timestamp=True):
    """<output_dir>/<prefix>_<cultivar>_<UTC time><ext>, without the time if
    timestamp is False."""
    os.makedirs(output_dir, exist_ok=True)
    name = f"{prefix}_{cultivar}"
    if timestamp:
        name += "_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, name + ext)


def split_settings(saved_args):
    """The settings that determine the train/validation split."""
    return {k: saved_args.get(k) for k in ("cultivar", "val_frac", "seed")}


def warn_if_split_differs(args, saved_args, what):
    """Warn if an upstream model (`what`) was trained with another split, in
    which case it may have seen tiles that are in the validation set now."""
    if not saved_args:
        return
    mine, theirs = split_settings(vars(args)), split_settings(saved_args)
    diff = {k: (theirs[k], mine[k]) for k in mine if mine[k] != theirs[k]}
    if diff:
        details = ", ".join(f"{k}: {a!r} there vs {b!r} here" for k, (a, b) in diff.items())
        print(f"WARNING: {what} was trained with a different split ({details}); it may have "
              f"seen tiles that are in the validation set here.")

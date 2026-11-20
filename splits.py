"""Random train/validation split of the tiles."""

import numpy as np


def train_val_split(labels, val_frac, seed):
    """Return (train_idx, val_idx) for a random split of the tiles.

    Uses the same permutation as the original xval.random_splitter_gen. Both
    sides must contain every class.
    """
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    labels = np.asarray(labels)
    n = len(labels)
    n_val = int(round(n * val_frac))
    if not 0 < n_val < n:
        raise ValueError(f"Cannot split {n} tiles with val_frac={val_frac}: "
                         f"the training or validation set would be empty")

    perm = np.random.default_rng(seed).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    classes = np.unique(labels)
    for name, idx in (("training", train_idx), ("validation", val_idx)):
        missing = np.setdiff1d(classes, labels[idx])
        if missing.size:
            raise ValueError(f"The {name} set has no tiles of class(es) {missing.tolist()}; "
                             f"try another seed or val_frac")
    return train_idx, val_idx

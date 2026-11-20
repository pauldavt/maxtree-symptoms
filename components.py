"""Probability maps and component embeddings (via the mta extension).

Patch and component embeddings both have shape
(n_tiles, n_rows, n_cols, n_features): there is one component per patch.
"""

import mta
import numpy as np


def probability_maps(clf, embeddings):
    """P(class 1) of every patch or component: (n_tiles, n_rows, n_cols, n_features)
    -> C-contiguous float32 array (n_tiles, n_rows, n_cols)."""
    *grid, n_features = embeddings.shape
    probs = clf.predict_proba(embeddings.reshape(-1, n_features))[:, 1]
    # [:, 1] is a strided view; make it contiguous for the C extension.
    return np.ascontiguousarray(probs, dtype=np.float32).reshape(grid)


def component_attributes(embeddings, patch_clf):
    """Return (component_embeddings, areas, parents) from mta.attributes."""
    return mta.attributes(probability_maps(patch_clf, embeddings), embeddings)


def stack_samples(embeddings, labels, tile_idx):
    """Patches/components of the selected tiles as (X, y); every sample gets the
    label of its tile."""
    selected = embeddings[tile_idx]
    n_tiles, n_rows, n_cols, n_features = selected.shape
    return selected.reshape(-1, n_features), np.repeat(labels[tile_idx], n_rows * n_cols)

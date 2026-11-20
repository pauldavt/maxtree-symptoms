"""Loading images, their YOLO box annotations and the tiles cut out of them.

Each image ``<name>.png`` must have an annotation file ``<name>.txt`` next to it,
with one box per line::

    <class_id> <x_center> <y_center> <width> <height>

where all coordinates are relative to the image size (0..1).

Boxes are represented as half-open pixel ranges ``(y0, y1, x0, x1)``, i.e. the
tile is ``img[y0:y1, x0:x1]``.
"""

import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CLASS_NAMES, MIN_TILE_SIZE


# ------------------------------------------------------------------ annotations


def read_boxes(txt_file, width, height):
    """Read a YOLO annotation file; return a list of (label, (y0, y1, x0, x1)).

    Pixel ranges are half-open and clipped to the image.
    """
    boxes = []
    with open(txt_file) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                label, xc, yc, bw, bh = (float(v) for v in line.split())
            except ValueError as e:
                raise ValueError(f"{txt_file}:{line_no}: malformed line {line!r}") from e
            if label != int(label):
                raise ValueError(f"{txt_file}:{line_no}: class id {label} is not an integer")

            y0, y1 = np.clip(np.round(height * (yc + np.array([-0.5, 0.5]) * bh)), 0, height)
            x0, x1 = np.clip(np.round(width * (xc + np.array([-0.5, 0.5]) * bw)), 0, width)
            boxes.append((int(label), (int(y0), int(y1), int(x0), int(x1))))
    return boxes


def load_image_tiles(image_path, min_tile_size=MIN_TILE_SIZE):
    """Load one image and cut out its annotated boxes.

    Boxes are skipped if they are smaller than min_tile_size in either
    direction, measured as in the original code (an interior box counts one
    pixel more than its half-open size), so the same tiles are kept as before.

    Returns:
        img: the uint8 RGB image
        boxes: list of (y0, y1, x0, x1)
        labels: list of class ids
        tiles: list of uint8 RGB arrays, one per box
    """
    with Image.open(image_path) as im:
        img = np.asarray(im.convert("RGB"))
    height, width = img.shape[:2]
    txt_file = os.path.splitext(image_path)[0] + ".txt"
    boxes, labels, tiles = [], [], []
    for label, (y0, y1, x0, x1) in read_boxes(txt_file, width, height):
        if (_original_size(y0, y1, height) < min_tile_size
                or _original_size(x0, x1, width) < min_tile_size):
            continue
        boxes.append((y0, y1, x0, x1))
        labels.append(label)
        tiles.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
    return img, boxes, labels, tiles


# --------------------------------------------------------------------- datasets


def extract_tiles(files, min_tile_size=MIN_TILE_SIZE):
    """Cut every annotated box (of at least min_tile_size px) out of the images.

    Returns (tiles, labels): a list of uint8 RGB arrays (sizes differ per tile)
    and an int64 array with the class id of each tile.
    """
    tiles, labels = [], []
    for file in tqdm(files, desc="Extracting tiles"):
        _, _, file_labels, file_tiles = load_image_tiles(file, min_tile_size)
        tiles += file_tiles
        labels += file_labels
    return tiles, np.array(labels, dtype=np.int64)


def _original_size(start, end, n):
    """Size of the half-open range [start, end) as the original code measured
    it: inclusive, with both ends clipped to n - 1."""
    return min(end, n - 1) - min(start, n - 1) + 1


def find_image_files(data_dir, cultivar):
    """Return the healthy ('None') images, then the Y-virus images of a cultivar,
    each group sorted (the order of the original code, which fixes the split).

    Images are expected in folder(s) whose name starts with the cultivar name.
    Assumption: each image contains only healthy leaves, or both healthy and
    Y-virus leaves.
    """
    folders = os.path.join(glob.escape(data_dir), f"{glob.escape(cultivar)}*")
    files = []
    for pattern in ("*None*.png", "*Y-Virus*.png"):
        files += [f for f in sorted(glob.glob(os.path.join(folders, pattern)))
                  if f not in files]  # a name matching both patterns is kept once

    if not files:
        available = sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ) if os.path.isdir(data_dir) else []
        raise FileNotFoundError(
            f"No images found for cultivar {cultivar!r} in {data_dir}. "
            f"Folders found there: {available or 'none'}"
        )
    return files


def load_cultivar(data_dir, cultivar):
    """Find, load and cut the tiles of one cultivar; check the labels are binary."""
    files = find_image_files(data_dir, cultivar)
    tiles, labels = extract_tiles(files)
    if not tiles:
        raise ValueError(f"No tiles of at least {MIN_TILE_SIZE} px in {len(files)} images")
    unexpected = set(np.unique(labels).tolist()) - {0, 1}
    if unexpected:
        raise ValueError(f"Expected binary labels 0/1, found {sorted(unexpected)}")
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).tolist()
    print(f"{len(files)} images, {len(tiles)} tiles ({counts} per class)")
    return tiles, labels

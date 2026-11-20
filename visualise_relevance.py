"""Visualise the class-1 probability of a model saved by train_logreg.py.

The tiles of one image (boxes from the matching .txt file) are cut out, resized
and embedded by the backbone exactly as in training.

  patch model      P(class 1) for every 16x16 patch.
  component model  The patch embeddings are turned into component embeddings
                   with the patch model the component model was trained on,
                   and the component model gives P(class 1) per component.
                   Components larger than --max-area are set to 0, then
                   mta.max_ancestor_filter is applied.

Give the model with --patch-logreg or --component-logreg. The component model
uses the patch model it was trained with. The probability map is scaled back
to each tile's size in the original image, and the gamma of each pixel in the
tiles is adjusted:

    out = in ** gamma,   gamma = max_gamma ** (1 - 2 * p)

so p = 1 is brightened (gamma = 1 / max_gamma), p = 0 darkened
(gamma = max_gamma) and p = 0.5 left unchanged. Where tiles overlap, their
probabilities are averaged. Pixels outside the tiles are set to black, or
keep their original values with --no-black-outside.

Example:
    python visualise_relevance.py image.png --patch-logreg patch_logreg_Rudolph_<time>.npz
    python visualise_relevance.py image.png --component-logreg component_logreg_Rudolph_<time>.npz
"""

import argparse
import os

import mta
import numpy as np
import torch
from PIL import Image, ImageDraw

from common import add_runtime_args, make_parser
from components import component_attributes, probability_maps
from config import CLASS_COLORS, MIN_TILE_SIZE
from data import load_image_tiles
from logreg_io import load_logreg
from model import load_backbone, patch_embeddings


def probability_image(shape, boxes, prob_maps, smooth=False):
    """Upsample each tile's probability map to the tile's size in the image.

    Returns (prob, covered): per-pixel probability (averaged over overlapping
    tiles, 0.5 outside them) and a boolean mask of pixels inside a tile.
    """
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)
    resample = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST

    for (y0, y1, x0, x1), pmap in zip(boxes, prob_maps, strict=True):
        up = Image.fromarray(pmap.astype(np.float32)).resize((x1 - x0, y1 - y0), resample)
        total[y0:y1, x0:x1] += np.asarray(up)
        count[y0:y1, x0:x1] += 1

    covered = count > 0
    prob = np.full(shape, 0.5)
    prob[covered] = total[covered] / count[covered]
    return prob, covered


def apply_gamma(img, prob, max_gamma):
    """Gamma-adjust a uint8 RGB image: gamma = max_gamma ** (1 - 2 * prob)."""
    gamma = max_gamma ** (1.0 - 2.0 * prob)
    out = (img.astype(np.float64) / 255.0) ** gamma[..., None]
    return np.round(out * 255).astype(np.uint8)


def draw_boxes(img, boxes, labels):
    """Copy of img with every box outlined in the colour of its class."""
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(img.shape[:2]) / 500))
    for (y0, y1, x0, x1), label in zip(boxes, labels, strict=True):
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=CLASS_COLORS[label], width=width)
    return np.asarray(out)


def parse_args():
    p = make_parser(__doc__)
    p.add_argument("image", help="image file; the boxes are read from the .txt next to it")
    model = p.add_mutually_exclusive_group(required=True)
    model.add_argument("--patch-logreg", help="model saved by 'train_logreg.py patch'")
    model.add_argument("--component-logreg", help="model saved by 'train_logreg.py component'")
    p.add_argument("--max-area", type=float, default=16,
                   help="--component-logreg only: components with a larger area get "
                        "P(class 1) = 0 before the ancestor filter (default: %(default)s)")
    p.add_argument("--output", default=None,
                   help="output image (default: <image>_<level>_relevance.png in the current folder)")
    p.add_argument("--max-gamma", type=float, default=2.5,
                   help="gamma for p = 0; p = 1 gets 1/max-gamma (default: %(default)s)")
    p.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                   help="interpolate the probabilities instead of showing blocks "
                        "(default: %(default)s)")
    p.add_argument("--black-outside", action=argparse.BooleanOptionalAction, default=True,
                   help="set pixels outside all tiles to black (default: %(default)s)")
    p.add_argument("--side-by-side", action="store_true",
                   help="put the original, with the annotated tiles outlined (healthy green, "
                        "Y-virus red), to the left of the adjusted image")
    add_runtime_args(p, output=False)
    args = p.parse_args()
    if args.max_gamma <= 0:
        p.error("--max-gamma must be positive")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    level = "patch" if args.patch_logreg else "component"
    path = args.patch_logreg or args.component_logreg
    clf, saved = load_logreg(path)
    if saved["level"] != level:
        raise ValueError(f"{path} is a {saved['level']}-level model; "
                         f"use --{saved['level']}-logreg")
    checkpoint, image_size = saved["checkpoint"], saved["image_size"]
    print(f"{level.capitalize()} model: {path} (cultivar {saved['cultivar']})")

    if level == "component":
        print(f"Patch model: {saved['patch_logreg']}")
        patch_clf, _ = load_logreg(saved["patch_logreg"])
    print(f"Backbone: {checkpoint or 'ImageNet-pretrained'}")

    img, boxes, labels, tiles = load_image_tiles(args.image)
    if not tiles:
        raise ValueError(f"No tiles of at least {MIN_TILE_SIZE} px found for {args.image}")

    model, _ = load_backbone(checkpoint, device)
    embeddings = patch_embeddings(model, tiles, args.batch_size, device, image_size)
    if level == "patch":
        prob_maps = probability_maps(clf, embeddings)
    else:
        comp_embeddings, areas, parents = component_attributes(embeddings, patch_clf)
        prob_maps = probability_maps(clf, comp_embeddings)
        prob_maps[areas > args.max_area] = 0
        prob_maps = mta.max_ancestor_filter(parents, prob_maps)

    prob, covered = probability_image(img.shape[:2], boxes, prob_maps, smooth=args.smooth)
    out = apply_gamma(img, prob, args.max_gamma)  # prob = 0.5 (gamma 1) outside the tiles
    if args.black_outside:
        out[~covered] = 0
    if args.side_by_side:
        out = np.concatenate([draw_boxes(img, boxes, labels), out], axis=1)

    name = os.path.splitext(os.path.basename(args.image))[0]
    output = args.output or f"{name}_{level}_relevance.png"
    Image.fromarray(out).save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

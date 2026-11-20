"""Logistic regression on patch or component embeddings of the ResNet backbone.

Every tile is resized to 192x192 and passed through the backbone, giving a
12x12 grid of 256-d embeddings, one per 16x16-pixel patch.

  patch      Each patch is a sample.
  component  The patch embeddings are turned into component embeddings by
             mta.attributes, using P(class 1) per patch from a patch model
             trained with the "patch" command (--patch-logreg). Each component
             is a sample. The backbone is the one that patch model used.

Every sample gets the label of its tile, and a logistic regression (with
feature standardisation and balanced class weights) is trained on the
training split and evaluated on the validation split, each sample classified
on its own. The tiles are split at random, so all samples of a tile end up on
the same side. Use the same --seed and --val-frac for every stage, so the
upstream models have not seen the validation tiles; a warning is printed if they differ. The trained model is
saved as a portable .npz file (see logreg_io.py).

Examples:
    python train_logreg.py patch --cultivar Rudolph
    python train_logreg.py patch --cultivar Rudolph --checkpoint model_Rudolph_<time>.pt
    python train_logreg.py component --cultivar Rudolph --patch-logreg patch_logreg_Rudolph_<time>.npz
"""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (add_data_args, add_runtime_args, add_split_args, make_parser,
                    timestamped_path, warn_if_split_differs)
from components import component_attributes, stack_samples
from config import IMAGE_SIZE
from data import load_cultivar
from logreg_io import load_logreg, save_logreg
from model import load_backbone, patch_embeddings
from splits import train_val_split


def make_classifier(C, max_iter, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=seed),
    )


def evaluate(clf, embeddings, labels, val_idx):
    """Sample-level balanced accuracy and normalised confusion matrix."""
    X_val, y_val = stack_samples(embeddings, labels, val_idx)
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1], normalize="true")
    return balanced_accuracy_score(y_val, y_pred), cm


def parse_args():
    p = make_parser(__doc__)
    sub = p.add_subparsers(dest="level", required=True)
    patch = sub.add_parser("patch", help="one sample per patch")
    patch.add_argument("--checkpoint", default=None,
                       help="model saved by finetune.py (default: ImageNet-pretrained backbone)")
    comp = sub.add_parser("component", help="one sample per component (needs mta)")
    comp.add_argument("--patch-logreg", required=True,
                      help="patch model saved by 'train_logreg.py patch'")
    for s in (patch, comp):
        add_data_args(s)
        add_split_args(s)
        s.add_argument("--C", type=float, default=1.0,
                       help="inverse regularisation strength (default: %(default)s)")
        s.add_argument("--max-iter", type=int, default=1000)
        add_runtime_args(s)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Cultivar: {args.cultivar}, level: {args.level}")

    patch_clf = None
    if args.level == "component":
        patch_clf, patch_saved = load_logreg(args.patch_logreg)
        if patch_saved["level"] != "patch":
            raise ValueError(f"{args.patch_logreg} is not a patch-level model")
        checkpoint = patch_saved["checkpoint"]
        print(f"Patch model: {args.patch_logreg} (cultivar {patch_saved['cultivar']})")
        warn_if_split_differs(args, patch_saved["args"], "the patch model")
    else:
        checkpoint = args.checkpoint
    print(f"Backbone: {checkpoint or 'ImageNet-pretrained'}")
    model, backbone_meta = load_backbone(checkpoint, device)
    warn_if_split_differs(args, backbone_meta.get("args"), "the backbone")
    image_size = backbone_meta.get("image_size", IMAGE_SIZE)

    tiles, labels = load_cultivar(args.data_dir, args.cultivar)
    embeddings = patch_embeddings(model, tiles, args.batch_size, device, image_size)
    print(f"Patch embeddings: {embeddings.shape}")  # (n_tiles, n_rows, n_cols, n_features)
    if args.level == "component":
        embeddings = component_attributes(embeddings, patch_clf)[0]

    train_idx, val_idx = train_val_split(labels, args.val_frac, args.seed)
    print(f"Split (seed {args.seed}): {len(train_idx)} training tiles, "
          f"{len(val_idx)} validation tiles")

    print("Fitting logistic regression on the training set...")
    clf = make_classifier(args.C, args.max_iter, args.seed)
    clf.fit(*stack_samples(embeddings, labels, train_idx))

    acc, cm = evaluate(clf, embeddings, labels, val_idx)
    print(f"Validation {args.level} balanced acc.: {acc:.3f}")
    print(f"{args.level.capitalize()}-level confusion matrix (rows: true class, normalised):")
    print(cm)

    path = timestamped_path(args.output_dir, f"{args.level}_logreg", args.cultivar, ".npz",
                            timestamp=not args.no_timestamp)
    meta = {"level": args.level, "cultivar": args.cultivar, "checkpoint": checkpoint,
            "image_size": image_size, "patch_logreg": getattr(args, "patch_logreg", None),
            "val_scores": {f"{args.level}_balanced_acc": float(acc)}, "args": vars(args)}
    save_logreg(path, clf, meta, train_idx, val_idx)
    print(f"Saved logistic regression to {path}")


if __name__ == "__main__":
    main()

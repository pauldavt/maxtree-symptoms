"""Fine-tune a truncated ResNet-18 to tell healthy leaves from Y-virus leaves.

The tiles of one cultivar are randomly split into a training and a validation
set. The model is trained on class-balanced minibatches with
random flips, transposes and shifts, and evaluated on the validation set after
every epoch. The final weights are saved and can be used as the backbone of
train_logreg.py via --checkpoint; use the same --seed and --val-frac there so
the validation tiles stay unseen.

Example:
    python finetune.py --cultivar Rudolph
    python finetune.py --cultivar Rudolph --epochs 30 --device cpu
"""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

from augmentation import RandomFlipTransposeShift
from common import (add_data_args, add_runtime_args, add_split_args, make_parser,
                    timestamped_path)
from config import IMAGE_SIZE
from data import load_cultivar
from model import TruncatedResNet18, images_to_tensor, predict, use_amp
from splits import train_val_split


def balanced_minibatches(labels, batch_size, rng=None):
    """Yield index arrays of shuffled minibatches with equal samples per class.

    Samples are drawn with replacement. Labels must be integers 0..n_classes-1.
    If batch_size isn't divisible by n_classes, each batch has
    (batch_size // n_classes) * n_classes samples.
    """
    rng = np.random.default_rng(rng)
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    n_per_class = batch_size // n_classes
    if n_per_class == 0:
        raise ValueError(f"batch_size ({batch_size}) is smaller than the number of "
                         f"classes ({n_classes})")
    class_indices = [np.flatnonzero(labels == c) for c in range(n_classes)]
    empty = [c for c, idx in enumerate(class_indices) if idx.size == 0]
    if empty:
        raise ValueError(f"No samples found for class(es): {empty}")

    while True:
        batch = np.concatenate([rng.choice(idx, size=n_per_class) for idx in class_indices])
        rng.shuffle(batch)
        yield batch


def train(model, tiles_train, labels_train, tiles_val, labels_val, *, epochs, batch_size,
          lr, weight_decay, device, seed):
    """Train the model; return the validation predictions after the last epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    amp = use_amp(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                  end_factor=0.01, total_iters=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    augment = RandomFlipTransposeShift().train()

    batches = balanced_minibatches(labels_train, batch_size, rng=seed)
    n_batches = max(1, len(tiles_train) // batch_size)

    progress = tqdm(range(epochs), desc="Training")
    for _ in progress:
        model.train()
        for _ in range(n_batches):
            idx = next(batches)
            x = augment(images_to_tensor([tiles_train[i] for i in idx]).to(device))
            target = torch.from_numpy(labels_train[idx]).float().to(device)

            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(x), target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        preds_val = predict(model, tiles_val, batch_size, device)
        progress.set_postfix(loss=f"{loss.item():.3f}",
                             val_bal_acc=f"{balanced_accuracy_score(labels_val, preds_val):.3f}")
    return preds_val


def parse_args():
    p = make_parser(__doc__)
    add_data_args(p)
    add_split_args(p)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    add_runtime_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"Cultivar: {args.cultivar}")

    tiles, labels = load_cultivar(args.data_dir, args.cultivar)
    train_idx, val_idx = train_val_split(labels, args.val_frac, args.seed)
    print(f"Split (seed {args.seed}): {len(train_idx)} training tiles, "
          f"{len(val_idx)} validation tiles")

    model = TruncatedResNet18().to(device)
    labels_val = labels[val_idx]
    preds_val = train(
        model, [tiles[i] for i in train_idx], labels[train_idx],
        [tiles[i] for i in val_idx], labels_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, device=device, seed=args.seed,
    )
    print(f"Validation balanced acc.: {balanced_accuracy_score(labels_val, preds_val):.3f}")
    print("Confusion matrix (rows: true class, normalised):")
    print(confusion_matrix(labels_val, preds_val, labels=[0, 1], normalize="true"))

    path = timestamped_path(args.output_dir, "model", args.cultivar, ".pt",
                            timestamp=not args.no_timestamp)
    torch.save({"model_state_dict": model.state_dict(), "cultivar": args.cultivar,
                "image_size": IMAGE_SIZE, "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(), "args": vars(args)}, path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()

"""The truncated ResNet-18 backbone and helpers to feed tiles through it."""

import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, N_FEATURES


class TruncatedResNet18(torch.nn.Module):
    """ResNet-18 up to layer3 (output stride 16), global average pooling, linear head.

    Expects RGB images scaled to [0, 1]; ImageNet normalisation is applied inside
    the model. Returns one logit per image. conv1 and the conv weights of layer1
    are frozen (their batch-norm layers are still trained).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.backbone = torch.nn.Sequential(
            r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3
        )
        self.head = torch.nn.Linear(N_FEATURES, 1)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        for p in r.conv1.parameters():
            p.requires_grad = False
        for name, p in r.layer1.named_parameters():
            if "conv" in name:
                p.requires_grad = False

    def features(self, x):
        """Return the backbone feature map, shape (N, 256, H/16, W/16).

        Each spatial position is the embedding of one 16x16-pixel patch.
        """
        return self.backbone((x - self.mean) / self.std)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(-1)


def use_amp(device):
    return device.type == "cuda"


def images_to_tensor(imgs, size=IMAGE_SIZE):
    """Resize uint8 HxWx3 arrays to size x size and stack them as a float32
    tensor of shape (N, 3, size, size) with values in [0, 1]."""
    out = torch.empty((len(imgs), 3, size, size), dtype=torch.float32)
    for k, img in enumerate(imgs):
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise TypeError(f"Expected uint8 HxWx3 images, got {img.dtype} {img.shape}")
        chw = torch.from_numpy(img).permute(2, 0, 1)
        out[k] = v2.functional.resize(chw, [size, size], antialias=True).float() / 255
    return out


def load_backbone(checkpoint, device):
    """Return (model in eval mode, checkpoint metadata).

    Without a checkpoint the ImageNet-pretrained weights are used and the
    metadata is empty; otherwise the weights saved by finetune.py are loaded.
    """
    model = TruncatedResNet18(pretrained=checkpoint is None)
    meta = {}
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        meta = {k: v for k, v in saved.items() if k != "model_state_dict"}
    return model.to(device).eval(), meta


@torch.no_grad()
def _run_batches(fn, imgs, batch_size, device, image_size, desc=None):
    """Apply fn to the resized tiles batch by batch; concatenate the numpy results.

    Shows a progress bar only if desc is given, so that no second bar appears
    when this runs inside another progress bar (e.g. validation while training).
    """
    starts = range(0, len(imgs), batch_size)
    out = []
    for start in (tqdm(starts, desc=desc) if desc else starts):
        x = images_to_tensor(imgs[start : start + batch_size], image_size).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp(device)):
            out.append(fn(x).cpu().numpy())
    return np.concatenate(out)


def patch_embeddings(model, imgs, batch_size, device, image_size=IMAGE_SIZE):
    """Embed every 16x16 patch of every tile.

    Returns a float32 array of shape (n_tiles, n_rows, n_cols, n_features),
    e.g. (n_tiles, 12, 12, 256), where [t, r, c] is the patch in row r and
    column c of tile t.
    """
    model.eval()
    return _run_batches(lambda x: model.features(x).float().permute(0, 2, 3, 1),
                        imgs, batch_size, device, image_size, desc="Embedding")


def predict(model, imgs, batch_size, device, image_size=IMAGE_SIZE):
    """Return binary predictions (0/1) of the model's own head for a list of tiles."""
    model.eval()
    return _run_batches(lambda x: model(x) >= 0, imgs, batch_size, device,
                        image_size).astype(np.int64)

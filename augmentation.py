"""GPU-friendly batch augmentation used during fine-tuning."""

import torch
import torch.nn.functional as F


class RandomFlipTransposeShift(torch.nn.Module):
    """Per-image random transpose, horizontal flip, vertical flip and translation
    for a batch of shape (N, C, H, W). Runs on whatever device the batch is on.

    Equivalent to
        RandomTranspose(), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
        RandomAffine(0, translate=(max_shift / W, max_shift / H))
    but with independent parameters for every image in the batch.

    All operations are combined into one affine warp, so each image is
    resampled only once. With integer shifts (the default, like torchvision's
    RandomAffine) every output pixel is an exact copy of an input pixel;
    exposed borders are filled with zeros. Only active in training mode.
    """

    def __init__(self, max_shift=32, p_transpose=0.5, p_hflip=0.5, p_vflip=0.5,
                 integer_shift=True):
        super().__init__()
        self.max_shift = max_shift
        self.p_transpose = p_transpose
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.integer_shift = integer_shift

    def forward(self, x):
        if not self.training:
            return x
        n, _, h, w = x.shape
        if self.p_transpose > 0 and h != w:
            raise ValueError(f"Transpose needs square images, got {h}x{w}")
        dev, dt = x.device, x.dtype

        # Linear part: optional axis swap (transpose), then per-axis sign (flips).
        # The 8 combinations are exactly the symmetries of the square.
        A = torch.eye(2, device=dev, dtype=dt).repeat(n, 1, 1)
        transpose = torch.rand(n, device=dev) < self.p_transpose
        A[transpose] = A[transpose].flip(1)  # swap rows: (x, y) -> (y, x)
        sx = torch.where(torch.rand(n, device=dev) < self.p_hflip, -1.0, 1.0)
        sy = torch.where(torch.rand(n, device=dev) < self.p_vflip, -1.0, 1.0)
        A = torch.stack([sx, sy], dim=1).to(dt).unsqueeze(-1) * A

        # Translation in pixels, uniform in [-max_shift, max_shift], converted to
        # normalised coordinates ([-1, 1] spans the image).
        shift = (torch.rand(n, 2, device=dev) * 2 - 1) * self.max_shift
        if self.integer_shift:
            shift = shift.round()
        t = shift.to(dt) * torch.tensor([2.0 / w, 2.0 / h], device=dev, dtype=dt)

        # affine_grid maps output to input coordinates: in = A @ out - t
        theta = torch.cat([A, -t.unsqueeze(-1)], dim=-1)  # (N, 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        # Integer shifts land exactly on pixel centres, so "nearest" is an exact
        # copy and avoids rounding noise from bilinear weights.
        mode = "nearest" if self.integer_shift else "bilinear"
        return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)

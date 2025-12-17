from __future__ import annotations

import numpy as np


def overlay_mask_on_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    m = mask.astype(bool)
    out = img.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[m] = (1.0 - float(alpha)) * out[m] + float(alpha) * c
    return np.clip(out, 0, 255).astype(np.uint8)


__all__ = ["overlay_mask_on_rgb"]


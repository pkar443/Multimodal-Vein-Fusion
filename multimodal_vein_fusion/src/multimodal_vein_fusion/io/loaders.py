from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .schema import ThermalLoadResult

logger = logging.getLogger(__name__)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def _normalize_01(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32, copy=False)
    vmin = float(np.min(image_f))
    vmax = float(np.max(image_f))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(image_f, dtype=np.float32)
    return (image_f - vmin) / (vmax - vmin)


def load_rgb(path: str | Path) -> np.ndarray:
    p = _as_path(path)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {p}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def load_nir(path: str | Path) -> np.ndarray:
    p = _as_path(path)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read NIR image: {p}")
    gray = _ensure_grayscale(img)
    if gray.dtype == np.uint8:
        return gray.astype(np.float32) / 255.0
    if gray.dtype == np.uint16:
        return gray.astype(np.float32) / 65535.0
    if np.issubdtype(gray.dtype, np.floating):
        return _normalize_01(gray)
    logger.warning("Unsupported NIR dtype %s; normalizing via min/max", gray.dtype)
    return _normalize_01(gray)


def load_thermal(path: str | Path) -> ThermalLoadResult:
    p = _as_path(path)
    warnings: list[str] = []

    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        if arr.ndim == 3:
            warnings.append("Thermal .npy has 3 dimensions; averaging channels to 2D.")
            arr = np.mean(arr, axis=2)
        if arr.ndim != 2:
            raise ValueError(f"Thermal .npy must be 2D (H,W), got shape {arr.shape}")
        return ThermalLoadResult(
            array=arr.astype(np.float32, copy=False),
            path=p,
            warnings=tuple(warnings),
            original_dtype=str(arr.dtype),
        )

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read Thermal image: {p}")

    if img.ndim == 3:
        img = _ensure_grayscale(img)

    if img.dtype == np.uint8:
        warnings.append(
            "Thermal PNG is 8-bit; prefer 16-bit PNG or .npy for best results. Proceeding with robust normalization."
        )
    elif img.dtype == np.uint16:
        pass
    elif np.issubdtype(img.dtype, np.floating):
        warnings.append("Thermal PNG is floating-point; proceeding with robust normalization.")
    else:
        warnings.append(f"Thermal PNG dtype {img.dtype} is uncommon; proceeding with robust normalization.")

    return ThermalLoadResult(
        array=img.astype(np.float32, copy=False),
        path=p,
        warnings=tuple(warnings),
        original_dtype=str(img.dtype),
    )


from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:  # optional
    from skimage.filters import frangi as _sk_frangi

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _sk_frangi = None
    _HAS_SKIMAGE = False


def _normalize_robust01(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(xf)
    if not np.any(finite):
        return np.zeros_like(xf)
    lo, hi = np.percentile(xf[finite], [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(xf)
    out = (xf - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0)


def _hessian_vesselness_multiscale(
    image01: np.ndarray,
    *,
    sigmas: Sequence[float] = (1.0, 2.0, 3.0),
    beta: float = 0.5,
    c: float = 0.25,
) -> np.ndarray:
    img = np.asarray(image01, dtype=np.float32)
    vessel = np.zeros_like(img, dtype=np.float32)
    eps = 1e-6

    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        ixx = cv2.Sobel(blur, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(blur, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(blur, cv2.CV_32F, 1, 1, ksize=3)

        scale = float(sigma) ** 2
        ixx *= scale
        iyy *= scale
        ixy *= scale

        tmp = np.sqrt((ixx - iyy) ** 2 + 4.0 * (ixy**2))
        l1 = 0.5 * (ixx + iyy + tmp)
        l2 = 0.5 * (ixx + iyy - tmp)

        swap = np.abs(l1) > np.abs(l2)
        l1s = l1.copy()
        l2s = l2.copy()
        l1s[swap] = l2[swap]
        l2s[swap] = l1[swap]

        rb = (np.abs(l1s) / (np.abs(l2s) + eps)) ** 2
        s2 = l1s**2 + l2s**2

        v = np.exp(-rb / (2.0 * beta * beta)) * (1.0 - np.exp(-s2 / (2.0 * c * c)))
        v[l2s > 0] = 0.0  # bright tubular structures on dark background
        vessel = np.maximum(vessel, v.astype(np.float32))

    return _normalize_robust01(vessel)


def compute_nir_vesselness(nir01: np.ndarray) -> np.ndarray:
    nir = np.asarray(nir01, dtype=np.float32)
    nir = np.clip(nir, 0.0, 1.0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    nir_u8 = (nir * 255.0 + 0.5).astype(np.uint8)
    nir_eq = clahe.apply(nir_u8).astype(np.float32) / 255.0

    inv = 1.0 - nir_eq
    if _HAS_SKIMAGE and _sk_frangi is not None:
        try:
            v = _sk_frangi(inv, sigmas=(1, 2, 3), black_ridges=False)
            return _normalize_robust01(v)
        except Exception as e:  # pragma: no cover
            logger.warning("skimage frangi failed (%s); using fallback vesselness.", e)

    return _hessian_vesselness_multiscale(inv, sigmas=(1.0, 2.0, 3.0))


__all__ = ["compute_nir_vesselness"]


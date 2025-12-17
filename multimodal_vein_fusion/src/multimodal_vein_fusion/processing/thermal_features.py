from __future__ import annotations

import numpy as np
import cv2


def _robust_normalize_01(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(xf)
    if not np.any(finite):
        return np.zeros_like(xf)
    lo, hi = np.percentile(xf[finite], [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(xf)
    out = (xf - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0)


def compute_thermal_perfusion(thermal: np.ndarray) -> np.ndarray:
    t01 = _robust_normalize_01(thermal, 2.0, 98.0)
    smooth = cv2.GaussianBlur(t01.astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=1.5)

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad01 = _robust_normalize_01(grad, 2.0, 98.0)

    perf = 0.7 * t01 + 0.3 * grad01
    return np.clip(perf, 0.0, 1.0).astype(np.float32)


__all__ = ["compute_thermal_perfusion"]


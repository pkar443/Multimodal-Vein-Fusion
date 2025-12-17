from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:  # optional
    from skimage.morphology import skeletonize as _sk_skeletonize

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _sk_skeletonize = None
    _HAS_SKIMAGE = False


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    display_mask: np.ndarray
    skeleton: np.ndarray | None
    threshold: float


def _otsu_threshold(values01: np.ndarray) -> float:
    v = np.asarray(values01, dtype=np.float32)
    if v.size == 0:
        return 0.5
    u8 = (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr) / 255.0


def _remove_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask_u8
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    out = np.zeros_like(mask_u8)
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == lbl] = 255
    return out


def segment_veins(
    *,
    fused_map: np.ndarray,
    hand_mask: np.ndarray,
    min_area: int = 150,
    skeletonize: bool = False,
) -> SegmentationResult:
    fused = np.clip(np.asarray(fused_map, dtype=np.float32), 0.0, 1.0)
    hand = hand_mask.astype(bool)

    vals = fused[hand]
    thr = _otsu_threshold(vals)

    mask = (fused >= thr) & hand
    mask_u8 = (mask.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = _remove_small_components(mask_u8, int(min_area))

    clean = mask_u8.astype(bool)
    skel: np.ndarray | None = None

    if skeletonize and _HAS_SKIMAGE and _sk_skeletonize is not None:
        skel = _sk_skeletonize(clean)
        display = skel.astype(bool)
    else:
        display = clean

    return SegmentationResult(mask=clean, display_mask=display, skeleton=skel, threshold=float(thr))


__all__ = ["SegmentationResult", "segment_veins"]


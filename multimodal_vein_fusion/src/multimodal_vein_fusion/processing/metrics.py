from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx), int(stats[idx, cv2.CC_STAT_AREA])


def _major_axis_length_proxy(component_mask: np.ndarray) -> float:
    ys, xs = np.where(component_mask)
    if xs.size < 2:
        return 0.0
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = coords.mean(axis=0, keepdims=True)
    centered = coords - mean
    cov = (centered.T @ centered) / float(max(1, coords.shape[0] - 1))
    eigvals = np.linalg.eigvalsh(cov)
    major = float(np.sqrt(max(0.0, float(eigvals[-1]))))
    return 4.0 * major


def compute_metrics(
    *,
    fused_map: np.ndarray,
    vein_mask: np.ndarray,
    hand_mask: np.ndarray,
    skeleton: np.ndarray | None,
    weights_used: tuple[float, float, float],
    registration_quality: dict[str, Any],
    fusion_info: dict[str, Any],
) -> dict[str, Any]:
    fused = np.clip(np.asarray(fused_map, dtype=np.float32), 0.0, 1.0)
    vein = vein_mask.astype(bool)
    hand = hand_mask.astype(bool)

    hand_area = int(hand.sum())
    vein_area = int((vein & hand).sum())
    coverage = 100.0 * vein_area / float(max(1, hand_area))

    inside = fused[vein & hand]
    outside = fused[hand & (~vein)]
    if inside.size > 0 and outside.size > 10:
        mean_in = float(inside.mean())
        mean_out = float(outside.mean())
        std_out = float(outside.std())
        cnr = (mean_in - mean_out) / float(std_out + 1e-6)
    else:
        mean_in = mean_out = std_out = cnr = None

    largest_mask, largest_area = _largest_component(vein & hand)

    if skeleton is not None:
        length_proxy = int(np.count_nonzero(skeleton & largest_mask))
        length_kind = "skeleton_pixels"
    else:
        length_proxy = float(_major_axis_length_proxy(largest_mask))
        length_kind = "major_axis_proxy"

    insertion_point = None
    if largest_area > 0:
        vals = fused.copy()
        vals[~largest_mask] = -1.0
        idx = int(np.argmax(vals))
        y, x = np.unravel_index(idx, fused.shape)
        insertion_point = {"x": int(x), "y": int(y), "value": float(fused[y, x])}

    return {
        "weights_used": {"nir": float(weights_used[0]), "thermal": float(weights_used[1]), "edges": float(weights_used[2])},
        "coverage_pct": float(coverage),
        "cnr_proxy": cnr,
        "cnr_components": {"mean_in": mean_in, "mean_out": mean_out, "std_out": std_out},
        "largest_component": {"area_px": int(largest_area), "length_proxy": length_proxy, "length_kind": length_kind},
        "recommended_insertion_point": insertion_point,
        "registration_quality": registration_quality,
        "fusion_info": fusion_info,
    }


__all__ = ["compute_metrics"]


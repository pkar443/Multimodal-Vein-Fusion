from __future__ import annotations

from typing import Any

import numpy as np


def _robust_contrast(x: np.ndarray) -> float:
    xf = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(xf)
    if not np.any(finite):
        return 0.0
    p5, p95 = np.percentile(xf[finite], [5.0, 95.0])
    c = float(p95 - p5)
    return max(0.0, min(1.0, c))


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def fuse_modalities(
    *,
    nir: np.ndarray,
    thermal: np.ndarray,
    edges: np.ndarray,
    weights: tuple[float, float, float],
    adaptive: bool,
    registration_quality: dict[str, Any],
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    nir_f = np.clip(np.asarray(nir, dtype=np.float32), 0.0, 1.0)
    th_f = np.clip(np.asarray(thermal, dtype=np.float32), 0.0, 1.0)
    ed_f = np.clip(np.asarray(edges, dtype=np.float32), 0.0, 1.0)

    w_n, w_t, w_e = (float(weights[0]), float(weights[1]), float(weights[2]))
    info: dict[str, Any] = {
        "requested_weights": {"nir": w_n, "thermal": w_t, "edges": w_e},
    }

    if adaptive:
        q_n = _robust_contrast(nir_f)
        q_t = _robust_contrast(th_f)

        nir_q = registration_quality.get("nir_to_rgb", {}) if isinstance(registration_quality, dict) else {}
        therm_q = registration_quality.get("thermal_to_rgb", {}) if isinstance(registration_quality, dict) else {}

        inlier_ratio = float(nir_q.get("inlier_ratio", 1.0) or 0.0)
        ecc_t = therm_q.get("ecc_score", None)
        ecc_q = 1.0 if ecc_t is None else _clamp01((float(ecc_t) + 1.0) * 0.5)

        q_n = _clamp01(q_n * _clamp01(inlier_ratio))
        q_t = _clamp01(q_t * ecc_q)

        info["quality_proxies"] = {"nir": q_n, "thermal": q_t, "thermal_ecc_q": ecc_q, "nir_inlier_ratio": inlier_ratio}

        w_n *= max(0.15, q_n)
        w_t *= max(0.15, q_t)

    w_sum = w_n + w_t + w_e
    if w_sum <= 1e-9:
        w_n, w_t, w_e = 0.65, 0.25, 0.10
        w_sum = 1.0

    w_n /= w_sum
    w_t /= w_sum
    w_e /= w_sum

    fused = (w_n * nir_f) + (w_t * th_f) + (w_e * ed_f)
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return fused, (float(w_n), float(w_t), float(w_e)), info


__all__ = ["fuse_modalities"]


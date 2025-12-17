from __future__ import annotations

import numpy as np

from multimodal_vein_fusion.processing.fusion import fuse_modalities


def test_fuse_modalities_weighted_sum_no_adaptive() -> None:
    nir = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    thermal = np.zeros_like(nir)
    edges = np.ones_like(nir, dtype=np.float32) * 0.2

    fused, weights_used, info = fuse_modalities(
        nir=nir,
        thermal=thermal,
        edges=edges,
        weights=(0.5, 0.25, 0.25),
        adaptive=False,
        registration_quality={},
    )

    assert fused.shape == nir.shape
    assert np.isfinite(fused).all()
    assert np.allclose(sum(weights_used), 1.0)
    expected = 0.5 * nir + 0.25 * edges
    assert np.allclose(fused, expected, atol=1e-6)
    assert "requested_weights" in info


def test_fuse_modalities_adaptive_downweights_low_quality() -> None:
    nir = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    thermal = np.ones_like(nir, dtype=np.float32) * 0.5  # near-constant => low contrast proxy
    edges = np.zeros_like(nir, dtype=np.float32)

    fused, weights_used, info = fuse_modalities(
        nir=nir,
        thermal=thermal,
        edges=edges,
        weights=(0.6, 0.4, 0.0),
        adaptive=True,
        registration_quality={"nir_to_rgb": {"inlier_ratio": 1.0}, "thermal_to_rgb": {"ecc_score": -1.0}},
    )

    assert np.allclose(sum(weights_used), 1.0)
    assert weights_used[1] < 0.4  # thermal downweighted
    assert "quality_proxies" in info
    assert fused.min() >= 0.0 and fused.max() <= 1.0


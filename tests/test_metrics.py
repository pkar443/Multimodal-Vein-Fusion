from __future__ import annotations

import numpy as np

from multimodal_vein_fusion.processing.metrics import compute_metrics


def test_metrics_basic_fields_present() -> None:
    fused = np.zeros((10, 10), dtype=np.float32)
    fused[2:5, 2:5] = 0.8
    fused[3, 3] = 1.0

    hand = np.ones_like(fused, dtype=bool)
    vein = np.zeros_like(fused, dtype=bool)
    vein[2:5, 2:5] = True

    metrics = compute_metrics(
        fused_map=fused,
        vein_mask=vein,
        hand_mask=hand,
        skeleton=None,
        weights_used=(0.65, 0.25, 0.10),
        registration_quality={"nir_to_rgb": {"method": "identity"}, "thermal_to_rgb": {"method": "identity"}},
        fusion_info={"requested_weights": {"nir": 0.65, "thermal": 0.25, "edges": 0.10}},
    )

    assert "weights_used" in metrics
    assert "coverage_pct" in metrics
    assert "cnr_proxy" in metrics
    assert "largest_component" in metrics
    assert "recommended_insertion_point" in metrics

    ins = metrics["recommended_insertion_point"]
    assert ins is not None
    assert ins["x"] == 3 and ins["y"] == 3
    assert abs(ins["value"] - 1.0) < 1e-6


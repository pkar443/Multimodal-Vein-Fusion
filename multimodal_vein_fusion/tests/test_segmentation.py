from __future__ import annotations

import numpy as np

from multimodal_vein_fusion.processing.segmentation import segment_veins


def test_segmentation_removes_small_components() -> None:
    fused = np.full((20, 20), 0.1, dtype=np.float32)
    hand = np.ones_like(fused, dtype=bool)

    # Big component (area 25)
    fused[2:7, 2:7] = 0.9
    # Small component (area 4)
    fused[15:17, 15:17] = 0.9

    res = segment_veins(fused_map=fused, hand_mask=hand, min_area=10, skeletonize=False)
    mask = res.mask.astype(bool)

    assert bool(mask[3, 3]) is True
    assert bool(mask[16, 16]) is False
    assert res.skeleton is None
    assert np.array_equal(res.display_mask, res.mask)

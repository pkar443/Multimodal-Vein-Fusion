from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .fusion import fuse_modalities
from .metrics import compute_metrics
from .nir_vesselness import compute_nir_vesselness
from .registration import register_nir_to_rgb, register_thermal_to_rgb
from .rgb_features import compute_hand_mask, compute_rgb_edges
from .segmentation import SegmentationResult, segment_veins
from .thermal_features import compute_thermal_perfusion
from ..viz.overlay import overlay_mask_on_rgb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    w_nir: float = 0.65
    w_thermal: float = 0.25
    w_edges: float = 0.10
    min_area: int = 150
    adaptive_weighting: bool = True
    skeletonize: bool = False


@dataclass(frozen=True)
class PipelineResult:
    rgb: np.ndarray
    hand_mask: np.ndarray
    rgb_edges: np.ndarray
    nir_vesselness: np.ndarray
    thermal_perfusion: np.ndarray
    fused_map: np.ndarray
    vein_mask: np.ndarray
    display_mask: np.ndarray
    overlay: np.ndarray
    metrics: dict[str, Any]
    weights_used: tuple[float, float, float]
    registration_quality: dict[str, Any]
    warnings: tuple[str, ...] = ()


def run_pipeline(
    *,
    rgb: np.ndarray,
    nir: np.ndarray,
    thermal: np.ndarray,
    config: PipelineConfig,
) -> PipelineResult:
    if rgb is None or nir is None or thermal is None:
        raise ValueError("rgb, nir, and thermal must all be provided.")

    warnings: list[str] = []

    hand_mask = compute_hand_mask(rgb)
    rgb_edges = compute_rgb_edges(rgb, hand_mask)

    nir_v = compute_nir_vesselness(nir)
    thermal_p = compute_thermal_perfusion(thermal)

    nir_reg, nir_q, nir_warn = register_nir_to_rgb(nir, nir_v, rgb, hand_mask)
    therm_reg, therm_q, therm_warn = register_thermal_to_rgb(thermal_p, rgb, hand_mask)
    warnings.extend(nir_warn)
    warnings.extend(therm_warn)

    registration_quality: dict[str, Any] = {
        "nir_to_rgb": nir_q,
        "thermal_to_rgb": therm_q,
    }

    fused_map, weights_used, fusion_info = fuse_modalities(
        nir=nir_reg,
        thermal=therm_reg,
        edges=rgb_edges,
        weights=(config.w_nir, config.w_thermal, config.w_edges),
        adaptive=config.adaptive_weighting,
        registration_quality=registration_quality,
    )

    seg: SegmentationResult = segment_veins(
        fused_map=fused_map,
        hand_mask=hand_mask,
        min_area=config.min_area,
        skeletonize=config.skeletonize,
    )

    overlay = overlay_mask_on_rgb(rgb, seg.display_mask, alpha=0.45)

    metrics = compute_metrics(
        fused_map=fused_map,
        vein_mask=seg.mask,
        hand_mask=hand_mask,
        skeleton=seg.skeleton,
        weights_used=weights_used,
        registration_quality=registration_quality,
        fusion_info=fusion_info,
    )

    return PipelineResult(
        rgb=rgb,
        hand_mask=hand_mask,
        rgb_edges=rgb_edges,
        nir_vesselness=nir_reg,
        thermal_perfusion=therm_reg,
        fused_map=fused_map,
        vein_mask=seg.mask,
        display_mask=seg.display_mask,
        overlay=overlay,
        metrics=metrics,
        weights_used=weights_used,
        registration_quality=registration_quality,
        warnings=tuple(warnings),
    )


__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline"]

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multimodal_vein_fusion.io import load_nir, load_rgb, load_thermal, save_outputs
from multimodal_vein_fusion.processing import PipelineConfig, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run multimodal vein fusion on a single RGB/NIR/Thermal sample.")
    p.add_argument("--rgb", required=True, help="Path to RGB image (.png/.jpg)")
    p.add_argument("--nir", required=True, help="Path to NIR image (.png/.jpg)")
    p.add_argument("--thermal", required=True, help="Path to Thermal (.npy preferred, or .png)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--w-nir", type=float, default=0.65)
    p.add_argument("--w-thermal", type=float, default=0.25)
    p.add_argument("--w-edges", type=float, default=0.10)
    p.add_argument("--min-area", type=int, default=150)
    p.add_argument("--no-adaptive", action="store_true", help="Disable adaptive weighting")
    p.add_argument("--skeletonize", action="store_true", help="Enable skeletonization (requires scikit-image)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    rgb = load_rgb(args.rgb)
    nir = load_nir(args.nir)
    thermal_res = load_thermal(args.thermal)

    config = PipelineConfig(
        w_nir=float(args.w_nir),
        w_thermal=float(args.w_thermal),
        w_edges=float(args.w_edges),
        min_area=int(args.min_area),
        adaptive_weighting=not bool(args.no_adaptive),
        skeletonize=bool(args.skeletonize),
    )

    result = run_pipeline(rgb=rgb, nir=nir, thermal=thermal_res.array, config=config)
    metrics = dict(result.metrics)
    if thermal_res.warnings:
        metrics["load_warnings"] = list(thermal_res.warnings)
    if result.warnings:
        metrics["warnings"] = list(result.warnings)

    out = Path(args.out)
    save_outputs(
        out,
        rgb=result.rgb,
        nir_vesselness=result.nir_vesselness,
        thermal_perfusion=result.thermal_perfusion,
        fused_map=result.fused_map,
        vein_mask=result.display_mask,
        overlay=result.overlay,
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


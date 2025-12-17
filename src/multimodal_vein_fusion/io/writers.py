from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _to_uint8_gray(image01: np.ndarray) -> np.ndarray:
    img = np.asarray(image01, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        return (m.astype(np.uint8) * 255)
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    return m


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64, np.uint32, np.uint64)):
        return int(obj)
    return str(obj)


def save_outputs(
    out_dir: str | Path,
    *,
    rgb: np.ndarray,
    nir_vesselness: np.ndarray,
    thermal_perfusion: np.ndarray,
    fused_map: np.ndarray,
    vein_mask: np.ndarray,
    overlay: np.ndarray,
    metrics: dict[str, Any],
) -> Path:
    out = _as_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out / "rgb.png")
    Image.fromarray(_to_uint8_gray(nir_vesselness), mode="L").save(out / "nir_vesselness.png")
    Image.fromarray(_to_uint8_gray(thermal_perfusion), mode="L").save(out / "thermal_perfusion.png")
    Image.fromarray(_to_uint8_gray(fused_map), mode="L").save(out / "fused_map.png")
    Image.fromarray(_mask_to_uint8(vein_mask), mode="L").save(out / "vein_mask.png")
    Image.fromarray(overlay).save(out / "overlay.png")

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    logger.info("Saved outputs to %s", out)
    return out


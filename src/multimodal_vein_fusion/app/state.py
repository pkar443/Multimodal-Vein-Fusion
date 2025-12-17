from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from multimodal_vein_fusion.processing import PipelineResult


@dataclass
class AppState:
    mode: str = "files"  # "files" or "live"

    rgb_path: Optional[Path] = None
    nir_path: Optional[Path] = None
    thermal_path: Optional[Path] = None

    rgb: Optional[np.ndarray] = None
    nir: Optional[np.ndarray] = None
    thermal: Optional[np.ndarray] = None

    live_rgb: Optional[np.ndarray] = None
    live_nir: Optional[np.ndarray] = None
    live_thermal: Optional[np.ndarray] = None

    loader_warnings: list[str] = field(default_factory=list)
    pipeline_result: Optional[PipelineResult] = None
    pipeline_running: bool = False
    last_error: Optional[str] = None
    last_metrics_text: str = ""

    def clear(self) -> None:
        self.rgb_path = None
        self.nir_path = None
        self.thermal_path = None
        self.rgb = None
        self.nir = None
        self.thermal = None
        self.pipeline_result = None
        self.pipeline_running = False
        self.loader_warnings.clear()
        self.last_error = None
        self.last_metrics_text = ""

    def current_inputs(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.mode == "live":
            return self.live_rgb, self.live_nir, self.live_thermal
        return self.rgb, self.nir, self.thermal

    def has_all_inputs(self) -> bool:
        rgb, nir, therm = self.current_inputs()
        return rgb is not None and nir is not None and therm is not None


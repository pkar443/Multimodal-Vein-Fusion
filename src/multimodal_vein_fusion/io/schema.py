from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ThermalLoadResult:
    array: np.ndarray
    path: Path
    warnings: Tuple[str, ...] = ()
    original_dtype: str | None = None


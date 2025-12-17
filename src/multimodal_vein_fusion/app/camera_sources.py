from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False

try:  # optional
    from pylepton import Lepton  # type: ignore

    _HAS_PYLEPTON = True
except Exception:  # pragma: no cover
    Lepton = None
    _HAS_PYLEPTON = False


class CameraBase:
    def open(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self) -> Optional[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def is_open(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class OpenCVCamera(CameraBase):
    index: int = 0
    name: str = "OpenCV Camera"
    width: int | None = None
    height: int | None = None

    _cap: object | None = None

    def open(self) -> bool:
        if not _HAS_CV2 or cv2 is None:
            logger.warning("OpenCV not available; cannot open camera.")
            return False
        cap = cv2.VideoCapture(int(self.index))
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def read(self) -> Optional[np.ndarray]:
        if not self.is_open or cv2 is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] >= 3:
            bgr = frame[:, :, :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb.astype(np.uint8)
        return None

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None


class LeptonThermal(CameraBase):
    def __init__(self) -> None:
        self._ctx = None

    def open(self) -> bool:
        if not _HAS_PYLEPTON or Lepton is None:
            return False
        try:
            self._ctx = Lepton()
            self._ctx.__enter__()
            return True
        except Exception:
            self._ctx = None
            return False

    @property
    def is_open(self) -> bool:
        return self._ctx is not None

    def read(self) -> Optional[np.ndarray]:
        if self._ctx is None:
            return None
        try:
            frame, _ = self._ctx.capture()
            return frame.astype(np.float32)
        except Exception:
            return None

    def release(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            finally:
                self._ctx = None


class UVCThermal(OpenCVCamera):
    pass


__all__ = ["CameraBase", "OpenCVCamera", "UVCThermal", "LeptonThermal"]


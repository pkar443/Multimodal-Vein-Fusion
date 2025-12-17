from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _largest_filled_contour(mask_u8: np.ndarray) -> np.ndarray:
    contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_u8)
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_u8)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def compute_hand_mask(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB must be HxWx3, got {rgb.shape}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([20, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    skin = cv2.bitwise_or(m1, m2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)

    skin_filled = _largest_filled_contour(skin)
    skin_area = int(np.count_nonzero(skin_filled))

    if skin_area >= 0.02 * rgb.shape[0] * rgb.shape[1]:
        return skin_filled.astype(bool)

    logger.warning("HSV skin mask failed; falling back to Otsu thresholding.")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _thr, otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.count_nonzero(otsu) < (otsu.size // 2):
        mask = otsu
    else:
        mask = cv2.bitwise_not(otsu)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _largest_filled_contour(mask)
    return mask.astype(bool)


def compute_rgb_edges(rgb: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    if hand_mask.dtype != np.bool_:
        hand_mask = hand_mask.astype(bool)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges = edges.astype(np.float32) / 255.0
    edges[~hand_mask] = 0.0

    if float(edges.max()) > 0:
        edges = edges / float(edges.max())
    return edges.astype(np.float32)


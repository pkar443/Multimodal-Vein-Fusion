from __future__ import annotations

import cv2
import numpy as np

from multimodal_vein_fusion.processing.registration import register_nir_to_rgb, register_thermal_to_rgb


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(inter) / float(max(1, union))


def _make_hand_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    # Palm
    cv2.rectangle(m, (int(w * 0.25), int(h * 0.50)), (int(w * 0.78), int(h * 0.90)), 255, thickness=cv2.FILLED)
    # Fingers
    finger_w = max(6, int(w * 0.06))
    for i in range(5):
        x0 = int(w * (0.28 + 0.09 * i))
        cv2.rectangle(m, (x0, int(h * 0.28)), (x0 + finger_w, int(h * 0.50)), 255, thickness=cv2.FILLED)
    # Thumb (break symmetry)
    cv2.rectangle(m, (int(w * 0.72), int(h * 0.60)), (int(w * 0.92), int(h * 0.76)), 255, thickness=cv2.FILLED)
    return m.astype(bool)


def _make_rgb_from_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.full((h, w, 3), (25, 25, 25), dtype=np.uint8)
    rgb[mask] = (210, 170, 150)
    return rgb


def _warp_u8(u8: np.ndarray, mtx: np.ndarray, out_hw: tuple[int, int], *, interp: int) -> np.ndarray:
    h, w = out_hw
    return cv2.warpAffine(u8, mtx, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def test_register_nir_to_rgb_recovers_affine() -> None:
    h, w = 220, 280
    hand = _make_hand_mask(h, w)
    rgb = _make_rgb_from_mask(hand)

    # Create a misaligned NIR image/map by warping the hand mask.
    center = (w / 2.0, h / 2.0)
    mtx = cv2.getRotationMatrix2D(center, 7.0, 0.92)
    mtx[0, 2] += 14.0
    mtx[1, 2] += -9.0

    nir_mask_u8 = _warp_u8(hand.astype(np.uint8) * 255, mtx, (h, w), interp=cv2.INTER_NEAREST)
    nir_map01 = (nir_mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    nir_img01 = (0.10 + 0.75 * nir_map01).clip(0.0, 1.0)

    nir_reg, q, _warn = register_nir_to_rgb(nir_img01, nir_map01, rgb, hand)

    assert nir_reg.shape == (h, w)
    assert q.get("status") in {"ok", "warning"}
    assert _iou(nir_reg > 0.5, hand) > 0.85


def test_register_nir_to_rgb_handles_different_input_size() -> None:
    h, w = 240, 300
    hand = _make_hand_mask(h, w)
    rgb = _make_rgb_from_mask(hand)

    center = (w / 2.0, h / 2.0)
    mtx = cv2.getRotationMatrix2D(center, -5.0, 1.05)
    mtx[0, 2] += -18.0
    mtx[1, 2] += 11.0

    nir_mask_u8 = _warp_u8(hand.astype(np.uint8) * 255, mtx, (h, w), interp=cv2.INTER_NEAREST)
    nir_map01_full = (nir_mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    nir_img01_full = (0.12 + 0.70 * nir_map01_full).clip(0.0, 1.0)

    # Simulate a lower-res NIR sensor.
    nir_small = cv2.resize(nir_img01_full, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
    map_small = cv2.resize(nir_map01_full, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)

    nir_reg, _q, _warn = register_nir_to_rgb(nir_small, map_small, rgb, hand)
    assert nir_reg.shape == (h, w)
    assert _iou(nir_reg > 0.5, hand) > 0.80


def test_register_thermal_to_rgb_recovers_affine() -> None:
    h, w = 210, 260
    hand = _make_hand_mask(h, w)
    rgb = _make_rgb_from_mask(hand)

    center = (w / 2.0, h / 2.0)
    mtx = cv2.getRotationMatrix2D(center, 9.0, 0.95)
    mtx[0, 2] += 10.0
    mtx[1, 2] += 7.0

    therm_mask_u8 = _warp_u8(hand.astype(np.uint8) * 255, mtx, (h, w), interp=cv2.INTER_NEAREST)
    thermal01 = (therm_mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    thermal01 = (0.05 + 0.90 * thermal01).clip(0.0, 1.0)

    therm_reg, q, _warn = register_thermal_to_rgb(thermal01, rgb, hand)
    assert therm_reg.shape == (h, w)
    assert q.get("status") in {"ok", "warning"}
    assert _iou(therm_reg > 0.5, hand) > 0.85


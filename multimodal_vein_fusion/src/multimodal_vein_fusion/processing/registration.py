from __future__ import annotations

import math
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _to_u8(image01: np.ndarray) -> np.ndarray:
    img = np.asarray(image01, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _resize_to(rgb_shape: tuple[int, int], src: np.ndarray) -> np.ndarray:
    h, w = rgb_shape
    if src.shape[:2] == (h, w):
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def _orb_affine(
    src_u8: np.ndarray,
    dst_u8: np.ndarray,
    *,
    mask_src_u8: np.ndarray | None,
    mask_dst_u8: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(src_u8, mask_src_u8)
    kp2, des2 = orb.detectAndCompute(dst_u8, mask_dst_u8)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, {"method": "orb", "status": "no_descriptors"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, {"method": "orb", "status": "too_few_matches", "matches_good": len(good)}

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    mtx, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99,
        maxIters=2000,
    )

    if mtx is None:
        return None, {"method": "orb", "status": "estimate_failed", "matches_good": len(good)}

    inliers_count = int(inliers.sum()) if inliers is not None else 0
    return mtx.astype(np.float32), {
        "method": "orb_ransac",
        "status": "ok",
        "matches_good": int(len(good)),
        "inliers": inliers_count,
        "inlier_ratio": float(inliers_count / max(1, len(good))),
    }


def _ecc_refine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    initial_warp: np.ndarray | None,
    mask_u8: np.ndarray | None,
    warp_mode: int,
    iterations: int = 60,
    eps: float = 1e-5,
) -> tuple[np.ndarray | None, float | None]:
    src_f = np.asarray(src, dtype=np.float32)
    dst_f = np.asarray(dst, dtype=np.float32)
    if src_f.shape != dst_f.shape:
        return None, None

    warp = initial_warp
    if warp is None:
        warp = np.eye(2, 3, dtype=np.float32) if warp_mode != cv2.MOTION_HOMOGRAPHY else np.eye(3, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), float(eps))
    try:
        cc, warp_out = cv2.findTransformECC(
            templateImage=dst_f,
            inputImage=src_f,
            warpMatrix=warp,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=mask_u8,
            gaussFiltSize=5,
        )
        return warp_out.astype(np.float32), float(cc)
    except cv2.error:
        return None, None


def _warp_affine(src: np.ndarray, mtx: np.ndarray, out_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_shape_hw
    return cv2.warpAffine(src, mtx, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _warp_mask_u8(mask_u8: np.ndarray, mtx: np.ndarray, out_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_shape_hw
    return cv2.warpAffine(
        mask_u8,
        mtx,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _mask_iou_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8.astype(bool)
    b = b_u8.astype(bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(inter) / float(max(1, union))


def _edge_iou01(src01: np.ndarray, dst01: np.ndarray, *, mask_u8: np.ndarray | None, thr: float = 0.15) -> float:
    s = np.asarray(src01, dtype=np.float32)
    d = np.asarray(dst01, dtype=np.float32)
    if s.shape != d.shape:
        return 0.0
    s_bin = s > float(thr)
    d_bin = d > float(thr)
    if mask_u8 is not None:
        m = mask_u8.astype(bool)
        s_bin &= m
        d_bin &= m
    inter = int(np.count_nonzero(s_bin & d_bin))
    union = int(np.count_nonzero(s_bin | d_bin))
    return float(inter) / float(max(1, union))


def _largest_filled_contour(mask_u8: np.ndarray) -> np.ndarray:
    contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_u8)
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_u8)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def _hand_mask_from_gray01(gray01: np.ndarray, *, ref_mask_u8: np.ndarray | None = None) -> np.ndarray | None:
    g = np.asarray(gray01, dtype=np.float32)
    if g.ndim != 2:
        return None

    u8 = _to_u8(g)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    _thr, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = [otsu, cv2.bitwise_not(otsu)]
    best: np.ndarray | None = None
    best_score: float = -1.0

    for cand in candidates:
        m = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        filled = _largest_filled_contour(m)
        ratio = float(np.count_nonzero(filled)) / float(max(1, filled.size))
        if ratio < 0.02 or ratio > 0.85:
            continue
        if ref_mask_u8 is not None:
            score = _mask_iou_u8(filled, ref_mask_u8)
        else:
            # Without a reference mask, prefer a plausible hand-sized foreground.
            score = 1.0 - abs(ratio - 0.35)
        if score > best_score:
            best = filled
            best_score = score

    if best is None:
        return None
    return best.astype(bool)


def _dilate_mask_u8(mask_u8: np.ndarray, *, ksize: int = 21, iterations: int = 1) -> np.ndarray:
    k = int(max(3, ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_u8, kernel, iterations=int(max(1, iterations)))


def _mask_boundary01(mask_u8: np.ndarray) -> np.ndarray:
    edge = cv2.Canny(mask_u8, 50, 150).astype(np.float32) / 255.0
    return cv2.GaussianBlur(edge, (0, 0), sigmaX=1.2, sigmaY=1.2)


def _moments_center_angle_area(mask_u8: np.ndarray) -> tuple[tuple[float, float], float, float] | None:
    m = cv2.moments(mask_u8, binaryImage=True)
    m00 = float(m.get("m00", 0.0))
    if m00 <= 1e-6:
        return None
    cx = float(m["m10"]) / m00
    cy = float(m["m01"]) / m00
    mu11 = float(m["mu11"]) / m00
    mu20 = float(m["mu20"]) / m00
    mu02 = float(m["mu02"]) / m00
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    return (cx, cy), float(theta), m00


def _wrap_pi(angle_rad: float) -> float:
    a = float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)
    return a


def _minimal_axis_delta(src_theta: float, dst_theta: float) -> float:
    # Major-axis orientation has 180° symmetry; choose the smallest equivalent rotation.
    delta = _wrap_pi(float(dst_theta) - float(src_theta))
    if delta > math.pi / 2.0:
        delta -= math.pi
    elif delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _coarse_similarity_from_masks(
    src_mask_u8: np.ndarray, dst_mask_u8: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    src_stats = _moments_center_angle_area(src_mask_u8)
    dst_stats = _moments_center_angle_area(dst_mask_u8)
    if src_stats is None or dst_stats is None:
        return None, {"method": "coarse_similarity", "status": "no_mask_moments"}

    (sx, sy), src_theta, src_area = src_stats
    (dx, dy), dst_theta, dst_area = dst_stats

    scale = float(math.sqrt(max(1e-6, dst_area) / max(1e-6, src_area)))
    scale = float(max(0.25, min(4.0, scale)))

    delta = _minimal_axis_delta(src_theta, dst_theta)

    c = math.cos(delta) * scale
    s = math.sin(delta) * scale

    tx = dx - (c * sx - s * sy)
    ty = dy - (s * sx + c * sy)
    mtx = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)

    return mtx, {
        "method": "coarse_similarity",
        "status": "ok",
        "scale": scale,
        "rotation_deg": float(delta * 180.0 / math.pi),
    }


def _nir_edges01(nir_u8: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(nir_u8, 30, 90).astype(np.float32) / 255.0
    return cv2.GaussianBlur(edges, (0, 0), sigmaX=1.2, sigmaY=1.2)


def _thermal_edges01(therm_u8: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(therm_u8, 30, 90).astype(np.float32) / 255.0
    return cv2.GaussianBlur(edges, (0, 0), sigmaX=1.2, sigmaY=1.2)


def _select_best_warp(
    *,
    out_shape_hw: tuple[int, int],
    src_mask_u8: np.ndarray | None,
    dst_mask_u8: np.ndarray,
    src_edge01: np.ndarray,
    dst_edge01: np.ndarray,
    eval_mask_u8: np.ndarray,
    candidates: dict[str, np.ndarray | None],
) -> tuple[np.ndarray, dict[str, Any]]:
    identity = np.eye(2, 3, dtype=np.float32)
    all_candidates: dict[str, np.ndarray] = {"identity": identity}
    for name, mtx in candidates.items():
        if mtx is None:
            continue
        all_candidates[name] = np.asarray(mtx, dtype=np.float32)

    edge_scores: dict[str, float] = {}
    mask_scores: dict[str, float] = {}

    for name, mtx in all_candidates.items():
        warped_edge = _warp_affine(src_edge01, mtx, out_shape_hw)
        edge_scores[name] = _edge_iou01(warped_edge, dst_edge01, mask_u8=eval_mask_u8)
        if src_mask_u8 is not None:
            warped_mask = _warp_mask_u8(src_mask_u8, mtx, out_shape_hw)
            mask_scores[name] = _mask_iou_u8(warped_mask, dst_mask_u8)

    pick_by = "edge_iou"
    if src_mask_u8 is not None and mask_scores:
        best_mask = max(mask_scores.values()) if mask_scores else 0.0
        # If the mask estimate is plausible, prefer mask IoU; otherwise fall back to edge IoU.
        if best_mask >= 0.15:
            pick_by = "mask_iou"

    if pick_by == "mask_iou":
        best_name = max(mask_scores, key=mask_scores.get)
    else:
        best_name = max(edge_scores, key=edge_scores.get)

    info: dict[str, Any] = {
        "selected": best_name,
        "selected_by": pick_by,
        "edge_iou": edge_scores,
    }
    if mask_scores:
        info["mask_iou"] = mask_scores

    return all_candidates[best_name], info


def register_nir_to_rgb(
    nir_image01: np.ndarray, nir_map01: np.ndarray, rgb: np.ndarray, hand_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = []
    h, w = rgb.shape[:2]
    nir_img_resized = _resize_to((h, w), np.asarray(nir_image01, dtype=np.float32))
    nir_map_resized = _resize_to((h, w), np.asarray(nir_map01, dtype=np.float32))

    rgb_gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    nir_u8 = _to_u8(nir_img_resized)
    rgb_u8 = (rgb_gray * 255.0 + 0.5).astype(np.uint8)

    dst_mask_u8 = (hand_mask.astype(np.uint8) * 255) if hand_mask is not None else None
    if dst_mask_u8 is None:
        warnings.append("No RGB hand mask for NIR registration; using identity transform.")
        return nir_map_resized.astype(np.float32), {"method": "identity", "status": "warning", "ecc_score": None}, tuple(warnings)

    dst_boundary01 = _mask_boundary01(dst_mask_u8)
    eval_mask_u8 = _dilate_mask_u8(dst_mask_u8, ksize=25, iterations=1)

    src_mask_bool = _hand_mask_from_gray01(nir_img_resized, ref_mask_u8=dst_mask_u8)
    if src_mask_bool is None:
        warnings.append("Failed to estimate NIR hand mask; using edge-based alignment.")
    src_mask_u8 = (src_mask_bool.astype(np.uint8) * 255) if src_mask_bool is not None else None

    coarse_warp: np.ndarray | None = None
    coarse_info: dict[str, Any] = {"method": "coarse_similarity", "status": "skipped"}
    if src_mask_u8 is not None and dst_mask_u8 is not None:
        coarse_warp, coarse_info = _coarse_similarity_from_masks(src_mask_u8, dst_mask_u8)

    src_edge01 = _mask_boundary01(src_mask_u8) if src_mask_u8 is not None else _nir_edges01(nir_u8)

    # ORB can help on some pairs; use per-modality masks (RGB mask != NIR mask when misaligned).
    mask_src_for_orb = _dilate_mask_u8(src_mask_u8, ksize=31, iterations=1) if src_mask_u8 is not None else None
    mask_dst_for_orb = _dilate_mask_u8(dst_mask_u8, ksize=31, iterations=1) if dst_mask_u8 is not None else None
    mtx_orb, orb_q = _orb_affine(nir_u8, rgb_u8, mask_src_u8=mask_src_for_orb, mask_dst_u8=mask_dst_for_orb)

    # Be conservative with ORB: only consider it as a candidate if it has strong support.
    orb_candidate: np.ndarray | None = None
    if mtx_orb is not None:
        inlier_ratio = float(orb_q.get("inlier_ratio", 0.0) or 0.0)
        matches_good = int(orb_q.get("matches_good", 0) or 0)
        if inlier_ratio >= 0.35 and matches_good >= 40:
            orb_candidate = mtx_orb

    init_warp, sel_info = _select_best_warp(
        out_shape_hw=(h, w),
        src_mask_u8=src_mask_u8,
        dst_mask_u8=dst_mask_u8,
        src_edge01=src_edge01,
        dst_edge01=dst_boundary01,
        eval_mask_u8=eval_mask_u8,
        candidates={"coarse": coarse_warp, "orb": orb_candidate},
    )

    warp_ecc, ecc_score = _ecc_refine(
        src=src_edge01,
        dst=dst_boundary01,
        initial_warp=init_warp,
        mask_u8=eval_mask_u8,
        warp_mode=cv2.MOTION_AFFINE,
        iterations=150,
        eps=1e-6,
    )

    use_warp = init_warp
    ecc_accepted = False
    if warp_ecc is not None and ecc_score is not None and np.isfinite(warp_ecc).all():
        warped_edge = _warp_affine(src_edge01, warp_ecc, (h, w))
        edge_iou = _edge_iou01(warped_edge, dst_boundary01, mask_u8=eval_mask_u8)

        pick_by = str(sel_info.get("selected_by", "edge_iou"))
        if pick_by == "mask_iou" and src_mask_u8 is not None:
            warped_mask = _warp_mask_u8(src_mask_u8, warp_ecc, (h, w))
            mask_iou = _mask_iou_u8(warped_mask, dst_mask_u8)
            base = float(sel_info.get("mask_iou", {}).get(sel_info.get("selected", "identity"), 0.0))
            ecc_accepted = mask_iou >= (base - 0.02)
        else:
            base = float(sel_info.get("edge_iou", {}).get(sel_info.get("selected", "identity"), 0.0))
            ecc_accepted = edge_iou >= (base - 0.02)

        if ecc_accepted:
            use_warp = warp_ecc

    nir_out = _warp_affine(nir_map_resized, use_warp, (h, w))

    method = f"{sel_info.get('selected', 'identity')}" + ("+ecc" if ecc_accepted else "_affine")
    status = "ok" if method != "identity_affine" else "warning"
    quality: dict[str, Any] = dict(orb_q)
    quality.update(
        {
            "method": method,
            "status": status,
            "ecc_score": float(ecc_score) if ecc_accepted and ecc_score is not None else None,
            "selection": sel_info,
            "coarse_init": coarse_info,
        }
    )

    return nir_out.astype(np.float32), quality, tuple(warnings)


def register_thermal_to_rgb(
    thermal_map01: np.ndarray, rgb: np.ndarray, hand_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = []
    h, w = rgb.shape[:2]
    therm_resized = _resize_to((h, w), np.asarray(thermal_map01, dtype=np.float32))

    dst_mask_u8 = (hand_mask.astype(np.uint8) * 255) if hand_mask is not None else None
    if dst_mask_u8 is None:
        warnings.append("No RGB hand mask for thermal registration; using identity transform.")
        return therm_resized.astype(np.float32), {"method": "identity", "status": "warning", "ecc_score": None}, tuple(warnings)

    dst_boundary01 = _mask_boundary01(dst_mask_u8)
    eval_mask_u8 = _dilate_mask_u8(dst_mask_u8, ksize=25, iterations=1)

    src_mask_bool = _hand_mask_from_gray01(therm_resized, ref_mask_u8=dst_mask_u8)
    if src_mask_bool is None:
        warnings.append("Failed to estimate thermal hand mask; using edge-based alignment.")
    src_mask_u8 = (src_mask_bool.astype(np.uint8) * 255) if src_mask_bool is not None else None

    coarse_warp: np.ndarray | None = None
    coarse_info: dict[str, Any] = {"method": "coarse_similarity", "status": "skipped"}
    if src_mask_u8 is not None:
        coarse_warp, coarse_info = _coarse_similarity_from_masks(src_mask_u8, dst_mask_u8)

    therm_u8 = _to_u8(therm_resized)
    src_edge01 = _mask_boundary01(src_mask_u8) if src_mask_u8 is not None else _thermal_edges01(therm_u8)

    init_warp, sel_info = _select_best_warp(
        out_shape_hw=(h, w),
        src_mask_u8=src_mask_u8,
        dst_mask_u8=dst_mask_u8,
        src_edge01=src_edge01,
        dst_edge01=dst_boundary01,
        eval_mask_u8=eval_mask_u8,
        candidates={"coarse": coarse_warp},
    )

    warp_ecc, ecc_score = _ecc_refine(
        src=src_edge01,
        dst=dst_boundary01,
        initial_warp=init_warp,
        mask_u8=eval_mask_u8,
        warp_mode=cv2.MOTION_AFFINE,
        iterations=150,
        eps=1e-6,
    )

    use_warp = init_warp
    ecc_accepted = False
    if warp_ecc is not None and ecc_score is not None and np.isfinite(warp_ecc).all():
        warped_edge = _warp_affine(src_edge01, warp_ecc, (h, w))
        edge_iou = _edge_iou01(warped_edge, dst_boundary01, mask_u8=eval_mask_u8)

        pick_by = str(sel_info.get("selected_by", "edge_iou"))
        if pick_by == "mask_iou" and src_mask_u8 is not None:
            warped_mask = _warp_mask_u8(src_mask_u8, warp_ecc, (h, w))
            mask_iou = _mask_iou_u8(warped_mask, dst_mask_u8)
            base = float(sel_info.get("mask_iou", {}).get(sel_info.get("selected", "identity"), 0.0))
            ecc_accepted = mask_iou >= (base - 0.02)
        else:
            base = float(sel_info.get("edge_iou", {}).get(sel_info.get("selected", "identity"), 0.0))
            ecc_accepted = edge_iou >= (base - 0.02)

        if ecc_accepted:
            use_warp = warp_ecc

    therm_out = _warp_affine(therm_resized, use_warp, (h, w))
    method = f"{sel_info.get('selected', 'identity')}" + ("+ecc" if ecc_accepted else "_affine")
    status = "ok" if method != "identity_affine" else "warning"
    return therm_out.astype(np.float32), {
        "method": method,
        "status": status,
        "ecc_score": float(ecc_score) if ecc_accepted and ecc_score is not None else None,
        "selection": sel_info,
        "coarse_init": coarse_info,
    }, tuple(warnings)


__all__ = ["register_nir_to_rgb", "register_thermal_to_rgb"]

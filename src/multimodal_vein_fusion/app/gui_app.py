from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from multimodal_vein_fusion.app.camera_sources import CameraBase, LeptonThermal, OpenCVCamera, UVCThermal
from multimodal_vein_fusion.app.state import AppState
from multimodal_vein_fusion.io import load_nir, load_rgb, load_thermal, save_outputs
from multimodal_vein_fusion.processing import PipelineConfig, PipelineResult, run_pipeline

logger = logging.getLogger(__name__)

try:  # optional
    import skimage  # noqa: F401

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


@dataclass
class _DisplayedImages:
    rgb: Optional[ImageTk.PhotoImage] = None
    nir: Optional[ImageTk.PhotoImage] = None
    thermal: Optional[ImageTk.PhotoImage] = None
    fused: Optional[ImageTk.PhotoImage] = None
    mask: Optional[ImageTk.PhotoImage] = None
    overlay: Optional[ImageTk.PhotoImage] = None


class _ImagePane:
    def __init__(self, master: tk.Widget, title: str, *, width: int = 320, height: int = 240) -> None:
        self.width = int(width)
        self.height = int(height)
        self.frame = ttk.LabelFrame(master, text=title)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self._label = ttk.Label(self.frame, anchor="center", justify="center")
        self._label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._photo: Optional[ImageTk.PhotoImage] = None

    def clear(self, text: str = "") -> None:
        self._photo = None
        self._label.configure(image="", text=text)

    def set_pil(self, img: Image.Image) -> None:
        img = img.copy()
        img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.width, self.height), (20, 20, 20))
        x = (self.width - img.size[0]) // 2
        y = (self.height - img.size[1]) // 2
        canvas.paste(img.convert("RGB"), (x, y))
        self._photo = ImageTk.PhotoImage(canvas)
        self._label.configure(image=self._photo, text="")

    def set_array(self, arr: np.ndarray, *, kind: str) -> None:
        if kind == "rgb":
            img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
            self.set_pil(img)
            return

        if kind in {"gray", "map"}:
            u8 = _to_u8_gray(arr)
            if kind == "map":
                color = _apply_colormap(u8)
                self.set_pil(Image.fromarray(color, mode="RGB"))
            else:
                self.set_pil(Image.fromarray(u8, mode="L").convert("RGB"))
            return

        if kind == "mask":
            u8 = (arr.astype(bool).astype(np.uint8) * 255) if arr.dtype != np.uint8 else arr
            self.set_pil(Image.fromarray(u8, mode="L").convert("RGB"))
            return

        raise ValueError(f"Unknown kind: {kind}")


def _to_u8_gray(x: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
    xf = np.clip(xf, 0.0, 1.0)
    return (xf * 255.0 + 0.5).astype(np.uint8)


def _robust_norm01(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(xf)
    if not np.any(finite):
        return np.zeros_like(xf, dtype=np.float32)
    lo, hi = np.percentile(xf[finite], [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(xf, dtype=np.float32)
    out = (xf - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_colormap(u8: np.ndarray) -> np.ndarray:
    try:
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    except Exception:
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def _nir_preview(nir01: np.ndarray) -> np.ndarray:
    nir = np.asarray(nir01, dtype=np.float32)
    if nir.ndim == 3 and nir.shape[2] == 3:
        nir = cv2.cvtColor(nir.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip(nir, 0.0, 1.0)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame] * 3, axis=2).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[:, :, :3].astype(np.uint8)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _to_gray01(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        g = frame.astype(np.float32)
        if g.max() > 1.5:
            g = g / 255.0
        return np.clip(g, 0.0, 1.0)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        g = cv2.cvtColor(frame[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return np.clip(g, 0.0, 1.0)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


class VeinFusionGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.state = AppState()
        self._display = _DisplayedImages()

        self.rgb_cam: CameraBase | None = None
        self.nir_cam: CameraBase | None = None
        self.thermal_cam: CameraBase | None = None
        self._live_after_id: str | None = None

        self._build_ui()
        self._update_buttons()

    def _build_ui(self) -> None:
        self.root.title("Multimodal Vein Fusion (RGB / NIR / Thermal)")
        self.root.geometry("1120x760")

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="both", expand=True)
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(8, 0))

        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(top)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.pane_rgb = _ImagePane(left, "RGB Input")
        self.pane_nir = _ImagePane(left, "NIR Input / Vesselness")
        self.pane_thermal = _ImagePane(left, "Thermal Input / Perfusion")
        self.pane_rgb.frame.pack(fill="both", expand=True)
        self.pane_nir.frame.pack(fill="both", expand=True)
        self.pane_thermal.frame.pack(fill="both", expand=True)

        self.pane_fused = _ImagePane(right, "Fused Map")
        self.pane_mask = _ImagePane(right, "Segmentation Mask")
        self.pane_overlay = _ImagePane(right, "RGB Overlay")
        self.pane_fused.frame.pack(fill="both", expand=True)
        self.pane_mask.frame.pack(fill="both", expand=True)
        self.pane_overlay.frame.pack(fill="both", expand=True)

        controls = ttk.Frame(bottom)
        controls.pack(side="left", fill="x", expand=True)

        self.btn_load_rgb = ttk.Button(controls, text="Load RGB", command=self._on_load_rgb)
        self.btn_load_nir = ttk.Button(controls, text="Load NIR", command=self._on_load_nir)
        self.btn_load_thermal = ttk.Button(controls, text="Load Thermal", command=self._on_load_thermal)
        self.btn_run = ttk.Button(controls, text="Run Pipeline", command=self._on_run_pipeline)
        self.btn_save = ttk.Button(controls, text="Save Outputs", command=self._on_save_outputs)
        self.btn_clear = ttk.Button(controls, text="Clear", command=self._on_clear)

        self.btn_load_rgb.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self.btn_load_nir.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.btn_load_thermal.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        self.btn_run.grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        self.btn_save.grid(row=0, column=4, padx=4, pady=4, sticky="ew")
        self.btn_clear.grid(row=0, column=5, padx=4, pady=4, sticky="ew")

        for c in range(6):
            controls.columnconfigure(c, weight=1)

        # Mode toggle + config
        self.live_var = tk.BooleanVar(value=False)
        self.chk_live = ttk.Checkbutton(
            controls,
            text="Mode: Files / Live",
            variable=self.live_var,
            command=self._on_toggle_mode,
        )
        self.chk_live.grid(row=1, column=0, padx=4, pady=(8, 4), sticky="w")

        config_frame = ttk.LabelFrame(controls, text="Config")
        config_frame.grid(row=1, column=1, columnspan=5, padx=4, pady=(8, 4), sticky="ew")
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(2, weight=1)
        config_frame.columnconfigure(3, weight=1)
        config_frame.columnconfigure(4, weight=1)

        self.w_nir = tk.DoubleVar(value=0.65)
        self.w_therm = tk.DoubleVar(value=0.25)
        self.w_edges = tk.DoubleVar(value=0.10)

        ttk.Label(config_frame, text="wNIR").grid(row=0, column=0, sticky="w")
        ttk.Label(config_frame, text="wThermal").grid(row=0, column=1, sticky="w")
        ttk.Label(config_frame, text="wEdges").grid(row=0, column=2, sticky="w")

        self.sld_w_nir = ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.w_nir)
        self.sld_w_therm = ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.w_therm)
        self.sld_w_edges = ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.w_edges)
        self.sld_w_nir.grid(row=1, column=0, sticky="ew", padx=4)
        self.sld_w_therm.grid(row=1, column=1, sticky="ew", padx=4)
        self.sld_w_edges.grid(row=1, column=2, sticky="ew", padx=4)

        self.min_area = tk.IntVar(value=150)
        ttk.Label(config_frame, text="Min area (px)").grid(row=0, column=3, sticky="w")
        self.spin_area = ttk.Spinbox(config_frame, from_=0, to=10000, increment=10, textvariable=self.min_area, width=8)
        self.spin_area.grid(row=1, column=3, sticky="w", padx=4)

        self.adaptive_var = tk.BooleanVar(value=True)
        self.chk_adaptive = ttk.Checkbutton(config_frame, text="Adaptive weighting", variable=self.adaptive_var)
        self.chk_adaptive.grid(row=0, column=4, rowspan=1, sticky="w", padx=4)

        self.skel_var = tk.BooleanVar(value=_HAS_SKIMAGE)
        self.chk_skel = ttk.Checkbutton(config_frame, text="Skeletonize (skimage)", variable=self.skel_var)
        self.chk_skel.grid(row=1, column=4, sticky="w", padx=4)
        if not _HAS_SKIMAGE:
            self.chk_skel.configure(state="disabled")
            self.skel_var.set(False)

        metrics_frame = ttk.LabelFrame(bottom, text="Metrics")
        metrics_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)

        self.txt_metrics = tk.Text(metrics_frame, height=10, width=60, wrap="word")
        self.txt_metrics.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.txt_metrics.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.txt_metrics.configure(yscrollcommand=scroll.set)

        self._set_metrics_text("Load RGB, NIR, and Thermal inputs (Files mode) or connect cameras (Live mode).")

        self.pane_rgb.clear("No RGB loaded")
        self.pane_nir.clear("No NIR loaded")
        self.pane_thermal.clear("No Thermal loaded")
        self.pane_fused.clear("Run pipeline to see fused map")
        self.pane_mask.clear("Run pipeline to see mask")
        self.pane_overlay.clear("Run pipeline to see overlay")

    def _set_metrics_text(self, text: str) -> None:
        self.txt_metrics.configure(state="normal")
        self.txt_metrics.delete("1.0", "end")
        self.txt_metrics.insert("1.0", text)
        self.txt_metrics.configure(state="disabled")
        self.state.last_metrics_text = text

    def _update_buttons(self) -> None:
        live = self.state.mode == "live"
        self.btn_load_rgb.configure(state=("disabled" if live else "normal"))
        self.btn_load_nir.configure(state=("disabled" if live else "normal"))
        self.btn_load_thermal.configure(state=("disabled" if live else "normal"))

        can_run = self.state.has_all_inputs() and not self.state.pipeline_running
        self.btn_run.configure(state=("normal" if can_run else "disabled"))
        self.btn_save.configure(state=("normal" if self.state.pipeline_result is not None and not self.state.pipeline_running else "disabled"))

    def _on_load_rgb(self) -> None:
        path = filedialog.askopenfilename(
            title="Select RGB image",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            rgb = load_rgb(path)
            self.state.rgb = rgb
            self.state.rgb_path = Path(path)
            self.pane_rgb.set_array(rgb, kind="rgb")
        except Exception as e:
            messagebox.showerror("Load RGB failed", str(e))
        finally:
            self._update_buttons()

    def _on_load_nir(self) -> None:
        path = filedialog.askopenfilename(
            title="Select NIR image",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            nir = load_nir(path)
            self.state.nir = nir
            self.state.nir_path = Path(path)
            self.pane_nir.set_array(_nir_preview(nir), kind="gray")
        except Exception as e:
            messagebox.showerror("Load NIR failed", str(e))
        finally:
            self._update_buttons()

    def _on_load_thermal(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Thermal file",
            filetypes=[("Thermal (.npy/.png)", "*.npy *.png"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            res = load_thermal(path)
            self.state.thermal = res.array
            self.state.thermal_path = Path(path)
            if res.warnings:
                self.state.loader_warnings.extend(list(res.warnings))
                messagebox.showwarning("Thermal load warning", "\n".join(res.warnings))
            self.pane_thermal.set_array(_robust_norm01(res.array), kind="map")
        except Exception as e:
            messagebox.showerror("Load Thermal failed", str(e))
        finally:
            self._update_buttons()

    def _on_clear(self) -> None:
        self.state.clear()
        self.pane_rgb.clear("No RGB loaded")
        self.pane_nir.clear("No NIR loaded")
        self.pane_thermal.clear("No Thermal loaded")
        self.pane_fused.clear("Run pipeline to see fused map")
        self.pane_mask.clear("Run pipeline to see mask")
        self.pane_overlay.clear("Run pipeline to see overlay")
        self._set_metrics_text("Cleared. Load inputs to run.")
        self._update_buttons()

    def _on_toggle_mode(self) -> None:
        want_live = bool(self.live_var.get())
        if want_live and self.state.mode != "live":
            self._enter_live_mode()
        elif not want_live and self.state.mode != "files":
            self._exit_live_mode()
        self._update_buttons()

    def _enter_live_mode(self) -> None:
        self.state.mode = "live"
        self.state.pipeline_result = None
        self._set_metrics_text("Live mode: opening cameras… (RGB=0, NIR=1, Thermal=2; Lepton optional)")
        self._open_cameras()
        self._schedule_live_update()

    def _exit_live_mode(self) -> None:
        self.state.mode = "files"
        self._stop_live_update()
        self._release_cameras()
        self.state.live_rgb = None
        self.state.live_nir = None
        self.state.live_thermal = None
        self._set_metrics_text("Files mode: load RGB/NIR/Thermal from disk.")

        if self.state.rgb is not None:
            self.pane_rgb.set_array(self.state.rgb, kind="rgb")
        else:
            self.pane_rgb.clear("No RGB loaded")
        if self.state.nir is not None:
            self.pane_nir.set_array(_nir_preview(self.state.nir), kind="gray")
        else:
            self.pane_nir.clear("No NIR loaded")
        if self.state.thermal is not None:
            self.pane_thermal.set_array(_robust_norm01(self.state.thermal), kind="map")
        else:
            self.pane_thermal.clear("No Thermal loaded")

    def _open_cameras(self) -> None:
        self._release_cameras()

        self.rgb_cam = OpenCVCamera(index=0, name="RGB")
        self.nir_cam = OpenCVCamera(index=1, name="NIR")
        # Thermal can be a UVC camera or Lepton if available; try UVC index=2 first, then Lepton.
        self.thermal_cam = UVCThermal(index=2, name="Thermal")

        if not self.rgb_cam.open():
            self.rgb_cam = None
        if not self.nir_cam.open():
            self.nir_cam = None
        if not self.thermal_cam.open():
            self.thermal_cam = None
            lepton = LeptonThermal()
            if lepton.open():
                self.thermal_cam = lepton

        self._update_live_panes_no_camera()

    def _release_cameras(self) -> None:
        for cam in (self.rgb_cam, self.nir_cam, self.thermal_cam):
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
        self.rgb_cam = None
        self.nir_cam = None
        self.thermal_cam = None

    def _update_live_panes_no_camera(self) -> None:
        if self.rgb_cam is None:
            self.pane_rgb.clear("No camera connected")
        if self.nir_cam is None:
            self.pane_nir.clear("No camera connected")
        if self.thermal_cam is None:
            self.pane_thermal.clear("No camera connected")

    def _schedule_live_update(self) -> None:
        self._stop_live_update()
        self._live_after_id = self.root.after(80, self._live_tick)

    def _stop_live_update(self) -> None:
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except Exception:
                pass
        self._live_after_id = None

    def _live_tick(self) -> None:
        if self.state.mode != "live":
            return

        try:
            if self.rgb_cam is not None:
                frame = self.rgb_cam.read()
                if frame is not None:
                    rgb = _ensure_rgb(frame)
                    self.state.live_rgb = rgb
                    self.pane_rgb.set_array(rgb, kind="rgb")
            if self.nir_cam is not None:
                frame = self.nir_cam.read()
                if frame is not None:
                    nir01 = _to_gray01(frame)
                    self.state.live_nir = nir01
                    self.pane_nir.set_array(nir01, kind="gray")
            if self.thermal_cam is not None:
                frame = self.thermal_cam.read()
                if frame is not None:
                    if frame.ndim == 2:
                        therm = frame.astype(np.float32)
                    else:
                        therm = cv2.cvtColor(frame[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                    self.state.live_thermal = therm
                    self.pane_thermal.set_array(_robust_norm01(therm), kind="map")

            self._update_buttons()
        finally:
            self._schedule_live_update()

    def _on_run_pipeline(self) -> None:
        if self.state.pipeline_running:
            return
        if not self.state.has_all_inputs():
            messagebox.showerror("Missing input", "RGB, NIR, and Thermal inputs are required.")
            self._update_buttons()
            return

        rgb, nir, therm = self.state.current_inputs()
        if rgb is None or nir is None or therm is None:
            messagebox.showerror("Missing input", "RGB, NIR, and Thermal inputs are required.")
            self._update_buttons()
            return

        config = PipelineConfig(
            w_nir=float(self.w_nir.get()),
            w_thermal=float(self.w_therm.get()),
            w_edges=float(self.w_edges.get()),
            min_area=int(self.min_area.get()),
            adaptive_weighting=bool(self.adaptive_var.get()),
            skeletonize=bool(self.skel_var.get()),
        )

        self.state.pipeline_running = True
        self.state.pipeline_result = None
        self._set_metrics_text("Running pipeline…")
        self._update_buttons()

        thread = threading.Thread(
            target=self._pipeline_worker,
            args=(rgb.copy(), np.asarray(nir).copy(), np.asarray(therm).copy(), config),
            daemon=True,
        )
        thread.start()

    def _pipeline_worker(self, rgb: np.ndarray, nir: np.ndarray, thermal: np.ndarray, config: PipelineConfig) -> None:
        try:
            result = run_pipeline(rgb=rgb, nir=nir, thermal=thermal, config=config)
            error = None
        except Exception as e:
            logger.exception("Pipeline failed")
            result = None
            error = str(e)

        self.root.after(0, lambda: self._on_pipeline_done(result, error))

    def _on_pipeline_done(self, result: PipelineResult | None, error: str | None) -> None:
        self.state.pipeline_running = False
        if error is not None or result is None:
            self.state.last_error = error or "Unknown error"
            messagebox.showerror("Pipeline failed", self.state.last_error)
            self._set_metrics_text(f"Pipeline failed:\n{self.state.last_error}")
            self._update_buttons()
            return

        self.state.pipeline_result = result

        # Update panes: show enhanced NIR and perfusion (registered) on the left
        self.pane_nir.set_array(result.nir_vesselness, kind="map")
        self.pane_thermal.set_array(result.thermal_perfusion, kind="map")

        self.pane_fused.set_array(result.fused_map, kind="map")
        self.pane_mask.set_array(result.display_mask, kind="mask")
        self.pane_overlay.set_array(result.overlay, kind="rgb")

        self._set_metrics_text(self._format_metrics(result))
        self._update_buttons()

    def _format_metrics(self, result: PipelineResult) -> str:
        m = result.metrics or {}
        w = m.get("weights_used", {})
        cov = m.get("coverage_pct", None)
        cnr = m.get("cnr_proxy", None)
        largest = m.get("largest_component", {})
        ins = m.get("recommended_insertion_point", None)
        rq = m.get("registration_quality", {})
        nir_q = rq.get("nir_to_rgb", {})
        th_q = rq.get("thermal_to_rgb", {})

        lines = []
        lines.append("Weights used:")
        lines.append(f"  NIR={w.get('nir')}  Thermal={w.get('thermal')}  Edges={w.get('edges')}")
        lines.append("")
        lines.append("Segmentation / metrics:")
        lines.append(f"  Coverage %: {cov}")
        lines.append(f"  CNR proxy: {cnr}")
        lines.append(f"  Largest component area (px): {largest.get('area_px')}")
        lines.append(f"  Largest length proxy ({largest.get('length_kind')}): {largest.get('length_proxy')}")
        if ins is not None:
            lines.append(f"  Recommended insertion point: (x={ins.get('x')}, y={ins.get('y')}), value={ins.get('value')}")
        lines.append("")
        lines.append("Registration quality:")
        lines.append(
            f"  NIR: method={nir_q.get('method')} matches={nir_q.get('matches_good')} inliers={nir_q.get('inliers')} ecc={nir_q.get('ecc_score')}"
        )
        lines.append(f"  Thermal: method={th_q.get('method')} ecc={th_q.get('ecc_score')}")

        if result.warnings:
            lines.append("")
            lines.append("Warnings:")
            for wmsg in result.warnings:
                lines.append(f"  - {wmsg}")
        if self.state.loader_warnings:
            lines.append("")
            lines.append("Load warnings:")
            for wmsg in self.state.loader_warnings:
                lines.append(f"  - {wmsg}")
        return "\n".join(lines)

    def _on_save_outputs(self) -> None:
        if self.state.pipeline_result is None:
            messagebox.showerror("Nothing to save", "Run the pipeline first.")
            return

        out_dir = filedialog.askdirectory(title="Select output folder")
        if not out_dir:
            return

        result = self.state.pipeline_result
        metrics = dict(result.metrics)
        if result.warnings:
            metrics["warnings"] = list(result.warnings)
        if self.state.loader_warnings:
            metrics["load_warnings"] = list(self.state.loader_warnings)

        try:
            save_outputs(
                out_dir,
                rgb=result.rgb,
                nir_vesselness=result.nir_vesselness,
                thermal_perfusion=result.thermal_perfusion,
                fused_map=result.fused_map,
                vein_mask=result.display_mask,
                overlay=result.overlay,
                metrics=metrics,
            )
            messagebox.showinfo("Saved", f"Saved outputs to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = tk.Tk()
    app = VeinFusionGUI(root)

    def _on_close() -> None:
        try:
            app._stop_live_update()
            app._release_cameras()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


__all__ = ["main", "VeinFusionGUI"]


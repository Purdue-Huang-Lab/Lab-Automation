"""
Paper-style spin subtraction GUI built with Qt and pyqtgraph.

The scientific definition is unchanged from the original Matplotlib version:
    spin contrast = (Left - Right) / (Left + Right + eps)
with denominator masking where abs(Left + Right) < denom_min.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg
from scipy.optimize import curve_fit
from scipy.signal import convolve2d

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets


QT_DASH_LINE = getattr(getattr(QtCore.Qt, "PenStyle", QtCore.Qt), "DashLine")
QT_DOT_LINE = getattr(getattr(QtCore.Qt, "PenStyle", QtCore.Qt), "DotLine")
QT_LEFT_BUTTON = getattr(getattr(QtCore.Qt, "MouseButton", QtCore.Qt), "LeftButton")
QT_HORIZONTAL = getattr(getattr(QtCore.Qt, "Orientation", QtCore.Qt), "Horizontal")
QT_KEY_RIGHT = getattr(getattr(QtCore.Qt, "Key", QtCore.Qt), "Key_Right")
QT_KEY_LEFT = getattr(getattr(QtCore.Qt, "Key", QtCore.Qt), "Key_Left")
QT_CHECKED = getattr(getattr(QtCore.Qt, "CheckState", QtCore.Qt), "Checked")
FRAME_STYLED_PANEL = getattr(getattr(QtWidgets.QFrame, "Shape", QtWidgets.QFrame), "StyledPanel")


@dataclass
class Config:
    """Application configuration and analysis defaults."""

    filename_left_img: str = "ND300HW290w 1 ND 1sum.txt"
    filename_left_time: str = "ND200HW245w 1 ND 1para.txt"
    filename_right_img: str = "ND300HW245w 1 ND 1sum.txt"
    filename_right_time: str = "ND200HW245w 1 ND 1para.txt"
    n1: int = 128
    n2: int = 480
    crop1: int = 0
    crop2: int = 10
    crop3: int = 0
    crop4: int = 128
    smooth_kernel_size: int = 8
    pixel1: float = 170.0
    wave1: float = 725.0
    pixel2: float = 400.0
    wave2: float = 775.0
    pixel_per_um: float = 1.0 / 10.0
    init_vmin: float = -10.0
    init_vmax: float = 10.0
    norm_vmin: float = -0.2
    norm_vmax: float = 0.2
    eps_init: float = 1e-9
    denom_min_init: float = 0.02
    sub_axis_min: float = 0.0
    sub_axis_max: float = 1.0
    background_threshold: float = 0.0
    auto_percentiles: tuple[float, float] = (2.0, 98.0)
    mode_options: tuple[str, ...] = ("norm", "difference", "left", "right")


@dataclass(frozen=True)
class ProcessedData:
    """Preprocessed reconstruction and calibrated axes."""

    left_frames: np.ndarray
    right_frames: np.ndarray
    times_ps: np.ndarray
    wavelength_nm: np.ndarray
    position_um: np.ndarray


@dataclass
class TracePayload:
    """Data exported from the currently selected point."""

    time_ps: np.ndarray
    left: np.ndarray
    right: np.ndarray
    display: np.ndarray
    spin_norm: np.ndarray
    subtraction: np.ndarray
    subtraction_01: np.ndarray
    mode: str
    x_idx: int
    y_idx: int
    wavelength_nm: float
    position_um: float


@dataclass
class AppState:
    """Mutable UI state."""

    mode: str
    eps: float
    denom_min: float
    auto_scale: bool
    x_idx: int
    y_idx: int
    time_index: int = 0
    last_trace_payload: TracePayload | None = None
    updating_time_box: bool = False


def find_closest_index(array: np.ndarray, value: float) -> int:
    """Return the index of the array element closest to value."""

    return int(np.argmin(np.abs(np.asarray(array, dtype=float) - float(value))))


def line_equation(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Return slope and intercept of the line through two points."""

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def robust_limits(
    data: np.ndarray,
    qlow: float,
    qhigh: float,
    fallback_vmin: float,
    fallback_vmax: float,
) -> tuple[float, float]:
    """Compute percentile-based image limits with safe fallbacks."""

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback_vmin, fallback_vmax
    lo, hi = np.percentile(arr, [qlow, qhigh])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return fallback_vmin, fallback_vmax
    return float(lo), float(hi)


def normalize_to_01(values: np.ndarray) -> np.ndarray:
    """Normalize finite values to [0, 1] and keep invalid values at 0."""

    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if np.isclose(lo, hi):
        return np.zeros_like(arr, dtype=float)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def compute_spin_norm(left: np.ndarray, right: np.ndarray, eps: float, denom_min: float) -> np.ndarray:
    """Compute masked spin contrast."""

    numerator = left - right
    denominator = left + right
    mask = np.abs(denominator) >= denom_min
    out = np.zeros_like(denominator, dtype=float)
    out[mask] = numerator[mask] / (denominator[mask] + eps)
    return out


def compute_spin_norm_cut(
    left: np.ndarray,
    right: np.ndarray,
    eps: float,
    denom_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spin contrast and keep only the valid masked region."""

    numerator = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    denominator = np.asarray(left, dtype=float) + np.asarray(right, dtype=float)
    mask = np.abs(denominator) >= denom_min
    out = np.full_like(denominator, np.nan, dtype=float)
    out[mask] = numerator[mask] / (denominator[mask] + eps)
    return out, mask


def apply_mask_cut(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only values inside a mask and mark the rest as NaN."""

    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    out[np.asarray(mask, dtype=bool)] = arr[np.asarray(mask, dtype=bool)]
    return out


def compute_display(left: np.ndarray, right: np.ndarray, mode: str, eps: float, denom_min: float) -> np.ndarray:
    """Select the current display mode."""

    if mode == "left":
        return left
    if mode == "right":
        return right
    if mode == "difference":
        return left - right
    return compute_spin_norm(left, right, eps, denom_min)


def gaussian_with_offset(x: np.ndarray, amplitude: float, mean: float, stddev: float, offset: float) -> np.ndarray:
    """Gaussian profile with constant offset using the legacy TAM parameterization."""

    sigma = max(abs(float(stddev)), 1e-12)
    return amplitude * np.exp(-((x - mean) / (2.0 * sigma)) ** 2) + offset


def extract_spatial_fit_profile(frame: np.ndarray, x_idx: int, average_half_width: int) -> np.ndarray:
    """Average a small x-window to create the spatial profile used for Gaussian fitting."""

    x_lo = max(0, int(x_idx) - int(average_half_width))
    x_hi = min(frame.shape[1], int(x_idx) + int(average_half_width) + 1)
    return np.mean(frame[:, x_lo:x_hi], axis=1)


class DataLoader:
    """File loading layer for raw image and time data."""

    def __init__(self, script_dir: Path) -> None:
        self.script_dir = script_dir

    def resolve_data_path(self, name: str) -> Path:
        """Resolve relative filenames against the script directory."""

        path = Path(name)
        if path.is_absolute():
            return path
        return self.script_dir / path

    def load_time_axis(self, time_path: Path) -> np.ndarray:
        """Load the time axis from the first non-empty line of the legacy para file."""

        with time_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                values = [token for token in stripped.split("\t") if token.strip()]
                if not values:
                    continue
                try:
                    return np.asarray([float(token) for token in values], dtype=float)
                except ValueError as exc:
                    raise ValueError(f"Could not parse time axis from {time_path}") from exc
        raise ValueError(f"Time file is empty or does not contain a valid time axis: {time_path}")

    def load_dataset(self, image_file: str, time_file: str) -> tuple[np.ndarray, np.ndarray]:
        """Load the raw stacked image array and its time axis."""

        image_path = self.resolve_data_path(image_file)
        time_path = self.resolve_data_path(time_file)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not time_path.exists():
            raise FileNotFoundError(f"Time file not found: {time_path}")

        img_df = pd.read_csv(image_path, delimiter="\t", header=None, on_bad_lines="skip")
        images = img_df.replace([np.nan], 0.0).values

        times = self.load_time_axis(time_path)
        return images, times

    def load_pair(self, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load left/right datasets and validate that they share a time axis."""

        images_left, times_left = self.load_dataset(config.filename_left_img, config.filename_left_time)
        images_right, times_right = self.load_dataset(config.filename_right_img, config.filename_right_time)
        if len(times_left) != len(times_right) or not np.allclose(times_left, times_right, rtol=0.0, atol=1e-9):
            raise ValueError("Left and right datasets must share the same time axis.")
        return images_left, images_right, times_left


class Preprocessor:
    """Raw-data reconstruction and coordinate calibration."""

    def __init__(self, config: Config) -> None:
        self.config = config
        k = config.smooth_kernel_size
        self.kernel = np.ones((k, k), dtype=np.float32) / float(k * k)

    def infer_available_frame_count(self, images: np.ndarray) -> int:
        """Infer how many full frames are present in the stacked image file."""

        cfg = self.config
        remainder = images.shape[0] % cfg.n2
        if remainder != 0:
            print(
                f"Warning: image stack has {images.shape[0]} rows which is not exactly divisible by "
                f"frame height {cfg.n2} (remainder {remainder}). Trailing rows will be ignored."
            )
        return images.shape[0] // cfg.n2

    def reconstruct_per_time(self, images: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Build a smoothed frame stack with shape (time, y, x)."""

        cfg = self.config
        available_frames = self.infer_available_frame_count(images)
        if available_frames <= 0:
            raise ValueError(
                f"Image stack has {images.shape[0]} rows, which is not enough for even one frame of height {cfg.n2}."
            )

        frame_count = min(len(times), available_frames)
        if frame_count != len(times):
            print(
                f"Warning: time axis has {len(times)} points but image stack contains {available_frames} frames. "
                f"Using the first {frame_count} frames."
            )
            times = times[:frame_count]

        frames: list[np.ndarray] = []
        for i in range(frame_count):
            start = i * cfg.n2
            raw_frame = np.rot90(
                images[
                    start + cfg.crop1 : start + cfg.n2 - cfg.crop2,
                    cfg.n1 - cfg.crop4 : cfg.n1 - cfg.crop3,
                ]
            )
            if raw_frame.size == 0:
                raise ValueError("Crop settings produced an empty frame.")
            smoothed = convolve2d(raw_frame, self.kernel, mode="same", boundary="symm")
            if cfg.background_threshold > 0.0:
                smoothed = np.where(smoothed < cfg.background_threshold, 0.0, smoothed)
            frames.append(smoothed.astype(np.float32, copy=False))
        return np.stack(frames, axis=0)

    def build_axes(self, example_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build wavelength and position axes from the calibrated frame size."""

        cfg = self.config
        slope, intercept = line_equation(cfg.pixel1, cfg.wave1, cfg.pixel2, cfg.wave2)
        pixvec = np.arange(example_frame.shape[1], dtype=float)
        final_pix = pixvec + cfg.crop1
        wavelength_nm = slope * final_pix + intercept

        pos_pixels = np.arange(example_frame.shape[0], dtype=float)
        offset = -0.5 * float(example_frame.shape[0])
        position_um = cfg.pixel_per_um * (pos_pixels + offset)
        return wavelength_nm, position_um

    def process(self, images_left: np.ndarray, images_right: np.ndarray, times_ps: np.ndarray) -> ProcessedData:
        """Generate the final analysis-ready frame stacks and calibrated axes."""

        usable_frame_count = min(
            len(times_ps),
            self.infer_available_frame_count(images_left),
            self.infer_available_frame_count(images_right),
        )
        if usable_frame_count <= 0:
            raise ValueError("No usable frames were found in the input image stacks.")

        used_times_ps = np.asarray(times_ps[:usable_frame_count], dtype=float)
        if usable_frame_count != len(times_ps):
            print(
                f"Warning: time axis has {len(times_ps)} points, but only {usable_frame_count} frames are usable across both stacks. "
                f"Truncating to the shared prefix."
            )

        left_frames = self.reconstruct_per_time(images_left, used_times_ps)
        right_frames = self.reconstruct_per_time(images_right, used_times_ps)
        wavelength_nm, position_um = self.build_axes(left_frames[0])
        return ProcessedData(
            left_frames=left_frames,
            right_frames=right_frames,
            times_ps=used_times_ps,
            wavelength_nm=wavelength_nm.astype(float, copy=False),
            position_um=position_um.astype(float, copy=False),
        )


class SpinPlotView(QtWidgets.QWidget):
    """Plotting layer that owns all pyqtgraph widgets."""

    def __init__(self, data: ProcessedData, config: Config, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.config = config
        self._image_click_callback: Callable[[int, int], None] | None = None

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        self._build_plots()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def _build_plots(self) -> None:
        """Create plots once so updates can reuse existing items."""

        height, width = self.data.left_frames.shape[1:]
        color_map = pg.colormap.get("viridis")

        self.image_plot = self.graphics.addPlot(row=0, col=0)
        self.image_plot.setLabel("bottom", "pixel (wavelength)")
        self.image_plot.setLabel("left", "pixel (space)")
        self.image_plot.showGrid(x=False, y=False, alpha=0.2)
        self.image_plot.getViewBox().invertY(True)
        self.image_plot.setLimits(xMin=0.0, xMax=float(width), yMin=0.0, yMax=float(height))
        self.image_plot.setXRange(0.0, float(width), padding=0.0)
        self.image_plot.setYRange(0.0, float(height), padding=0.0)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setLookupTable(color_map.getLookupTable())
        self.image_plot.addItem(self.image_item)
        cross_pen = pg.mkPen(color="w", width=1.5, style=QT_DASH_LINE)
        self.image_hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.image_vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.image_plot.addItem(self.image_hline)
        self.image_plot.addItem(self.image_vline)
        self.image_plot.scene().sigMouseClicked.connect(self._handle_scene_click)

        self.wavelength_plot = self.graphics.addPlot(row=0, col=1)
        self.wavelength_plot.setLabel("bottom", "Wavelength (nm)")
        self.wavelength_plot.setLabel("left", "Intensity")
        self.wavelength_plot.showGrid(x=True, y=True, alpha=0.2)
        self.wavelength_curve = self.wavelength_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.wavelength_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.wavelength_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.wavelength_plot.addItem(self.wavelength_hline)
        self.wavelength_plot.addItem(self.wavelength_vline)
        self.wavelength_plot.setXRange(
            float(self.data.wavelength_nm[0]),
            float(self.data.wavelength_nm[-1]),
            padding=0.0,
        )

        self.spatial_plot = self.graphics.addPlot(row=1, col=0)
        self.spatial_plot.setLabel("bottom", "Space (um)")
        self.spatial_plot.setLabel("left", "Intensity")
        self.spatial_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spatial_curve = self.spatial_plot.plot(
            pen=None, symbol="o", symbolSize=5,
            symbolPen=(200, 50, 50), symbolBrush=(200, 50, 50),
        )
        self.spatial_fit_curve = self.spatial_plot.plot(pen=pg.mkPen((0, 0, 0), width=2))
        self.spatial_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.spatial_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.spatial_plot.addItem(self.spatial_hline)
        self.spatial_plot.addItem(self.spatial_vline)
        self.spatial_plot.setXRange(
            float(self.data.position_um[0]),
            float(self.data.position_um[-1]),
            padding=0.0,
        )

        self.time_trace_plot = self.graphics.addPlot(row=1, col=1)
        self.time_trace_plot.setLabel("bottom", "Time (ps)")
        self.time_trace_plot.setLabel("left", "Intensity")
        self.time_trace_plot.showGrid(x=True, y=True, alpha=0.2)
        self.display_trace_curve = self.time_trace_plot.plot(pen=pg.mkPen("#2ca02c", width=2))
        self.left_trace_curve = self.time_trace_plot.plot(pen=pg.mkPen((31, 119, 180, 120), width=1))
        self.right_trace_curve = self.time_trace_plot.plot(pen=pg.mkPen((255, 127, 14, 120), width=1, style=QT_DASH_LINE))
        self.time_trace_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=1.2, style=QT_DASH_LINE))
        self.time_trace_plot.addItem(self.time_trace_marker)
        self.time_trace_plot.setXRange(float(self.data.times_ps[0]), float(self.data.times_ps[-1]), padding=0.0)

        self.sub_dynamics_plot = self.graphics.addPlot(row=2, col=0)
        self.sub_dynamics_plot.setLabel("bottom", "Time (ps)")
        self.sub_dynamics_plot.setLabel("left", "Normalized intensity")
        self.sub_dynamics_plot.showGrid(x=True, y=True, alpha=0.2)
        self.sub_dynamics_curve = self.sub_dynamics_plot.plot(pen=pg.mkPen("#9467bd", width=2))
        self.sub_dynamics_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=1.2, style=QT_DASH_LINE))
        self.sub_dynamics_plot.addItem(self.sub_dynamics_marker)
        self.sub_dynamics_plot.setXRange(float(self.data.times_ps[0]), float(self.data.times_ps[-1]), padding=0.0)
        self.sub_dynamics_plot.setYRange(self.config.sub_axis_min, self.config.sub_axis_max, padding=0.0)

        self.sub_spatial_plot = self.graphics.addPlot(row=2, col=1)
        self.sub_spatial_plot.setLabel("bottom", "Space (um)")
        self.sub_spatial_plot.setLabel("left", "Normalized intensity")
        self.sub_spatial_plot.showGrid(x=True, y=True, alpha=0.2)
        self.sub_spatial_curve = self.sub_spatial_plot.plot(pen=pg.mkPen("#2ca02c", width=2))
        self.sub_spatial_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.sub_spatial_plot.addItem(self.sub_spatial_marker)
        self.sub_spatial_plot.setXRange(
            float(self.data.position_um[0]),
            float(self.data.position_um[-1]),
            padding=0.0,
        )
        self.sub_spatial_plot.setYRange(self.config.sub_axis_min, self.config.sub_axis_max, padding=0.0)

    def set_image_click_callback(self, callback: Callable[[int, int], None]) -> None:
        """Register a callback for image clicks."""

        self._image_click_callback = callback

    def _handle_scene_click(self, event: object) -> None:
        """Map scene clicks back to image pixel coordinates."""

        if self._image_click_callback is None:
            return
        button = getattr(event, "button", lambda: None)()
        if button != QT_LEFT_BUTTON:
            return
        scene_pos = getattr(event, "scenePos", lambda: None)()
        if scene_pos is None:
            return
        if not self.image_plot.sceneBoundingRect().contains(scene_pos):
            return

        view_pos = self.image_plot.vb.mapSceneToView(scene_pos)
        width = self.data.left_frames.shape[2]
        height = self.data.left_frames.shape[1]
        x_idx = int(np.clip(np.round(view_pos.x()), 0, width - 1))
        y_idx = int(np.clip(np.round(view_pos.y()), 0, height - 1))
        self._image_click_callback(x_idx, y_idx)

    def update_image(self, frame: np.ndarray, mode: str, time_ps: float, vmin: float, vmax: float, x_idx: int, y_idx: int) -> None:
        """Update the main image and crosshairs."""

        self.image_item.setImage(frame, autoLevels=False)
        self.image_item.setLevels((float(vmin), float(vmax)))
        self.image_plot.setTitle(f"{mode} @ t={time_ps:.2f} ps")
        self.image_vline.setValue(float(x_idx))
        self.image_hline.setValue(float(y_idx))

    def update_wavelength_slice(
        self,
        values: np.ndarray,
        wavelength_nm: np.ndarray,
        mode: str,
        x_idx: int,
        y_idx: int,
    ) -> None:
        """Update the wavelength slice for the selected y position."""

        self.wavelength_plot.setTitle(f"Wavelength slice @ Y={y_idx}")
        self.wavelength_curve.setData(wavelength_nm, values)
        self.wavelength_vline.setValue(float(wavelength_nm[x_idx]))
        self.wavelength_hline.setValue(float(values[x_idx]))
        self.wavelength_curve.setPen(pg.mkPen("#1f77b4", width=2, cosmetic=True))

    def update_spatial_slice(
        self,
        values: np.ndarray,
        position_um: np.ndarray,
        mode: str,
        x_idx: int,
        y_idx: int,
    ) -> None:
        """Update the spatial slice for the selected x position."""

        self.spatial_plot.setTitle(f"Spatial slice @ X={x_idx}")
        self.spatial_curve.setData(position_um, values)
        self.spatial_vline.setValue(float(position_um[y_idx]))
        self.spatial_hline.setValue(float(values[y_idx]))

    def update_spatial_fit(self, position_um: np.ndarray, amplitude: float, mean: float, std: float, offset: float) -> None:
        """Draw Gaussian fit curve on the spatial plot using stored per-frame params."""

        if std > 0 and np.isfinite(std):
            x_fine = np.linspace(float(position_um[0]), float(position_um[-1]), 300)
            y_fit = gaussian_with_offset(x_fine, amplitude, mean, std, offset)
            self.spatial_fit_curve.setData(x_fine, y_fit)
        else:
            self.spatial_fit_curve.setData([], [])

    def update_time_trace(
        self,
        times_ps: np.ndarray,
        display_trace: np.ndarray,
        left_trace: np.ndarray,
        right_trace: np.ndarray,
        show_components: bool,
        mode: str,
        x_idx: int,
        y_idx: int,
        current_time: float,
    ) -> None:
        """Update the time-trace panel."""

        self.time_trace_plot.setTitle(f"Time trace @ (X={x_idx}, Y={y_idx})")
        self.display_trace_curve.setData(times_ps, display_trace)
        if show_components:
            self.left_trace_curve.setData(times_ps, left_trace)
            self.right_trace_curve.setData(times_ps, right_trace)
        else:
            self.left_trace_curve.setData([], [])
            self.right_trace_curve.setData([], [])
        self.time_trace_marker.setValue(float(current_time))

    def update_subtraction_dynamics(self, times_ps: np.ndarray, values: np.ndarray, current_time: float, x_idx: int, y_idx: int) -> None:
        """Update the normalized subtraction-vs-time plot."""

        self.sub_dynamics_plot.setTitle(f"Subtraction dynamics (Left-Right) @ (X={x_idx}, Y={y_idx})")
        self.sub_dynamics_curve.setData(times_ps, values)
        self.sub_dynamics_marker.setValue(float(current_time))

    def update_subtraction_spatial(self, position_um: np.ndarray, values: np.ndarray, current_time: float, y_idx: int) -> None:
        """Update the normalized subtraction-vs-space plot."""

        self.sub_spatial_plot.setTitle(f"Subtraction spatial (Left-Right) @ t={current_time:.2f} ps")
        self.sub_spatial_curve.setData(position_um, values)
        self.sub_spatial_marker.setValue(float(position_um[y_idx]))

    def export_png(self, out_path: Path) -> None:
        """Export the current widget view as a PNG."""

        pixmap = self.grab()
        pixmap.save(str(out_path), "PNG")


class NormCutPlotView(QtWidgets.QWidget):
    """Secondary tab showing norm-mode masked cuts."""

    def __init__(self, data: ProcessedData, config: Config, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.config = config
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        self._build_plots()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def _build_plots(self) -> None:
        self.cut_spatial_plot = self.graphics.addPlot(row=0, col=0)
        self.cut_spatial_plot.setLabel("bottom", "Wavelength (nm)")
        self.cut_spatial_plot.setLabel("left", "Norm cut")
        self.cut_spatial_plot.showGrid(x=True, y=True, alpha=0.2)
        self.cut_spatial_curve = self.cut_spatial_plot.plot(pen=pg.mkPen("#d62728", width=2))
        self.cut_spatial_fit_curve = self.cut_spatial_plot.plot(pen=pg.mkPen((0, 0, 0), width=2))
        self.cut_spatial_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.cut_spatial_plot.addItem(self.cut_spatial_marker)
        self.cut_spatial_plot.setXRange(float(self.data.wavelength_nm[0]), float(self.data.wavelength_nm[-1]), padding=0.0)

        self.cut_time_plot = self.graphics.addPlot(row=0, col=1)
        self.cut_time_plot.setLabel("bottom", "Time (ps)")
        self.cut_time_plot.setLabel("left", "Norm cut")
        self.cut_time_plot.showGrid(x=True, y=True, alpha=0.2)
        self.cut_time_curve = self.cut_time_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.cut_time_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=1.2, style=QT_DASH_LINE))
        self.cut_time_plot.addItem(self.cut_time_marker)
        self.cut_time_plot.setXRange(float(self.data.times_ps[0]), float(self.data.times_ps[-1]), padding=0.0)

        self.cut_sub_dynamics_plot = self.graphics.addPlot(row=1, col=0)
        self.cut_sub_dynamics_plot.setLabel("bottom", "Time (ps)")
        self.cut_sub_dynamics_plot.setLabel("left", "Normalized intensity")
        self.cut_sub_dynamics_plot.showGrid(x=True, y=True, alpha=0.2)
        self.cut_sub_dynamics_curve = self.cut_sub_dynamics_plot.plot(pen=pg.mkPen("#9467bd", width=2))
        self.cut_sub_dynamics_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=1.2, style=QT_DASH_LINE))
        self.cut_sub_dynamics_plot.addItem(self.cut_sub_dynamics_marker)
        self.cut_sub_dynamics_plot.setXRange(float(self.data.times_ps[0]), float(self.data.times_ps[-1]), padding=0.0)
        self.cut_sub_dynamics_plot.setYRange(self.config.sub_axis_min, self.config.sub_axis_max, padding=0.0)

        self.cut_sub_spatial_plot = self.graphics.addPlot(row=1, col=1)
        self.cut_sub_spatial_plot.setLabel("bottom", "Space (um)")
        self.cut_sub_spatial_plot.setLabel("left", "Normalized intensity")
        self.cut_sub_spatial_plot.showGrid(x=True, y=True, alpha=0.2)
        self.cut_sub_spatial_curve = self.cut_sub_spatial_plot.plot(pen=pg.mkPen("#2ca02c", width=2))
        self.cut_sub_spatial_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.cut_sub_spatial_plot.addItem(self.cut_sub_spatial_marker)
        self.cut_sub_spatial_plot.setXRange(float(self.data.position_um[0]), float(self.data.position_um[-1]), padding=0.0)
        self.cut_sub_spatial_plot.setYRange(self.config.sub_axis_min, self.config.sub_axis_max, padding=0.0)

    def clear(self) -> None:
        """Clear the norm-cut tab when not in norm mode."""

        self.cut_spatial_plot.setTitle("Spatial slice cut (norm mode only)")
        self.cut_time_plot.setTitle("Time trace cut (norm mode only)")
        self.cut_sub_dynamics_plot.setTitle("Subtraction dynamics cut (norm mode only)")
        self.cut_sub_spatial_plot.setTitle("Subtraction spatial cut (norm mode only)")
        self.cut_spatial_curve.setData([], [])
        self.cut_spatial_fit_curve.setData([], [])
        self.cut_time_curve.setData([], [])
        self.cut_sub_dynamics_curve.setData([], [])
        self.cut_sub_spatial_curve.setData([], [])

    def update_plots(
        self,
        *,
        wavelength_nm: np.ndarray,
        position_um: np.ndarray,
        times_ps: np.ndarray,
        wavelength_cut: np.ndarray,
        time_cut: np.ndarray,
        subtraction_time_cut: np.ndarray,
        subtraction_spatial_cut: np.ndarray,
        current_time: float,
        x_idx: int,
        y_idx: int,
    ) -> None:
        """Update all norm-cut plots using the current mask."""

        self.cut_spatial_plot.setTitle(f"Wavelength slice cut @ Y={y_idx}")
        self.cut_time_plot.setTitle(f"Time trace cut @ (X={x_idx}, Y={y_idx})")
        self.cut_sub_dynamics_plot.setTitle(f"Subtraction dynamics cut @ (X={x_idx}, Y={y_idx})")
        self.cut_sub_spatial_plot.setTitle(f"Subtraction spatial cut @ t={current_time:.2f} ps")

        self.cut_spatial_curve.setData(wavelength_nm, wavelength_cut)
        self.cut_time_curve.setData(times_ps, time_cut)
        self.cut_sub_dynamics_curve.setData(times_ps, subtraction_time_cut)
        self.cut_sub_spatial_curve.setData(position_um, subtraction_spatial_cut)

        self.cut_spatial_marker.setValue(float(wavelength_nm[x_idx]))
        self.cut_sub_spatial_marker.setValue(float(position_um[y_idx]))
        self.cut_time_marker.setValue(float(current_time))
        self.cut_sub_dynamics_marker.setValue(float(current_time))

    def clear_fit_overlay(self) -> None:
        """Clear the first norm-cut plot overlay."""

        self.cut_spatial_fit_curve.setData([], [])


class NormCutFitView(QtWidgets.QWidget):
    """Dedicated tab for Gaussian fitting on the masked subtraction spatial profile."""

    def __init__(self, data: ProcessedData, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        self._build_plots()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def _build_plots(self) -> None:
        self.fit_plot = self.graphics.addPlot(row=0, col=0)
        self.fit_plot.setLabel("bottom", "Space (um)")
        self.fit_plot.setLabel("left", "Normalized intensity")
        self.fit_plot.showGrid(x=True, y=True, alpha=0.2)
        self.fit_curve = self.fit_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolPen=(200, 50, 50),
            symbolBrush=(200, 50, 50),
        )
        self.fit_overlay_curve = self.fit_plot.plot(pen=pg.mkPen((0, 0, 0), width=2))
        self.fit_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=QT_DOT_LINE))
        self.fit_plot.addItem(self.fit_marker)
        self.fit_plot.setXRange(float(self.data.position_um[0]), float(self.data.position_um[-1]), padding=0.0)

    def clear(self) -> None:
        """Clear the norm-cut fit tab when not in norm mode."""

        self.fit_plot.setTitle("Norm cut Gaussian fit (norm mode only)")
        self.fit_curve.setData([], [])
        self.fit_overlay_curve.setData([], [])

    def update_profile(self, position_um: np.ndarray, spatial_cut: np.ndarray, x_idx: int, y_idx: int) -> None:
        """Update the masked subtraction spatial profile used for Gaussian fitting."""

        self.fit_plot.setTitle(f"Norm cut subtraction spatial profile @ X={x_idx}")
        self.fit_curve.setData(position_um, spatial_cut)
        self.fit_marker.setValue(float(position_um[y_idx]))

    def update_fit(self, position_um: np.ndarray, amplitude: float, mean: float, std: float, offset: float) -> None:
        """Draw Gaussian fit on the masked subtraction spatial profile."""

        if std > 0 and np.isfinite(std):
            x_fine = np.linspace(float(position_um[0]), float(position_um[-1]), 300)
            y_fit = gaussian_with_offset(x_fine, amplitude, mean, std, offset)
            self.fit_overlay_curve.setData(x_fine, y_fit)
        else:
            self.fit_overlay_curve.setData([], [])


class SpinSubtractionWindow(QtWidgets.QMainWindow):
    """Controller layer for the paper-style spin subtraction GUI."""

    def __init__(self, config: Config, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.script_dir = Path(__file__).resolve().parent
        self.loader = DataLoader(self.script_dir)
        self.preprocessor = Preprocessor(config)

        self._data_loaded = False
        try:
            self.data = self._load_processed_data()
            self._data_loaded = True
        except Exception:
            self.data = self._make_dummy_data()

        height, width = self.data.left_frames.shape[1:]
        self.state = AppState(
            mode="norm",
            eps=config.eps_init,
            denom_min=config.denom_min_init,
            auto_scale=True,
            x_idx=max(width - 10, 0),
            y_idx=max(height - 10, 0),
        )

        n = len(self.data.times_ps)
        self.G_AMP = np.zeros(n)
        self.G_MEAN = np.zeros(n)
        self.G_STD = np.zeros(n)
        self.G_OFFSET = np.zeros(n)
        self.G_AMP_ERR = np.zeros(n)
        self.G_MEAN_ERR = np.zeros(n)
        self.G_STD_ERR = np.zeros(n)
        self.G_OFFSET_ERR = np.zeros(n)
        self.G_CUT_AMP = np.zeros(n)
        self.G_CUT_MEAN = np.zeros(n)
        self.G_CUT_STD = np.zeros(n)
        self.G_CUT_OFFSET = np.zeros(n)
        self.G_CUT_AMP_ERR = np.zeros(n)
        self.G_CUT_MEAN_ERR = np.zeros(n)
        self.G_CUT_STD_ERR = np.zeros(n)
        self.G_CUT_OFFSET_ERR = np.zeros(n)

        self.setWindowTitle("Paper Spin GUI (pyqtgraph)")
        self.resize(1500, 980)
        self._build_ui()
        if self._data_loaded:
            self.refresh_view()
        else:
            self.statusBar().showMessage(
                "No data loaded - use Browse to select files, then click Load data."
            )

    def _load_processed_data(self) -> ProcessedData:
        """Load, validate, and preprocess the left/right datasets."""

        images_left, images_right, times_ps = self.loader.load_pair(self.config)
        return self.preprocessor.process(images_left, images_right, times_ps)

    @staticmethod
    def _make_dummy_data() -> ProcessedData:
        """Create minimal placeholder data when no files are loaded."""
        return ProcessedData(
            left_frames=np.zeros((1, 10, 10), dtype=np.float32),
            right_frames=np.zeros((1, 10, 10), dtype=np.float32),
            times_ps=np.array([0.0]),
            wavelength_nm=np.arange(10, dtype=float),
            position_um=np.arange(10, dtype=float),
        )

    def _build_ui(self) -> None:
        """Build controls, layouts, and signal wiring."""

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        controls = self._build_controls_panel()
        main_layout.addWidget(controls, 0)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        time_row = self._build_time_row()
        right_layout.addWidget(time_row)

        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_view = SpinPlotView(self.data, self.config)
        self.plot_view.set_image_click_callback(self.on_image_clicked)
        self.norm_cut_view = NormCutPlotView(self.data, self.config)
        self.norm_cut_fit_view = NormCutFitView(self.data)
        self.plot_tabs.addTab(self.plot_view, "Main")
        self.plot_tabs.addTab(self.norm_cut_view, "Norm cut")
        self.plot_tabs.addTab(self.norm_cut_fit_view, "Norm cut fit")
        right_layout.addWidget(self.plot_tabs, 1)

        main_layout.addWidget(right_panel, 1)
        self.statusBar().showMessage("Ready")

    def _build_time_row(self) -> QtWidgets.QWidget:
        """Create the time controls above the plots."""

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel("Time (ps):")
        self.time_box = QtWidgets.QLineEdit(f"{self.data.times_ps[0]:.2f}")
        self.time_box.setFixedWidth(110)
        self.time_box.setValidator(QtGui.QDoubleValidator())
        self.frame_label = QtWidgets.QLabel("Frame: 0")

        self.time_slider = QtWidgets.QSlider(QT_HORIZONTAL)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(len(self.data.times_ps) - 1)
        self.time_slider.setValue(0)
        self.time_slider.setSingleStep(1)
        self.time_slider.setPageStep(1)
        self.time_slider.setTickInterval(1)

        layout.addWidget(label)
        layout.addWidget(self.time_box)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.time_slider, 1)

        self.time_box.editingFinished.connect(self.on_time_box_submit)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        return widget

    def _build_controls_panel(self) -> QtWidgets.QWidget:
        """Create the control column."""

        panel = QtWidgets.QFrame()
        panel.setFrameShape(FRAME_STYLED_PANEL)
        panel.setMinimumWidth(340)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        files_group = QtWidgets.QGroupBox("Data Files")
        files_layout = QtWidgets.QVBoxLayout(files_group)
        for attr, label, placeholder in [
            ("path_left_img_edit", "Left img:", "Path to left image file"),
            ("path_left_time_edit", "Left para:", "Path to left time/para file"),
            ("path_right_img_edit", "Right img:", "Path to right image file"),
            ("path_right_time_edit", "Right para:", "Path to right time/para file"),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(placeholder)
            setattr(self, attr, edit)
            row.addWidget(edit, 1)
            btn = QtWidgets.QPushButton("Browse\u2026")
            btn.clicked.connect(
                lambda checked=False, e=edit, t=f"Select {label[:-1]}": self._browse_file(e, t)
            )
            row.addWidget(btn)
            files_layout.addLayout(row)

        self.btn_load_data = QtWidgets.QPushButton("Load data")
        self.btn_load_data.setStyleSheet("font-weight: bold;")
        self.btn_load_data.clicked.connect(self._apply_paths_and_load)
        files_layout.addWidget(self.btn_load_data)

        self._sync_path_edits_from_config()
        layout.addWidget(files_group)

        form = QtWidgets.QFormLayout()
        self.vmax_box = QtWidgets.QLineEdit(str(self.config.init_vmax))
        self.vmin_box = QtWidgets.QLineEdit(str(self.config.init_vmin))
        self.eps_box = QtWidgets.QLineEdit(f"{self.config.eps_init:.1e}")
        self.denom_box = QtWidgets.QLineEdit(f"{self.config.denom_min_init:.3g}")
        validator = QtGui.QDoubleValidator()
        for widget in (self.vmax_box, self.vmin_box, self.eps_box, self.denom_box):
            widget.setValidator(validator)
            widget.editingFinished.connect(self.refresh_view)
        form.addRow("Vmax", self.vmax_box)
        form.addRow("Vmin", self.vmin_box)
        form.addRow("eps", self.eps_box)
        form.addRow("|L+R|min", self.denom_box)
        layout.addLayout(form)

        self.set_denom_button = QtWidgets.QPushButton("Set denom")
        self.set_denom_button.clicked.connect(self.on_set_denom_clicked)
        layout.addWidget(self.set_denom_button)

        mode_group_box = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group_box)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_buttons: dict[str, QtWidgets.QRadioButton] = {}
        for mode in self.config.mode_options:
            button = QtWidgets.QRadioButton(mode)
            if mode == self.state.mode:
                button.setChecked(True)
            button.toggled.connect(self.on_mode_changed)
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            mode_layout.addWidget(button)
        layout.addWidget(mode_group_box)

        self.auto_scale_checkbox = QtWidgets.QCheckBox("Auto scale")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.stateChanged.connect(self.on_auto_scale_changed)
        layout.addWidget(self.auto_scale_checkbox)

        self.export_png_button = QtWidgets.QPushButton("Export PNG")
        self.export_csv_button = QtWidgets.QPushButton("Export trace CSV")
        self.export_dyn_button = QtWidgets.QPushButton("Export spin DY")
        self.export_png_button.clicked.connect(self.on_export_png)
        self.export_csv_button.clicked.connect(self.on_export_csv)
        self.export_dyn_button.clicked.connect(self.on_export_spin_dynamics)
        layout.addWidget(self.export_png_button)
        layout.addWidget(self.export_csv_button)
        layout.addWidget(self.export_dyn_button)

        gaussian_group = QtWidgets.QGroupBox("Gaussian Fit")
        gaussian_layout = QtWidgets.QFormLayout(gaussian_group)
        self.fit_window_box = QtWidgets.QLineEdit("3")
        self.fit_start_index_box = QtWidgets.QLineEdit("0")
        int_validator = QtGui.QIntValidator(0, 1000000, self)
        self.fit_window_box.setValidator(int_validator)
        self.fit_start_index_box.setValidator(int_validator)
        gaussian_layout.addRow("Half-width (px)", self.fit_window_box)
        gaussian_layout.addRow("Start frame", self.fit_start_index_box)
        layout.addWidget(gaussian_group)

        self.fit_current_button = QtWidgets.QPushButton("Fit current")
        self.fit_all_button = QtWidgets.QPushButton("Fit all times")
        self.export_fit_button = QtWidgets.QPushButton("Export Gaussian Fit")
        self.fit_current_button.clicked.connect(self.on_fit_current_gaussian)
        self.fit_all_button.clicked.connect(self.on_fit_all_gaussian)
        self.export_fit_button.clicked.connect(self.on_export_gaussian_fit)
        layout.addWidget(self.fit_current_button)
        layout.addWidget(self.fit_all_button)
        layout.addWidget(self.export_fit_button)

        self.gaussian_result_label = QtWidgets.QLabel("Gaussian fit: not computed")
        self.gaussian_result_label.setWordWrap(True)
        layout.addWidget(self.gaussian_result_label)

        info_box = QtWidgets.QLabel(
            "Click the image to move the analysis point.\n"
            "Use Left/Right arrow keys to move through time."
        )
        info_box.setWordWrap(True)
        layout.addWidget(info_box)
        layout.addStretch(1)
        return panel

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Support left/right keyboard stepping through time."""

        if event.key() == QT_KEY_RIGHT:
            self.time_slider.setValue(min(self.time_slider.value() + 1, self.time_slider.maximum()))
            return
        if event.key() == QT_KEY_LEFT:
            self.time_slider.setValue(max(self.time_slider.value() - 1, self.time_slider.minimum()))
            return
        super().keyPressEvent(event)

    def parse_float(self, widget: QtWidgets.QLineEdit, fallback: float, minimum: float | None = None) -> float:
        """Parse a float from a line edit with clamping and fallback."""

        text = widget.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = fallback
        if minimum is not None:
            value = max(value, minimum)
        return value

    def parse_int(self, widget: QtWidgets.QLineEdit, fallback: int, minimum: int = 0) -> int:
        """Parse an integer from a line edit with clamping and fallback."""

        text = widget.text().strip()
        try:
            value = int(float(text))
        except ValueError:
            value = fallback
        return max(value, minimum)

    def build_display_frame(self, time_index: int) -> np.ndarray:
        """Build the display frame for the current mode at one time index."""

        left_frame = self.data.left_frames[time_index]
        right_frame = self.data.right_frames[time_index]
        return compute_display(left_frame, right_frame, self.state.mode, self.state.eps, self.state.denom_min)

    def refresh_view(self) -> None:
        """Recompute the current view and push updates into the plots."""

        self.state.eps = self.parse_float(self.eps_box, self.config.eps_init, minimum=1e-15)
        self.state.denom_min = self.parse_float(self.denom_box, self.config.denom_min_init, minimum=0.0)
        self.state.time_index = int(self.time_slider.value())
        self.state.x_idx = int(np.clip(self.state.x_idx, 0, self.data.left_frames.shape[2] - 1))
        self.state.y_idx = int(np.clip(self.state.y_idx, 0, self.data.left_frames.shape[1] - 1))

        if not self.state.updating_time_box:
            self.state.updating_time_box = True
            self.time_box.setText(f"{self.data.times_ps[self.state.time_index]:.2f}")
            self.state.updating_time_box = False
        self.frame_label.setText(f"Frame: {self.state.time_index}")

        left_frame = self.data.left_frames[self.state.time_index]
        right_frame = self.data.right_frames[self.state.time_index]
        display_frame = compute_display(left_frame, right_frame, self.state.mode, self.state.eps, self.state.denom_min)
        is_norm = self.state.mode == "norm"

        default_vmin = self.config.norm_vmin if is_norm else self.config.init_vmin
        default_vmax = self.config.norm_vmax if is_norm else self.config.init_vmax
        manual_vmin = self.parse_float(self.vmin_box, default_vmin)
        manual_vmax = self.parse_float(self.vmax_box, default_vmax)

        if self.state.auto_scale:
            vmin, vmax = robust_limits(
                display_frame,
                self.config.auto_percentiles[0],
                self.config.auto_percentiles[1],
                manual_vmin,
                manual_vmax,
            )
        else:
            vmin, vmax = manual_vmin, manual_vmax

        if vmin >= vmax:
            vmax = vmin + 1e-12

        x_idx = self.state.x_idx
        y_idx = self.state.y_idx
        current_time = float(self.data.times_ps[self.state.time_index])
        average_half_width = self.parse_int(self.fit_window_box, 3, minimum=0)

        wavelength_slice = display_frame[y_idx, :]
        spatial_slice = display_frame[:, x_idx]

        left_trace = self.data.left_frames[:, y_idx, x_idx].astype(float, copy=False)
        right_trace = self.data.right_frames[:, y_idx, x_idx].astype(float, copy=False)
        display_trace = compute_display(left_trace, right_trace, self.state.mode, self.state.eps, self.state.denom_min)
        subtraction_trace = left_trace - right_trace
        subtraction_trace_01 = normalize_to_01(subtraction_trace)
        subtraction_spatial = left_frame[:, x_idx] - right_frame[:, x_idx]
        subtraction_spatial_01 = normalize_to_01(subtraction_spatial)

        self.plot_view.update_image(display_frame, self.state.mode, current_time, vmin, vmax, x_idx, y_idx)
        self.plot_view.update_wavelength_slice(wavelength_slice, self.data.wavelength_nm, self.state.mode, x_idx, y_idx)
        self.plot_view.update_spatial_slice(spatial_slice, self.data.position_um, self.state.mode, x_idx, y_idx)

        t = self.state.time_index
        if self.G_STD[t] > 0 and np.isfinite(self.G_STD[t]):
            self.plot_view.update_spatial_fit(
                self.data.position_um, self.G_AMP[t], self.G_MEAN[t], self.G_STD[t], self.G_OFFSET[t],
            )
            std_text = f"{self.G_STD[t]:.3g}"
            if np.isfinite(self.G_STD_ERR[t]) and self.G_STD_ERR[t] > 0:
                std_text += f" +/- {self.G_STD_ERR[t]:.2g}"
            self.gaussian_result_label.setText(
                f"Fit @ frame {t}: A={self.G_AMP[t]:.3g}, "
                f"mean={self.G_MEAN[t]:.3g} um, std={std_text} um, "
                f"offset={self.G_OFFSET[t]:.3g}"
            )
        else:
            self.plot_view.update_spatial_fit(self.data.position_um, 0, 0, 0, 0)
            self.gaussian_result_label.setText("Gaussian fit: not computed for this frame")
        self.plot_view.update_time_trace(
            self.data.times_ps,
            display_trace,
            left_trace,
            right_trace,
            show_components=is_norm or self.state.mode == "difference",
            mode=self.state.mode,
            x_idx=x_idx,
            y_idx=y_idx,
            current_time=current_time,
        )
        self.plot_view.update_subtraction_dynamics(self.data.times_ps, subtraction_trace_01, current_time, x_idx, y_idx)
        self.plot_view.update_subtraction_spatial(self.data.position_um, subtraction_spatial_01, current_time, y_idx)

        if is_norm:
            wavelength_cut, wavelength_mask = compute_spin_norm_cut(
                left_frame[y_idx, :],
                right_frame[y_idx, :],
                self.state.eps,
                self.state.denom_min,
            )
            spatial_cut, spatial_mask = compute_spin_norm_cut(
                left_frame[:, x_idx],
                right_frame[:, x_idx],
                self.state.eps,
                self.state.denom_min,
            )
            time_cut, time_mask = compute_spin_norm_cut(
                left_trace,
                right_trace,
                self.state.eps,
                self.state.denom_min,
            )
            subtraction_time_cut = normalize_to_01(apply_mask_cut(subtraction_trace, time_mask))
            subtraction_spatial_cut = normalize_to_01(apply_mask_cut(subtraction_spatial, spatial_mask))
            self.norm_cut_view.update_plots(
                wavelength_nm=self.data.wavelength_nm,
                position_um=self.data.position_um,
                times_ps=self.data.times_ps,
                wavelength_cut=wavelength_cut,
                time_cut=time_cut,
                subtraction_time_cut=subtraction_time_cut,
                subtraction_spatial_cut=subtraction_spatial_cut,
                current_time=current_time,
                x_idx=x_idx,
                y_idx=y_idx,
            )
            self.norm_cut_fit_view.update_profile(self.data.position_um, subtraction_spatial_cut, x_idx, y_idx)
            self.norm_cut_view.clear_fit_overlay()
            if self.G_CUT_STD[t] > 0 and np.isfinite(self.G_CUT_STD[t]):
                self.norm_cut_fit_view.update_fit(
                    self.data.position_um,
                    self.G_CUT_AMP[t],
                    self.G_CUT_MEAN[t],
                    self.G_CUT_STD[t],
                    self.G_CUT_OFFSET[t],
                )
            else:
                self.norm_cut_fit_view.update_fit(self.data.position_um, 0, 0, 0, 0)
            if self.G_CUT_STD[t] > 0 and np.isfinite(self.G_CUT_STD[t]) and self.plot_tabs.currentWidget() is self.norm_cut_fit_view:
                std_text = f"{self.G_CUT_STD[t]:.3g}"
                if np.isfinite(self.G_CUT_STD_ERR[t]) and self.G_CUT_STD_ERR[t] > 0:
                    std_text += f" +/- {self.G_CUT_STD_ERR[t]:.2g}"
                self.gaussian_result_label.setText(
                        f"Norm cut subtraction fit @ frame {t}: A={self.G_CUT_AMP[t]:.3g}, "
                    f"mean={self.G_CUT_MEAN[t]:.3g} um, std={std_text} um, "
                    f"offset={self.G_CUT_OFFSET[t]:.3g}"
                )
        else:
            self.norm_cut_view.clear()
            self.norm_cut_fit_view.clear()

        self.state.last_trace_payload = TracePayload(
            time_ps=self.data.times_ps.copy(),
            left=left_trace,
            right=right_trace,
            display=display_trace,
            spin_norm=compute_spin_norm(left_trace, right_trace, self.state.eps, self.state.denom_min),
            subtraction=subtraction_trace,
            subtraction_01=subtraction_trace_01,
            mode=self.state.mode,
            x_idx=x_idx,
            y_idx=y_idx,
            wavelength_nm=float(np.round(self.data.wavelength_nm[x_idx])),
            position_um=float(np.round(self.data.position_um[y_idx])),
        )
        self.statusBar().showMessage(
            f"Mode={self.state.mode}, t={current_time:.2f} ps, X={x_idx}, Y={y_idx}",
            4000,
        )

    def on_image_clicked(self, x_idx: int, y_idx: int) -> None:
        """Update the selected position from an image click."""

        self.state.x_idx = x_idx
        self.state.y_idx = y_idx
        self.refresh_view()

    def on_time_slider_changed(self, value: int) -> None:
        """Handle time slider movement."""

        self.state.time_index = int(value)
        self.refresh_view()

    def on_time_box_submit(self) -> None:
        """Jump to the time closest to the user-entered ps value."""

        if self.state.updating_time_box:
            return
        try:
            time_value = float(self.time_box.text())
        except ValueError:
            self.time_box.setText(f"{self.data.times_ps[self.state.time_index]:.2f}")
            return
        index = find_closest_index(self.data.times_ps, time_value)
        self.time_slider.setValue(index)

    def on_mode_changed(self) -> None:
        """Switch display mode when a radio button is toggled."""

        for mode, button in self.mode_buttons.items():
            if button.isChecked():
                self.state.mode = mode
                break
        self.refresh_view()

    def on_set_denom_clicked(self) -> None:
        """Clamp and normalize the denominator threshold box."""

        denom_min = self.parse_float(self.denom_box, self.config.denom_min_init, minimum=0.0)
        self.denom_box.setText(f"{denom_min:.3g}")
        self.refresh_view()

    def on_auto_scale_changed(self, state: int) -> None:
        """Toggle automatic image scaling."""

        self.state.auto_scale = state == int(QT_CHECKED)
        self.refresh_view()

    def _browse_file(self, edit_widget: QtWidgets.QLineEdit, title: str) -> None:
        """Open a file dialog and write the chosen path into the given line edit."""
        current = edit_widget.text().strip()
        if current and Path(current).parent.exists():
            start_dir = str(Path(current).parent)
        else:
            start_dir = str(self.script_dir)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start_dir, "Text (*.txt);;All (*.*)"
        )
        if path:
            edit_widget.setText(str(Path(path)))

    def _sync_path_edits_from_config(self) -> None:
        """Populate the file path edits from the current Config."""
        cfg = self.config
        self.path_left_img_edit.setText(str(self.loader.resolve_data_path(cfg.filename_left_img)))
        self.path_left_time_edit.setText(str(self.loader.resolve_data_path(cfg.filename_left_time)))
        self.path_right_img_edit.setText(str(self.loader.resolve_data_path(cfg.filename_right_img)))
        self.path_right_time_edit.setText(str(self.loader.resolve_data_path(cfg.filename_right_time)))

    def _apply_paths_and_load(self) -> None:
        """Read file paths from the edits, reload data, and refresh the entire GUI."""
        left_img = self.path_left_img_edit.text().strip()
        left_time = self.path_left_time_edit.text().strip()
        right_img = self.path_right_img_edit.text().strip()
        right_time = self.path_right_time_edit.text().strip()

        if not all([left_img, left_time, right_img, right_time]):
            self._show_warning("Please set all four file paths before loading.")
            return
        for label, p in [("Left image", left_img), ("Left para", left_time),
                          ("Right image", right_img), ("Right para", right_time)]:
            if not Path(p).is_file():
                self._show_warning(f"{label} file not found:\n{p}")
                return

        self.config.filename_left_img = left_img
        self.config.filename_left_time = left_time
        self.config.filename_right_img = right_img
        self.config.filename_right_time = right_time

        try:
            new_data = self._load_processed_data()
        except Exception as exc:
            self._show_warning(f"Failed to load data:\n{exc}")
            return

        self.data = new_data
        self._data_loaded = True

        old_tab_index = self.plot_tabs.currentIndex()
        old_views = [self.plot_view, self.norm_cut_view, self.norm_cut_fit_view]
        for i in range(self.plot_tabs.count() - 1, -1, -1):
            self.plot_tabs.removeTab(i)
        for v in old_views:
            v.deleteLater()

        self.plot_view = SpinPlotView(self.data, self.config)
        self.plot_view.set_image_click_callback(self.on_image_clicked)
        self.norm_cut_view = NormCutPlotView(self.data, self.config)
        self.norm_cut_fit_view = NormCutFitView(self.data)
        self.plot_tabs.addTab(self.plot_view, "Main")
        self.plot_tabs.addTab(self.norm_cut_view, "Norm cut")
        self.plot_tabs.addTab(self.norm_cut_fit_view, "Norm cut fit")
        self.plot_tabs.setCurrentIndex(min(old_tab_index, 2))

        height, width = self.data.left_frames.shape[1:]
        self.state.x_idx = max(width - 10, 0)
        self.state.y_idx = max(height - 10, 0)
        self.state.time_index = 0

        n = len(self.data.times_ps)
        for attr in ("G_AMP", "G_MEAN", "G_STD", "G_OFFSET",
                      "G_AMP_ERR", "G_MEAN_ERR", "G_STD_ERR", "G_OFFSET_ERR",
                      "G_CUT_AMP", "G_CUT_MEAN", "G_CUT_STD", "G_CUT_OFFSET",
                      "G_CUT_AMP_ERR", "G_CUT_MEAN_ERR", "G_CUT_STD_ERR", "G_CUT_OFFSET_ERR"):
            setattr(self, attr, np.zeros(n))

        self.time_slider.setMaximum(n - 1)
        self.time_slider.setValue(0)
        self.time_box.setText(f"{self.data.times_ps[0]:.2f}")

        self.refresh_view()
        self.statusBar().showMessage("Data loaded successfully", 5000)

    def on_export_png(self) -> None:
        """Export the current GUI view to a PNG file."""

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path.cwd() / f"paper_spin_view_{self.state.mode}_t{self.state.time_index}_{stamp}.png"
        current_plot_widget = self.plot_tabs.currentWidget()
        if current_plot_widget is not None:
            current_plot_widget.grab().save(str(out_path), "PNG")
        self.statusBar().showMessage(f"Saved PNG: {out_path}", 6000)

    def on_export_csv(self) -> None:
        """Export the currently selected time trace to CSV."""

        payload = self.state.last_trace_payload
        if payload is None:
            self._show_warning("No trace data is available yet.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (
            Path.cwd()
            / f"paper_spin_trace_{payload.mode}_x{payload.x_idx}_y{payload.y_idx}_{stamp}.csv"
        )
        df = pd.DataFrame(
            {
                "time_ps": payload.time_ps,
                "Left": payload.left,
                "Right": payload.right,
                f"{payload.mode}_trace": payload.display,
                "Left_minus_Right": payload.subtraction,
                "Left_minus_Right_norm01": payload.subtraction_01,
            }
        )
        df.to_csv(out_path, index=False)
        self.statusBar().showMessage(f"Saved CSV: {out_path}", 6000)

    def on_export_spin_dynamics(self) -> None:
        """Export the masked spin-contrast dynamics at the selected point."""

        payload = self.state.last_trace_payload
        if payload is None:
            self._show_warning("No spin dynamics are available yet.")
            return

        left_img_path = self.loader.resolve_data_path(self.config.filename_left_img)
        out_dir = left_img_path.parent
        base_filename = left_img_path.stem
        wavelength = payload.wavelength_nm
        position = payload.position_um
        out_path = out_dir / f"{base_filename}DYatwav{wavelength}pos{position}_spin.csv"
        export_df = pd.DataFrame(
            {
                "Time": payload.time_ps,
                "Left": payload.left,
                "Right": payload.right,
                "Difference": payload.subtraction,
                "Spin_Dynamics": payload.spin_norm,
            }
        )
        export_df.to_csv(out_path, index=False)
        self.statusBar().showMessage(f"Saved spin dynamics: {out_path}", 6000)

    def _fit_spatial_at_frame(self, t: int) -> bool:
        """Fit Gaussian-with-offset to the spatial profile at frame t. Returns True on success."""

        hw = self.parse_int(self.fit_window_box, 3, minimum=0)
        ix = int(np.clip(self.state.x_idx, 0, self.data.left_frames.shape[2] - 1))
        display_frame = self.build_display_frame(t)
        why = extract_spatial_fit_profile(display_frame, ix, hw)
        amp0 = float(why[np.argmax(np.abs(why))])
        mu0 = float(self.data.position_um[np.argmax(np.abs(why))])
        sig0 = max(float(self.data.position_um[-1] - self.data.position_um[0]) / 10.0, 1e-3)
        off0 = float(np.median(why))
        try:
            p, cov = curve_fit(
                gaussian_with_offset, self.data.position_um, why,
                p0=[amp0, mu0, sig0, off0], maxfev=20000,
            )
            e = np.sqrt(np.diag(cov))
            self.G_AMP[t], self.G_MEAN[t] = p[0], p[1]
            self.G_STD[t], self.G_OFFSET[t] = abs(p[2]), p[3]
            self.G_AMP_ERR[t], self.G_MEAN_ERR[t] = e[0], e[1]
            self.G_STD_ERR[t], self.G_OFFSET_ERR[t] = abs(e[2]), e[3]
            return True
        except (RuntimeError, ValueError):
            self.G_STD[t] = np.nan
            self.G_STD_ERR[t] = np.nan
            return False

    def _fit_spatial_cut_at_frame(self, t: int) -> bool:
        """Fit Gaussian-with-offset to the masked subtraction spatial profile at frame t."""

        ix = int(np.clip(self.state.x_idx, 0, self.data.left_frames.shape[2] - 1))
        _, spatial_mask = compute_spin_norm_cut(
            self.data.left_frames[t, :, ix],
            self.data.right_frames[t, :, ix],
            self.state.eps,
            self.state.denom_min,
        )
        subtraction_spatial = self.data.left_frames[t, :, ix] - self.data.right_frames[t, :, ix]
        subtraction_spatial_cut = normalize_to_01(apply_mask_cut(subtraction_spatial, spatial_mask))
        valid = np.isfinite(subtraction_spatial_cut) & np.isfinite(self.data.position_um)
        if np.count_nonzero(valid) < 4:
            self.G_CUT_STD[t] = np.nan
            self.G_CUT_STD_ERR[t] = np.nan
            return False

        x_fit = self.data.position_um[valid]
        y_fit = subtraction_spatial_cut[valid]
        amp0 = float(y_fit[np.argmax(np.abs(y_fit))])
        mu0 = float(x_fit[np.argmax(np.abs(y_fit))])
        sig0 = max(float(x_fit[-1] - x_fit[0]) / 10.0, 1e-3)
        off0 = float(np.median(y_fit))
        try:
            p, cov = curve_fit(
                gaussian_with_offset,
                x_fit,
                y_fit,
                p0=[amp0, mu0, sig0, off0],
                maxfev=20000,
            )
            e = np.sqrt(np.diag(cov))
            self.G_CUT_AMP[t], self.G_CUT_MEAN[t] = p[0], p[1]
            self.G_CUT_STD[t], self.G_CUT_OFFSET[t] = abs(p[2]), p[3]
            self.G_CUT_AMP_ERR[t], self.G_CUT_MEAN_ERR[t] = e[0], e[1]
            self.G_CUT_STD_ERR[t], self.G_CUT_OFFSET_ERR[t] = abs(e[2]), e[3]
            return True
        except (RuntimeError, ValueError):
            self.G_CUT_STD[t] = np.nan
            self.G_CUT_STD_ERR[t] = np.nan
            return False

    def on_fit_current_gaussian(self) -> None:
        """Fit a Gaussian-with-offset to the current spatial profile."""

        t = self.state.time_index
        use_norm_cut_fit = self.plot_tabs.currentWidget() is self.norm_cut_fit_view
        success = self._fit_spatial_cut_at_frame(t) if use_norm_cut_fit else self._fit_spatial_at_frame(t)
        fit_std = self.G_CUT_STD[t] if use_norm_cut_fit else self.G_STD[t]
        if success:
            self.statusBar().showMessage(
                f"{'Norm cut fit' if use_norm_cut_fit else 'Fit'} @ frame {t}: sigma={fit_std:.4g} um",
                7000,
            )
        else:
            self.statusBar().showMessage("Gaussian fit did not converge", 5000)
        self.refresh_view()

    def on_fit_all_gaussian(self) -> None:
        """Fit Gaussian profiles for all times from the selected start frame."""

        start_index = self.parse_int(self.fit_start_index_box, 0, minimum=0)
        start_index = min(start_index, len(self.data.times_ps) - 1)
        use_norm_cut_fit = self.plot_tabs.currentWidget() is self.norm_cut_fit_view
        success_count = 0
        for i in range(start_index, len(self.data.times_ps)):
            if (self._fit_spatial_cut_at_frame(i) if use_norm_cut_fit else self._fit_spatial_at_frame(i)):
                success_count += 1

        self._plot_fit_w2(start_index, use_norm_cut_fit=use_norm_cut_fit)
        self.statusBar().showMessage(
            f"{'Norm cut fit' if use_norm_cut_fit else 'Fit'} All complete: "
            f"{success_count}/{len(self.data.times_ps) - start_index} converged",
            7000,
        )
        self.refresh_view()

    def _plot_fit_w2(self, start_index: int, *, use_norm_cut_fit: bool = False) -> None:
        """Show Gaussian W squared vs time with error bars in a pop-up window."""

        gw = self.G_CUT_STD[start_index:] if use_norm_cut_fit else self.G_STD[start_index:]
        gwe = self.G_CUT_STD_ERR[start_index:] if use_norm_cut_fit else self.G_STD_ERR[start_index:]
        t_all = self.data.times_ps[start_index:]
        valid = np.isfinite(gw) & (gw > 0)
        if not np.any(valid):
            self._show_warning("No Gaussian fits converged in the selected range.")
            return

        t = t_all[valid]
        w2 = gw[valid] ** 2
        w2e = 2.0 * gw[valid] * gwe[valid]

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Gaussian W\u00b2 vs Time" + (" (Norm cut)" if use_norm_cut_fit else ""))
        dialog.resize(700, 450)
        layout = QtWidgets.QVBoxLayout(dialog)
        pw = pg.PlotWidget()
        pw.setBackground("w")
        pw.showGrid(x=True, y=True, alpha=0.2)
        pw.setLabel("bottom", "Time (ps)")
        pw.setLabel("left", "W\u00b2 (um\u00b2)")
        pw.plot(
            t, w2,
            pen=pg.mkPen((148, 103, 189), width=1.5),
            symbol="o", symbolSize=5,
            symbolPen=(148, 103, 189), symbolBrush=(148, 103, 189),
        )
        err_item = pg.ErrorBarItem(x=t, y=w2, height=w2e, pen=pg.mkPen((148, 103, 189), width=1))
        pw.addItem(err_item)
        layout.addWidget(pw)
        self._fit_w2_dialog = dialog
        dialog.show()

    def on_export_gaussian_fit(self) -> None:
        """Export the per-frame Gaussian fit arrays to an Excel file."""

        if not np.any(np.isfinite(self.G_STD) & (self.G_STD > 0)):
            self._show_warning("No Gaussian fit results to export. Run 'Fit current' or 'Fit all times' first.")
            return

        left_img_path = self.loader.resolve_data_path(self.config.filename_left_img)
        out_dir = left_img_path.parent
        base_filename = left_img_path.stem
        x_idx = self.state.x_idx
        wavelength_nm = float(np.round(self.data.wavelength_nm[x_idx]))
        out_path = out_dir / f"{base_filename}GaussianW_atwav{wavelength_nm}.xlsx"

        df = pd.DataFrame()
        df["Time"] = self.data.times_ps
        df["Standard_W_um"] = self.G_STD
        df["Standard_W_Err_um"] = self.G_STD_ERR
        df["Amplitude"] = self.G_AMP
        df["Mean_um"] = self.G_MEAN
        df["Offset"] = self.G_OFFSET
        df.to_excel(out_path, index=False)
        self.statusBar().showMessage(f"Saved Gaussian fit: {out_path}", 7000)

    def _show_warning(self, message: str) -> None:
        """Show a user-facing warning dialog."""

        QtWidgets.QMessageBox.warning(self, "Spin GUI", message)


def main() -> int:
    """Launch the Qt application."""

    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = SpinSubtractionWindow(Config())
    window.show()
    exec_method = getattr(app, "exec", None)
    if exec_method is None:
        return int(app.exec_())
    return int(exec_method())


if __name__ == "__main__":
    sys.exit(main())

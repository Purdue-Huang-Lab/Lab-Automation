"""SPAD23 GUI: live count map + counts analysis + TRPL tabbed workflow."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from matplotlib.tri import Triangulation

try:
    from scipy.optimize import curve_fit

    _HAS_SCIPY = True
except Exception:
    curve_fit = None  # type: ignore
    _HAS_SCIPY = False

from spad23.spad23_tcspc_wrapper import Spad23TcspcClient, TrplSnapshot
from spad23.spad23_wrapper import (
    NUM_PIXELS,
    PIXEL_LIMIT_MCPS_23G,
    CountSnapshot,
    Spad23CountClient,
    Spad23Error,
    Spad23OverflowError,
)


PIXEL_COORDS: Dict[int, Tuple[float, float]] = {
    0: (0.0, 4.0),
    1: (1.0, 4.0),
    2: (2.0, 4.0),
    3: (3.0, 4.0),
    4: (4.0, 4.0),
    5: (0.5, 3.0),
    6: (1.5, 3.0),
    7: (2.5, 3.0),
    8: (3.5, 3.0),
    9: (0.0, 2.0),
    10: (1.0, 2.0),
    11: (2.0, 2.0),
    12: (3.0, 2.0),
    13: (4.0, 2.0),
    14: (0.5, 1.0),
    15: (1.5, 1.0),
    16: (2.5, 1.0),
    17: (3.5, 1.0),
    18: (0.0, 0.0),
    19: (1.0, 0.0),
    20: (2.0, 0.0),
    21: (3.0, 0.0),
    22: (4.0, 0.0),
}

FIT_ROWS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12, 13],
    [14, 15, 16, 17],
    [18, 19, 20, 21, 22],
]

PITCH_X_UM = 23.0
PITCH_Y_UM = 19.92
LIVE_HIST_MAX_POINTS = 6000
LIVE_PLOT_UPDATE_EVERY = 3


@dataclass
class FitResult:
    popt: np.ndarray
    z_fit: np.ndarray
    sigma_eq: float


def _build_fit_coords() -> Tuple[np.ndarray, np.ndarray]:
    coords: Dict[int, Tuple[float, float]] = {}
    max_cols = max(len(r) for r in FIT_ROWS)
    for row_idx, row in enumerate(FIT_ROWS):
        x0 = (PITCH_X_UM / 2.0) if len(row) < max_cols else 0.0
        for col_idx, pixel in enumerate(row):
            coords[pixel] = (x0 + col_idx * PITCH_X_UM, row_idx * PITCH_Y_UM)
    x = np.array([coords[i][0] for i in range(NUM_PIXELS)], dtype=np.float64)
    y = np.array([coords[i][1] for i in range(NUM_PIXELS)], dtype=np.float64)
    return x, y


def _auto_rate_scale(values_mcps: np.ndarray) -> Tuple[str, float]:
    max_mcps = float(np.nanmax(values_mcps)) if values_mcps.size else 0.0
    if max_mcps < 0.001:
        return "cps", 1e6
    if max_mcps < 0.1:
        return "kcps", 1e3
    return "Mcps", 1.0


def _format_rate(rate_mcps: float) -> str:
    if rate_mcps < 0.001:
        return f"{rate_mcps * 1e6:.1f} cps"
    if rate_mcps < 0.1:
        return f"{rate_mcps * 1e3:.3f} kcps"
    return f"{rate_mcps:.3f} Mcps"


def _fit_gaussian_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[FitResult]:
    if not _HAS_SCIPY:
        return None
    if np.allclose(z, z[0]):
        return None

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0) ** 2) / (2.0 * sx**2) + ((yg - y0) ** 2) / (2.0 * sy**2))) + offset

    p0 = [
        max(float(np.max(z) - np.min(z)), 1e-6),
        float(np.mean(x)),
        float(np.mean(y)),
        20.0,
        20.0,
        float(np.min(z)),
    ]
    bounds = (
        [0.0, float(np.min(x) - PITCH_X_UM), float(np.min(y) - PITCH_Y_UM), 1e-3, 1e-3, -np.inf],
        [np.inf, float(np.max(x) + PITCH_X_UM), float(np.max(y) + PITCH_Y_UM), np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)  # type: ignore[misc]
    except Exception:
        return None

    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return FitResult(popt=np.asarray(popt, dtype=np.float64), z_fit=np.asarray(z_fit, dtype=np.float64), sigma_eq=sigma_eq)


def _fit_gaussian_2d_with_reference(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x0_fixed: float,
    y0_fixed: float,
    sigma_guess: float,
) -> Optional[FitResult]:
    if not _HAS_SCIPY:
        return None
    if np.allclose(z, z[0]):
        return None

    sigma_guess = max(float(sigma_guess), 1e-3)

    def gauss2d_fixed(coords, A, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0_fixed) ** 2) / (2.0 * sx**2) + ((yg - y0_fixed) ** 2) / (2.0 * sy**2))) + offset

    p0 = [max(float(np.max(z) - np.min(z)), 1e-6), sigma_guess, sigma_guess, float(np.min(z))]
    bounds = ([0.0, 1e-3, 1e-3, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    try:
        p_fit, _ = curve_fit(gauss2d_fixed, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)  # type: ignore[misc]
    except Exception:
        return None

    A, sx, sy, offset = [float(v) for v in p_fit]
    z_fit = gauss2d_fixed((x, y), A, sx, sy, offset)
    popt = np.array([A, float(x0_fixed), float(y0_fixed), sx, sy, offset], dtype=np.float64)
    sigma_eq = float(np.sqrt((sx**2 + sy**2) / 2.0))
    return FitResult(popt=popt, z_fit=np.asarray(z_fit, dtype=np.float64), sigma_eq=sigma_eq)


def _fit_gaussian_2d_constrained_next(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prev_popt: np.ndarray,
    tightness: float = 1.0,
) -> Optional[FitResult]:
    """
    Fit next TRPL frame using previous frame fit as initial guess, with small bounds.
    """
    if not _HAS_SCIPY:
        return None
    if np.allclose(z, z[0]):
        return None

    A_prev, x0_prev, y0_prev, sx_prev, sy_prev, off_prev = [float(v) for v in prev_popt]
    sx_prev = max(sx_prev, 1e-3)
    sy_prev = max(sy_prev, 1e-3)

    # Small allowed drift around previous frame fit.
    dA = max(abs(A_prev) * 0.18 * tightness, 1e-3)
    dx = max(PITCH_X_UM * 0.18 * tightness, 0.8)
    dy = max(PITCH_Y_UM * 0.18 * tightness, 0.8)
    dsx = max(abs(sx_prev) * 0.18 * tightness, 0.5)
    dsy = max(abs(sy_prev) * 0.18 * tightness, 0.5)
    doff = max(abs(off_prev) * 0.25 * tightness, 0.5)

    p0 = [A_prev, x0_prev, y0_prev, sx_prev, sy_prev, off_prev]
    lower = [
        max(0.0, A_prev - dA),
        x0_prev - dx,
        y0_prev - dy,
        max(1e-3, sx_prev - dsx),
        max(1e-3, sy_prev - dsy),
        off_prev - doff,
    ]
    upper = [
        A_prev + dA,
        x0_prev + dx,
        y0_prev + dy,
        sx_prev + dsx,
        sy_prev + dsy,
        off_prev + doff,
    ]

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0) ** 2) / (2.0 * sx**2) + ((yg - y0) ** 2) / (2.0 * sy**2))) + offset

    try:
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0, bounds=(lower, upper), maxfev=20000)  # type: ignore[misc]
    except Exception:
        return None

    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return FitResult(popt=np.asarray(popt, dtype=np.float64), z_fit=np.asarray(z_fit, dtype=np.float64), sigma_eq=sigma_eq)


class CalibAlignWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, host: str, port: int, align_ms: int = 1000):
        super().__init__()
        self.host = host
        self.port = port
        self.align_ms = int(align_ms)

    def run(self) -> None:
        try:
            with Spad23TcspcClient(host=self.host, port=self.port) as client:
                messages = client.calibrate_and_align(align_ms=self.align_ms, calibrate_if_invalid=True)
            self.done.emit(" | ".join(messages))
        except Exception as exc:
            self.error.emit(str(exc))


class TrplAcquireWorker(QThread):
    update = Signal(object)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, host: str, port: int, measurement_ms: int, bin_width_ps: int):
        super().__init__()
        self.host = host
        self.port = port
        self.measurement_ms = int(measurement_ms)
        self.bin_width_ps = int(bin_width_ps)

    def run(self) -> None:
        try:
            with Spad23TcspcClient(host=self.host, port=self.port) as client:
                snapshot = client.acquire_trpl_stream(
                    measurement_ms=self.measurement_ms,
                    bin_width_ps=self.bin_width_ps,
                    on_update=lambda s: self.update.emit(s),
                    update_interval_s=0.4,
                )
            self.done.emit(snapshot)
        except Exception as exc:
            self.error.emit(str(exc))


class Spad23MainWindow(QMainWindow):
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        super().__init__()
        self.host = host
        self.port = int(port)

        self.count_client = Spad23CountClient(host=self.host, port=self.port, pixel_limit_mcps=PIXEL_LIMIT_MCPS_23G)
        self.latest_count_snapshot: Optional[CountSnapshot] = None
        self.latest_trpl_snapshot: Optional[TrplSnapshot] = None

        self.trpl_worker: Optional[TrplAcquireWorker] = None
        self.calib_worker: Optional[CalibAlignWorker] = None
        self.trpl_running = False
        self.live_updates_enabled = True
        self.selected_pixel = 11
        self.hover_pixel: Optional[int] = None
        self.excluded_pixels: Set[int] = set()
        self.reference_fit: Optional[Tuple[float, float, float]] = None  # x0, y0, sigma_eq
        self.reference_frame_idx: Optional[int] = None
        self.reference_frame_popt: Optional[np.ndarray] = None
        self.reference_fit_cache: Dict[int, FitResult] = {}
        self.sigma_time_ns = np.array([], dtype=np.float64)
        self.sigma_values = np.array([], dtype=np.float64)
        self.slider_start_idx = 0
        self.slider_end_idx = 0
        self._trpl_update_counter = 0

        self.fit_x_um, self.fit_y_um = _build_fit_coords()
        self.fit_tri = Triangulation(self.fit_x_um, self.fit_y_um)
        self.map_coords = np.array([PIXEL_COORDS[i] for i in range(NUM_PIXELS)], dtype=np.float64)

        self._build_ui()
        self._setup_count_timer()

    def _build_ui(self) -> None:
        self.setWindowTitle("SPAD23 Counts + TRPL")
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left: live count map
        left = QVBoxLayout()
        self.left_fig = Figure(figsize=(6.8, 7.2), dpi=100)
        self.left_canvas = FigureCanvas(self.left_fig)
        self.left_ax = self.left_fig.add_subplot(111)
        left.addWidget(self.left_canvas, 1)

        self.lbl_left_info = QLabel("Total: --")
        self.lbl_left_state = QLabel("Status: idle")
        self.btn_toggle_live = QPushButton("Pause Live Counts")
        self.btn_toggle_live.clicked.connect(self._toggle_live_updates)
        left.addWidget(self.lbl_left_info)
        left.addWidget(self.lbl_left_state)
        left.addWidget(self.btn_toggle_live)

        left_container = QWidget()
        left_container.setLayout(left)
        left_container.setMinimumWidth(560)
        splitter.addWidget(left_container)

        # Right: tabs
        right = QVBoxLayout()
        self.tabs = QTabWidget()
        right.addWidget(self.tabs, 1)

        self._build_counts_tab()
        self._build_fitting_tab()
        self._build_trpl_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        right_container = QWidget()
        right_container.setLayout(right)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([760, 980])

        self._init_left_map()

    def _build_counts_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.counts_fig = Figure(figsize=(8.0, 6.8), dpi=100)
        self.counts_canvas = FigureCanvas(self.counts_fig)
        g = self.counts_fig.add_gridspec(2, 2, wspace=0.35, hspace=0.4)
        self.counts_fit_ax = self.counts_fig.add_subplot(g[0, 0])
        self.counts_radial_ax = self.counts_fig.add_subplot(g[0, 1])
        self.counts_hcut_ax = self.counts_fig.add_subplot(g[1, 0])
        self.counts_vcut_ax = self.counts_fig.add_subplot(g[1, 1])
        layout.addWidget(self.counts_canvas)

        self.tabs.addTab(tab, "Counts")

    def _build_fitting_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.lbl_fit_status = QLabel("Run TRPL to generate sigma-vs-time.")
        self.lbl_fit_status.setWordWrap(True)
        layout.addWidget(self.lbl_fit_status)

        self.fitting_fig = Figure(figsize=(8.0, 6.8), dpi=100)
        self.fitting_canvas = FigureCanvas(self.fitting_fig)
        self.fitting_ax = self.fitting_fig.add_subplot(111)
        self.fitting_ax.set_title("Sigma_eq vs Time")
        self.fitting_ax.set_xlabel("Time (ns)")
        self.fitting_ax.set_ylabel("Sigma_eq (um)")
        self.fitting_ax.grid(True, alpha=0.25)
        layout.addWidget(self.fitting_canvas)

        self.tabs.addTab(tab, "Fitting")

    def _build_trpl_tab(self) -> None:
        tab = QWidget()
        root = QVBoxLayout(tab)

        ctrl_group = QGroupBox("TRPL Controls")
        ctrl_layout = QGridLayout(ctrl_group)

        self.spin_acq_s = QDoubleSpinBox()
        self.spin_acq_s.setRange(0.1, 3600.0)
        self.spin_acq_s.setDecimals(2)
        self.spin_acq_s.setValue(10.0)

        self.spin_bin_ps = QSpinBox()
        self.spin_bin_ps.setRange(1, 5000)
        self.spin_bin_ps.setValue(10)

        self.btn_calib_align = QPushButton("Calibrate + Align")
        self.btn_take_trpl = QPushButton("Take TRPL")
        self.btn_export_trpl = QPushButton("Export TRPL CSV")
        self.chk_hist_log = QCheckBox("Log histogram y")

        self.lbl_trpl_state = QLabel("Idle")
        self.lbl_trpl_state.setWordWrap(True)
        self.lbl_selected_pixel = QLabel(f"Selected pixel: {self.selected_pixel}")

        ctrl_layout.addWidget(QLabel("Acquisition time (s)"), 0, 0)
        ctrl_layout.addWidget(self.spin_acq_s, 0, 1)
        ctrl_layout.addWidget(QLabel("Bin size (ps)"), 0, 2)
        ctrl_layout.addWidget(self.spin_bin_ps, 0, 3)
        ctrl_layout.addWidget(self.btn_calib_align, 0, 4)
        ctrl_layout.addWidget(self.btn_take_trpl, 0, 5)
        ctrl_layout.addWidget(self.btn_export_trpl, 0, 6)
        ctrl_layout.addWidget(self.lbl_selected_pixel, 1, 0, 1, 2)
        ctrl_layout.addWidget(self.chk_hist_log, 1, 2, 1, 1)
        ctrl_layout.addWidget(self.lbl_trpl_state, 1, 3, 1, 4)

        root.addWidget(ctrl_group)

        slider_group = QGroupBox("Time Slice")
        slider_layout = QGridLayout(slider_group)
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setMinimum(0)
        self.slider_time.setMaximum(0)
        self.slider_time.setEnabled(False)
        self.lbl_time = QLabel("t = 0.000 ns")
        self.spin_range_start_ns = QDoubleSpinBox()
        self.spin_range_start_ns.setRange(0.0, 1e9)
        self.spin_range_start_ns.setDecimals(4)
        self.spin_range_start_ns.setValue(0.0)
        self.spin_range_end_ns = QDoubleSpinBox()
        self.spin_range_end_ns.setRange(0.0, 1e9)
        self.spin_range_end_ns.setDecimals(4)
        self.spin_range_end_ns.setValue(0.0)
        self.btn_apply_range = QPushButton("Apply Time Range")
        self.btn_apply_range.setEnabled(False)
        self.chk_ref_frame = QCheckBox("Use this frame as reference")
        self.chk_ref_frame.setEnabled(False)

        slider_layout.addWidget(QLabel("Start (ns)"), 0, 0)
        slider_layout.addWidget(self.spin_range_start_ns, 0, 1)
        slider_layout.addWidget(QLabel("End (ns)"), 0, 2)
        slider_layout.addWidget(self.spin_range_end_ns, 0, 3)
        slider_layout.addWidget(self.btn_apply_range, 0, 4)
        slider_layout.addWidget(QLabel("Time bin"), 1, 0)
        slider_layout.addWidget(self.slider_time, 1, 1, 1, 4)
        slider_layout.addWidget(QLabel("Selected time"), 2, 0)
        slider_layout.addWidget(self.lbl_time, 2, 1, 1, 2)
        slider_layout.addWidget(self.chk_ref_frame, 2, 3, 1, 2)
        root.addWidget(slider_group)

        self.trpl_fig = Figure(figsize=(8.0, 7.2), dpi=100)
        self.trpl_canvas = FigureCanvas(self.trpl_fig)
        gg = self.trpl_fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], wspace=0.35, hspace=0.45)
        self.trpl_hist_ax = self.trpl_fig.add_subplot(gg[0, :])
        self.trpl_pix_ax = self.trpl_fig.add_subplot(gg[1, 0])
        self.trpl_map_ax = self.trpl_fig.add_subplot(gg[1, 1])
        self.trpl_fit_ax = self.trpl_fig.add_subplot(gg[1, 2])
        root.addWidget(self.trpl_canvas, 1)

        self.tabs.addTab(tab, "TRPL")

        self.btn_calib_align.clicked.connect(self._on_calib_align)
        self.btn_take_trpl.clicked.connect(self._on_take_trpl)
        self.btn_export_trpl.clicked.connect(self._on_export_trpl_csv)
        self.slider_time.valueChanged.connect(self._on_time_slider_changed)
        self.btn_apply_range.clicked.connect(self._on_apply_time_range)
        self.chk_ref_frame.toggled.connect(self._on_toggle_reference_frame)
        self.chk_hist_log.toggled.connect(lambda _v: self._update_trpl_histogram_only())

    def _setup_count_timer(self) -> None:
        self.count_timer = QTimer(self)
        self.count_timer.setInterval(500)
        self.count_timer.timeout.connect(self._poll_counts)
        self.count_timer.start()

    def _stop_count_updates(self) -> None:
        if self.count_timer.isActive():
            self.count_timer.stop()

    def _start_count_updates_if_allowed(self) -> None:
        if self.live_updates_enabled and (not self.trpl_running):
            if not self.count_timer.isActive():
                self.count_timer.start()

    def _toggle_live_updates(self) -> None:
        self.live_updates_enabled = not self.live_updates_enabled
        if self.live_updates_enabled:
            self.btn_toggle_live.setText("Pause Live Counts")
            self.lbl_left_state.setText("Status: live updates resumed")
            self._start_count_updates_if_allowed()
        else:
            self.btn_toggle_live.setText("Resume Live Counts")
            self.lbl_left_state.setText("Status: live updates paused")
            self._stop_count_updates()

    def _init_left_map(self) -> None:
        self.left_ax.clear()
        self.left_scatter = self.left_ax.scatter(
            self.map_coords[:, 0],
            self.map_coords[:, 1],
            c=np.zeros(NUM_PIXELS),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            s=950,
            edgecolors="#5f6368",
            linewidths=2.0,
        )
        self.left_selected_marker = self.left_ax.scatter([], [], s=1200, facecolors="none", edgecolors="#22d3ee", linewidths=2.2)
        self.left_excluded_marker = self.left_ax.scatter([], [], s=340, marker="x", c="#ef4444", linewidths=2.2)
        self.left_hover_annot = self.left_ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#9ca3af", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#6b7280"},
        )
        self.left_hover_annot.set_visible(False)
        for pix, (x, y) in PIXEL_COORDS.items():
            self.left_ax.text(x, y, str(pix), ha="center", va="center", color="#f7f7f7", fontsize=10, fontweight="bold")

        self.left_ax.set_aspect("equal")
        self.left_ax.set_xlim(-0.7, 4.7)
        self.left_ax.set_ylim(-0.7, 4.7)
        self.left_ax.set_xticks([])
        self.left_ax.set_yticks([])
        self.left_ax.set_title("SPAD23 Count-Rate Map")

        self.left_fig.colorbar(self.left_scatter, ax=self.left_ax, fraction=0.046, pad=0.04, label="Normalized [0,1]")
        self.left_canvas.mpl_connect("button_press_event", self._on_left_map_click)
        self.left_canvas.mpl_connect("motion_notify_event", self._on_left_map_hover)
        self._update_selected_marker()
        self._update_excluded_markers()
        self.left_canvas.draw_idle()

    def _update_selected_marker(self) -> None:
        x, y = PIXEL_COORDS[self.selected_pixel]
        self.left_selected_marker.set_offsets(np.array([[x, y]], dtype=np.float64))

    def _update_excluded_markers(self) -> None:
        if not self.excluded_pixels:
            self.left_excluded_marker.set_offsets(np.empty((0, 2)))
            return
        pts = np.array([PIXEL_COORDS[p] for p in sorted(self.excluded_pixels)], dtype=np.float64)
        self.left_excluded_marker.set_offsets(pts)

    def _get_excluded_fit_points(self) -> Optional[np.ndarray]:
        if not self.excluded_pixels:
            return None
        idx = np.array(sorted(self.excluded_pixels), dtype=int)
        return np.column_stack((self.fit_x_um[idx], self.fit_y_um[idx]))

    def _poll_counts(self) -> None:
        if self.trpl_running:
            return
        try:
            if not self.count_client.is_connected:
                self.count_client.connect()
            snap = self.count_client.get_counts_snapshot(integration_ms=200)
            self.latest_count_snapshot = snap
            self._update_left_from_count(snap)
            if self.tabs.tabText(self.tabs.currentIndex()) == "Counts":
                self._update_counts_tab(snap.pixel_rates_mcps)
            self.lbl_left_state.setText("Status: live counts")
        except Spad23OverflowError as exc:
            snap = exc.snapshot
            if snap is not None:
                self.latest_count_snapshot = snap
                self._update_left_from_count(snap)
                if self.tabs.tabText(self.tabs.currentIndex()) == "Counts":
                    self._update_counts_tab(snap.pixel_rates_mcps)
            self.lbl_left_state.setText(f"Status: {exc}")
        except Spad23Error as exc:
            self.lbl_left_state.setText(f"Status: {exc}")
        except Exception as exc:
            self.lbl_left_state.setText(f"Status: {exc}")

    def _update_left_from_count(self, snap: CountSnapshot) -> None:
        rates = snap.pixel_rates_mcps
        rmin = float(np.min(rates))
        rmax = float(np.max(rates))
        if rmax > rmin:
            norm = (rates - rmin) / (rmax - rmin)
        else:
            norm = np.zeros_like(rates)
        self.left_scatter.set_array(norm)
        self._update_selected_marker()
        self._update_excluded_markers()
        if self.hover_pixel is None:
            self.left_hover_annot.set_visible(False)
        else:
            self._update_hover_annotation(self.hover_pixel)
        self.left_canvas.draw_idle()
        self._update_left_info_label()

    def _clear_reference_state(self) -> None:
        self.reference_fit = None
        self.reference_frame_idx = None
        self.reference_frame_popt = None
        self.reference_fit_cache.clear()
        if hasattr(self, "chk_ref_frame"):
            self.chk_ref_frame.blockSignals(True)
            self.chk_ref_frame.setChecked(False)
            self.chk_ref_frame.blockSignals(False)

    def _fit_inputs(self, values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        mask = np.ones(NUM_PIXELS, dtype=bool)
        if self.excluded_pixels:
            excluded = np.array(sorted(self.excluded_pixels), dtype=int)
            mask[excluded] = False
        if int(np.sum(mask)) < 6:
            return None
        return self.fit_x_um[mask], self.fit_y_um[mask], values[mask], mask

    def _current_tab_name(self) -> str:
        return self.tabs.tabText(self.tabs.currentIndex())

    def _maybe_refresh_sigma_trace(self) -> None:
        if self._current_tab_name() == "Fitting":
            self._refresh_sigma_trace()

    def _update_counts_tab(self, rates_mcps: np.ndarray) -> None:
        h_pixels = np.array([9, 10, 11, 12, 13], dtype=np.int64)
        h_vals = rates_mcps[h_pixels]
        v_labels = ["2", "(6+7)/2", "11", "(15+16)/2", "20"]
        v_vals = np.array(
            [
                rates_mcps[2],
                0.5 * (rates_mcps[6] + rates_mcps[7]),
                rates_mcps[11],
                0.5 * (rates_mcps[15] + rates_mcps[16]),
                rates_mcps[20],
            ],
            dtype=np.float64,
        )

        unit, factor = _auto_rate_scale(np.concatenate([h_vals, v_vals]))

        self.counts_hcut_ax.clear()
        self.counts_hcut_ax.plot(h_pixels, h_vals * factor, "-o", color="#2563eb")
        self.counts_hcut_ax.set_title("Horizontal linecut")
        self.counts_hcut_ax.set_xlabel("Pixel")
        self.counts_hcut_ax.set_ylabel(f"Intensity ({unit})")
        self.counts_hcut_ax.grid(True, alpha=0.3)

        self.counts_vcut_ax.clear()
        xi = np.arange(5)
        self.counts_vcut_ax.plot(xi, v_vals * factor, "-o", color="#d97706")
        self.counts_vcut_ax.set_xticks(xi)
        self.counts_vcut_ax.set_xticklabels(v_labels, rotation=20, ha="right")
        self.counts_vcut_ax.set_title("Vertical linecut")
        self.counts_vcut_ax.set_ylabel(f"Intensity ({unit})")
        self.counts_vcut_ax.grid(True, alpha=0.3)

        self._plot_map_and_fit(
            ax_map=self.counts_fit_ax,
            ax_fit=self.counts_radial_ax,
            values=rates_mcps,
            unit=unit,
            factor=factor,
            map_title="2D Gaussian fit",
            fit_title="Gaussian radial fit",
        )
        self.counts_canvas.draw_idle()

    def _plot_map_and_fit(
        self,
        ax_map,
        ax_fit,
        values: np.ndarray,
        unit: str,
        factor: float,
        map_title: str,
        fit_title: str,
    ) -> None:
        z = values.astype(np.float64)
        z_plot = z * factor

        ax_map.clear()
        fit_inputs = self._fit_inputs(z)
        fit = None
        if fit_inputs is not None:
            x_fit, y_fit, z_fit_data, mask = fit_inputs
            fit = _fit_gaussian_2d(x_fit, y_fit, z_fit_data)
        else:
            mask = np.ones(NUM_PIXELS, dtype=bool)
        if fit is None:
            zmin = float(np.min(z_plot))
            zmax = float(np.max(z_plot))
            if zmax <= zmin:
                zmax = zmin + 1e-9
            levels = np.linspace(zmin, zmax, 50)
            ax_map.tricontourf(self.fit_tri, z_plot, levels=levels, cmap="viridis")
            ax_map.scatter(
                self.fit_x_um,
                self.fit_y_um,
                c=z_plot,
                cmap="viridis",
                edgecolors="k",
                s=50,
                vmin=zmin,
                vmax=zmax,
            )
            ax_map.set_title(map_title)
            ax_map.set_xlabel("x (um)")
            ax_map.set_ylabel("y (um)")
            ax_map.set_aspect("equal")
            ax_map.invert_yaxis()
            ex = self._get_excluded_fit_points()
            if ex is not None:
                ax_map.scatter(ex[:, 0], ex[:, 1], s=120, marker="x", c="#ef4444", linewidths=1.8)

            ax_fit.clear()
            ax_fit.text(0.5, 0.5, "Fit unavailable", ha="center", va="center", transform=ax_fit.transAxes)
            ax_fit.set_title(fit_title)
            ax_fit.set_xlabel("Distance (um)")
            ax_fit.set_ylabel(f"Intensity ({unit})")
            ax_fit.grid(True, alpha=0.25)
            return

        popt = fit.popt
        xpad = PITCH_X_UM * 0.35
        ypad = PITCH_Y_UM * 0.35
        xmin = float(np.min(self.fit_x_um) - xpad)
        xmax = float(np.max(self.fit_x_um) + xpad)
        ymin = float(np.min(self.fit_y_um) - ypad)
        ymax = float(np.max(self.fit_y_um) + ypad)

        gx = np.linspace(xmin, xmax, 220)
        gy = np.linspace(ymin, ymax, 220)
        gxx, gyy = np.meshgrid(gx, gy)
        z_fit_grid = (
            popt[0]
            * np.exp(-(((gxx - popt[1]) ** 2) / (2.0 * popt[3] ** 2) + ((gyy - popt[2]) ** 2) / (2.0 * popt[4] ** 2)))
            + popt[5]
        ) * factor

        zmin = float(np.min(z_fit_grid))
        zmax = float(np.max(z_fit_grid))
        if zmax <= zmin:
            zmax = zmin + 1e-9
        levels = np.linspace(zmin, zmax, 80)
        ax_map.contourf(gxx, gyy, z_fit_grid, levels=levels, cmap="viridis")
        ax_map.scatter(
            self.fit_x_um,
            self.fit_y_um,
            c=z_plot,
            cmap="viridis",
            edgecolors="k",
            s=50,
            vmin=zmin,
            vmax=zmax,
        )
        ex = self._get_excluded_fit_points()
        if ex is not None:
            ax_map.scatter(ex[:, 0], ex[:, 1], s=120, marker="x", c="#ef4444", linewidths=1.8)
        ax_map.plot(float(popt[1]), float(popt[2]), "wx", markersize=9, markeredgewidth=2)
        ax_map.set_title(map_title, y=1.12)
        ax_map.text(
            0.5,
            1.03,
            f"x0={popt[1]:.1f} um, y0={popt[2]:.1f} um, sigma_eq={fit.sigma_eq:.1f} um",
            transform=ax_map.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#111827",
            clip_on=False,
        )
        ax_map.set_xlabel("x (um)")
        ax_map.set_ylabel("y (um)")
        ax_map.set_aspect("equal")
        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)
        ax_map.invert_yaxis()

        ax_fit.clear()
        x_used = self.fit_x_um[mask]
        y_used = self.fit_y_um[mask]
        z_used = z_plot[mask]
        r = np.sqrt((x_used - popt[1]) ** 2 + (y_used - popt[2]) ** 2)
        idx = np.argsort(r)
        r_sorted = r[idx]
        z_sorted = z_used[idx]
        r_sym = np.concatenate([-r_sorted[::-1], r_sorted])
        z_sym = np.concatenate([z_sorted[::-1], z_sorted])

        rmax = float(np.max(r))
        r_line = np.linspace(-rmax, rmax, 401)
        sigma_safe = max(float(fit.sigma_eq), 1e-6)
        fit_line = (popt[0] * np.exp(-(r_line**2) / (2.0 * sigma_safe**2)) + popt[5]) * factor

        ax_fit.scatter(r_sym, z_sym, s=20, alpha=0.75, color="#2563eb")
        ax_fit.plot(r_line, fit_line, "-", color="#111827", linewidth=1.7)
        ax_fit.set_title(f"{fit_title} (sigma={fit.sigma_eq:.1f} um)")
        ax_fit.set_xlabel("Distance from fitted center (um)")
        ax_fit.set_ylabel(f"Intensity ({unit})")
        ax_fit.grid(True, alpha=0.25)

    def _on_left_map_click(self, event) -> None:
        if event.inaxes != self.left_ax or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        dist2 = (self.map_coords[:, 0] - x) ** 2 + (self.map_coords[:, 1] - y) ** 2
        nearest = int(np.argmin(dist2))
        if float(np.sqrt(dist2[nearest])) > 0.42:
            return
        if event.button in (MouseButton.RIGHT, 3):
            if nearest in self.excluded_pixels:
                self.excluded_pixels.remove(nearest)
            else:
                self.excluded_pixels.add(nearest)
            self._clear_reference_state()
            self._update_excluded_markers()
            if self.latest_count_snapshot is not None:
                self._update_counts_tab(self.latest_count_snapshot.pixel_rates_mcps)
            if self.latest_trpl_snapshot is not None:
                self._update_trpl_histogram_only()
                self._update_trpl_frame_plots()
                self._maybe_refresh_sigma_trace()
            self.left_canvas.draw_idle()
            return

        self.selected_pixel = nearest
        self.lbl_selected_pixel.setText(f"Selected pixel: {self.selected_pixel}")
        self._update_selected_marker()
        self.left_canvas.draw_idle()
        self._update_trpl_histogram_only()

    def _on_left_map_hover(self, event) -> None:
        if event.inaxes != self.left_ax or self.latest_count_snapshot is None:
            if self.hover_pixel is not None:
                self.hover_pixel = None
                self.left_hover_annot.set_visible(False)
                self._update_left_info_label()
                self.left_canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        dist2 = (self.map_coords[:, 0] - x) ** 2 + (self.map_coords[:, 1] - y) ** 2
        nearest = int(np.argmin(dist2))
        if float(np.sqrt(dist2[nearest])) <= 0.42:
            if self.hover_pixel != nearest:
                self.hover_pixel = nearest
                self._update_hover_annotation(nearest)
                self._update_left_info_label()
                self.left_canvas.draw_idle()
        else:
            if self.hover_pixel is not None:
                self.hover_pixel = None
                self.left_hover_annot.set_visible(False)
                self._update_left_info_label()
                self.left_canvas.draw_idle()

    def _update_hover_annotation(self, pixel: int) -> None:
        if self.latest_count_snapshot is None:
            return
        x, y = PIXEL_COORDS[pixel]
        count = int(self.latest_count_snapshot.pixel_counts[pixel])
        rate = float(self.latest_count_snapshot.pixel_rates_mcps[pixel])
        self.left_hover_annot.xy = (x, y)
        self.left_hover_annot.set_text(f"Px {pixel}\n{count} counts\n{_format_rate(rate)}")
        self.left_hover_annot.set_visible(True)

    def _update_left_info_label(self) -> None:
        snap = self.latest_count_snapshot
        if snap is None:
            self.lbl_left_info.setText("Total: --")
            return
        ratio_pct = 100.0 * snap.overload_ratio
        if self.hover_pixel is None:
            self.lbl_left_info.setText(
                f"Total {_format_rate(snap.total_rate_mcps)} | Max pixel {snap.max_pixel_index}: "
                f"{_format_rate(snap.max_pixel_rate_mcps)} | Limit ratio {ratio_pct:.1f}%"
            )
        else:
            pix = int(self.hover_pixel)
            self.lbl_left_info.setText(
                f"Pixel {pix}: {int(snap.pixel_counts[pix])} counts, {_format_rate(float(snap.pixel_rates_mcps[pix]))} "
                f"| Total {_format_rate(snap.total_rate_mcps)}"
            )

    def _on_calib_align(self) -> None:
        if self.trpl_running:
            return
        self._set_controls_enabled(False)
        self.lbl_trpl_state.setText("Calibrating + aligning...")
        self._stop_count_updates()
        self.count_client.close()
        self.calib_worker = CalibAlignWorker(host=self.host, port=self.port, align_ms=1000)
        self.calib_worker.done.connect(self._on_calib_align_done)
        self.calib_worker.error.connect(self._on_calib_align_error)
        self.calib_worker.start()

    def _on_calib_align_done(self, msg: str) -> None:
        self.lbl_trpl_state.setText(msg)
        self._set_controls_enabled(True)
        self._start_count_updates_if_allowed()

    def _on_calib_align_error(self, msg: str) -> None:
        self.lbl_trpl_state.setText(f"Calibration/alignment failed: {msg}")
        self._set_controls_enabled(True)
        self._start_count_updates_if_allowed()

    def _on_take_trpl(self) -> None:
        if self.trpl_running:
            return
        measurement_ms = int(round(self.spin_acq_s.value() * 1000.0))
        bin_ps = int(self.spin_bin_ps.value())

        self.trpl_running = True
        self._trpl_update_counter = 0
        self._set_controls_enabled(False)
        self.lbl_trpl_state.setText("TRPL acquisition running...")
        self._stop_count_updates()
        self.count_client.close()

        self.latest_trpl_snapshot = None
        self.reference_fit = None
        self.reference_frame_idx = None
        self.reference_frame_popt = None
        self.reference_fit_cache.clear()
        self.sigma_time_ns = np.array([], dtype=np.float64)
        self.sigma_values = np.array([], dtype=np.float64)
        if hasattr(self, "fitting_ax"):
            self.fitting_ax.clear()
            self.fitting_ax.set_title("Sigma_eq vs Time")
            self.fitting_ax.set_xlabel("Time (ns)")
            self.fitting_ax.set_ylabel("Sigma_eq (um)")
            self.fitting_ax.grid(True, alpha=0.25)
            self.fitting_canvas.draw_idle()
            self.lbl_fit_status.setText("Acquisition running... sigma trace will update after completion.")
        self.chk_ref_frame.blockSignals(True)
        self.chk_ref_frame.setChecked(False)
        self.chk_ref_frame.blockSignals(False)
        self.chk_ref_frame.setEnabled(False)
        self.slider_time.setEnabled(False)
        self.btn_apply_range.setEnabled(False)
        self.spin_range_start_ns.setEnabled(False)
        self.spin_range_end_ns.setEnabled(False)
        self.slider_time.setMinimum(0)
        self.slider_time.setMaximum(0)
        self.slider_time.setValue(0)
        self.lbl_time.setText("t = 0.000 ns")

        self.trpl_worker = TrplAcquireWorker(
            host=self.host,
            port=self.port,
            measurement_ms=measurement_ms,
            bin_width_ps=bin_ps,
        )
        self.trpl_worker.update.connect(self._on_trpl_update)
        self.trpl_worker.done.connect(self._on_trpl_done)
        self.trpl_worker.error.connect(self._on_trpl_error)
        self.trpl_worker.start()

    def _on_trpl_update(self, snapshot_obj: object) -> None:
        snapshot = snapshot_obj if isinstance(snapshot_obj, TrplSnapshot) else None
        if snapshot is None:
            return
        if snapshot.done:
            # Final result is handled by _on_trpl_done; skip duplicate heavy redraw here.
            return
        self.latest_trpl_snapshot = snapshot
        self._trpl_update_counter += 1
        # Invalidate sequential cache as counts are still evolving during acquisition.
        if self.chk_ref_frame.isChecked() and self.reference_frame_idx is not None:
            self.reference_fit_cache.clear()
            if self.reference_frame_popt is not None:
                ref_sigma = float(np.sqrt((self.reference_frame_popt[3] ** 2 + self.reference_frame_popt[4] ** 2) / 2.0))
                self.reference_fit_cache[self.reference_frame_idx] = FitResult(
                    popt=self.reference_frame_popt.copy(),
                    z_fit=np.zeros(NUM_PIXELS, dtype=np.float64),
                    sigma_eq=ref_sigma,
                )
        self.lbl_trpl_state.setText(
            f"TRPL running: {snapshot.elapsed_s:.1f} s, bins={snapshot.counts_matrix.shape[0]}, "
            f"events={snapshot.total_counts}"
        )

        max_bin = max(snapshot.counts_matrix.shape[0] - 1, 0)
        slider_at_end = self.slider_time.value() >= self.slider_time.maximum()
        self.slider_start_idx = 0
        self.slider_end_idx = max_bin
        self.slider_time.setEnabled(False)
        self.btn_apply_range.setEnabled(False)
        self.spin_range_start_ns.setEnabled(False)
        self.spin_range_end_ns.setEnabled(False)
        self.chk_ref_frame.setEnabled(False)
        self.slider_time.blockSignals(True)
        self.slider_time.setMinimum(self.slider_start_idx)
        self.slider_time.setMaximum(self.slider_end_idx)
        if slider_at_end:
            self.slider_time.setValue(max_bin)
        self.slider_time.blockSignals(False)
        self.spin_range_start_ns.setValue(0.0)
        if max_bin > 0:
            self.spin_range_end_ns.setValue((float(max_bin) * float(snapshot.bin_width_ps)) / 1000.0)

        # Only redraw live TRPL plots when TRPL tab is visible, and throttle cadence.
        if self.tabs.tabText(self.tabs.currentIndex()) == "TRPL":
            if self._trpl_update_counter % LIVE_PLOT_UPDATE_EVERY == 0:
                self._update_trpl_histogram_only()
            if self._trpl_update_counter % (2 * LIVE_PLOT_UPDATE_EVERY) == 0:
                self._update_trpl_frame_plots()

    def _on_trpl_done(self, snapshot_obj: object) -> None:
        snapshot = snapshot_obj if isinstance(snapshot_obj, TrplSnapshot) else None
        if snapshot is None:
            self._on_trpl_error("Invalid TRPL result")
            return

        self.latest_trpl_snapshot = snapshot
        self.trpl_running = False
        self._set_controls_enabled(True)
        self._start_count_updates_if_allowed()

        max_bin = max(snapshot.counts_matrix.shape[0] - 1, 0)
        self.slider_start_idx = 0
        self.slider_end_idx = max_bin
        self.slider_time.setEnabled(max_bin > 0)
        self.btn_apply_range.setEnabled(max_bin > 0)
        self.spin_range_start_ns.setEnabled(max_bin > 0)
        self.spin_range_end_ns.setEnabled(max_bin > 0)
        self.chk_ref_frame.setEnabled(max_bin > 0)
        self.slider_time.blockSignals(True)
        self.slider_time.setMinimum(self.slider_start_idx)
        self.slider_time.setMaximum(max_bin)
        self.slider_time.setValue(min(self.slider_time.value(), max_bin))
        self.slider_time.blockSignals(False)
        self.spin_range_start_ns.setValue(0.0)
        self.spin_range_end_ns.setValue((float(max_bin) * float(snapshot.bin_width_ps)) / 1000.0 if max_bin > 0 else 0.0)

        self.lbl_trpl_state.setText(
            f"TRPL done: {snapshot.elapsed_s:.1f} s, bins={snapshot.counts_matrix.shape[0]}, "
            f"events={snapshot.total_counts}"
        )
        self._update_trpl_histogram_only()
        self._update_trpl_frame_plots()
        self._maybe_refresh_sigma_trace()

    def _on_trpl_error(self, msg: str) -> None:
        self.trpl_running = False
        self._set_controls_enabled(True)
        self._start_count_updates_if_allowed()
        has_data = self.latest_trpl_snapshot is not None and self.latest_trpl_snapshot.counts_matrix.shape[0] > 0
        self.slider_time.setEnabled(has_data)
        self.btn_apply_range.setEnabled(has_data)
        self.spin_range_start_ns.setEnabled(has_data)
        self.spin_range_end_ns.setEnabled(has_data)
        self.chk_ref_frame.setEnabled(has_data)
        self.lbl_trpl_state.setText(f"TRPL error: {msg}")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_calib_align.setEnabled(enabled)
        self.btn_take_trpl.setEnabled(enabled)
        self.spin_acq_s.setEnabled(enabled)
        self.spin_bin_ps.setEnabled(enabled)

    def _on_export_trpl_csv(self) -> None:
        snap = self.latest_trpl_snapshot
        if snap is None or snap.counts_matrix.shape[0] == 0:
            QMessageBox.warning(self, "Export TRPL CSV", "No TRPL data available to export.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"spad23_trpl_{ts}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save TRPL CSV",
            default_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not file_path:
            return

        n_bins = int(snap.counts_matrix.shape[0])
        time_ns = (np.arange(n_bins, dtype=np.float64) * float(snap.bin_width_ps)) / 1000.0
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["time_ns"] + [f"pixel_{i}" for i in range(NUM_PIXELS)]
                writer.writerow(header)
                for i in range(n_bins):
                    writer.writerow([f"{time_ns[i]:.6f}"] + snap.counts_matrix[i, :].astype(np.int64).tolist())
        except Exception as exc:
            QMessageBox.critical(self, "Export TRPL CSV", f"Failed to save CSV:\n{exc}")
            return

        self.lbl_trpl_state.setText(f"TRPL CSV exported: {file_path}")

    def _on_tab_changed(self, _idx: int) -> None:
        tab_name = self.tabs.tabText(self.tabs.currentIndex())
        if tab_name == "Counts" and self.latest_count_snapshot is not None:
            self._update_counts_tab(self.latest_count_snapshot.pixel_rates_mcps)
        if tab_name == "TRPL" and self.latest_trpl_snapshot is not None:
            self._update_trpl_histogram_only()
            self._update_trpl_frame_plots()
        if tab_name == "Fitting":
            self._refresh_sigma_trace()

    def _on_time_slider_changed(self, value: int) -> None:
        if self.trpl_running and (not self.slider_time.isEnabled()):
            return
        snap = self.latest_trpl_snapshot
        if snap is None or snap.counts_matrix.shape[0] == 0:
            self.lbl_time.setText("t = 0.000 ns")
            return
        idx = int(np.clip(value, self.slider_start_idx, self.slider_end_idx))
        t_ns = (float(idx) * float(snap.bin_width_ps)) / 1000.0
        self.lbl_time.setText(f"t = {t_ns:.4f} ns (bin {idx})")
        self._update_trpl_histogram_only()
        self._update_trpl_frame_plots()

    def _on_toggle_reference_frame(self, checked: bool) -> None:
        if not checked:
            self.reference_fit = None
            self.reference_frame_idx = None
            self.reference_frame_popt = None
            self.reference_fit_cache.clear()
            self.lbl_trpl_state.setText("Reference frame disabled.")
            self._update_trpl_frame_plots()
            self._maybe_refresh_sigma_trace()
            return

        snap = self.latest_trpl_snapshot
        if snap is None or snap.counts_matrix.shape[0] == 0:
            self.chk_ref_frame.blockSignals(True)
            self.chk_ref_frame.setChecked(False)
            self.chk_ref_frame.blockSignals(False)
            self.reference_fit = None
            self.lbl_trpl_state.setText("No TRPL frame available for reference.")
            return

        idx = int(np.clip(self.slider_time.value(), self.slider_start_idx, self.slider_end_idx))
        frame = snap.counts_matrix[idx, :].astype(np.float64)
        fit_inputs = self._fit_inputs(frame)
        if fit_inputs is None:
            self.chk_ref_frame.blockSignals(True)
            self.chk_ref_frame.setChecked(False)
            self.chk_ref_frame.blockSignals(False)
            self.reference_fit = None
            self.lbl_trpl_state.setText("Not enough enabled pixels for reference fit.")
            return
        x_fit, y_fit, z_fit_data, _ = fit_inputs
        fit = _fit_gaussian_2d(x_fit, y_fit, z_fit_data)
        if fit is None:
            self.chk_ref_frame.blockSignals(True)
            self.chk_ref_frame.setChecked(False)
            self.chk_ref_frame.blockSignals(False)
            self.reference_fit = None
            self.lbl_trpl_state.setText("Could not fit selected frame; reference not set.")
            return

        x0, y0, sigma_eq = float(fit.popt[1]), float(fit.popt[2]), float(fit.sigma_eq)
        self.reference_fit = (x0, y0, sigma_eq)
        self.reference_frame_idx = idx
        self.reference_frame_popt = fit.popt.copy()
        self.reference_fit_cache.clear()
        self.reference_fit_cache[idx] = fit
        self.lbl_trpl_state.setText(
            f"Reference set from bin {idx}: x0={x0:.1f} um, y0={y0:.1f} um, sigma_eq={sigma_eq:.1f} um"
        )
        self._update_trpl_frame_plots()
        self._maybe_refresh_sigma_trace()

    def _on_apply_time_range(self) -> None:
        snap = self.latest_trpl_snapshot
        if snap is None or snap.counts_matrix.shape[0] == 0:
            return
        t_start_ns = float(self.spin_range_start_ns.value())
        t_end_ns = float(self.spin_range_end_ns.value())
        if t_end_ns < t_start_ns:
            t_start_ns, t_end_ns = t_end_ns, t_start_ns

        bin_ns = float(snap.bin_width_ps) / 1000.0
        if bin_ns <= 0.0:
            return
        start_idx = int(np.ceil(t_start_ns / bin_ns))
        end_idx = int(np.floor(t_end_ns / bin_ns))
        start_idx = int(np.clip(start_idx, 0, snap.counts_matrix.shape[0] - 1))
        end_idx = int(np.clip(end_idx, start_idx, snap.counts_matrix.shape[0] - 1))

        self.slider_start_idx = start_idx
        self.slider_end_idx = end_idx
        self.slider_time.blockSignals(True)
        self.slider_time.setMinimum(start_idx)
        self.slider_time.setMaximum(end_idx)
        self.slider_time.setValue(int(np.clip(self.slider_time.value(), start_idx, end_idx)))
        self.slider_time.blockSignals(False)
        self._update_trpl_histogram_only()
        self._update_trpl_frame_plots()
        self._maybe_refresh_sigma_trace()

    def _update_trpl_histogram_only(self) -> None:
        snap = self.latest_trpl_snapshot
        if snap is None or snap.counts_matrix.shape[0] == 0:
            return

        pix = int(np.clip(self.selected_pixel, 0, NUM_PIXELS - 1))
        y = snap.counts_matrix[:, pix]
        n_bins = y.shape[0]
        idx_axis = np.arange(n_bins, dtype=np.int64)
        if self.trpl_running and n_bins > LIVE_HIST_MAX_POINTS:
            stride = int(np.ceil(float(n_bins) / float(LIVE_HIST_MAX_POINTS)))
            idx_axis = idx_axis[::stride]
        x_ns = idx_axis.astype(np.float64) * (float(snap.bin_width_ps) / 1000.0)
        y_plot = y[idx_axis]
        idx = int(np.clip(self.slider_time.value(), self.slider_start_idx, self.slider_end_idx))
        selected_y = float(y[idx])
        selected_t_ns = (float(idx) * float(snap.bin_width_ps)) / 1000.0
        start_t_ns = (float(self.slider_start_idx) * float(snap.bin_width_ps)) / 1000.0
        end_t_ns = (float(self.slider_end_idx) * float(snap.bin_width_ps)) / 1000.0
        self.trpl_hist_ax.clear()
        self.trpl_hist_ax.plot(x_ns, y_plot, color="#2563eb", linewidth=1.2)
        self.trpl_hist_ax.axvline(selected_t_ns, color="#111827", linestyle="--", linewidth=1.0, alpha=0.8)
        y_line = max(selected_y, 1.0) if self.chk_hist_log.isChecked() else selected_y
        self.trpl_hist_ax.axhline(y_line, color="#059669", linestyle=":", linewidth=1.0, alpha=0.8)
        self.trpl_hist_ax.axvspan(start_t_ns, end_t_ns, color="#dbeafe", alpha=0.18)
        self.trpl_hist_ax.set_title(f"Pixel {pix} histogram")
        self.trpl_hist_ax.set_xlabel("Time (ns)")
        self.trpl_hist_ax.set_ylabel("Counts")
        if self.chk_hist_log.isChecked():
            self.trpl_hist_ax.set_yscale("log")
            positive = y_plot[y_plot > 0]
            if positive.size > 0:
                ymin = max(1.0, float(np.min(positive)) * 0.8)
                ymax = max(float(np.max(positive)) * 1.2, ymin * 2.0)
                self.trpl_hist_ax.set_ylim(ymin, ymax)
            else:
                self.trpl_hist_ax.set_ylim(1.0, 10.0)
        else:
            self.trpl_hist_ax.set_yscale("linear")
        self.trpl_hist_ax.grid(True, alpha=0.25)
        self.trpl_canvas.draw_idle()

    def _update_trpl_frame_plots(self) -> None:
        snap = self.latest_trpl_snapshot
        if snap is None or snap.counts_matrix.shape[0] == 0:
            return

        idx = int(np.clip(self.slider_time.value(), self.slider_start_idx, self.slider_end_idx))
        frame = snap.counts_matrix[idx, :].astype(np.float64)
        t_ns = (float(idx) * float(snap.bin_width_ps)) / 1000.0
        if self.chk_ref_frame.isChecked() and self.reference_frame_idx is not None and self.reference_frame_popt is not None:
            fit = self._fit_trpl_frame_with_reference(idx=idx, snap=snap)
        else:
            fit_inputs = self._fit_inputs(frame)
            if fit_inputs is None:
                fit = None
                mask = np.ones(NUM_PIXELS, dtype=bool)
            else:
                x_fit, y_fit, z_fit_data, mask = fit_inputs
                fit = _fit_gaussian_2d(x_fit, y_fit, z_fit_data)

        self.trpl_pix_ax.clear()
        self.trpl_pix_ax.plot(np.arange(NUM_PIXELS), frame, "-o", color="#1d4ed8", markersize=4)
        self.trpl_pix_ax.set_title(f"Counts vs pixel @ {t_ns:.4f} ns")
        self.trpl_pix_ax.set_xlabel("Pixel")
        self.trpl_pix_ax.set_ylabel("Counts")
        self.trpl_pix_ax.grid(True, alpha=0.25)

        # 2D Gaussian-view map at selected time (smooth + margin style)
        self.trpl_map_ax.clear()
        if fit is None:
            zmin = float(np.min(frame))
            zmax = float(np.max(frame))
            if zmax <= zmin:
                zmax = zmin + 1e-9
            levels = np.linspace(zmin, zmax, 40)
            self.trpl_map_ax.tricontourf(self.fit_tri, frame, levels=levels, cmap="viridis")
            self.trpl_map_ax.scatter(
                self.fit_x_um,
                self.fit_y_um,
                c=frame,
                cmap="viridis",
                edgecolors="k",
                s=45,
                vmin=zmin,
                vmax=zmax,
            )
            ex = self._get_excluded_fit_points()
            if ex is not None:
                self.trpl_map_ax.scatter(ex[:, 0], ex[:, 1], s=110, marker="x", c="#ef4444", linewidths=1.8)
            self.trpl_map_ax.set_title(f"2D Gaussian fit @ {t_ns:.4f} ns")
        else:
            popt = fit.popt
            xpad = PITCH_X_UM * 0.35
            ypad = PITCH_Y_UM * 0.35
            xmin = float(np.min(self.fit_x_um) - xpad)
            xmax = float(np.max(self.fit_x_um) + xpad)
            ymin = float(np.min(self.fit_y_um) - ypad)
            ymax = float(np.max(self.fit_y_um) + ypad)

            gx = np.linspace(xmin, xmax, 220)
            gy = np.linspace(ymin, ymax, 220)
            gxx, gyy = np.meshgrid(gx, gy)
            z_fit_grid = (
                popt[0]
                * np.exp(-(((gxx - popt[1]) ** 2) / (2.0 * popt[3] ** 2) + ((gyy - popt[2]) ** 2) / (2.0 * popt[4] ** 2)))
                + popt[5]
            )

            zmin = float(np.min(z_fit_grid))
            zmax = float(np.max(z_fit_grid))
            if zmax <= zmin:
                zmax = zmin + 1e-9
            levels = np.linspace(zmin, zmax, 80)
            self.trpl_map_ax.contourf(gxx, gyy, z_fit_grid, levels=levels, cmap="viridis")
            self.trpl_map_ax.scatter(
                self.fit_x_um,
                self.fit_y_um,
                c=frame,
                cmap="viridis",
                edgecolors="k",
                s=45,
                vmin=zmin,
                vmax=zmax,
            )
            ex = self._get_excluded_fit_points()
            if ex is not None:
                self.trpl_map_ax.scatter(ex[:, 0], ex[:, 1], s=110, marker="x", c="#ef4444", linewidths=1.8)
            self.trpl_map_ax.plot(float(popt[1]), float(popt[2]), "wx", markersize=8, markeredgewidth=2)
            self.trpl_map_ax.set_title(f"2D Gaussian fit @ {t_ns:.4f} ns", y=1.12)
            self.trpl_map_ax.text(
                0.5,
                1.03,
                f"x0={popt[1]:.1f} um, y0={popt[2]:.1f} um, sigma_eq={fit.sigma_eq:.1f} um",
                transform=self.trpl_map_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#111827",
                clip_on=False,
            )
            self.trpl_map_ax.set_xlim(xmin, xmax)
            self.trpl_map_ax.set_ylim(ymin, ymax)
        if fit is None:
            self.trpl_map_ax.set_title(f"2D Gaussian fit @ {t_ns:.4f} ns")
        self.trpl_map_ax.set_xlabel("x (um)")
        self.trpl_map_ax.set_ylabel("y (um)")
        self.trpl_map_ax.set_aspect("equal")
        self.trpl_map_ax.invert_yaxis()

        self.trpl_fit_ax.clear()
        if fit is None:
            self.trpl_fit_ax.text(0.5, 0.5, "Fit unavailable", ha="center", va="center", transform=self.trpl_fit_ax.transAxes)
            self.trpl_fit_ax.set_title("Gaussian fit")
            self.trpl_fit_ax.set_xlabel("Distance (um)")
            self.trpl_fit_ax.set_ylabel("Counts")
            self.trpl_fit_ax.grid(True, alpha=0.25)
        else:
            popt = fit.popt
            fit_inputs = self._fit_inputs(frame)
            if fit_inputs is None:
                x_used = self.fit_x_um
                y_used = self.fit_y_um
                z_used = frame
            else:
                x_used, y_used, z_used, _ = fit_inputs
            r = np.sqrt((x_used - popt[1]) ** 2 + (y_used - popt[2]) ** 2)
            idx_sort = np.argsort(r)
            r_sorted = r[idx_sort]
            z_sorted = z_used[idx_sort]
            r_sym = np.concatenate([-r_sorted[::-1], r_sorted])
            z_sym = np.concatenate([z_sorted[::-1], z_sorted])

            rmax = float(np.max(r))
            r_line = np.linspace(-rmax, rmax, 401)
            sigma_safe = max(float(fit.sigma_eq), 1e-6)
            fit_line = popt[0] * np.exp(-(r_line**2) / (2.0 * sigma_safe**2)) + popt[5]

            self.trpl_fit_ax.scatter(r_sym, z_sym, s=20, alpha=0.75, color="#2563eb")
            self.trpl_fit_ax.plot(r_line, fit_line, "-", color="#111827", linewidth=1.8)
            self.trpl_fit_ax.set_title(
                f"Gaussian fit @ {t_ns:.4f} ns\n"
                f"x0={popt[1]:.1f} um, y0={popt[2]:.1f} um, sigma={fit.sigma_eq:.1f} um"
            )
            self.trpl_fit_ax.set_xlabel("Distance from fitted center (um)")
            self.trpl_fit_ax.set_ylabel("Counts")
            self.trpl_fit_ax.grid(True, alpha=0.25)

        self.lbl_time.setText(f"t = {t_ns:.4f} ns (bin {idx})")
        self.trpl_canvas.draw_idle()

    def _fit_trpl_frame_with_reference(self, idx: int, snap: TrplSnapshot) -> Optional[FitResult]:
        assert self.reference_frame_idx is not None
        assert self.reference_frame_popt is not None

        ref_idx = int(self.reference_frame_idx)
        if idx < ref_idx:
            # Request says only frames after reference are constrained.
            frame = snap.counts_matrix[idx, :].astype(np.float64)
            fit_inputs = self._fit_inputs(frame)
            if fit_inputs is None:
                return None
            x_fit, y_fit, z_fit_data, _ = fit_inputs
            return _fit_gaussian_2d(x_fit, y_fit, z_fit_data)

        if ref_idx not in self.reference_fit_cache:
            ref_sigma = float(np.sqrt((self.reference_frame_popt[3] ** 2 + self.reference_frame_popt[4] ** 2) / 2.0))
            self.reference_fit_cache[ref_idx] = FitResult(
                popt=self.reference_frame_popt.copy(),
                z_fit=np.zeros(NUM_PIXELS, dtype=np.float64),
                sigma_eq=ref_sigma,
            )

        if idx in self.reference_fit_cache:
            cached = self.reference_fit_cache[idx]
            if np.any(cached.z_fit):
                return cached
            # Ref seed entry may have zero z_fit placeholder; fit it explicitly if idx == ref.
            if idx == ref_idx:
                frame_ref = snap.counts_matrix[idx, :].astype(np.float64)
                fit_inputs = self._fit_inputs(frame_ref)
                if fit_inputs is None:
                    return None
                x_fit, y_fit, z_fit_data, _ = fit_inputs
                fit_ref = _fit_gaussian_2d(x_fit, y_fit, z_fit_data)
                if fit_ref is not None:
                    self.reference_fit_cache[idx] = fit_ref
                    return fit_ref
                return None

        valid_cached = [k for k in self.reference_fit_cache.keys() if k <= idx]
        start_idx = max(valid_cached) if valid_cached else ref_idx
        prev_fit = self.reference_fit_cache.get(start_idx)
        if prev_fit is None:
            frame0 = snap.counts_matrix[ref_idx, :].astype(np.float64)
            fit_inputs = self._fit_inputs(frame0)
            if fit_inputs is None:
                return None
            x_fit, y_fit, z_fit_data, _ = fit_inputs
            prev_fit = _fit_gaussian_2d(x_fit, y_fit, z_fit_data)
            if prev_fit is None:
                return None
            self.reference_fit_cache[ref_idx] = prev_fit
            start_idx = ref_idx

        for j in range(start_idx + 1, idx + 1):
            frame_j = snap.counts_matrix[j, :].astype(np.float64)
            fit_inputs = self._fit_inputs(frame_j)
            if fit_inputs is None:
                return None
            x_fit, y_fit, z_fit_data, _ = fit_inputs
            fit_j = _fit_gaussian_2d_constrained_next(x_fit, y_fit, z_fit_data, prev_fit.popt, tightness=1.0)
            if fit_j is None:
                fit_j = _fit_gaussian_2d_constrained_next(x_fit, y_fit, z_fit_data, prev_fit.popt, tightness=1.8)
            if fit_j is None:
                return None
            self.reference_fit_cache[j] = fit_j
            prev_fit = fit_j

        return self.reference_fit_cache.get(idx, prev_fit)

    def _refresh_sigma_trace(self) -> None:
        snap = self.latest_trpl_snapshot
        self.fitting_ax.clear()
        self.fitting_ax.set_title("Sigma_eq vs Time")
        self.fitting_ax.set_xlabel("Time (ns)")
        self.fitting_ax.set_ylabel("Sigma_eq (um)")
        self.fitting_ax.grid(True, alpha=0.25)

        if not _HAS_SCIPY:
            self.lbl_fit_status.setText("scipy is required for sigma-vs-time fitting.")
            self.fitting_canvas.draw_idle()
            return

        if snap is None or snap.counts_matrix.shape[0] == 0:
            self.lbl_fit_status.setText("Run TRPL to generate sigma-vs-time.")
            self.fitting_canvas.draw_idle()
            return

        start = int(np.clip(self.slider_start_idx, 0, snap.counts_matrix.shape[0] - 1))
        end = int(np.clip(self.slider_end_idx, start, snap.counts_matrix.shape[0] - 1))
        idxs_full = list(range(start, end + 1))
        if not idxs_full:
            self.lbl_fit_status.setText("No frames in selected range.")
            self.fitting_canvas.draw_idle()
            return

        max_points = 500
        stride = max(1, int(np.ceil(len(idxs_full) / max_points)))
        idxs = idxs_full[::stride]

        sigma_vals: List[float] = []
        time_vals: List[float] = []
        failures = 0

        for i in idxs:
            frame = snap.counts_matrix[i, :].astype(np.float64)
            if self.chk_ref_frame.isChecked() and self.reference_frame_idx is not None and self.reference_frame_popt is not None:
                fit = self._fit_trpl_frame_with_reference(i, snap)
            else:
                fit_inputs = self._fit_inputs(frame)
                if fit_inputs is None:
                    fit = None
                else:
                    x_fit, y_fit, z_fit_data, _ = fit_inputs
                    fit = _fit_gaussian_2d(x_fit, y_fit, z_fit_data)

            if fit is None:
                failures += 1
                continue
            sigma_vals.append(float(fit.sigma_eq))
            time_vals.append((float(i) * float(snap.bin_width_ps)) / 1000.0)

        if not sigma_vals:
            self.lbl_fit_status.setText("Sigma trace unavailable (fits failed or too few enabled pixels).")
            self.fitting_canvas.draw_idle()
            return

        t_arr = np.asarray(time_vals, dtype=np.float64)
        s_arr = np.asarray(sigma_vals, dtype=np.float64)
        self.sigma_time_ns = t_arr
        self.sigma_values = s_arr
        self.fitting_ax.plot(t_arr, s_arr, "-o", markersize=2.8, linewidth=1.1, color="#1d4ed8")
        self.fitting_ax.grid(True, alpha=0.25)
        mode = "reference-constrained" if self.chk_ref_frame.isChecked() else "free fit"
        sampled = f", sampled every {stride} frame(s)" if stride > 1 else ""
        excl = f", excluded pixels={len(self.excluded_pixels)}" if self.excluded_pixels else ""
        self.lbl_fit_status.setText(
            f"Sigma trace computed ({mode}{sampled}{excl}). Points={len(s_arr)}, failures={failures}."
        )
        self.fitting_canvas.draw_idle()

    def closeEvent(self, event) -> None:
        try:
            self._stop_count_updates()
        except Exception:
            pass
        try:
            self.count_client.close()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPAD23 Counts + TRPL GUI")
    parser.add_argument("--host", default="127.0.0.1", help="TCP host of the LabVIEW server")
    parser.add_argument("--port", type=int, default=9999, help="TCP port of the LabVIEW server")
    args = parser.parse_args()

    app = QApplication([])
    window = Spad23MainWindow(host=args.host, port=args.port)
    window.resize(1800, 980)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

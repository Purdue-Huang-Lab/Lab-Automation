"""Offline SPAD23 TRPL CSV viewer with histogram and fitting tabs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

try:
    from scipy.optimize import curve_fit

    _HAS_SCIPY = True
except Exception:
    curve_fit = None  # type: ignore
    _HAS_SCIPY = False


NUM_PIXELS = 23
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


@dataclass
class FitResult:
    popt: np.ndarray
    z_fit: np.ndarray
    sigma_eq: float


@dataclass
class RadialFitResult:
    popt: np.ndarray  # [A, sigma, offset]
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
        [0.0, float(np.min(x) - PITCH_X_UM),
         float(np.min(y) - PITCH_Y_UM), 1e-3, 1e-3, -np.inf],
        [np.inf, float(np.max(x) + PITCH_X_UM),
         float(np.max(y) + PITCH_Y_UM), np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0,
                            bounds=bounds, maxfev=20000)  # type: ignore[misc]
    except Exception:
        return None

    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return FitResult(popt=np.asarray(popt, dtype=np.float64), z_fit=np.asarray(z_fit, dtype=np.float64), sigma_eq=sigma_eq)


def _fit_gaussian_2d_constrained_next(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prev_popt: np.ndarray,
    tightness: float = 1.0,
) -> Optional[FitResult]:
    if not _HAS_SCIPY:
        return None
    if np.allclose(z, z[0]):
        return None

    A_prev, x0_prev, y0_prev, sx_prev, sy_prev, off_prev = [
        float(v) for v in prev_popt]
    sx_prev = max(sx_prev, 1e-3)
    sy_prev = max(sy_prev, 1e-3)

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
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0, bounds=(
            lower, upper), maxfev=20000)  # type: ignore[misc]
    except Exception:
        return None

    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return FitResult(popt=np.asarray(popt, dtype=np.float64), z_fit=np.asarray(z_fit, dtype=np.float64), sigma_eq=sigma_eq)


def _fit_radial_gaussian_1d(
    r: np.ndarray,
    z: np.ndarray,
    p0: Optional[np.ndarray] = None,
    sigma_guess: Optional[float] = None,
) -> Optional[RadialFitResult]:
    if not _HAS_SCIPY:
        return None
    if r.size < 3 or z.size < 3:
        return None
    if np.allclose(z, z[0]):
        return None

    r = r.astype(np.float64)
    z = z.astype(np.float64)

    def gauss1d(rg, A, sigma, offset):
        return A * np.exp(-(rg**2) / (2.0 * sigma**2)) + offset

    if p0 is not None and p0.size == 3:
        p0_use = [float(p0[0]), max(float(p0[1]), 1e-3), float(p0[2])]
    else:
        A0 = max(float(np.max(z) - np.min(z)), 1e-6)
        sigma0 = max(float(sigma_guess)
                     if sigma_guess is not None else float(np.std(r)), 1e-3)
        offset0 = float(np.min(z))
        p0_use = [A0, sigma0, offset0]

    bounds = ([0.0, 1e-3, -np.inf], [np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(gauss1d, r, z, p0=p0_use,
                            bounds=bounds, maxfev=20000)  # type: ignore[misc]
    except Exception:
        return None

    sigma_eq = float(abs(popt[1]))
    return RadialFitResult(popt=np.asarray(popt, dtype=np.float64), sigma_eq=sigma_eq)


class Spad23TrplViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.time_ns = np.array([], dtype=np.float64)
        self.counts_matrix = np.zeros((0, NUM_PIXELS), dtype=np.int64)
        self.current_idx = 0
        self.range_start_idx = 0
        self.range_end_idx = 0
        self.reference_idx: Optional[int] = None
        self.selected_pixels: Set[int] = {11}
        self.excluded_pixels: Set[int] = set()

        self.fit_x_um, self.fit_y_um = _build_fit_coords()
        self.fit_tri = Triangulation(self.fit_x_um, self.fit_y_um)
        self.map_coords = np.array([PIXEL_COORDS[i]
                                   for i in range(NUM_PIXELS)], dtype=np.float64)

        self.fit_results_m1: Dict[int, FitResult] = {}
        self.fit_results_m2: Dict[int, RadialFitResult] = {}
        self.m2_reference_fit: Optional[FitResult] = None
        self.m2_reference_center: Optional[Tuple[float, float]] = None
        self.m2_reference_mask: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("SPAD23 TRPL CSV Viewer")
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        file_row = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText(
            "Select a TRPL CSV file with columns: time_ns,pixel_0,...,pixel_22")
        self.btn_browse = QPushButton("Browse")
        self.btn_load = QPushButton("Load")
        self.lbl_file_state = QLabel("No file loaded.")
        file_row.addWidget(QLabel("CSV file"))
        file_row.addWidget(self.edit_path, 1)
        file_row.addWidget(self.btn_browse)
        file_row.addWidget(self.btn_load)
        root.addLayout(file_row)
        root.addWidget(self.lbl_file_state)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left_layout = QVBoxLayout()
        self.left_fig = Figure(figsize=(6.5, 7.0), dpi=100)
        self.left_canvas = FigureCanvas(self.left_fig)
        self.left_ax = self.left_fig.add_subplot(111)
        left_layout.addWidget(self.left_canvas, 1)
        self.lbl_left_info = QLabel("Time: -- | Selected pixels: --")
        left_layout.addWidget(self.lbl_left_info)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(540)
        splitter.addWidget(left_widget)

        right_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs, 1)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([760, 980])

        self._build_hist_tab()
        self._build_fitting_tab()
        self._init_left_map()

        self.btn_browse.clicked.connect(self._on_browse_file)
        self.btn_load.clicked.connect(self._on_load_file)
        self.tabs.currentChanged.connect(
            lambda _i: self._refresh_after_state_change())

    def _build_hist_tab(self) -> None:
        tab = QWidget()
        root = QVBoxLayout(tab)

        group = QGroupBox("Time Slice")
        g = QGridLayout(group)

        self.spin_start_ns = QDoubleSpinBox()
        self.spin_start_ns.setRange(0.0, 1e12)
        self.spin_start_ns.setDecimals(6)
        self.spin_end_ns = QDoubleSpinBox()
        self.spin_end_ns.setRange(0.0, 1e12)
        self.spin_end_ns.setDecimals(6)
        self.btn_apply_range = QPushButton("Apply Time Range")
        self.chk_reference = QCheckBox("Use this frame as reference")
        self.chk_hist_log = QCheckBox("Log histogram y")
        self.btn_export_hist = QPushButton("Export Line Traces CSV")

        self.slider_hist = QSlider(Qt.Horizontal)
        self.slider_hist.setMinimum(0)
        self.slider_hist.setMaximum(0)
        self.slider_hist.setEnabled(False)
        self.lbl_time_hist = QLabel("t = 0.0000 ns")
        self.lbl_hist_state = QLabel(
            "Left-click pixels on map to show/hide histogram traces.")

        g.addWidget(QLabel("Start (ns)"), 0, 0)
        g.addWidget(self.spin_start_ns, 0, 1)
        g.addWidget(QLabel("End (ns)"), 0, 2)
        g.addWidget(self.spin_end_ns, 0, 3)
        g.addWidget(self.btn_apply_range, 0, 4)
        g.addWidget(self.chk_reference, 0, 5)
        g.addWidget(self.chk_hist_log, 0, 6)
        g.addWidget(self.btn_export_hist, 0, 7)
        g.addWidget(QLabel("Time bin"), 1, 0)
        g.addWidget(self.slider_hist, 1, 1, 1, 6)
        g.addWidget(self.lbl_time_hist, 1, 7)
        g.addWidget(self.lbl_hist_state, 2, 0, 1, 8)
        root.addWidget(group)

        self.hist_fig = Figure(figsize=(8.0, 7.0), dpi=100)
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.hist_ax = self.hist_fig.add_subplot(111)
        root.addWidget(self.hist_canvas, 1)

        self.tabs.addTab(tab, "Histogram")

        self.btn_apply_range.clicked.connect(self._on_apply_range)
        self.slider_hist.valueChanged.connect(self._on_hist_slider_changed)
        self.chk_hist_log.toggled.connect(
            lambda _v: self._update_histogram_plot())
        self.chk_reference.toggled.connect(self._on_toggle_reference)
        self.btn_export_hist.clicked.connect(self._on_export_hist_traces)

    def _build_fitting_tab(self) -> None:
        tab = QWidget()
        root = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.cmb_fit_method = QComboBox()
        self.cmb_fit_method.addItem("1: sequential 2D Gaussian fit")
        self.cmb_fit_method.addItem("2: fixed-center radial 1D Gaussian fit")
        self.btn_fit_all = QPushButton("Fit All")
        self.btn_export_sigma = QPushButton("Export Sigma^2 CSV")
        self.slider_fit = QSlider(Qt.Horizontal)
        self.slider_fit.setMinimum(0)
        self.slider_fit.setMaximum(0)
        self.slider_fit.setEnabled(False)
        self.lbl_time_fit = QLabel("t = 0.0000 ns")
        top.addWidget(QLabel("Method"))
        top.addWidget(self.cmb_fit_method, 1)
        top.addWidget(self.btn_fit_all)
        top.addWidget(self.btn_export_sigma)
        top.addWidget(QLabel("Time bin"))
        top.addWidget(self.slider_fit, 2)
        top.addWidget(self.lbl_time_fit)
        root.addLayout(top)

        self.lbl_fit_state = QLabel(
            "Choose a reference frame in Histogram tab, then click Fit All.")
        root.addWidget(self.lbl_fit_state)

        self.fit_fig = Figure(figsize=(8.0, 7.2), dpi=100)
        self.fit_canvas = FigureCanvas(self.fit_fig)
        gs = self.fit_fig.add_gridspec(
            2, 2, height_ratios=[1.0, 1.2], wspace=0.35, hspace=0.4)
        self.ax_sigma = self.fit_fig.add_subplot(gs[0, :])
        self.ax_fit_map = self.fit_fig.add_subplot(gs[1, 0])
        self.ax_fit_radial = self.fit_fig.add_subplot(gs[1, 1])
        root.addWidget(self.fit_canvas, 1)

        self.tabs.addTab(tab, "Fitting")

        self.btn_fit_all.clicked.connect(self._on_fit_all)
        self.btn_export_sigma.clicked.connect(self._on_export_sigma_trace)
        self.slider_fit.valueChanged.connect(self._on_fit_slider_changed)
        self.cmb_fit_method.currentIndexChanged.connect(
            lambda _i: self._refresh_after_state_change())

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
        self.left_selected_marker = self.left_ax.scatter(
            [], [], s=1200, facecolors="none", edgecolors="#22d3ee", linewidths=2.2)
        self.left_excluded_marker = self.left_ax.scatter(
            [], [], s=340, marker="x", c="#ef4444", linewidths=2.2)
        for pix, (x, y) in PIXEL_COORDS.items():
            self.left_ax.text(x, y, str(pix), ha="center", va="center",
                              color="#f7f7f7", fontsize=10, fontweight="bold")
        self.left_ax.set_aspect("equal")
        self.left_ax.set_xlim(-0.7, 4.7)
        self.left_ax.set_ylim(-0.7, 4.7)
        self.left_ax.set_xticks([])
        self.left_ax.set_yticks([])
        self.left_ax.set_title("Count Map @ Selected Time Bin")
        self.left_fig.colorbar(self.left_scatter, ax=self.left_ax,
                               fraction=0.046, pad=0.04, label="Normalized [0,1]")
        self.left_canvas.mpl_connect(
            "button_press_event", self._on_left_map_click)
        self._update_left_markers()
        self.left_canvas.draw_idle()

    def _update_left_markers(self) -> None:
        if not self.selected_pixels:
            self.left_selected_marker.set_offsets(np.empty((0, 2)))
        else:
            pts = np.array([PIXEL_COORDS[p] for p in sorted(
                self.selected_pixels)], dtype=np.float64)
            self.left_selected_marker.set_offsets(pts)

        if not self.excluded_pixels:
            self.left_excluded_marker.set_offsets(np.empty((0, 2)))
        else:
            ex = np.array([PIXEL_COORDS[p] for p in sorted(
                self.excluded_pixels)], dtype=np.float64)
            self.left_excluded_marker.set_offsets(ex)

    def _fit_inputs(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        mask = np.ones(NUM_PIXELS, dtype=bool)
        if self.excluded_pixels:
            idx = np.array(sorted(self.excluded_pixels), dtype=int)
            mask[idx] = False
        if int(np.sum(mask)) < 6:
            return None
        return self.fit_x_um[mask], self.fit_y_um[mask], frame[mask], mask

    def _clear_fit_results(self) -> None:
        self.fit_results_m1.clear()
        self.fit_results_m2.clear()
        self.m2_reference_fit = None
        self.m2_reference_center = None
        self.m2_reference_mask = None

    def _default_export_dir(self) -> str:
        path = self.edit_path.text().strip()
        if path:
            parent = Path(path).expanduser().parent
            if parent.exists():
                return str(parent)
        return ""

    def _choose_export_path(self, default_name: str) -> Optional[Path]:
        default_dir = self._default_export_dir()
        default_path = str(Path(default_dir) /
                           default_name) if default_dir else default_name
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        if not file_path:
            return None
        out = Path(file_path)
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        return out

    def _hist_trace_data(self) -> Optional[Tuple[np.ndarray, List[int], np.ndarray]]:
        if self.counts_matrix.shape[0] == 0:
            return None
        selected = sorted(self.selected_pixels)
        if not selected:
            return None
        sl = slice(self.range_start_idx, self.range_end_idx + 1)
        t = np.asarray(self.time_ns[sl], dtype=np.float64)
        y = np.asarray(self.counts_matrix[sl, :]
                       [:, selected], dtype=np.float64)
        return t, selected, y

    def _sigma_trace_data(self) -> Optional[Tuple[int, np.ndarray, np.ndarray, List[int], Dict[int, object]]]:
        if self.counts_matrix.shape[0] == 0:
            return None
        method = int(self.cmb_fit_method.currentIndex())
        results: Dict[int, object]
        if method == 0:
            results = self.fit_results_m1
        else:
            results = self.fit_results_m2
        if not results:
            return None
        valid = [i for i in sorted(
            results) if self.range_start_idx <= i <= self.range_end_idx]
        if method == 1 and self.reference_idx is not None:
            valid = [i for i in valid if i >= int(self.reference_idx)]
        if not valid:
            return None
        t = np.array([self.time_ns[i] for i in valid], dtype=np.float64)
        # type: ignore[attr-defined]
        sigma = np.array([float(results[i].sigma_eq)
                         for i in valid], dtype=np.float64)
        return method, t, sigma, valid, results

    def _on_browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open TRPL CSV", "", "CSV files (*.csv);;All files (*)")
        if file_path:
            self.edit_path.setText(file_path)

    def _on_load_file(self) -> None:
        path = self.edit_path.text().strip()
        if not path:
            QMessageBox.warning(
                self, "Load CSV", "Please choose a CSV file first.")
            return
        self._load_csv(path)

    def _load_csv(self, path: str) -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            QMessageBox.critical(self, "Load CSV", f"File not found:\n{path}")
            return

        required = ["time_ns"] + [f"pixel_{i}" for i in range(NUM_PIXELS)]
        time_vals: List[float] = []
        pixel_vals: List[List[int]] = []
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("Missing CSV header.")
                missing = [
                    name for name in required if name not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
                for row in reader:
                    time_vals.append(float(row["time_ns"]))
                    pixel_vals.append(
                        [int(float(row[f"pixel_{i}"])) for i in range(NUM_PIXELS)])
        except Exception as exc:
            QMessageBox.critical(
                self, "Load CSV", f"Could not parse file:\n{exc}")
            return

        if not time_vals:
            QMessageBox.warning(self, "Load CSV", "CSV has no data rows.")
            return

        self.time_ns = np.asarray(time_vals, dtype=np.float64)
        self.counts_matrix = np.asarray(pixel_vals, dtype=np.int64)
        self.current_idx = 0
        self.range_start_idx = 0
        self.range_end_idx = int(self.counts_matrix.shape[0] - 1)
        self.reference_idx = None
        self._clear_fit_results()
        self.selected_pixels = {11}
        self.excluded_pixels = set()

        self.chk_reference.blockSignals(True)
        self.chk_reference.setChecked(False)
        self.chk_reference.blockSignals(False)

        self.slider_hist.setEnabled(True)
        self.slider_fit.setEnabled(True)
        self.btn_apply_range.setEnabled(True)
        self.btn_fit_all.setEnabled(True)
        self.spin_start_ns.setEnabled(True)
        self.spin_end_ns.setEnabled(True)

        self.spin_start_ns.setValue(float(self.time_ns[0]))
        self.spin_end_ns.setValue(float(self.time_ns[-1]))
        self._set_slider_ranges(self.range_start_idx, self.range_end_idx)
        self._set_current_idx(0)

        self.lbl_file_state.setText(
            f"Loaded: {csv_path.name} | bins={self.counts_matrix.shape[0]}")
        self.lbl_hist_state.setText(
            "Left-click pixels on map to show/hide histogram traces.")
        self.lbl_fit_state.setText(
            "Choose a reference frame in Histogram tab, then click Fit All.")
        self._refresh_after_state_change()

    def _set_slider_ranges(self, start_idx: int, end_idx: int) -> None:
        self.slider_hist.blockSignals(True)
        self.slider_fit.blockSignals(True)
        self.slider_hist.setMinimum(start_idx)
        self.slider_hist.setMaximum(end_idx)
        self.slider_fit.setMinimum(start_idx)
        self.slider_fit.setMaximum(end_idx)
        self.slider_hist.blockSignals(False)
        self.slider_fit.blockSignals(False)

    def _on_hist_slider_changed(self, value: int) -> None:
        self._set_current_idx(value)

    def _on_fit_slider_changed(self, value: int) -> None:
        self._set_current_idx(value)

    def _set_current_idx(self, idx: int) -> None:
        if self.counts_matrix.shape[0] == 0:
            return
        idx = int(np.clip(idx, self.range_start_idx, self.range_end_idx))
        self.current_idx = idx

        self.slider_hist.blockSignals(True)
        self.slider_fit.blockSignals(True)
        self.slider_hist.setValue(idx)
        self.slider_fit.setValue(idx)
        self.slider_hist.blockSignals(False)
        self.slider_fit.blockSignals(False)

        t_ns = float(self.time_ns[idx])
        self.lbl_time_hist.setText(f"t = {t_ns:.4f} ns (bin {idx})")
        self.lbl_time_fit.setText(f"t = {t_ns:.4f} ns (bin {idx})")
        self._refresh_after_state_change()

    def _on_apply_range(self) -> None:
        if self.counts_matrix.shape[0] == 0:
            return
        t0 = float(self.spin_start_ns.value())
        t1 = float(self.spin_end_ns.value())
        if t1 < t0:
            t0, t1 = t1, t0

        start_idx = int(np.searchsorted(self.time_ns, t0, side="left"))
        end_idx = int(np.searchsorted(self.time_ns, t1, side="right") - 1)
        start_idx = int(np.clip(start_idx, 0, self.counts_matrix.shape[0] - 1))
        end_idx = int(np.clip(end_idx, start_idx,
                      self.counts_matrix.shape[0] - 1))

        self.range_start_idx = start_idx
        self.range_end_idx = end_idx
        self._set_slider_ranges(start_idx, end_idx)
        self._set_current_idx(self.current_idx)

    def _on_toggle_reference(self, checked: bool) -> None:
        if self.counts_matrix.shape[0] == 0:
            self.reference_idx = None
            return
        if checked:
            self.reference_idx = int(self.current_idx)
            self._clear_fit_results()
            self.lbl_hist_state.setText(
                f"Reference set at bin {self.reference_idx}, t={self.time_ns[self.reference_idx]:.4f} ns.")
        else:
            self.reference_idx = None
            self._clear_fit_results()
            self.lbl_hist_state.setText("Reference cleared.")
        self._refresh_after_state_change()

    def _on_left_map_click(self, event) -> None:
        if event.inaxes != self.left_ax or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        dist2 = (self.map_coords[:, 0] - x) ** 2 + \
            (self.map_coords[:, 1] - y) ** 2
        nearest = int(np.argmin(dist2))
        if float(np.sqrt(dist2[nearest])) > 0.42:
            return
        if event.button in (MouseButton.RIGHT, 3):
            if nearest in self.excluded_pixels:
                self.excluded_pixels.remove(nearest)
            else:
                self.excluded_pixels.add(nearest)
            self._clear_fit_results()
            self._update_left_markers()
            self._update_fitting_plots()
            self._update_sigma_plot()
            self.left_canvas.draw_idle()
            return

        if event.button in (MouseButton.LEFT, 1):
            if nearest in self.selected_pixels:
                self.selected_pixels.remove(nearest)
            else:
                self.selected_pixels.add(nearest)
            self._update_left_markers()
            self._update_histogram_plot()
            self.left_canvas.draw_idle()

    def _on_fit_all(self) -> None:
        if not _HAS_SCIPY:
            QMessageBox.warning(
                self, "Fit All", "scipy is required for fitting.")
            return
        if self.counts_matrix.shape[0] == 0:
            QMessageBox.warning(self, "Fit All", "No data loaded.")
            return
        if self.reference_idx is None:
            QMessageBox.warning(
                self, "Fit All", "Please set a reference frame in Histogram tab first.")
            return

        start_idx = max(self.range_start_idx, int(self.reference_idx))
        end_idx = self.range_end_idx
        if start_idx > end_idx:
            QMessageBox.warning(
                self, "Fit All", "Selected time range does not include frames after the reference.")
            return

        method = self.cmb_fit_method.currentIndex()
        ref_frame = self.counts_matrix[int(
            self.reference_idx), :].astype(np.float64)
        ref_inputs = self._fit_inputs(ref_frame)
        if ref_inputs is None:
            QMessageBox.warning(
                self, "Fit All", "Not enough enabled pixels for fitting (need >= 6).")
            return
        x_fit_ref, y_fit_ref, z_fit_ref, mask_ref = ref_inputs
        ref_fit_2d = _fit_gaussian_2d(x_fit_ref, y_fit_ref, z_fit_ref)
        if ref_fit_2d is None:
            QMessageBox.warning(
                self, "Fit All", "Could not fit the reference frame.")
            return

        if method == 0:
            self.fit_results_m1.clear()
            self.fit_results_m1[int(self.reference_idx)] = ref_fit_2d
            prev_fit = ref_fit_2d
            failures = 0
            for i in range(start_idx, end_idx + 1):
                if i == int(self.reference_idx):
                    continue
                frame = self.counts_matrix[i, :].astype(np.float64)
                fit_inputs = self._fit_inputs(frame)
                if fit_inputs is None:
                    failures += 1
                    continue
                x_fit, y_fit, z_fit, _ = fit_inputs
                fit = _fit_gaussian_2d_constrained_next(
                    x_fit, y_fit, z_fit, prev_fit.popt, tightness=1.0)
                if fit is None:
                    fit = _fit_gaussian_2d_constrained_next(
                        x_fit, y_fit, z_fit, prev_fit.popt, tightness=1.8)
                if fit is None:
                    failures += 1
                    continue
                self.fit_results_m1[i] = fit
                prev_fit = fit
                if (i - start_idx) % 200 == 0:
                    QApplication.processEvents()

            fitted = len(
                [k for k in self.fit_results_m1 if start_idx <= k <= end_idx])
            self.lbl_fit_state.setText(
                f"Method 1 done: fitted {fitted} frame(s), failures={failures}, "
                f"range={start_idx}-{end_idx}, ref={self.reference_idx}."
            )
        elif method == 1:
            # Method 2: fixed center from reference frame + per-frame radial 1D Gaussian fit.
            x0_ref = float(ref_fit_2d.popt[1])
            y0_ref = float(ref_fit_2d.popt[2])
            r_fixed = np.sqrt((x_fit_ref - x0_ref) ** 2 +
                              (y_fit_ref - y0_ref) ** 2)
            order = np.argsort(r_fixed)
            r_fixed = r_fixed[order]

            self.fit_results_m2.clear()
            self.m2_reference_fit = ref_fit_2d
            self.m2_reference_center = (x0_ref, y0_ref)
            self.m2_reference_mask = mask_ref.copy()

            ref_radial = _fit_radial_gaussian_1d(
                r_fixed,
                z_fit_ref[order],
                p0=None,
                sigma_guess=ref_fit_2d.sigma_eq,
            )
            if ref_radial is None:
                QMessageBox.warning(
                    self, "Fit All", "Reference radial 1D fit failed for method 2.")
                return
            self.fit_results_m2[int(self.reference_idx)] = ref_radial
            prev_radial = ref_radial
            failures = 0

            for i in range(start_idx, end_idx + 1):
                if i == int(self.reference_idx):
                    continue
                frame = self.counts_matrix[i, :].astype(np.float64)
                fit_inputs = self._fit_inputs(frame)
                if fit_inputs is None:
                    failures += 1
                    continue
                _, _, z_fit_i, mask_i = fit_inputs
                if mask_i.shape != mask_ref.shape or np.any(mask_i != mask_ref):
                    failures += 1
                    continue
                z_sorted = z_fit_i[order]
                fit1d = _fit_radial_gaussian_1d(
                    r_fixed,
                    z_sorted,
                    p0=prev_radial.popt,
                    sigma_guess=prev_radial.sigma_eq,
                )
                if fit1d is None:
                    fit1d = _fit_radial_gaussian_1d(
                        r_fixed,
                        z_sorted,
                        p0=None,
                        sigma_guess=prev_radial.sigma_eq,
                    )
                if fit1d is None:
                    failures += 1
                    continue
                self.fit_results_m2[i] = fit1d
                prev_radial = fit1d
                if (i - start_idx) % 200 == 0:
                    QApplication.processEvents()

            fitted = len(
                [k for k in self.fit_results_m2 if start_idx <= k <= end_idx])
            self.lbl_fit_state.setText(
                f"Method 2 done: fitted {fitted} frame(s), failures={failures}, "
                f"range={start_idx}-{end_idx}, ref={self.reference_idx}, fixed center=({x0_ref:.2f},{y0_ref:.2f}) um."
            )
        else:
            QMessageBox.warning(self, "Fit All", "Unknown fitting method.")
            return

        self._update_sigma_plot()
        self._update_fitting_plots()

    def _refresh_after_state_change(self) -> None:
        self._update_left_map()
        self._update_histogram_plot()
        self._update_sigma_plot()
        self._update_fitting_plots()

    def _update_left_map(self) -> None:
        if self.counts_matrix.shape[0] == 0:
            return
        frame = self.counts_matrix[self.current_idx, :].astype(np.float64)
        vmin = float(np.min(frame))
        vmax = float(np.max(frame))
        if vmax > vmin:
            norm = (frame - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(frame)
        self.left_scatter.set_array(norm)
        self._update_left_markers()
        sel = ",".join(str(p) for p in sorted(self.selected_pixels)
                       ) if self.selected_pixels else "none"
        excl = ",".join(str(p) for p in sorted(
            self.excluded_pixels)) if self.excluded_pixels else "none"
        self.lbl_left_info.setText(
            f"Time {self.time_ns[self.current_idx]:.4f} ns (bin {self.current_idx}) | "
            f"Selected pixels: {sel} | Excluded from fit: {excl}"
        )
        self.left_canvas.draw_idle()

    def _update_histogram_plot(self) -> None:
        self.hist_ax.clear()
        if self.counts_matrix.shape[0] == 0:
            self.hist_ax.text(0.5, 0.5, "Load a TRPL CSV file.",
                              ha="center", va="center", transform=self.hist_ax.transAxes)
            self.hist_canvas.draw_idle()
            return

        t = self.time_ns
        t_slice = t[self.range_start_idx: self.range_end_idx + 1]
        selected = sorted(self.selected_pixels)
        if not selected:
            self.hist_ax.text(0.5, 0.5, "No pixel selected. Left-click map to toggle.",
                              ha="center", va="center", transform=self.hist_ax.transAxes)
        else:
            for pix in selected:
                y_slice = self.counts_matrix[self.range_start_idx: self.range_end_idx + 1, pix]
                self.hist_ax.plot(t_slice, y_slice,
                                  linewidth=1.15, label=f"Px {pix}")
            self.hist_ax.legend(loc="upper right", fontsize=8)

        t_sel = float(self.time_ns[self.current_idx])
        t_start = float(self.time_ns[self.range_start_idx])
        t_end = float(self.time_ns[self.range_end_idx])
        self.hist_ax.axvline(t_sel, color="#111827",
                             linestyle="--", linewidth=1.0, alpha=0.85)
        self.hist_ax.axvspan(t_start, t_end, color="#dbeafe", alpha=0.18)
        self.hist_ax.set_xlim(t_start, t_end if t_end >
                              t_start else t_start + 1e-9)
        if self.reference_idx is not None:
            self.hist_ax.axvline(float(
                self.time_ns[self.reference_idx]), color="#059669", linestyle="-.", linewidth=1.0, alpha=0.9)

        if self.chk_hist_log.isChecked():
            self.hist_ax.set_yscale("log")
            if selected:
                y = self.counts_matrix[self.range_start_idx:
                                       self.range_end_idx + 1, selected].reshape(-1)
                positive = y[y > 0]
                if positive.size > 0:
                    ymin = max(1.0, float(np.min(positive)) * 0.8)
                    ymax = max(float(np.max(positive)) * 1.2, ymin * 2.0)
                    self.hist_ax.set_ylim(ymin, ymax)
                else:
                    self.hist_ax.set_ylim(1.0, 10.0)
        else:
            self.hist_ax.set_yscale("linear")

        self.hist_ax.set_title(
            "Pixel Histograms (toggle pixels by left-clicking map)")
        self.hist_ax.set_xlabel("Time (ns)")
        self.hist_ax.set_ylabel("Counts per bin")
        self.hist_ax.grid(True, alpha=0.25)
        self.hist_canvas.draw_idle()

    def _on_export_hist_traces(self) -> None:
        data = self._hist_trace_data()
        if data is None:
            QMessageBox.warning(self, "Export Line Traces",
                                "No selected histogram line traces to export.")
            return
        path = self._choose_export_path("spad23_histogram_line_traces.csv")
        if path is None:
            return
        t, selected, y = data
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["# type", "histogram_line_traces"])
                writer.writerow(
                    ["# source_file", self.edit_path.text().strip()])
                writer.writerow(["# range_start_idx", self.range_start_idx])
                writer.writerow(["# range_end_idx", self.range_end_idx])
                writer.writerow(["# current_idx", self.current_idx])
                writer.writerow(
                    ["# current_time_ns", f"{float(self.time_ns[self.current_idx]):.12g}" if self.time_ns.size else ""])
                writer.writerow(
                    ["# log_y_display", bool(self.chk_hist_log.isChecked())])
                writer.writerow([])
                writer.writerow(
                    ["time_ns"] + [f"pixel_{pix}" for pix in selected])
                for row_idx, t_ns in enumerate(t):
                    writer.writerow(
                        [f"{float(t_ns):.12g}"] + [f"{float(v):.12g}" for v in y[row_idx, :]])
            self.lbl_hist_state.setText(
                f"Exported histogram line traces: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Line Traces",
                                 f"Failed to write CSV:\n{exc}")

    def _update_sigma_plot(self) -> None:
        self.ax_sigma.clear()
        self.ax_sigma.set_title(r"Sigma_eq^2 vs Time")
        self.ax_sigma.set_xlabel("Time (ns)")
        self.ax_sigma.set_ylabel(r"Sigma_eq^2 (um^2)")
        self.ax_sigma.grid(True, alpha=0.25)

        data = self._sigma_trace_data()
        if data is None:
            self.ax_sigma.text(0.5, 0.5, "No fit results yet. Click Fit All.",
                               ha="center", va="center", transform=self.ax_sigma.transAxes)
            self.fit_canvas.draw_idle()
            return
        _method, t, sigma, _valid, _results = data
        sigma2 = sigma**2
        self.ax_sigma.plot(t, sigma2, "-o", markersize=2.5,
                           linewidth=1.1, color="#1d4ed8")
        self.ax_sigma.axvline(float(
            self.time_ns[self.current_idx]), color="#111827", linestyle="--", linewidth=0.95, alpha=0.8)
        self.fit_canvas.draw_idle()

    def _on_export_sigma_trace(self) -> None:
        data = self._sigma_trace_data()
        if data is None:
            QMessageBox.warning(self, "Export Sigma^2",
                                "No sigma^2 trace to export. Run Fit All first.")
            return
        path = self._choose_export_path("spad23_sigma_eq2_vs_time.csv")
        if path is None:
            return
        method, t, sigma, valid, results = data
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["# type", "sigma_eq2_vs_time"])
                writer.writerow(
                    ["# source_file", self.edit_path.text().strip()])
                writer.writerow(["# method_index", method + 1])
                writer.writerow(
                    ["# method", self.cmb_fit_method.currentText()])
                writer.writerow(["# range_start_idx", self.range_start_idx])
                writer.writerow(["# range_end_idx", self.range_end_idx])
                writer.writerow(
                    ["# reference_idx", "" if self.reference_idx is None else int(self.reference_idx)])
                writer.writerow(["# current_idx", self.current_idx])
                writer.writerow([])
                if method == 0:
                    writer.writerow(["bin_idx", "time_ns", "sigma_eq_um", "sigma_eq2_um2",
                                    "amplitude", "x0_um", "y0_um", "sx_um", "sy_um", "offset"])
                    for idx, t_ns, s in zip(valid, t, sigma):
                        fit = results[idx]
                        popt = fit.popt  # type: ignore[attr-defined]
                        writer.writerow(
                            [
                                idx,
                                f"{float(t_ns):.12g}",
                                f"{float(s):.12g}",
                                f"{float(s * s):.12g}",
                                f"{float(popt[0]):.12g}",
                                f"{float(popt[1]):.12g}",
                                f"{float(popt[2]):.12g}",
                                f"{float(popt[3]):.12g}",
                                f"{float(popt[4]):.12g}",
                                f"{float(popt[5]):.12g}",
                            ]
                        )
                else:
                    writer.writerow(["bin_idx", "time_ns", "sigma_eq_um",
                                    "sigma_eq2_um2", "amplitude", "sigma_um", "offset"])
                    for idx, t_ns, s in zip(valid, t, sigma):
                        fit = results[idx]
                        popt = fit.popt  # type: ignore[attr-defined]
                        writer.writerow(
                            [
                                idx,
                                f"{float(t_ns):.12g}",
                                f"{float(s):.12g}",
                                f"{float(s * s):.12g}",
                                f"{float(popt[0]):.12g}",
                                f"{float(popt[1]):.12g}",
                                f"{float(popt[2]):.12g}",
                            ]
                        )
            self.lbl_fit_state.setText(f"Exported sigma^2 trace: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Sigma^2",
                                 f"Failed to write CSV:\n{exc}")

    def _update_fitting_plots(self) -> None:
        self.ax_fit_map.clear()
        self.ax_fit_radial.clear()
        if self.counts_matrix.shape[0] == 0:
            self.fit_canvas.draw_idle()
            return

        method = self.cmb_fit_method.currentIndex()
        idx = int(self.current_idx)
        frame = self.counts_matrix[idx, :].astype(np.float64)
        if method == 0:
            fit = self.fit_results_m1.get(idx)
            if fit is None:
                self.ax_fit_map.text(0.5, 0.5, "No fit at selected time.\nRun Fit All first.",
                                     ha="center", va="center", transform=self.ax_fit_map.transAxes)
                self.ax_fit_map.set_title("2D Gaussian fit")
                self.ax_fit_map.set_xlabel("x (um)")
                self.ax_fit_map.set_ylabel("y (um)")
                self.ax_fit_radial.text(0.5, 0.5, "No fit at selected time.",
                                        ha="center", va="center", transform=self.ax_fit_radial.transAxes)
                self.ax_fit_radial.set_title("Gaussian radial fit")
                self.ax_fit_radial.set_xlabel("Distance (um)")
                self.ax_fit_radial.set_ylabel("Counts")
                self.ax_fit_radial.grid(True, alpha=0.25)
                self.fit_canvas.draw_idle()
                return

            popt = fit.popt
            fit_inputs = self._fit_inputs(frame)
            if fit_inputs is None:
                self.ax_fit_map.text(
                    0.5,
                    0.5,
                    "Not enough enabled pixels for fitting.",
                    ha="center",
                    va="center",
                    transform=self.ax_fit_map.transAxes,
                )
                self.fit_canvas.draw_idle()
                return
            x_used, y_used, z_used, _ = fit_inputs
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
            self.ax_fit_map.contourf(
                gxx, gyy, z_fit_grid, levels=levels, cmap="viridis")
            self.ax_fit_map.scatter(
                self.fit_x_um,
                self.fit_y_um,
                c=frame,
                cmap="viridis",
                edgecolors="k",
                s=45,
                vmin=zmin,
                vmax=zmax,
            )
            if self.excluded_pixels:
                ex_idx = np.array(sorted(self.excluded_pixels), dtype=int)
                self.ax_fit_map.scatter(
                    self.fit_x_um[ex_idx], self.fit_y_um[ex_idx], s=110, marker="x", c="#ef4444", linewidths=1.8)
            self.ax_fit_map.plot(float(popt[1]), float(
                popt[2]), "wx", markersize=8, markeredgewidth=2)
            self.ax_fit_map.set_title("2D Gaussian fit", y=1.12)
            self.ax_fit_map.text(
                0.5,
                1.03,
                f"x0={popt[1]:.1f} um, y0={popt[2]:.1f} um, sigma_eq={fit.sigma_eq:.1f} um",
                transform=self.ax_fit_map.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#111827",
                clip_on=False,
            )
            self.ax_fit_map.set_xlabel("x (um)")
            self.ax_fit_map.set_ylabel("y (um)")
            self.ax_fit_map.set_aspect("equal")
            self.ax_fit_map.set_xlim(xmin, xmax)
            self.ax_fit_map.set_ylim(ymin, ymax)
            self.ax_fit_map.invert_yaxis()

            r = np.sqrt((x_used - popt[1]) ** 2 + (y_used - popt[2]) ** 2)
            idx_sort = np.argsort(r)
            r_sorted = r[idx_sort]
            z_sorted = z_used[idx_sort]
            r_sym = np.concatenate([-r_sorted[::-1], r_sorted])
            z_sym = np.concatenate([z_sorted[::-1], z_sorted])
            rmax = float(np.max(r))
            r_line = np.linspace(-rmax, rmax, 401)
            sigma_safe = max(float(fit.sigma_eq), 1e-6)
            fit_line = popt[0] * \
                np.exp(-(r_line**2) / (2.0 * sigma_safe**2)) + popt[5]

            self.ax_fit_radial.scatter(
                r_sym, z_sym, s=20, alpha=0.75, color="#2563eb")
            self.ax_fit_radial.plot(
                r_line, fit_line, "-", color="#111827", linewidth=1.7)
            self.ax_fit_radial.set_title(
                f"Gaussian radial fit (sigma={fit.sigma_eq:.1f} um)")
            self.ax_fit_radial.set_xlabel("Distance from fitted center (um)")
            self.ax_fit_radial.set_ylabel("Counts")
            self.ax_fit_radial.grid(True, alpha=0.25)
            self.fit_canvas.draw_idle()
            return

        # Method 2 display: reference-frame 2D fit map + selected-time 1D radial fit.
        if self.reference_idx is None or self.m2_reference_fit is None or self.m2_reference_center is None:
            self.ax_fit_map.text(0.5, 0.5, "No method-2 fit results yet.\nSet reference and click Fit All.",
                                 ha="center", va="center", transform=self.ax_fit_map.transAxes)
            self.ax_fit_map.set_title("Reference 2D fit (fixed center)")
            self.ax_fit_map.set_xlabel("x (um)")
            self.ax_fit_map.set_ylabel("y (um)")
            self.ax_fit_radial.text(0.5, 0.5, "No radial 1D fit at selected time.",
                                    ha="center", va="center", transform=self.ax_fit_radial.transAxes)
            self.ax_fit_radial.set_title("Radial 1D Gaussian fit")
            self.ax_fit_radial.set_xlabel("Distance from fixed center (um)")
            self.ax_fit_radial.set_ylabel("Counts")
            self.ax_fit_radial.grid(True, alpha=0.25)
            self.fit_canvas.draw_idle()
            return

        ref_idx = int(self.reference_idx)
        ref_frame = self.counts_matrix[ref_idx, :].astype(np.float64)
        ref_fit = self.m2_reference_fit
        x0_ref, y0_ref = self.m2_reference_center

        xpad = PITCH_X_UM * 0.35
        ypad = PITCH_Y_UM * 0.35
        xmin = float(np.min(self.fit_x_um) - xpad)
        xmax = float(np.max(self.fit_x_um) + xpad)
        ymin = float(np.min(self.fit_y_um) - ypad)
        ymax = float(np.max(self.fit_y_um) + ypad)
        gx = np.linspace(xmin, xmax, 220)
        gy = np.linspace(ymin, ymax, 220)
        gxx, gyy = np.meshgrid(gx, gy)
        popt_ref = ref_fit.popt
        z_fit_grid = (
            popt_ref[0]
            * np.exp(-(((gxx - popt_ref[1]) ** 2) / (2.0 * popt_ref[3] ** 2) + ((gyy - popt_ref[2]) ** 2) / (2.0 * popt_ref[4] ** 2)))
            + popt_ref[5]
        )
        zmin = float(np.min(z_fit_grid))
        zmax = float(np.max(z_fit_grid))
        if zmax <= zmin:
            zmax = zmin + 1e-9
        levels = np.linspace(zmin, zmax, 80)
        self.ax_fit_map.contourf(gxx, gyy, z_fit_grid,
                                 levels=levels, cmap="viridis")
        self.ax_fit_map.scatter(
            self.fit_x_um,
            self.fit_y_um,
            c=ref_frame,
            cmap="viridis",
            edgecolors="k",
            s=45,
            vmin=zmin,
            vmax=zmax,
        )
        if self.excluded_pixels:
            ex_idx = np.array(sorted(self.excluded_pixels), dtype=int)
            self.ax_fit_map.scatter(
                self.fit_x_um[ex_idx], self.fit_y_um[ex_idx], s=110, marker="x", c="#ef4444", linewidths=1.8)
        self.ax_fit_map.plot(float(x0_ref), float(
            y0_ref), "wx", markersize=8, markeredgewidth=2)
        self.ax_fit_map.set_title(
            "Reference 2D Gaussian fit (fixed center)", y=1.12)
        self.ax_fit_map.text(
            0.5,
            1.03,
            f"ref bin={ref_idx}, t={self.time_ns[ref_idx]:.4f} ns, x0={x0_ref:.1f} um, y0={y0_ref:.1f} um",
            transform=self.ax_fit_map.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#111827",
            clip_on=False,
        )
        self.ax_fit_map.set_xlabel("x (um)")
        self.ax_fit_map.set_ylabel("y (um)")
        self.ax_fit_map.set_aspect("equal")
        self.ax_fit_map.set_xlim(xmin, xmax)
        self.ax_fit_map.set_ylim(ymin, ymax)
        self.ax_fit_map.invert_yaxis()

        fit1d = self.fit_results_m2.get(idx)
        if fit1d is None:
            self.ax_fit_radial.text(0.5, 0.5, "No method-2 fit at selected time.",
                                    ha="center", va="center", transform=self.ax_fit_radial.transAxes)
            self.ax_fit_radial.set_title("Radial 1D Gaussian fit")
            self.ax_fit_radial.set_xlabel("Distance from fixed center (um)")
            self.ax_fit_radial.set_ylabel("Counts")
            self.ax_fit_radial.grid(True, alpha=0.25)
            self.fit_canvas.draw_idle()
            return

        fit_inputs = self._fit_inputs(frame)
        if fit_inputs is None:
            self.ax_fit_radial.text(0.5, 0.5, "Not enough enabled pixels for method-2 radial fit.",
                                    ha="center", va="center", transform=self.ax_fit_radial.transAxes)
            self.ax_fit_radial.set_title("Radial 1D Gaussian fit")
            self.ax_fit_radial.set_xlabel("Distance from fixed center (um)")
            self.ax_fit_radial.set_ylabel("Counts")
            self.ax_fit_radial.grid(True, alpha=0.25)
            self.fit_canvas.draw_idle()
            return
        x_used, y_used, z_used, _ = fit_inputs
        r = np.sqrt((x_used - x0_ref) ** 2 + (y_used - y0_ref) ** 2)
        idx_sort = np.argsort(r)
        r_sorted = r[idx_sort]
        z_sorted = z_used[idx_sort]

        A1, sigma1, offset1 = [float(v) for v in fit1d.popt]
        rmax = float(np.max(r_sorted))
        r_line = np.linspace(0.0, rmax, 401)
        fit_line = A1 * np.exp(-(r_line**2) /
                               (2.0 * max(abs(sigma1), 1e-6) ** 2)) + offset1

        self.ax_fit_radial.scatter(
            r_sorted, z_sorted, s=22, alpha=0.8, color="#2563eb")
        self.ax_fit_radial.plot(r_line, fit_line, "-",
                                color="#111827", linewidth=1.7)
        self.ax_fit_radial.set_title(
            f"Radial 1D Gaussian @ t={self.time_ns[idx]:.4f} ns (sigma={fit1d.sigma_eq:.2f} um)")
        self.ax_fit_radial.set_xlabel("Distance from fixed center (um)")
        self.ax_fit_radial.set_ylabel("Counts")
        self.ax_fit_radial.grid(True, alpha=0.25)
        self.fit_canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPAD23 TRPL CSV offline viewer")
    parser.add_argument("--file", default="",
                        help="Optional path to TRPL CSV to load at startup")
    args = parser.parse_args()

    app = QApplication([])
    win = Spad23TrplViewer()
    win.resize(1800, 980)
    win.show()
    if args.file:
        win.edit_path.setText(args.file)
        win._load_csv(args.file)
    app.exec()


if __name__ == "__main__":
    main()

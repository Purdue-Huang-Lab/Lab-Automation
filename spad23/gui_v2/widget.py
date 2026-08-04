from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .config import BTN_W, FIT_ROWS, NUM_PIXELS, PITCH_X_UM, PITCH_Y_UM
from .plots import FittingWidget, HistogramWidget, SpadMapWidget

try:
    from scipy.optimize import curve_fit
    from scipy.special import erfc as _erfc
    _HAS_SCIPY = True
except Exception:
    curve_fit = None  # type: ignore
    _erfc = None      # type: ignore
    _HAS_SCIPY = False


# ── EMG model functions ────────────────────────────────────────────────────────
# Single EMG kernel: (1/2)*exp(σ²/(2τ²) - dt/τ) * erfc(σ/(√2 τ) - dt/(√2 σ))
# Numerically safe for all t: clamp the exponent and zero-out where erfc→0.

import math as _math
_SQRT2 = _math.sqrt(2.0)


def _emg_k(t: np.ndarray, t0: float, sigma: float, tau: float) -> np.ndarray:
    dt = t - t0
    erfc_arg = sigma / (_SQRT2 * tau) - dt / (_SQRT2 * sigma)
    exp_arg = np.clip(sigma ** 2 / (2.0 * tau ** 2) - dt / tau, -700.0, 700.0)
    result = 0.5 * np.exp(exp_arg) * _erfc(erfc_arg)
    result[erfc_arg > 26.0] = 0.0   # erfc(26)≈0, avoid exp(+∞)*0 = NaN
    return result


def _emg_1dec(t, C, A1, tau1, t0, sigma):
    return C + A1 * _emg_k(t, t0, sigma, tau1)


def _emg_2dec(t, C, A1, tau1, A2, tau2, t0, sigma):
    return C + A1 * _emg_k(t, t0, sigma, tau1) + A2 * _emg_k(t, t0, sigma, tau2)


def _emg_rise_1dec(t, C, Ar, taur, A1, tau1, t0, sigma):
    # rise modelled as negative-amplitude short-tau EMG component
    return C - Ar * _emg_k(t, t0, sigma, taur) + A1 * _emg_k(t, t0, sigma, tau1)


def _emg_rise_2dec(t, C, Ar, taur, A1, tau1, A2, tau2, t0, sigma):
    return (C - Ar * _emg_k(t, t0, sigma, taur)
            + A1 * _emg_k(t, t0, sigma, tau1)
            + A2 * _emg_k(t, t0, sigma, tau2))


def _emg_3dec(t, C, A1, tau1, A2, tau2, A3, tau3, t0, sigma):
    return (C + A1 * _emg_k(t, t0, sigma, tau1)
            + A2 * _emg_k(t, t0, sigma, tau2)
            + A3 * _emg_k(t, t0, sigma, tau3))


def _emg_rise_3dec(t, C, Ar, taur, A1, tau1, A2, tau2, A3, tau3, t0, sigma):
    return (C - Ar * _emg_k(t, t0, sigma, taur)
            + A1 * _emg_k(t, t0, sigma, tau1)
            + A2 * _emg_k(t, t0, sigma, tau2)
            + A3 * _emg_k(t, t0, sigma, tau3))


def _build_joint_model(n_pixels: int, n_bins: int,
                       use_rise: bool, use_2nd: bool, use_3rd: bool,
                       sigma_fixed: float):
    """
    Build a joint model for simultaneous fitting of N pixels with sigma fixed.

    sigma_fixed is captured as a constant — not a free parameter.

    Parameter layout fed to curve_fit (no sigma):
      [C_0, (Ar_0, taur_0,) A1_0, tau1_0, (A2_0, tau2_0,) (A3_0, tau3_0,) t0_0,
       C_1, ...,                                                             t0_{N-1}]

    Returns (model_fn, n_per_pixel) where n_per_pixel is the per-pixel block size.
    """
    n_per_pixel = 1  # C only  (t0 fixed at 0 — not a free parameter)
    if use_rise:
        n_per_pixel += 2
    n_per_pixel += 2  # A1, tau1
    if use_2nd:
        n_per_pixel += 2
    if use_3rd:
        n_per_pixel += 2

    def model_fn(t_all, *flat):
        out = np.empty(n_pixels * n_bins)
        for i in range(n_pixels):
            t_i = t_all[i * n_bins:(i + 1) * n_bins]
            pp = flat[i * n_per_pixel:(i + 1) * n_per_pixel]
            out[i * n_bins:(i + 1) * n_bins] = _eval_emg_pixel_fixed_t0(
                t_i, sigma_fixed, pp, use_rise, use_2nd, use_3rd)
        return out

    return model_fn, n_per_pixel


def _eval_emg_pixel(t, sigma: float, pp, use_rise: bool, use_2nd: bool, use_3rd: bool):
    """Per-pixel EMG evaluation with t0 as pp[-1]  (used by the alignment fit)."""
    C = float(pp[0])
    t0 = float(pp[-1])
    y = np.full(len(t), C)
    j = 1
    if use_rise:
        y -= float(pp[j]) * _emg_k(t, t0, sigma, float(pp[j + 1]))
        j += 2
    y += float(pp[j]) * _emg_k(t, t0, sigma, float(pp[j + 1]))
    j += 2
    if use_2nd:
        y += float(pp[j]) * _emg_k(t, t0, sigma, float(pp[j + 1]))
        j += 2
    if use_3rd:
        y += float(pp[j]) * _emg_k(t, t0, sigma, float(pp[j + 1]))
    return y


def _eval_emg_pixel_fixed_t0(t, sigma: float, pp,
                              use_rise: bool, use_2nd: bool, use_3rd: bool):
    """Per-pixel EMG evaluation with t0 fixed at 0 (used by the histogram fit).

    t0 is NOT in pp — the layout is [C, (Ar, taur,) A1, tau1, (A2, tau2,) (A3, tau3,)].
    t=0 in the aligned data already encodes t0 from the chosen alignment method.
    """
    C = float(pp[0])
    y = np.full(len(t), C)
    j = 1
    if use_rise:
        y -= float(pp[j]) * _emg_k(t, 0.0, sigma, float(pp[j + 1]))
        j += 2
    y += float(pp[j]) * _emg_k(t, 0.0, sigma, float(pp[j + 1]))
    j += 2
    if use_2nd:
        y += float(pp[j]) * _emg_k(t, 0.0, sigma, float(pp[j + 1]))
        j += 2
    if use_3rd:
        y += float(pp[j]) * _emg_k(t, 0.0, sigma, float(pp[j + 1]))
    return y


@dataclass
class FitResult:
    popt: np.ndarray
    z_fit: np.ndarray
    sigma_eq: float


@dataclass
class RadialFitResult:
    popt: np.ndarray
    sigma_eq: float


def _build_fit_coords() -> Tuple[np.ndarray, np.ndarray]:
    max_cols = max(len(r) for r in FIT_ROWS)
    coords: Dict[int, Tuple[float, float]] = {}
    for row_idx, row in enumerate(FIT_ROWS):
        x0 = (PITCH_X_UM / 2.0) if len(row) < max_cols else 0.0
        for col_idx, pixel in enumerate(row):
            coords[pixel] = (x0 + col_idx * PITCH_X_UM, row_idx * PITCH_Y_UM)
    x = np.array([coords[i][0] for i in range(NUM_PIXELS)], dtype=np.float64)
    y = np.array([coords[i][1] for i in range(NUM_PIXELS)], dtype=np.float64)
    return x, y


def _fit_gaussian_2d(x: np.ndarray, y: np.ndarray,
                     z: np.ndarray) -> Optional[FitResult]:
    if not _HAS_SCIPY or np.allclose(z, z[0]):
        return None

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0) ** 2) / (2 * sx ** 2) +
                            ((yg - y0) ** 2) / (2 * sy ** 2))) + offset

    p0 = [max(float(np.max(z) - np.min(z)), 1e-6),
          float(np.mean(x)), float(np.mean(y)), 20.0, 20.0, float(np.min(z))]
    bounds = (
        [0, float(np.min(x) - PITCH_X_UM), float(np.min(y) - PITCH_Y_UM), 1e-3, 1e-3, -np.inf],
        [np.inf, float(np.max(x) + PITCH_X_UM), float(np.max(y) + PITCH_Y_UM), np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)  # type: ignore
    except Exception:
        return None
    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2))
    return FitResult(popt=np.asarray(popt, dtype=np.float64),
                     z_fit=np.asarray(z_fit, dtype=np.float64),
                     sigma_eq=sigma_eq)


def _fit_gaussian_2d_constrained_next(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                      prev_popt: np.ndarray,
                                      tightness: float = 1.0) -> Optional[FitResult]:
    if not _HAS_SCIPY or np.allclose(z, z[0]):
        return None
    A_p, x0_p, y0_p, sx_p, sy_p, off_p = [float(v) for v in prev_popt]
    sx_p = max(sx_p, 1e-3)
    sy_p = max(sy_p, 1e-3)
    dA = max(abs(A_p) * 0.18 * tightness, 1e-3)
    dx = max(PITCH_X_UM * 0.18 * tightness, 0.8)
    dy = max(PITCH_Y_UM * 0.18 * tightness, 0.8)
    dsx = max(abs(sx_p) * 0.18 * tightness, 0.5)
    dsy = max(abs(sy_p) * 0.18 * tightness, 0.5)
    doff = max(abs(off_p) * 0.25 * tightness, 0.5)
    p0 = [A_p, x0_p, y0_p, sx_p, sy_p, off_p]
    lower = [max(0.0, A_p - dA), x0_p - dx, y0_p - dy,
             max(1e-3, sx_p - dsx), max(1e-3, sy_p - dsy), off_p - doff]
    upper = [A_p + dA, x0_p + dx, y0_p + dy,
             sx_p + dsx, sy_p + dsy, off_p + doff]

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0) ** 2) / (2 * sx ** 2) +
                            ((yg - y0) ** 2) / (2 * sy ** 2))) + offset

    try:
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0,  # type: ignore
                            bounds=(lower, upper), maxfev=20000)
    except Exception:
        return None
    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2))
    return FitResult(popt=np.asarray(popt, dtype=np.float64),
                     z_fit=np.asarray(z_fit, dtype=np.float64),
                     sigma_eq=sigma_eq)


def _fit_radial_gaussian_1d(r: np.ndarray, z: np.ndarray,
                             p0: Optional[np.ndarray] = None,
                             sigma_guess: Optional[float] = None) -> Optional[RadialFitResult]:
    if not _HAS_SCIPY or r.size < 3 or z.size < 3 or np.allclose(z, z[0]):
        return None
    r = r.astype(np.float64)
    z = z.astype(np.float64)

    def gauss1d(rg, A, sigma, offset):
        return A * np.exp(-(rg ** 2) / (2 * sigma ** 2)) + offset

    if p0 is not None and p0.size == 3:
        p0_use = [float(p0[0]), max(float(p0[1]), 1e-3), float(p0[2])]
    else:
        A0 = max(float(np.max(z) - np.min(z)), 1e-6)
        s0 = max(float(sigma_guess) if sigma_guess is not None else float(np.std(r)), 1e-3)
        p0_use = [A0, s0, float(np.min(z))]
    try:
        popt, _ = curve_fit(gauss1d, r, z, p0=p0_use,  # type: ignore
                            bounds=([0, 1e-3, -np.inf], [np.inf, np.inf, np.inf]),
                            maxfev=20000)
    except Exception:
        return None
    return RadialFitResult(popt=np.asarray(popt, dtype=np.float64),
                           sigma_eq=float(abs(popt[1])))


class Spad23TrplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time_ns: np.ndarray = np.array([], dtype=np.float64)
        self.counts_matrix: np.ndarray = np.zeros((0, NUM_PIXELS), dtype=np.int64)

        # alignment / t=0 detection
        self.smooth_window: int = 5
        self.noise_level: float = 0.0
        self.align_t0: bool = True
        self.magnification: float = 25.0
        self.t0_idx: np.ndarray = np.zeros(NUM_PIXELS, dtype=int)
        self.t0_ns: np.ndarray = np.zeros(NUM_PIXELS, dtype=np.float64)
        self.smoothed_matrix: np.ndarray = np.zeros((0, NUM_PIXELS), dtype=np.float64)
        self.aligned_counts: np.ndarray = np.zeros((0, NUM_PIXELS), dtype=np.float64)
        self.aligned_smoothed_counts: np.ndarray = np.zeros((0, NUM_PIXELS), dtype=np.float64)
        self.aligned_time_ns: np.ndarray = np.array([], dtype=np.float64)
        self.n_aligned_bins: int = 0
        self.end_time_ns: float = 0.0
        self.current_aligned_idx: int = 0

        self.align_method: int = 0   # 0 = smooth peak, 1 = EMG fit t₀

        # baseline (pre-peak median, per pixel)
        self.baseline_counts: np.ndarray = np.zeros(NUM_PIXELS, dtype=np.float64)
        self.baseline_window_pct: float = 15.0

        # x-t colormap cache — populated by _on_fit_all, consumed by _update_fitting
        self._xt_t: Optional[np.ndarray] = None          # time axis (ns), shape (n_t,)
        self._xt_r: Optional[np.ndarray] = None          # uniform r grid in sample µm
        self._xt_z: Optional[np.ndarray] = None          # intensity (n_t, n_r_grid)
        self._xt_r_raw_sample: Optional[np.ndarray] = None  # actual r values (sample µm)
        self._xt_enabled_idx: Optional[np.ndarray] = None   # enabled pixel flat indices
        self._xt_order: Optional[np.ndarray] = None         # sort order by r

        self.selected_pixels: Set[int] = {11}
        self.excluded_pixels: Set[int] = set()

        self.fit_results_m1: Dict[int, FitResult] = {}
        self.fit_results_m2: Dict[int, RadialFitResult] = {}
        self.m2_reference_fit: Optional[FitResult] = None
        self.m2_reference_center: Optional[Tuple[float, float]] = None
        self.m2_reference_mask: Optional[np.ndarray] = None

        self.fit_x_um, self.fit_y_um = _build_fit_coords()
        self.m2_r_max_detector: float = 0.0   # max radial distance (detector µm) from method-2 fit
        self.m2_r_fixed_detector_sorted: Optional[np.ndarray] = None
        self.m2_reference_order: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── File bar ──────────────────────────────────────────────────────────
        file_group = QtWidgets.QGroupBox("File")
        fg = QtWidgets.QHBoxLayout(file_group)
        fg.setContentsMargins(8, 6, 8, 6)
        fg.setSpacing(6)
        self.edit_path = QtWidgets.QLineEdit()
        self.edit_path.setPlaceholderText(
            "Select a TRPL CSV file: columns time_ns, pixel_0 … pixel_22")
        self.btn_browse = QtWidgets.QPushButton("Browse")
        self.btn_browse.setFixedWidth(BTN_W)
        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_load.setFixedWidth(BTN_W)
        fg.addWidget(QtWidgets.QLabel("CSV file"))
        fg.addWidget(self.edit_path, 1)
        fg.addWidget(self.btn_browse)
        fg.addWidget(self.btn_load)
        root.addWidget(file_group)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl_group = QtWidgets.QGroupBox("Controls")
        cg = QtWidgets.QGridLayout(ctrl_group)
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(6)
        cg.setContentsMargins(8, 8, 8, 8)

        self.spin_smooth = QtWidgets.QSpinBox()
        self.spin_smooth.setRange(1, 500)
        self.spin_smooth.setValue(5)
        self.spin_smooth.setSuffix(" bins")
        self.spin_smooth.setFixedWidth(90)
        self.spin_smooth.setToolTip(
            "Moving-average window (bins) used only for finding t=0 per pixel")

        self.spin_noise = QtWidgets.QDoubleSpinBox()
        self.spin_noise.setRange(0.0, 1e9)
        self.spin_noise.setDecimals(1)
        self.spin_noise.setValue(0.0)
        self.spin_noise.setFixedWidth(90)
        self.spin_noise.setToolTip(
            "Pixels whose smoothed peak is ≤ this threshold share pixel 11's t=0")

        self.spin_end_ns = QtWidgets.QDoubleSpinBox()
        self.spin_end_ns.setRange(0.0, 1e6)
        self.spin_end_ns.setDecimals(4)
        self.spin_end_ns.setSuffix(" ns")
        self.spin_end_ns.setFixedWidth(120)
        self.spin_end_ns.setEnabled(False)
        self.spin_end_ns.setToolTip("Fitting end time relative to t=0 (raw data used)")

        self.spin_magnification = QtWidgets.QDoubleSpinBox()
        self.spin_magnification.setRange(0.01, 10000.0)
        self.spin_magnification.setDecimals(2)
        self.spin_magnification.setValue(25.0)
        self.spin_magnification.setFixedWidth(80)
        self.spin_magnification.setToolTip(
            "Detector magnification M: sample length = detector length / M")

        self.chk_align_t0 = QtWidgets.QCheckBox("Align t=0")
        self.chk_align_t0.setChecked(True)
        self.chk_align_t0.setToolTip(
            "Per-pixel t=0 alignment. When off, all pixels share pixel 11's t=0.")

        self.cmb_t0_method = QtWidgets.QComboBox()
        self.cmb_t0_method.addItem("Smooth peak")
        self.cmb_t0_method.addItem("EMG fit t₀")
        self.cmb_t0_method.setToolTip(
            "Smooth peak: t=0 at the smoothed-maximum bin.\n"
            "EMG fit t₀: fit a single-decay EMG per pixel; use the fitted t₀ "
            "parameter as t=0. Falls back to pixel 11's t₀ on fit failure.")

        self.chk_hist_log = QtWidgets.QCheckBox("Log y")
        self.chk_normalize = QtWidgets.QCheckBox("Normalize")
        self.chk_normalize.setToolTip("Divide each trace by its smoothed peak value")
        self.btn_export_hist = QtWidgets.QPushButton("Export Traces")
        self.btn_export_hist.setFixedWidth(BTN_W)

        self.spin_baseline_pct = QtWidgets.QSpinBox()
        self.spin_baseline_pct.setRange(1, 90)
        self.spin_baseline_pct.setValue(15)
        self.spin_baseline_pct.setSuffix("%")
        self.spin_baseline_pct.setFixedWidth(70)
        self.spin_baseline_pct.setToolTip(
            "Window width (% of trace length) of pre-peak region used for baseline estimate")

        self.chk_subtract_baseline = QtWidgets.QCheckBox("Subtract baseline")
        self.chk_subtract_baseline.setToolTip(
            "Subtract the pre-peak median baseline from each trace before display")
        self.chk_show_baseline = QtWidgets.QCheckBox("Show baseline")
        self.chk_show_baseline.setToolTip(
            "Draw a dashed horizontal line at the estimated baseline level for each trace")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.lbl_time = QtWidgets.QLabel("t = 0.0000 ns (rel. t₀)")

        cg.addWidget(self.chk_align_t0, 0, 0)
        cg.addWidget(self.cmb_t0_method, 0, 1)
        cg.addWidget(QtWidgets.QLabel("Smooth:"), 0, 2)
        cg.addWidget(self.spin_smooth, 0, 3)
        cg.addWidget(QtWidgets.QLabel("Noise floor:"), 0, 4)
        cg.addWidget(self.spin_noise, 0, 5)
        cg.addWidget(QtWidgets.QLabel("Fit end (rel. t₀):"), 0, 6)
        cg.addWidget(self.spin_end_ns, 0, 7)
        cg.addWidget(QtWidgets.QLabel("M:"), 0, 8)
        cg.addWidget(self.spin_magnification, 0, 9)
        cg.addWidget(self.chk_hist_log, 0, 10)
        cg.addWidget(self.chk_normalize, 0, 11)
        cg.addWidget(self.btn_export_hist, 0, 12)
        cg.addWidget(QtWidgets.QLabel("Aligned bin:"), 1, 0)
        cg.addWidget(self.slider, 1, 1, 1, 11)
        cg.addWidget(self.lbl_time, 1, 12)

        cg.addWidget(QtWidgets.QLabel("Baseline win:"), 2, 0)
        cg.addWidget(self.spin_baseline_pct, 2, 1)
        cg.addWidget(self.chk_subtract_baseline, 2, 2)
        cg.addWidget(self.chk_show_baseline, 2, 3)
        root.addWidget(ctrl_group)

        # ── Main splitter ──────────────────────────────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        self._spad_map = SpadMapWidget()
        self._spad_map.setMinimumWidth(420)
        splitter.addWidget(self._spad_map)

        self._tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([720, 980])

        self._build_hist_tab()
        self._build_fitting_tab()

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = QtWidgets.QStatusBar()
        self._status.setSizeGripEnabled(False)
        root.addWidget(self._status)

        # ── Connections ───────────────────────────────────────────────────────
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_load.clicked.connect(self._on_load)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.chk_align_t0.toggled.connect(self._on_align_changed)
        self.cmb_t0_method.currentIndexChanged.connect(self._on_t0_method_changed)
        self.spin_smooth.valueChanged.connect(self._on_smooth_changed)
        self.spin_noise.valueChanged.connect(self._on_noise_changed)
        self.spin_magnification.valueChanged.connect(self._on_magnification_changed)
        self.chk_hist_log.toggled.connect(lambda _: self._update_histogram())
        self.chk_normalize.toggled.connect(lambda _: self._update_histogram())
        self.btn_export_hist.clicked.connect(self._on_export_hist)
        self.spin_baseline_pct.valueChanged.connect(self._on_baseline_pct_changed)
        self.chk_subtract_baseline.toggled.connect(lambda _: self._update_histogram())
        self.chk_show_baseline.toggled.connect(lambda _: self._update_histogram())
        self._spad_map.pixel_left_clicked.connect(self._on_pixel_left)
        self._spad_map.pixel_right_clicked.connect(self._on_pixel_right)
        self._tabs.currentChanged.connect(lambda _: self._refresh_all())

    def _build_hist_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._hist_widget = HistogramWidget()
        layout.addWidget(self._hist_widget, 1)

        fit_group = QtWidgets.QGroupBox("EMG Fit")
        fg = QtWidgets.QGridLayout(fit_group)
        fg.setHorizontalSpacing(8)
        fg.setVerticalSpacing(4)
        fg.setContentsMargins(8, 6, 8, 6)

        self.chk_fit_emg = QtWidgets.QCheckBox("Enable fit")
        self.chk_fit_emg.setToolTip(
            "Overlay Gaussian-convolved exponential (EMG) fit on each selected pixel")
        self.chk_fit_risetime = QtWidgets.QCheckBox("Rise time τᵣ")
        self.chk_fit_risetime.setEnabled(False)
        self.chk_fit_risetime.setToolTip(
            "Add a negative-amplitude EMG component to model exponential rise")
        self.chk_fit_2nd_decay = QtWidgets.QCheckBox("2nd decay τ₂")
        self.chk_fit_2nd_decay.setEnabled(False)
        self.chk_fit_2nd_decay.setToolTip("Add a second decay component")

        self.chk_fit_3rd_decay = QtWidgets.QCheckBox("3rd decay τ₃")
        self.chk_fit_3rd_decay.setEnabled(False)
        self.chk_fit_3rd_decay.setToolTip("Add a third decay component")

        self.spin_fit_sigma_ps = QtWidgets.QDoubleSpinBox()
        self.spin_fit_sigma_ps.setRange(0.1, 10000.0)
        self.spin_fit_sigma_ps.setDecimals(1)
        self.spin_fit_sigma_ps.setValue(10.0)
        self.spin_fit_sigma_ps.setSuffix(" ps")
        self.spin_fit_sigma_ps.setFixedWidth(90)
        self.spin_fit_sigma_ps.setEnabled(False)
        self.spin_fit_sigma_ps.setToolTip("Initial Gaussian IRF width σ₀")

        self.lbl_fit_results = QtWidgets.QLabel("")
        self.lbl_fit_results.setWordWrap(True)

        fg.addWidget(self.chk_fit_emg, 0, 0)
        fg.addWidget(self.chk_fit_risetime, 0, 1)
        fg.addWidget(self.chk_fit_2nd_decay, 0, 2)
        fg.addWidget(self.chk_fit_3rd_decay, 0, 3)
        fg.addWidget(QtWidgets.QLabel("σ₀:"), 0, 4)
        fg.addWidget(self.spin_fit_sigma_ps, 0, 5)
        fg.addWidget(self.lbl_fit_results, 1, 0, 1, 7)
        layout.addWidget(fit_group)

        self._tabs.addTab(tab, "Histogram")

        self.chk_fit_emg.toggled.connect(self._on_fit_emg_toggled)
        self.chk_fit_risetime.toggled.connect(self._on_sigma0_changed)
        self.chk_fit_2nd_decay.toggled.connect(self._on_sigma0_changed)
        self.chk_fit_3rd_decay.toggled.connect(self._on_sigma0_changed)
        self.spin_fit_sigma_ps.valueChanged.connect(self._on_sigma0_changed)

    def _on_fit_emg_toggled(self, checked: bool) -> None:
        self.chk_fit_risetime.setEnabled(checked)
        self.chk_fit_2nd_decay.setEnabled(checked)
        self.chk_fit_3rd_decay.setEnabled(checked)
        self.spin_fit_sigma_ps.setEnabled(checked)
        if not checked:
            self._hist_widget.clear_fits()
            self.lbl_fit_results.setText("")
        else:
            self._refresh_emg_fits()

    def _on_sigma0_changed(self, _=None) -> None:
        # If EMG-fit alignment is active, re-align; always refresh histogram fits.
        if (self.align_method == 1 and self.align_t0
                and self.counts_matrix.shape[0] > 0):
            self._clear_fit_results()
            self._compute_alignment()
            self.slider.setMaximum(max(0, self.n_aligned_bins - 1))
            idx = min(self.current_aligned_idx, self.n_aligned_bins - 1)
            self._set_current_aligned_idx(idx)
        else:
            self._refresh_emg_fits()

    def _build_fitting_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        self.cmb_fit_method = QtWidgets.QComboBox()
        self.cmb_fit_method.addItem("1: sequential 2D Gaussian fit")
        self.cmb_fit_method.addItem("2: fixed-center radial 1D Gaussian fit")
        self.btn_fit_all = QtWidgets.QPushButton("Fit All")
        self.btn_fit_all.setFixedWidth(BTN_W)
        self.spin_sigma_smooth = QtWidgets.QSpinBox()
        self.spin_sigma_smooth.setRange(1, 500)
        self.spin_sigma_smooth.setValue(1)
        self.spin_sigma_smooth.setSuffix(" pts")
        self.spin_sigma_smooth.setFixedWidth(80)
        self.spin_sigma_smooth.setToolTip(
            "Smoothing window for σ_eq² vs time (uniform filter, edge-padded to avoid boundary artifacts)")

        self.spin_asym_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_asym_threshold.setRange(1.0, 100.0)
        self.spin_asym_threshold.setDecimals(2)
        self.spin_asym_threshold.setValue(1.5)
        self.spin_asym_threshold.setSuffix("×")
        self.spin_asym_threshold.setFixedWidth(75)
        self.spin_asym_threshold.setToolTip(
            "Method 1 only: if max(σ_x, σ_y) / min(σ_x, σ_y) exceeds this, "
            "the point is shown gray on the σ² plot (fit unreliable due to asymmetry)")

        self.btn_export_sigma = QtWidgets.QPushButton("Export σ² CSV")
        self.btn_export_sigma.setFixedWidth(BTN_W)

        self.edit_extra_times = QtWidgets.QLineEdit()
        self.edit_extra_times.setPlaceholderText("Extra radial times (ns): 0.1, 0.5, 1.0")
        self.edit_extra_times.setToolTip(
            "Comma-separated times (ns). For each, the closest fitted time is shown "
            "in the radial plot (max 5 lines, colored in order).")

        self.chk_normalize_spatial = QtWidgets.QCheckBox("Norm. spatial")
        self.chk_normalize_spatial.setToolTip(
            "Normalize intensity to [0, 1] at each time slice in the space–time colormap")

        top.addWidget(QtWidgets.QLabel("Method"))
        top.addWidget(self.cmb_fit_method, 1)
        top.addWidget(self.btn_fit_all)
        top.addWidget(QtWidgets.QLabel("Smooth σ²:"))
        top.addWidget(self.spin_sigma_smooth)
        top.addWidget(QtWidgets.QLabel("Asym ≤:"))
        top.addWidget(self.spin_asym_threshold)
        top.addWidget(self.btn_export_sigma)
        top.addWidget(QtWidgets.QLabel("Extra t:"))
        top.addWidget(self.edit_extra_times, 1)
        top.addWidget(self.chk_normalize_spatial)
        layout.addLayout(top)

        self.lbl_fit_state = QtWidgets.QLabel("Load data and click Fit All.")
        layout.addWidget(self.lbl_fit_state)

        self._fitting_widget = FittingWidget()
        layout.addWidget(self._fitting_widget, 1)

        self._tabs.addTab(tab, "Fitting")

        self.btn_fit_all.clicked.connect(self._on_fit_all)
        self.btn_export_sigma.clicked.connect(self._on_export_sigma)
        self.cmb_fit_method.currentIndexChanged.connect(lambda _: self._refresh_all())
        self.spin_sigma_smooth.valueChanged.connect(lambda _: self._update_fitting())
        self.spin_asym_threshold.valueChanged.connect(lambda _: self._update_fitting())
        self.edit_extra_times.editingFinished.connect(self._update_fitting)
        self.chk_normalize_spatial.toggled.connect(self._update_spacetime_plot)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open TRPL CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.edit_path.setText(path)

    def _on_load(self) -> None:
        path = self.edit_path.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Load CSV",
                                          "Please choose a CSV file first.")
            return
        self._load_csv(path)

    def _load_csv(self, path: str) -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            QtWidgets.QMessageBox.critical(self, "Load CSV",
                                           f"File not found:\n{path}")
            return
        required = ["time_ns"] + [f"pixel_{i}" for i in range(NUM_PIXELS)]
        time_vals: List[float] = []
        pixel_vals: List[List[int]] = []
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("Missing CSV header.")
                missing = [n for n in required if n not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Missing columns: {missing}")
                for row in reader:
                    time_vals.append(float(row["time_ns"]))
                    pixel_vals.append(
                        [int(float(row[f"pixel_{i}"])) for i in range(NUM_PIXELS)])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load CSV",
                                           f"Could not parse file:\n{exc}")
            return
        if not time_vals:
            QtWidgets.QMessageBox.warning(self, "Load CSV", "CSV has no data rows.")
            return

        self.time_ns = np.asarray(time_vals, dtype=np.float64)
        self.counts_matrix = np.asarray(pixel_vals, dtype=np.int64)
        self.current_aligned_idx = 0
        self._clear_fit_results()
        self.selected_pixels = {11}
        self.excluded_pixels = set()

        self._compute_alignment()

        self.slider.setEnabled(True)
        self.spin_end_ns.setEnabled(True)
        self.slider.setMaximum(max(0, self.n_aligned_bins - 1))
        self.slider.setValue(0)
        self._set_current_aligned_idx(0)
        self._set_status(
            f"Loaded: {csv_path.name} | "
            f"{self.counts_matrix.shape[0]} bins | "
            f"{self.n_aligned_bins} aligned bins after t=0")

    # ── Alignment (t=0 detection) ─────────────────────────────────────────────

    def _compute_alignment(self) -> None:
        N, P = self.counts_matrix.shape
        if N == 0:
            return
        w = max(1, self.smooth_window)
        kernel = np.ones(w, dtype=np.float64) / w
        self.smoothed_matrix = np.apply_along_axis(
            lambda c: np.convolve(c.astype(np.float64), kernel, mode='same'),
            0, self.counts_matrix)

        # smooth-peak t0 is always computed — used as initial guess / fallback
        smooth_t0_idx = np.argmax(self.smoothed_matrix, axis=0).astype(int)

        if not self.align_t0:
            # All pixels share pixel 11's smooth-peak t=0
            self.t0_idx = np.full(P, int(smooth_t0_idx[11]), dtype=int)

        elif self.align_method == 1 and _HAS_SCIPY:
            # EMG fit t₀: fit each pixel on the raw time axis, use fitted t0 param.
            # Uses the same model options (rise / 2nd / 3rd decay) as the fit panel.
            # Fallback: pixel 11's t0 (fitted or smooth-peak if px11 also fails).
            sigma0_ns = self.spin_fit_sigma_ps.value() / 1000.0
            use_rise = self.chk_fit_risetime.isChecked()
            use_2nd  = self.chk_fit_2nd_decay.isChecked()
            use_3rd  = self.chk_fit_3rd_decay.isChecked()
            t = self.time_ns
            t_range = float(t[-1] - t[0]) if N > 1 else 1.0
            fn = lambda tt, *pp: _eval_emg_pixel(tt, sigma0_ns, pp, use_rise, use_2nd, use_3rd)  # noqa: E731

            def _fit_pixel_t0(p: int) -> Optional[int]:
                y = self.smoothed_matrix[:, p]
                peak = float(np.max(y))
                if peak <= 0:
                    return None
                y_norm = y / peak
                t0_init = float(t[smooth_t0_idx[p]]) - sigma0_ns
                tau_init = max(t_range * 0.15, 0.1)
                pp0: List[float] = [0.0]
                plo: List[float] = [-0.5]
                phi: List[float] = [0.5]
                if use_rise:
                    taur_init = min(sigma0_ns * 3.0, tau_init * 0.3)
                    pp0 += [0.5, taur_init]; plo += [0.0, 1e-4]; phi += [5.0, tau_init]
                pp0 += [1.0, tau_init];  plo += [0.0, 1e-4];  phi += [5.0, float(t[-1])]
                if use_2nd:
                    tau2 = max(tau_init * 4.0, 1.0)
                    pp0 += [0.2, tau2]; plo += [0.0, 1e-4]; phi += [5.0, float(t[-1]) * 4]
                if use_3rd:
                    tau3 = max(tau_init * 10.0, 5.0)
                    pp0 += [0.1, tau3]; plo += [0.0, 1e-4]; phi += [5.0, float(t[-1]) * 10]
                pp0 += [t0_init]; plo += [float(t[0]) - 1.0]; phi += [float(t[-1]) + 1.0]
                try:
                    popt, _ = curve_fit(fn, t, y_norm, p0=pp0,  # type: ignore
                                        bounds=(plo, phi), maxfev=2000)
                    fitted_t0 = float(popt[-1])  # t0 is always last in pp layout
                    return int(np.argmin(np.abs(t - fitted_t0)))
                except Exception:
                    return None

            px11_idx = _fit_pixel_t0(11)
            if px11_idx is None:
                px11_idx = int(smooth_t0_idx[11])

            self.t0_idx = np.empty(P, dtype=int)
            for p in range(P):
                if p == 11:
                    self.t0_idx[p] = px11_idx
                else:
                    result = _fit_pixel_t0(p)
                    self.t0_idx[p] = result if result is not None else px11_idx

        else:
            # Method 0: smooth-peak per pixel
            self.t0_idx = smooth_t0_idx.copy()
            if self.noise_level > 0:
                peak_smooth = np.max(self.smoothed_matrix, axis=0)
                low_signal = peak_smooth <= self.noise_level
                low_signal[11] = False
                self.t0_idx[low_signal] = int(self.t0_idx[11])

        self.t0_ns = self.time_ns[self.t0_idx]

        # aligned bins where ≥6 pixels still have fresh data
        # 6th-smallest t0_idx sets the limit (the 6th pixel runs out last of
        # the "top-6 earliest-peak" group, so exactly 6 pixels have data up to that point)
        t0_sorted = np.sort(self.t0_idx)        # ascending
        ref_t0 = int(t0_sorted[min(5, P - 1)])  # 6th smallest (0-indexed = 5)
        self.n_aligned_bins = max(1, N - ref_t0)

        dt = float(self.time_ns[1] - self.time_ns[0]) if N > 1 else 1.0
        self.aligned_time_ns = np.arange(self.n_aligned_bins, dtype=np.float64) * dt

        # aligned_counts[k, p] = counts_matrix[t0_idx[p] + k, p]  (raw, not smoothed)
        k_idx = np.arange(self.n_aligned_bins)[:, np.newaxis]      # (K, 1)
        p_idx = np.arange(P)[np.newaxis, :]                        # (1, P)
        raw_idx = np.clip(self.t0_idx[np.newaxis, :] + k_idx, 0, N - 1)
        self.aligned_counts = self.counts_matrix[raw_idx, p_idx].astype(np.float64)
        self.aligned_smoothed_counts = self.smoothed_matrix[raw_idx, p_idx]

        default_end = float(self.aligned_time_ns[-1]) if len(self.aligned_time_ns) else 0.0
        self.end_time_ns = default_end
        self.spin_end_ns.blockSignals(True)
        self.spin_end_ns.setValue(default_end)
        self.spin_end_ns.blockSignals(False)

        self._compute_baselines()

    def _compute_baselines(self) -> None:
        N, P = self.counts_matrix.shape
        self.baseline_counts = np.zeros(P, dtype=np.float64)
        if N == 0:
            return
        frac = self.baseline_window_pct / 100.0
        w = max(5, int(round(frac * N)))
        for p in range(P):
            i_peak = int(self.t0_idx[p])
            start = max(0, i_peak - w)
            if i_peak - start >= 3:
                seg = self.counts_matrix[start:i_peak, p].astype(np.float64)
            else:
                seg = self.counts_matrix[:min(w, N), p].astype(np.float64)
            self.baseline_counts[p] = float(np.median(seg)) if seg.size > 0 else 0.0

    def _on_baseline_pct_changed(self, value: int) -> None:
        self.baseline_window_pct = float(value)
        if self.counts_matrix.shape[0] > 0:
            self._compute_baselines()
            self._update_histogram()

    def _clear_fit_results(self) -> None:
        self.fit_results_m1.clear()
        self.fit_results_m2.clear()
        self.m2_reference_fit = None
        self.m2_reference_center = None
        self.m2_reference_mask = None
        self._xt_t = self._xt_r = self._xt_z = None
        self._xt_r_raw_sample = self._xt_enabled_idx = self._xt_order = None
        self.m2_r_fixed_detector_sorted = None
        self.m2_reference_order = None

    # ── Slider & aligned-bin navigation ──────────────────────────────────────

    def _on_slider_changed(self, value: int) -> None:
        self._set_current_aligned_idx(value)

    def _set_current_aligned_idx(self, idx: int) -> None:
        if self.n_aligned_bins == 0:
            return
        idx = int(np.clip(idx, 0, self.n_aligned_bins - 1))
        self.current_aligned_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        if len(self.aligned_time_ns) > idx:
            t = float(self.aligned_time_ns[idx])
            if not self.align_t0:
                ref = "px11 t₀"
            elif self.align_method == 1:
                ref = "EMG t₀"
            else:
                ref = "t₀"
            self.lbl_time.setText(f"t = {t:.4f} ns (after {ref},  bin {idx})")
        self._refresh_all()

    # ── Alignment toggle & smoothing/noise controls ───────────────────────────

    def _on_align_changed(self, checked: bool) -> None:
        self.align_t0 = checked
        self.cmb_t0_method.setEnabled(checked)
        # noise floor only applies to the smooth-peak method
        self.spin_noise.setEnabled(checked and self.align_method == 0)
        if self.counts_matrix.shape[0] == 0:
            return
        had_fit = bool(self.fit_results_m1 or self.fit_results_m2)
        self._clear_fit_results()
        self._compute_alignment()
        self.slider.setMaximum(max(0, self.n_aligned_bins - 1))
        idx = min(self.current_aligned_idx, self.n_aligned_bins - 1)
        self._set_current_aligned_idx(idx)
        if had_fit and self.aligned_counts.shape[0] > 0:
            self._on_fit_all()

    def _on_t0_method_changed(self, index: int) -> None:
        self.align_method = index
        # noise floor spinbox only meaningful for smooth-peak method
        self.spin_noise.setEnabled(self.align_t0 and index == 0)
        if self.counts_matrix.shape[0] == 0:
            return
        had_fit = bool(self.fit_results_m1 or self.fit_results_m2)
        self._clear_fit_results()
        self._compute_alignment()
        self.slider.setMaximum(max(0, self.n_aligned_bins - 1))
        idx = min(self.current_aligned_idx, self.n_aligned_bins - 1)
        self._set_current_aligned_idx(idx)
        if had_fit and self.aligned_counts.shape[0] > 0:
            self._on_fit_all()

    def _on_smooth_changed(self, value: int) -> None:
        self.smooth_window = value
        if self.counts_matrix.shape[0] == 0:
            return
        self._clear_fit_results()
        self._compute_alignment()
        self.slider.setMaximum(max(0, self.n_aligned_bins - 1))
        idx = min(self.current_aligned_idx, self.n_aligned_bins - 1)
        self._set_current_aligned_idx(idx)

    def _on_noise_changed(self, value: float) -> None:
        self.noise_level = value
        if self.counts_matrix.shape[0] == 0:
            return
        had_fit = bool(self.fit_results_m1 or self.fit_results_m2)
        self._clear_fit_results()
        self._compute_alignment()
        self.slider.setMaximum(max(0, self.n_aligned_bins - 1))
        idx = min(self.current_aligned_idx, self.n_aligned_bins - 1)
        self._set_current_aligned_idx(idx)   # calls _refresh_all internally
        if had_fit and self.aligned_counts.shape[0] > 0:
            self._on_fit_all()

    def _on_magnification_changed(self, value: float) -> None:
        self.magnification = max(1e-6, value)
        self._fitting_widget.set_magnification(self.magnification)
        self._update_spacetime_plot()
        self._update_fitting()

    # ── Pixel selection ───────────────────────────────────────────────────────

    def _on_pixel_left(self, pixel: int) -> None:
        if pixel in self.selected_pixels:
            self.selected_pixels.remove(pixel)
        else:
            self.selected_pixels.add(pixel)
        self._update_map()
        self._update_histogram()

    def _on_pixel_right(self, pixel: int) -> None:
        if pixel in self.excluded_pixels:
            self.excluded_pixels.remove(pixel)
        else:
            self.excluded_pixels.add(pixel)
        had_fit = bool(self.fit_results_m1 or self.fit_results_m2)
        self._clear_fit_results()
        self._refresh_all()
        if had_fit and self.aligned_counts.shape[0] > 0:
            self._on_fit_all()

    # ── Plot updates ──────────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._update_map()
        self._update_histogram()
        self._update_fitting()

    def _update_map(self) -> None:
        if self.aligned_counts.shape[0] == 0:
            return
        k = min(self.current_aligned_idx, self.aligned_counts.shape[0] - 1)
        frame = self.aligned_counts[k, :]
        aligned_t = float(self.aligned_time_ns[k]) if len(self.aligned_time_ns) > k else 0.0
        self._spad_map.update(frame, aligned_t, self.selected_pixels, self.excluded_pixels)

    def _update_histogram(self) -> None:
        if self.counts_matrix.shape[0] == 0 or self.smoothed_matrix.shape[0] == 0:
            return
        current_ns = (float(self.aligned_time_ns[self.current_aligned_idx])
                      if len(self.aligned_time_ns) > self.current_aligned_idx else 0.0)
        self._hist_widget.update(
            self.time_ns, self.counts_matrix, self.smoothed_matrix, self.t0_ns,
            current_ns, self.selected_pixels,
            self.chk_hist_log.isChecked(), self.chk_normalize.isChecked(),
            baseline_counts=self.baseline_counts,
            subtract_baseline=self.chk_subtract_baseline.isChecked(),
            show_baseline=self.chk_show_baseline.isChecked())
        self._refresh_emg_fits()

    def _refresh_emg_fits(self) -> None:
        if not _HAS_SCIPY or not self.chk_fit_emg.isChecked():
            return
        if self.aligned_smoothed_counts.shape[0] == 0 or len(self.aligned_time_ns) == 0:
            self._hist_widget.clear_fits()
            self.lbl_fit_results.setText("")
            return

        use_rise = self.chk_fit_risetime.isChecked()
        use_2nd  = self.chk_fit_2nd_decay.isChecked()
        use_3rd  = self.chk_fit_3rd_decay.isChecked()
        sigma0_ns = self.spin_fit_sigma_ps.value() / 1000.0  # ps → ns
        t = self.aligned_time_ns
        n_bins = len(t)
        t_range = float(t[-1] - t[0]) if n_bins > 1 else 1.0
        t_lo = float(t[0]) - 1.0
        t_hi = float(t[-1]) + 1.0

        # Collect pixels with nonzero signal
        pixels = [p for p in sorted(self.selected_pixels)
                  if float(np.max(self.aligned_smoothed_counts[:, p])) > 0]
        peak_vals: Dict[int, float] = {
            p: float(np.max(self.aligned_smoothed_counts[:, p]))
            for p in sorted(self.selected_pixels)}

        fit_curves: Dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {
            p: None for p in sorted(self.selected_pixels)}
        result_lines: List[str] = []

        if not pixels:
            self._hist_widget.update_fits(fit_curves, self.chk_normalize.isChecked(), peak_vals)
            self.lbl_fit_results.setText("no signal")
            return

        n_pixels = len(pixels)
        model_fn, n_per_pixel = _build_joint_model(
            n_pixels, n_bins, use_rise, use_2nd, use_3rd, sigma0_ns)

        # Stack normalized data for joint fit
        y_norm_list = [self.aligned_smoothed_counts[:, p] / peak_vals[p] for p in pixels]
        t_all = np.tile(t, n_pixels)
        y_all = np.concatenate(y_norm_list)

        # Build p0 and bounds: [per-pixel-block-0, per-pixel-block-1, ...]
        # sigma and t0 are both fixed — neither is a free parameter.
        # Per-pixel layout: [C, (Ar, taur,) A1, tau1, (A2, tau2,) (A3, tau3,)]  (no t0)
        p0: List[float] = []
        lo_b: List[float] = []
        hi_b: List[float] = []

        for p in pixels:
            tau1_init = max(t_range * 0.15, 0.1)

            pp0: List[float] = [0.0]          # C
            plo: List[float] = [-0.5]
            phi: List[float] = [0.5]
            if use_rise:
                taur_init = min(sigma0_ns * 3.0, tau1_init * 0.3)
                pp0 += [0.5, taur_init];  plo += [0.0, 1e-4];  phi += [5.0, tau1_init]
            pp0 += [1.0, tau1_init];      plo += [0.0, 1e-4];  phi += [5.0, float(t[-1])]
            if use_2nd:
                tau2_init = max(tau1_init * 4.0, 1.0)
                pp0 += [0.2, tau2_init];  plo += [0.0, 1e-4];  phi += [5.0, float(t[-1]) * 4]
            if use_3rd:
                tau3_init = max(tau1_init * 10.0, 5.0)
                pp0 += [0.1, tau3_init];  plo += [0.0, 1e-4];  phi += [5.0, float(t[-1]) * 10]

            p0  += pp0
            lo_b += plo
            hi_b += phi

        t_dense = np.linspace(float(t[0]), float(t[-1]), 600)

        try:
            popt, _ = curve_fit(  # type: ignore
                model_fn, t_all, y_all,
                p0=p0, bounds=(lo_b, hi_b),
                maxfev=6000 * n_pixels)

            for i, p in enumerate(pixels):
                pp = popt[i * n_per_pixel: (i + 1) * n_per_pixel]
                y_fit = _eval_emg_pixel_fixed_t0(
                    t_dense, sigma0_ns, pp, use_rise, use_2nd, use_3rd)
                fit_curves[p] = (t_dense, y_fit * peak_vals[p])

                # Build result label for this pixel
                j = 1
                parts: List[str] = [f"Px{p}:"]
                if use_rise:
                    parts.append(f"τᵣ={float(pp[j+1])*1e3:.1f}")
                    j += 2
                parts.append(f"τ₁={float(pp[j+1])*1e3:.1f}")
                j += 2
                if use_2nd:
                    parts.append(f"τ₂={float(pp[j+1])*1e3:.1f}")
                    j += 2
                if use_3rd:
                    parts.append(f"τ₃={float(pp[j+1])*1e3:.1f}")
                result_lines.append(" ".join(parts) + " ps")

            result_lines.append(f"σ={sigma0_ns*1e3:.1f} ps (fixed)")

        except Exception:
            result_lines.append("fit failed")

        self._hist_widget.update_fits(fit_curves, self.chk_normalize.isChecked(), peak_vals)
        self.lbl_fit_results.setText("  |  ".join(result_lines))

    def _parse_extra_times(self) -> List[float]:
        text = self.edit_extra_times.text().strip()
        if not text:
            return []
        out: List[float] = []
        for tok in text.replace(';', ',').split(','):
            tok = tok.strip()
            if tok:
                try:
                    out.append(float(tok))
                except ValueError:
                    pass
        return out[:5]

    def _update_spacetime_plot(self, _=None) -> None:
        """Build the x–t (space–time) colormap from fit parameters and refresh p_xt."""
        method = self.cmb_fit_method.currentIndex()
        M = self.magnification
        normalize = self.chk_normalize_spatial.isChecked()
        current_t_ns = (float(self.aligned_time_ns[self.current_aligned_idx])
                        if len(self.aligned_time_ns) > self.current_aligned_idx else 0.0)
        R_GRID = 120

        if method == 0 and self.fit_results_m1:
            valid = sorted(self.fit_results_m1.keys())
            if not valid:
                return
            ref = self.fit_results_m1[valid[0]]
            x0_ref = float(ref.popt[1]);  y0_ref = float(ref.popt[2])
            r_max_det = float(np.max(
                np.sqrt((self.fit_x_um - x0_ref) ** 2 + (self.fit_y_um - y0_ref) ** 2)))
            r_arr = np.linspace(0.0, r_max_det / M, R_GRID)
            end_bin = max(valid) + 1
            t_arr = self.aligned_time_ns[:end_bin]
            xt_z = np.zeros((end_bin, R_GRID), dtype=np.float32)
            sigma_full = np.full(end_bin, np.nan)
            for k in valid:
                fit = self.fit_results_m1[k]
                A = float(fit.popt[0]);  offset = float(fit.popt[5])
                sig_s = max(fit.sigma_eq / M, 1e-9)
                xt_z[k] = A * np.exp(-r_arr ** 2 / (2 * sig_s ** 2)) + offset
                sigma_full[k] = sig_s
            vm = np.isfinite(sigma_full)
            self._fitting_widget.update_xt(
                t_arr, r_arr, xt_z,
                sigma_full[vm], t_arr[vm],
                normalize, current_t_ns)

        elif method == 1 and self.fit_results_m2 and self.m2_r_max_detector > 0:
            valid = sorted(self.fit_results_m2.keys())
            if not valid:
                return
            r_arr = np.linspace(0.0, self.m2_r_max_detector / M, R_GRID)
            end_bin = max(valid) + 1
            t_arr = self.aligned_time_ns[:end_bin]
            xt_z = np.zeros((end_bin, R_GRID), dtype=np.float32)
            sigma_full = np.full(end_bin, np.nan)
            for k in valid:
                fit1d = self.fit_results_m2[k]
                A, sigma, offset = [float(v) for v in fit1d.popt]
                sig_s = max(abs(sigma) / M, 1e-9)
                xt_z[k] = A * np.exp(-r_arr ** 2 / (2 * sig_s ** 2)) + offset
                sigma_full[k] = sig_s
            vm = np.isfinite(sigma_full)
            self._fitting_widget.update_xt(
                t_arr, r_arr, xt_z,
                sigma_full[vm], t_arr[vm],
                normalize, current_t_ns)

    def _update_fitting(self) -> None:
        if self.aligned_smoothed_counts.shape[0] == 0:
            self._fitting_widget.no_fit_maps()
            return
        method = self.cmb_fit_method.currentIndex()
        M = self.magnification
        k = int(self.current_aligned_idx)
        frame = self.aligned_smoothed_counts[k, :]

        t_arr: Optional[np.ndarray] = None
        sigma2: Optional[np.ndarray] = None
        reliable_mask: Optional[np.ndarray] = None
        if method == 0 and self.fit_results_m1:
            valid = sorted(self.fit_results_m1.keys())
            if valid:
                valid_idx = np.array(valid, dtype=int)
                t_arr = self.aligned_time_ns[valid_idx]
                sigma2 = np.array([self.fit_results_m1[i].sigma_eq ** 2
                                   for i in valid], dtype=np.float64)
                threshold = self.spin_asym_threshold.value()
                reliable_mask = np.ones(len(valid), dtype=bool)
                for j, i in enumerate(valid):
                    popt = self.fit_results_m1[i].popt
                    sx, sy = abs(float(popt[3])), abs(float(popt[4]))
                    mn = min(sx, sy)
                    if mn > 0 and max(sx, sy) / mn > threshold:
                        reliable_mask[j] = False
        elif method == 1 and self.fit_results_m2:
            valid = sorted(self.fit_results_m2.keys())
            if valid:
                valid_idx = np.array(valid, dtype=int)
                t_arr = self.aligned_time_ns[valid_idx]
                sigma2 = np.array([self.fit_results_m2[i].sigma_eq ** 2
                                   for i in valid], dtype=np.float64)
        self._fitting_widget.update_sigma(
            self.aligned_time_ns, k, t_arr, sigma2,
            self.spin_sigma_smooth.value(), reliable_mask)

        # ── Extra radial lines (popt tuples for the new _draw_radial_extra API) ─
        extra_traces: List = []
        req_times = self._parse_extra_times()
        if req_times:
            if method == 0 and self.fit_results_m1:
                valid_m1 = sorted(self.fit_results_m1.keys())
                t_valid = self.aligned_time_ns[np.array(valid_m1, dtype=int)]
                ref = self.fit_results_m1[valid_m1[0]]
                r_max_s = float(np.max(np.sqrt(
                    (self.fit_x_um - float(ref.popt[1])) ** 2 +
                    (self.fit_y_um - float(ref.popt[2])) ** 2))) / M
                for t_req in req_times:
                    idx = int(np.argmin(np.abs(t_valid - t_req)))
                    ki = valid_m1[idx]
                    fit = self.fit_results_m1[ki]
                    popt_s = np.array([float(fit.popt[0]),
                                       max(fit.sigma_eq / M, 1e-9),
                                       float(fit.popt[5])])
                    t_actual = float(self.aligned_time_ns[ki])
                    inp_ki = self._fit_inputs(self.aligned_smoothed_counts[ki, :])
                    if inp_ki is not None:
                        x_ki, y_ki, z_ki, _ = inp_ki
                        r_ki = np.sqrt((x_ki - float(fit.popt[1])) ** 2 +
                                       (y_ki - float(fit.popt[2])) ** 2)
                        ord_ki = np.argsort(r_ki)
                        extra_traces.append((r_ki[ord_ki] / M, z_ki[ord_ki],
                                             popt_s, t_actual))
                    else:
                        extra_traces.append((np.array([r_max_s]), None,
                                             popt_s, t_actual))
            elif method == 1 and self.fit_results_m2 and self.m2_r_max_detector > 0:
                valid_m2 = sorted(self.fit_results_m2.keys())
                t_valid = self.aligned_time_ns[np.array(valid_m2, dtype=int)]
                r_max_s = self.m2_r_max_detector / M
                for t_req in req_times:
                    idx = int(np.argmin(np.abs(t_valid - t_req)))
                    ki = valid_m2[idx]
                    fit1d = self.fit_results_m2[ki]
                    A, sigma, offset = [float(v) for v in fit1d.popt]
                    popt_s = np.array([A, max(abs(sigma) / M, 1e-9), offset])
                    t_actual = float(self.aligned_time_ns[ki])
                    if (self.m2_r_fixed_detector_sorted is not None and
                            self.m2_reference_order is not None and
                            self.m2_reference_mask is not None):
                        enabled_idx = np.where(self.m2_reference_mask)[0]
                        z_ki = self.aligned_smoothed_counts[
                            ki, enabled_idx][self.m2_reference_order]
                        r_samp = self.m2_r_fixed_detector_sorted / M
                        extra_traces.append((r_samp, z_ki, popt_s, t_actual))
                    else:
                        extra_traces.append((np.array([r_max_s]), None,
                                             popt_s, t_actual))

        normalize = self.chk_normalize_spatial.isChecked()
        if method == 0:
            self._fitting_widget.update_m1_maps(
                frame, self.fit_results_m1.get(k),
                self.fit_x_um, self.fit_y_um, self.excluded_pixels,
                extra_traces=extra_traces, normalize=normalize)
        else:
            if self.m2_reference_fit is None or self.m2_reference_center is None:
                self._fitting_widget.no_fit_maps()
                return
            ref_frame = self.aligned_smoothed_counts[0, :]
            x0_ref, y0_ref = self.m2_reference_center
            self._fitting_widget.update_m2_maps(
                ref_frame, frame,
                self.m2_reference_fit, self.fit_results_m2.get(k),
                self.fit_x_um, self.fit_y_um, self.excluded_pixels,
                0, self.aligned_time_ns, k, x0_ref, y0_ref,
                extra_traces=extra_traces, normalize=normalize)

        # Keep the current-time marker on the x-t colormap in sync with the slider
        if len(self.aligned_time_ns) > k:
            self._fitting_widget.update_xt_marker(float(self.aligned_time_ns[k]))

    # ── Fitting ───────────────────────────────────────────────────────────────

    def _fit_inputs(self, frame: np.ndarray):
        mask = np.ones(NUM_PIXELS, dtype=bool)
        if self.excluded_pixels:
            mask[np.array(sorted(self.excluded_pixels), dtype=int)] = False
        if int(np.sum(mask)) < 6:
            return None
        return self.fit_x_um[mask], self.fit_y_um[mask], frame[mask], mask

    def _on_fit_all(self) -> None:
        if not _HAS_SCIPY:
            QtWidgets.QMessageBox.warning(self, "Fit All",
                                          "scipy is required for fitting.")
            return
        if self.aligned_counts.shape[0] == 0:
            QtWidgets.QMessageBox.warning(self, "Fit All", "No data loaded.")
            return

        dt = (float(self.aligned_time_ns[1] - self.aligned_time_ns[0])
              if len(self.aligned_time_ns) > 1 else 1.0)
        max_available_ns = float(self.aligned_time_ns[-1]) if len(self.aligned_time_ns) else 0.0
        end_time = float(self.spin_end_ns.value())
        if end_time > max_available_ns:
            self.spin_end_ns.blockSignals(True)
            self.spin_end_ns.setValue(max_available_ns)
            self.spin_end_ns.blockSignals(False)
            end_time = max_available_ns
        end_bin = min(self.n_aligned_bins,
                      max(1, int(round(end_time / dt)) + 1))

        ref_frame = self.aligned_smoothed_counts[0, :]
        ref_inputs = self._fit_inputs(ref_frame)
        if ref_inputs is None:
            QtWidgets.QMessageBox.warning(
                self, "Fit All",
                "Not enough enabled pixels for fitting (need ≥ 6).")
            return
        x_ref, y_ref, z_ref, mask_ref = ref_inputs
        ref_fit_2d = _fit_gaussian_2d(x_ref, y_ref, z_ref)
        if ref_fit_2d is None:
            QtWidgets.QMessageBox.warning(
                self, "Fit All",
                "Could not fit the t=0 reference frame.")
            return

        method = self.cmb_fit_method.currentIndex()
        failures = 0

        if method == 0:
            self.fit_results_m1.clear()
            self.fit_results_m1[0] = ref_fit_2d
            prev = ref_fit_2d
            for k in range(1, end_bin):
                inp = self._fit_inputs(self.aligned_smoothed_counts[k, :])
                if inp is None:
                    failures += 1
                    continue
                fit = _fit_gaussian_2d_constrained_next(
                    inp[0], inp[1], inp[2], prev.popt, tightness=1.0)
                if fit is None:
                    fit = _fit_gaussian_2d_constrained_next(
                        inp[0], inp[1], inp[2], prev.popt, tightness=1.8)
                if fit is None:
                    failures += 1
                    continue
                self.fit_results_m1[k] = fit
                prev = fit
                if k % 200 == 0:
                    QtWidgets.QApplication.processEvents()
            fitted = len(self.fit_results_m1)
            self.lbl_fit_state.setText(
                f"Method 1 done: {fitted} fitted, {failures} failures, "
                f"bins 0–{end_bin - 1}.")

        elif method == 1:
            x0_ref = float(ref_fit_2d.popt[1])
            y0_ref = float(ref_fit_2d.popt[2])
            r_fixed = np.sqrt((x_ref - x0_ref) ** 2 + (y_ref - y0_ref) ** 2)
            order = np.argsort(r_fixed)
            r_fixed_sorted = r_fixed[order]
            self.m2_r_max_detector = float(np.max(r_fixed_sorted)) if len(r_fixed_sorted) > 0 else 0.0
            self.m2_r_fixed_detector_sorted = r_fixed_sorted.copy()
            self.m2_reference_order = order.copy()

            self.fit_results_m2.clear()
            self.m2_reference_fit = ref_fit_2d
            self.m2_reference_center = (x0_ref, y0_ref)
            self.m2_reference_mask = mask_ref.copy()

            ref_radial = _fit_radial_gaussian_1d(
                r_fixed_sorted, z_ref[order],
                p0=None, sigma_guess=ref_fit_2d.sigma_eq)
            if ref_radial is None:
                QtWidgets.QMessageBox.warning(
                    self, "Fit All", "Reference radial 1D fit failed.")
                return
            self.fit_results_m2[0] = ref_radial
            prev_r = ref_radial

            for k in range(1, end_bin):
                inp = self._fit_inputs(self.aligned_smoothed_counts[k, :])
                if inp is None:
                    failures += 1
                    continue
                _, _, z_k, mask_k = inp
                if np.any(mask_k != mask_ref):
                    failures += 1
                    continue
                fit1d = _fit_radial_gaussian_1d(
                    r_fixed_sorted, z_k[order],
                    p0=prev_r.popt, sigma_guess=prev_r.sigma_eq)
                if fit1d is None:
                    fit1d = _fit_radial_gaussian_1d(
                        r_fixed_sorted, z_k[order],
                        p0=None, sigma_guess=prev_r.sigma_eq)
                if fit1d is None:
                    failures += 1
                    continue
                self.fit_results_m2[k] = fit1d
                prev_r = fit1d
                if k % 200 == 0:
                    QtWidgets.QApplication.processEvents()
            fitted = len(self.fit_results_m2)
            self.lbl_fit_state.setText(
                f"Method 2 done: {fitted} fitted, {failures} failures, "
                f"center=({x0_ref:.1f}, {y0_ref:.1f}) µm.")

        self._update_spacetime_plot()
        self._update_fitting()

    # ── Exports ───────────────────────────────────────────────────────────────

    def _default_export_dir(self) -> str:
        path = self.edit_path.text().strip()
        if path:
            p = Path(path).expanduser().parent
            if p.exists():
                return str(p)
        return ""

    def _choose_export_path(self, default_name: str) -> Optional[Path]:
        default_dir = self._default_export_dir()
        default_path = (str(Path(default_dir) / default_name)
                        if default_dir else default_name)
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", default_path,
            "CSV files (*.csv);;All files (*)")
        if not file_path:
            return None
        out = Path(file_path)
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        return out

    def _on_export_hist(self) -> None:
        if self.aligned_counts.shape[0] == 0 or not self.selected_pixels:
            QtWidgets.QMessageBox.warning(
                self, "Export", "No selected histogram traces to export.")
            return
        path = self._choose_export_path("spad23_histogram_aligned.csv")
        if path is None:
            return
        selected = sorted(self.selected_pixels)
        N_raw = self.counts_matrix.shape[0]
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["# source_file", self.edit_path.text().strip()])
                writer.writerow(["# smooth_window_bins", self.smooth_window])
                for p in selected:
                    writer.writerow([f"# t0_px{p}_ns", f"{float(self.t0_ns[p]):.12g}"])
                writer.writerow([])
                header = ["aligned_t_ns"]
                for p in selected:
                    header += [f"raw_px{p}", f"smooth_px{p}"]
                writer.writerow(header)
                for k in range(self.n_aligned_bins):
                    row_data = [f"{float(self.aligned_time_ns[k]):.12g}"]
                    for p in selected:
                        raw_val = float(self.aligned_counts[k, p])
                        raw_src = int(self.t0_idx[p]) + k
                        smooth_val = (float(self.smoothed_matrix[
                            min(raw_src, N_raw - 1), p])
                            if self.smoothed_matrix.shape[0] > 0 else raw_val)
                        row_data += [f"{raw_val:.12g}", f"{smooth_val:.12g}"]
                    writer.writerow(row_data)
            self._set_status(f"Exported aligned traces: {path.name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed:\n{exc}")

    def _on_export_sigma(self) -> None:
        method = self.cmb_fit_method.currentIndex()
        results = self.fit_results_m1 if method == 0 else self.fit_results_m2
        if not results:
            QtWidgets.QMessageBox.warning(
                self, "Export",
                "No σ² trace to export — run Fit All first.")
            return
        path = self._choose_export_path("spad23_sigma_eq2_vs_time.csv")
        if path is None:
            return
        valid = sorted(results.keys())
        M = self.magnification
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["# method_index", method + 1])
                writer.writerow(["# method", self.cmb_fit_method.currentText()])
                writer.writerow(["# smooth_window_bins", self.smooth_window])
                writer.writerow(["# magnification_M", f"{M:.12g}"])
                writer.writerow(["# length_unit", "sample_um (= detector_um / M)"])
                writer.writerow([])
                if method == 0:
                    writer.writerow([
                        "aligned_bin", "aligned_t_ns",
                        "sigma_eq_sample_um", "sigma_eq2_sample_um2",
                        "amplitude", "x0_sample_um", "y0_sample_um",
                        "sx_sample_um", "sy_sample_um", "offset"])
                    for k in valid:
                        fit = results[k]
                        p = fit.popt
                        t = float(self.aligned_time_ns[k]) if k < len(self.aligned_time_ns) else 0.0
                        seq = fit.sigma_eq / M
                        writer.writerow([
                            k, f"{t:.12g}",
                            f"{seq:.12g}", f"{seq**2:.12g}",
                            f"{float(p[0]):.12g}",
                            f"{float(p[1])/M:.12g}", f"{float(p[2])/M:.12g}",
                            f"{float(p[3])/M:.12g}", f"{float(p[4])/M:.12g}",
                            f"{float(p[5]):.12g}"])
                else:
                    writer.writerow([
                        "aligned_bin", "aligned_t_ns",
                        "sigma_eq_sample_um", "sigma_eq2_sample_um2",
                        "amplitude", "sigma_sample_um", "offset"])
                    for k in valid:
                        fit = results[k]
                        p = fit.popt
                        t = float(self.aligned_time_ns[k]) if k < len(self.aligned_time_ns) else 0.0
                        seq = fit.sigma_eq / M
                        writer.writerow([
                            k, f"{t:.12g}",
                            f"{seq:.12g}", f"{seq**2:.12g}",
                            f"{float(p[0]):.12g}",
                            f"{float(p[1])/M:.12g}", f"{float(p[2]):.12g}"])
            self._set_status(f"Exported σ² trace: {path.name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed:\n{exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, timeout_ms: int = 4000) -> None:
        self._status.showMessage(msg, timeout_ms)

    def load_csv_path(self, path: str) -> None:
        self.edit_path.setText(path)
        self._load_csv(path)


import csv
import json
import os
import re
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider, Slider

from andor.andor_wrapper import AndorSystem
from andor.gui.live_view_widget import AndorLiveViewWidget
from andor.gui.workers import LiveAcqThread
from rot.rot_wrapper import MotionController, ROT_RANGE_DEFAULT, RotationStage, list_kinesis_serials

from measurements.dual_wheel_intensity_calib_config import (
    DEFAULT_A_COARSE_STEPS,
    DEFAULT_ACQ_NUMBER,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_CLAMP_T_TO_1,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_FINE_STEP_DEG,
    DEFAULT_GOOD_OD_HIGH,
    DEFAULT_GOOD_OD_A_HIGH,
    DEFAULT_GOOD_OD_A_LOW,
    DEFAULT_GOOD_OD_LOW,
    DEFAULT_MAX_ANGLE_A,
    DEFAULT_MAX_ANGLE_B,
    DEFAULT_MIN_ANGLE_A,
    DEFAULT_MIN_ANGLE_B,
    DEFAULT_MIN_DP_HIGH,
    DEFAULT_MIN_DP_LOW,
    DEFAULT_MIN_STEP_NEAR_OPEN_DEG,
    DEFAULT_OUTPUT_AMP,
    DEFAULT_OD_SMOOTH_PTS,
    DEFAULT_P00_POWER,
    DEFAULT_POINT_SPLIT,
    DEFAULT_POWER_SPLIT,
    DEFAULT_POWER_UNIT,
    DEFAULT_PREAMP_GAIN,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_READOUT_RATE,
    DEFAULT_REF_EVERY,
    DEFAULT_ROI_X1,
    DEFAULT_ROI_X2,
    DEFAULT_ROI_Y1,
    DEFAULT_ROI_Y2,
    DEFAULT_SENS_THRESH,
    DEFAULT_SETTLE_MS,
    DEFAULT_SLIT_ID,
    DEFAULT_SLIT_UM,
    DEFAULT_SPEC_INDEX,
    DEFAULT_STAGE_A_SERIAL,
    DEFAULT_STAGE_B_SERIAL,
    DEFAULT_STAGE_SCALE,
    DEFAULT_TOTAL_POINTS,
    DEFAULT_TOTAL_POINTS_MAX,
    DEFAULT_TOTAL_POINTS_MIN,
    POLL_MS,
    DATA_DIR,
)
from measurements.dual_wheel_intensity_workers import DualWheelFineSweepThread, DualWheelListThread
from measurements.intensity_workers import IntensitySweepThread

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

try:
    from scipy.optimize import isotonic_regression as scipy_isotonic_regression
except Exception:
    scipy_isotonic_regression = None

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None


_UNIT_SCALE = {
    "W": 1.0,
    "mW": 1e-3,
    "uW": 1e-6,
    "nW": 1e-9,
}


def power_from_counts(counts: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * np.asarray(counts, dtype=float) + intercept


def od_from_transmittance(transmittance: np.ndarray) -> np.ndarray:
    t = np.asarray(transmittance, dtype=float)
    t = np.clip(t, 1e-12, None)
    return -np.log10(t)


def sanitize_transmittance_for_od(transmittance: np.ndarray, *, clamp_to_one: bool) -> Optional[np.ndarray]:
    t = np.asarray(transmittance, dtype=float).copy()
    if t.size == 0:
        return None
    valid = np.isfinite(t) & (t > 0.0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid <= 0:
        return None
    if n_valid == 1:
        t[:] = float(t[valid][0])
    elif np.any(~valid):
        x = np.arange(t.size, dtype=float)
        t[~valid] = np.interp(x[~valid], x[valid], t[valid])
    if clamp_to_one:
        t = np.minimum(t, 1.0)
    t = np.clip(t, 1e-12, None)
    return t


def build_wheel_arrays(model_dict: dict, slope: float, intercept: float):
    ang_u = np.asarray(model_dict["angles"], dtype=float)
    ang_w = np.asarray(model_dict["angles_wrapped"], dtype=float)
    intens = np.asarray(model_dict["intensities"], dtype=float)
    power = power_from_counts(intens, slope, intercept)
    return ang_u, ang_w, intens, power


def invert_monotonic(x: np.ndarray, y: np.ndarray, yq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(y)
    return np.interp(yq, y[idx], x[idx])


def compute_sensitivity(ang_u: np.ndarray, od: np.ndarray) -> np.ndarray:
    return np.abs(np.gradient(od, ang_u))


def find_od_open_cut(od: np.ndarray, sens: np.ndarray, sens_thresh: float) -> float:
    od = np.asarray(od, dtype=float)
    sens = np.asarray(sens, dtype=float)
    idx = np.argsort(od)
    od_s = od[idx]
    sens_s = sens[idx]
    hit = np.where(sens_s >= sens_thresh)[0]
    if hit.size == 0:
        return float(np.max(od_s))
    return float(od_s[int(hit[0])])


def quantize_angles_min_step(
    ang_deg: np.ndarray,
    od_f_used: np.ndarray,
    od_open_cut: float,
    min_step_deg: float = 1.0,
) -> np.ndarray:
    ang_deg = np.asarray(ang_deg, dtype=float).copy()
    if min_step_deg <= 0:
        ang_deg[:] = np.mod(ang_deg, 360.0)
        return ang_deg
    mask = od_f_used <= od_open_cut
    ang_deg[mask] = np.round(ang_deg[mask] / min_step_deg) * min_step_deg
    ang_deg[:] = np.mod(ang_deg, 360.0)
    return ang_deg


def map_wrapped_to_unwrapped(
    angles_deg: np.ndarray,
    unwrapped_grid: np.ndarray,
    wrap_deg: float = 360.0,
) -> np.ndarray:
    arr = np.asarray(angles_deg, dtype=float)
    grid = np.asarray(unwrapped_grid, dtype=float)
    if arr.size == 0 or grid.size == 0:
        return arr
    lo = float(np.nanmin(grid))
    hi = float(np.nanmax(grid))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return arr
    if hi < lo:
        lo, hi = hi, lo
    if wrap_deg <= 0:
        return arr
    adj = ((arr - lo) % wrap_deg) + lo
    adj = np.where(adj > hi, adj - wrap_deg, adj)
    return adj


def isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    if scipy_isotonic_regression is not None:
        res = scipy_isotonic_regression(-y, increasing=True)
        x = res.x if hasattr(res, "x") else np.asarray(res, float)
        return -x

    def pava_increasing(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = v.size
        blocks = [[i, i, v[i], 1.0] for i in range(n)]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][2] <= blocks[i + 1][2] + 1e-15:
                i += 1
                continue
            s1, e1, m1, w1 = blocks[i]
            s2, e2, m2, w2 = blocks[i + 1]
            m = (m1 * w1 + m2 * w2) / (w1 + w2)
            blocks[i] = [s1, e2, m, w1 + w2]
            del blocks[i + 1]
            i = max(i - 1, 0)

        out = np.empty(n, float)
        for s, e, m, _w in blocks:
            out[s : e + 1] = m
        return out

    return -pava_increasing(-y)


def smooth_then_monotone(ang_u: np.ndarray, od_raw: np.ndarray, smooth_pts: int):
    od = np.asarray(od_raw, dtype=float)

    sp = int(smooth_pts)
    if sp < 3:
        od_s = od.copy()
    else:
        if sp % 2 == 0:
            sp += 1
        sp = min(sp, max(3, len(od) - (1 - len(od) % 2)))
        sp = max(sp, 3)

        if savgol_filter is not None and sp >= 5:
            poly = 3 if sp >= 9 else 2
            od_s = savgol_filter(od, window_length=sp, polyorder=poly, mode="interp")
        else:
            k = sp
            pad = k // 2
            x = np.pad(od, (pad, pad), mode="edge")
            od_s = np.convolve(x, np.ones(k) / k, mode="valid")

    od_m = isotonic_decreasing(od_s)
    return od_s, od_m


def allocate_od_with_good_window_fine_residual(
    od_total: np.ndarray,
    od_f_max: float,
    od_c_levels: np.ndarray,
    good_f_lo: float,
    good_f_hi: float,
    good_c_lo: Optional[float] = None,
    good_c_hi: Optional[float] = None,
):
    od_total = np.asarray(od_total, dtype=float)
    od_f = np.zeros_like(od_total)
    od_c = np.zeros_like(od_total)
    c_idx = np.zeros_like(od_total, dtype=int)

    for i, odt in enumerate(od_total):
        best = None
        for j, odc in enumerate(od_c_levels):
            odf = odt - odc
            if not (0.0 <= odf <= od_f_max):
                continue

            if odf < good_f_lo:
                pen = good_f_lo - odf
            elif odf > good_f_hi:
                pen = odf - good_f_hi
            else:
                pen = 0.0

            pen_c = 0.0
            if good_c_lo is not None and good_c_hi is not None:
                if odc < good_c_lo:
                    pen_c = good_c_lo - odc
                elif odc > good_c_hi:
                    pen_c = odc - good_c_hi

            score = (pen + pen_c, pen, pen_c, -odc)
            if (best is None) or (score < best[0]):
                best = (score, j, odf, odc)

        if best is None:
            j = int(np.clip(np.searchsorted(od_c_levels, odt) - 1, 0, len(od_c_levels) - 1))
            odc = float(od_c_levels[j])
            odf = float(np.clip(odt - odc, 0.0, od_f_max))
        else:
            _, j, odf, odc = best

        c_idx[i] = j
        od_f[i] = odf
        od_c[i] = odc

    return od_f, od_c, c_idx


def map_targets_to_wheels_swapped(
    p_targets: np.ndarray,
    *,
    p00: float,
    a_u: np.ndarray,
    a_w: np.ndarray,
    od_a_for_inversion: np.ndarray,
    b_u: np.ndarray,
    b_w: np.ndarray,
    od_b_for_inversion: np.ndarray,
    od_b_for_snap: np.ndarray,
    od_a_levels: np.ndarray,
    od_b_max: float,
    good_a_lo: float,
    good_a_hi: float,
    good_b_lo: float,
    good_b_hi: float,
    sens_thresh: float,
    s_b: np.ndarray,
    od_b_for_open_cut: np.ndarray,
    min_step_deg: float,
):
    p_targets = np.asarray(p_targets, dtype=float)
    od_total = od_from_transmittance(np.clip(p_targets / p00, 1e-12, 1.0))

    od_b_used, od_a_used, a_idx = allocate_od_with_good_window_fine_residual(
        od_total,
        od_b_max,
        od_a_levels,
        good_b_lo,
        good_b_hi,
        good_a_lo,
        good_a_hi,
    )

    a_u_used = invert_monotonic(a_u, od_a_for_inversion, od_a_used)
    b_u_used = invert_monotonic(b_u, od_b_for_inversion, od_b_used)

    a_deg = np.mod(a_u_used, 360.0)
    b_deg = np.mod(b_u_used, 360.0)

    od_open_cut = find_od_open_cut(od_b_for_open_cut, s_b, sens_thresh)
    b_deg_snap = quantize_angles_min_step(b_deg, od_b_used, od_open_cut, min_step_deg)

    b_u_snap = map_wrapped_to_unwrapped(b_deg_snap, b_u, wrap_deg=360.0)
    order_b = np.argsort(np.asarray(b_u, dtype=float))
    b_u_sorted = np.asarray(b_u, dtype=float)[order_b]
    od_b_sorted = np.asarray(od_b_for_snap, dtype=float)[order_b]
    od_b_snap = np.interp(b_u_snap, b_u_sorted, od_b_sorted)

    p_ach = p00 * 10 ** (-(od_a_used + od_b_snap))

    return {
        "p_targets": p_targets,
        "od_total": od_total,
        "od_a_used": od_a_used,
        "od_b_used": od_b_used,
        "a_idx": a_idx,
        "a_deg": a_deg,
        "b_deg": b_deg,
        "b_deg_snap": b_deg_snap,
        "od_b_snap": od_b_snap,
        "p_ach": p_ach,
        "od_open_cut": od_open_cut,
    }


def build_region_targets(p_min: float, p_max: float, power_split_frac: float, n1: int, n2_dense: int):
    p_break = p_min + power_split_frac * (p_max - p_min)

    n1 = max(2, int(n1))
    p1 = np.linspace(p_min, p_break, n1, endpoint=False)

    n2_dense = max(10, int(n2_dense))
    od_break = -np.log10(np.clip(p_break / p_max, 1e-12, 1.0))
    od2 = np.linspace(od_break, 0.0, n2_dense, endpoint=True)
    p2 = p_max * (10 ** (-od2))

    return p1, p2, p_break


def filter_by_min_dp_and_pair(mapped: dict, min_dp: float, n_desired: int, pair_mode: str = "AB"):
    if n_desired <= 0:
        return []

    p_ach = mapped["p_ach"]
    a_deg = mapped["a_deg"]
    b_deg_snap = mapped["b_deg_snap"]

    keep = []
    last_p = None
    last_pair = None

    for i in range(len(p_ach)):
        if pair_mode == "BA":
            pair = (float(a_deg[i]), float(b_deg_snap[i]))
        else:
            pair = (float(b_deg_snap[i]), float(a_deg[i]))

        if last_p is None:
            keep.append(i)
            last_p = float(p_ach[i])
            last_pair = pair
            if len(keep) >= n_desired:
                break
            continue

        if pair == last_pair:
            continue
        if abs(float(p_ach[i]) - last_p) < float(min_dp):
            continue

        keep.append(i)
        last_p = float(p_ach[i])
        last_pair = pair

        if len(keep) >= n_desired:
            break

    return keep


class DualWheelIntensityCalibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)
        self.stage_a: Optional[RotationStage] = None
        self.stage_b: Optional[RotationStage] = None

        self._live_thread: Optional[LiveAcqThread] = None
        self._fine_thread: Optional[QtCore.QThread] = None
        self._grid_thread: Optional[QtCore.QThread] = None
        self._stage_controller: Optional[MotionController] = None
        self._stage_busy = False
        self._busy = False
        self._live_active = False

        self._save_dir = None
        self._results: List[dict] = []
        self._fine_a: List[dict] = []
        self._fine_b: List[dict] = []
        self._angles_a: Optional[List[float]] = None
        self._angles_b: Optional[List[float]] = None
        self._grid_data = None
        self._pair_list: Optional[List[Tuple[float, float]]] = None
        self._grid_angles_a: Optional[List[float]] = None
        self._grid_angles_b: Optional[List[float]] = None
        self._grid_map_a = {}
        self._grid_map_b = {}

        self._model_a = None
        self._model_b = None

        self._power_unit = DEFAULT_POWER_UNIT
        self._fit_slope = None
        self._fit_intercept = None
        self._fit_r2 = None
        self._readout_rate_label = str(DEFAULT_READOUT_RATE)
        self._preamp_gain_label = str(DEFAULT_PREAMP_GAIN)
        self._output_amp_label = str(DEFAULT_OUTPUT_AMP)
        self._last_roi_title_s = 0.0
        self._roi_title_interval_s = 1.0
        self._live_title_base = "Andor Image"

        self._crop = (DEFAULT_CROP_TOP, DEFAULT_CROP_BOTTOM, DEFAULT_CROP_LEFT, DEFAULT_CROP_RIGHT)
        self._settle_ms = float(DEFAULT_SETTLE_MS)
        self._use_loaded_pairs = False
        self._ref_every = int(DEFAULT_REF_EVERY)
        self._p00_power = float(DEFAULT_P00_POWER)
        self._od_last_plan = None
        self._od_updating = False
        self._od_slider_block = False
        self._od_plan_dirty = False
        self._coarse_control_active = True
        self._single_grid_idx = 0
        self._fine_a_wrap_start: Optional[float] = None
        self._fine_a_force_wrap: bool = False
        self._fine_b_wrap_start: Optional[float] = None
        self._fine_b_force_wrap: bool = False

        self._build_ui()
        self._update_roi_info()
        self._update_od_planner()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.start(int(POLL_MS))

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self._build_dashboard(vbox)
        self._build_controls(vbox)
        self._build_views(vbox)

        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)
        self.statusBar().showMessage("Idle")

        self._wire_signals()
        self.on_calib_scale_toggle(self.calibLogCheck.isChecked())
        self._update_coarse_control_ui()

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    def _build_dashboard(self, parent_layout):
        box = QtWidgets.QGroupBox("Dashboard")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        self.camDot = self._make_dot()
        self.camStatusLbl = QtWidgets.QLabel("Andor: disconnected")
        self.camStatusLbl.setFont(mono)

        self.exposureSpin = QtWidgets.QDoubleSpinBox()
        self.exposureSpin.setDecimals(3)
        self.exposureSpin.setRange(0.001, 1.0e7)
        self.exposureSpin.setValue(DEFAULT_EXPOSURE_MS)
        self.exposureSpin.setFixedWidth(90)

        self.accumSpin = QtWidgets.QSpinBox()
        self.accumSpin.setRange(1, 100000)
        self.accumSpin.setValue(DEFAULT_ACQ_NUMBER)
        self.accumSpin.setFixedWidth(70)

        self.preampCombo = QtWidgets.QComboBox()
        self.preampCombo.setEditable(False)
        self.preampCombo.addItem(str(DEFAULT_PREAMP_GAIN))
        self.preampCombo.setFixedWidth(70)
        self.preampInfoLbl = QtWidgets.QLabel("Gain: --")
        self.preampInfoLbl.setFont(mono)
        self.readoutInfoLbl = QtWidgets.QLabel("Rate: --")
        self.readoutInfoLbl.setFont(mono)

        line1 = QtWidgets.QHBoxLayout()
        line1.addWidget(self.camDot)
        line1.addWidget(QtWidgets.QLabel("Andor:"))
        line1.addWidget(self.camStatusLbl, 1)
        line1.addStretch()
        line1.addWidget(QtWidgets.QLabel("Exp (ms):"))
        line1.addWidget(self.exposureSpin)
        line1.addWidget(QtWidgets.QLabel("Acq N:"))
        line1.addWidget(self.accumSpin)
        line1.addWidget(QtWidgets.QLabel("Preamp:"))
        line1.addWidget(self.preampCombo)
        line1.addWidget(self.preampInfoLbl)
        line1.addWidget(self.readoutInfoLbl)
        grid.addLayout(line1, 0, 0, 1, 1)

        self.specDot = self._make_dot()
        self.specStatusLbl = QtWidgets.QLabel("Spectrograph: disconnected")
        self.specStatusLbl.setFont(mono)

        self.centerSpin = QtWidgets.QDoubleSpinBox()
        self.centerSpin.setDecimals(3)
        self.centerSpin.setRange(0.0, 20000.0)
        self.centerSpin.setValue(float(DEFAULT_CENTER_WL_NM))
        self.centerSpin.setFixedWidth(90)

        self.slitSpin = QtWidgets.QDoubleSpinBox()
        self.slitSpin.setDecimals(2)
        self.slitSpin.setRange(0.0, 5000.0)
        self.slitSpin.setValue(float(DEFAULT_SLIT_UM))
        self.slitSpin.setFixedWidth(90)

        self.applySpecBtn = QtWidgets.QPushButton("Apply Spec")
        self.applySpecBtn.setFixedWidth(110)

        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.specDot)
        line2.addWidget(QtWidgets.QLabel("Spectrograph:"))
        line2.addWidget(self.specStatusLbl, 1)
        line2.addStretch()
        line2.addWidget(QtWidgets.QLabel("Center (nm):"))
        line2.addWidget(self.centerSpin)
        line2.addWidget(QtWidgets.QLabel("Slit (um):"))
        line2.addWidget(self.slitSpin)
        line2.addWidget(self.applySpecBtn)
        grid.addLayout(line2, 1, 0, 1, 1)

        self.stageADot = self._make_dot()
        self.stageAStatusLbl = QtWidgets.QLabel("Stage A: disconnected")
        self.stageAStatusLbl.setFont(mono)

        self.stageASerialCombo = QtWidgets.QComboBox()
        self.stageASerialCombo.setEditable(True)
        self.stageASerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_A_SERIAL:
            self.stageASerialCombo.addItem(DEFAULT_STAGE_A_SERIAL)

        self.stageADetectBtn = QtWidgets.QPushButton("Search")
        self.stageADetectBtn.setFixedWidth(80)
        self.stageAHomeBtn = QtWidgets.QPushButton("Home A")
        self.stageAHomeBtn.setFixedWidth(90)
        self.stageATargetSpin = QtWidgets.QDoubleSpinBox()
        self.stageATargetSpin.setDecimals(3)
        self.stageATargetSpin.setRange(-360.0, 360.0)
        self.stageATargetSpin.setValue(0.0)
        self.stageATargetSpin.setFixedWidth(90)
        self.stageAMoveBtn = QtWidgets.QPushButton("Move A")
        self.stageAMoveBtn.setFixedWidth(80)

        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.stageADot)
        line2.addWidget(QtWidgets.QLabel("Stage A:"))
        line2.addWidget(self.stageAStatusLbl, 1)
        line2.addStretch()
        line2.addWidget(QtWidgets.QLabel("Serial:"))
        line2.addWidget(self.stageASerialCombo)
        line2.addWidget(self.stageADetectBtn)
        line2.addWidget(self.stageAHomeBtn)
        line2.addWidget(QtWidgets.QLabel("Target (deg):"))
        line2.addWidget(self.stageATargetSpin)
        line2.addWidget(self.stageAMoveBtn)
        grid.addLayout(line2, 2, 0, 1, 1)

        self.stageBDot = self._make_dot()
        self.stageBStatusLbl = QtWidgets.QLabel("Stage B: disconnected")
        self.stageBStatusLbl.setFont(mono)

        self.stageBSerialCombo = QtWidgets.QComboBox()
        self.stageBSerialCombo.setEditable(True)
        self.stageBSerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_B_SERIAL:
            self.stageBSerialCombo.addItem(DEFAULT_STAGE_B_SERIAL)

        self.stageBDetectBtn = QtWidgets.QPushButton("Search")
        self.stageBDetectBtn.setFixedWidth(80)
        self.stageBHomeBtn = QtWidgets.QPushButton("Home B")
        self.stageBHomeBtn.setFixedWidth(90)
        self.stageBTargetSpin = QtWidgets.QDoubleSpinBox()
        self.stageBTargetSpin.setDecimals(3)
        self.stageBTargetSpin.setRange(-360.0, 360.0)
        self.stageBTargetSpin.setValue(0.0)
        self.stageBTargetSpin.setFixedWidth(90)
        self.stageBMoveBtn = QtWidgets.QPushButton("Move B")
        self.stageBMoveBtn.setFixedWidth(80)

        line3 = QtWidgets.QHBoxLayout()
        line3.addWidget(self.stageBDot)
        line3.addWidget(QtWidgets.QLabel("Stage B:"))
        line3.addWidget(self.stageBStatusLbl, 1)
        line3.addStretch()
        line3.addWidget(QtWidgets.QLabel("Serial:"))
        line3.addWidget(self.stageBSerialCombo)
        line3.addWidget(self.stageBDetectBtn)
        line3.addWidget(self.stageBHomeBtn)
        line3.addWidget(QtWidgets.QLabel("Target (deg):"))
        line3.addWidget(self.stageBTargetSpin)
        line3.addWidget(self.stageBMoveBtn)
        grid.addLayout(line3, 3, 0, 1, 1)

        self.initBtn = QtWidgets.QPushButton("Initialize")
        self.initBtn.setFixedWidth(120)
        self.coarseControlCheck = QtWidgets.QCheckBox("Coarse control")
        self.coarseControlCheck.setChecked(True)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setFixedWidth(120)
        self.liveBtn = QtWidgets.QPushButton("Live")
        self.liveBtn.setFixedWidth(90)
        self.stopLiveBtn = QtWidgets.QPushButton("Stop Live")
        self.stopLiveBtn.setFixedWidth(110)
        self.calibABtn = QtWidgets.QPushButton("Calib A")
        self.calibABtn.setFixedWidth(90)
        self.calibBBtn = QtWidgets.QPushButton("Calib B")
        self.calibBBtn.setFixedWidth(90)
        self.gridBtn = QtWidgets.QPushButton("Start Sweep")
        self.gridBtn.setFixedWidth(110)
        self.abortBtn = QtWidgets.QPushButton("Abort")
        self.abortBtn.setFixedWidth(90)

        line4 = QtWidgets.QHBoxLayout()
        line4.addStretch()
        line4.addWidget(self.coarseControlCheck)
        line4.addSpacing(8)
        line4.addWidget(self.initBtn)
        line4.addWidget(self.disconnectBtn)
        line4.addSpacing(10)
        line4.addWidget(self.liveBtn)
        line4.addWidget(self.stopLiveBtn)
        line4.addSpacing(10)
        line4.addWidget(self.calibABtn)
        line4.addWidget(self.calibBBtn)
        line4.addWidget(self.gridBtn)
        line4.addWidget(self.abortBtn)
        grid.addLayout(line4, 4, 0, 1, 1)

        parent_layout.addWidget(box)
    def _build_controls(self, parent_layout):
        setup_box = QtWidgets.QGroupBox("Setup / ROI")
        grid = QtWidgets.QGridLayout(setup_box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.saveNameEdit = QtWidgets.QLineEdit(self._default_save_name())
        self.saveNameEdit.setPlaceholderText("subfolder name under measurements/")
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(110)
        self.exportSessionBtn = QtWidgets.QPushButton("Export Session")
        self.exportSessionBtn.setFixedWidth(130)
        self.loadSessionBtn = QtWidgets.QPushButton("Load Session")
        self.loadSessionBtn.setFixedWidth(120)

        self.loadPathEdit = QtWidgets.QLineEdit("")
        self.loadBrowseBtn = QtWidgets.QPushButton("Browse")
        self.loadBrowseBtn.setFixedWidth(90)
        self.loadBtn = QtWidgets.QPushButton("Load")
        self.loadBtn.setFixedWidth(90)

        self.aMinSpin = QtWidgets.QDoubleSpinBox()
        self.aMaxSpin = QtWidgets.QDoubleSpinBox()
        self.bMinSpin = QtWidgets.QDoubleSpinBox()
        self.bMaxSpin = QtWidgets.QDoubleSpinBox()
        for spin in (self.aMinSpin, self.aMaxSpin, self.bMinSpin, self.bMaxSpin):
            spin.setDecimals(3)
            spin.setRange(-360.0, 360.0)
            spin.setFixedWidth(90)
        self.aMinSpin.setValue(DEFAULT_MIN_ANGLE_A)
        self.aMaxSpin.setValue(DEFAULT_MAX_ANGLE_A)
        self.bMinSpin.setValue(DEFAULT_MIN_ANGLE_B)
        self.bMaxSpin.setValue(DEFAULT_MAX_ANGLE_B)

        self.fineStepSpin = QtWidgets.QDoubleSpinBox()
        self.fineStepSpin.setDecimals(3)
        self.fineStepSpin.setRange(0.001, 360.0)
        self.fineStepSpin.setValue(DEFAULT_FINE_STEP_DEG)
        self.fineStepSpin.setFixedWidth(90)

        self.totalPointsSpin = QtWidgets.QSpinBox()
        self.totalPointsSpin.setRange(int(DEFAULT_TOTAL_POINTS_MIN), int(DEFAULT_TOTAL_POINTS_MAX))
        self.totalPointsSpin.setValue(int(DEFAULT_TOTAL_POINTS))
        self.totalPointsSpin.setFixedWidth(90)

        self.roiX1 = QtWidgets.QSpinBox()
        self.roiX2 = QtWidgets.QSpinBox()
        self.roiY1 = QtWidgets.QSpinBox()
        self.roiY2 = QtWidgets.QSpinBox()
        for spin in (self.roiX1, self.roiX2, self.roiY1, self.roiY2):
            spin.setRange(0, 10000)
        self.roiX1.setValue(DEFAULT_ROI_X1)
        self.roiX2.setValue(DEFAULT_ROI_X2)
        self.roiY1.setValue(DEFAULT_ROI_Y1)
        self.roiY2.setValue(DEFAULT_ROI_Y2)
        self.roiApplyBtn = QtWidgets.QPushButton("Apply ROI")
        self.roiApplyBtn.setFixedWidth(100)
        self.roiInfoLbl = QtWidgets.QLabel("ROI: --")

        self.linearInfoLbl = QtWidgets.QLabel("A sweep: -- | B sweep: -- | Final points: --")
        self.aMinMaxLbl = QtWidgets.QLabel("Wheel A min/max (deg):")
        self.bMinMaxLbl = QtWidgets.QLabel("Wheel B min/max (deg):")

        grid.addWidget(QtWidgets.QLabel("Save subfolder:"), 0, 0)
        grid.addWidget(self.saveNameEdit, 0, 1, 1, 3)
        grid.addWidget(self.exportBtn, 0, 4)
        grid.addWidget(self.exportSessionBtn, 0, 5)
        grid.addWidget(self.loadSessionBtn, 0, 6)

        grid.addWidget(QtWidgets.QLabel("Load calib file:"), 1, 0)
        grid.addWidget(self.loadPathEdit, 1, 1)
        grid.addWidget(self.loadBrowseBtn, 1, 2)
        grid.addWidget(self.loadBtn, 1, 3)

        grid.addWidget(self.aMinMaxLbl, 2, 0)
        grid.addWidget(self.aMinSpin, 2, 1)
        grid.addWidget(self.aMaxSpin, 2, 2)
        grid.addWidget(self.bMinMaxLbl, 2, 3)
        grid.addWidget(self.bMinSpin, 2, 4)
        grid.addWidget(self.bMaxSpin, 2, 5)

        grid.addWidget(QtWidgets.QLabel("Fine step (deg):"), 3, 0)
        grid.addWidget(self.fineStepSpin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Total points (N):"), 3, 2)
        grid.addWidget(self.totalPointsSpin, 3, 3)

        grid.addWidget(QtWidgets.QLabel("ROI x1/x2:"), 4, 0)
        grid.addWidget(self.roiX1, 4, 1)
        grid.addWidget(self.roiX2, 4, 2)
        grid.addWidget(QtWidgets.QLabel("ROI y1/y2:"), 4, 3)
        grid.addWidget(self.roiY1, 4, 4)
        grid.addWidget(self.roiY2, 4, 5)
        grid.addWidget(self.roiApplyBtn, 4, 6)
        grid.addWidget(self.roiInfoLbl, 4, 7)

        grid.addWidget(self.linearInfoLbl, 5, 0, 1, 4)

        parent_layout.addWidget(setup_box)

        fit_box = QtWidgets.QGroupBox("Power Fit (counts to power)")
        hbox = QtWidgets.QHBoxLayout(fit_box)
        hbox.setContentsMargins(8, 8, 8, 8)
        hbox.setSpacing(8)

        left = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(left)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        unit_row = QtWidgets.QHBoxLayout()
        self.powerUnitCombo = QtWidgets.QComboBox()
        self.powerUnitCombo.addItems(["nW", "uW", "mW", "W"])
        self.powerUnitCombo.setCurrentText(self._power_unit)
        self.fitResultLbl = QtWidgets.QLabel("Fit: --")
        unit_row.addWidget(QtWidgets.QLabel("Power unit:"))
        unit_row.addWidget(self.powerUnitCombo)
        unit_row.addStretch()
        unit_row.addWidget(self.fitResultLbl)
        vbox.addLayout(unit_row)

        self.calibLogCheck = QtWidgets.QCheckBox("Log scale (calib)")
        self.calibLogCheck.setChecked(True)
        vbox.addWidget(self.calibLogCheck)

        self.fitTable = QtWidgets.QTableWidget(0, 2)
        self.fitTable.setHorizontalHeaderLabels(["Counts", f"Power ({self._power_unit})"])
        self.fitTable.horizontalHeader().setStretchLastSection(True)
        self.fitTable.verticalHeader().setVisible(False)
        self.fitTable.setFixedWidth(260)
        vbox.addWidget(self.fitTable)

        btn_row = QtWidgets.QHBoxLayout()
        self.fitAddBtn = QtWidgets.QPushButton("Add Row")
        self.fitRemoveBtn = QtWidgets.QPushButton("Remove Row")
        self.fitBtn = QtWidgets.QPushButton("Fit")
        btn_row.addWidget(self.fitAddBtn)
        btn_row.addWidget(self.fitRemoveBtn)
        btn_row.addStretch()
        btn_row.addWidget(self.fitBtn)
        vbox.addLayout(btn_row)

        self.fig_fit = Figure(figsize=(4, 2.5), dpi=100)
        self.canvas_fit = FigureCanvas(self.fig_fit)
        self.ax_fit = self.fig_fit.add_subplot(111)
        self.ax_fit.set_title("Fit")
        self.ax_fit.set_xlabel("Counts")
        self.ax_fit.set_ylabel(f"Power ({self._power_unit})")
        self.fit_points, = self.ax_fit.plot([], [], marker="o", ls="", ms=4, color="tab:blue", zorder=2)
        self.fit_line, = self.ax_fit.plot([], [], ls="-", lw=1.6, color="tab:red", zorder=3)

        self.fig_calib = Figure(figsize=(4, 2.5), dpi=100)
        self.canvas_calib = FigureCanvas(self.fig_calib)
        self.ax_calib = self.fig_calib.add_subplot(111)
        self.ax_calib_b = self.ax_calib.twinx()
        self.ax_calib.set_title("Wheel Calibration")
        self.ax_calib.set_xlabel("Angle (deg)")
        self.ax_calib.set_ylabel("A intensity (counts)")
        self.ax_calib_b.set_ylabel("B intensity (counts)")
        self.calib_a_points, = self.ax_calib.plot(
            [], [], marker="o", ls="", ms=3, color="tab:blue", alpha=0.7, zorder=2
        )
        self.calib_a_fit, = self.ax_calib.plot(
            [], [], ls="-", lw=1.6, color="tab:purple", zorder=3
        )
        self.calib_b_points, = self.ax_calib_b.plot(
            [], [], marker="o", ls="", ms=3, color="tab:orange", alpha=0.7, zorder=2
        )
        self.calib_b_fit, = self.ax_calib_b.plot(
            [], [], ls="-", lw=1.6, color="tab:green", zorder=3
        )

        hbox.addWidget(left, 0)
        hbox.addWidget(self.canvas_fit, 1)
        hbox.addWidget(self.canvas_calib, 1)

        od_box = self._build_od_planner()

        self.calibTabs = QtWidgets.QTabWidget()
        self.calibTabs.addTab(fit_box, "Power Fit")
        self.calibTabs.addTab(od_box, "OD Planner")
        parent_layout.addWidget(self.calibTabs)

    def _build_od_planner(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("OD Planner (A coarse, B fine)")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        control_row = QtWidgets.QHBoxLayout()

        self.p00Spin = QtWidgets.QDoubleSpinBox()
        self.p00Spin.setDecimals(3)
        self.p00Spin.setRange(0.0, 1.0e9)
        self.p00Spin.setValue(float(DEFAULT_P00_POWER))
        self.p00Spin.setFixedWidth(120)
        self.p00UnitLbl = QtWidgets.QLabel(self._power_unit)

        self.coarseASpin = QtWidgets.QSpinBox()
        self.coarseASpin.setRange(2, 200)
        self.coarseASpin.setValue(int(DEFAULT_A_COARSE_STEPS))
        self.coarseASpin.setFixedWidth(80)

        self.minStepSpin = QtWidgets.QDoubleSpinBox()
        self.minStepSpin.setDecimals(3)
        self.minStepSpin.setRange(0.0, 30.0)
        self.minStepSpin.setValue(float(DEFAULT_MIN_STEP_NEAR_OPEN_DEG))
        self.minStepSpin.setFixedWidth(90)

        self.clampCheck = QtWidgets.QCheckBox("Clamp T<=1")
        self.clampCheck.setChecked(bool(DEFAULT_CLAMP_T_TO_1))

        self.applyOdPlanBtn = QtWidgets.QPushButton("Apply OD Plan")
        self.applyOdPlanBtn.setFixedWidth(130)
        self.aCoarseStepsLbl = QtWidgets.QLabel("A coarse steps:")

        control_row.addWidget(QtWidgets.QLabel("P00:"))
        control_row.addWidget(self.p00Spin)
        control_row.addWidget(self.p00UnitLbl)
        control_row.addSpacing(12)
        control_row.addWidget(self.aCoarseStepsLbl)
        control_row.addWidget(self.coarseASpin)
        control_row.addSpacing(12)
        control_row.addWidget(QtWidgets.QLabel("B min step near open (deg):"))
        control_row.addWidget(self.minStepSpin)
        control_row.addSpacing(12)
        control_row.addWidget(self.clampCheck)
        control_row.addSpacing(12)
        control_row.addWidget(self.applyOdPlanBtn)
        control_row.addStretch()

        layout.addLayout(control_row)

        self.fig_od = Figure(figsize=(12, 4.5), dpi=100)
        self.canvas_od = FigureCanvas(self.fig_od)
        self.ax_od1 = self.fig_od.add_subplot(1, 3, 1)
        self.ax_od2 = self.fig_od.add_subplot(1, 3, 2)
        self.ax_od3 = self.fig_od.add_subplot(1, 3, 3)
        self.fig_od.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.30, wspace=0.30)

        self.ax_od1.set_title("Calibration converted to OD (referenced to P00)")
        self.ax_od1.set_xlabel("Angle (deg; A native, B native)")
        self.ax_od1.set_ylabel("Optical Density (OD)")
        self.ax_od1.grid(True, alpha=0.3)

        self.od_line_a_raw, = self.ax_od1.plot([], [], alpha=0.25, lw=1.0, label="Wheel A raw (coarse)")
        self.od_line_b_raw, = self.ax_od1.plot([], [], alpha=0.25, lw=1.0, label="Wheel B raw (fine)")
        self.od_line_a_smooth, = self.ax_od1.plot([], [], lw=2.0, label="Wheel A (used, coarse)")
        self.od_line_b_smooth, = self.ax_od1.plot([], [], lw=2.0, label="Wheel B (used, fine)")
        self.od_good_band_b = self.ax_od1.axhspan(
            float(DEFAULT_GOOD_OD_LOW),
            float(DEFAULT_GOOD_OD_HIGH),
            alpha=0.15,
            label="B good OD window",
        )
        self.od_good_band_a = self.ax_od1.axhspan(
            float(DEFAULT_GOOD_OD_A_LOW),
            float(DEFAULT_GOOD_OD_A_HIGH),
            alpha=0.10,
            color="tab:blue",
            label="A good OD window",
        )
        self.od_open_cut_line = self.ax_od1.axhline(0.0, ls="--", lw=1.5, label="B near-open OD cut")
        self.ax_od1.legend(loc="best")

        self.ax_od2.set_title("OD allocation per coarse step (A fixed; B residual)")
        self.ax_od2.set_xlabel("Optical Density (OD)")
        self.ax_od2.set_ylabel("Coarse step index (A)")
        self.ax_od2.grid(True, alpha=0.3)

        self.od_range_lines = []
        self.od_scat_b = None
        self.od_scat_a_levels = None

        self.ax_od3.set_title("Target vs Achieved")
        self.ax_od3.set_xlabel("Point index")
        self.ax_od3.set_ylabel(f"Power ({self._power_unit})")
        self.ax_od3.grid(True, alpha=0.3)
        self.od_scat_target = self.ax_od3.scatter([], [], s=18, alpha=0.8, label="Target power")
        self.od_scat_ach = self.ax_od3.scatter([], [], s=18, alpha=0.8, label="Achieved power")
        self.ax_od3.legend(loc="best")

        ax_sens = self.fig_od.add_axes([0.10, 0.18, 0.35, 0.03])
        ax_pspl = self.fig_od.add_axes([0.10, 0.14, 0.35, 0.03])
        ax_ptspl = self.fig_od.add_axes([0.10, 0.10, 0.35, 0.03])
        ax_total = self.fig_od.add_axes([0.55, 0.20, 0.40, 0.03])
        ax_good_b = self.fig_od.add_axes([0.55, 0.14, 0.40, 0.04])
        ax_good_a = self.fig_od.add_axes([0.55, 0.09, 0.40, 0.04])
        ax_smooth = self.fig_od.add_axes([0.55, 0.04, 0.19, 0.03])
        ax_mindp_low = self.fig_od.add_axes([0.76, 0.04, 0.19, 0.03])
        ax_mindp_high = self.fig_od.add_axes([0.76, 0.01, 0.19, 0.03])

        self.od_smooth_slider = Slider(ax_smooth, "Smooth pts", 1, 51, valinit=DEFAULT_OD_SMOOTH_PTS, valstep=2)
        self.od_mindp_low_slider = Slider(
            ax_mindp_low, "Min dP low", 0.0, 50.0, valinit=DEFAULT_MIN_DP_LOW
        )
        self.od_mindp_high_slider = Slider(
            ax_mindp_high, "Min dP high", 0.0, 150.0, valinit=DEFAULT_MIN_DP_HIGH
        )
        self.od_sens_slider = Slider(
            ax_sens, "Sens thresh |dOD/dtheta| (B)", 0.0, 1.0, valinit=DEFAULT_SENS_THRESH
        )
        self.od_power_split_slider = Slider(
            ax_pspl, "Power split (region1)", 0.50, 0.95, valinit=DEFAULT_POWER_SPLIT
        )
        self.od_point_split_slider = Slider(
            ax_ptspl, "Point split (region1)", 0.50, 0.98, valinit=DEFAULT_POINT_SPLIT
        )
        self.od_total_points_slider = Slider(
            ax_total,
            "Total points",
            int(DEFAULT_TOTAL_POINTS_MIN),
            int(DEFAULT_TOTAL_POINTS_MAX),
            valinit=int(self.totalPointsSpin.value()),
            valstep=1,
        )
        good_b_init = (float(DEFAULT_GOOD_OD_LOW), float(DEFAULT_GOOD_OD_HIGH))
        self.od_good_slider = RangeSlider(
            ax_good_b,
            "B good OD window",
            0.0,
            max(1.0, float(DEFAULT_GOOD_OD_HIGH)),
            valinit=good_b_init,
        )
        good_a_init = (float(DEFAULT_GOOD_OD_A_LOW), float(DEFAULT_GOOD_OD_A_HIGH))
        self.od_good_a_slider = RangeSlider(
            ax_good_a,
            "A good OD window",
            0.0,
            max(1.0, float(DEFAULT_GOOD_OD_A_HIGH)),
            valinit=good_a_init,
        )

        for slider in [
            self.od_smooth_slider,
            self.od_mindp_low_slider,
            self.od_mindp_high_slider,
            self.od_sens_slider,
            self.od_power_split_slider,
            self.od_point_split_slider,
            self.od_total_points_slider,
            self.od_good_slider,
            self.od_good_a_slider,
        ]:
            slider.on_changed(self._on_od_slider_change)

        self.od_summary_lbl = QtWidgets.QLabel("OD plan: --")
        self.od_summary_lbl.setWordWrap(True)
        layout.addWidget(self.canvas_od, 1)
        layout.addWidget(self.od_summary_lbl)
        return box

    def _build_views(self, parent_layout):
        box = QtWidgets.QGroupBox("Dual Wheel Intensity Calibration")
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.liveView = AndorLiveViewWidget(self.cam, title="Andor Image")
        self.liveView.set_crop(*self._crop)
        self.liveView.set_roi(self.roiX1.value(), self.roiX2.value(), self.roiY1.value(), self.roiY2.value())
        self._live_title_base = self.liveView.title()

        self.fig_map = Figure(figsize=(4.5, 4.5), dpi=100)
        self.canvas_map = FigureCanvas(self.fig_map)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("Sweep Intensity (B x A)")
        self.ax_map.set_xlabel("A index")
        self.ax_map.set_ylabel("B index")
        self.map_artist = self.ax_map.imshow(
            np.zeros((1, 1)),
            cmap="viridis",
            origin="lower",
            aspect="auto",
        )
        self.map_colorbar = self.fig_map.colorbar(self.map_artist, ax=self.ax_map)
        self.map_colorbar.set_label("Intensity (counts)")

        self.fig_power = Figure(figsize=(4.5, 3.0), dpi=100)
        self.canvas_power = FigureCanvas(self.fig_power)
        self.ax_power = self.fig_power.add_subplot(111)
        self.ax_power.set_title("Sorted Powers")
        self.ax_power.set_xlabel("Index")
        self.ax_power.set_ylabel(f"Power ({self._power_unit})")
        self.line_power, = self.ax_power.plot([], [], marker="o", ls="", ms=3)
        self.ax_power_ref = self.ax_power.twinx()
        self.ax_power_ref.set_ylabel("Reference (counts)")
        self.ref_scatter = self.ax_power_ref.scatter([], [], s=16, color="tab:green", alpha=0.6)

        plot_col = QtWidgets.QVBoxLayout()
        plot_col.addWidget(self.canvas_map, 3)
        plot_col.addWidget(self.canvas_power, 2)
        plot_widget = QtWidgets.QWidget()
        plot_widget.setLayout(plot_col)

        self.dataTable = QtWidgets.QTableWidget(0, 5)
        self.dataTable.setHorizontalHeaderLabels(
            ["A deg", "B deg", f"Power ({self._power_unit})", "Intensity", "Base level (counts)"]
        )
        self.dataTable.horizontalHeader().setStretchLastSection(True)
        self.dataTable.verticalHeader().setVisible(False)
        self.dataTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.dataTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dataTable.setMinimumWidth(320)

        self.pointCountLbl = QtWidgets.QLabel("Points in table: 0")
        table_col = QtWidgets.QVBoxLayout()
        table_col.setContentsMargins(0, 0, 0, 0)
        table_col.setSpacing(4)
        table_col.addWidget(self.pointCountLbl)
        table_col.addWidget(self.dataTable, 1)
        table_widget = QtWidgets.QWidget()
        table_widget.setLayout(table_col)

        layout.addWidget(self.liveView, 3)
        layout.addWidget(plot_widget, 2)
        layout.addWidget(table_widget, 1)
        if hasattr(self, "calibTabs") and self.calibTabs is not None:
            self.calibTabs.addTab(box, "Dual Wheel Calib")
        else:
            parent_layout.addWidget(box, 1)

    def _wire_signals(self):
        self.initBtn.clicked.connect(self.on_initialize)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.liveBtn.clicked.connect(self.on_live)
        self.stopLiveBtn.clicked.connect(self.on_stop_live)
        self.calibABtn.clicked.connect(self.on_calib_a)
        self.calibBBtn.clicked.connect(self.on_calib_b)
        self.gridBtn.clicked.connect(self.on_grid)
        self.abortBtn.clicked.connect(self.on_abort)
        self.stageADetectBtn.clicked.connect(self.on_stage_detect)
        self.stageBDetectBtn.clicked.connect(self.on_stage_detect)
        self.stageAHomeBtn.clicked.connect(self.on_stage_home_a)
        self.stageBHomeBtn.clicked.connect(self.on_stage_home_b)
        self.stageAMoveBtn.clicked.connect(self.on_stage_move_a)
        self.stageBMoveBtn.clicked.connect(self.on_stage_move_b)
        self.roiApplyBtn.clicked.connect(self.on_apply_roi)
        self.applySpecBtn.clicked.connect(self.on_apply_spec)
        self.exportBtn.clicked.connect(self.on_export)
        self.loadBrowseBtn.clicked.connect(self.on_browse_load)
        self.loadBtn.clicked.connect(self.on_load)
        self.fitAddBtn.clicked.connect(self.on_fit_add)
        self.fitRemoveBtn.clicked.connect(self.on_fit_remove)
        self.fitBtn.clicked.connect(self.on_fit)
        self.exportSessionBtn.clicked.connect(self.on_export_session)
        self.loadSessionBtn.clicked.connect(self.on_load_session)
        self.calibLogCheck.toggled.connect(self.on_calib_scale_toggle)
        self.preampCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.powerUnitCombo.currentTextChanged.connect(self.on_power_unit_changed)
        self.totalPointsSpin.valueChanged.connect(self._on_total_points_spin_changed)
        self.p00Spin.valueChanged.connect(self._on_od_control_change)
        self.coarseASpin.valueChanged.connect(self._on_od_control_change)
        self.minStepSpin.valueChanged.connect(self._on_od_control_change)
        self.clampCheck.toggled.connect(self._on_od_control_change)
        self.applyOdPlanBtn.clicked.connect(self.on_apply_od_plan)
        self.coarseControlCheck.toggled.connect(self._on_coarse_control_toggled)
    # -----------------
    # Helpers
    # -----------------
    def _coarse_control_enabled(self) -> bool:
        if not hasattr(self, "coarseControlCheck"):
            return True
        return bool(self.coarseControlCheck.isChecked())

    def _coarse_mode_active(self) -> bool:
        return bool(self._coarse_control_active)

    def _a_related_widgets(self) -> List[QtWidgets.QWidget]:
        widgets = [
            self.stageADot,
            self.stageASerialCombo,
            self.stageADetectBtn,
            self.stageAHomeBtn,
            self.stageATargetSpin,
            self.stageAMoveBtn,
            self.stageAStatusLbl,
            self.calibABtn,
            self.aMinSpin,
            self.aMaxSpin,
            self.aMinMaxLbl,
            self.coarseASpin,
            self.aCoarseStepsLbl,
        ]
        return [w for w in widgets if w is not None]

    def _update_coarse_control_ui(self) -> None:
        enable_a = self._coarse_control_enabled()
        for w in self._a_related_widgets():
            w.setEnabled(enable_a)
        self._set_slider_enabled(self.od_power_split_slider, enable_a)
        self._set_slider_enabled(self.od_point_split_slider, enable_a)
        self._set_slider_enabled(self.od_mindp_low_slider, enable_a)
        self._set_slider_enabled(self.od_mindp_high_slider, enable_a)
        self._set_slider_enabled(self.od_good_a_slider, enable_a)
        if self._coarse_mode_active() != enable_a:
            mode_text = "enabled" if enable_a else "disabled"
            self.statusBar().showMessage(f"Coarse control {mode_text} on next Initialize")

    def _on_coarse_control_toggled(self, _checked: bool) -> None:
        self._update_coarse_control_ui()
        self._sync_controls()

    @staticmethod
    def _close_stage(stage: Optional[RotationStage]) -> None:
        if stage is None:
            return
        try:
            stage.close()
        except Exception:
            pass

    @staticmethod
    def _set_slider_enabled(slider, enabled: bool) -> None:
        if slider is None:
            return
        enabled = bool(enabled)
        try:
            slider.set_active(enabled)
        except Exception:
            pass
        alpha = 1.0 if enabled else 0.35
        try:
            slider.ax.patch.set_alpha(alpha)
        except Exception:
            pass
        for attr in ("label", "valtext"):
            try:
                txt = getattr(slider, attr)
                txt.set_alpha(alpha)
            except Exception:
                pass

    def _on_od_control_change(self, *_args) -> None:
        self._od_plan_dirty = True
        self._update_calib_plot()
        self._update_od_planner()

    def _on_od_slider_change(self, _val) -> None:
        if self._od_slider_block:
            return
        self._od_plan_dirty = True
        if self.od_total_points_slider is not None:
            total_val = int(round(float(self.od_total_points_slider.val)))
            if self.totalPointsSpin.value() != total_val:
                self._od_slider_block = True
                self.totalPointsSpin.setValue(total_val)
                self._od_slider_block = False
        self._update_calib_plot()
        self._update_od_planner()

    def _on_total_points_spin_changed(self, value: int) -> None:
        if self._od_slider_block:
            return
        self._od_plan_dirty = True
        if self.od_total_points_slider is not None:
            self._od_slider_block = True
            self.od_total_points_slider.set_val(int(value))
            self._od_slider_block = False
        self._update_od_planner()

    def _update_slider_range(self, slider: Slider, vmin: float, vmax: float, fallback: float) -> float:
        if slider is None:
            return fallback
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        slider.valmin = vmin
        slider.valmax = vmax
        slider.ax.set_xlim(vmin, vmax)
        val = float(slider.val)
        if not (vmin <= val <= vmax):
            val = vmin + 0.2 * (vmax - vmin)
            self._od_slider_block = True
            slider.set_val(val)
            self._od_slider_block = False
        return float(slider.val)

    def _update_range_slider(self, slider: RangeSlider, vmin: float, vmax: float) -> Tuple[float, float]:
        if slider is None:
            return (float(vmin), float(vmax))
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        slider.valmin = vmin
        slider.valmax = vmax
        slider.ax.set_xlim(vmin, vmax)
        lo, hi = map(float, slider.val)
        lo = max(vmin, min(lo, vmax))
        hi = max(lo + 1e-9, min(hi, vmax))
        if (lo, hi) != tuple(slider.val):
            self._od_slider_block = True
            slider.set_val((lo, hi))
            self._od_slider_block = False
        return lo, hi

    def _compute_od_plan(self, total_points: Optional[int] = None) -> Optional[dict]:
        if not self._coarse_mode_active():
            return self._compute_od_plan_single(total_points=total_points)
        if self._model_a is None or self._model_b is None:
            return None
        if self._fit_slope is None or self._fit_intercept is None:
            return None

        try:
            p00 = float(self.p00Spin.value())
        except Exception:
            return None
        if not np.isfinite(p00) or p00 <= 0:
            return None

        slope = float(self._fit_slope)
        intercept = float(self._fit_intercept)

        a_u, a_w, _a_i, a_p = build_wheel_arrays(self._model_a, slope, intercept)
        b_u, b_w, _b_i, b_p = build_wheel_arrays(self._model_b, slope, intercept)
        if a_u.size < 2 or b_u.size < 2:
            return None

        t_a = sanitize_transmittance_for_od(a_p / p00, clamp_to_one=bool(self.clampCheck.isChecked()))
        t_b = sanitize_transmittance_for_od(b_p / p00, clamp_to_one=bool(self.clampCheck.isChecked()))
        if t_a is None or t_b is None:
            return None

        od_a_raw = od_from_transmittance(t_a)
        od_b_raw = od_from_transmittance(t_b)

        if not np.isfinite(od_a_raw).any() or not np.isfinite(od_b_raw).any():
            return None

        od_a_raw_min = float(np.nanmin(od_a_raw))
        od_a_raw_max = float(np.nanmax(od_a_raw))
        od_b_raw_min = float(np.nanmin(od_b_raw))
        od_b_raw_max = float(np.nanmax(od_b_raw))
        if (
            not np.isfinite(od_a_raw_min)
            or not np.isfinite(od_a_raw_max)
            or not np.isfinite(od_b_raw_min)
            or not np.isfinite(od_b_raw_max)
        ):
            return None
        if od_a_raw_max <= 0 or od_b_raw_max <= 0:
            return None

        p_max = float(p00)
        p_min = float(p00) * (10.0 ** (-(od_a_raw_max + od_b_raw_max)))
        if not np.isfinite(p_min) or p_min <= 0:
            return None

        smooth_pts = int(self.od_smooth_slider.val) if self.od_smooth_slider else int(DEFAULT_OD_SMOOTH_PTS)
        min_dp_low = float(self.od_mindp_low_slider.val) if self.od_mindp_low_slider else float(DEFAULT_MIN_DP_LOW)
        min_dp_high = float(self.od_mindp_high_slider.val) if self.od_mindp_high_slider else float(DEFAULT_MIN_DP_HIGH)
        power_split = float(self.od_power_split_slider.val) if self.od_power_split_slider else float(DEFAULT_POWER_SPLIT)
        point_split = float(self.od_point_split_slider.val) if self.od_point_split_slider else float(DEFAULT_POINT_SPLIT)

        total_pts = int(total_points) if total_points is not None else int(self.totalPointsSpin.value())
        total_pts = int(np.clip(total_pts, DEFAULT_TOTAL_POINTS_MIN, DEFAULT_TOTAL_POINTS_MAX))

        _, od_a_mono = smooth_then_monotone(a_u, od_a_raw, smooth_pts)
        _, od_b_mono = smooth_then_monotone(b_u, od_b_raw, smooth_pts)

        od_a_max = float(np.nanmax(od_a_mono))
        od_b_max = float(np.nanmax(od_b_mono))
        if not np.isfinite(od_a_max) or not np.isfinite(od_b_max):
            return None

        s_b = compute_sensitivity(b_u, od_b_mono)
        if not np.isfinite(s_b).any():
            return None
        s_min = float(np.nanmin(s_b))
        s_max = float(np.nanmax(s_b))
        sens_thresh = self._update_slider_range(self.od_sens_slider, s_min, s_max, DEFAULT_SENS_THRESH)

        good_a_lo, good_a_hi = self._update_range_slider(self.od_good_a_slider, od_a_raw_min, od_a_raw_max)
        if good_a_hi < good_a_lo + 1e-9:
            good_a_hi = good_a_lo + 1e-9
        good_a_lo = float(np.clip(good_a_lo, 0.0, od_a_max))
        good_a_hi = float(np.clip(good_a_hi, good_a_lo + 1e-9, od_a_max))

        good_b_lo, good_b_hi = self._update_range_slider(self.od_good_slider, od_b_raw_min, od_b_raw_max)
        if good_b_hi < good_b_lo + 1e-9:
            good_b_hi = good_b_lo + 1e-9

        od_open_cut = find_od_open_cut(od_b_mono, s_b, sens_thresh)

        a_coarse_steps = int(self.coarseASpin.value())
        od_a_levels = np.linspace(0.0, od_a_max, a_coarse_steps)
        a_u_levels = invert_monotonic(a_u, od_a_mono, od_a_levels)
        a_deg_levels = np.mod(a_u_levels, 360.0)

        if total_pts <= 3:
            n1_desired = max(1, total_pts - 1)
        else:
            n1_desired = int(np.clip(round(point_split * total_pts), 2, total_pts - 2))
        n2_desired_initial = max(0, total_pts - n1_desired)

        _, _, p_break = build_region_targets(p_min, p_max, power_split, n1_desired, 200)
        n1_dense = int(np.clip(max(6 * n1_desired, 120), 120, 4000))
        p1_dense = np.linspace(p_min, p_break, n1_dense, endpoint=False)

        n2_dense = int(np.clip(max(8 * n2_desired_initial, 120), 120, 5000))
        _, p2_dense, _ = build_region_targets(p_min, p_max, power_split, 2, n2_dense)

        mapped1 = map_targets_to_wheels_swapped(
            p1_dense,
            p00=p00,
            a_u=a_u,
            a_w=a_w,
            od_a_for_inversion=od_a_mono,
            b_u=b_u,
            b_w=b_w,
            od_b_for_inversion=od_b_mono,
            od_b_for_snap=od_b_mono,
            od_a_levels=od_a_levels,
            od_b_max=od_b_max,
            good_a_lo=good_a_lo,
            good_a_hi=good_a_hi,
            good_b_lo=good_b_lo,
            good_b_hi=good_b_hi,
            sens_thresh=sens_thresh,
            s_b=s_b,
            od_b_for_open_cut=od_b_mono,
            min_step_deg=float(self.minStepSpin.value()),
        )
        keep1 = filter_by_min_dp_and_pair(mapped1, min_dp=min_dp_low, n_desired=n1_desired, pair_mode="AB")
        p1_final = mapped1["p_targets"][keep1]
        n1_final = int(len(p1_final))

        n2_desired = int(max(0, total_pts - n1_final))

        mapped2 = map_targets_to_wheels_swapped(
            p2_dense,
            p00=p00,
            a_u=a_u,
            a_w=a_w,
            od_a_for_inversion=od_a_mono,
            b_u=b_u,
            b_w=b_w,
            od_b_for_inversion=od_b_mono,
            od_b_for_snap=od_b_mono,
            od_a_levels=od_a_levels,
            od_b_max=od_b_max,
            good_a_lo=good_a_lo,
            good_a_hi=good_a_hi,
            good_b_lo=good_b_lo,
            good_b_hi=good_b_hi,
            sens_thresh=sens_thresh,
            s_b=s_b,
            od_b_for_open_cut=od_b_mono,
            min_step_deg=float(self.minStepSpin.value()),
        )
        keep2 = filter_by_min_dp_and_pair(mapped2, min_dp=min_dp_high, n_desired=n2_desired, pair_mode="AB")
        p2_final = mapped2["p_targets"][keep2]
        n2_final = int(len(p2_final))

        if n2_final > 0:
            p_targets = np.concatenate([p1_final, p2_final])
        else:
            p_targets = p1_final

        if p_targets.size == 0:
            return None

        mapped = map_targets_to_wheels_swapped(
            p_targets,
            p00=p00,
            a_u=a_u,
            a_w=a_w,
            od_a_for_inversion=od_a_mono,
            b_u=b_u,
            b_w=b_w,
            od_b_for_inversion=od_b_mono,
            od_b_for_snap=od_b_mono,
            od_a_levels=od_a_levels,
            od_b_max=od_b_max,
            good_a_lo=good_a_lo,
            good_a_hi=good_a_hi,
            good_b_lo=good_b_lo,
            good_b_hi=good_b_hi,
            sens_thresh=sens_thresh,
            s_b=s_b,
            od_b_for_open_cut=od_b_mono,
            min_step_deg=float(self.minStepSpin.value()),
        )

        pairs = [(float(a), float(b)) for a, b in zip(mapped["a_deg"], mapped["b_deg_snap"])]

        plan = {
            "a_u": a_u,
            "b_u": b_u,
            "od_a_raw": od_a_raw,
            "od_b_raw": od_b_raw,
            "od_a_mono": od_a_mono,
            "od_b_mono": od_b_mono,
            "od_a_levels": od_a_levels,
            "a_deg_levels": a_deg_levels,
            "od_b_max": od_b_max,
            "good_a_lo": good_a_lo,
            "good_a_hi": good_a_hi,
            "good_lo": good_b_lo,
            "good_hi": good_b_hi,
            "od_open_cut": od_open_cut,
            "mapped": mapped,
            "p_targets": p_targets,
            "p_ach": mapped["p_ach"],
            "a_idx": mapped["a_idx"],
            "n1_desired": n1_desired,
            "n1_final": n1_final,
            "n2_desired": n2_desired,
            "n2_final": n2_final,
            "total_pts": total_pts,
            "pairs": pairs,
            "p00": p00,
        }

        self._od_last_plan = plan
        return plan

    def _map_targets_to_wheel_b(
        self,
        p_targets: np.ndarray,
        *,
        p00: float,
        b_u: np.ndarray,
        b_w: np.ndarray,
        od_b_for_inversion: np.ndarray,
        od_b_for_snap: np.ndarray,
        od_b_for_open_cut: np.ndarray,
        s_b: np.ndarray,
        sens_thresh: float,
        min_step_deg: float,
    ) -> dict:
        p_targets = np.asarray(p_targets, dtype=float)
        t_targets = np.clip(p_targets / float(p00), 1e-12, 1.0)
        od_targets = od_from_transmittance(t_targets)
        b_u_req = invert_monotonic(b_u, od_b_for_inversion, od_targets)
        b_deg = np.mod(b_u_req, 360.0)

        od_open_cut = find_od_open_cut(od_b_for_open_cut, s_b, sens_thresh)
        b_deg_snap = quantize_angles_min_step(b_deg, od_targets, od_open_cut, min_step_deg)

        b_w = np.asarray(b_w, dtype=float)
        od_b_for_snap = np.asarray(od_b_for_snap, dtype=float)
        b_idx = np.zeros(b_deg_snap.shape, dtype=int)
        od_b_snap = np.zeros(b_deg_snap.shape, dtype=float)
        for i, ang in enumerate(b_deg_snap):
            d = np.abs(((b_w - ang + 180.0) % 360.0) - 180.0)
            j = int(np.argmin(d))
            b_idx[i] = j
            od_b_snap[i] = float(od_b_for_snap[j])

        p_ach = float(p00) * (10.0 ** (-od_b_snap))
        return {
            "p_targets": p_targets,
            "od_b_used": od_targets,
            "a_idx": np.zeros(p_targets.shape, dtype=int),
            "a_deg": np.zeros(p_targets.shape, dtype=float),
            "b_deg": b_deg,
            "b_deg_snap": b_deg_snap,
            "od_b_snap": od_b_snap,
            "p_ach": p_ach,
            "b_idx": b_idx,
            "od_open_cut": od_open_cut,
        }

    def _compute_od_plan_single(self, total_points: Optional[int] = None) -> Optional[dict]:
        if self._model_b is None:
            return None
        if self._fit_slope is None or self._fit_intercept is None:
            return None

        try:
            p00 = float(self.p00Spin.value())
        except Exception:
            return None
        if not np.isfinite(p00) or p00 <= 0:
            return None

        slope = float(self._fit_slope)
        intercept = float(self._fit_intercept)
        b_u, b_w, _b_i, b_p = build_wheel_arrays(self._model_b, slope, intercept)
        if b_u.size < 2:
            return None

        t_b = sanitize_transmittance_for_od(b_p / p00, clamp_to_one=bool(self.clampCheck.isChecked()))
        if t_b is None:
            return None
        od_b_raw = od_from_transmittance(t_b)
        if not np.isfinite(od_b_raw).any():
            return None

        od_b_raw_min = float(np.nanmin(od_b_raw))
        od_b_raw_max = float(np.nanmax(od_b_raw))
        if not np.isfinite(od_b_raw_min) or not np.isfinite(od_b_raw_max) or od_b_raw_max <= 0:
            return None

        p_max = float(p00)
        p_min = float(p00) * (10.0 ** (-od_b_raw_max))
        if not np.isfinite(p_min) or p_min <= 0:
            return None

        smooth_pts = int(self.od_smooth_slider.val) if self.od_smooth_slider else int(DEFAULT_OD_SMOOTH_PTS)
        total_pts = int(total_points) if total_points is not None else int(self.totalPointsSpin.value())
        total_pts = int(np.clip(total_pts, DEFAULT_TOTAL_POINTS_MIN, DEFAULT_TOTAL_POINTS_MAX))

        _, od_b_mono = smooth_then_monotone(b_u, od_b_raw, smooth_pts)
        od_b_max = float(np.nanmax(od_b_mono))
        if not np.isfinite(od_b_max) or od_b_max <= 0:
            return None

        good_lo, good_hi = self._update_range_slider(self.od_good_slider, od_b_raw_min, od_b_raw_max)
        if good_hi < good_lo + 1e-9:
            good_hi = good_lo + 1e-9
        good_lo = float(np.clip(good_lo, od_b_raw_min, od_b_max))
        good_hi = float(np.clip(good_hi, good_lo + 1e-9, od_b_max))

        p_max_win = float(p00) * (10.0 ** (-good_lo))
        p_min_win = float(p00) * (10.0 ** (-good_hi))
        p_targets = np.linspace(p_max_win, p_min_win, total_pts, endpoint=True)
        t_targets = np.clip(p_targets / float(p00), 1e-12, 1.0)
        od_targets = od_from_transmittance(t_targets)
        b_u_req = invert_monotonic(b_u, od_b_mono, od_targets)
        b_deg = np.mod(b_u_req, 360.0)
        min_step = float(self.minStepSpin.value())
        if min_step > 0:
            # Quantize relative to first planned angle so float offsets are preserved.
            anchor = float(b_deg[0])
            b_deg_snap = np.mod(np.round((b_deg - anchor) / min_step) * min_step + anchor, 360.0)
        else:
            b_deg_snap = b_deg
        od_open_cut = od_b_max

        b_u_snap = self._map_to_unwrapped(self._model_b, np.asarray(b_deg_snap, dtype=float))
        od_b_snap = np.interp(b_u_snap, b_u, od_b_mono)
        b_idx = np.searchsorted(b_u, b_u_snap, side="left")
        b_idx = np.clip(b_idx, 0, max(0, len(b_u) - 1)).astype(int)
        p_ach = float(p00) * (10.0 ** (-od_b_snap))

        mapped = {
            "p_targets": p_targets,
            "od_b_used": od_targets,
            "a_idx": np.zeros(p_targets.shape, dtype=int),
            "a_deg": np.zeros(p_targets.shape, dtype=float),
            "b_deg": b_deg,
            "b_deg_snap": b_deg_snap,
            "od_b_snap": od_b_snap,
            "p_ach": p_ach,
            "b_idx": b_idx,
            "od_open_cut": od_open_cut,
        }
        pairs = [(0.0, float(b)) for b in b_deg_snap]

        plan = {
            "single_wheel": True,
            "a_u": np.asarray([], dtype=float),
            "b_u": b_u,
            "od_a_raw": np.asarray([], dtype=float),
            "od_b_raw": od_b_raw,
            "od_a_mono": np.asarray([], dtype=float),
            "od_b_mono": od_b_mono,
            "od_a_levels": np.asarray([0.0], dtype=float),
            "a_deg_levels": np.asarray([0.0], dtype=float),
            "od_b_max": od_b_max,
            "good_lo": good_lo,
            "good_hi": good_hi,
            "od_open_cut": od_open_cut,
            "mapped": mapped,
            "p_targets": p_targets,
            "p_ach": p_ach,
            "a_idx": mapped["a_idx"],
            "n1_desired": total_pts,
            "n1_final": total_pts,
            "n2_desired": 0,
            "n2_final": 0,
            "total_pts": total_pts,
            "pairs": pairs,
            "p00": p00,
        }

        self._od_last_plan = plan
        return plan

    def _clear_od_planner(self) -> None:
        if hasattr(self, "od_line_a_raw"):
            self.od_line_a_raw.set_data([], [])
            self.od_line_b_raw.set_data([], [])
            self.od_line_a_smooth.set_data([], [])
            self.od_line_b_smooth.set_data([], [])
        if hasattr(self, "od_open_cut_line"):
            self.od_open_cut_line.set_ydata([0.0, 0.0])
        if self.od_scat_b is not None:
            self.od_scat_b.set_offsets(np.empty((0, 2)))
        if self.od_scat_a_levels is not None:
            self.od_scat_a_levels.set_offsets(np.empty((0, 2)))
        if hasattr(self, "od_scat_target"):
            self.od_scat_target.set_offsets(np.empty((0, 2)))
            self.od_scat_ach.set_offsets(np.empty((0, 2)))
        self.ax_od3.set_ylabel(f"Power ({self._power_unit})")
        self.ax_od3.set_title("Target vs Achieved")
        if hasattr(self, "od_summary_lbl"):
            self.od_summary_lbl.setText("OD plan: --")
        self.canvas_od.draw_idle()

    def _update_od_planner(self) -> None:
        if self._od_updating:
            return
        self._od_updating = True
        try:
            plan = self._compute_od_plan()
            if plan is None:
                self._clear_od_planner()
                self._od_updating = False
                return

            p00 = float(plan["p00"])
            coarse_active = self._coarse_mode_active()
            self.ax_od1.set_title(f"Calibration converted to OD (referenced to P00 = {p00:.3f} {self._power_unit})")

            if coarse_active:
                self.od_line_a_raw.set_data(plan["a_u"], plan["od_a_raw"])
                self.od_line_a_smooth.set_data(plan["a_u"], plan["od_a_mono"])
            else:
                self.od_line_a_raw.set_data([], [])
                self.od_line_a_smooth.set_data([], [])
            self.od_line_b_raw.set_data(plan["b_u"], plan["od_b_raw"])
            self.od_line_b_smooth.set_data(plan["b_u"], plan["od_b_mono"])

            try:
                self.od_good_band_b.remove()
            except Exception:
                pass
            self.od_good_band_b = self.ax_od1.axhspan(
                plan["good_lo"], plan["good_hi"], alpha=0.15, label="B good OD window"
            )
            try:
                self.od_good_band_a.remove()
            except Exception:
                pass
            if coarse_active:
                self.od_good_band_a = self.ax_od1.axhspan(
                    plan["good_a_lo"],
                    plan["good_a_hi"],
                    alpha=0.10,
                    color="tab:blue",
                    label="A good OD window",
                )
            else:
                self.od_good_band_a = self.ax_od1.axhspan(
                    0.0,
                    0.0,
                    alpha=0.0,
                    color="tab:blue",
                    label="_nolegend_",
                )
            self.od_open_cut_line.set_ydata([plan["od_open_cut"], plan["od_open_cut"]])
            self.ax_od1.legend(loc="best")
            self.ax_od1.relim()
            self.ax_od1.autoscale_view()

            od_b_used = plan["mapped"]["od_b_used"]
            if coarse_active:
                a_steps = int(self.coarseASpin.value())
                a_deg_levels = plan["a_deg_levels"]
                od_a_levels = plan["od_a_levels"]
                a_idx = plan["a_idx"].astype(float)

                if self.od_scat_b is None or len(self.od_range_lines) != a_steps:
                    self.ax_od2.clear()
                    self.ax_od2.set_title("OD allocation per coarse step (A fixed; B residual)")
                    self.ax_od2.set_xlabel("Optical Density (OD)")
                    self.ax_od2.set_ylabel("Coarse step index (A)")
                    self.ax_od2.grid(True, alpha=0.3)

                    self.ax_od2.set_xlim(0.0, plan["od_b_max"])
                    self.ax_od2.set_ylim(-0.5, a_steps - 0.5)
                    self.ax_od2.set_yticks(np.arange(a_steps))
                    self.ax_od2.set_yticklabels([f"{j} (A={a_deg_levels[j]:.1f} deg)" for j in range(a_steps)])

                    self.od_range_lines = []
                    for j in range(a_steps):
                        ln, = self.ax_od2.plot([np.nan, np.nan], [j, j], lw=6, solid_capstyle="butt", zorder=2)
                        self.od_range_lines.append(ln)

                    self.od_scat_b = self.ax_od2.scatter([], [], s=14, alpha=0.55, label="OD_B points (fine)", zorder=6)
                    self.od_scat_a_levels = self.ax_od2.scatter(
                        od_a_levels,
                        np.arange(a_steps),
                        s=70,
                        marker="o",
                        label="OD_A levels (coarse)",
                        zorder=4,
                    )
                    self.ax_od2.legend(loc="best")
                else:
                    self.ax_od2.set_xlim(0.0, plan["od_b_max"])
                    self.ax_od2.set_yticklabels([f"{j} (A={a_deg_levels[j]:.1f} deg)" for j in range(a_steps)])
                    self.od_scat_a_levels.set_offsets(np.column_stack([od_a_levels, np.arange(a_steps)]))

                self.od_scat_b.set_offsets(np.column_stack([od_b_used, a_idx]))
                for j in range(a_steps):
                    mask = plan["mapped"]["a_idx"] == j
                    if np.any(mask):
                        lo = float(np.min(od_b_used[mask]))
                        hi = float(np.max(od_b_used[mask]))
                        self.od_range_lines[j].set_data([lo, hi], [j, j])
                    else:
                        self.od_range_lines[j].set_data([np.nan, np.nan], [j, j])
            else:
                self.ax_od2.clear()
                self.ax_od2.set_title("B-only OD allocation")
                self.ax_od2.set_xlabel("Point index")
                self.ax_od2.set_ylabel("Optical Density (OD)")
                self.ax_od2.grid(True, alpha=0.3)
                idx = np.arange(len(od_b_used))
                self.ax_od2.plot(idx, od_b_used, marker="o", ls="-", ms=3, alpha=0.9)
                self.ax_od2.set_xlim(-1, max(5, len(od_b_used)))
                self.ax_od2.set_ylim(0.0, max(1e-6, float(np.nanmax(od_b_used)) * 1.05))
                self.od_range_lines = []
                self.od_scat_b = None
                self.od_scat_a_levels = None

            p_targets = plan["p_targets"]
            p_ach = plan["p_ach"]
            idx = np.arange(len(p_targets))
            self.od_scat_target.set_offsets(np.column_stack([idx, p_targets]))
            self.od_scat_ach.set_offsets(np.column_stack([idx, p_ach]))

            if len(p_targets) > 0:
                ymin = float(min(np.min(p_targets), np.min(p_ach)))
                ymax = float(max(np.max(p_targets), np.max(p_ach)))
                span = ymax - ymin
                margin = 0.05 * span if span > 1e-9 else max(1.0, 0.05 * ymax)
                self.ax_od3.set_xlim(-1, max(5, len(p_targets)))
                self.ax_od3.set_ylim(max(0.0, ymin - margin), ymax + margin)

            self.ax_od3.set_ylabel(f"Power ({self._power_unit})")
            if hasattr(self, "od_summary_lbl"):
                if coarse_active:
                    self.od_summary_lbl.setText(
                        "OD plan: "
                            f"R1 kept={plan['n1_final']}/{plan['n1_desired']} (min dP low={self.od_mindp_low_slider.val:.2f}) | "
                            f"R2 kept={plan['n2_final']}/{plan['n2_desired']} (min dP high={self.od_mindp_high_slider.val:.2f}) | "
                            f"A window=[{plan['good_a_lo']:.3f}, {plan['good_a_hi']:.3f}] | "
                            f"B window=[{plan['good_lo']:.3f}, {plan['good_hi']:.3f}] | "
                            f"Total={len(p_targets)}/{plan['total_pts']} | Smooth={int(self.od_smooth_slider.val)}"
                        )
                else:
                    self.od_summary_lbl.setText(
                        "OD plan (B only, linear power): "
                        f"Total={len(p_targets)}/{plan['total_pts']} | "
                        f"OD window=[{plan['good_lo']:.3f}, {plan['good_hi']:.3f}] | "
                        f"Smooth={int(self.od_smooth_slider.val)}"
                    )

            self.canvas_od.draw_idle()
        finally:
            self._od_updating = False

    def _ensure_power_fit(self) -> bool:
        if self._fit_slope is not None and self._fit_intercept is not None:
            return True
        if self.fitTable.rowCount() >= 2:
            self.on_fit()
        return self._fit_slope is not None and self._fit_intercept is not None

    def _build_calib_fit_curve(self, model) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if model is None or self._fit_slope is None or self._fit_intercept is None:
            return None
        slope = float(self._fit_slope)
        intercept = float(self._fit_intercept)
        if not np.isfinite(slope) or slope == 0:
            return None
        try:
            p00 = float(self.p00Spin.value())
        except Exception:
            return None
        if not np.isfinite(p00) or p00 <= 0:
            return None
        ang_u = np.asarray(model.get("angles", []), dtype=float)
        intens = np.asarray(model.get("intensities", []), dtype=float)
        if ang_u.size < 2 or intens.size < 2:
            return None
        valid = np.isfinite(ang_u) & np.isfinite(intens)
        if np.count_nonzero(valid) < 2:
            return None
        ang_u = ang_u[valid]
        intens = intens[valid]

        power = power_from_counts(intens, slope, intercept)
        t = power / p00
        # For calibration-fit display, keep high-power points (>P00) instead of flattening.
        # OD planner still uses clamp settings in its own computation path.
        t = np.clip(t, 1e-12, None)
        od_raw = od_from_transmittance(t)

        smooth_pts = int(self.od_smooth_slider.val) if self.od_smooth_slider else int(DEFAULT_OD_SMOOTH_PTS)
        _, od_mono = smooth_then_monotone(ang_u, od_raw, smooth_pts)

        power_fit = p00 * 10.0 ** (-od_mono)
        counts_fit = (power_fit - intercept) / slope
        counts_fit = np.clip(counts_fit, 1e-9, None)

        x = ang_u
        if model.get("wrap_deg", 0.0) > 0:
            x = np.mod(ang_u, 360.0)
            x = x.astype(float, copy=True)
            counts_fit = counts_fit.astype(float, copy=True)
            drop_idx = np.where(np.diff(x) < -1e-6)[0]
            if drop_idx.size:
                cut = int(drop_idx[0] + 1)
                x[cut] = np.nan
                counts_fit[cut] = np.nan
        return x, counts_fit
    def _make_dot(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setFixedSize(10, 10)
        lbl.setStyleSheet("border-radius:5px; background-color:#c62828;")
        return lbl

    def _set_dot(self, lbl: QtWidgets.QLabel, ok: bool) -> None:
        color = "#2e7d32" if ok else "#c62828"
        lbl.setStyleSheet(f"border-radius:5px; background-color:{color};")

    def _default_save_name(self) -> str:
        return time.strftime("dual_wheel_calib_%Y%m%d_%H%M%S")

    def _resolve_save_dir(self) -> str:
        name = self.saveNameEdit.text().strip()
        if not name:
            name = self._default_save_name()
            self.saveNameEdit.setText(name)
        base = os.path.join(DATA_DIR, "measurements", name)
        os.makedirs(base, exist_ok=True)
        self._save_dir = base
        return base

    def _threads_running(self) -> bool:
        live = self._live_thread is not None and self._live_thread.isRunning()
        fine = self._fine_thread is not None and self._fine_thread.isRunning()
        grid = self._grid_thread is not None and self._grid_thread.isRunning()
        return live or fine or grid

    def _is_busy(self) -> bool:
        return self._busy or self._stage_busy or self._threads_running()

    def _sync_controls(self) -> None:
        cam_ok = self.cam.cam is not None
        stage_a_ok = self.stage_a is not None
        stage_b_ok = self.stage_b is not None
        spec_ok = self.cam.spec is not None
        busy = self._is_busy()
        coarse_active = self._coarse_mode_active()
        coarse_ui_enabled = self._coarse_control_enabled()
        has_points = self._model_b is not None
        if coarse_active:
            has_points = has_points and (self._model_a is not None)

        self.initBtn.setEnabled(not busy)
        self.disconnectBtn.setEnabled(not busy)
        self.liveBtn.setEnabled(cam_ok and not busy)
        self.stopLiveBtn.setEnabled(self._threads_running() and self._live_active)
        self.calibABtn.setEnabled(
            coarse_ui_enabled and coarse_active and cam_ok and stage_a_ok and stage_b_ok and not busy
        )
        if coarse_active:
            self.calibBBtn.setEnabled(cam_ok and stage_a_ok and stage_b_ok and not busy)
            self.gridBtn.setEnabled(cam_ok and stage_a_ok and stage_b_ok and has_points and not busy)
        else:
            self.calibBBtn.setEnabled(cam_ok and stage_b_ok and not busy)
            self.gridBtn.setEnabled(cam_ok and stage_b_ok and has_points and not busy)
        self.abortBtn.setEnabled(busy)
        self.stageAHomeBtn.setEnabled(coarse_ui_enabled and stage_a_ok and not busy)
        self.stageBHomeBtn.setEnabled(stage_b_ok and not busy)
        self.stageAMoveBtn.setEnabled(coarse_ui_enabled and stage_a_ok and not busy)
        self.stageBMoveBtn.setEnabled(stage_b_ok and not busy)
        self.applySpecBtn.setEnabled(spec_ok and not busy)
        self.preampCombo.setEnabled(cam_ok and not busy)

    def _roi_tuple(self) -> Optional[Tuple[int, int, int, int]]:
        x1 = int(self.roiX1.value())
        x2 = int(self.roiX2.value())
        y1 = int(self.roiY1.value())
        y2 = int(self.roiY2.value())
        if x1 == x2 or y1 == y2:
            return None
        return (x1, x2, y1, y2)

    def _reference_angles(self) -> Tuple[float, float]:
        a_fallback = 0.5 * (float(self.aMinSpin.value()) + float(self.aMaxSpin.value()))
        b_fallback = 0.5 * (float(self.bMinSpin.value()) + float(self.bMaxSpin.value()))
        a_ref = self._half_power_reference_angle(self._model_a, a_fallback)
        b_ref = self._half_power_reference_angle(self._model_b, b_fallback)
        return (float(a_ref), float(b_ref))

    def _half_power_reference_angle(self, model, fallback: float) -> float:
        try:
            fb = float(fallback)
        except Exception:
            fb = 0.0
        if model is None:
            return fb
        angles = np.asarray(model.get("angles", []), dtype=float)
        counts = np.asarray(model.get("intensities", []), dtype=float)
        valid = np.isfinite(angles) & np.isfinite(counts)
        if np.count_nonzero(valid) < 2:
            return fb
        angles = angles[valid]
        counts = counts[valid]

        values = counts.copy()
        if self._fit_slope is not None and self._fit_intercept is not None:
            powers = np.asarray([self._counts_to_power(v) for v in counts], dtype=float)
            p_valid = np.isfinite(powers)
            if np.count_nonzero(p_valid) >= 2:
                angles = angles[p_valid]
                values = powers[p_valid]

        if values.size < 2:
            return fb
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmax):
            return fb
        target = 0.5 * vmax

        order = np.argsort(values)
        vals_sorted = values[order]
        ang_sorted = angles[order]
        vals_unique, idx_unique = np.unique(vals_sorted, return_index=True)
        if vals_unique.size >= 2 and (vals_unique[-1] - vals_unique[0]) > 1e-12:
            ang_unique = ang_sorted[idx_unique]
            target_c = float(np.clip(target, vals_unique[0], vals_unique[-1]))
            ang_ref_u = float(np.interp(target_c, vals_unique, ang_unique))
        else:
            j = int(np.argmin(np.abs(values - target)))
            ang_ref_u = float(angles[j])

        return float(self._wrap_angle(model, ang_ref_u))

    def _update_roi_info(self):
        roi = self._roi_tuple()
        if roi is None:
            self.roiInfoLbl.setText("ROI: full frame")
            return
        x1, x2, y1, y2 = roi
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        self.roiInfoLbl.setText(f"ROI: {w}x{h}")

    def _apply_camera_defaults(self) -> None:
        try:
            self.cam.set_frame_api("Snap")
            self.cam.set_acquisition_mode("Single")
            self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            self.cam.set_fastest_readout_default()
        except Exception:
            pass
        try:
            if andor_cfg is not None:
                self.cam.set_temperature_setpoint(float(andor_cfg.DEFAULT_SETPOINT_C))
                self.cam.set_cooler(bool(andor_cfg.DEFAULT_COOLER_ON))
                self.cam.set_baseline_clamp(bool(andor_cfg.DEFAULT_BASELINE_CLAMP))
        except Exception:
            pass

    def _refresh_amp_choices(self) -> None:
        if self.cam is None or self.cam.cam is None:
            return
        current = self.preampCombo.currentText().strip()
        gains = []
        rates = []
        try:
            _, rates, gains = self.cam.get_amp_mode_choices()
        except Exception:
            gains = []
            rates = []
        if rates and str(DEFAULT_READOUT_RATE) in rates:
            self._readout_rate_label = str(DEFAULT_READOUT_RATE)
        if not gains:
            return
        self.preampCombo.blockSignals(True)
        self.preampCombo.clear()
        self.preampCombo.addItems(gains)
        if str(DEFAULT_PREAMP_GAIN) in gains:
            self.preampCombo.setCurrentText(str(DEFAULT_PREAMP_GAIN))
        elif current and current in gains:
            self.preampCombo.setCurrentText(current)
        else:
            self.preampCombo.setCurrentIndex(0)
        self.preampCombo.blockSignals(False)

    def _apply_amp_settings(self) -> None:
        if self.cam is None or self.cam.cam is None:
            return
        req_gain = self.preampCombo.currentText().strip()
        req_rate = self._readout_rate_label
        req_amp = self._output_amp_label

        def _gain_value(label: str):
            m = re.search(r"(\\d+(?:\\.\\d+)?)", str(label))
            if m:
                return float(m.group(1))
            return None

        want_gain = _gain_value(req_gain)
        ok = True
        try:
            ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
        except Exception as exc:
            ok = False
            self.statusBar().showMessage(f"Preamp set failed: {exc}")

        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None

        if want_gain is not None:
            cur_gain = None
            if info is not None:
                cur_gain = info.get("preamp_gain")
                if cur_gain is None:
                    cur_gain = _gain_value(info.get("gain_label"))
            if cur_gain is None or abs(float(cur_gain) - float(want_gain)) > 0.05:
                try:
                    ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
                except Exception:
                    ok = False
                try:
                    info = self.cam.get_amp_mode_info()
                except Exception:
                    info = None

        if info:
            self._readout_rate_label = info.get("rate_label") or self._readout_rate_label
            self._preamp_gain_label = info.get("gain_label") or self._preamp_gain_label
            self._output_amp_label = info.get("amp_label") or self._output_amp_label
            gain_label = info.get("gain_label")
            if gain_label:
                self.preampCombo.blockSignals(True)
                self.preampCombo.setCurrentText(str(gain_label))
                self.preampCombo.blockSignals(False)

        self._update_preamp_label()
        self._update_readout_label()
        if not ok:
            self.statusBar().showMessage("Preamp gain not applied")

    def _update_preamp_label(self) -> None:
        if self.cam is None or self.cam.cam is None:
            self.preampInfoLbl.setText("Gain: --")
            self.readoutInfoLbl.setText("Rate: --")
            return
        label = self._preamp_gain_label or self.preampCombo.currentText().strip()
        self.preampInfoLbl.setText(f"Gain: {label}")
        self._update_readout_label()

    def _update_readout_label(self) -> None:
        label = (self._readout_rate_label or "").strip()
        if not label:
            label = "--"
        self.readoutInfoLbl.setText(f"Rate: {label}")

    def _apply_spec_settings(self) -> None:
        if self.cam.spec is None:
            return
        try:
            self.cam.set_center_wavelength_nm(float(self.centerSpin.value()))
        except Exception:
            pass
        try:
            self.cam.spec_set_slit_width_um(DEFAULT_SLIT_ID, float(self.slitSpin.value()))
        except Exception:
            pass
        if self.liveView is not None:
            center = float(self.centerSpin.value())
            if abs(center) <= 1e-9:
                self.liveView.set_wavelength_axis_enabled(False)
            else:
                self.liveView.set_wavelength_axis_enabled(True)

    def _angle_range(self, stage: Optional[RotationStage]) -> Tuple[float, float]:
        if stage is not None and hasattr(stage, "angle_range"):
            lo, hi = stage.angle_range
            return float(lo), float(hi)
        return float(ROT_RANGE_DEFAULT[0]), float(ROT_RANGE_DEFAULT[1])

    def _gen_points(self, start: float, stop: float, step: float, stage: Optional[RotationStage]) -> List[float]:
        start = float(start)
        stop = float(stop)
        step = float(step)
        if step == 0:
            return []
        lo, hi = self._angle_range(stage)
        if lo > hi:
            lo, hi = hi, lo
        width = hi - lo
        start = max(lo, min(hi, start))
        stop = max(lo, min(hi, stop))
        wrap = width >= 360.0 - 1e-6 and start > stop

        step_mag = abs(step)
        if step_mag == 0:
            return []

        def _forward(a: float, b: float) -> List[float]:
            pts: List[float] = []
            x = a
            while x <= b + 1e-9:
                pts.append(x)
                x += step_mag
            return pts

        def _backward(a: float, b: float) -> List[float]:
            pts: List[float] = []
            x = a
            while x >= b - 1e-9:
                pts.append(x)
                x -= step_mag
            return pts

        if not wrap and start <= stop:
            return _forward(start, stop)
        if not wrap and start > stop:
            return _backward(start, stop)

        if start <= stop:
            return _forward(start, stop)
        pts = _forward(start, hi)
        pts += _forward(lo, stop)
        return pts

    def _build_wheel_model(
        self,
        angles: List[float],
        intensities: List[float],
        *,
        wrap_start_deg: Optional[float] = None,
        force_wrap: bool = False,
    ):
        if not angles or not intensities or len(angles) != len(intensities):
            return None
        arr_a_raw = np.asarray(angles, dtype=float)
        arr_i = np.asarray(intensities, dtype=float)
        valid = np.isfinite(arr_a_raw) & np.isfinite(arr_i)
        arr_a_raw = arr_a_raw[valid]
        arr_i = arr_i[valid]
        if arr_a_raw.size < 2:
            return None

        wrap_deg = 360.0
        order = np.argsort(arr_a_raw)
        arr_a_sorted = arr_a_raw[order]
        arr_i_sorted = arr_i[order]
        diffs = np.diff(arr_a_sorted)
        pos_diffs = diffs[diffs > 0]
        median_step = float(np.median(pos_diffs)) if pos_diffs.size else 0.0
        circ_gap = (float(arr_a_sorted[0]) + wrap_deg) - float(arr_a_sorted[-1])
        diffs_all = np.concatenate([diffs, [circ_gap]])
        max_gap_idx = int(np.argmax(diffs_all))
        max_gap = float(diffs_all[max_gap_idx])
        wrap_detected = False
        if arr_a_sorted.size >= 3:
            angle_span = float(arr_a_sorted[-1] - arr_a_sorted[0])
            gap_thresh = 0.45 * angle_span if angle_span > 0 else 0.0
            if median_step > 0:
                gap_thresh = max(gap_thresh, 3.0 * median_step)
            if max_gap >= max(gap_thresh, 1e-6):
                wrap_detected = True
        if force_wrap:
            wrap_detected = True

        if wrap_detected:
            cut = (max_gap_idx + 1) % arr_a_sorted.size
            if wrap_start_deg is not None:
                start_mod = float(wrap_start_deg) % wrap_deg
                d = np.abs(((arr_a_sorted - start_mod + 180.0) % wrap_deg) - 180.0)
                cut = int(np.argmin(d))
            arr_a_wrapped = np.concatenate([arr_a_sorted[cut:], arr_a_sorted[:cut]])
            arr_i = np.concatenate([arr_i_sorted[cut:], arr_i_sorted[:cut]])
            arr_a = arr_a_wrapped.astype(float)
            if cut > 0:
                arr_a[-cut:] += wrap_deg
            arr_a_raw = arr_a_wrapped
        else:
            arr_a = arr_a_sorted.astype(float)
            arr_i = arr_i_sorted
            arr_a_raw = arr_a_sorted

        unique_angles, inv = np.unique(arr_a, return_inverse=True)
        if unique_angles.size < arr_a.size:
            sums = np.zeros(unique_angles.shape, dtype=float)
            counts = np.zeros(unique_angles.shape, dtype=float)
            np.add.at(sums, inv, arr_i)
            np.add.at(counts, inv, 1.0)
            arr_a = unique_angles
            arr_i = sums / np.clip(counts, 1.0, None)
            if wrap_detected:
                arr_a_raw = arr_a % wrap_deg
            else:
                arr_a_raw = arr_a.copy()
        return {
            "angles": arr_a,
            "angles_wrapped": arr_a_raw,
            "intensities": arr_i,
            "angle_lo": float(arr_a.min()),
            "angle_hi": float(arr_a.max()),
            "wrap_deg": wrap_deg if wrap_detected else 0.0,
        }

    def _map_to_unwrapped(self, model, angles: np.ndarray) -> np.ndarray:
        arr = np.asarray(angles, dtype=float)
        wrap = float(model.get("wrap_deg", 0.0) or 0.0)
        if wrap <= 0:
            return arr
        lo = float(model.get("angle_lo", arr.min()))
        hi = float(model.get("angle_hi", arr.max()))
        if hi < lo:
            return arr
        adj = ((arr - lo) % wrap) + lo
        adj = np.where(adj > hi, adj - wrap, adj)
        return adj

    def _wrap_angle(self, model, angle: float) -> float:
        wrap = float(model.get("wrap_deg", 0.0) or 0.0)
        if wrap <= 0:
            return float(angle)
        return float(angle % wrap)

    def _eval_transmission(self, model, angles: np.ndarray) -> np.ndarray:
        arr = np.asarray(angles, dtype=float)
        if arr.size == 0:
            return np.array([], dtype=float)
        arr_u = self._map_to_unwrapped(model, arr)
        interp = model.get("t_log_interp")
        if interp is not None:
            log_t = np.asarray(interp(arr_u), dtype=float)
        else:
            log_t = np.interp(arr_u, model["angles"], model["t_log"])
        t = np.power(10.0, log_t)
        t = np.clip(t, 0.0, 1.0)
        min_angle = float(model.get("min_angle_unwrapped", arr_u.min()))
        max_angle = float(model.get("max_angle_unwrapped", arr_u.max()))
        t = np.where(np.isclose(arr_u, min_angle, atol=1e-6), 0.0, t)
        t = np.where(np.isclose(arr_u, max_angle, atol=1e-6), 1.0, t)
        return t

    def _angle_for_transmission(self, model, t_req: np.ndarray) -> np.ndarray:
        inv_t = model.get("inv_t")
        inv_angles = model.get("inv_angles")
        if inv_t is None or inv_angles is None or len(inv_t) < 2:
            base = float(model.get("min_angle", 0.0))
            return np.full_like(np.asarray(t_req, dtype=float), base, dtype=float)
        t_arr = np.asarray(t_req, dtype=float)
        t_min = float(inv_t[0])
        t_max = float(inv_t[-1])
        t_clipped = np.clip(t_arr, t_min, t_max)
        return np.interp(t_clipped, inv_t, inv_angles)

    @staticmethod
    def _second_half_penalty(angles: np.ndarray, angle_lo: float, angle_hi: float) -> np.ndarray:
        if angle_hi <= angle_lo:
            return np.zeros_like(angles, dtype=float)
        mid = angle_lo + 0.5 * (angle_hi - angle_lo)
        half = max(1e-9, 0.5 * (angle_hi - angle_lo))
        excess = np.maximum(0.0, (angles - mid) / half)
        return excess ** 2

    @staticmethod
    def _biased_targets(count: int, power_frac: float, point_frac: float) -> List[float]:
        if count <= 0:
            return []
        power_frac = float(power_frac)
        point_frac = float(point_frac)
        power_frac = min(max(power_frac, 1e-6), 1.0 - 1e-6)
        point_frac = min(max(point_frac, 0.0), 1.0)
        low_count = int(round(count * point_frac))
        high_count = max(0, count - low_count)
        targets = []
        if low_count > 0:
            low = np.linspace(0.0, power_frac, low_count + 1, endpoint=True)[1:]
            targets.extend(float(x) for x in low)
        if high_count > 0:
            high = np.linspace(power_frac, 1.0, high_count + 1, endpoint=False)[1:]
            targets.extend(float(x) for x in high)
        return targets

    def _estimate_p00_power(self) -> Optional[float]:
        if self._fit_slope is None or self._fit_intercept is None:
            return None
        powers = []
        for model in (self._model_a, self._model_b):
            if model is None:
                continue
            i_max = model.get("i_max")
            if i_max is None:
                intens = model.get("intensities")
                if intens is None:
                    continue
                arr = np.asarray(intens, dtype=float)
                if arr.size == 0:
                    continue
                i_max = float(np.nanmax(arr))
            power = self._counts_to_power(i_max)
            if np.isfinite(power):
                powers.append(float(power))
        if not powers:
            return None
        p00 = float(np.max(powers))
        if not np.isfinite(p00) or p00 <= 0:
            return None
        return p00

    def _build_od_curve(self, model, p00_power: float):
        if model is None or p00_power is None:
            return None
        angles = np.asarray(model.get("angles", []), dtype=float)
        intens = np.asarray(model.get("intensities", []), dtype=float)
        if angles.size < 2 or intens.size < 2:
            return None
        valid = np.isfinite(angles) & np.isfinite(intens)
        if np.count_nonzero(valid) < 2:
            return None
        angles = angles[valid]
        intens = intens[valid]
        slope = float(self._fit_slope)
        intercept = float(self._fit_intercept)
        power = slope * intens + intercept
        if not np.any(np.isfinite(power)):
            return None
        t = power / float(p00_power)
        t = np.clip(t, 1e-12, 1.0)
        od = -np.log10(t)
        return angles, od

    @staticmethod
    def _interp_angle_from_od(od_levels: np.ndarray, od_curve: np.ndarray, angles: np.ndarray):
        od_curve = np.asarray(od_curve, dtype=float)
        angles = np.asarray(angles, dtype=float)
        valid = np.isfinite(od_curve) & np.isfinite(angles)
        if np.count_nonzero(valid) < 2:
            return None
        od_curve = od_curve[valid]
        angles = angles[valid]
        order = np.argsort(od_curve)
        od_sorted = od_curve[order]
        ang_sorted = angles[order]
        od_unique, idx = np.unique(od_sorted, return_index=True)
        if od_unique.size < 2:
            return None
        ang_unique = ang_sorted[idx]
        return np.interp(np.asarray(od_levels, dtype=float), od_unique, ang_unique)

    @staticmethod
    def _allocate_od_levels(od_total: np.ndarray, od_a_max: float, od_b_levels: np.ndarray):
        od_total = np.asarray(od_total, dtype=float)
        od_a = np.zeros_like(od_total, dtype=float)
        od_b = np.zeros_like(od_total, dtype=float)
        b_idx = np.zeros_like(od_total, dtype=int)
        for i, od_tot in enumerate(od_total):
            lo = od_tot - od_a_max
            hi = od_tot
            feasible = np.where((od_b_levels >= lo) & (od_b_levels <= hi))[0]
            if feasible.size == 0:
                j = int(np.clip(np.searchsorted(od_b_levels, od_tot) - 1, 0, len(od_b_levels) - 1))
            else:
                j = int(feasible.max())
            od_b_val = float(od_b_levels[j])
            od_a_val = float(np.clip(od_tot - od_b_val, 0.0, od_a_max))
            b_idx[i] = j
            od_a[i] = od_a_val
            od_b[i] = od_b_val
        return od_a, od_b, b_idx

    def _select_pairs_od(self, total_points: int):
        p00 = self._estimate_p00_power()
        if p00 is None:
            return []
        a_curve = self._build_od_curve(self._model_a, p00)
        b_curve = self._build_od_curve(self._model_b, p00)
        if a_curve is None or b_curve is None:
            return []
        a_angles, od_a = a_curve
        b_angles, od_b = b_curve
        od_a_max = float(np.nanmax(od_a))
        od_b_max = float(np.nanmax(od_b))
        if not np.isfinite(od_a_max) or not np.isfinite(od_b_max):
            return []
        if od_a_max <= 0 or od_b_max <= 0:
            return []
        coarse_steps = int(self.coarseASpin.value())
        if coarse_steps < 2:
            return []
        od_b_levels = np.linspace(0.0, od_b_max, coarse_steps)
        b_u_levels = self._interp_angle_from_od(od_b_levels, od_b, b_angles)
        if b_u_levels is None:
            return []

        p_max = float(p00)
        p_min = float(p00) * (10.0 ** (-(od_a_max + od_b_max)))
        p_targets = np.linspace(p_max, p_min, int(total_points))
        t_targets = np.clip(p_targets / float(p00), 1e-12, 1.0)
        od_total = -np.log10(t_targets)
        od_a_used, _, b_idx = self._allocate_od_levels(od_total, od_a_max, od_b_levels)
        a_u_used = self._interp_angle_from_od(od_a_used, od_a, a_angles)
        if a_u_used is None:
            return []

        b_u_used = b_u_levels[np.clip(b_idx, 0, len(b_u_levels) - 1)]
        pairs = []
        for a_u, b_u in zip(a_u_used, b_u_used):
            pairs.append((
                self._wrap_angle(self._model_a, float(a_u)),
                self._wrap_angle(self._model_b, float(b_u)),
            ))
        return pairs

    def _select_pairs_continuous(self, targets: List[float], penalty_weight: float, eps: float):
        if not targets:
            return []
        grid_n = max(200, int(len(self._model_a["angles"]) * 4))
        a_grid = np.linspace(self._model_a["angle_lo"], self._model_a["angle_hi"], grid_n)
        t_a = self._eval_transmission(self._model_a, a_grid)
        if self._model_a.get("direction") == "increasing":
            t_a = np.maximum.accumulate(t_a)
        else:
            t_a = np.minimum.accumulate(t_a)

        penalty_a = self._second_half_penalty(a_grid, self._model_a["angle_lo"], self._model_a["angle_hi"])

        selected_pairs = []
        for target in targets:
            t_req = float(target)
            t_b_req = t_req / np.maximum(t_a, eps)
            t_b_clipped = np.clip(t_b_req, 0.0, 1.0)
            b_angles = self._angle_for_transmission(self._model_b, t_b_clipped)
            t_b_actual = t_b_clipped
            product = t_a * t_b_actual
            error = np.abs(product - t_req)
            penalty_b = self._second_half_penalty(
                b_angles, self._model_b["angle_lo"], self._model_b["angle_hi"]
            )
            scores = error + penalty_weight * (penalty_a + penalty_b)
            idx = int(np.argmin(scores))
            pair = (
                self._wrap_angle(self._model_a, float(a_grid[idx])),
                self._wrap_angle(self._model_b, float(b_angles[idx])),
            )
            selected_pairs.append(pair)
        return selected_pairs

    def _select_pairs_coarse_b(self, targets: List[float], penalty_weight: float, eps: float, coarse_steps: int):
        if not targets:
            return []
        b_levels = np.linspace(0.0, 1.0, int(coarse_steps))
        b_angles = self._angle_for_transmission(self._model_b, b_levels)
        b_t_actual = self._eval_transmission(self._model_b, b_angles)
        unique = {}
        for angle, t_val in zip(b_angles, b_t_actual):
            key = round(float(angle), 6)
            if key in unique:
                continue
            unique[key] = (float(angle), float(t_val))
        if len(unique) < 2:
            return self._select_pairs_continuous(targets, penalty_weight, eps)
        b_angles = np.array([v[0] for v in unique.values()], dtype=float)
        b_t_actual = np.array([v[1] for v in unique.values()], dtype=float)
        order = np.argsort(b_t_actual)
        b_angles = b_angles[order]
        b_t_actual = b_t_actual[order]

        targets_arr = np.asarray(targets, dtype=float)
        targets_arr = targets_arr[np.isfinite(targets_arr)]
        if targets_arr.size == 0:
            return []
        targets_arr.sort()

        grid_n = max(200, int(len(self._model_a["angles"]) * 4))
        a_lo = float(self._model_a["angle_lo"])
        a_hi = float(self._model_a["angle_hi"])
        if a_hi < a_lo:
            a_lo, a_hi = a_hi, a_lo
        a_grid = np.linspace(a_lo, a_hi, grid_n)
        t_a = self._eval_transmission(self._model_a, a_grid)
        if self._model_a.get("direction") == "increasing":
            t_a = np.maximum.accumulate(t_a)
        else:
            t_a = np.minimum.accumulate(t_a)

        # If the upper half of A is nearly flat, avoid it to keep power spacing linear.
        half_idx = grid_n // 2
        full_span = float(np.nanmax(t_a) - np.nanmin(t_a))
        upper_span = float(np.nanmax(t_a[half_idx:]) - np.nanmin(t_a[half_idx:]))
        if full_span > 0 and upper_span < 0.2 * full_span:
            mid = a_lo + 0.5 * (a_hi - a_lo)
            a_grid = np.linspace(a_lo, mid, grid_n)
            t_a = self._eval_transmission(self._model_a, a_grid)
            if self._model_a.get("direction") == "increasing":
                t_a = np.maximum.accumulate(t_a)
            else:
                t_a = np.minimum.accumulate(t_a)

        penalty_a = self._second_half_penalty(a_grid, self._model_a["angle_lo"], self._model_a["angle_hi"])

        selected_pairs = []
        start = 0
        for idx, b_t in enumerate(b_t_actual):
            end = int(np.searchsorted(targets_arr, b_t, side="right"))
            count = max(0, end - start)
            if count <= 0:
                continue
            lo = 0.0 if idx == 0 else float(b_t_actual[idx - 1])
            hi = float(b_t)
            if hi <= lo:
                start = end
                continue
            sub_targets = np.linspace(lo, hi, count + 1, endpoint=True)[1:]
            b_angle = float(b_angles[idx])
            t_total = t_a * float(b_t)
            used = np.zeros(a_grid.shape, dtype=bool)
            used_count = 0
            for t_req in sub_targets:
                if used_count >= used.size:
                    used[:] = False
                    used_count = 0
                scores = np.abs(t_total - float(t_req))
                if penalty_weight > 0:
                    scores = scores + penalty_weight * penalty_a
                scores = np.where(used, np.inf, scores)
                best = int(np.argmin(scores))
                if not np.isfinite(scores[best]):
                    break
                used[best] = True
                used_count += 1
                selected_pairs.append((
                    self._wrap_angle(self._model_a, float(a_grid[best])),
                    self._wrap_angle(self._model_b, b_angle),
                ))
            start = end
        return selected_pairs

    def _select_weighted_pairs(self, total_points: int):
        plan = self._compute_od_plan(total_points=total_points)
        if plan is None:
            return []
        return list(plan.get("pairs") or [])

    @staticmethod
    def _path_cost(pairs: List[Tuple[float, float]]) -> float:
        if len(pairs) < 2:
            return 0.0
        cost = 0.0
        last_a, last_b = pairs[0]
        for a, b in pairs[1:]:
            cost += abs(a - last_a) + abs(b - last_b)
            last_a, last_b = a, b
        return float(cost)

    @staticmethod
    def _zigzag_pairs(pairs: List[Tuple[float, float]], axis: int) -> List[Tuple[float, float]]:
        if not pairs:
            return []
        groups = {}
        for a, b in pairs:
            key = round(a, 6) if axis == 0 else round(b, 6)
            if key not in groups:
                groups[key] = {"a": a, "b": b, "pairs": []}
            groups[key]["pairs"].append((a, b))

        keys = sorted(groups.keys())
        ordered = []
        for idx, key in enumerate(keys):
            chunk = groups[key]["pairs"]
            if axis == 0:
                chunk_sorted = sorted(chunk, key=lambda x: x[1])
            else:
                chunk_sorted = sorted(chunk, key=lambda x: x[0])
            if idx % 2 == 1:
                chunk_sorted.reverse()
            ordered.extend(chunk_sorted)
        return ordered

    def _order_pairs(self, pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(pairs) < 2:
            return pairs
        try:
            coarse_steps = int(self.coarseASpin.value())
        except Exception:
            coarse_steps = 0
        if coarse_steps >= 2:
            return self._zigzag_pairs(pairs, axis=1)
        order_a = self._zigzag_pairs(pairs, axis=0)
        order_b = self._zigzag_pairs(pairs, axis=1)
        if self._path_cost(order_b) < self._path_cost(order_a):
            return order_b
        return order_a

    def _select_linear_angles(self, angles: List[float], intensities: List[float], count: int):
        if count <= 0 or not angles or not intensities:
            return [], [], []
        if len(angles) != len(intensities):
            return [], [], []
        arr_i = np.asarray(intensities, dtype=float)
        arr_a = np.asarray(angles, dtype=float)
        order = np.argsort(arr_i)
        arr_i = arr_i[order]
        arr_a = arr_a[order]
        targets = np.linspace(arr_i[0], arr_i[-1], int(count))
        available = set(range(arr_i.size))
        picked_angles = []
        picked_intens = []
        errors = []
        for target in targets:
            if not available:
                break
            best = min(available, key=lambda i: abs(arr_i[i] - target))
            available.remove(best)
            picked_angles.append(float(arr_a[best]))
            picked_intens.append(float(arr_i[best]))
            errors.append(float(abs(arr_i[best] - target)))
        return picked_angles, picked_intens, errors

    def _counts_to_power(self, counts: float) -> float:
        if self._fit_slope is None or self._fit_intercept is None:
            return float("nan")
        if counts is None:
            return float("nan")
        try:
            val = float(counts)
        except Exception:
            return float("nan")
        if not np.isfinite(val):
            return float("nan")
        return float(self._fit_slope * val + self._fit_intercept)

    @staticmethod
    def _smooth_transmission(t: np.ndarray, direction: str, eps: float) -> np.ndarray:
        if t is None:
            return t
        arr = np.asarray(t, dtype=float)
        n = arr.size
        if n < 5:
            return arr
        window = max(3, min(9, 2 * (n // 5) + 1))
        if window <= 2:
            return arr
        kernel = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        t_log = np.log10(np.clip(arr, eps, 1.0))
        padded = np.pad(t_log, (pad, pad), mode="edge")
        smooth_log = np.convolve(padded, kernel, mode="valid")
        smooth = np.power(10.0, smooth_log)
        smooth = np.clip(smooth, 0.0, 1.0)
        if direction == "increasing":
            smooth = np.maximum.accumulate(smooth)
        else:
            smooth = np.minimum.accumulate(smooth)
        return smooth

    @staticmethod
    def _roi_sum(img, roi: Optional[Tuple[int, int, int, int]]) -> float:
        if img is None:
            return 0.0
        if roi is None:
            return float(np.nansum(img))
        x1, x2, y1, y2 = roi
        try:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
        except Exception:
            return float(np.nansum(img))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        h, w = img.shape
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return float(np.nansum(img))
        return float(np.nansum(img[y1:y2, x1:x2]))

    def _update_linear_info(self) -> None:
        a_n = len(self._angles_a) if self._angles_a else 0
        b_n = len(self._angles_b) if self._angles_b else 0
        p_n = len(self._pair_list) if self._pair_list else 0
        if self._coarse_mode_active():
            self.linearInfoLbl.setText(f"A sweep: {a_n} | B sweep: {b_n} | Final points: {p_n}")
        else:
            self.linearInfoLbl.setText(f"A sweep: n/a | B sweep: {b_n} | Final points: {p_n}")

    def on_calib_scale_toggle(self, checked: bool) -> None:
        self._update_calib_plot()

    def _apply_calib_scale(self, a_has_positive: bool, b_has_positive: bool) -> None:
        want_log = bool(self.calibLogCheck.isChecked()) if hasattr(self, "calibLogCheck") else False
        if want_log and a_has_positive:
            self.ax_calib.set_yscale("log", nonpositive="clip")
        else:
            self.ax_calib.set_yscale("linear")
        if want_log and b_has_positive:
            self.ax_calib_b.set_yscale("log", nonpositive="clip")
        else:
            self.ax_calib_b.set_yscale("linear")

    def _update_calib_plot(self) -> None:
        a_angles = [e["angle_deg"] for e in self._fine_a] if self._fine_a else []
        a_intens = [e["intensity"] for e in self._fine_a] if self._fine_a else []
        b_angles = [e["angle_deg"] for e in self._fine_b] if self._fine_b else []
        b_intens = [e["intensity"] for e in self._fine_b] if self._fine_b else []

        a_plot = [val if val is not None and val > 0 else np.nan for val in a_intens]
        b_plot = [val if val is not None and val > 0 else np.nan for val in b_intens]
        self.calib_a_points.set_data(a_angles, a_plot)
        self.calib_b_points.set_data(b_angles, b_plot)
        a_has_positive = False
        b_has_positive = False
        try:
            if a_plot and np.any(np.asarray(a_plot, dtype=float) > 0):
                a_has_positive = True
            if b_plot and np.any(np.asarray(b_plot, dtype=float) > 0):
                b_has_positive = True
        except Exception:
            pass

        fit_a = self._build_calib_fit_curve(self._model_a)
        if fit_a is not None:
            x, fit = fit_a
            self.calib_a_fit.set_data(x, fit)
            if np.any(np.asarray(fit, dtype=float) > 0):
                a_has_positive = True
        else:
            self.calib_a_fit.set_data([], [])

        fit_b = self._build_calib_fit_curve(self._model_b)
        if fit_b is not None:
            x, fit = fit_b
            self.calib_b_fit.set_data(x, fit)
            if np.any(np.asarray(fit, dtype=float) > 0):
                b_has_positive = True
        else:
            self.calib_b_fit.set_data([], [])

        self._apply_calib_scale(a_has_positive, b_has_positive)
        self.ax_calib.relim()
        self.ax_calib.autoscale_view()
        self.ax_calib_b.relim()
        self.ax_calib_b.autoscale_view()
        self.canvas_calib.draw_idle()

    def _reset_table(self) -> None:
        self.dataTable.setRowCount(0)
        self._update_point_count_label()

    @staticmethod
    def _format_table_value(val: Optional[float], fmt: str) -> str:
        if val is None:
            return ""
        try:
            num = float(val)
        except Exception:
            return ""
        if not np.isfinite(num):
            return ""
        return format(num, fmt)

    def _set_table_row(self, row: int, entry: dict) -> None:
        a_deg = entry.get("a_deg")
        b_deg = entry.get("b_deg")
        power = entry.get("power")
        intensity = entry.get("intensity")
        base_level = entry.get("ref_level")
        items = [
            QtWidgets.QTableWidgetItem(self._format_table_value(a_deg, ".6f")),
            QtWidgets.QTableWidgetItem(self._format_table_value(b_deg, ".6f")),
            QtWidgets.QTableWidgetItem(self._format_table_value(power, ".9g")),
            QtWidgets.QTableWidgetItem(self._format_table_value(intensity, ".9g")),
            QtWidgets.QTableWidgetItem(self._format_table_value(base_level, ".9g")),
        ]
        for col, item in enumerate(items):
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.dataTable.setItem(row, col, item)

    def _append_table_row(self, entry: dict) -> None:
        row = self.dataTable.rowCount()
        self.dataTable.insertRow(row)
        self._set_table_row(row, entry)
        self._update_point_count_label()

    def _update_point_count_label(self) -> None:
        if hasattr(self, "pointCountLbl") and self.dataTable is not None:
            self.pointCountLbl.setText(f"Points in table: {self.dataTable.rowCount()}")

    def _apply_pairs_to_table(self, pairs: List[Tuple[float, float]], *, use_loaded: bool) -> None:
        self._pair_list = list(pairs)
        self._use_loaded_pairs = bool(use_loaded)
        self._results = [
            {"a_deg": float(a), "b_deg": float(b), "power": None, "intensity": None, "ref_level": None}
            for a, b in pairs
        ]
        self._reset_table()
        for entry in self._results:
            self._append_table_row(entry)

        a_vals = sorted({float(a) for a, _ in pairs})
        b_vals = sorted({float(b) for _, b in pairs})
        self._grid_angles_a = a_vals
        self._grid_angles_b = b_vals
        self._grid_map_a = {round(a, 6): idx for idx, a in enumerate(a_vals)}
        self._grid_map_b = {round(b, 6): idx for idx, b in enumerate(b_vals)}
        self._grid_data = np.full((len(b_vals), len(a_vals)), np.nan)

        self._update_heatmap()
        self._update_power_plot()
        self._update_linear_info()

    def _refresh_table_power(self) -> None:
        for row, entry in enumerate(self._results):
            power = entry.get("power", float("nan"))
            item = QtWidgets.QTableWidgetItem(self._format_table_value(power, ".9g"))
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.dataTable.setItem(row, 2, item)

    def _update_heatmap(self) -> None:
        if self._grid_data is None:
            self.map_artist.set_data(np.zeros((1, 1)))
            self.map_artist.set_extent([0.5, 1.5, 0.5, 1.5])
            self.canvas_map.draw_idle()
            return
        data = np.asarray(self._grid_data, dtype=float)
        nrows, ncols = data.shape
        self.map_artist.set_data(data)
        self.map_artist.set_extent([0.5, ncols + 0.5, 0.5, nrows + 0.5])
        if np.isfinite(data).any():
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
            if vmin == vmax:
                vmax = vmin + 1.0
            self.map_artist.set_clim(vmin, vmax)
        self.canvas_map.draw_idle()

    def _update_power_plot(self) -> None:
        points = []
        for entry in self._results:
            val = entry.get("power")
            if val is None or not np.isfinite(val):
                continue
            points.append((float(val), entry.get("ref_level")))
        if points:
            points.sort(key=lambda item: item[0])
            ys = [p for p, _ in points]
            xs = list(range(1, len(ys) + 1))
            self.line_power.set_data(xs, ys)
            self.ax_power.set_ylabel(f"Power ({self._power_unit})")
        else:
            points = []
            for entry in self._results:
                val = entry.get("intensity")
                if val is None or not np.isfinite(val):
                    continue
                points.append((float(val), entry.get("ref_level")))
            points.sort(key=lambda item: item[0])
            ys = [p for p, _ in points]
            xs = list(range(1, len(ys) + 1))
            self.line_power.set_data(xs, ys)
            self.ax_power.set_ylabel("Intensity (counts)")

        ref_xs = []
        ref_ys = []
        for idx, (_, ref_val) in enumerate(points, start=1):
            if ref_val is None:
                continue
            try:
                ref_num = float(ref_val)
            except Exception:
                continue
            if not np.isfinite(ref_num):
                continue
            ref_xs.append(idx)
            ref_ys.append(ref_num)
        if ref_xs:
            self.ref_scatter.set_offsets(np.column_stack([ref_xs, ref_ys]))
        else:
            self.ref_scatter.set_offsets(np.empty((0, 2)))

        self.ax_power.relim()
        self.ax_power.autoscale_view()
        self.ax_power_ref.relim()
        self.ax_power_ref.autoscale_view()
        self.canvas_power.draw_idle()

    def _update_headers(self) -> None:
        self.fitTable.setHorizontalHeaderLabels(["Counts", f"Power ({self._power_unit})"])
        self.dataTable.setHorizontalHeaderLabels(
            ["A deg", "B deg", f"Power ({self._power_unit})", "Intensity", "Base level (counts)"]
        )
        self.ax_power.set_ylabel(f"Power ({self._power_unit})")
        self.canvas_power.draw_idle()
        if hasattr(self, "ax_fit"):
            self.ax_fit.set_ylabel(f"Power ({self._power_unit})")
            self._update_fit_plot()

    @staticmethod
    def _normalize_unit(unit: Optional[str]) -> Optional[str]:
        if not unit:
            return None
        unit = unit.strip()
        unit_l = unit.lower()
        if unit_l == "w":
            return "W"
        if unit_l == "mw":
            return "mW"
        if unit_l == "uw":
            return "uW"
        if unit_l == "nw":
            return "nW"
        return unit

    def _update_fit_plot(self) -> None:
        if not hasattr(self, "ax_fit"):
            return
        xs = []
        ys = []
        for row in range(self.fitTable.rowCount()):
            x_item = self.fitTable.item(row, 0)
            y_item = self.fitTable.item(row, 1)
            if x_item is None or y_item is None:
                continue
            try:
                x = float(x_item.text())
                y = float(y_item.text())
            except Exception:
                continue
            xs.append(x)
            ys.append(y)
        if xs and ys:
            self.fit_points.set_data(xs, ys)
            if self._fit_slope is not None and self._fit_intercept is not None:
                xmin = min(xs)
                xmax = max(xs)
                xgrid = np.linspace(xmin, xmax, 50)
                ygrid = self._fit_slope * xgrid + self._fit_intercept
                self.fit_line.set_data(xgrid, ygrid)
            else:
                self.fit_line.set_data([], [])
        else:
            self.fit_points.set_data([], [])
            self.fit_line.set_data([], [])
        self.ax_fit.relim()
        self.ax_fit.autoscale_view()
        self.canvas_fit.draw_idle()
    # -----------------
    # Status polling
    # -----------------
    def _poll_status(self) -> None:
        cam_ok = self.cam.cam is not None
        stage_a_ok = self.stage_a is not None
        stage_b_ok = self.stage_b is not None
        spec_ok = self.cam.spec is not None

        self._set_dot(self.camDot, cam_ok)
        self._set_dot(self.stageADot, stage_a_ok)
        self._set_dot(self.stageBDot, stage_b_ok)
        self._set_dot(self.specDot, spec_ok)

        if cam_ok:
            temp = self.cam.get_temperature_c()
            temp_s = f"{temp:.1f} C" if temp is not None else "--"
            exp_ms = float(self.exposureSpin.value())
            acc = int(self.accumSpin.value())
            self.camStatusLbl.setText(f"T={temp_s} | Exp {exp_ms:g} ms x{acc}")
        else:
            self.camStatusLbl.setText("Disconnected")

        self._update_stage_status(self.stage_a, self.stageAStatusLbl, "A")
        self._update_stage_status(self.stage_b, self.stageBStatusLbl, "B")
        if spec_ok:
            try:
                center = self.cam.get_center_wavelength_nm()
            except Exception:
                center = None
            try:
                slit = self.cam.spec_get_slit_width_um(DEFAULT_SLIT_ID)
            except Exception:
                slit = None
            center_s = f"{center:.3f}" if center is not None else "--"
            slit_s = f"{slit:.1f}" if slit is not None else "--"
            self.specStatusLbl.setText(f"Center {center_s} nm | Slit {slit_s} um")
        else:
            self.specStatusLbl.setText("Disconnected")
        self._update_preamp_label()
        self._update_readout_label()

    def _update_stage_status(self, stage: Optional[RotationStage], label: QtWidgets.QLabel, tag: str) -> None:
        if tag == "A" and not self._coarse_mode_active():
            label.setText("Stage A: coarse control disabled")
            return
        if stage is None:
            label.setText(f"Stage {tag}: disconnected")
            return
        try:
            pos = float(stage.get_position())
        except Exception:
            pos = None
        try:
            homed = stage.is_homed()
        except Exception:
            homed = None
        try:
            moving = bool(stage.is_moving())
        except Exception:
            moving = False
        pos_s = f"{pos:.3f} deg" if pos is not None else "--"
        home_s = "yes" if homed else ("no" if homed is False else "?")
        status = "Moving..." if (self._stage_busy or moving) else "Ready"
        label.setText(f"SN {stage.serial} | Homed {home_s} | {status} | Angle {pos_s}")
    # -----------------
    # Actions
    # -----------------
    def on_initialize(self):
        if self._busy:
            return
        coarse_enabled = self._coarse_control_enabled()
        try:
            self.cam.connect()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Andor connect", str(exc))
            return
        try:
            self.cam.connect_spectrograph(int(DEFAULT_SPEC_INDEX))
        except Exception:
            pass
        self._apply_camera_defaults()
        self._apply_spec_settings()

        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None
        if info:
            self._readout_rate_label = info.get("rate_label") or self._readout_rate_label
            self._preamp_gain_label = info.get("gain_label") or self._preamp_gain_label
            self._output_amp_label = info.get("amp_label") or self._output_amp_label

        self._readout_rate_label = str(DEFAULT_READOUT_RATE)
        self._preamp_gain_label = str(DEFAULT_PREAMP_GAIN)
        self._output_amp_label = str(DEFAULT_OUTPUT_AMP)

        self._refresh_amp_choices()
        self._apply_amp_settings()

        self._close_stage(self.stage_a)
        self._close_stage(self.stage_b)
        self.stage_a = None
        self.stage_b = None

        if coarse_enabled:
            serial_a = self.stageASerialCombo.currentText().strip()
            if serial_a:
                try:
                    self.stage_a = RotationStage(serial_a, scale=DEFAULT_STAGE_SCALE)
                    self.stage_a.open()
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Stage A connect", str(exc))
                    self.stage_a = None

        serial_b = self.stageBSerialCombo.currentText().strip()
        if serial_b:
            try:
                self.stage_b = RotationStage(serial_b, scale=DEFAULT_STAGE_SCALE)
                self.stage_b.open()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Stage B connect", str(exc))
                self.stage_b = None

        self._coarse_control_active = coarse_enabled
        mode_label = "dual-wheel mode" if self._coarse_mode_active() else "single-wheel mode (B only)"
        self.statusBar().showMessage(f"Initialized ({mode_label})")
        self._update_coarse_control_ui()
        self._sync_controls()

    def on_disconnect(self):
        self.on_abort()
        for stage in (self.stage_a, self.stage_b):
            try:
                if stage:
                    stage.close()
            except Exception:
                pass
        self.stage_a = None
        self.stage_b = None
        try:
            self.cam.disconnect_spectrograph()
        except Exception:
            pass
        try:
            self.cam.disconnect()
        except Exception:
            pass
        self.statusBar().showMessage("Disconnected")
        self._sync_controls()

    def on_stage_detect(self):
        try:
            serials = list_kinesis_serials()
        except Exception as exc:
            self.statusBar().showMessage(f"Detect failed: {exc}")
            return
        if not serials:
            self.statusBar().showMessage("No stages detected")
            return
        current_a = self.stageASerialCombo.currentText().strip()
        current_b = self.stageBSerialCombo.currentText().strip()
        self.stageASerialCombo.blockSignals(True)
        self.stageBSerialCombo.blockSignals(True)
        self.stageASerialCombo.clear()
        self.stageBSerialCombo.clear()
        self.stageASerialCombo.addItems(serials)
        self.stageBSerialCombo.addItems(serials)
        if current_a:
            self.stageASerialCombo.setCurrentText(current_a)
        if current_b:
            self.stageBSerialCombo.setCurrentText(current_b)
        self.stageASerialCombo.blockSignals(False)
        self.stageBSerialCombo.blockSignals(False)
        self.statusBar().showMessage(f"Detected {len(serials)} stage(s)")

    def on_stage_home_a(self):
        self._home_stage(self.stage_a, "A")

    def on_stage_home_b(self):
        self._home_stage(self.stage_b, "B")

    def on_stage_move_a(self):
        self._move_stage(self.stage_a, float(self.stageATargetSpin.value()), "A")

    def on_stage_move_b(self):
        self._move_stage(self.stage_b, float(self.stageBTargetSpin.value()), "B")

    def _home_stage(self, stage: Optional[RotationStage], tag: str) -> None:
        if stage is None or self._stage_busy:
            return
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        try:
            stage.home(controller=self._stage_controller)
            self.statusBar().showMessage(f"Home {tag} OK")
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
        self._stage_busy = False
        self._sync_controls()

    def _move_stage(
        self,
        stage: Optional[RotationStage],
        target: float,
        tag: str,
        *,
        show_status: bool = True,
    ) -> bool:
        if stage is None or self._stage_busy:
            return False
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        ok = False
        try:
            stage.move_to(
                float(target),
                step_deg=float(DEFAULT_RAMP_STEP_DEG),
                controller=self._stage_controller,
            )
            ok = True
            if show_status:
                self.statusBar().showMessage(f"Move {tag} OK")
        except Exception as exc:
            if show_status:
                self.statusBar().showMessage(str(exc))
        self._stage_busy = False
        self._sync_controls()
        return ok

    def _move_wheels_to_min_positions(self) -> str:
        targets = [
            ("A", self.stage_a, float(self.aMinSpin.value()) if self.aMinSpin is not None else 0.0),
            ("B", self.stage_b, float(self.bMinSpin.value()) if self.bMinSpin is not None else 0.0),
        ]
        moved = []
        failed = []
        for tag, stage, target in targets:
            if stage is None:
                continue
            ok = self._move_stage(stage, target, tag, show_status=False)
            if ok:
                moved.append(f"{tag}={target:.3f}")
            else:
                failed.append(tag)
        if moved and not failed:
            return f"moved wheels to min ({', '.join(moved)} deg)"
        if moved and failed:
            return f"moved {', '.join(moved)} deg; failed: {', '.join(failed)}"
        if failed:
            return f"move to min failed: {', '.join(failed)}"
        return ""

    def on_live(self):
        if self._busy or self.cam.cam is None:
            return
        try:
            self.cam.set_frame_api("Stream+Buffer")
            self.cam.set_acquisition_mode("Run till abort")
            self.cam.set_trigger_mode("Internal")
            self.cam.set_exposure_ms(float(self.exposureSpin.value()))
        except Exception:
            pass
        self._live_thread = LiveAcqThread(self.cam)
        self._live_thread.frame_ready.connect(self._on_live_frame)
        self._live_thread.status.connect(self.statusBar().showMessage)
        self._live_thread.start()
        self._live_active = True
        self._set_busy(True)
        self.statusBar().showMessage("Live started")

    def on_stop_live(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
        try:
            if self.cam is not None:
                self.cam.stop_stream()
        except Exception:
            pass
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Live stopped")

    def on_abort(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
        if self._fine_thread is not None and self._fine_thread.isRunning():
            self._fine_thread.stop()
        if self._grid_thread is not None and self._grid_thread.isRunning():
            self._grid_thread.stop()
        if self._stage_controller is not None:
            try:
                self._stage_controller.abort()
            except Exception:
                pass
        for stage in (self.stage_a, self.stage_b):
            try:
                if stage is not None:
                    stage.stop()
            except Exception:
                pass
        self._stage_busy = False
        try:
            if self.cam is not None:
                self.cam.stop_stream()
        except Exception:
            pass
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Aborted")

    def on_apply_roi(self):
        roi = self._roi_tuple()
        if roi is None:
            self.liveView.clear_roi()
        else:
            self.liveView.set_roi(*roi)
        self._update_roi_info()

    def on_apply_spec(self):
        self._apply_spec_settings()
        self.statusBar().showMessage("Spectrograph settings applied")

    def on_preamp_change(self):
        if self.cam is None or self.cam.cam is None:
            self._preamp_gain_label = self.preampCombo.currentText().strip()
            self._update_preamp_label()
            return
        self._apply_amp_settings()

    def on_browse_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select calibration file",
            os.path.join(DATA_DIR, "measurements"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.loadPathEdit.setText(path)

    def on_load(self):
        path = self.loadPathEdit.text().strip()
        if not path:
            return
        try:
            rows, unit = self._read_calib_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        if not rows:
            QtWidgets.QMessageBox.warning(self, "Load error", "No data rows found.")
            return
        unit = self._normalize_unit(unit)
        if unit and unit in _UNIT_SCALE:
            self._power_unit = unit
            self.powerUnitCombo.blockSignals(True)
            self.powerUnitCombo.setCurrentText(unit)
            self.powerUnitCombo.blockSignals(False)
            self._update_headers()

        self._results = rows
        self._reset_table()
        for entry in self._results:
            self._append_table_row(entry)

        a_vals = sorted({float(r["a_deg"]) for r in self._results})
        b_vals = sorted({float(r["b_deg"]) for r in self._results})
        self._angles_a = a_vals
        self._angles_b = b_vals
        self._pair_list = [(float(r["a_deg"]), float(r["b_deg"])) for r in self._results]
        self._grid_angles_a = a_vals
        self._grid_angles_b = b_vals
        self._grid_map_a = {round(a, 6): idx for idx, a in enumerate(a_vals)}
        self._grid_map_b = {round(b, 6): idx for idx, b in enumerate(b_vals)}
        self._grid_data = np.full((len(b_vals), len(a_vals)), np.nan)
        for entry in self._results:
            a_idx = a_vals.index(float(entry["a_deg"]))
            b_idx = b_vals.index(float(entry["b_deg"]))
            self._grid_data[b_idx, a_idx] = float(entry.get("intensity", float("nan")))
        self._update_linear_info()
        self._update_heatmap()
        self._update_power_plot()
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}")
        self._sync_controls()

    def on_export(self):
        if not self._results:
            self.statusBar().showMessage("No data to export")
            return
        self._resolve_save_dir()
        fname = os.path.join(self._save_dir, "dual_wheel_intensity_calib.csv")
        if self._coarse_mode_active():
            self._write_dual_csv(fname, self._results)
        else:
            self._write_single_base_csv(fname, self._results)
        self.statusBar().showMessage(f"Saved {os.path.basename(fname)}")

    def on_export_session(self):
        self._resolve_save_dir()
        default_path = os.path.join(self._save_dir, "dual_wheel_calib_session.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export session",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        session = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "setup": {
                "coarse_control_enabled": bool(self.coarseControlCheck.isChecked()),
                "a_min_deg": float(self.aMinSpin.value()),
                "a_max_deg": float(self.aMaxSpin.value()),
                "b_min_deg": float(self.bMinSpin.value()),
                "b_max_deg": float(self.bMaxSpin.value()),
                "fine_step_deg": float(self.fineStepSpin.value()),
                "total_points": int(self.totalPointsSpin.value()),
                "a_coarse_steps": int(self.coarseASpin.value()),
                "min_step_near_open_deg": float(self.minStepSpin.value()),
                "p00": float(self.p00Spin.value()),
                "clamp_t_to_1": bool(self.clampCheck.isChecked()),
                "settle_ms": float(self._settle_ms),
                "acq_n": int(self.accumSpin.value()),
                "exposure_ms": float(self.exposureSpin.value()),
            },
            "od_planner": {
                "smooth_pts": int(self.od_smooth_slider.val),
                "sens_thresh": float(self.od_sens_slider.val),
                "power_split": float(self.od_power_split_slider.val),
                "point_split": float(self.od_point_split_slider.val),
                "total_points": int(self.od_total_points_slider.val),
                "good_od_low": float(self.od_good_slider.val[0]),
                "good_od_high": float(self.od_good_slider.val[1]),
                "good_od_a_low": float(self.od_good_a_slider.val[0]),
                "good_od_a_high": float(self.od_good_a_slider.val[1]),
                "min_dp_low": float(self.od_mindp_low_slider.val),
                "min_dp_high": float(self.od_mindp_high_slider.val),
            },
            "roi": {
                "x1": int(self.roiX1.value()),
                "x2": int(self.roiX2.value()),
                "y1": int(self.roiY1.value()),
                "y2": int(self.roiY2.value()),
                "crop": list(self._crop),
            },
            "power_fit": {
                "unit": str(self._power_unit),
                "slope": self._fit_slope,
                "intercept": self._fit_intercept,
                "r2": self._fit_r2,
                "table": self._export_fit_table(),
            },
            "calib_a": self._export_model(self._model_a, self._fine_a),
            "calib_b": self._export_model(self._model_b, self._fine_b),
            "pair_list": self._pair_list or [],
            "results": self._results or [],
        }

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(session, handle, indent=2)
        except Exception as exc:
            self.statusBar().showMessage(f"Export failed: {exc}")
            return
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}")

    def _export_fit_table(self):
        rows = []
        for row in range(self.fitTable.rowCount()):
            x_item = self.fitTable.item(row, 0)
            y_item = self.fitTable.item(row, 1)
            x = x_item.text().strip() if x_item is not None else ""
            y = y_item.text().strip() if y_item is not None else ""
            rows.append({"counts": x, "power": y})
        return rows

    def _export_model(self, model, points):
        payload = {
            "points": points or [],
        }
        if model is None:
            payload["model"] = None
            return payload
        def _tolist(val):
            if val is None:
                return None
            if hasattr(val, "tolist"):
                return val.tolist()
            return val
        payload["model"] = {
            "angles": _tolist(model.get("angles")),
            "angles_wrapped": _tolist(model.get("angles_wrapped")),
            "intensities": _tolist(model.get("intensities")),
            "angle_lo": model.get("angle_lo"),
            "angle_hi": model.get("angle_hi"),
            "wrap_deg": model.get("wrap_deg"),
        }
        return payload

    def on_load_session(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load session",
            os.path.join(DATA_DIR, "measurements"),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                session = json.load(handle)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load session", str(exc))
            return

        setup = session.get("setup", {})
        roi = session.get("roi", {})
        power_fit = session.get("power_fit", {})
        od_planner = session.get("od_planner", {})

        if setup:
            if "coarse_control_enabled" in setup:
                self.coarseControlCheck.setChecked(bool(setup.get("coarse_control_enabled")))
            self.aMinSpin.setValue(float(setup.get("a_min_deg", self.aMinSpin.value())))
            self.aMaxSpin.setValue(float(setup.get("a_max_deg", self.aMaxSpin.value())))
            self.bMinSpin.setValue(float(setup.get("b_min_deg", self.bMinSpin.value())))
            self.bMaxSpin.setValue(float(setup.get("b_max_deg", self.bMaxSpin.value())))
            self.fineStepSpin.setValue(float(setup.get("fine_step_deg", self.fineStepSpin.value())))
            self.totalPointsSpin.setValue(int(setup.get("total_points", self.totalPointsSpin.value())))
            coarse_steps = setup.get("a_coarse_steps", setup.get("b_coarse_steps", self.coarseASpin.value()))
            self.coarseASpin.setValue(int(coarse_steps))
            self.minStepSpin.setValue(float(setup.get("min_step_near_open_deg", self.minStepSpin.value())))
            self.p00Spin.setValue(float(setup.get("p00", self.p00Spin.value())))
            self.clampCheck.setChecked(bool(setup.get("clamp_t_to_1", self.clampCheck.isChecked())))
            self.accumSpin.setValue(int(setup.get("acq_n", self.accumSpin.value())))
            self.exposureSpin.setValue(float(setup.get("exposure_ms", self.exposureSpin.value())))
            try:
                self._settle_ms = float(setup.get("settle_ms", self._settle_ms))
            except Exception:
                pass

        if roi:
            self.roiX1.setValue(int(roi.get("x1", self.roiX1.value())))
            self.roiX2.setValue(int(roi.get("x2", self.roiX2.value())))
            self.roiY1.setValue(int(roi.get("y1", self.roiY1.value())))
            self.roiY2.setValue(int(roi.get("y2", self.roiY2.value())))
            crop = roi.get("crop")
            if isinstance(crop, (list, tuple)) and len(crop) == 4:
                self._crop = tuple(int(v) for v in crop)
                try:
                    if self.liveView is not None:
                        self.liveView.set_crop(*self._crop)
                except Exception:
                    pass
            self.on_apply_roi()

        if od_planner:
            self._od_slider_block = True
            try:
                if "total_points" in od_planner:
                    total_points = int(od_planner.get("total_points"))
                    self.totalPointsSpin.setValue(total_points)
                    if self.od_total_points_slider is not None:
                        self.od_total_points_slider.set_val(total_points)
                if self.od_smooth_slider is not None and "smooth_pts" in od_planner:
                    self.od_smooth_slider.set_val(int(od_planner.get("smooth_pts")))
                if self.od_sens_slider is not None and "sens_thresh" in od_planner:
                    self.od_sens_slider.set_val(float(od_planner.get("sens_thresh")))
                if self.od_power_split_slider is not None and "power_split" in od_planner:
                    self.od_power_split_slider.set_val(float(od_planner.get("power_split")))
                if self.od_point_split_slider is not None and "point_split" in od_planner:
                    self.od_point_split_slider.set_val(float(od_planner.get("point_split")))
                if self.od_good_slider is not None and "good_od_low" in od_planner and "good_od_high" in od_planner:
                    self.od_good_slider.set_val(
                        (float(od_planner.get("good_od_low")), float(od_planner.get("good_od_high")))
                    )
                if self.od_good_a_slider is not None:
                    if "good_od_a_low" in od_planner and "good_od_a_high" in od_planner:
                        self.od_good_a_slider.set_val(
                            (float(od_planner.get("good_od_a_low")), float(od_planner.get("good_od_a_high")))
                        )
                if self.od_mindp_low_slider is not None and "min_dp_low" in od_planner:
                    self.od_mindp_low_slider.set_val(float(od_planner.get("min_dp_low")))
                if self.od_mindp_high_slider is not None and "min_dp_high" in od_planner:
                    self.od_mindp_high_slider.set_val(float(od_planner.get("min_dp_high")))
            finally:
                self._od_slider_block = False

        if power_fit:
            unit = self._normalize_unit(power_fit.get("unit"))
            if unit and unit in _UNIT_SCALE:
                self._power_unit = unit
                self.powerUnitCombo.blockSignals(True)
                self.powerUnitCombo.setCurrentText(unit)
                self.powerUnitCombo.blockSignals(False)
                self._update_headers()
                if hasattr(self, "p00UnitLbl"):
                    self.p00UnitLbl.setText(unit)
            self._fit_slope = power_fit.get("slope")
            self._fit_intercept = power_fit.get("intercept")
            self._fit_r2 = power_fit.get("r2")
            if self._fit_slope is not None and self._fit_intercept is not None:
                self.fitResultLbl.setText(
                    f"Fit: power = {self._fit_slope:.6g}*counts + {self._fit_intercept:.6g}, R2={self._fit_r2:.4f}"
                )
            else:
                self.fitResultLbl.setText("Fit: --")
            rows = power_fit.get("table", [])
            if isinstance(rows, list):
                self.fitTable.setRowCount(0)
                for row in rows:
                    r = self.fitTable.rowCount()
                    self.fitTable.insertRow(r)
                    self.fitTable.setItem(r, 0, QtWidgets.QTableWidgetItem(str(row.get("counts", ""))))
                    self.fitTable.setItem(r, 1, QtWidgets.QTableWidgetItem(str(row.get("power", ""))))
                self._update_fit_plot()

        self._fine_a = session.get("calib_a", {}).get("points", []) or []
        self._fine_b = session.get("calib_b", {}).get("points", []) or []
        if self._fine_a:
            a_angles = [e.get("angle_deg") for e in self._fine_a]
            a_intens = [e.get("intensity") for e in self._fine_a]
            a_min = float(self.aMinSpin.value())
            a_max = float(self.aMaxSpin.value())
            self._model_a = self._build_wheel_model(
                a_angles,
                a_intens,
                wrap_start_deg=a_min,
                force_wrap=(a_min > a_max),
            )
            self._angles_a = list(self._model_a["angles"]) if self._model_a else None
        else:
            self._model_a = None
            self._angles_a = None
        if self._fine_b:
            b_angles = [e.get("angle_deg") for e in self._fine_b]
            b_intens = [e.get("intensity") for e in self._fine_b]
            b_min = float(self.bMinSpin.value())
            b_max = float(self.bMaxSpin.value())
            self._model_b = self._build_wheel_model(
                b_angles,
                b_intens,
                wrap_start_deg=b_min,
                force_wrap=(b_min > b_max),
            )
            self._angles_b = list(self._model_b["angles"]) if self._model_b else None
        else:
            self._model_b = None
            self._angles_b = None

        pair_list = session.get("pair_list") or None
        if pair_list:
            self._pair_list = [(float(a), float(b)) for a, b in pair_list]
            self._use_loaded_pairs = True
            self._od_plan_dirty = False
        else:
            self._pair_list = None
            self._use_loaded_pairs = False
        self._results = session.get("results") or []
        self._reset_table()
        for entry in self._results:
            self._append_table_row(entry)

        if (self._fit_slope is None or self._fit_intercept is None) and self.fitTable.rowCount() >= 2:
            self.on_fit()
        elif self._fit_slope is not None and self._fit_intercept is not None and self._results:
            for entry in self._results:
                entry["power"] = self._counts_to_power(entry.get("intensity", 0.0))
            self._refresh_table_power()

        if self._results:
            a_vals = sorted({float(r["a_deg"]) for r in self._results})
            b_vals = sorted({float(r["b_deg"]) for r in self._results})
            self._grid_angles_a = a_vals
            self._grid_angles_b = b_vals
            self._grid_map_a = {round(a, 6): idx for idx, a in enumerate(a_vals)}
            self._grid_map_b = {round(b, 6): idx for idx, b in enumerate(b_vals)}
            self._grid_data = np.full((len(b_vals), len(a_vals)), np.nan)
            for entry in self._results:
                a_idx = self._grid_map_a.get(round(float(entry["a_deg"]), 6))
                b_idx = self._grid_map_b.get(round(float(entry["b_deg"]), 6))
                if a_idx is None or b_idx is None:
                    continue
                intensity = entry.get("intensity")
                if intensity is None:
                    continue
                try:
                    val = float(intensity)
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                self._grid_data[b_idx, a_idx] = val
        else:
            self._grid_data = None

        self._update_calib_plot()
        self._update_linear_info()
        self._update_heatmap()
        self._update_power_plot()
        self._update_od_planner()
        self._sync_controls()
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}")

    def on_fit_add(self):
        row = self.fitTable.rowCount()
        self.fitTable.insertRow(row)
        self._update_fit_plot()

    def on_fit_remove(self):
        row = self.fitTable.rowCount() - 1
        if row < 0:
            return
        self.fitTable.removeRow(row)
        self._update_fit_plot()

    def on_fit(self):
        xs = []
        ys = []
        for row in range(self.fitTable.rowCount()):
            x_item = self.fitTable.item(row, 0)
            y_item = self.fitTable.item(row, 1)
            if x_item is None or y_item is None:
                continue
            try:
                x = float(x_item.text())
                y = float(y_item.text())
            except Exception:
                continue
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            QtWidgets.QMessageBox.warning(self, "Fit", "Need at least two points.")
            return
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
        y_pred = slope * xs_arr + intercept
        ss_res = float(np.sum((ys_arr - y_pred) ** 2))
        ss_tot = float(np.sum((ys_arr - np.mean(ys_arr)) ** 2))
        r2 = 1.0 if ss_tot == 0 else (1.0 - ss_res / ss_tot)

        self._fit_slope = float(slope)
        self._fit_intercept = float(intercept)
        self._fit_r2 = float(r2)
        self.fitResultLbl.setText(
            f"Fit: power = {slope:.6g}*counts + {intercept:.6g}, R2={r2:.4f}"
        )

        for entry in self._results:
            entry["power"] = self._counts_to_power(entry.get("intensity", 0.0))
        self._refresh_table_power()
        self._update_fit_plot()
        self._update_power_plot()
        self._update_calib_plot()
        self._update_od_planner()

    def on_power_unit_changed(self, unit: str):
        if not unit:
            return
        if unit not in _UNIT_SCALE:
            return
        self._power_unit = unit
        self._fit_slope = None
        self._fit_intercept = None
        self._fit_r2 = None
        self.fitResultLbl.setText("Fit: --")
        for entry in self._results:
            entry["power"] = float("nan")
        self._refresh_table_power()
        self._update_headers()
        self._update_power_plot()
        if hasattr(self, "p00UnitLbl"):
            self.p00UnitLbl.setText(unit)
        if hasattr(self, "ax_od3"):
            self.ax_od3.set_ylabel(f"Power ({unit})")
        self._update_calib_plot()
        self._update_od_planner()
    def on_calib_a(self):
        if not self._coarse_mode_active():
            QtWidgets.QMessageBox.warning(self, "Calib A", "Coarse control is disabled. Enable it and Initialize first.")
            return
        if self._busy or self.cam.cam is None or self.stage_a is None or self.stage_b is None:
            return
        step = float(self.fineStepSpin.value())
        if step <= 0:
            QtWidgets.QMessageBox.warning(self, "Calib A", "Fine step must be > 0.")
            return
        start = float(self.aMinSpin.value())
        stop = float(self.aMaxSpin.value())
        points = self._gen_points(start, stop, step, self.stage_a)
        if not points:
            QtWidgets.QMessageBox.warning(self, "Calib A", "No sweep points generated.")
            return
        fixed_angle = float(self.bMaxSpin.value())
        self._fine_a_wrap_start = float(points[0]) if points else float(start)
        self._fine_a_force_wrap = bool(len(points) >= 2 and np.any(np.diff(np.asarray(points, dtype=float)) < -1e-9))

        self._fine_a = []
        self._model_a = None
        self._angles_a = None
        self._pair_list = None
        self._use_loaded_pairs = False
        self._update_calib_plot()
        roi = self._roi_tuple()
        self.on_apply_roi()
        self._fine_thread = DualWheelFineSweepThread(
            self.cam,
            self.stage_a,
            self.stage_b,
            fixed_angle=fixed_angle,
            angles=points,
            accum_n=int(self.accumSpin.value()),
            exposure_ms=float(self.exposureSpin.value()),
            ramp_step_deg=float(DEFAULT_RAMP_STEP_DEG),
            ref_angles=None,
            ref_every=self._ref_every,
            settle_ms=self._settle_ms,
            crop=self._crop,
            roi=roi,
            max_retries=3,
        )
        self._fine_thread.frame_ready.connect(self._on_live_frame)
        self._fine_thread.point_ready.connect(self._on_fine_a_point)
        self._fine_thread.status.connect(self.statusBar().showMessage)
        self._fine_thread.done.connect(self._on_fine_a_done)
        self._fine_thread.start()
        self._set_busy(True)
        self.statusBar().showMessage("Calib A started")

    def on_calib_b(self):
        if self._coarse_mode_active():
            if self._busy or self.cam.cam is None or self.stage_a is None or self.stage_b is None:
                return
        else:
            if self._busy or self.cam.cam is None or self.stage_b is None:
                return
        step = float(self.fineStepSpin.value())
        if step <= 0:
            QtWidgets.QMessageBox.warning(self, "Calib B", "Fine step must be > 0.")
            return
        start = float(self.bMinSpin.value())
        stop = float(self.bMaxSpin.value())
        points = self._gen_points(start, stop, step, self.stage_b)
        if not points:
            QtWidgets.QMessageBox.warning(self, "Calib B", "No sweep points generated.")
            return
        self._fine_b_wrap_start = float(points[0]) if points else float(start)
        self._fine_b_force_wrap = bool(len(points) >= 2 and np.any(np.diff(np.asarray(points, dtype=float)) < -1e-9))
        self._fine_b = []
        self._model_b = None
        self._angles_b = None
        self._pair_list = None
        self._use_loaded_pairs = False
        self._update_calib_plot()
        roi = self._roi_tuple()
        self.on_apply_roi()
        if self._coarse_mode_active():
            fixed_angle = float(self.aMaxSpin.value())
            self._fine_thread = DualWheelFineSweepThread(
                self.cam,
                self.stage_b,
                self.stage_a,
                fixed_angle=fixed_angle,
                angles=points,
                accum_n=int(self.accumSpin.value()),
                exposure_ms=float(self.exposureSpin.value()),
                ramp_step_deg=float(DEFAULT_RAMP_STEP_DEG),
                ref_angles=None,
                ref_every=self._ref_every,
                settle_ms=self._settle_ms,
                crop=self._crop,
                roi=roi,
                max_retries=3,
            )
            self._fine_thread.frame_ready.connect(self._on_live_frame)
            self._fine_thread.point_ready.connect(self._on_fine_b_point)
            self._fine_thread.status.connect(self.statusBar().showMessage)
            self._fine_thread.done.connect(self._on_fine_b_done)
            self._fine_thread.start()
        else:
            self._fine_thread = IntensitySweepThread(
                self.cam,
                self.stage_b,
                points,
                int(self.accumSpin.value()),
                float(self.exposureSpin.value()),
                float(DEFAULT_RAMP_STEP_DEG),
                self._crop,
                roi,
            )
            self._fine_thread.frame_ready.connect(self._on_live_frame)
            self._fine_thread.point_ready.connect(self._on_fine_b_point_single)
            self._fine_thread.status.connect(self.statusBar().showMessage)
            self._fine_thread.done.connect(self._on_fine_b_done)
            self._fine_thread.start()
        self._set_busy(True)
        if self._coarse_mode_active():
            self.statusBar().showMessage("Calib B started")
        else:
            self.statusBar().showMessage("Calib B started (B only)")

    def on_grid(self):
        if self._coarse_mode_active():
            if self._busy or self.cam.cam is None or self.stage_a is None or self.stage_b is None:
                return
            if self._model_a is None or self._model_b is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Run Calib A and Calib B first.")
                return
        else:
            if self._busy or self.cam.cam is None or self.stage_b is None:
                return
            if self._model_b is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Run Calib B first.")
                return
        total_points = int(self.totalPointsSpin.value())
        if total_points < 2:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Total points must be >= 2.")
            return

        use_loaded = bool(self._use_loaded_pairs and self._pair_list and not self._od_plan_dirty)
        if use_loaded:
            pairs = list(self._pair_list)
        else:
            pairs = self._select_weighted_pairs(total_points)
            self._use_loaded_pairs = False
            self._od_plan_dirty = False
        if not pairs:
            QtWidgets.QMessageBox.warning(self, "Sweep", "No target pairs generated.")
            return
        ordered_pairs = pairs

        self._apply_pairs_to_table(ordered_pairs, use_loaded=use_loaded)

        roi = self._roi_tuple()
        self.on_apply_roi()
        if self._coarse_mode_active():
            ref_a, ref_b = self._reference_angles()
            self._grid_thread = DualWheelListThread(
                self.cam,
                self.stage_a,
                self.stage_b,
                pairs=ordered_pairs,
                accum_n=int(self.accumSpin.value()),
                exposure_ms=float(self.exposureSpin.value()),
                ramp_step_deg=float(DEFAULT_RAMP_STEP_DEG),
                ref_angles=(ref_a, ref_b),
                ref_every=self._ref_every,
                settle_ms=self._settle_ms,
                crop=self._crop,
                roi=roi,
                max_retries=3,
            )
            self._grid_thread.frame_ready.connect(self._on_live_frame)
            self._grid_thread.point_ready.connect(self._on_list_point)
            self._grid_thread.status.connect(self.statusBar().showMessage)
            self._grid_thread.done.connect(self._on_grid_done)
            self._grid_thread.start()
        else:
            b_positions = [float(b) for _a, b in ordered_pairs]
            self._single_grid_idx = 0
            self._grid_thread = IntensitySweepThread(
                self.cam,
                self.stage_b,
                b_positions,
                int(self.accumSpin.value()),
                float(self.exposureSpin.value()),
                float(DEFAULT_RAMP_STEP_DEG),
                self._crop,
                roi,
            )
            self._grid_thread.frame_ready.connect(self._on_live_frame)
            self._grid_thread.point_ready.connect(self._on_single_grid_point)
            self._grid_thread.status.connect(self.statusBar().showMessage)
            self._grid_thread.done.connect(self._on_grid_done)
            self._grid_thread.start()
        self._set_busy(True)
        self.statusBar().showMessage("Sweep started")
    def _on_live_frame(self, fr: dict):
        if self.liveView is not None:
            self.liveView.update_frame(fr)
            now = time.monotonic()
            if now - self._last_roi_title_s < self._roi_title_interval_s:
                return
            self._last_roi_title_s = now
            img = fr.get("image")
            if img is None:
                return
            disp = self.liveView.prepare_display_image(img)
            intensity = self._roi_sum(disp, self._roi_tuple())
            title = f"{self._live_title_base} | ROI sum: {intensity:.4g}"
            try:
                self.liveView.setTitle(title)
            except Exception:
                pass
            try:
                self.liveView.ax_img.set_title(f"Live (ROI sum: {intensity:.4g})")
                self.liveView.canvas.draw_idle()
            except Exception:
                pass

    def _on_fine_a_point(self, angle_deg: float, intensity: float, _ref_level: float, _image):
        self._fine_a.append({"angle_deg": float(angle_deg), "intensity": float(intensity)})
        self._update_calib_plot()

    def _on_fine_b_point(self, angle_deg: float, intensity: float, _ref_level: float, _image):
        self._fine_b.append({"angle_deg": float(angle_deg), "intensity": float(intensity)})
        self._update_calib_plot()

    def _on_fine_b_point_single(self, angle_deg: float, intensity: float, _image):
        self._on_fine_b_point(angle_deg, intensity, float("nan"), _image)

    def _on_fine_a_done(self, status: str):
        self._set_busy(False)
        move_note = self._move_wheels_to_min_positions()
        if status != "ok":
            msg = str(status)
            if move_note:
                msg = f"{msg} | {move_note}"
            self.statusBar().showMessage(msg)
            return
        angles = [e["angle_deg"] for e in self._fine_a]
        intens = [e["intensity"] for e in self._fine_a]
        a_start = self._fine_a_wrap_start
        a_force_wrap = bool(self._fine_a_force_wrap)
        if a_start is None:
            a_min = float(self.aMinSpin.value())
            a_max = float(self.aMaxSpin.value())
            a_start = a_min
            a_force_wrap = bool(a_min > a_max)
        model = self._build_wheel_model(
            angles,
            intens,
            wrap_start_deg=a_start,
            force_wrap=a_force_wrap,
        )
        if model is None:
            self.statusBar().showMessage("Calib A failed: invalid sweep data")
            return
        self._model_a = model
        self._angles_a = list(model["angles"])
        msg = f"Calib A done ({len(self._angles_a)} points)"
        if move_note:
            msg = f"{msg} | {move_note}"
        self.statusBar().showMessage(msg)
        self._update_calib_plot()
        self._update_linear_info()
        self._sync_controls()
        self._update_od_planner()

    def _on_fine_b_done(self, status: str):
        self._set_busy(False)
        move_note = self._move_wheels_to_min_positions()
        if status != "ok":
            msg = str(status)
            if move_note:
                msg = f"{msg} | {move_note}"
            self.statusBar().showMessage(msg)
            return
        angles = [e["angle_deg"] for e in self._fine_b]
        intens = [e["intensity"] for e in self._fine_b]
        b_start = self._fine_b_wrap_start
        b_force_wrap = bool(self._fine_b_force_wrap)
        if b_start is None:
            b_min = float(self.bMinSpin.value())
            b_max = float(self.bMaxSpin.value())
            b_start = b_min
            b_force_wrap = bool(b_min > b_max)
        model = self._build_wheel_model(
            angles,
            intens,
            wrap_start_deg=b_start,
            force_wrap=b_force_wrap,
        )
        if model is None:
            self.statusBar().showMessage("Calib B failed: invalid sweep data")
            return
        self._model_b = model
        self._angles_b = list(model["angles"])
        msg = f"Calib B done ({len(self._angles_b)} points)"
        if move_note:
            msg = f"{msg} | {move_note}"
        self.statusBar().showMessage(msg)
        self._update_calib_plot()
        self._update_linear_info()
        self._sync_controls()
        self._update_od_planner()

    def _on_list_point(self, idx: int, a_deg: float, b_deg: float, intensity: float, ref_level: float, _image):
        self._ensure_power_fit()
        power = self._counts_to_power(intensity)
        if not np.isfinite(power):
            power = None
        entry = {
            "a_deg": float(a_deg),
            "b_deg": float(b_deg),
            "power": float(power) if power is not None else None,
            "intensity": float(intensity),
            "ref_level": float(ref_level) if ref_level is not None and np.isfinite(ref_level) else None,
        }
        if 0 <= idx < len(self._results):
            self._results[idx] = entry
            self._set_table_row(idx, entry)
        else:
            self._results.append(entry)
            self._append_table_row(entry)
        if self._grid_data is not None and self._grid_map_a and self._grid_map_b:
            a_idx = self._grid_map_a.get(round(float(a_deg), 6))
            b_idx = self._grid_map_b.get(round(float(b_deg), 6))
            if a_idx is not None and b_idx is not None:
                try:
                    self._grid_data[b_idx, a_idx] = float(intensity)
                except Exception:
                    pass
        self._update_heatmap()
        self._update_power_plot()

    def _on_single_grid_point(self, b_deg: float, intensity: float, _image):
        idx = int(self._single_grid_idx)
        self._single_grid_idx += 1
        self._ensure_power_fit()
        power = self._counts_to_power(intensity)
        if not np.isfinite(power):
            power = None

        a_deg = 0.0
        if 0 <= idx < len(self._results):
            try:
                a_deg = float(self._results[idx].get("a_deg", 0.0))
            except Exception:
                a_deg = 0.0
        entry = {
            "a_deg": float(a_deg),
            "b_deg": float(b_deg),
            "power": float(power) if power is not None else None,
            "intensity": float(intensity),
            "ref_level": None,
        }

        if 0 <= idx < len(self._results):
            self._results[idx] = entry
            self._set_table_row(idx, entry)
        else:
            self._results.append(entry)
            self._append_table_row(entry)

        if self._grid_data is not None and self._grid_map_a and self._grid_map_b:
            a_idx = self._grid_map_a.get(round(float(a_deg), 6))
            b_idx = self._grid_map_b.get(round(float(b_deg), 6))
            if a_idx is not None and b_idx is not None:
                try:
                    self._grid_data[b_idx, a_idx] = float(intensity)
                except Exception:
                    pass
        self._update_heatmap()
        self._update_power_plot()

    def _on_grid_point(
        self,
        a_idx: int,
        b_idx: int,
        a_deg: float,
        b_deg: float,
        intensity: float,
        ref_level: float,
        _image,
    ):
        self._ensure_power_fit()
        power = self._counts_to_power(intensity)
        if not np.isfinite(power):
            power = None
        entry = {
            "a_deg": float(a_deg),
            "b_deg": float(b_deg),
            "power": float(power) if power is not None else None,
            "intensity": float(intensity),
            "ref_level": float(ref_level) if ref_level is not None and np.isfinite(ref_level) else None,
        }
        self._results.append(entry)
        if self._grid_data is not None:
            try:
                self._grid_data[b_idx, a_idx] = float(intensity)
            except Exception:
                pass
        self._append_table_row(entry)
        self._update_heatmap()
        self._update_power_plot()

    def _on_grid_done(self, status: str):
        self._set_busy(False)
        move_note = self._move_wheels_to_min_positions()
        if status == "ok":
            msg = "Sweep complete"
            if move_note:
                msg = f"{msg} | {move_note}"
            self.statusBar().showMessage(msg)
        elif status == "aborted":
            msg = "Sweep aborted"
            if move_note:
                msg = f"{msg} | {move_note}"
            self.statusBar().showMessage(msg)
        else:
            msg = str(status)
            if move_note:
                msg = f"{msg} | {move_note}"
            self.statusBar().showMessage(msg)

    def on_apply_od_plan(self) -> None:
        plan = self._compute_od_plan()
        if plan is None or not plan.get("pairs"):
            QtWidgets.QMessageBox.warning(self, "Apply OD Plan", "No OD plan available to apply.")
            return
        self._apply_pairs_to_table(plan["pairs"], use_loaded=True)
        self._od_plan_dirty = False
        self.statusBar().showMessage(f"Applied OD plan ({len(plan['pairs'])} points)")

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._is_busy():
            self._poll_timer.stop()
        else:
            if not self._poll_timer.isActive():
                self._poll_timer.start(int(POLL_MS))
        self._sync_controls()

    def _read_calib_csv(self, path: str):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(path)
        rows = []
        unit = None
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header_seen = False
            fmt = None
            for row in reader:
                if not row:
                    continue
                first = row[0].strip()
                if first.startswith("#"):
                    continue
                if not header_seen:
                    hdr = [cell.strip().lower() for cell in row]
                    low = ",".join(hdr)
                    if "position_deg" in low and "series" in low and "power" in low:
                        fmt = "single"
                        header_seen = True
                        for cell in hdr:
                            if cell.startswith("power_"):
                                unit = cell.split("power_")[-1]
                        continue
                    if "a" in low and "b" in low and "power" in low:
                        fmt = "dual"
                        header_seen = True
                        for cell in hdr:
                            if cell.startswith("power_"):
                                unit = cell.split("power_")[-1]
                        continue
                if len(row) < 2:
                    continue
                if fmt == "single":
                    try:
                        b_deg = float(row[0])
                        power = float(row[1])
                    except Exception:
                        continue
                    intensity = float("nan")
                    if len(row) >= 4:
                        try:
                            intensity = float(row[3])
                        except Exception:
                            intensity = float("nan")
                    rows.append({
                        "a_deg": 0.0,
                        "b_deg": b_deg,
                        "power": power,
                        "intensity": intensity,
                        "ref_level": None,
                    })
                    continue
                if len(row) < 4:
                    continue
                try:
                    a_deg = float(row[0])
                    b_deg = float(row[1])
                    power = float(row[2])
                    intensity = float(row[3])
                except Exception:
                    continue
                ref_level = None
                if len(row) >= 5:
                    try:
                        ref_level = float(row[4])
                    except Exception:
                        ref_level = None
                rows.append({
                    "a_deg": a_deg,
                    "b_deg": b_deg,
                    "power": power,
                    "intensity": intensity,
                    "ref_level": ref_level,
                })
        return rows, unit

    def _write_dual_csv(self, path: str, rows: List[dict]):
        header = f"a_deg,b_deg,power_{self._power_unit},intensity,base_level"
        lines = [header]
        for row in rows:
            a_deg = row.get("a_deg", float("nan"))
            b_deg = row.get("b_deg", float("nan"))
            power = row.get("power", float("nan"))
            intensity = row.get("intensity", float("nan"))
            base_level = row.get("ref_level", float("nan"))
            try:
                base_level = float(base_level)
            except Exception:
                base_level = float("nan")
            lines.append(f"{a_deg:.6f},{b_deg:.6f},{power:.9g},{intensity:.9g},{base_level:.9g}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}")

    def _write_single_base_csv(self, path: str, rows: List[dict]):
        header = f"position_deg,power_{self._power_unit},series,intensity"
        lines = [header]
        for row in rows:
            b_deg = row.get("b_deg", float("nan"))
            power = row.get("power", float("nan"))
            intensity = row.get("intensity", float("nan"))
            lines.append(f"{b_deg:.6f},{power:.9g},base,{intensity:.9g}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}")

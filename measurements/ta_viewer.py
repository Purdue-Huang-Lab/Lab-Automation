"""
measurements/ta_viewer.py

Simple viewer for transient absorption data.
Loads *para.txt + *sum.txt file pairs from a folder.
No gate / power / helicity dependence — file selection by name.
Cursor position is remembered per file.

Usage:
    python -m measurements.ta_viewer [folder]
    or via run_ta_viewer.ps1
"""
import sys
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.optimize import curve_fit

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k", useOpenGL=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Colour maps
# ─────────────────────────────────────────────────────────────────────────────

def _rdbu_lut() -> np.ndarray:
    """RdBu – blue=positive (bleach), red=negative (induced absorption)."""
    pts = np.linspace(0.0, 1.0, 11)
    rgba = np.array([
        [5,   48,  97, 255],
        [33, 102, 172, 255],
        [67, 147, 195, 255],
        [146, 197, 222, 255],
        [209, 229, 240, 255],
        [247, 247, 247, 255],
        [253, 219, 199, 255],
        [244, 165, 130, 255],
        [214,  96,  77, 255],
        [178,  24,  43, 255],
        [103,   0,  31, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(pts, rgba).getLookupTable(nPts=256)


def _viridis_lut() -> np.ndarray:
    return pg.colormap.get("viridis").getLookupTable(nPts=256)


LUTS = {"RdBu": None, "viridis": None, "gray": None}


def _get_lut(name: str) -> Optional[np.ndarray]:
    if name == "RdBu":
        if LUTS["RdBu"] is None:
            LUTS["RdBu"] = _rdbu_lut()
        return LUTS["RdBu"]
    if name == "viridis":
        if LUTS["viridis"] is None:
            LUTS["viridis"] = _viridis_lut()
        return LUTS["viridis"]
    return None  # gray → no LUT (pyqtgraph default grayscale)


# ─────────────────────────────────────────────────────────────────────────────
#  para.txt parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_para(path: Path) -> dict:
    """Parse *para.txt written by the pump-probe acquisition GUI."""
    data_lines: List[str] = []
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if not s.startswith("#"):
                data_lines.append(s)

    def _f(idx: int, default: float = 0.0) -> float:
        try:
            return float(data_lines[idx])
        except Exception:
            return default

    def _i(idx: int, default: int = 128) -> int:
        try:
            return int(float(data_lines[idx]))
        except Exception:
            return default

    delays = np.array(
        [float(x) for x in data_lines[0].split("\t") if x.strip()],
        dtype=np.float64,
    ) if data_lines else np.array([], dtype=np.float64)

    return dict(
        delays_ps=delays,
        pixel_size=_f(4, 0.07),
        crop_w=_i(5, 480),
        crop_h=_i(7, 128),
        t0_ps=_f(9, 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ScanRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanRecord:
    path_para:  Path
    path_sum:   Path
    stem:       str        # the * portion from *para.txt / *sum.txt
    delays_ps:  np.ndarray
    crop_h:     int        # spatial pixels  (N1)
    crop_w:     int        # spectral pixels (N2)
    t0_ps:      float
    pixel_size: float      # µm/pixel (spatial)
    _stack: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_delays(self) -> int:
        return len(self.delays_ps)

    def load_stack(self) -> Optional[np.ndarray]:
        if self._stack is not None:
            return self._stack
        try:
            raw = (
                pd.read_csv(
                    self.path_sum,
                    delimiter="\t",
                    header=None,
                    na_values=["--", "---"],
                )
                .fillna(0.0)
                .values.astype(np.float32)
            )
            N1, N2, Nt = self.crop_h, self.crop_w, self.n_delays
            stack = np.zeros((Nt, N1, N2), dtype=np.float32)
            for i in range(Nt):
                s = i * N2
                if s + N2 > raw.shape[0]:
                    break
                stack[i] = np.rot90(raw[s: s + N2, :N1])
            self._stack = stack
            return stack
        except Exception as exc:
            print(f"load_stack {self.path_sum.name}: {exc}")
            return None

    def value_at(self, t_idx: int, iy: int, ix: int) -> float:
        if self._stack is not None:
            stk = self._stack
            t = int(np.clip(t_idx, 0, stk.shape[0] - 1))
            y = int(np.clip(iy,    0, stk.shape[1] - 1))
            x = int(np.clip(ix,    0, stk.shape[2] - 1))
            return float(stk[t, y, x])
        N1, N2 = self.crop_h, self.crop_w
        t = int(np.clip(t_idx, 0, self.n_delays - 1))
        y = int(np.clip(iy, 0, N1 - 1))
        x = int(np.clip(ix, 0, N2 - 1))
        try:
            chunk = (
                pd.read_csv(
                    self.path_sum,
                    delimiter="\t",
                    header=None,
                    skiprows=t * N2,
                    nrows=N2,
                    na_values=["--", "---"],
                )
                .fillna(0.0)
                .values.astype(np.float32)
            )
            frame = np.rot90(chunk[:N2, :N1])
            return float(frame[y, x])
        except Exception:
            return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Folder scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(folder: str) -> List[ScanRecord]:
    """Discover all *para.txt + *sum.txt pairs and build ScanRecord list."""
    records = []
    for para_path in sorted(Path(folder).glob("*para.txt")):
        sum_path = para_path.with_name(
            para_path.name.replace("para.txt", "sum.txt"))
        if not sum_path.exists():
            continue
        try:
            meta = _parse_para(para_path)
        except Exception as exc:
            print(f"Skip {para_path.name}: {exc}")
            continue
        # Extract the shared stem: strip trailing _para or para
        stem = re.sub(r"[_\-]?para$", "", para_path.stem, flags=re.I)
        records.append(ScanRecord(
            path_para=para_path,
            path_sum=sum_path,
            stem=stem,
            delays_ps=meta["delays_ps"],
            crop_h=meta["crop_h"],
            crop_w=meta["crop_w"],
            t0_ps=meta["t0_ps"],
            pixel_size=meta["pixel_size"],
        ))
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian fitting helper
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_with_offset(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / (2.0 * stddev)) ** 2) + offset


# ─────────────────────────────────────────────────────────────────────────────
#  Pop-out TA map window
# ─────────────────────────────────────────────────────────────────────────────

class TAMapWindow(QtWidgets.QWidget):
    """Standalone window: ΔT/T vs (spectral axis, time) at one spatial position."""

    def __init__(self, title: str, data: np.ndarray, xax: np.ndarray,
                 delays: np.ndarray, x_label: str, lut, levels,
                 parent=None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.resize(760, 520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        gw = pg.GraphicsLayoutWidget()
        pl = gw.addPlot()
        img = pg.ImageItem()
        if lut is not None:
            img.setLookupTable(lut)
        Nt = data.shape[0]
        t  = delays[:Nt].astype(float)
        Nt_uni = int(np.clip(max(Nt * 4, 400), 400, 2000))
        t_uni  = np.linspace(t[0], t[-1], Nt_uni)
        from scipy.interpolate import interp1d
        data_uni = interp1d(t, data.astype(float), axis=0,
                            kind="linear", fill_value="extrapolate")(t_uni).astype(np.float32)
        img.setImage(data_uni, autoLevels=False)
        img.setLevels(levels)
        img.setRect(QtCore.QRectF(
            float(xax[0]),   float(t_uni[0]),
            float(xax[-1] - xax[0]),
            float(t_uni[-1] - t_uni[0]),
        ))
        pl.addItem(img)
        pl.setLabel("bottom", x_label)
        pl.setLabel("left", "Delay (ps)")
        for ax in ("bottom", "left"):
            pl.getAxis(ax).setPen(pg.mkPen("k"))
            pl.getAxis(ax).setTextPen(pg.mkPen("k"))
        layout.addWidget(gw)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class TAViewer(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TA Viewer")
        self.resize(1300, 980)

        self.records:      List[ScanRecord] = []
        self.current:      Optional[ScanRecord] = None
        self.corx:         int = 0
        self.cory:         int = 0
        self.wl_m:         float = 1.0
        self.wl_b:         float = 0.0
        self.smooth_sigma: float = 0.0
        self._cursor_pos:  dict = {}   # stem → (corx, cory)
        self._map_windows: List[QtWidgets.QWidget] = []

        self.G_STD:      Optional[np.ndarray] = None
        self.G_STD_ERR:  Optional[np.ndarray] = None
        self.G_AMP:      Optional[np.ndarray] = None
        self.G_MEAN:     Optional[np.ndarray] = None
        self.G_OFFSET:   Optional[np.ndarray] = None
        self.G_AMP_ERR:  Optional[np.ndarray] = None
        self.G_MEAN_ERR: Optional[np.ndarray] = None
        self.G_OFFSET_ERR: Optional[np.ndarray] = None

        self._build_ui()
        self._connect_signals()

        if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
            self._load_folder(sys.argv[1])

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_center_panel(), stretch=1)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(240)
        w.setMaximumWidth(300)
        layout = QtWidgets.QVBoxLayout(w)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 4, 0)

        # ── Folder ──
        fg = QtWidgets.QGroupBox("Data Folder")
        fl = QtWidgets.QGridLayout(fg)
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder…")
        fl.addWidget(self.folder_edit, 0, 0)
        self.folder_btn = QtWidgets.QPushButton("Browse")
        self.folder_btn.setFixedWidth(58)
        fl.addWidget(self.folder_btn, 0, 1)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setFixedWidth(45)
        fl.addWidget(self.load_btn, 0, 2)
        layout.addWidget(fg)

        # ── File list ──
        sg = QtWidgets.QGroupBox("Files")
        sl = QtWidgets.QVBoxLayout(sg)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.file_list.setAlternatingRowColors(True)
        sl.addWidget(self.file_list)
        layout.addWidget(sg, stretch=1)

        # ── Wavelength calibration ──
        cg = QtWidgets.QGroupBox("Wavelength Cal  (two-point linear)")
        cl = QtWidgets.QGridLayout(cg)
        cl.addWidget(QtWidgets.QLabel("px₁:"), 0, 0)
        self.wl_px1_edit = QtWidgets.QLineEdit("")
        self.wl_px1_edit.setPlaceholderText("pixel 1")
        self.wl_px1_edit.setFixedWidth(68)
        cl.addWidget(self.wl_px1_edit, 0, 1)
        cl.addWidget(QtWidgets.QLabel("nm₁:"), 0, 2)
        self.wl_wl1_edit = QtWidgets.QLineEdit("")
        self.wl_wl1_edit.setPlaceholderText("λ₁ (nm)")
        self.wl_wl1_edit.setFixedWidth(68)
        cl.addWidget(self.wl_wl1_edit, 0, 3)
        cl.addWidget(QtWidgets.QLabel("px₂:"), 1, 0)
        self.wl_px2_edit = QtWidgets.QLineEdit("")
        self.wl_px2_edit.setPlaceholderText("pixel 2")
        self.wl_px2_edit.setFixedWidth(68)
        cl.addWidget(self.wl_px2_edit, 1, 1)
        cl.addWidget(QtWidgets.QLabel("nm₂:"), 1, 2)
        self.wl_wl2_edit = QtWidgets.QLineEdit("")
        self.wl_wl2_edit.setPlaceholderText("λ₂ (nm)")
        self.wl_wl2_edit.setFixedWidth(68)
        cl.addWidget(self.wl_wl2_edit, 1, 3)
        self.wl_apply_btn = QtWidgets.QPushButton("Apply")
        self.wl_apply_btn.setFixedWidth(48)
        cl.addWidget(self.wl_apply_btn, 1, 4)
        layout.addWidget(cg)

        # ── Gaussian smoothing ──
        smg = QtWidgets.QGroupBox("Gaussian Smoothing")
        sml = QtWidgets.QHBoxLayout(smg)
        sml.addWidget(QtWidgets.QLabel("σ:"))
        self.smooth_edit = QtWidgets.QLineEdit("0")
        self.smooth_edit.setFixedWidth(50)
        self.smooth_edit.setToolTip("Gaussian sigma in pixels (0 = off).\nApplied to image and lineouts; dynamics use raw data.")
        sml.addWidget(self.smooth_edit)
        sml.addWidget(QtWidgets.QLabel("px"))
        self.smooth_apply_btn = QtWidgets.QPushButton("Apply")
        self.smooth_apply_btn.setFixedWidth(48)
        sml.addWidget(self.smooth_apply_btn)
        layout.addWidget(smg)

        # ── Gaussian Fit (Spatial Profile) ──
        gfg = QtWidgets.QGroupBox("Gaussian Fit (Spatial)")
        gfl = QtWidgets.QGridLayout(gfg)
        gfl.addWidget(QtWidgets.QLabel("A₀:"), 0, 0)
        self.gauss_amp_edit = QtWidgets.QLineEdit()
        self.gauss_amp_edit.setPlaceholderText("auto")
        self.gauss_amp_edit.setFixedWidth(68)
        self.gauss_amp_edit.setToolTip("Initial amplitude guess (leave blank for auto)")
        gfl.addWidget(self.gauss_amp_edit, 0, 1)
        gfl.addWidget(QtWidgets.QLabel("x₀ (µm):"), 0, 2)
        self.gauss_x0_edit = QtWidgets.QLineEdit()
        self.gauss_x0_edit.setPlaceholderText("auto")
        self.gauss_x0_edit.setFixedWidth(68)
        self.gauss_x0_edit.setToolTip("Initial center position guess in µm (leave blank for auto)")
        gfl.addWidget(self.gauss_x0_edit, 0, 3)
        gfl.addWidget(QtWidgets.QLabel("t start (ps):"), 1, 0)
        self.fit_t_start_edit = QtWidgets.QLineEdit()
        self.fit_t_start_edit.setPlaceholderText("first")
        self.fit_t_start_edit.setFixedWidth(68)
        self.fit_t_start_edit.setToolTip("Start time in ps (leave blank for first delay)")
        gfl.addWidget(self.fit_t_start_edit, 1, 1)
        gfl.addWidget(QtWidgets.QLabel("t end (ps):"), 1, 2)
        self.fit_t_end_edit = QtWidgets.QLineEdit()
        self.fit_t_end_edit.setPlaceholderText("last")
        self.fit_t_end_edit.setFixedWidth(68)
        self.fit_t_end_edit.setToolTip("End time in ps (leave blank for last delay)")
        gfl.addWidget(self.fit_t_end_edit, 1, 3)
        self.fit_here_btn = QtWidgets.QPushButton("Fit Here")
        self.fit_here_btn.setToolTip("Fit Gaussian to spatial profile at the current time step")
        gfl.addWidget(self.fit_here_btn, 2, 0, 1, 2)
        self.fit_all_btn = QtWidgets.QPushButton("Fit All Times")
        self.fit_all_btn.setToolTip("Fit Gaussian at every time step in [t start, t end] for the current spectral pixel")
        gfl.addWidget(self.fit_all_btn, 2, 2, 1, 2)
        layout.addWidget(gfg)

        # ── Display ──
        cmg = QtWidgets.QGroupBox("Display")
        cml = QtWidgets.QGridLayout(cmg)
        cml.addWidget(QtWidgets.QLabel("Colormap:"), 0, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["RdBu", "viridis", "gray"])
        cml.addWidget(self.cmap_combo, 0, 1)
        self.lock_axes_cb = QtWidgets.QCheckBox("Lock axes")
        self.lock_axes_cb.setToolTip("Freeze x/y ranges on all plots.")
        cml.addWidget(self.lock_axes_cb, 1, 0)
        self.reset_axes_btn = QtWidgets.QPushButton("Reset all scales")
        self.reset_axes_btn.setToolTip("Unlock and auto-scale all plots.")
        cml.addWidget(self.reset_axes_btn, 1, 1)
        self.open_map_btn = QtWidgets.QPushButton("Open TA map window")
        self.open_map_btn.setToolTip("Open a standalone window showing ΔT/T vs (spectral, time) at the current spatial cursor position.")
        cml.addWidget(self.open_map_btn, 2, 0, 1, 2)
        layout.addWidget(cmg)

        return w

    # ── Center panel ──────────────────────────────────────────────────────────

    def _build_center_panel(self) -> QtWidgets.QSplitter:
        center = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Time slider row
        ctrl_w = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(ctrl_w)
        cl.setContentsMargins(2, 2, 2, 2)
        cl.addWidget(QtWidgets.QLabel("t:"))
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        cl.addWidget(self.time_slider, stretch=1)
        self.time_label = QtWidgets.QLabel("-- ps  (step 0)")
        cl.addWidget(self.time_label)
        cl.addSpacing(12)
        cl.addWidget(QtWidgets.QLabel("Min:"))
        self.cmin_edit = QtWidgets.QLineEdit("-3")
        self.cmin_edit.setFixedWidth(55)
        cl.addWidget(self.cmin_edit)
        cl.addWidget(QtWidgets.QLabel("Max:"))
        self.cmax_edit = QtWidgets.QLineEdit("3")
        self.cmax_edit.setFixedWidth(55)
        cl.addWidget(self.cmax_edit)
        self.clevel_btn = QtWidgets.QPushButton("Apply")
        self.clevel_btn.setFixedWidth(48)
        cl.addWidget(self.clevel_btn)
        cl.addSpacing(12)
        self.cursor_label = QtWidgets.QLabel("Cursor: --")
        cl.addWidget(self.cursor_label)
        center.addWidget(ctrl_w)

        # TAM image
        self.img_gw = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_gw.addPlot()
        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(_get_lut("RdBu"))
        self.img_plot.addItem(self.img_item)
        self.img_plot.setLabel("bottom", "Spectral pixel")
        self.img_plot.setLabel("left", "Position (µm)")
        self.img_plot.setTitle("TA map (select a file to load)")
        dash = QtCore.Qt.DashLine
        self.ch_h = pg.InfiniteLine(angle=0,  pen=pg.mkPen("k", width=1, style=dash))
        self.ch_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1, style=dash))
        self.img_plot.addItem(self.ch_h)
        self.img_plot.addItem(self.ch_v)
        center.addWidget(self.img_gw)

        # 2×2 plot grid
        plots_w = QtWidgets.QWidget()
        pgrid = QtWidgets.QGridLayout(plots_w)
        pgrid.setSpacing(4)
        pgrid.setContentsMargins(0, 0, 0, 0)

        self.spec_pw = pg.PlotWidget(title="Spectrum (x-lineout at cursor row)")
        self.spec_pw.setLabel("bottom", "Spectral pixel")
        self.spec_pw.setLabel("left", "ΔT/T (mOD)")
        self.spec_curve = self.spec_pw.plot(pen=pg.mkPen((0, 130, 0), width=1.5))
        self.spec_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.spec_pw.addItem(self.spec_vline)
        pgrid.addWidget(self.spec_pw, 0, 0)

        self.spat_pw = pg.PlotWidget(title="Spatial profile (y-lineout at cursor column)")
        self.spat_pw.setLabel("bottom", "Position (µm)")
        self.spat_pw.setLabel("left", "ΔT/T (mOD)")
        self.spat_curve = self.spat_pw.plot(
            pen=None, symbol="o", symbolSize=4,
            symbolPen=(180, 50, 50), symbolBrush=(180, 50, 50),
        )
        self.spat_fit_curve = self.spat_pw.plot(pen=pg.mkPen("k", width=2))
        self.spat_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.spat_pw.addItem(self.spat_vline)
        pgrid.addWidget(self.spat_pw, 0, 1)

        self.dyn_pw = pg.PlotWidget(title="Dynamics at cursor")
        self.dyn_pw.setLabel("bottom", "Time delay (ps)")
        self.dyn_pw.setLabel("left", "ΔT/T (mOD)")
        self.dyn_pw.showGrid(x=True, y=True, alpha=0.25)
        self.dyn_curve = self.dyn_pw.plot(
            pen=pg.mkPen((0, 90, 200), width=1.8),
            symbol="o", symbolSize=5,
            symbolPen=(0, 90, 200), symbolBrush=(0, 90, 200),
        )
        self.dyn_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", style=dash))
        self.dyn_pw.addItem(self.dyn_vline)
        pgrid.addWidget(self.dyn_pw, 1, 0)

        self.diff_pw = pg.PlotWidget(title="Diffusion at cursor")
        self.diff_pw.setLabel("bottom", "Time delay (ps)")
        self.diff_pw.setLabel("left", "σ² (µm²)")
        self.diff_pw.showGrid(x=True, y=True, alpha=0.25)
        self.diff_curve = self.diff_pw.plot(
            pen=pg.mkPen((148, 103, 189), width=1.5),
            symbol="o", symbolSize=5,
            symbolPen=(148, 103, 189), symbolBrush=(148, 103, 189),
        )
        self.diff_err_item = pg.ErrorBarItem(pen=pg.mkPen((148, 103, 189), width=1))
        self.diff_pw.addItem(self.diff_err_item)
        pgrid.addWidget(self.diff_pw, 1, 1)

        center.addWidget(plots_w)
        center.setSizes([34, 380, 420])
        return center

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.folder_btn.clicked.connect(self._browse_folder)
        self.load_btn.clicked.connect(self._do_load)
        self.folder_edit.returnPressed.connect(self._do_load)
        self.file_list.currentRowChanged.connect(self._on_scan_selected)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.clevel_btn.clicked.connect(self._apply_levels)
        self.wl_apply_btn.clicked.connect(self._apply_wl_cal)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.smooth_apply_btn.clicked.connect(self._apply_smoothing)
        self.lock_axes_cb.stateChanged.connect(self._on_lock_axes_changed)
        self.reset_axes_btn.clicked.connect(self._reset_all_scales)
        self.open_map_btn.clicked.connect(self._open_ta_map_window)
        self.fit_here_btn.clicked.connect(self._fit_spatial_here)
        self.fit_all_btn.clicked.connect(self._fit_all_times)
        self.img_gw.scene().sigMouseClicked.connect(self._on_image_click)

    # ── Axis helpers ──────────────────────────────────────────────────────────

    def _x_axis(self, rec: ScanRecord) -> np.ndarray:
        px = np.arange(rec.crop_w, dtype=float)
        if self.wl_m != 1.0 or self.wl_b != 0.0:
            return self.wl_m * px + self.wl_b
        return px

    def _y_axis(self, rec: ScanRecord) -> np.ndarray:
        px = np.arange(rec.crop_h, dtype=float)
        return (px - rec.crop_h / 2.0) * rec.pixel_size

    def _levels(self):
        try:
            vmin = float(self.cmin_edit.text())
        except ValueError:
            vmin = -3.0
        try:
            vmax = float(self.cmax_edit.text())
        except ValueError:
            vmax = 3.0
        return vmin, vmax

    def _t_idx(self) -> int:
        return self.time_slider.value()

    def _x_label(self) -> str:
        return "Wavelength (nm)" if (self.wl_m != 1.0 or self.wl_b != 0.0) else "Spectral pixel"

    # ── Axis locking ──────────────────────────────────────────────────────────

    def _plot_viewboxes(self) -> list:
        vbs = [self.img_plot.getViewBox()]
        for pw in [self.spec_pw, self.spat_pw, self.dyn_pw, self.diff_pw]:
            vbs.append(pw.getViewBox())
        return vbs

    def _on_lock_axes_changed(self, state: int) -> None:
        locked = bool(state)
        for vb in self._plot_viewboxes():
            vb.enableAutoRange(not locked)

    def _reset_all_scales(self) -> None:
        self.lock_axes_cb.blockSignals(True)
        self.lock_axes_cb.setChecked(False)
        self.lock_axes_cb.blockSignals(False)
        for vb in self._plot_viewboxes():
            vb.enableAutoRange(True)

    # ── Folder loading ────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select data folder", self.folder_edit.text() or ""
        )
        if d:
            self.folder_edit.setText(d)
            self._do_load()

    def _do_load(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self.statusBar().showMessage("Invalid folder path.")
            return
        self.records = scan_folder(folder)
        self._populate_file_list()
        n = len(self.records)
        self.statusBar().showMessage(
            f"Found {n} scan{'s' if n != 1 else ''} in {folder}"
        )
        if n > 0:
            self.file_list.setCurrentRow(0)

    def _load_folder(self, folder: str) -> None:
        self.folder_edit.setText(folder)
        self._do_load()

    # ── File list ─────────────────────────────────────────────────────────────

    def _populate_file_list(self) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for r in self.records:
            self.file_list.addItem(r.stem)
        self.file_list.blockSignals(False)

    # ── Scan selection ────────────────────────────────────────────────────────

    def _on_scan_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.records):
            return
        rec = self.records[idx]
        self.statusBar().showMessage(f"Loading {rec.path_para.name} …")
        QtWidgets.QApplication.processEvents()

        stk = rec.load_stack()
        if stk is None:
            self.statusBar().showMessage(f"Failed to load {rec.path_para.name}")
            return

        # Save current cursor before switching
        if self.current is not None:
            self._cursor_pos[self.current.stem] = (self.corx, self.cory)

        self.current = rec

        # Restore cursor if we've been here before, else default to centre
        if rec.stem in self._cursor_pos:
            self.corx, self.cory = self._cursor_pos[rec.stem]
        else:
            self.corx = rec.crop_w // 2
            self.cory = rec.crop_h // 2

        # Clear fit results for new scan
        self.G_STD = self.G_STD_ERR = self.G_AMP = self.G_MEAN = None
        self.G_OFFSET = self.G_AMP_ERR = self.G_MEAN_ERR = self.G_OFFSET_ERR = None
        self.spat_fit_curve.setData([], [])
        self.diff_curve.setData([], [])
        self.diff_err_item.setData(x=np.array([]), y=np.array([]), height=np.array([]))

        t0_idx = int(np.argmin(np.abs(rec.delays_ps - rec.t0_ps))) if rec.n_delays else 0
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, rec.n_delays - 1))
        self.time_slider.setValue(t0_idx)
        self.time_slider.blockSignals(False)

        self._update_display(t0_idx)
        self.statusBar().showMessage(
            f"{rec.stem}  |  {rec.n_delays} delays  |  "
            f"{rec.crop_h}×{rec.crop_w} px  |  pixel={rec.pixel_size:.3f} µm"
        )

    def _on_time_changed(self, t_idx: int) -> None:
        self._update_display(t_idx)

    # ── Main display update ───────────────────────────────────────────────────

    def _update_display(self, t_idx: int) -> None:
        rec = self.current
        if rec is None:
            return
        stk = rec.load_stack()
        if stk is None:
            return

        t_idx = int(np.clip(t_idx, 0, stk.shape[0] - 1))
        frame = self._smooth_frame(stk[t_idx])

        xax = self._x_axis(rec)
        yax = self._y_axis(rec)
        vmin, vmax = self._levels()

        self.img_item.setImage(frame, autoLevels=False)
        self.img_item.setLevels([vmin, vmax])
        self.img_item.setRect(QtCore.QRectF(
            float(xax[0]), float(yax[0]),
            float(xax[-1] - xax[0]),
            float(yax[-1] - yax[0]),
        ))
        self.img_plot.setLabel("bottom", self._x_label())
        t_ps = rec.delays_ps[t_idx]
        self.img_plot.setTitle(f"{rec.stem}  |  t = {t_ps:.2f} ps  (step {t_idx})")

        ix = int(np.clip(self.corx, 0, rec.crop_w - 1))
        iy = int(np.clip(self.cory, 0, rec.crop_h - 1))
        self.ch_v.setValue(xax[ix])
        self.ch_h.setValue(yax[iy])
        self.time_label.setText(f"{t_ps:.2f} ps  (step {t_idx})")

        self.spec_curve.setData(xax, frame[iy, :])
        self.spec_vline.setValue(xax[ix])
        self.spec_pw.setLabel("bottom", self._x_label())
        self.spec_pw.setTitle(f"Spectrum at y = {yax[iy]:.2f} µm")

        self.spat_curve.setData(yax, frame[:, ix])
        self.spat_vline.setValue(yax[iy])
        xlbl = (f"λ={xax[ix]:.1f} nm" if (self.wl_m != 1.0 or self.wl_b != 0.0)
                else f"px {ix}")
        self.spat_pw.setTitle(f"Spatial profile at {xlbl}")

        if (self.G_STD is not None and t_idx < len(self.G_STD)
                and np.isfinite(self.G_STD[t_idx]) and self.G_STD[t_idx] > 0):
            y_fine = np.linspace(yax[0], yax[-1], 300)
            self.spat_fit_curve.setData(
                y_fine,
                _gaussian_with_offset(
                    y_fine, self.G_AMP[t_idx], self.G_MEAN[t_idx],
                    self.G_STD[t_idx], self.G_OFFSET[t_idx],
                ),
            )
        else:
            self.spat_fit_curve.setData([], [])

        self.dyn_curve.setData(rec.delays_ps, stk[:, iy, ix])
        self.dyn_vline.setValue(t_ps)
        self.dyn_pw.setTitle(f"Dynamics at {xlbl}, y = {yax[iy]:.2f} µm")

        val = frame[iy, ix]
        self.cursor_label.setText(
            f"({'λ=' if (self.wl_m != 1 or self.wl_b != 0) else 'px'}"
            f"{xax[ix]:.1f}, {yax[iy]:.2f} µm) = {val:.3g} mOD"
        )

    # ── Image click ───────────────────────────────────────────────────────────

    def _on_image_click(self, ev) -> None:
        if self.current is None:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if not self.img_plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = self.img_plot.vb.mapSceneToView(ev.scenePos())
        xax = self._x_axis(self.current)
        yax = self._y_axis(self.current)
        self.corx = int(np.clip(
            np.argmin(np.abs(xax - pos.x())), 0, self.current.crop_w - 1
        ))
        self.cory = int(np.clip(
            np.argmin(np.abs(yax - pos.y())), 0, self.current.crop_h - 1
        ))
        # Persist cursor for this file
        self._cursor_pos[self.current.stem] = (self.corx, self.cory)
        self._update_display(self._t_idx())

    # ── Levels / calibration / colormap ──────────────────────────────────────

    def _apply_levels(self) -> None:
        if self.current is None:
            return
        vmin, vmax = self._levels()
        self.img_item.setLevels([vmin, vmax])

    def _apply_wl_cal(self) -> None:
        try:
            px1 = float(self.wl_px1_edit.text())
            wl1 = float(self.wl_wl1_edit.text())
            px2 = float(self.wl_px2_edit.text())
            wl2 = float(self.wl_wl2_edit.text())
        except ValueError:
            self.statusBar().showMessage("Wavelength cal: enter all four values (px1, nm1, px2, nm2).")
            return
        if abs(px2 - px1) < 1e-9:
            self.statusBar().showMessage("Wavelength cal: px1 and px2 must be different.")
            return
        self.wl_m = (wl2 - wl1) / (px2 - px1)
        self.wl_b = wl1 - self.wl_m * px1
        self.statusBar().showMessage(
            f"Wavelength cal applied: m={self.wl_m:.4f} nm/px, b={self.wl_b:.2f} nm"
        )
        self._update_display(self._t_idx())

    def _on_cmap_changed(self, name: str) -> None:
        lut = _get_lut(name)
        self.img_item.setLookupTable(lut)
        if self.current is not None:
            vmin, vmax = self._levels()
            self.img_item.setLevels([vmin, vmax])

    def _open_ta_map_window(self) -> None:
        if self.current is None:
            self.statusBar().showMessage("No scan loaded.")
            return
        rec = self.current
        stk = rec.load_stack()
        if stk is None:
            self.statusBar().showMessage("Could not load stack.")
            return
        iy     = int(np.clip(self.cory, 0, rec.crop_h - 1))
        xax    = self._x_axis(rec)
        data   = stk[:, iy, :]
        lut    = _get_lut(self.cmap_combo.currentText())
        yax    = self._y_axis(rec)
        title  = f"{rec.stem}  |  y = {yax[iy]:.2f} µm"
        win = TAMapWindow(title, data, xax, rec.delays_ps,
                          self._x_label(), lut, self._levels())
        win.show()
        self._map_windows.append(win)

    # ── Gaussian smoothing ────────────────────────────────────────────────────

    def _smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.smooth_sigma <= 0:
            return frame
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(frame.astype(np.float32), sigma=self.smooth_sigma)
        except ImportError:
            self.statusBar().showMessage("Smoothing requires scipy — install with: pip install scipy")
            return frame

    def _apply_smoothing(self) -> None:
        try:
            sigma = max(0.0, float(self.smooth_edit.text()))
        except ValueError:
            return
        self.smooth_sigma = sigma
        self.statusBar().showMessage(
            f"Smoothing σ = {sigma} px" if sigma > 0 else "Smoothing off"
        )
        self._update_display(self._t_idx())

    # ── Gaussian spatial fitting ──────────────────────────────────────────────

    def _init_fit_arrays(self, n: int) -> None:
        self.G_STD = np.zeros(n)
        self.G_STD_ERR = np.zeros(n)
        self.G_AMP = np.zeros(n)
        self.G_MEAN = np.zeros(n)
        self.G_OFFSET = np.zeros(n)
        self.G_AMP_ERR = np.zeros(n)
        self.G_MEAN_ERR = np.zeros(n)
        self.G_OFFSET_ERR = np.zeros(n)

    def _spatial_p0(self, profile: np.ndarray, yax: np.ndarray):
        try:
            amp0 = float(self.gauss_amp_edit.text())
        except ValueError:
            amp0 = float(profile[np.argmax(np.abs(profile))])
        try:
            x0_0 = float(self.gauss_x0_edit.text())
        except ValueError:
            x0_0 = float(yax[np.argmax(np.abs(profile))])
        sig0 = max((float(yax[-1]) - float(yax[0])) / 10.0, 1e-3)
        off0 = float(np.median(profile))
        return [amp0, x0_0, sig0, off0]

    def _fit_spatial_here(self) -> None:
        rec = self.current
        if rec is None:
            return
        stk = rec.load_stack()
        if stk is None:
            return
        t_idx = self._t_idx()
        ix = int(np.clip(self.corx, 0, rec.crop_w - 1))
        frame = self._smooth_frame(stk[t_idx])
        yax = self._y_axis(rec)
        profile = frame[:, ix].astype(np.float64)
        p0 = self._spatial_p0(profile, yax)
        try:
            p, cov = curve_fit(_gaussian_with_offset, yax, profile,
                               p0=p0, maxfev=20000)
            e = np.sqrt(np.diag(cov))
            Nt = rec.n_delays
            if self.G_STD is None or len(self.G_STD) != Nt:
                self._init_fit_arrays(Nt)
            self.G_AMP[t_idx], self.G_MEAN[t_idx] = p[0], p[1]
            self.G_STD[t_idx], self.G_OFFSET[t_idx] = abs(p[2]), p[3]
            self.G_AMP_ERR[t_idx], self.G_MEAN_ERR[t_idx] = e[0], e[1]
            self.G_STD_ERR[t_idx], self.G_OFFSET_ERR[t_idx] = abs(e[2]), e[3]
            y_fine = np.linspace(yax[0], yax[-1], 300)
            self.spat_fit_curve.setData(y_fine, _gaussian_with_offset(y_fine, *p))
            self.statusBar().showMessage(
                f"Fit @ step {t_idx}: A={p[0]:.3g}  x₀={p[1]:.3g} µm  σ={abs(p[2]):.3g} µm"
            )
        except (RuntimeError, ValueError) as exc:
            self.statusBar().showMessage(f"Gaussian fit failed: {exc}")

    def _fit_all_times(self) -> None:
        rec = self.current
        if rec is None:
            return
        stk = rec.load_stack()
        if stk is None:
            return
        ix = int(np.clip(self.corx, 0, rec.crop_w - 1))
        yax = self._y_axis(rec)
        delays = rec.delays_ps
        Nt = len(delays)
        self._init_fit_arrays(Nt)

        try:
            t_start_ps = float(self.fit_t_start_edit.text())
            i_start = int(np.searchsorted(delays, t_start_ps))
        except ValueError:
            i_start = 0
        try:
            t_end_ps = float(self.fit_t_end_edit.text())
            i_end = int(np.searchsorted(delays, t_end_ps, side="right"))
        except ValueError:
            i_end = Nt
        i_start = int(np.clip(i_start, 0, Nt))
        i_end   = int(np.clip(i_end, i_start + 1, Nt))

        self.statusBar().showMessage(
            f"Fitting steps {i_start}–{i_end - 1} "
            f"({delays[i_start]:.2f} – {delays[i_end - 1]:.2f} ps)…"
        )
        QtWidgets.QApplication.processEvents()

        prev_p = None
        for t_idx in range(i_start, i_end):
            frame   = self._smooth_frame(stk[t_idx])
            profile = frame[:, ix].astype(np.float64)
            if prev_p is None:
                p0     = self._spatial_p0(profile, yax)
                bounds = (-np.inf, np.inf)
            else:
                prev_sig  = abs(prev_p[2])
                prev_mean = prev_p[1]
                p0 = [prev_p[0], prev_mean, prev_sig, prev_p[3]]
                bounds = (
                    [-np.inf, prev_mean - 2*prev_sig, prev_sig * 0.5, -np.inf],
                    [ np.inf, prev_mean + 2*prev_sig, prev_sig * 2.0,  np.inf],
                )
            try:
                p, cov = curve_fit(_gaussian_with_offset, yax, profile,
                                   p0=p0, bounds=bounds, maxfev=20000)
                e = np.sqrt(np.diag(cov))
                self.G_AMP[t_idx], self.G_MEAN[t_idx] = p[0], p[1]
                self.G_STD[t_idx], self.G_OFFSET[t_idx] = abs(p[2]), p[3]
                self.G_AMP_ERR[t_idx], self.G_MEAN_ERR[t_idx] = e[0], e[1]
                self.G_STD_ERR[t_idx], self.G_OFFSET_ERR[t_idx] = abs(e[2]), e[3]
                prev_p = p
            except (RuntimeError, ValueError):
                self.G_STD[t_idx] = self.G_STD_ERR[t_idx] = np.nan

        self._plot_diffusion()
        self._update_display(self._t_idx())
        xax = self._x_axis(rec)
        xlbl = (f"λ={xax[ix]:.1f} nm"
                if (self.wl_m != 1.0 or self.wl_b != 0.0) else f"px {ix}")
        self.statusBar().showMessage(f"Fit All Times complete at {xlbl}")

    def _plot_diffusion(self) -> None:
        if self.G_STD is None or self.current is None:
            self.diff_curve.setData([], [])
            self.diff_err_item.setData(x=np.array([]), y=np.array([]), height=np.array([]))
            return
        rec = self.current
        delays = rec.delays_ps
        Nt = min(len(self.G_STD), len(delays))
        t = delays[:Nt]
        gw  = self.G_STD[:Nt]
        gwe = self.G_STD_ERR[:Nt]
        m = np.isfinite(gw) & (gw > 0)
        t_m, w2 = t[m], gw[m] ** 2
        w2e = 2.0 * gw[m] * gwe[m]
        self.diff_curve.setData(t_m, w2)
        self.diff_err_item.setData(x=t_m, y=w2, height=w2e)
        xax = self._x_axis(rec)
        ix  = int(np.clip(self.corx, 0, rec.crop_w - 1))
        xlbl = (f"λ={xax[ix]:.1f} nm"
                if (self.wl_m != 1.0 or self.wl_b != 0.0) else f"px {ix}")
        self.diff_pw.setTitle(f"Diffusion at {xlbl}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    from PyQt5 import QtGui
    font = QtGui.QFont("Segoe UI", 9)
    app.setFont(font)
    win = TAViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

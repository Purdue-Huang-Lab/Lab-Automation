"""
data processing/polaron_ta_viewer.py

Polaron TA viewer — gate- and power-dependent transient absorption.

Two spectral features are tracked by selecting wavelength (or pixel) ranges.
Feature intensity = −min ΔT/T within the range (maximum bleach),
computed from the spatially-averaged spectrum at each time delay.

Bottom 2 × 2 panel layout:
  ┌──────────────────────┬──────────────────────┐
  │ Spectrum (x-lineout) │ Spatial profile      │
  ├──────────────────────┼──────────────────────┤
  │ Dynamics at cursor   │ Feature 1 & 2 dyn.   │
  └──────────────────────┴──────────────────────┘

Gate / power dependence plots both features at the selected time.

Usage:
    python "data processing/polaron_ta_viewer.py" [folder]
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

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")


# ─────────────────────────────────────────────────────────────────────────────
#  Colour maps
# ─────────────────────────────────────────────────────────────────────────────

def _rdbu_lut() -> np.ndarray:
    """RdBu_r – red=positive, blue=negative (standard TA convention)."""
    pts = np.linspace(0.0, 1.0, 11)
    rgba = np.array([
        [103,   0,  31, 255],
        [178,  24,  43, 255],
        [214,  96,  77, 255],
        [244, 165, 130, 255],
        [253, 219, 199, 255],
        [247, 247, 247, 255],
        [209, 229, 240, 255],
        [146, 197, 222, 255],
        [ 67, 147, 195, 255],
        [ 33, 102, 172, 255],
        [  5,  48,  97, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(pts, rgba).getLookupTable(nPts=256)


def _viridis_lut() -> np.ndarray:
    return pg.colormap.get("viridis").getLookupTable(nPts=256)


LUTS = {"RdBu_r": None, "viridis": None, "gray": None}


def _get_lut(name: str) -> Optional[np.ndarray]:
    if name == "RdBu_r":
        if LUTS["RdBu_r"] is None:
            LUTS["RdBu_r"] = _rdbu_lut()
        return LUTS["RdBu_r"]
    if name == "viridis":
        if LUTS["viridis"] is None:
            LUTS["viridis"] = _viridis_lut()
        return LUTS["viridis"]
    return None  # gray → no LUT


# ─────────────────────────────────────────────────────────────────────────────
#  Filename / para.txt parsers
# ─────────────────────────────────────────────────────────────────────────────

_VF_RE  = re.compile(r"Vf([pn])(\d+)d(\d+)V", re.I)
_VB_RE  = re.compile(r"Vb([pn])(\d+)d(\d+)V", re.I)
_ND_RE  = re.compile(r"ND(n?)(\d+)(?:p(\d+))?deg", re.I)
_HEL_RE = re.compile(r"(?<![A-Za-z])(LH|RH)(?![A-Za-z])", re.I)


def _volt(m: re.Match) -> float:
    sign = -1.0 if m.group(1).lower() == "n" else 1.0
    return sign * float(f"{m.group(2)}.{m.group(3)}")


def _ang(m: re.Match) -> float:
    sign = -1.0 if m.group(1).lower() == "n" else 1.0
    dec = m.group(3) or "0"
    return sign * float(f"{m.group(2)}.{dec}")


def _parse_fname(stem: str):
    """Return (v_front, v_back, angle, helicity) from filename; NaN/'' if absent."""
    vf = vb = angle = np.nan
    helicity = ""
    m = _VF_RE.search(stem)
    if m:
        vf = _volt(m)
    m = _VB_RE.search(stem)
    if m:
        vb = _volt(m)
    m = _ND_RE.search(stem)
    if m:
        angle = _ang(m)
    m = _HEL_RE.search(stem)
    if m:
        helicity = m.group(1).upper()
    return vf, vb, angle, helicity


def _parse_para(path: Path) -> dict:
    """
    Parse a *para.txt produced by _save_gate_step / _save_angle.

    Line layout (0-indexed, comment lines excluded):
      0  : tab-separated delays in ps
      1  : avg counts per delay
      2  : 0
      3  : scale factor (1.0)
      4  : pixel_size (µm/pixel)
      5  : crop_w  – spectral pixels  (≈ 480)
      6  : pixel_size (repeated)
      7  : crop_h  – spatial pixels   (≈ 128)
      8  : 0
      9  : t0_ps
      16 : angle (degrees)
      19 : n_frames
    Comment lines: # FrontGateV=…  # BackGateV=…  # Helicity=LH/RH
    """
    data_lines: List[str] = []
    comments: dict = {}
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("#"):
                if "=" in s:
                    k, v = s[1:].split("=", 1)
                    comments[k.strip()] = v.strip()
            else:
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

    hel = (comments.get("Helicity", "") or comments.get("helicity", "")).strip().upper()

    return dict(
        delays_ps  = delays,
        pixel_size = _f(4, 0.07),
        crop_w     = _i(5, 480),
        crop_h     = _i(7, 128),
        t0_ps      = _f(9, 0.0),
        angle      = _f(16, np.nan),
        v_front    = float(comments["FrontGateV"]) if "FrontGateV" in comments else np.nan,
        v_back     = float(comments["BackGateV"])  if "BackGateV"  in comments else np.nan,
        helicity   = hel,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ScanRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanRecord:
    path_para:  Path
    path_sum:   Path
    delays_ps:  np.ndarray
    crop_h:     int
    crop_w:     int
    t0_ps:      float
    pixel_size: float
    v_front:    float
    v_back:     float
    angle:      float
    power_uw:   float
    helicity:   str        # "LH", "RH", or ""
    _stack: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_delays(self) -> int:
        return len(self.delays_ps)

    @property
    def label(self) -> str:
        parts = []
        if self.helicity:
            parts.append(self.helicity)
        if np.isfinite(self.v_front):
            parts.append(f"Vf={self.v_front:+.3f} V")
        if np.isfinite(self.v_back):
            parts.append(f"Vb={self.v_back:+.3f} V")
        if np.isfinite(self.power_uw):
            parts.append(f"{self.power_uw:.3g} µW")
        elif np.isfinite(self.angle):
            parts.append(f"ND {self.angle:.1f}°")
        return "  ".join(parts) if parts else self.path_para.stem[:50]

    def load_stack(self) -> Optional[np.ndarray]:
        if self._stack is not None:
            return self._stack
        try:
            raw = (
                pd.read_csv(
                    self.path_sum, delimiter="\t", header=None,
                    na_values=["--", "---"],
                ).fillna(0.0).values.astype(np.float32)
            )
            N1, N2, Nt = self.crop_h, self.crop_w, self.n_delays
            stack = np.zeros((Nt, N1, N2), dtype=np.float32)
            for i in range(Nt):
                s = i * N2
                if s + N2 > raw.shape[0]:
                    break
                stack[i] = np.rot90(raw[s : s + N2, :N1])
            self._stack = stack
            return stack
        except Exception as exc:
            print(f"load_stack {self.path_sum.name}: {exc}")
            return None

    def frame_at(self, t_idx: int) -> Optional[np.ndarray]:
        """Return the (crop_h, crop_w) frame at time index t_idx (lazy, no full cache)."""
        if self._stack is not None:
            t = int(np.clip(t_idx, 0, self._stack.shape[0] - 1))
            return self._stack[t]
        N1, N2 = self.crop_h, self.crop_w
        t = int(np.clip(t_idx, 0, self.n_delays - 1))
        try:
            chunk = (
                pd.read_csv(
                    self.path_sum, delimiter="\t", header=None,
                    skiprows=t * N2, nrows=N2, na_values=["--", "---"],
                ).fillna(0.0).values.astype(np.float32)
            )
            return np.rot90(chunk[:N2, :N1])
        except Exception:
            return None

    def feat_intensity_at(self, t_idx: int, px_lo: int, px_hi: int) -> float:
        """−min of spatially-averaged spectrum in [px_lo, px_hi] at time t_idx."""
        frame = self.frame_at(t_idx)
        if frame is None:
            return np.nan
        avg = frame.mean(axis=0)          # (crop_w,)
        region = avg[px_lo : px_hi + 1]
        if len(region) == 0:
            return np.nan
        return float(-np.min(region))

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
                    self.path_sum, delimiter="\t", header=None,
                    skiprows=t * N2, nrows=N2, na_values=["--", "---"],
                ).fillna(0.0).values.astype(np.float32)
            )
            return float(np.rot90(chunk[:N2, :N1])[y, x])
        except Exception:
            return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Folder / XLSX helpers
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(folder: str, wheel_map: dict) -> List[ScanRecord]:
    records = []
    for para_path in sorted(Path(folder).glob("*para.txt")):
        sum_path = para_path.with_name(para_path.name.replace("para.txt", "sum.txt"))
        if not sum_path.exists():
            continue
        try:
            meta = _parse_para(para_path)
        except Exception as exc:
            print(f"Skip {para_path.name}: {exc}")
            continue

        fn_vf, fn_vb, fn_ang, fn_hel = _parse_fname(para_path.stem)
        vf  = meta["v_front"]  if np.isfinite(meta["v_front"]) else fn_vf
        vb  = meta["v_back"]   if np.isfinite(meta["v_back"])  else fn_vb
        ang = meta["angle"]    if np.isfinite(meta["angle"])   else fn_ang
        hel = meta["helicity"] or fn_hel

        puw = np.nan
        if np.isfinite(ang) and wheel_map:
            best = min(wheel_map, key=lambda k: abs(k - ang))
            if abs(best - ang) <= 0.5:
                puw = wheel_map[best]

        records.append(ScanRecord(
            path_para  = para_path,
            path_sum   = sum_path,
            delays_ps  = meta["delays_ps"],
            crop_h     = meta["crop_h"],
            crop_w     = meta["crop_w"],
            t0_ps      = meta["t0_ps"],
            pixel_size = meta["pixel_size"],
            v_front    = vf,
            v_back     = vb,
            angle      = ang,
            power_uw   = puw,
            helicity   = hel,
        ))
    return records


def load_wheel_map(xlsx_path: str) -> dict:
    try:
        df = pd.read_excel(xlsx_path, header=None)
        hdr_row = None
        for i, row in df.iterrows():
            if any("angle" in str(v).lower() for v in row):
                hdr_row = i
                break
        if hdr_row is None:
            return {}
        header = [str(v).lower().strip() for v in df.iloc[hdr_row]]
        data   = df.iloc[hdr_row + 1 :].reset_index(drop=True)
        data.columns = header
        angle_col = next((c for c in header if "angle" in c), None)
        power_col = next((c for c in header if "power" in c), None)
        if not angle_col or not power_col:
            return {}
        result: dict = {}
        for _, row in data.iterrows():
            try:
                a = float(row[angle_col])
                p = float(row[power_col])
                if np.isfinite(a) and np.isfinite(p):
                    result[a] = p
            except (ValueError, TypeError):
                pass
        return result
    except Exception as exc:
        print(f"XLSX load failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature colours (consistent across all plots)
# ─────────────────────────────────────────────────────────────────────────────
FEAT1_COL = (30,  100, 200)   # blue
FEAT2_COL = (220, 120,   0)   # orange
CUR_COL   = (200,  50,  50)   # red  – current scan highlight


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class PolaronTAViewer(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Polaron TA Viewer")
        self.resize(1750, 1050)

        self.records:      List[ScanRecord]               = []
        self.current:      Optional[ScanRecord]           = None
        self.wheel_map:    dict                           = {}
        self.corx:         int                            = 0
        self.cory:         int                            = 0
        self.wl_m:         float                         = 1.0
        self.wl_b:         float                         = 0.0
        self.smooth_sigma: float                         = 0.0
        self._unique_vf:   list                          = []
        self._unique_pwr:  list                          = []
        self._unique_hel:  list                          = []
        # Feature wavelength / pixel ranges: (lo, hi) in display units, or None
        self.feat1_range:  Optional[Tuple[float, float]] = None
        self.feat2_range:  Optional[Tuple[float, float]] = None
        # Derived helicity mode ("single" | "LH-RH" | "norm")
        self._hel_mode:      str                          = "single"
        self._lh_rec:        Optional[ScanRecord]         = None
        self._rh_rec:        Optional[ScanRecord]         = None
        self._derived_stack: Optional[np.ndarray]         = None

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

        cr_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        cr_split.addWidget(self._build_center_panel())
        cr_split.addWidget(self._build_right_panel())
        cr_split.setSizes([1050, 450])
        root.addWidget(cr_split, stretch=1)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(270)
        w.setMaximumWidth(330)
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

        # ── XLSX ──
        xg = QtWidgets.QGroupBox("ND Wheel → Power (XLSX)")
        xl = QtWidgets.QGridLayout(xg)
        self.xlsx_edit = QtWidgets.QLineEdit()
        self.xlsx_edit.setPlaceholderText("Optional wheel-power map…")
        xl.addWidget(self.xlsx_edit, 0, 0)
        self.xlsx_btn = QtWidgets.QPushButton("Browse")
        self.xlsx_btn.setFixedWidth(58)
        xl.addWidget(self.xlsx_btn, 0, 1)
        layout.addWidget(xg)

        # ── Scan table ──
        sg = QtWidgets.QGroupBox("Scans")
        sl = QtWidgets.QVBoxLayout(sg)
        self.scan_table = QtWidgets.QTableWidget(0, 6)
        self.scan_table.setHorizontalHeaderLabels(
            ["Vf (V)", "Vb (V)", "ND (°)", "Power (µW)", "Hel", "#t"]
        )
        hh = self.scan_table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.scan_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.scan_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.scan_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.scan_table.setAlternatingRowColors(True)
        self.scan_table.verticalHeader().setDefaultSectionSize(22)
        sl.addWidget(self.scan_table)
        layout.addWidget(sg, stretch=1)

        # ── Wavelength calibration ──
        cg = QtWidgets.QGroupBox("Wavelength Cal  (two-point linear)")
        cl = QtWidgets.QGridLayout(cg)
        cl.addWidget(QtWidgets.QLabel("px₁:"), 0, 0)
        self.wl_px1_edit = QtWidgets.QLineEdit("")
        self.wl_px1_edit.setPlaceholderText("pixel 1")
        self.wl_px1_edit.setFixedWidth(62)
        cl.addWidget(self.wl_px1_edit, 0, 1)
        cl.addWidget(QtWidgets.QLabel("nm₁:"), 0, 2)
        self.wl_wl1_edit = QtWidgets.QLineEdit("")
        self.wl_wl1_edit.setPlaceholderText("λ₁ (nm)")
        self.wl_wl1_edit.setFixedWidth(62)
        cl.addWidget(self.wl_wl1_edit, 0, 3)
        cl.addWidget(QtWidgets.QLabel("px₂:"), 1, 0)
        self.wl_px2_edit = QtWidgets.QLineEdit("")
        self.wl_px2_edit.setPlaceholderText("pixel 2")
        self.wl_px2_edit.setFixedWidth(62)
        cl.addWidget(self.wl_px2_edit, 1, 1)
        cl.addWidget(QtWidgets.QLabel("nm₂:"), 1, 2)
        self.wl_wl2_edit = QtWidgets.QLineEdit("")
        self.wl_wl2_edit.setPlaceholderText("λ₂ (nm)")
        self.wl_wl2_edit.setFixedWidth(62)
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
        self.smooth_edit.setToolTip(
            "Gaussian sigma in pixels (0 = off).\n"
            "Applied to image and lineouts; dynamics use raw data."
        )
        sml.addWidget(self.smooth_edit)
        sml.addWidget(QtWidgets.QLabel("px"))
        self.smooth_apply_btn = QtWidgets.QPushButton("Apply")
        self.smooth_apply_btn.setFixedWidth(48)
        sml.addWidget(self.smooth_apply_btn)
        layout.addWidget(smg)

        # ── Feature ranges ──
        frg = QtWidgets.QGroupBox("Feature Ranges  (−min ΔT/T, spatial avg.)")
        frl = QtWidgets.QGridLayout(frg)
        # Coloured labels to match plot curves
        lbl1 = QtWidgets.QLabel("Feat 1:")
        lbl1.setStyleSheet(f"color: rgb{FEAT1_COL};")
        frl.addWidget(lbl1, 0, 0)
        self.feat1_lo_edit = QtWidgets.QLineEdit("")
        self.feat1_lo_edit.setPlaceholderText("min")
        self.feat1_lo_edit.setFixedWidth(62)
        frl.addWidget(self.feat1_lo_edit, 0, 1)
        frl.addWidget(QtWidgets.QLabel("–"), 0, 2)
        self.feat1_hi_edit = QtWidgets.QLineEdit("")
        self.feat1_hi_edit.setPlaceholderText("max")
        self.feat1_hi_edit.setFixedWidth(62)
        frl.addWidget(self.feat1_hi_edit, 0, 3)

        lbl2 = QtWidgets.QLabel("Feat 2:")
        lbl2.setStyleSheet(f"color: rgb{FEAT2_COL};")
        frl.addWidget(lbl2, 1, 0)
        self.feat2_lo_edit = QtWidgets.QLineEdit("")
        self.feat2_lo_edit.setPlaceholderText("min")
        self.feat2_lo_edit.setFixedWidth(62)
        frl.addWidget(self.feat2_lo_edit, 1, 1)
        frl.addWidget(QtWidgets.QLabel("–"), 1, 2)
        self.feat2_hi_edit = QtWidgets.QLineEdit("")
        self.feat2_hi_edit.setPlaceholderText("max")
        self.feat2_hi_edit.setFixedWidth(62)
        frl.addWidget(self.feat2_hi_edit, 1, 3)
        self.feat_apply_btn = QtWidgets.QPushButton("Apply")
        self.feat_apply_btn.setFixedWidth(48)
        frl.addWidget(self.feat_apply_btn, 1, 4)
        layout.addWidget(frg)

        # ── Colormap + axis lock ──
        cmg = QtWidgets.QGroupBox("Display")
        cml = QtWidgets.QGridLayout(cmg)
        cml.addWidget(QtWidgets.QLabel("Colormap:"), 0, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["RdBu_r", "viridis", "gray"])
        cml.addWidget(self.cmap_combo, 0, 1)
        self.lock_axes_cb = QtWidgets.QCheckBox("Lock axes")
        self.lock_axes_cb.setToolTip(
            "Freeze x/y ranges on all plots so you can compare across\n"
            "conditions (LH/RH, gate, power) without auto-rescaling."
        )
        cml.addWidget(self.lock_axes_cb, 1, 0)
        self.reset_axes_btn = QtWidgets.QPushButton("Reset all scales")
        self.reset_axes_btn.setToolTip("Unlock and auto-scale all plots.")
        cml.addWidget(self.reset_axes_btn, 1, 1)
        layout.addWidget(cmg)

        return w

    # ── Center panel ──────────────────────────────────────────────────────────

    def _build_center_panel(self) -> QtWidgets.QSplitter:
        center = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Control rows: time slider (row 1) + Vf/Power/Hel scan sliders (row 2)
        ctrl_w = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(ctrl_w)
        cv.setContentsMargins(2, 2, 2, 2)
        cv.setSpacing(2)

        row1 = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(row1)
        cl.setContentsMargins(0, 0, 0, 0)
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
        cv.addWidget(row1)

        row2 = QtWidgets.QWidget()
        cl2 = QtWidgets.QHBoxLayout(row2)
        cl2.setContentsMargins(0, 0, 0, 0)
        cl2.addWidget(QtWidgets.QLabel("Hel:"))
        self.hel_combo = QtWidgets.QComboBox()
        self.hel_combo.setFixedWidth(55)
        self.hel_combo.setEnabled(False)
        self.hel_combo.setToolTip("Switch between LH and RH helicity (keeps Vf and Power fixed).")
        cl2.addWidget(self.hel_combo)
        cl2.addSpacing(8)
        cl2.addWidget(QtWidgets.QLabel("Vf:"))
        self.vf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vf_slider.setEnabled(False)
        cl2.addWidget(self.vf_slider, stretch=1)
        self.vf_label = QtWidgets.QLabel("--")
        self.vf_label.setMinimumWidth(85)
        cl2.addWidget(self.vf_label)
        cl2.addSpacing(12)
        cl2.addWidget(QtWidgets.QLabel("P:"))
        self.pwr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pwr_slider.setEnabled(False)
        cl2.addWidget(self.pwr_slider, stretch=1)
        self.pwr_label = QtWidgets.QLabel("--")
        self.pwr_label.setMinimumWidth(85)
        cl2.addWidget(self.pwr_label)
        cv.addWidget(row2)

        center.addWidget(ctrl_w)

        # TAM image
        self.img_gw = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_gw.addPlot()
        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(_get_lut("RdBu_r"))
        self.img_plot.addItem(self.img_item)
        self.img_plot.setLabel("bottom", "Spectral pixel")
        self.img_plot.setLabel("left", "Position (µm)")
        self.img_plot.setTitle("TA map (select a scan to load)")
        dash = QtCore.Qt.DashLine
        self.ch_h = pg.InfiniteLine(angle=0,  pen=pg.mkPen("k", width=1, style=dash))
        self.ch_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1, style=dash))
        self.img_plot.addItem(self.ch_h)
        self.img_plot.addItem(self.ch_v)
        center.addWidget(self.img_gw)

        # ── Top lineout row: spectrum | spatial profile ────────────────────────
        lo_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.spec_pw = pg.PlotWidget(title="Spectrum (x-lineout at cursor row)")
        self.spec_pw.setLabel("bottom", "Spectral pixel")
        self.spec_pw.setLabel("left", "ΔT/T (mOD)")
        self.spec_curve = self.spec_pw.plot(pen=pg.mkPen((0, 130, 0), width=1.5))
        self.spec_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.spec_pw.addItem(self.spec_vline)
        # Shaded regions for feature ranges
        self.feat1_region = pg.LinearRegionItem(
            values=[0, 0], brush=pg.mkBrush(*FEAT1_COL, 40),
            pen=pg.mkPen(*FEAT1_COL, width=1), movable=False,
        )
        self.feat2_region = pg.LinearRegionItem(
            values=[0, 0], brush=pg.mkBrush(*FEAT2_COL, 40),
            pen=pg.mkPen(*FEAT2_COL, width=1), movable=False,
        )
        self.feat1_region.setVisible(False)
        self.feat2_region.setVisible(False)
        self.spec_pw.addItem(self.feat1_region)
        self.spec_pw.addItem(self.feat2_region)
        lo_split.addWidget(self.spec_pw)

        self.spat_pw = pg.PlotWidget(title="Spatial profile (y-lineout at cursor column)")
        self.spat_pw.setLabel("bottom", "Position (µm)")
        self.spat_pw.setLabel("left", "ΔT/T (mOD)")
        self.spat_curve = self.spat_pw.plot(
            pen=None, symbol="o", symbolSize=4,
            symbolPen=(180, 50, 50), symbolBrush=(180, 50, 50),
        )
        self.spat_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.spat_pw.addItem(self.spat_vline)
        lo_split.addWidget(self.spat_pw)
        center.addWidget(lo_split)

        # ── Bottom row: dynamics at cursor | feature dynamics ──────────────────
        bot_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

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
        self.dyn_lh_curve = self.dyn_pw.plot(
            pen=pg.mkPen((0, 180, 0), width=1, style=QtCore.Qt.DashLine), symbol=None,
        )
        self.dyn_rh_curve = self.dyn_pw.plot(
            pen=pg.mkPen((220, 140, 0), width=1, style=QtCore.Qt.DashLine), symbol=None,
        )
        bot_split.addWidget(self.dyn_pw)

        self.feat_pw = pg.PlotWidget(
            title="Feature dynamics  (blue = Feat 1,  orange = Feat 2)"
        )
        self.feat_pw.setLabel("bottom", "Time delay (ps)")
        self.feat_pw.setLabel("left", "−min ΔT/T (mOD)")
        self.feat_pw.showGrid(x=True, y=True, alpha=0.25)
        self.feat1_curve = self.feat_pw.plot(
            pen=pg.mkPen(FEAT1_COL, width=2),
            symbol="o", symbolSize=5,
            symbolPen=FEAT1_COL, symbolBrush=FEAT1_COL,
        )
        self.feat2_curve = self.feat_pw.plot(
            pen=pg.mkPen(FEAT2_COL, width=2),
            symbol="s", symbolSize=5,
            symbolPen=FEAT2_COL, symbolBrush=FEAT2_COL,
        )
        self.feat_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", style=dash))
        self.feat_pw.addItem(self.feat_vline)
        bot_split.addWidget(self.feat_pw)

        center.addWidget(bot_split)
        center.setSizes([34, 380, 210, 240])
        return center

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QtWidgets.QSplitter:
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.gate_pw = pg.PlotWidget(
            title="Gate dep  (● Feat 1,  ■ Feat 2,  red = current)"
        )
        self.gate_pw.setLabel("bottom", "V_front (V)")
        self.gate_pw.setLabel("left", "−min ΔT/T (mOD)")
        self.gate_pw.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.gate_pw)

        self.pwr_pw = pg.PlotWidget(
            title="Power dep  (● Feat 1,  ■ Feat 2,  red = current)"
        )
        self.pwr_pw.setLabel("bottom", "Power (µW)")
        self.pwr_pw.setLabel("left", "−min ΔT/T (mOD)")
        self.pwr_pw.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.pwr_pw)

        btn_w = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(btn_w)
        bl.setContentsMargins(4, 2, 4, 2)
        self.dep_btn = QtWidgets.QPushButton("Update Gate/Power Plots")
        self.dep_btn.setToolTip(
            "Extracts feature intensities at the current cursor position and time "
            "from ALL loaded scans.\nMay load remaining stacks on first use."
        )
        bl.addWidget(self.dep_btn)
        right.addWidget(btn_w)

        right.setSizes([430, 430, 40])
        return right

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.folder_btn.clicked.connect(self._browse_folder)
        self.load_btn.clicked.connect(self._do_load)
        self.folder_edit.returnPressed.connect(self._do_load)
        self.xlsx_btn.clicked.connect(self._browse_xlsx)
        self.xlsx_edit.returnPressed.connect(self._load_xlsx_from_edit)
        self.scan_table.itemSelectionChanged.connect(self._on_scan_selected)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.clevel_btn.clicked.connect(self._apply_levels)
        self.wl_apply_btn.clicked.connect(self._apply_wl_cal)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.dep_btn.clicked.connect(self._update_dep_plots)
        self.feat_apply_btn.clicked.connect(self._apply_feat_ranges)
        self.vf_slider.valueChanged.connect(self._on_vf_slider_changed)
        self.pwr_slider.valueChanged.connect(self._on_pwr_slider_changed)
        self.hel_combo.currentTextChanged.connect(self._on_hel_changed)
        self.smooth_apply_btn.clicked.connect(self._apply_smoothing)
        self.lock_axes_cb.stateChanged.connect(self._on_lock_axes_changed)
        self.reset_axes_btn.clicked.connect(self._reset_all_scales)
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

    def _feat_px_range(
        self, rec: ScanRecord, lo: float, hi: float
    ) -> Optional[Tuple[int, int]]:
        """
        Convert a display-unit range [lo, hi] to pixel indices [px_lo, px_hi].
        Works for both calibrated (nm) and uncalibrated (pixel) axes.
        Returns None if the range contains no pixels.
        """
        xax = self._x_axis(rec)
        wl_lo, wl_hi = min(lo, hi), max(lo, hi)
        idxs = np.where((xax >= wl_lo) & (xax <= wl_hi))[0]
        if len(idxs) == 0:
            return None
        return int(idxs[0]), int(idxs[-1])

    # ── Axis locking ──────────────────────────────────────────────────────────

    def _plot_viewboxes(self) -> list:
        """All ViewBox instances managed by Lock axes / Reset scales."""
        vbs = [self.img_plot.getViewBox()]
        for pw in [self.spec_pw, self.spat_pw, self.dyn_pw, self.feat_pw,
                   self.gate_pw, self.pwr_pw]:
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

    # ── Folder / XLSX loading ─────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select data folder", self.folder_edit.text() or ""
        )
        if d:
            self.folder_edit.setText(d)
            self._do_load()

    def _browse_xlsx(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ND-wheel power map",
            os.path.dirname(self.xlsx_edit.text() or ""),
            "Excel (*.xlsx *.xls);;All (*.*)",
        )
        if p:
            self.xlsx_edit.setText(p)
            self._load_xlsx_from_edit()

    def _load_xlsx_from_edit(self) -> None:
        p = self.xlsx_edit.text().strip()
        if p and os.path.isfile(p):
            self.wheel_map = load_wheel_map(p)
            self.statusBar().showMessage(
                f"Wheel map: {len(self.wheel_map)} angle→power entries from {Path(p).name}"
            )
        else:
            self.wheel_map = {}
        if self.records:
            self._apply_wheel_map()

    def _apply_wheel_map(self) -> None:
        """Re-apply the current wheel_map to all loaded records and refresh the table."""
        for r in self.records:
            r.power_uw = np.nan
            if np.isfinite(r.angle) and self.wheel_map:
                best = min(self.wheel_map, key=lambda k: abs(k - r.angle))
                if abs(best - r.angle) <= 0.5:
                    r.power_uw = self.wheel_map[best]
        self._populate_table()
        self._rebuild_sliders()

    def _do_load(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self.statusBar().showMessage("Invalid folder path.")
            return
        self._load_xlsx_from_edit()
        self.records = scan_folder(folder, self.wheel_map)
        self._populate_table()
        self._rebuild_sliders()
        n = len(self.records)
        self.statusBar().showMessage(
            f"Found {n} scan{'s' if n != 1 else ''} in {folder}"
        )
        if n > 0:
            self.scan_table.selectRow(0)

    def _load_folder(self, folder: str) -> None:
        self.folder_edit.setText(folder)
        self._do_load()

    # ── Scan table ────────────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        self.scan_table.setRowCount(len(self.records))
        for i, r in enumerate(self.records):
            def _cell(txt: str):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                return item
            self.scan_table.setItem(i, 0, _cell(f"{r.v_front:+.3f}" if np.isfinite(r.v_front) else "--"))
            self.scan_table.setItem(i, 1, _cell(f"{r.v_back:+.3f}"  if np.isfinite(r.v_back)  else "--"))
            self.scan_table.setItem(i, 2, _cell(f"{r.angle:.1f}"    if np.isfinite(r.angle)    else "--"))
            self.scan_table.setItem(i, 3, _cell(f"{r.power_uw:.3g}" if np.isfinite(r.power_uw) else "--"))
            self.scan_table.setItem(i, 4, _cell(r.helicity or "--"))
            self.scan_table.setItem(i, 5, _cell(str(r.n_delays)))

    # ── Scan selection ────────────────────────────────────────────────────────

    def _on_scan_selected(self) -> None:
        idx = self.scan_table.currentRow()
        if idx < 0 or idx >= len(self.records):
            return
        rec = self.records[idx]
        self.statusBar().showMessage(f"Loading {rec.path_para.name} …")
        QtWidgets.QApplication.processEvents()

        stk = rec.load_stack()
        if stk is None:
            self.statusBar().showMessage(f"Failed to load {rec.path_para.name}")
            return

        self.current = rec
        self._sync_sliders_to_current()
        self.corx = rec.crop_w // 2
        self.cory = rec.crop_h // 2

        t0_idx = int(np.argmin(np.abs(rec.delays_ps - rec.t0_ps))) if rec.n_delays else 0
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, rec.n_delays - 1))
        self.time_slider.setValue(t0_idx)
        self.time_slider.blockSignals(False)

        self._update_display(t0_idx)
        self.statusBar().showMessage(
            f"{rec.label}  |  {rec.n_delays} delays  |  "
            f"{rec.crop_h}×{rec.crop_w} px  |  pixel={rec.pixel_size:.3f} µm"
        )

    def _on_time_changed(self, t_idx: int) -> None:
        self._update_display(t_idx)

    # ── Main display update ───────────────────────────────────────────────────

    def _update_display(self, t_idx: int) -> None:
        rec = self.current
        if rec is None:
            return
        raw_stk = rec.load_stack()
        if raw_stk is None:
            return

        # In derived mode use the pre-computed derived stack for image/lineouts
        stk = self._derived_stack if self._derived_stack is not None else raw_stk

        t_idx = int(np.clip(t_idx, 0, stk.shape[0] - 1))
        frame = self._smooth_frame(stk[t_idx])   # smoothed for image + lineouts

        xax = self._x_axis(rec)
        yax = self._y_axis(rec)
        vmin, vmax = self._levels()

        # ── Image ──
        self.img_item.setImage(frame, autoLevels=False)
        self.img_item.setLevels([vmin, vmax])
        self.img_item.setRect(QtCore.QRectF(
            float(xax[0]), float(yax[0]),
            float(xax[-1] - xax[0]),
            float(yax[-1] - yax[0]),
        ))
        self.img_plot.setLabel("bottom", self._x_label())
        delays = self._current_delays()
        t_ps = delays[t_idx] if len(delays) > t_idx else 0.0
        mode_sfx = f"  [{self._hel_mode}]" if self._hel_mode != "single" else ""
        self.img_plot.setTitle(f"{rec.label}{mode_sfx}  |  t = {t_ps:.2f} ps  (step {t_idx})")

        ix = int(np.clip(self.corx, 0, rec.crop_w - 1))
        iy = int(np.clip(self.cory, 0, rec.crop_h - 1))
        self.ch_v.setValue(xax[ix])
        self.ch_h.setValue(yax[iy])
        self.time_label.setText(f"{t_ps:.2f} ps  (step {t_idx})")

        # ── Spectrum ──
        self.spec_curve.setData(xax, frame[iy, :])
        self.spec_vline.setValue(xax[ix])
        self.spec_pw.setLabel("bottom", self._x_label())
        self.spec_pw.setTitle(f"Spectrum at y = {yax[iy]:.2f} µm")
        self._update_feat_regions(xax)

        # ── Spatial profile ──
        self.spat_curve.setData(yax, frame[:, ix])
        self.spat_vline.setValue(yax[iy])
        xlbl = f"λ={xax[ix]:.1f} nm" if (self.wl_m != 1.0 or self.wl_b != 0.0) else f"px {ix}"
        self.spat_pw.setTitle(f"Spatial profile at {xlbl}")

        # ── Dynamics at cursor ──
        n = len(delays)
        self.dyn_curve.setData(delays, stk[:n, iy, ix])
        self.dyn_vline.setValue(t_ps)
        self.dyn_pw.setTitle(f"Dynamics at {xlbl}, y = {yax[iy]:.2f} µm")
        # Ghost LH/RH curves shown in derived mode
        if self._derived_stack is not None and self._lh_rec is not None and self._rh_rec is not None:
            lh_stk = self._lh_rec.load_stack()
            rh_stk = self._rh_rec.load_stack()
            if lh_stk is not None:
                self.dyn_lh_curve.setData(self._lh_rec.delays_ps[:n], lh_stk[:n, iy, ix])
            else:
                self.dyn_lh_curve.setData([], [])
            if rh_stk is not None:
                self.dyn_rh_curve.setData(self._rh_rec.delays_ps[:n], rh_stk[:n, iy, ix])
            else:
                self.dyn_rh_curve.setData([], [])
        else:
            self.dyn_lh_curve.setData([], [])
            self.dyn_rh_curve.setData([], [])

        # ── Feature dynamics ──
        self._update_feature_plot(rec, stk, t_ps)

        val = frame[iy, ix]
        self.cursor_label.setText(
            f"({'λ=' if (self.wl_m!=1 or self.wl_b!=0) else 'px'}"
            f"{xax[ix]:.1f}, {yax[iy]:.2f} µm) = {val:.3g} mOD"
        )

    def _update_feat_regions(self, xax: np.ndarray) -> None:
        """Update the shaded region overlays on the spectrum plot."""
        if self.feat1_range is not None:
            lo, hi = min(self.feat1_range), max(self.feat1_range)
            self.feat1_region.setRegion([lo, hi])
            self.feat1_region.setVisible(True)
        else:
            self.feat1_region.setVisible(False)

        if self.feat2_range is not None:
            lo, hi = min(self.feat2_range), max(self.feat2_range)
            self.feat2_region.setRegion([lo, hi])
            self.feat2_region.setVisible(True)
        else:
            self.feat2_region.setVisible(False)

    def _update_feature_plot(
        self, rec: ScanRecord, stk: np.ndarray, t_ps: float
    ) -> None:
        """Compute and plot feature 1 and feature 2 dynamics from the loaded stack."""
        delays = self._current_delays()
        n = len(delays)
        avg_spectra = stk[:n].mean(axis=1)   # (T, crop_w)

        for curve, feat_range in (
            (self.feat1_curve, self.feat1_range),
            (self.feat2_curve, self.feat2_range),
        ):
            if feat_range is None:
                curve.setData([], [])
                continue
            pr = self._feat_px_range(rec, *feat_range)
            if pr is None:
                curve.setData([], [])
                continue
            px_lo, px_hi = pr
            region = avg_spectra[:, px_lo : px_hi + 1]   # (T, n_px)
            dyn = -np.min(region, axis=1)                 # (T,)
            curve.setData(delays, dyn)

        self.feat_vline.setValue(t_ps)

        labels = []
        if self.feat1_range is not None:
            labels.append(f"F1: {self.feat1_range[0]:.1f}–{self.feat1_range[1]:.1f}")
        if self.feat2_range is not None:
            labels.append(f"F2: {self.feat2_range[0]:.1f}–{self.feat2_range[1]:.1f}")
        unit = self._x_label()
        suffix = f"  [{unit}]" if labels else ""
        self.feat_pw.setTitle(
            "Feature dynamics  (blue = Feat 1,  orange = Feat 2)"
            + (f"   {',  '.join(labels)}{suffix}" if labels else "")
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
        self._update_display(self._t_idx())

    # ── Levels / calibration / colormap / feature ranges ─────────────────────

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

    # ── Gaussian smoothing ────────────────────────────────────────────────────

    def _smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.smooth_sigma <= 0:
            return frame
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(frame.astype(np.float32), sigma=self.smooth_sigma)
        except ImportError:
            self.statusBar().showMessage(
                "Smoothing requires scipy — install with: pip install scipy"
            )
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

    # ── Vf / Power / Hel scan sliders ────────────────────────────────────────

    def _rebuild_sliders(self) -> None:
        self._unique_vf  = sorted({r.v_front  for r in self.records if np.isfinite(r.v_front)})
        self._unique_pwr = sorted({r.power_uw for r in self.records if np.isfinite(r.power_uw)})
        self._unique_hel = sorted({r.helicity for r in self.records if r.helicity})

        for slider, vals in (
            (self.vf_slider,  self._unique_vf),
            (self.pwr_slider, self._unique_pwr),
        ):
            slider.blockSignals(True)
            if len(vals) > 1:
                slider.setMinimum(0)
                slider.setMaximum(len(vals) - 1)
                slider.setEnabled(True)
            else:
                slider.setEnabled(False)
            slider.blockSignals(False)

        # Helicity combo
        self.hel_combo.blockSignals(True)
        self.hel_combo.clear()
        both_hel = "LH" in self._unique_hel and "RH" in self._unique_hel
        if len(self._unique_hel) > 1:
            self.hel_combo.addItems(self._unique_hel)
            if both_hel:
                self.hel_combo.addItems(["LH-RH", "norm"])
            self.hel_combo.setEnabled(True)
        else:
            if self._unique_hel:
                self.hel_combo.addItem(self._unique_hel[0])
            self.hel_combo.setEnabled(False)
        self.hel_combo.blockSignals(False)

        self._sync_sliders_to_current()

    def _sync_sliders_to_current(self) -> None:
        rec = self.current

        if self._unique_vf and rec is not None and np.isfinite(rec.v_front):
            idx = min(range(len(self._unique_vf)),
                      key=lambda i: abs(self._unique_vf[i] - rec.v_front))
            self.vf_slider.blockSignals(True)
            self.vf_slider.setValue(idx)
            self.vf_slider.blockSignals(False)
            self.vf_label.setText(f"{rec.v_front:+.3f} V")
        else:
            self.vf_label.setText("--" if not self._unique_vf else
                                  f"{self._unique_vf[self.vf_slider.value()]:+.3f} V")

        if self._unique_pwr and rec is not None and np.isfinite(rec.power_uw):
            idx = min(range(len(self._unique_pwr)),
                      key=lambda i: abs(self._unique_pwr[i] - rec.power_uw))
            self.pwr_slider.blockSignals(True)
            self.pwr_slider.setValue(idx)
            self.pwr_slider.blockSignals(False)
            self.pwr_label.setText(f"{rec.power_uw:.3g} µW")
        else:
            self.pwr_label.setText("--" if not self._unique_pwr else
                                   f"{self._unique_pwr[self.pwr_slider.value()]:.3g} µW")

        if self._hel_mode != "single":
            self.hel_combo.blockSignals(True)
            self.hel_combo.setCurrentText(self._hel_mode)
            self.hel_combo.blockSignals(False)
        elif self._unique_hel and rec is not None and rec.helicity in self._unique_hel:
            self.hel_combo.blockSignals(True)
            self.hel_combo.setCurrentText(rec.helicity)
            self.hel_combo.blockSignals(False)

    def _find_record_row(
        self, target_vf: float, target_pwr: float, target_hel: str = ""
    ) -> int:
        best_row, best_dist = -1, float("inf")
        for i, r in enumerate(self.records):
            d = 0.0
            # Helicity mismatch is a hard penalty
            if target_hel and r.helicity and r.helicity != target_hel:
                d += 1e8
            if np.isfinite(target_vf) and np.isfinite(r.v_front):
                d += (r.v_front - target_vf) ** 2
            elif np.isfinite(target_vf) != np.isfinite(r.v_front):
                d += 1e6
            if np.isfinite(target_pwr) and np.isfinite(r.power_uw):
                ref = max(abs(target_pwr), 1e-9)
                d += ((r.power_uw - target_pwr) / ref) ** 2
            elif np.isfinite(target_pwr) != np.isfinite(r.power_uw):
                d += 1e6
            if d < best_dist:
                best_dist, best_row = d, i
        return best_row

    def _switch_to_scan(self, row: int) -> None:
        """
        Switch to records[row] while preserving:
          - current time in ps (snaps to nearest step in the new scan)
          - cursor pixel position (corx, cory unchanged)
        Updates the time slider range and redraws.
        """
        # Capture current time in ps before switching
        old_rec = self.current
        old_t_ps: Optional[float] = None
        if old_rec is not None and old_rec.n_delays > 0:
            old_t_ps = old_rec.delays_ps[int(np.clip(
                self._t_idx(), 0, old_rec.n_delays - 1
            ))]

        rec = self.records[row]
        self.scan_table.blockSignals(True)
        self.scan_table.selectRow(row)
        self.scan_table.blockSignals(False)
        self.current = rec
        self._sync_sliders_to_current()

        stk = rec.load_stack()
        if stk is None:
            return

        # Find nearest time step in new scan
        if old_t_ps is not None and rec.n_delays > 0:
            new_t_idx = int(np.argmin(np.abs(rec.delays_ps - old_t_ps)))
        else:
            new_t_idx = 0

        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, rec.n_delays - 1))
        self.time_slider.setValue(new_t_idx)
        self.time_slider.blockSignals(False)

        self._update_display(new_t_idx)

    def _on_vf_slider_changed(self, idx: int) -> None:
        if not self._unique_vf:
            return
        target_vf  = self._unique_vf[idx]
        target_pwr = self.current.power_uw if self.current is not None else np.nan
        self.vf_label.setText(f"{target_vf:+.3f} V")
        if self._hel_mode != "single":
            lh_row = self._find_record_row(target_vf, target_pwr, "LH")
            if lh_row >= 0:
                self.current = self.records[lh_row]
                self._sync_sliders_to_current()
                self._enter_derived_mode(self._hel_mode)
            return
        target_hel = self.hel_combo.currentText() if self.hel_combo.isEnabled() else ""
        if target_hel not in self._unique_hel:
            target_hel = ""
        row = self._find_record_row(target_vf, target_pwr, target_hel)
        if row >= 0:
            self._switch_to_scan(row)

    def _on_pwr_slider_changed(self, idx: int) -> None:
        if not self._unique_pwr:
            return
        target_pwr = self._unique_pwr[idx]
        target_vf  = self.current.v_front  if self.current is not None else np.nan
        self.pwr_label.setText(f"{target_pwr:.3g} µW")
        if self._hel_mode != "single":
            lh_row = self._find_record_row(target_vf, target_pwr, "LH")
            if lh_row >= 0:
                self.current = self.records[lh_row]
                self._sync_sliders_to_current()
                self._enter_derived_mode(self._hel_mode)
            return
        target_hel = self.hel_combo.currentText() if self.hel_combo.isEnabled() else ""
        if target_hel not in self._unique_hel:
            target_hel = ""
        row = self._find_record_row(target_vf, target_pwr, target_hel)
        if row >= 0:
            self._switch_to_scan(row)

    def _enter_derived_mode(self, mode: str) -> None:
        target_vf  = self.current.v_front  if self.current is not None else np.nan
        target_pwr = self.current.power_uw if self.current is not None else np.nan
        lh_row = self._find_record_row(target_vf, target_pwr, "LH")
        rh_row = self._find_record_row(target_vf, target_pwr, "RH")
        if lh_row < 0 or rh_row < 0:
            self.statusBar().showMessage("Could not find matching LH and RH scans.")
            return
        self._lh_rec = self.records[lh_row]
        self._rh_rec = self.records[rh_row]
        lh_stk = self._lh_rec.load_stack()
        rh_stk = self._rh_rec.load_stack()
        if lh_stk is None or rh_stk is None:
            self.statusBar().showMessage("Failed to load LH/RH stacks for derived mode.")
            return
        Nt = min(lh_stk.shape[0], rh_stk.shape[0])
        lh = lh_stk[:Nt].astype(np.float32)
        rh = rh_stk[:Nt].astype(np.float32)
        if mode == "LH-RH":
            self._derived_stack = lh - rh
        else:  # norm
            denom = lh + rh
            self._derived_stack = np.clip(
                (lh - rh) / (np.abs(denom) + 1e-9), -1.0, 1.0
            ).astype(np.float32)
        self._hel_mode = mode
        old_t_ps: Optional[float] = None
        if self.current is not None and self.current.n_delays > 0:
            old_t_ps = self.current.delays_ps[int(np.clip(
                self._t_idx(), 0, self.current.n_delays - 1
            ))]
        self.current = self._lh_rec
        if old_t_ps is not None and Nt > 0:
            new_t_idx = int(np.argmin(np.abs(self._lh_rec.delays_ps[:Nt] - old_t_ps)))
        else:
            new_t_idx = 0
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, Nt - 1))
        self.time_slider.setValue(new_t_idx)
        self.time_slider.blockSignals(False)
        self._sync_sliders_to_current()
        self._update_display(new_t_idx)
        self.statusBar().showMessage(
            f"Derived [{mode}]  LH: {self._lh_rec.label}  |  RH: {self._rh_rec.label}"
        )

    def _current_delays(self) -> np.ndarray:
        rec = self.current
        if rec is None or rec.n_delays == 0:
            return np.array([])
        if self._derived_stack is not None:
            return rec.delays_ps[:self._derived_stack.shape[0]]
        return rec.delays_ps

    def _on_hel_changed(self, hel: str) -> None:
        if not hel or not self._unique_hel:
            return
        if hel in ("LH-RH", "norm"):
            self._enter_derived_mode(hel)
            return
        # Switching to a real helicity resets derived mode
        self._hel_mode = "single"
        self._derived_stack = None
        self._lh_rec = None
        self._rh_rec = None
        target_vf  = self.current.v_front  if self.current is not None else np.nan
        target_pwr = self.current.power_uw if self.current is not None else np.nan
        row = self._find_record_row(target_vf, target_pwr, hel)
        if row >= 0:
            self._switch_to_scan(row)

    def _apply_feat_ranges(self) -> None:
        def _parse_range(lo_edit, hi_edit):
            try:
                lo = float(lo_edit.text())
                hi = float(hi_edit.text())
                return (lo, hi)
            except ValueError:
                return None

        self.feat1_range = _parse_range(self.feat1_lo_edit, self.feat1_hi_edit)
        self.feat2_range = _parse_range(self.feat2_lo_edit, self.feat2_hi_edit)

        msgs = []
        unit = self._x_label()
        if self.feat1_range:
            msgs.append(f"Feat1: {self.feat1_range[0]:.1f}–{self.feat1_range[1]:.1f} {unit}")
        if self.feat2_range:
            msgs.append(f"Feat2: {self.feat2_range[0]:.1f}–{self.feat2_range[1]:.1f} {unit}")
        if not msgs:
            self.statusBar().showMessage("Feature ranges cleared.")
        else:
            self.statusBar().showMessage("Feature ranges: " + "   ".join(msgs))

        self._update_display(self._t_idx())

    # ── Gate / power dependence plots ─────────────────────────────────────────

    def _update_dep_plots(self) -> None:
        """
        For each scan record, compute feature 1 and feature 2 intensities at
        the current cursor time, then plot gate and power dependence.
        Falls back to cursor-pixel value if no feature ranges are set.
        """
        if not self.records:
            return
        rec  = self.current
        t_idx = self._t_idx()
        ix    = self.corx
        iy    = self.cory
        n     = len(self.records)

        self.statusBar().showMessage(f"Computing dependence plots ({n} scans)…")
        QtWidgets.QApplication.processEvents()

        ref_puw = rec.power_uw  if rec is not None else np.nan
        ref_ang = rec.angle     if rec is not None else np.nan
        ref_vf  = rec.v_front   if rec is not None else np.nan
        ref_hel = rec.helicity  if rec is not None else ""

        use_features = self.feat1_range is not None or self.feat2_range is not None

        # ── Helper: signal value(s) for one record ────────────────────────────
        def _vals(r: ScanRecord):
            if use_features:
                v1 = v2 = np.nan
                if self.feat1_range is not None:
                    pr = self._feat_px_range(r, *self.feat1_range)
                    if pr:
                        v1 = r.feat_intensity_at(t_idx, *pr)
                if self.feat2_range is not None:
                    pr = self._feat_px_range(r, *self.feat2_range)
                    if pr:
                        v2 = r.feat_intensity_at(t_idx, *pr)
                return v1, v2
            else:
                v = r.value_at(t_idx, iy, ix)
                return v, np.nan

        # ── Gate dependence ───────────────────────────────────────────────────
        gate_vf:    List[float] = []
        gate_v1:    List[float] = []
        gate_v2:    List[float] = []
        gate_is_cur: List[bool] = []

        for r in self.records:
            if not np.isfinite(r.v_front):
                continue
            if ref_hel and r.helicity and r.helicity != ref_hel:
                continue
            same_pwr = (
                np.isfinite(ref_puw) and np.isfinite(r.power_uw)
                and abs(r.power_uw - ref_puw) / max(abs(ref_puw), 1e-9) < 0.06
            )
            same_ang = (
                np.isfinite(ref_ang) and np.isfinite(r.angle)
                and abs(r.angle - ref_ang) < 0.6
            )
            no_pwr_info = (
                not np.isfinite(ref_puw) and not np.isfinite(r.power_uw)
                and not np.isfinite(ref_ang) and not np.isfinite(r.angle)
            )
            if not (same_pwr or same_ang or no_pwr_info):
                continue
            v1, v2 = _vals(r)
            if np.isfinite(v1) or np.isfinite(v2):
                gate_vf.append(r.v_front)
                gate_v1.append(v1)
                gate_v2.append(v2)
                gate_is_cur.append(r is rec)

        if len(gate_vf) < 2:
            gate_vf.clear(); gate_v1.clear(); gate_v2.clear(); gate_is_cur.clear()
            for r in self.records:
                if not np.isfinite(r.v_front):
                    continue
                if ref_hel and r.helicity and r.helicity != ref_hel:
                    continue
                v1, v2 = _vals(r)
                if np.isfinite(v1) or np.isfinite(v2):
                    gate_vf.append(r.v_front)
                    gate_v1.append(v1)
                    gate_v2.append(v2)
                    gate_is_cur.append(r is rec)

        self.gate_pw.clear()
        if gate_vf:
            order = np.argsort(gate_vf)
            vf_s  = np.array(gate_vf)[order]
            v1_s  = np.array(gate_v1)[order]
            v2_s  = np.array(gate_v2)[order]
            cur_s = [gate_is_cur[i] for i in order]

            def _scatter(pw, xs, ys, base_col, symbol):
                valid = np.isfinite(ys)
                if not valid.any():
                    return
                pw.plot(xs[valid], ys[valid], pen=pg.mkPen(base_col, width=1))
                brushes = [pg.mkBrush(*CUR_COL) if c else pg.mkBrush(*base_col)
                           for c, ok in zip(cur_s, valid) if ok]
                pw.addItem(pg.ScatterPlotItem(
                    x=xs[valid], y=ys[valid], size=11,
                    symbol=symbol,
                    brush=brushes,
                    pen=pg.mkPen("k", width=1),
                ))

            _scatter(self.gate_pw, vf_s, v1_s, FEAT1_COL, "o")
            _scatter(self.gate_pw, vf_s, v2_s, FEAT2_COL, "s")

        # ── Power dependence ──────────────────────────────────────────────────
        pwr_p:     List[float] = []
        pwr_v1:    List[float] = []
        pwr_v2:    List[float] = []
        pwr_is_cur: List[bool] = []

        for r in self.records:
            if not np.isfinite(r.power_uw):
                continue
            if ref_hel and r.helicity and r.helicity != ref_hel:
                continue
            same_vf = (
                np.isfinite(ref_vf) and np.isfinite(r.v_front)
                and abs(r.v_front - ref_vf) < 0.006
            )
            no_gate = not np.isfinite(ref_vf) and not np.isfinite(r.v_front)
            if not (same_vf or no_gate):
                continue
            v1, v2 = _vals(r)
            if np.isfinite(v1) or np.isfinite(v2):
                pwr_p.append(r.power_uw)
                pwr_v1.append(v1)
                pwr_v2.append(v2)
                pwr_is_cur.append(r is rec)

        if len(pwr_p) < 2:
            pwr_p.clear(); pwr_v1.clear(); pwr_v2.clear(); pwr_is_cur.clear()
            for r in self.records:
                if not np.isfinite(r.power_uw):
                    continue
                if ref_hel and r.helicity and r.helicity != ref_hel:
                    continue
                v1, v2 = _vals(r)
                if np.isfinite(v1) or np.isfinite(v2):
                    pwr_p.append(r.power_uw)
                    pwr_v1.append(v1)
                    pwr_v2.append(v2)
                    pwr_is_cur.append(r is rec)

        self.pwr_pw.clear()
        if pwr_p:
            order = np.argsort(pwr_p)
            pw_s  = np.array(pwr_p)[order]
            v1_s  = np.array(pwr_v1)[order]
            v2_s  = np.array(pwr_v2)[order]
            cur_s = [pwr_is_cur[i] for i in order]

            _scatter(self.pwr_pw, pw_s, v1_s, FEAT1_COL, "o")
            _scatter(self.pwr_pw, pw_s, v2_s, FEAT2_COL, "s")

        # ── Titles ────────────────────────────────────────────────────────────
        if rec is not None:
            xax  = self._x_axis(rec)
            yax  = self._y_axis(rec)
            t_ps = rec.delays_ps[t_idx] if rec.n_delays else 0.0
            xlbl = (f"λ={xax[ix]:.1f} nm"
                    if (self.wl_m != 1.0 or self.wl_b != 0.0) else f"px {ix}")
            coord = f"y={yax[iy]:.1f} µm, t={t_ps:.1f} ps"
            hel_sfx = f"  [{ref_hel}]" if ref_hel else ""
            if use_features:
                feat_lbl = f"feature ranges{hel_sfx}"
            else:
                feat_lbl = f"{xlbl}, {coord}{hel_sfx}"
            self.gate_pw.setTitle(
                f"Gate dep  ● Feat1  ■ Feat2  |  {feat_lbl}  |  red=current"
            )
            self.pwr_pw.setTitle(
                f"Power dep  ● Feat1  ■ Feat2  |  {feat_lbl}  |  red=current"
            )

        self.statusBar().showMessage(
            f"Dep plots updated — {len(gate_vf)} gate pts, {len(pwr_p)} power pts  "
            f"({'feature ranges' if use_features else 'cursor pixel'}, red = current scan)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PolaronTAViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

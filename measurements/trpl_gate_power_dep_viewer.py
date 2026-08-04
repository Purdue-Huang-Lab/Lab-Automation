"""
measurements/trpl_gate_power_dep_viewer.py

Viewer for TRPL gate- and power-dependent data produced by
measurements/trpl_gate_power_dep GUI.

Tabs:
  1. Data Browser   — overlay histogram traces, normalize / log / x-y range
  2. Power × Time   — 2-D colorplot: x = wheel angle, y = time (ps); select Va slice
  3. Gate × Time    — 2-D colorplot: x = Va (front gate), y = time (ps); Vb as top axis; select angle slice
"""
from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")

# ── repo path ──────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ph300.gui_v2.plots import HistogramPlotWidget

try:
    from measurements.config import DATA_DIR
except Exception:
    DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop")


# ══════════════════════════════════════════════════════════════════════════════
#  Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MeasPoint:
    idx:     int
    angle:   float
    va:      float
    vb:      float
    time_ps: np.ndarray
    counts:  np.ndarray


# filename: pt0000_ang0.00_Va-2.0000_Vb-3.6000.csv
_FNAME_RE = re.compile(r"^pt(\d+)_ang([^_]+)_Va([^_]+)_Vb(.+)\.csv$")


def _load_folder(folder: str) -> List[MeasPoint]:
    points = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        try:
            idx   = int(m.group(1))
            angle = float(m.group(2))
            va    = float(m.group(3))
            vb    = float(m.group(4))
        except ValueError:
            continue
        time_ps, counts = [], []
        try:
            with open(os.path.join(folder, fname), "r", newline="") as f:
                rd = csv.reader(f)
                next(rd)  # skip header row
                for row in rd:
                    if len(row) >= 2:
                        try:
                            time_ps.append(float(row[0]))
                            counts.append(int(float(row[1])))
                        except ValueError:
                            pass
        except Exception:
            continue
        points.append(MeasPoint(
            idx=idx, angle=angle, va=va, vb=vb,
            time_ps=np.array(time_ps, dtype=np.float64),
            counts=np.array(counts, dtype=np.uint32),
        ))
    return sorted(points, key=lambda p: p.idx)


def _unique_sorted(vals: List[float]) -> List[float]:
    seen, out = set(), []
    for v in sorted(vals):
        key = round(v, 9)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _near(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))


def _get_lut(name: str) -> np.ndarray:
    return pg.colormap.get(name).getLookupTable(nPts=256)


_CMAPS = ["viridis", "plasma", "inferno", "magma", "hot", "CET-L9"]
_BTN_W = 95


# ══════════════════════════════════════════════════════════════════════════════
#  Colorplot widget
# ══════════════════════════════════════════════════════════════════════════════

class ColorPlotWidget(QtWidgets.QWidget):
    """
    2-D colorplot: x = one sweep parameter, y = time_ps, color = TRPL intensity.
    The selector combo holds the orthogonal parameter fixed.
    Optional second_x_label triggers a top axis showing Vb = ratio × Va.
    """

    def __init__(
        self,
        x_label:         str,
        selector_label:  str,
        second_x_label:  str = "",
        parent=None,
    ):
        super().__init__(parent)

        # ── pyqtgraph ──────────────────────────────────────────────────────────
        self._glw  = pg.GraphicsLayoutWidget()
        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setLabel("left",   "Time (ps)")
        self._plot.setLabel("bottom", x_label)

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)
        self._img.setLookupTable(_get_lut("viridis"))

        if second_x_label:
            self._plot.showAxis("top")
            self._plot.getAxis("top").setLabel(second_x_label)
            self._plot.sigXRangeChanged.connect(self._refresh_top_axis)

        # ── controls ───────────────────────────────────────────────────────────
        ctrl = QtWidgets.QGroupBox("Controls")
        cg   = QtWidgets.QGridLayout(ctrl)

        self._combo_sel = QtWidgets.QComboBox()
        self._combo_sel.setMinimumWidth(110)
        self._combo_sel.currentIndexChanged.connect(self._on_sel_changed)

        self._chk_norm = QtWidgets.QCheckBox("Normalize")
        self._chk_log  = QtWidgets.QCheckBox("Log color")

        def _dspin(lo, hi, dec, val):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val)
            return s

        self._spin_tmin = _dspin(0, 1e12, 1,  60000.0)
        self._spin_tmax = _dspin(0, 1e12, 1, 100000.0)
        self._spin_cmin = _dspin(-1e9, 1e9, 4, 0.0)
        self._spin_cmax = _dspin(-1e9, 1e9, 4, 1.0)

        self._combo_cmap = QtWidgets.QComboBox()
        self._combo_cmap.addItems(_CMAPS)

        btn_auto  = QtWidgets.QPushButton("Auto levels")
        btn_apply = QtWidgets.QPushButton("Apply")
        for b in (btn_auto, btn_apply):
            b.setFixedWidth(_BTN_W)
        btn_auto.clicked.connect(self._auto_levels)
        btn_apply.clicked.connect(self.replot)

        r = 0
        cg.addWidget(QtWidgets.QLabel(selector_label), r, 0, 1, 2); r += 1
        cg.addWidget(self._combo_sel,  r, 0, 1, 2); r += 1
        cg.addWidget(self._chk_norm,   r, 0)
        cg.addWidget(self._chk_log,    r, 1); r += 1
        cg.addWidget(QtWidgets.QLabel("T min (ps):"), r, 0)
        cg.addWidget(self._spin_tmin,  r, 1); r += 1
        cg.addWidget(QtWidgets.QLabel("T max (ps):"), r, 0)
        cg.addWidget(self._spin_tmax,  r, 1); r += 1
        cg.addWidget(QtWidgets.QLabel("C min:"),  r, 0)
        cg.addWidget(self._spin_cmin,  r, 1); r += 1
        cg.addWidget(QtWidgets.QLabel("C max:"),  r, 0)
        cg.addWidget(self._spin_cmax,  r, 1); r += 1
        cg.addWidget(QtWidgets.QLabel("Colormap:"), r, 0)
        cg.addWidget(self._combo_cmap, r, 1); r += 1
        cg.addWidget(btn_auto,  r, 0)
        cg.addWidget(btn_apply, r, 1); r += 1
        cg.setRowStretch(r, 1)

        rw = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rw)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(ctrl)
        rv.addStretch(1)

        hl = QtWidgets.QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self._glw, 1)
        hl.addWidget(rw)

        # ── state ──────────────────────────────────────────────────────────────
        self._points:        List[MeasPoint]      = []
        self._selector_vals: List[float]          = []
        self._sel_getter:    Callable             = lambda pt: 0.0
        self._x_getter:      Callable             = lambda pt: 0.0
        self._vb_ratio:      Optional[float]      = None
        self._last_x_vals:   List[float]          = []
        self._last_data:     Optional[np.ndarray] = None
        self._last_time_ps:  Optional[np.ndarray] = None

    # ── public ─────────────────────────────────────────────────────────────────

    def configure(
        self,
        points:        List[MeasPoint],
        selector_vals: List[float],
        sel_getter:    Callable,
        x_getter:      Callable,
        selector_fmt:  str            = ".3f",
        vb_ratio:      Optional[float] = None,
    ):
        self._points        = points
        self._selector_vals = selector_vals
        self._sel_getter    = sel_getter
        self._x_getter      = x_getter
        self._vb_ratio      = vb_ratio

        self._combo_sel.blockSignals(True)
        self._combo_sel.clear()
        for v in selector_vals:
            self._combo_sel.addItem(format(v, selector_fmt))
        self._combo_sel.setCurrentIndex(0)
        self._combo_sel.blockSignals(False)
        self.replot()

    def replot(self):
        if not self._points or not self._selector_vals:
            return
        i = self._combo_sel.currentIndex()
        if i < 0 or i >= len(self._selector_vals):
            return
        sel_val = self._selector_vals[i]

        subset = sorted(
            [pt for pt in self._points if _near(self._sel_getter(pt), sel_val)],
            key=self._x_getter,
        )
        if not subset:
            return

        x_vals  = [self._x_getter(pt) for pt in subset]
        time_ps = subset[0].time_ps
        # shape (nt, nx) — rows = time, cols = x (row-major)
        data_2d = np.stack([pt.counts.astype(float) for pt in subset], axis=1)

        self._last_x_vals  = x_vals
        self._last_data    = data_2d
        self._last_time_ps = time_ps

        self._draw(x_vals, time_ps, data_2d)

    # ── private ────────────────────────────────────────────────────────────────

    def _draw(self, x_vals: List[float], time_ps: np.ndarray, data_2d: np.ndarray):
        tmin = float(self._spin_tmin.value())
        tmax = float(self._spin_tmax.value())
        norm = self._chk_norm.isChecked()
        logc = self._chk_log.isChecked()
        cmin = float(self._spin_cmin.value())
        cmax = float(self._spin_cmax.value())

        # Time crop — data_2d shape (nt, nx)
        t_mask = (time_ps >= tmin) & (time_ps <= tmax)
        t_crop = time_ps[t_mask]
        d      = data_2d[t_mask, :].astype(float)   # (nt_crop, nx)

        if d.size == 0 or len(t_crop) < 2 or len(x_vals) == 0:
            return

        # Normalize per trace (each column = one x value)
        if norm:
            mx = d.max(axis=0, keepdims=True)
            mx[mx == 0] = 1.0
            d = d / mx

        # Log color
        if logc:
            d = np.log10(np.maximum(d, 1.0))

        nt_c, nx = d.shape
        dt = float(t_crop[1] - t_crop[0])

        # Update colormap LUT
        try:
            lut = _get_lut(self._combo_cmap.currentText())
        except Exception:
            lut = _get_lut("viridis")
        self._img.setLookupTable(lut)
        self._img.setImage(d, autoLevels=False)
        self._img.setLevels([cmin, cmax])

        # Map image to world coords: x = index [0..nx-1], y = time_ps
        self._img.setRect(QtCore.QRectF(
            -0.5,
            float(t_crop[0]) - dt / 2,
            float(nx),
            float(t_crop[-1] - t_crop[0]) + dt,
        ))

        # Custom x-axis ticks: index → actual value
        ticks = [[(float(i), f"{v:.3g}") for i, v in enumerate(x_vals)]]
        self._plot.getAxis("bottom").setTicks(ticks)
        self._plot.setXRange(-0.5, float(nx) - 0.5, padding=0)
        self._plot.setYRange(float(t_crop[0]) - dt / 2,
                             float(t_crop[-1]) + dt / 2, padding=0)

        self._refresh_top_axis()

    def _auto_levels(self):
        if self._last_data is None or self._last_time_ps is None:
            return
        tmin = float(self._spin_tmin.value())
        tmax = float(self._spin_tmax.value())
        norm = self._chk_norm.isChecked()
        logc = self._chk_log.isChecked()
        t_mask = (self._last_time_ps >= tmin) & (self._last_time_ps <= tmax)
        d = self._last_data[t_mask, :].astype(float)
        if norm:
            mx = d.max(axis=0, keepdims=True); mx[mx == 0] = 1.0; d = d / mx
        if logc:
            d = np.log10(np.maximum(d, 1.0))
        if d.size:
            self._spin_cmin.setValue(float(np.nanmin(d)))
            self._spin_cmax.setValue(float(np.nanmax(d)))
        self.replot()

    def _on_sel_changed(self):
        self.replot()

    def _refresh_top_axis(self):
        if self._vb_ratio is None or not self._last_x_vals:
            return
        ticks = [
            (float(i), f"{self._vb_ratio * v:.3g}")
            for i, v in enumerate(self._last_x_vals)
        ]
        self._plot.getAxis("top").setTicks([ticks])


# ══════════════════════════════════════════════════════════════════════════════
#  Main viewer widget
# ══════════════════════════════════════════════════════════════════════════════

class TRPLViewer(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[MeasPoint] = []
        self._folder: str = ""

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 4)
        outer.setSpacing(4)

        # toolbar
        tb = QtWidgets.QHBoxLayout()
        btn_open   = QtWidgets.QPushButton("Open Folder…")
        btn_reload = QtWidgets.QPushButton("Reload")
        btn_open.setFixedWidth(120); btn_reload.setFixedWidth(70)
        self._lbl_folder = QtWidgets.QLabel("No folder loaded.")
        self._lbl_folder.setStyleSheet("color: #555;")
        btn_open.clicked.connect(self._on_open)
        btn_reload.clicked.connect(self._on_reload)
        tb.addWidget(btn_open); tb.addWidget(btn_reload)
        tb.addWidget(self._lbl_folder, 1)
        outer.addLayout(tb)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_browser_tab(),    "1 · Data Browser")
        self._tabs.addTab(self._build_power_time_tab(), "2 · Power × Time")
        self._tabs.addTab(self._build_gate_time_tab(),  "3 · Gate × Time")
        outer.addWidget(self._tabs, 1)

        self._status = QtWidgets.QStatusBar()
        outer.addWidget(self._status)
        self._status.showMessage("Open a data folder to begin.")

    # ── browser tab ────────────────────────────────────────────────────────────

    def _build_browser_tab(self) -> QtWidgets.QWidget:
        w  = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(w)

        left = QtWidgets.QVBoxLayout()
        self._br_plot = HistogramPlotWidget()
        left.addWidget(self._br_plot, 1)

        ctrl = QtWidgets.QGroupBox("Display Controls")
        cg   = QtWidgets.QGridLayout(ctrl)

        self._chk_norm_br = QtWidgets.QCheckBox("Normalize")
        self._chk_logy_br = QtWidgets.QCheckBox("Log Y")
        self._chk_norm_br.toggled.connect(self._refresh_browser)
        self._chk_logy_br.toggled.connect(self._refresh_browser)

        def _dspin(lo, hi, dec, val):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val)
            return s

        self._br_xmin = _dspin(0, 1e12, 1,  60000.0)
        self._br_xmax = _dspin(0, 1e12, 1, 100000.0)
        self._br_ymin = _dspin(0, 1e12, 0, 0.0)
        self._br_ymax = _dspin(0, 1e12, 0, 1e6)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setFixedWidth(_BTN_W)
        btn_apply.clicked.connect(self._apply_br_range)

        cg.addWidget(self._chk_norm_br,              0, 0)
        cg.addWidget(self._chk_logy_br,              0, 1)
        cg.addWidget(QtWidgets.QLabel("X min (ps):"), 1, 0)
        cg.addWidget(self._br_xmin,                  1, 1)
        cg.addWidget(QtWidgets.QLabel("X max (ps):"), 1, 2)
        cg.addWidget(self._br_xmax,                  1, 3)
        cg.addWidget(QtWidgets.QLabel("Y min:"),      2, 0)
        cg.addWidget(self._br_ymin,                  2, 1)
        cg.addWidget(QtWidgets.QLabel("Y max:"),      2, 2)
        cg.addWidget(self._br_ymax,                  2, 3)
        cg.addWidget(btn_apply,                      2, 4)
        left.addWidget(ctrl)
        hl.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Loaded Data</b>"))

        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["#", "Angle (°)", "Va (V)", "Vb (V)", "Show"])
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.itemChanged.connect(self._on_table_changed)

        btn_row = QtWidgets.QHBoxLayout()
        btn_save  = QtWidgets.QPushButton("Save Selected CSV")
        btn_clear = QtWidgets.QPushButton("Clear Selection")
        btn_save.clicked.connect(self._on_save_csv)
        btn_clear.clicked.connect(self._on_clear_sel)
        btn_row.addWidget(btn_save); btn_row.addWidget(btn_clear)

        right.addWidget(self._table, 1)
        right.addLayout(btn_row)
        hl.addLayout(right, 2)
        return w

    # ── colorplot tabs ─────────────────────────────────────────────────────────

    def _build_power_time_tab(self) -> QtWidgets.QWidget:
        self._cp_power = ColorPlotWidget(
            x_label        = "Wheel Angle (°)",
            selector_label = "Gate Va (V):",
        )
        return self._cp_power

    def _build_gate_time_tab(self) -> QtWidgets.QWidget:
        self._cp_gate = ColorPlotWidget(
            x_label        = "Front Gate Va (V)",
            selector_label = "Wheel Angle (°):",
            second_x_label = "Back Gate Vb (V)",
        )
        return self._cp_gate

    # ── load / rebuild ─────────────────────────────────────────────────────────

    def _on_open(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open TRPL data folder", str(DATA_DIR))
        if folder:
            self._folder = folder
            self._load()

    def _on_reload(self):
        if self._folder:
            self._load()

    def _load(self):
        try:
            pts = _load_folder(self._folder)
        except Exception as e:
            self._status.showMessage(f"Load error: {e}"); return
        self._points = pts
        self._lbl_folder.setText(self._folder)
        self._status.showMessage(
            f"Loaded {len(pts)} points from {os.path.basename(self._folder)}")
        self._rebuild()

    def _rebuild(self):
        pts = self._points
        if not pts:
            return

        # browser table
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for pt in pts:
            self._add_table_row(pt)
        self._table.blockSignals(False)
        self._br_plot.clear_all()

        # colorplot 1: power × time  (x = angle, selector = Va)
        va_vals = _unique_sorted([pt.va for pt in pts])
        self._cp_power.configure(
            points        = pts,
            selector_vals = va_vals,
            sel_getter    = lambda pt: pt.va,
            x_getter      = lambda pt: pt.angle,
            selector_fmt  = ".4f",
        )

        # colorplot 2: gate × time  (x = Va, selector = angle)
        angle_vals = _unique_sorted([pt.angle for pt in pts])
        vb_ratio: Optional[float] = None
        for pt in pts:
            if abs(pt.va) > 1e-6:
                vb_ratio = pt.vb / pt.va
                break
        self._cp_gate.configure(
            points        = pts,
            selector_vals = angle_vals,
            sel_getter    = lambda pt: pt.angle,
            x_getter      = lambda pt: pt.va,
            selector_fmt  = ".2f",
            vb_ratio      = vb_ratio,
        )

    # ── browser helpers ────────────────────────────────────────────────────────

    def _add_table_row(self, pt: MeasPoint):
        row = self._table.rowCount()
        self._table.insertRow(row)

        def _item(txt):
            it = QtWidgets.QTableWidgetItem(txt)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            return it

        self._table.setItem(row, 0, _item(str(pt.idx)))
        self._table.setItem(row, 1, _item(f"{pt.angle:.3f}"))
        self._table.setItem(row, 2, _item(f"{pt.va:.4f}"))
        self._table.setItem(row, 3, _item(f"{pt.vb:.4f}"))
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk.setCheckState(QtCore.Qt.Unchecked)
        self._table.setItem(row, 4, chk)

    def _pt_label(self, pt: MeasPoint) -> str:
        return f"θ={pt.angle:.1f}° Va={pt.va:.2f}V Vb={pt.vb:.2f}V"

    def _maybe_norm(self, counts: np.ndarray) -> np.ndarray:
        c = counts.astype(float)
        if self._chk_norm_br.isChecked():
            mx = c.max()
            return c / mx if mx > 0 else c
        return c

    def _on_table_changed(self, item: QtWidgets.QTableWidgetItem):
        if item.column() != 4:
            return
        row  = item.row()
        show = item.checkState() == QtCore.Qt.Checked
        if row >= len(self._points):
            return
        pt    = self._points[row]
        label = self._pt_label(pt)
        if show:
            self._br_plot.add_trace(label, pt.time_ps, self._maybe_norm(pt.counts))
        else:
            self._br_plot.remove_trace(label)

    def _refresh_browser(self):
        self._br_plot.clear_all()
        for row in range(self._table.rowCount()):
            chk = self._table.item(row, 4)
            if chk and chk.checkState() == QtCore.Qt.Checked and row < len(self._points):
                pt = self._points[row]
                self._br_plot.add_trace(
                    self._pt_label(pt), pt.time_ps, self._maybe_norm(pt.counts))
        self._apply_br_range()

    def _apply_br_range(self):
        self._br_plot.apply_range(
            xmin=float(self._br_xmin.value()),
            xmax=float(self._br_xmax.value()),
            ymin=float(self._br_ymin.value()),
            ymax=float(self._br_ymax.value()),
            logy=self._chk_logy_br.isChecked(),
        )

    def _on_save_csv(self):
        rows = [r for r in range(self._table.rowCount())
                if self._table.item(r, 4) and
                   self._table.item(r, 4).checkState() == QtCore.Qt.Checked]
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Save CSV", "Check rows in the table to select traces.")
            return
        pts = [self._points[r] for r in rows if r < len(self._points)]
        if not pts:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "trpl_selection.csv", "CSV Files (*.csv)")
        if not path:
            return
        n = len(pts[0].time_ps)
        for pt in pts[1:]:
            if len(pt.time_ps) != n:
                QtWidgets.QMessageBox.warning(
                    self, "Save CSV", "Selected traces have different lengths."); return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ps"] + [self._pt_label(pt) for pt in pts])
            for i in range(n):
                w.writerow([float(pts[0].time_ps[i])] + [int(pt.counts[i]) for pt in pts])
        self._status.showMessage(f"Saved: {path}")

    def _on_clear_sel(self):
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            chk = self._table.item(row, 4)
            if chk:
                chk.setCheckState(QtCore.Qt.Unchecked)
        self._table.blockSignals(False)
        self._br_plot.clear_all()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("TRPL Gate & Power Dep Viewer")
    win.resize(1300, 820)
    w = TRPLViewer()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        w._folder = sys.argv[1]
        w._load()
    win.setCentralWidget(w)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

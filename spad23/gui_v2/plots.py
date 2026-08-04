from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

from .config import NUM_PIXELS, PITCH_X_UM, PITCH_Y_UM, PIXEL_COORDS

_MAP_COORDS = np.array([PIXEL_COORDS[i] for i in range(NUM_PIXELS)], dtype=np.float64)

_TRACE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b6ecf',
]

_EXTRA_COLORS = ['#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']


def _fix_axes(pi) -> None:
    for ax in ("bottom", "left"):
        pi.getAxis(ax).setPen(pg.mkPen("k"))
        pi.getAxis(ax).setTextPen(pg.mkPen("k"))


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """Edge-padded rolling median — avoids the boundary artifact of np.convolve mode='same'."""
    if w <= 1 or len(x) < w:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, w - 1 - pad), mode='edge')
    shape = (len(x), w)
    strides = (xp.strides[0], xp.strides[0])
    windows = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    return np.median(windows, axis=1)


class SpadMapWidget(QtWidgets.QWidget):
    pixel_left_clicked = pyqtSignal(int)
    pixel_right_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._pw = pg.PlotWidget()
        pi = self._pw.plotItem
        _fix_axes(pi)
        pi.hideAxis('bottom')
        pi.hideAxis('left')
        pi.setTitle("Count Map @ Selected Time Bin")
        pi.setAspectLocked(True)
        pi.vb.setRange(xRange=(-0.7, 4.7), yRange=(-0.7, 4.7), padding=0)
        layout.addWidget(self._pw, 1)

        self._cmap = pg.colormap.get('inferno')

        self._scatter = pg.ScatterPlotItem(
            x=_MAP_COORDS[:, 0], y=_MAP_COORDS[:, 1],
            size=30, pxMode=True,
            brush=[pg.mkBrush(self._cmap.map(0.0, mode='qcolor'))] * NUM_PIXELS,
            pen=[pg.mkPen('#5f6368', width=2)] * NUM_PIXELS,
            data=list(range(NUM_PIXELS)),
        )
        self._pw.addItem(self._scatter)

        self._sel = pg.ScatterPlotItem(
            symbol='o', size=35, pxMode=True,
            pen=pg.mkPen('#22d3ee', width=2.2),
            brush=pg.mkBrush(None))
        self._pw.addItem(self._sel)

        self._excl = pg.ScatterPlotItem(
            symbol='x', size=15, pxMode=True,
            pen=pg.mkPen('#ef4444', width=2.2),
            brush=pg.mkBrush(None))
        self._pw.addItem(self._excl)

        for i, (x, y) in PIXEL_COORDS.items():
            t = pg.TextItem(str(i), anchor=(0.5, 0.5), color='#f7f7f7')
            f = t.textItem.font()
            f.setBold(True)
            f.setPointSize(8)
            t.textItem.setFont(f)
            t.setPos(x, y)
            self._pw.addItem(t)

        self._lbl = QtWidgets.QLabel("Time: -- | Selected: -- | Excluded: --")
        layout.addWidget(self._lbl)

        self._pw.scene().sigMouseClicked.connect(self._on_click)

    def update(self, frame: np.ndarray, aligned_t_ns: float,
               selected: Set[int], excluded: Set[int]) -> None:
        vmin, vmax = float(np.min(frame)), float(np.max(frame))
        norm = ((frame - vmin) / (vmax - vmin)
                if vmax > vmin else np.zeros_like(frame))
        colors = self._cmap.map(norm.astype(np.float32), mode='byte')
        brushes = [pg.mkBrush(*colors[i]) for i in range(NUM_PIXELS)]
        self._scatter.setData(
            x=_MAP_COORDS[:, 0], y=_MAP_COORDS[:, 1],
            brush=brushes,
            pen=[pg.mkPen('#5f6368', width=2)] * NUM_PIXELS,
            size=30, pxMode=True,
            data=list(range(NUM_PIXELS)),
        )
        if selected:
            pts = np.array([PIXEL_COORDS[p] for p in sorted(selected)], dtype=float)
            self._sel.setData(x=pts[:, 0], y=pts[:, 1])
        else:
            self._sel.setData(x=[], y=[])
        if excluded:
            pts = np.array([PIXEL_COORDS[p] for p in sorted(excluded)], dtype=float)
            self._excl.setData(x=pts[:, 0], y=pts[:, 1])
        else:
            self._excl.setData(x=[], y=[])

        sel_s = ", ".join(str(p) for p in sorted(selected)) if selected else "none"
        excl_s = ", ".join(str(p) for p in sorted(excluded)) if excluded else "none"
        self._lbl.setText(
            f"t = {aligned_t_ns:.4f} ns (after t₀) | "
            f"Selected: {sel_s} | Excluded: {excl_s}")

    def _on_click(self, event) -> None:
        vb = self._pw.plotItem.vb
        pos = vb.mapSceneToView(event.scenePos())
        if not vb.viewRect().contains(pos):
            return
        x, y = pos.x(), pos.y()
        d2 = (_MAP_COORDS[:, 0] - x) ** 2 + (_MAP_COORDS[:, 1] - y) ** 2
        nearest = int(np.argmin(d2))
        if float(np.sqrt(d2[nearest])) > 0.42:
            return
        if event.button() == QtCore.Qt.RightButton:
            self.pixel_right_clicked.emit(nearest)
        elif event.button() == QtCore.Qt.LeftButton:
            self.pixel_left_clicked.emit(nearest)


class HistogramWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._pw = pg.PlotWidget()
        self._pi = self._pw.plotItem
        _fix_axes(self._pi)
        self._pi.setTitle("Pixel Histograms — aligned to t=0 per pixel  (left-click map to toggle)")
        self._pi.setLabel('bottom', 'Time relative to t₀ (ns)')
        self._pi.setLabel('left', 'Counts per bin')
        self._pi.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._pi.addLegend()
        layout.addWidget(self._pw, 1)

        self._time_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#111827', style=QtCore.Qt.DashLine, width=1))
        self._zero_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#64748b', style=QtCore.Qt.DotLine, width=1))
        self._zero_line.setValue(0.0)
        self._pw.addItem(self._zero_line)
        self._pw.addItem(self._time_line)

        self._normalize = False
        self._subtract_baseline = False
        self._show_baseline = False

        self._raw: Dict[int, pg.PlotDataItem] = {}
        self._smooth: Dict[int, pg.PlotDataItem] = {}
        self._fit: Dict[int, pg.PlotDataItem] = {}
        self._baseline_lines: Dict[int, pg.InfiniteLine] = {}

    def _clear_traces(self) -> None:
        for pix in list(self._raw):
            self._pw.removeItem(self._raw.pop(pix))
        for pix in list(self._smooth):
            self._legend.removeItem(self._smooth[pix])
            self._pw.removeItem(self._smooth.pop(pix))
        for pix in list(self._baseline_lines):
            self._pw.removeItem(self._baseline_lines.pop(pix))

    def update(self, time_ns: np.ndarray, counts_matrix: np.ndarray,
               smoothed: np.ndarray, t0_ns: np.ndarray,
               current_aligned_ns: float,
               selected: Set[int], log_y: bool, normalize: bool,
               baseline_counts: Optional[np.ndarray] = None,
               subtract_baseline: bool = False,
               show_baseline: bool = False) -> None:
        self._time_line.setValue(current_aligned_ns)

        # Full redraw when any display-mode flag changes
        if (normalize != self._normalize
                or subtract_baseline != self._subtract_baseline
                or show_baseline != self._show_baseline):
            self._normalize = normalize
            self._subtract_baseline = subtract_baseline
            self._show_baseline = show_baseline
            self._clear_traces()

        # Remove deselected pixels
        for pix in list(self._raw):
            if pix not in selected:
                self._pw.removeItem(self._raw.pop(pix))
                if pix in self._smooth:
                    self._legend.removeItem(self._smooth[pix])
                    self._pw.removeItem(self._smooth.pop(pix))
                if pix in self._fit:
                    self._pw.removeItem(self._fit.pop(pix))
                if pix in self._baseline_lines:
                    self._pw.removeItem(self._baseline_lines.pop(pix))

        for pix in sorted(selected):
            aligned_t = time_ns - float(t0_ns[pix])
            raw_y = counts_matrix[:, pix].astype(float)
            smooth_y = smoothed[:, pix].copy()
            bl = float(baseline_counts[pix]) if baseline_counts is not None else 0.0

            # Record peak before any subtraction (needed for normalized baseline display)
            peak_raw = float(np.max(smooth_y))

            if subtract_baseline:
                raw_y = np.maximum(raw_y - bl, 0.0)
                smooth_y = np.maximum(smooth_y - bl, 0.0)

            if normalize:
                peak = float(np.max(smooth_y))
                if peak > 0:
                    raw_y = raw_y / peak
                    smooth_y = smooth_y / peak

            color = _TRACE_COLORS[pix % len(_TRACE_COLORS)]

            if pix in self._raw:
                self._raw[pix].setData(aligned_t, raw_y)
                self._smooth[pix].setData(aligned_t, smooth_y)
            else:
                raw_item = pg.PlotDataItem(
                    aligned_t, raw_y,
                    pen=None,
                    symbol='o', symbolSize=3,
                    symbolBrush=pg.mkBrush(color + '88'),
                    symbolPen=None)
                self._pw.addItem(raw_item)
                self._raw[pix] = raw_item

                curve = self._pw.plot(
                    aligned_t, smooth_y,
                    pen=pg.mkPen(color, width=1.6),
                    name=f"Px {pix}")
                self._smooth[pix] = curve

            # Baseline indicator — only when not already subtracted
            if show_baseline and not subtract_baseline and baseline_counts is not None:
                bl_display = (bl / peak_raw if (normalize and peak_raw > 0) else bl)
                if pix in self._baseline_lines:
                    self._baseline_lines[pix].setValue(bl_display)
                else:
                    line = pg.InfiniteLine(
                        angle=0, pos=bl_display, movable=False,
                        pen=pg.mkPen(color, style=QtCore.Qt.DashLine, width=1.2))
                    self._pw.addItem(line)
                    self._baseline_lines[pix] = line
            else:
                if pix in self._baseline_lines:
                    self._pw.removeItem(self._baseline_lines.pop(pix))

        self._pi.setLabel('left',
                          'Normalized counts' if normalize else 'Counts per bin')
        self._pw.setLogMode(y=log_y)

    def update_fits(self,
                    fit_curves: Dict[int, Optional[tuple]],
                    normalize: bool,
                    peak_vals: Dict[int, float]) -> None:
        for pix in list(self._fit):
            if pix not in fit_curves:
                self._pw.removeItem(self._fit.pop(pix))

        for pix, result in fit_curves.items():
            if result is None:
                if pix in self._fit:
                    self._pw.removeItem(self._fit.pop(pix))
                continue
            t_fit, y_fit = result
            if normalize:
                peak = peak_vals.get(pix, 1.0)
                if peak > 0:
                    y_fit = y_fit / peak
            color = _TRACE_COLORS[pix % len(_TRACE_COLORS)]
            if pix in self._fit:
                self._fit[pix].setData(t_fit, y_fit)
            else:
                item = self._pw.plot(
                    t_fit, y_fit,
                    pen=pg.mkPen(color, width=2.0, style=QtCore.Qt.DashLine))
                self._fit[pix] = item

    def clear_fits(self) -> None:
        for pix in list(self._fit):
            self._pw.removeItem(self._fit.pop(pix))


class FittingWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw, 1)

        # 2×2 grid:
        #  (0,0) p_map    — 2D Gaussian fit map
        #  (0,1) p_xt     — space–time colormap  [new]
        #  (1,0) p_radial — radial fit (current bold + extra traces)
        #  (1,1) p_sigma  — σ_eq² vs time

        self.p_map = glw.addPlot(row=0, col=0)
        _fix_axes(self.p_map)
        self.p_map.setLabel('bottom', 'x (µm)')
        self.p_map.setLabel('left', 'y (µm)')
        self.p_map.setAspectLocked(True)
        self.p_map.invertY(True)

        self.p_xt = glw.addPlot(row=0, col=1)
        _fix_axes(self.p_xt)
        self.p_xt.setTitle("Space–time intensity map")
        self.p_xt.setLabel('bottom', 'r (sample µm)')
        self.p_xt.setLabel('left', 'Time after t₀ (ns)')
        self.p_xt.showGrid(x=True, y=True, alpha=0.18)

        self.p_radial = glw.addPlot(row=1, col=0)
        _fix_axes(self.p_radial)
        self.p_radial.setLabel('bottom', 'Distance (µm)')
        self.p_radial.setLabel('left', 'Counts')
        self.p_radial.showGrid(x=True, y=True, alpha=0.25)

        self.p_sigma = glw.addPlot(row=1, col=1)
        _fix_axes(self.p_sigma)
        self.p_sigma.setTitle("σ_eq² vs Time")
        self.p_sigma.setLabel('bottom', 'Time (ns)')
        self.p_sigma.setLabel('left', 'σ_eq² (µm²)')
        self.p_sigma.showGrid(x=True, y=True, alpha=0.25)

        glw.ci.layout.setRowStretchFactor(0, 1)
        glw.ci.layout.setRowStretchFactor(1, 1)
        glw.ci.layout.setColumnStretchFactor(0, 1)
        glw.ci.layout.setColumnStretchFactor(1, 1)

        self._magnification: float = 25.0

        # ── σ² items ──────────────────────────────────────────────────────────
        self._sigma_reliable = self.p_sigma.plot(
            pen=pg.mkPen('#1d4ed8', width=1.1),
            symbol='o', symbolSize=5,
            symbolBrush=pg.mkBrush('#1d4ed8'), symbolPen=None)
        self._sigma_unreliable = self.p_sigma.plot(
            pen=None,
            symbol='o', symbolSize=5,
            symbolBrush=pg.mkBrush('#9ca3af'), symbolPen=None)
        self._sigma_vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#111827', style=QtCore.Qt.DashLine, width=0.95))
        self.p_sigma.addItem(self._sigma_vline)

        # ── 2D map items ──────────────────────────────────────────────────────
        self._viridis = pg.colormap.get('viridis')
        self._map_img = pg.ImageItem()
        self._map_img.setColorMap(self._viridis)
        self.p_map.addItem(self._map_img)
        self._map_scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen('k', width=1.0), brush=pg.mkBrush(None))
        self.p_map.addItem(self._map_scatter)
        self._map_excl = pg.ScatterPlotItem(
            symbol='x', size=12,
            pen=pg.mkPen('#ef4444', width=1.8), brush=pg.mkBrush(None))
        self.p_map.addItem(self._map_excl)
        self._map_center = pg.ScatterPlotItem(
            symbol='x', size=12,
            pen=pg.mkPen('w', width=2.0), brush=pg.mkBrush(None))
        self.p_map.addItem(self._map_center)

        # ── Space–time colormap items ─────────────────────────────────────────
        self._xt_cmap = pg.colormap.get('inferno')
        self._xt_img = pg.ImageItem()
        self._xt_img.setColorMap(self._xt_cmap)
        self.p_xt.addItem(self._xt_img)
        # white dashed σ(t) contour: x = σ_eq, y = time
        self._xt_sigma_curve = self.p_xt.plot(
            pen=pg.mkPen('w', width=1.8, style=QtCore.Qt.DashLine))
        # horizontal dotted marker at current time
        self._xt_vline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#aaaaaa', style=QtCore.Qt.DotLine, width=1.0))
        self.p_xt.addItem(self._xt_vline)

        # ── Radial items — current time: scatter + solid fit, both dark ─────────
        _CUR_COLOR = '#111827'
        self._radial_scatter = pg.ScatterPlotItem(
            size=8, pen=None, brush=pg.mkBrush(_CUR_COLOR + 'cc'))
        self.p_radial.addItem(self._radial_scatter)
        self._radial_fit = self.p_radial.plot(pen=pg.mkPen(_CUR_COLOR, width=2.5))

        # Extra time traces: scatter + solid fit, one color per slot (up to 5)
        self._radial_extra: List[pg.PlotDataItem] = []
        self._radial_extra_scatter: List[pg.ScatterPlotItem] = []
        for c in _EXTRA_COLORS:
            sc = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(c + 'bb'))
            self.p_radial.addItem(sc)
            self._radial_extra_scatter.append(sc)
            ln = self.p_radial.plot(pen=pg.mkPen(c, width=1.6))
            self._radial_extra.append(ln)

        self._radial_legend = self.p_radial.addLegend(offset=(10, 10))

    # ── Public API ────────────────────────────────────────────────────────────

    def set_magnification(self, M: float) -> None:
        self._magnification = max(1e-9, M)
        self.p_sigma.setLabel('left', f'σ_eq² (sample µm²,  M={M:.4g})')
        self.p_map.setLabel('bottom', 'x (sample µm)')
        self.p_map.setLabel('left', 'y (sample µm)')
        self.p_radial.setLabel('bottom', 'Distance from center (sample µm)')
        self.p_xt.setLabel('bottom', f'r (sample µm,  M={M:.4g})')

    def update_sigma(self, aligned_time_ns: np.ndarray, current_aligned_idx: int,
                     t_arr: Optional[np.ndarray],
                     sigma2: Optional[np.ndarray],
                     smooth_window: int = 1,
                     reliable_mask: Optional[np.ndarray] = None) -> None:
        M2 = self._magnification ** 2
        if t_arr is not None and sigma2 is not None and len(t_arr) > 0:
            s2 = sigma2 / M2
            w = max(1, smooth_window)
            if w > 1:
                s2 = _rolling_median(s2, w)
            if reliable_mask is not None and len(reliable_mask) == len(s2):
                self._sigma_reliable.setData(t_arr[reliable_mask], s2[reliable_mask])
                unrel = ~reliable_mask
                self._sigma_unreliable.setData(t_arr[unrel], s2[unrel])
            else:
                self._sigma_reliable.setData(t_arr, s2)
                self._sigma_unreliable.setData([], [])
        else:
            self._sigma_reliable.setData([], [])
            self._sigma_unreliable.setData([], [])
        if len(aligned_time_ns) > current_aligned_idx:
            self._sigma_vline.setValue(float(aligned_time_ns[current_aligned_idx]))

    def update_xt(self,
                  t_arr: np.ndarray,
                  r_arr_sample: np.ndarray,
                  xt_matrix: np.ndarray,
                  sigma_arr_sample: Optional[np.ndarray],
                  sigma_t_arr: Optional[np.ndarray],
                  normalize: bool,
                  current_t_ns: float) -> None:
        """Update the space–time colormap.

        t_arr           — uniform time axis (ns), shape (n_t,)
        r_arr_sample    — uniform r grid in sample µm, shape (n_r,)
        xt_matrix       — intensity, shape (n_t, n_r); xt_matrix[t_idx, r_idx]
        sigma_arr_sample— σ_eq (sample µm) at each σ time step; or None
        sigma_t_arr     — times (ns) for σ overlay; or None
        normalize       — normalize each row to [0, 1] independently
        current_t_ns    — current time marker position
        """
        if t_arr is None or len(t_arr) == 0 or xt_matrix.size == 0:
            self._xt_img.clear()
            self._xt_sigma_curve.setData([], [])
            return

        img = xt_matrix.astype(np.float32)
        if normalize:
            row_min = img.min(axis=1, keepdims=True)
            row_max = img.max(axis=1, keepdims=True)
            span = row_max - row_min
            span[span <= 0] = 1.0
            img = (img - row_min) / span

        n_t, n_r = img.shape
        t_min = float(t_arr[0])
        t_max = float(t_arr[-1]) if n_t > 1 else t_min + 1.0
        r_max = float(r_arr_sample[-1]) if len(r_arr_sample) > 0 else 1.0

        # row-major: data[t_idx, r_idx] → view (r, t)
        # translate to (r=0, t=t_min), scale by (dr, dt) per pixel
        tr = QtGui.QTransform()
        tr.translate(0.0, t_min)
        tr.scale(r_max / n_r if n_r > 0 else 1.0,
                 (t_max - t_min) / n_t if n_t > 1 else 1.0)
        self._xt_img.setTransform(tr)

        vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        self._xt_img.setImage(img, levels=(vmin, vmax))

        # σ(t) contour overlay: x = σ_eq(t), y = t
        if (sigma_arr_sample is not None and sigma_t_arr is not None
                and len(sigma_t_arr) > 0):
            self._xt_sigma_curve.setData(sigma_arr_sample, sigma_t_arr)
        else:
            self._xt_sigma_curve.setData([], [])

        self._xt_vline.setValue(current_t_ns)
        self.p_xt.setXRange(0.0, r_max, padding=0.02)
        self.p_xt.setYRange(t_min, t_max, padding=0.02)

    def update_xt_marker(self, current_t_ns: float) -> None:
        """Update only the horizontal current-time marker on p_xt (fast, slider-safe)."""
        self._xt_vline.setValue(current_t_ns)

    def no_fit_maps(self) -> None:
        self._map_img.clear()
        self._map_scatter.setData([], [])
        self._map_excl.setData([], [])
        self._map_center.setData([], [])
        self._radial_scatter.setData([], [])
        self._radial_fit.setData([], [])
        for ln in self._radial_extra:
            ln.setData([], [])
        for sc in self._radial_extra_scatter:
            sc.setData([], [])
        self._clear_radial_legend()
        self._sigma_reliable.setData([], [])
        self._sigma_unreliable.setData([], [])
        self._xt_img.clear()
        self._xt_sigma_curve.setData([], [])
        self.p_map.setTitle("2D Gaussian fit")
        self.p_radial.setTitle("Radial fit")

    def update_m1_maps(self, frame: np.ndarray, fit,
                       fit_x_all: np.ndarray, fit_y_all: np.ndarray,
                       excluded_pixels: Set[int],
                       extra_traces: Optional[List[Tuple]] = None,
                       normalize: bool = False) -> None:
        if fit is None:
            self._clear_map_items()
            self.p_map.setTitle("2D Gaussian fit — no fit at selected time")
            self.p_radial.setTitle("Radial fit — run Fit All first")
            self._clear_radial_legend()
            self._draw_radial_extra([], symmetric=True, rmax_global=1.0, normalize=False)
            return
        self._draw_map(frame, fit.popt, fit_x_all, fit_y_all,
                       excluded_pixels, fit.sigma_eq)
        self._draw_radial_m1(frame, fit, fit_x_all, fit_y_all,
                             excluded_pixels, extra_traces or [], normalize=normalize)

    def update_m2_maps(self, ref_frame: np.ndarray, current_frame: np.ndarray,
                       ref_fit, fit1d,
                       fit_x_all: np.ndarray, fit_y_all: np.ndarray,
                       excluded_pixels: Set[int],
                       ref_aligned_idx: int, aligned_time_ns: np.ndarray,
                       current_aligned_idx: int,
                       x0_ref: float, y0_ref: float,
                       extra_traces: Optional[List[Tuple]] = None,
                       normalize: bool = False) -> None:
        M = self._magnification
        self._draw_map(ref_frame, ref_fit.popt, fit_x_all, fit_y_all,
                       excluded_pixels, ref_fit.sigma_eq,
                       map_title=(f"Reference 2D fit — t=0 (peak), "
                                  f"x0={x0_ref/M:.2f}, y0={y0_ref/M:.2f} µm (sample)"))
        if fit1d is None:
            self._radial_scatter.setData([], [])
            self._radial_fit.setData([], [])
            self.p_radial.setTitle("Radial 1D fit — no result at selected time")
            self._clear_radial_legend()
            self._draw_radial_extra([], symmetric=False, rmax_global=1.0, normalize=False)
            return
        self._draw_radial_m2(current_frame, fit1d, fit_x_all, fit_y_all,
                             excluded_pixels, x0_ref, y0_ref,
                             aligned_time_ns, current_aligned_idx,
                             extra_traces or [], normalize=normalize)

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _clear_map_items(self) -> None:
        self._map_img.clear()
        self._map_scatter.setData([], [])
        self._map_excl.setData([], [])
        self._map_center.setData([], [])

    def _draw_map(self, frame: np.ndarray, popt: np.ndarray,
                  fit_x: np.ndarray, fit_y: np.ndarray,
                  excluded_pixels: Set[int], sigma_eq: float,
                  map_title: str = "") -> None:
        M = self._magnification
        xpad, ypad = PITCH_X_UM * 0.35, PITCH_Y_UM * 0.35
        xmin = float(np.min(fit_x) - xpad)
        xmax = float(np.max(fit_x) + xpad)
        ymin = float(np.min(fit_y) - ypad)
        ymax = float(np.max(fit_y) + ypad)
        N = 220
        gx = np.linspace(xmin, xmax, N)
        gy = np.linspace(ymin, ymax, N)
        gxx, gyy = np.meshgrid(gx, gy)
        z_grid = (popt[0] * np.exp(-(
            ((gxx - popt[1]) ** 2) / (2 * popt[3] ** 2) +
            ((gyy - popt[2]) ** 2) / (2 * popt[4] ** 2)
        )) + popt[5])
        zmin, zmax = float(np.min(z_grid)), float(np.max(z_grid))
        if zmax <= zmin:
            zmax = zmin + 1e-9

        tr = QtGui.QTransform()
        tr.translate(xmin / M, ymin / M)
        tr.scale((xmax - xmin) / M / N, (ymax - ymin) / M / N)
        self._map_img.setTransform(tr)
        self._map_img.setImage(z_grid, levels=(zmin, zmax))

        norm = np.clip((frame - zmin) / (zmax - zmin), 0.0, 1.0)
        colors = self._viridis.map(norm.astype(np.float32), mode='byte')
        spots = [{'pos': (fit_x[i] / M, fit_y[i] / M),
                  'brush': pg.mkBrush(*colors[i]),
                  'pen': pg.mkPen('k', width=1.0),
                  'size': 8}
                 for i in range(len(fit_x))]
        self._map_scatter.setData(spots)

        if excluded_pixels:
            ex = np.array(sorted(excluded_pixels), dtype=int)
            self._map_excl.setData(x=fit_x[ex] / M, y=fit_y[ex] / M)
        else:
            self._map_excl.setData([], [])
        self._map_center.setData([float(popt[1]) / M], [float(popt[2]) / M])

        sx_s = float(popt[3]) / M
        sy_s = float(popt[4]) / M
        first_line = (map_title if map_title else
                      f"2D Gaussian fit — x0={popt[1]/M:.2f}, "
                      f"y0={popt[2]/M:.2f}, σ_eq={sigma_eq/M:.2f} µm (sample)")
        self.p_map.setTitle(
            f"{first_line}<br>σ_x={sx_s:.2f}, σ_y={sy_s:.2f} µm (sample)")

    def _draw_radial_m1(self, frame: np.ndarray, fit,
                        fit_x: np.ndarray, fit_y: np.ndarray,
                        excluded_pixels: Set[int],
                        extra_traces: List[Tuple],
                        normalize: bool = False) -> None:
        M = self._magnification
        mask = np.ones(len(fit_x), dtype=bool)
        if excluded_pixels:
            mask[np.array(sorted(excluded_pixels), dtype=int)] = False
        x_u, y_u, z_u = fit_x[mask], fit_y[mask], frame[mask]
        r = np.sqrt((x_u - fit.popt[1]) ** 2 + (y_u - fit.popt[2]) ** 2)
        order = np.argsort(r)
        r_s, z_s = r[order] / M, z_u[order]
        rmax = float(np.max(r_s))
        r_sym = np.concatenate([-r_s[::-1], r_s])
        z_sym = np.concatenate([z_s[::-1], z_s])
        r_line = np.linspace(-rmax, rmax, 401)
        sigma_safe = max(float(fit.sigma_eq), 1e-6) / M
        A = float(fit.popt[0])
        offset = float(fit.popt[5])
        fit_line = A * np.exp(-(r_line ** 2) / (2 * sigma_safe ** 2)) + offset
        if normalize and A > 0:
            z_sym = (z_sym - offset) / A
            fit_line = (fit_line - offset) / A
        self._radial_scatter.setData(x=r_sym, y=z_sym)
        self._radial_fit.setData(r_line, fit_line)
        self.p_radial.setTitle(f"Radial fit (σ = {fit.sigma_eq/M:.2f} µm (sample))")
        self.p_radial.setLabel('bottom', 'Distance from fitted center (sample µm)')
        self.p_radial.setLabel('left', 'Normalized intensity' if normalize else 'Counts')
        self._clear_radial_legend()
        self._radial_legend.addItem(self._radial_fit, "current")
        self._draw_radial_extra(extra_traces, symmetric=True, rmax_global=rmax,
                                normalize=normalize)

    def _draw_radial_m2(self, current_frame: np.ndarray, fit1d,
                        fit_x: np.ndarray, fit_y: np.ndarray,
                        excluded_pixels: Set[int],
                        x0_ref: float, y0_ref: float,
                        aligned_time_ns: np.ndarray, current_aligned_idx: int,
                        extra_traces: List[Tuple],
                        normalize: bool = False) -> None:
        M = self._magnification
        mask = np.ones(len(fit_x), dtype=bool)
        if excluded_pixels:
            mask[np.array(sorted(excluded_pixels), dtype=int)] = False
        x_u, y_u, z_u = fit_x[mask], fit_y[mask], current_frame[mask]
        r = np.sqrt((x_u - x0_ref) ** 2 + (y_u - y0_ref) ** 2)
        order = np.argsort(r)
        r_s, z_s = r[order] / M, z_u[order]
        A1, sigma1, offset1 = [float(v) for v in fit1d.popt]
        rmax = float(np.max(r_s))
        r_line = np.linspace(0.0, rmax, 401)
        sigma_sample = max(abs(sigma1), 1e-6) / M
        fit_line = A1 * np.exp(-(r_line ** 2) / (2 * sigma_sample ** 2)) + offset1
        if normalize and A1 > 0:
            z_s = (z_s - offset1) / A1
            fit_line = (fit_line - offset1) / A1
        self._radial_scatter.setData(x=r_s, y=z_s)
        self._radial_fit.setData(r_line, fit_line)
        t = (float(aligned_time_ns[current_aligned_idx])
             if len(aligned_time_ns) > current_aligned_idx else 0.0)
        self.p_radial.setTitle(
            f"Radial 1D fit — t={t:.4f} ns (after t₀), σ={fit1d.sigma_eq/M:.2f} µm (sample)")
        self.p_radial.setLabel('bottom', 'Distance from fixed center (sample µm)')
        self.p_radial.setLabel('left', 'Normalized intensity' if normalize else 'Counts')
        self._clear_radial_legend()
        self._radial_legend.addItem(self._radial_fit, f"t = {t:.3f} ns (cur)")
        self._draw_radial_extra(extra_traces, symmetric=False, rmax_global=rmax,
                                normalize=normalize)

    def _clear_radial_legend(self) -> None:
        self._radial_legend.removeItem(self._radial_fit)
        for ln in self._radial_extra:
            self._radial_legend.removeItem(ln)

    def _draw_radial_extra(self, extra_traces: List[Tuple],
                           symmetric: bool = False,
                           rmax_global: float = 1.0,
                           normalize: bool = False) -> None:
        """Draw scatter + fit-line overlays for up to 5 extra time slices.

        Each extra_traces entry: (r_sample_arr, z_arr, popt_sample, t_ns)
          r_sample_arr: r values in sample µm (used for scatter and rmax)
          z_arr: intensity values (same length as r_sample_arr), or None
          popt_sample: [A, sigma_sample_µm, offset]
        """
        for i, (ln, sc) in enumerate(zip(self._radial_extra, self._radial_extra_scatter)):
            if i >= len(extra_traces):
                ln.setData([], [])
                sc.setData([], [])
                continue
            r_sample, z_arr, popt, t_ns = extra_traces[i]
            if popt is None or len(r_sample) == 0:
                ln.setData([], [])
                sc.setData([], [])
                continue
            A = float(popt[0])
            sigma = max(abs(float(popt[1])), 1e-6)
            offset = float(popt[2])
            rmax = max(float(np.max(r_sample)), rmax_global)
            r_line = np.linspace(-rmax, rmax, 401) if symmetric else np.linspace(0.0, rmax, 401)
            if normalize and A > 0:
                fit_line = np.exp(-(r_line ** 2) / (2 * sigma ** 2))
            else:
                fit_line = A * np.exp(-(r_line ** 2) / (2 * sigma ** 2)) + offset
            ln.setData(r_line, fit_line)
            self._radial_legend.addItem(ln, f"t = {t_ns:.3f} ns")
            if z_arr is not None and len(z_arr) > 0:
                r_sc = np.asarray(r_sample, dtype=np.float64)
                z_sc = np.asarray(z_arr, dtype=np.float64)
                if symmetric:
                    r_sc = np.concatenate([-r_sc[::-1], r_sc])
                    z_sc = np.concatenate([z_sc[::-1], z_sc])
                if normalize and A > 0:
                    z_sc = (z_sc - offset) / A
                sc.setData(x=r_sc, y=z_sc)
            else:
                sc.setData([], [])

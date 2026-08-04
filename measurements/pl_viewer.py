"""
measurements/pl_viewer.py

Simple PL viewer: loads all .asc files from a folder, shows a file list
on the left, and displays the PL map (spatial vs wavelength) on the right.

Fits available at the current cursor position:
  • Spectral peak  — Lorentzian on the horizontal linecut (spectrum)
  • Spatial width  — Gaussian on the vertical linecut (spatial profile)

Usage:
    python -m measurements.pl_viewer [folder]
    or via run_pl_viewer.ps1
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

try:
    from scipy.optimize import curve_fit as _curve_fit
except ImportError:
    _curve_fit = None

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k", useOpenGL=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Andor .asc loader
# ─────────────────────────────────────────────────────────────────────────────

def _parse_header(path: Path) -> dict:
    meta: dict = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    return meta


def _load_andor_ascii(path: Path) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    """Return (meta, wavelength_nm, image) where image is (n_spatial, n_wl).

    Handles two .asc dialects:
    • Headered (lines starting with '#', whitespace-separated): standard Andor
    • Headerless CSV (comma-separated, optional trailing comma): direct export
    """
    meta = _parse_header(path)

    # Peek at the first data line to detect delimiter
    first_data = ""
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                first_data = line.strip()
                break

    if "," in first_data:
        # CSV format (may have trailing comma per row → pandas handles it gracefully)
        raw = pd.read_csv(path, header=None, comment="#").values.astype(np.float64)
        # Drop all-NaN trailing column introduced by trailing comma
        valid_cols = ~np.all(np.isnan(raw), axis=0)
        raw = raw[:, valid_cols]
        wl    = raw[:, 0]
        image = raw[:, 1:].T   # (n_spatial, n_wl)
    else:
        # Whitespace-separated, may have '#' header
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        cols = str(meta.get("columns", "")).lower()
        if data.shape[1] > 1 and "wavelength" in cols:
            wl    = data[:, 0]
            image = data[:, 1:].T   # (n_spatial, n_wl)
        else:
            wl    = None
            image = data

    return meta, wl, image


# ─────────────────────────────────────────────────────────────────────────────
#  Image / linecut helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_crop(image: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    """crop = (top, bottom, left, right) — rows in spatial, cols in wavelength."""
    a = np.asarray(image)
    if a.ndim != 2:
        return a
    h, w = a.shape
    t, b, l, r = (max(0, min(int(v), h - 1)) if i < 2 else max(0, min(int(v), w - 1))
                  for i, v in enumerate(crop))
    return a[t: max(t + 1, h - b), l: max(l + 1, w - r)]


def _crop_wl(wl: Optional[np.ndarray], crop: Tuple[int, int, int, int],
             orig_w: int) -> Optional[np.ndarray]:
    if wl is None:
        return None
    arr = np.asarray(wl, dtype=float).ravel()
    if arr.size != orig_w:
        return arr
    l, r = int(crop[2]), int(crop[3])
    if r > 0:
        arr = arr[: max(0, arr.size - r)]
    if l > 0:
        arr = arr[min(l, arr.size):]
    return arr


def _linecut_h(image: np.ndarray, row: int, width: int) -> Optional[np.ndarray]:
    a = np.asarray(image)
    if a.ndim != 2:
        return None
    h = a.shape[0]
    width = max(1, int(width))
    half  = width // 2
    r1 = max(0, int(row) - half)
    r2 = min(h, int(row) + half + (1 if width % 2 else 0))
    return None if r2 <= r1 else a[r1:r2, :].sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Fitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        return np.zeros_like(x)
    dx = x - float(center)
    return float(amplitude) * float(gamma) ** 2 / (dx ** 2 + float(gamma) ** 2)


def _gaussian_with_offset(x, amplitude, center, sigma, offset):
    return amplitude * np.exp(-((np.asarray(x, dtype=float) - center) ** 2) / (2.0 * sigma ** 2)) + offset


def _noise_floor(y: np.ndarray) -> float:
    v = np.asarray(y, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 4:
        return 0.0
    diffs = np.diff(v)
    mad   = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
    return float(1.4826 * mad / np.sqrt(2.0)) if (np.isfinite(mad) and mad > 0) else float(np.nanstd(v))


def fit_lorentzian_peak(wl: np.ndarray, intensity: np.ndarray,
                        center_guess: float, window_nm: float,
                        *, min_points: int = 7, min_snr: float = 2.0) -> Optional[dict]:
    """Fit a single Lorentzian peak near center_guess within ±window_nm."""
    x = np.asarray(wl,        dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size < min_points:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x - center_guess) <= window_nm)
    if mask.sum() < min_points:
        return None
    xs, ys = x[mask], y[mask]
    bl   = float(np.nanpercentile(ys, 10.0))
    yb   = ys - bl
    amp0 = float(np.nanmax(yb))
    if not np.isfinite(amp0) or amp0 <= 0:
        return None
    nf = _noise_floor(ys)
    if nf > 0 and amp0 < min_snr * nf:
        return None
    idx   = int(np.nanargmax(yb))
    c0    = float(xs[idx])
    y_half = 0.5 * amp0
    left_  = np.where(yb[:idx] <= y_half)[0]
    right_ = np.where(yb[idx + 1:] <= y_half)[0]
    if left_.size > 0 and right_.size > 0:
        g0 = max(0.1, float(xs[idx + 1 + right_[0]]) - float(xs[left_[-1]])) * 0.5
    else:
        g0 = max(0.1, 0.25 * window_nm)
    a_fit, c_fit, g_fit = amp0, c0, g0
    if _curve_fit is not None:
        try:
            lo = [0.0,  center_guess - window_nm, 0.05]
            hi = [max(1.0, 4.0 * amp0), center_guess + window_nm, 12.0]
            p0 = np.clip([a_fit, c_fit, g_fit], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            popt, _ = _curve_fit(lambda xx, a, m, g: _lorentzian(xx, a, m, g),
                                  xs, yb, p0=p0, bounds=(lo, hi), maxfev=8000)
            a_fit, c_fit, g_fit = [float(v) for v in popt]
        except Exception:
            pass
    if not (0.05 <= g_fit <= 15.0):
        return None
    if abs(c_fit - center_guess) > window_nm * 1.2:
        return None
    fwhm_nm = 2.0 * g_fit
    # FWHM in meV
    c = c_fit
    half = 0.5 * fwhm_nm
    fwhm_mev = (1239.841984 / max(c - half, 1e-9) - 1239.841984 / (c + half)) * 1e3 if c > half else np.nan
    return {
        "center_nm": c_fit, "gamma_nm": g_fit, "fwhm_nm": fwhm_nm,
        "fwhm_mev": fwhm_mev, "amplitude": max(0.0, a_fit), "offset": bl,
    }


def fit_gaussian_spatial(rows: np.ndarray, intensities: np.ndarray,
                          center_guess: float, window_px: float) -> Optional[dict]:
    """Fit a Gaussian to the spatial profile (intensity vs row)."""
    x = np.asarray(rows,       dtype=float).ravel()
    y = np.asarray(intensities, dtype=float).ravel()
    if x.size != y.size or x.size < 5:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x - center_guess) <= window_px)
    if mask.sum() < 5:
        mask = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[mask], y[mask]
    if xs.size < 5:
        return None
    idx  = int(np.argmax(np.abs(ys)))
    amp0 = float(ys[idx])
    ctr0 = float(xs[idx])
    sig0 = max(float(window_px) / 4.0, 1.0)
    off0 = float(np.median(ys))
    if _curve_fit is None:
        return None
    try:
        half_range = max(window_px, 5.0)
        lo = [-np.inf, ctr0 - half_range, 0.3,         -np.inf]
        hi = [ np.inf, ctr0 + half_range, half_range,   np.inf]
        p0 = np.clip([amp0, ctr0, sig0, off0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        popt, pcov = _curve_fit(_gaussian_with_offset, xs, ys,
                                p0=p0, bounds=(lo, hi), maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        amp, ctr, sig, off = [float(v) for v in popt]
        sig_err = float(perr[2])
        return {
            "center_px": ctr, "sigma_px": abs(sig), "fwhm_px": 2.355 * abs(sig),
            "amplitude": amp, "offset": off, "sigma_err": sig_err,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PLFile:
    path:          Path
    stem:          str
    wavelength_nm: Optional[np.ndarray]   # (n_wl,) or None
    image:         np.ndarray             # (n_spatial, n_wl) — full, uncropped
    meta:          dict


def load_asc_folder(folder: str) -> List[PLFile]:
    """Load all *.asc files in folder."""
    files: List[PLFile] = []
    for path in sorted(Path(folder).glob("*.asc")):
        try:
            meta, wl, image = _load_andor_ascii(path)
        except Exception as exc:
            print(f"Skip {path.name}: {exc}")
            continue
        files.append(PLFile(
            path=path,
            stem=path.stem,
            wavelength_nm=wl,
            image=image,
            meta=meta,
        ))
    return files


# ─────────────────────────────────────────────────────────────────────────────
#  PL map display widget
# ─────────────────────────────────────────────────────────────────────────────

class PLMapView(QtWidgets.QWidget):
    """PL image (spatial × wavelength) with movable cursor and linecuts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._img_data: Optional[np.ndarray] = None  # (n_spatial, n_wl)
        self._wl:       Optional[np.ndarray] = None
        self._h = self._w = 0
        self._build_ui()

    def _build_ui(self) -> None:
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(2, 2, 2, 2)
        vlay.setSpacing(4)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Linecut width:"))
        self.lc_width_spin = QtWidgets.QSpinBox()
        self.lc_width_spin.setRange(1, 1000)
        self.lc_width_spin.setValue(1)
        self.lc_width_spin.setFixedWidth(60)
        ctrl.addWidget(self.lc_width_spin)
        ctrl.addStretch()
        vlay.addLayout(ctrl)

        self._glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self._glw)

        # Image plot (row 0, col 0)
        self._img_plot = self._glw.addPlot(row=0, col=0)
        self._img_plot.getViewBox().invertY(True)
        self._img_plot.setLabel("bottom", "Wavelength (nm)")
        self._img_plot.setLabel("left", "Spatial pixel")
        for ax in ("bottom", "left"):
            self._img_plot.getAxis(ax).setPen(pg.mkPen("k"))
            self._img_plot.getAxis(ax).setTextPen(pg.mkPen("k"))
        self._img_item = pg.ImageItem()
        self._img_plot.addItem(self._img_item)

        # Movable lines
        self._hline = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen("y", width=1.5), hoverPen=pg.mkPen("y", width=2.5),
        )
        self._vline = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("c", width=1.5), hoverPen=pg.mkPen("c", width=2.5),
        )
        self._img_plot.addItem(self._hline)
        self._img_plot.addItem(self._vline)
        self._hline.sigPositionChangeFinished.connect(self._refresh_hlinecut)
        self._vline.sigPositionChangeFinished.connect(self._refresh_vlinecut)

        # Histogram LUT (row 0, col 1) — narrow
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._img_item)
        self._glw.addItem(self._hist, row=0, col=1)

        # Vertical linecut — spatial profile at cursor wavelength (row 0, col 2)
        self._vlc_plot = self._glw.addPlot(row=0, col=2)
        self._vlc_plot.setLabel("bottom", "Intensity")
        self._vlc_plot.setLabel("left", "")
        for ax in ("bottom", "left"):
            self._vlc_plot.getAxis(ax).setPen(pg.mkPen("k"))
            self._vlc_plot.getAxis(ax).setTextPen(pg.mkPen("k"))
        self._vlc_plot.getViewBox().invertY(True)
        self._vlc_plot.setYLink(self._img_plot)
        self._vlc_curve     = self._vlc_plot.plot(pen=pg.mkPen("#ff7f0e", width=1.2))
        self._vlc_fit_curve = self._vlc_plot.plot(
            pen=pg.mkPen("r", width=1.8, style=QtCore.Qt.DashLine))

        # Horizontal linecut — spectrum at cursor row (row 1, col 0, colspan=3)
        self._hlc_plot = self._glw.addPlot(row=1, col=0, colspan=3)
        self._hlc_plot.setLabel("bottom", "Wavelength (nm)")
        self._hlc_plot.setLabel("left", "Intensity")
        for ax in ("bottom", "left"):
            self._hlc_plot.getAxis(ax).setPen(pg.mkPen("k"))
            self._hlc_plot.getAxis(ax).setTextPen(pg.mkPen("k"))
        self._hlc_curve     = self._hlc_plot.plot(pen=pg.mkPen("#1f77b4", width=1.2))
        self._hlc_fit_curve = self._hlc_plot.plot(
            pen=pg.mkPen("#e07000", width=1.8, style=QtCore.Qt.DashLine))

        # Stretch factors
        self._glw.ci.layout.setRowStretchFactor(0, 4)
        self._glw.ci.layout.setRowStretchFactor(1, 2)
        self._glw.ci.layout.setColumnMaximumWidth(1, 65)
        self._glw.ci.layout.setColumnStretchFactor(0, 4)
        self._glw.ci.layout.setColumnStretchFactor(2, 2)

        self.lc_width_spin.valueChanged.connect(lambda _: self._refresh_hlinecut())

    # ── public ────────────────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray, wl: Optional[np.ndarray] = None,
                  title: str = "") -> None:
        self._img_data = np.asarray(img, dtype=float)
        self._h, self._w = self._img_data.shape
        self._wl = None if wl is None else np.asarray(wl, dtype=float).ravel()

        # row-major: axis-0=y, axis-1=x.  flipud puts spatial row 0 at bottom so
        # invertY(True) makes it appear at the top (standard camera convention).
        disp = np.flipud(self._img_data)   # (n_spatial, n_wl): y=spatial, x=wavelength
        if self._wl is not None and self._wl.size == self._w:
            x0     = float(self._wl[0])
            x_span = float(self._wl[-1] - self._wl[0]) or 1.0
        else:
            x0, x_span = 0.0, float(self._w) or 1.0

        self._img_item.setImage(
            disp if np.isfinite(disp).any() else np.zeros_like(disp),
            autoLevels=True,
        )
        self._img_item.setRect(QtCore.QRectF(x0, 0, x_span, float(self._h)))
        self._img_plot.setXRange(x0, x0 + x_span, padding=0.01)
        self._img_plot.setYRange(0, float(self._h), padding=0)
        # Reset cursor to centre
        self._hline.setValue(self._h / 2.0)
        self._vline.setValue(x0 + x_span / 2.0)
        # Clear fit overlays
        self._hlc_fit_curve.setData([], [])
        self._vlc_fit_curve.setData([], [])
        if title:
            self._img_plot.setTitle(title)
        self._refresh_hlinecut()
        self._refresh_vlinecut()

    def set_spec_fit_overlay(self, x: np.ndarray, y: np.ndarray) -> None:
        if x is not None and len(x) > 0:
            self._hlc_fit_curve.setData(np.asarray(x, dtype=float),
                                        np.asarray(y, dtype=float))
        else:
            self._hlc_fit_curve.setData([], [])

    def set_spatial_fit_overlay(self, intensity_fit: np.ndarray,
                                  rows_fine: np.ndarray) -> None:
        """intensity_fit on x-axis, rows_fine on y-axis (matches vlc orientation)."""
        if intensity_fit is not None and len(intensity_fit) > 0:
            self._vlc_fit_curve.setData(np.asarray(intensity_fit, dtype=float),
                                        np.asarray(rows_fine, dtype=float))
        else:
            self._vlc_fit_curve.setData([], [])

    def current_hlinecut(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (wl_axis, spectrum) for the current horizontal linecut."""
        if self._img_data is None or self._h == 0:
            return None, None
        row_d   = int(np.clip(round(self._hline.value()), 0, self._h - 1))
        row_raw = self._h - 1 - row_d
        lc = _linecut_h(self._img_data, row_raw, int(self.lc_width_spin.value()))
        if lc is None:
            return None, None
        x = (self._wl if (self._wl is not None and self._wl.size == lc.size)
             else np.arange(lc.size, dtype=float))
        return x, lc

    def current_vlinecut(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (rows, intensity) for the current vertical linecut at cursor λ."""
        if self._img_data is None or self._h == 0 or self._w == 0:
            return None, None
        pos = float(self._vline.value())
        if self._wl is not None and self._wl.size == self._w:
            col = int(np.argmin(np.abs(self._wl - pos)))
        else:
            col = int(np.clip(round(pos), 0, self._w - 1))
        col = max(0, min(col, self._w - 1))
        # flip so display row 0 = top (row h-1 in data)
        profile = self._img_data[::-1, col]
        rows    = np.arange(self._h, dtype=float)
        return rows, profile

    def cursor_wavelength(self) -> Optional[float]:
        return float(self._vline.value()) if self._w > 0 else None

    def cursor_row_display(self) -> Optional[int]:
        if self._h == 0:
            return None
        return int(np.clip(round(self._hline.value()), 0, self._h - 1))

    # ── internal ──────────────────────────────────────────────────────────────

    def _refresh_hlinecut(self) -> None:
        x, lc = self.current_hlinecut()
        if lc is None:
            self._hlc_curve.setData([], [])
        else:
            self._hlc_curve.setData(x, lc)
        self._hlc_fit_curve.setData([], [])

    def _refresh_vlinecut(self) -> None:
        rows, profile = self.current_vlinecut()
        if profile is None:
            self._vlc_curve.setData([], [])
        else:
            # vlinecut: intensity on x-axis, row on y-axis (inverted to match image)
            self._vlc_curve.setData(profile, rows)
        self._vlc_fit_curve.setData([], [])


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class PLViewer(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PL Viewer")
        self.resize(1400, 960)

        self.files:   List[PLFile] = []
        self.current: Optional[PLFile] = None
        self._crop = (0, 0, 0, 0)

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
        self.map_view = PLMapView()
        root.addWidget(self.map_view, stretch=1)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(240)
        w.setMaximumWidth(300)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(4)
        lay.setContentsMargins(0, 0, 4, 0)

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
        lay.addWidget(fg)

        # ── File list ──
        sg = QtWidgets.QGroupBox("Files")
        sl = QtWidgets.QVBoxLayout(sg)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.file_list.setAlternatingRowColors(True)
        sl.addWidget(self.file_list)
        lay.addWidget(sg, stretch=1)

        # ── Crop ──
        cg = QtWidgets.QGroupBox("Crop  (spatial: T/B,  wavelength: L/R)")
        cl = QtWidgets.QGridLayout(cg)
        cl.setHorizontalSpacing(4)
        cl.setVerticalSpacing(4)
        cl.setContentsMargins(6, 6, 6, 6)
        for col, lbl in enumerate(("T", "B", "L", "R")):
            cl.addWidget(QtWidgets.QLabel(lbl), 0, col)
        self.crop_t = QtWidgets.QSpinBox(); self.crop_b = QtWidgets.QSpinBox()
        self.crop_l = QtWidgets.QSpinBox(); self.crop_r = QtWidgets.QSpinBox()
        for i, sp in enumerate((self.crop_t, self.crop_b, self.crop_l, self.crop_r)):
            sp.setRange(0, 10000)
            sp.setFixedWidth(56)
            cl.addWidget(sp, 1, i)
        self.crop_apply_btn = QtWidgets.QPushButton("Apply Crop")
        cl.addWidget(self.crop_apply_btn, 2, 0, 1, 4)
        lay.addWidget(cg)

        # ── Spectral Peak Fit ──
        spg = QtWidgets.QGroupBox("Spectral Peak Fit  (Lorentzian)")
        spl = QtWidgets.QGridLayout(spg)
        spl.setHorizontalSpacing(6)
        spl.setVerticalSpacing(4)
        spl.setContentsMargins(6, 6, 6, 6)
        spl.addWidget(QtWidgets.QLabel("Center guess (nm):"), 0, 0)
        self.spec_center_edit = QtWidgets.QLineEdit()
        self.spec_center_edit.setPlaceholderText("e.g. 855")
        spl.addWidget(self.spec_center_edit, 0, 1)
        spl.addWidget(QtWidgets.QLabel("Window (nm):"), 1, 0)
        self.spec_window_edit = QtWidgets.QLineEdit("15")
        spl.addWidget(self.spec_window_edit, 1, 1)
        self.spec_fit_btn = QtWidgets.QPushButton("Fit Spectrum at Cursor")
        spl.addWidget(self.spec_fit_btn, 2, 0, 1, 2)
        self.spec_result_lbl = QtWidgets.QLabel("")
        self.spec_result_lbl.setWordWrap(True)
        self.spec_result_lbl.setStyleSheet("color: #444; font-size: 8pt;")
        spl.addWidget(self.spec_result_lbl, 3, 0, 1, 2)
        lay.addWidget(spg)

        # ── Spatial Width Fit ──
        wfg = QtWidgets.QGroupBox("Spatial Width Fit  (Gaussian)")
        wfl = QtWidgets.QGridLayout(wfg)
        wfl.setHorizontalSpacing(6)
        wfl.setVerticalSpacing(4)
        wfl.setContentsMargins(6, 6, 6, 6)
        wfl.addWidget(QtWidgets.QLabel("Center guess (px):"), 0, 0)
        self.spat_center_edit = QtWidgets.QLineEdit()
        self.spat_center_edit.setPlaceholderText("auto (cursor)")
        wfl.addWidget(self.spat_center_edit, 0, 1)
        wfl.addWidget(QtWidgets.QLabel("Window (px):"), 1, 0)
        self.spat_window_edit = QtWidgets.QLineEdit("30")
        wfl.addWidget(self.spat_window_edit, 1, 1)
        self.spat_fit_btn = QtWidgets.QPushButton("Fit Spatial at Cursor")
        wfl.addWidget(self.spat_fit_btn, 2, 0, 1, 2)
        self.spat_result_lbl = QtWidgets.QLabel("")
        self.spat_result_lbl.setWordWrap(True)
        self.spat_result_lbl.setStyleSheet("color: #444; font-size: 8pt;")
        wfl.addWidget(self.spat_result_lbl, 3, 0, 1, 2)
        lay.addWidget(wfg)

        return w

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.folder_btn.clicked.connect(self._browse_folder)
        self.load_btn.clicked.connect(self._do_load)
        self.folder_edit.returnPressed.connect(self._do_load)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        self.crop_apply_btn.clicked.connect(self._on_apply_crop)
        self.spec_fit_btn.clicked.connect(self._fit_spectral)
        self.spat_fit_btn.clicked.connect(self._fit_spatial)

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
        self.files = load_asc_folder(folder)
        self._populate_file_list()
        n = len(self.files)
        self.statusBar().showMessage(
            f"Found {n} .asc file{'s' if n != 1 else ''} in {folder}"
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
        for f in self.files:
            self.file_list.addItem(f.stem)
        self.file_list.blockSignals(False)

    def _on_file_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.files):
            return
        self.current = self.files[idx]
        self._reload_current()

    # ── Display ───────────────────────────────────────────────────────────────

    def _reload_current(self) -> None:
        pf = self.current
        if pf is None:
            return
        self.statusBar().showMessage(f"Loading {pf.path.name}…")
        QtWidgets.QApplication.processEvents()

        orig_h, orig_w = pf.image.shape
        img = _apply_crop(pf.image, self._crop)
        wl  = _crop_wl(pf.wavelength_nm, self._crop, orig_w)

        self.spec_result_lbl.setText("")
        self.spat_result_lbl.setText("")

        self.map_view.set_image(img, wl, title=pf.stem)
        h, w = img.shape
        meta_bits = [f"{h}×{w} px"]
        exp = pf.meta.get("exposure time") or pf.meta.get("exposure_time")
        if exp:
            meta_bits.append(f"exp={exp} s")
        grat = pf.meta.get("grating") or pf.meta.get("grating_lines")
        if grat:
            meta_bits.append(f"grating={grat}")
        self.statusBar().showMessage(f"{pf.stem}  |  {'  |  '.join(meta_bits)}")

    def _on_apply_crop(self) -> None:
        self._crop = (
            self.crop_t.value(), self.crop_b.value(),
            self.crop_l.value(), self.crop_r.value(),
        )
        self._reload_current()

    # ── Spectral peak fit ─────────────────────────────────────────────────────

    def _fit_spectral(self) -> None:
        wl, lc = self.map_view.current_hlinecut()
        if lc is None or wl is None:
            self.statusBar().showMessage("No linecut data — load a file first.")
            return
        try:
            center_guess = float(self.spec_center_edit.text())
        except ValueError:
            idx_peak = int(np.argmax(lc))
            center_guess = float(wl[idx_peak])
        try:
            window = max(1.0, float(self.spec_window_edit.text()))
        except ValueError:
            window = 15.0

        result = fit_lorentzian_peak(wl, lc, center_guess, window, min_snr=1.0)
        if result is None:
            self.spec_result_lbl.setText("Fit failed (low SNR or out of range).")
            self.map_view.set_spec_fit_overlay([], [])
            return

        # Overlay
        x_fine = np.linspace(float(wl[0]), float(wl[-1]), 600)
        y_fit  = result["offset"] + _lorentzian(x_fine, result["amplitude"],
                                                  result["center_nm"], result["gamma_nm"])
        self.map_view.set_spec_fit_overlay(x_fine, y_fit)

        fwhm_mev = result["fwhm_mev"]
        msg = (f"Center: {result['center_nm']:.3f} nm\n"
               f"FWHM:   {result['fwhm_nm']:.3f} nm")
        if np.isfinite(fwhm_mev):
            msg += f"  ({fwhm_mev:.2f} meV)"
        self.spec_result_lbl.setText(msg)
        self.statusBar().showMessage(
            f"Spec fit: λ₀ = {result['center_nm']:.3f} nm, "
            f"FWHM = {result['fwhm_nm']:.3f} nm"
            + (f" = {fwhm_mev:.2f} meV" if np.isfinite(fwhm_mev) else "")
        )

    # ── Spatial width fit ─────────────────────────────────────────────────────

    def _fit_spatial(self) -> None:
        rows, profile = self.map_view.current_vlinecut()
        if profile is None or rows is None:
            self.statusBar().showMessage("No spatial profile — load a file first.")
            return
        try:
            center_guess = float(self.spat_center_edit.text())
        except ValueError:
            center_guess = float(rows[int(np.argmax(np.abs(profile)))])
        try:
            window = max(2.0, float(self.spat_window_edit.text()))
        except ValueError:
            window = 30.0

        result = fit_gaussian_spatial(rows, profile, center_guess, window)
        if result is None:
            self.spat_result_lbl.setText("Fit failed.")
            self.map_view.set_spatial_fit_overlay([], [])
            return

        # Overlay (intensity on x-axis, rows on y-axis — matches vlinecut orientation)
        rows_fine     = np.linspace(float(rows[0]), float(rows[-1]), 400)
        intensity_fit = _gaussian_with_offset(rows_fine, result["amplitude"],
                                               result["center_px"], result["sigma_px"],
                                               result["offset"])
        self.map_view.set_spatial_fit_overlay(intensity_fit, rows_fine)

        msg = (f"Center: {result['center_px']:.2f} px\n"
               f"σ:      {result['sigma_px']:.2f} ± {result['sigma_err']:.2f} px\n"
               f"FWHM:   {result['fwhm_px']:.2f} px")
        self.spat_result_lbl.setText(msg)
        self.statusBar().showMessage(
            f"Spatial fit: center = {result['center_px']:.2f} px, "
            f"FWHM = {result['fwhm_px']:.2f} px  (σ = {result['sigma_px']:.2f} px)"
        )


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
    win = PLViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

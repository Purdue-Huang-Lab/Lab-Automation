"""
pl_trpl_power_gate_dep_viewer_v2.py

PL + TRPL power/gate-dependence offline viewer — version 2.
All plots use pyqtgraph. Keeps full v1 layout/logic; adds:
  • Per-pixel t=0 alignment + smoothing for TRPL Histogram and Last Five Histograms
  • "Fit End Time" spinbox, "Fit Current Map" and "Fit All Maps" buttons (no auto-fit,
    no downsampling) in the Gaussian Fit panel
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal

try:
    from scipy.optimize import curve_fit as _curve_fit
except ImportError:
    _curve_fit = None

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CROP = (50, 50, 200, 200)
EV_NM = 1239.841984
NUM_SPAD = 23
SPAD_PIXEL_COORDS: Dict[int, Tuple[float, float]] = {
    0: (0.0, 4.0), 1: (1.0, 4.0), 2: (2.0, 4.0), 3: (3.0, 4.0), 4: (4.0, 4.0),
    5: (0.5, 3.0), 6: (1.5, 3.0), 7: (2.5, 3.0), 8: (3.5, 3.0),
    9: (0.0, 2.0), 10: (1.0, 2.0), 11: (2.0, 2.0), 12: (3.0, 2.0), 13: (4.0, 2.0),
    14: (0.5, 1.0), 15: (1.5, 1.0), 16: (2.5, 1.0), 17: (3.5, 1.0),
    18: (0.0, 0.0), 19: (1.0, 0.0), 20: (2.0, 0.0), 21: (3.0, 0.0), 22: (4.0, 0.0),
}
SPAD_MAP_COORDS = np.array([SPAD_PIXEL_COORDS[i] for i in range(NUM_SPAD)], dtype=np.float64)
SPAD_FIT_ROWS = [
    [0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21, 22],
]
PITCH_X_UM = 23.0
PITCH_Y_UM = 19.92
GRID_DECIMALS = 9
BTN_W = 110
R_GRID = 120  # radial points per half for synthetic space-time map

_TRACE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b6ecf',
]

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TRPLRecord:
    path: Path
    power_w: Optional[float]
    v_front: Optional[float]
    v_back: Optional[float]
    time_ns: np.ndarray
    counts: np.ndarray
    mtime: float


@dataclass
class PLFrame:
    path: Path
    power_w: Optional[float]
    v_front: Optional[float]
    v_back: Optional[float]
    wavelength_nm: Optional[np.ndarray]
    image: np.ndarray
    meta: dict
    mtime: float


@dataclass
class ViewerDefaults:
    crop: Tuple[int, int, int, int]
    linecut_row: Optional[int]
    linecut_width: int


@dataclass
class SpadFitResult:
    popt: np.ndarray
    sigma_eq: float


@dataclass
class AlignResult:
    t0_idx: np.ndarray
    t0_ns: np.ndarray
    smoothed_matrix: np.ndarray
    aligned_counts: np.ndarray
    aligned_smoothed_counts: np.ndarray
    aligned_time_ns: np.ndarray
    n_aligned_bins: int


@dataclass
class GaussFitResult:
    popt: np.ndarray
    sigma_eq: float
    sigma_x: float = 0.0
    sigma_y: float = 0.0

# ── File parsing helpers ───────────────────────────────────────────────────────

_POWER_RE = re.compile(
    r"(?:^|_)P(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)\s*(?P<unit>nW|uW|mW|W)(?=_|\.|$)")
_VF_RE = re.compile(r"(?:^|_)Vf(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)(?=_|\.|$)")
_VB_RE = re.compile(r"(?:^|_)Vb(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)(?=_|\.|$)")


def _parse_token(tok: str) -> Optional[float]:
    if not tok:
        return None
    sign = 1.0
    if tok.startswith("-"):
        sign, tok = -1.0, tok[1:]
    elif tok.startswith("+"):
        tok = tok[1:]
    if tok.startswith("m"):
        sign *= -1.0
        tok = tok[1:]
    try:
        return sign * float(tok.replace("p", "."))
    except Exception:
        return None


def parse_from_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    power_w = v_front = v_back = None
    pm = _POWER_RE.search(name)
    if pm:
        val = _parse_token(pm.group("val"))
        scale = {"w": 1.0, "mw": 1e-3, "uw": 1e-6, "nw": 1e-9}.get(pm.group("unit").lower())
        if val is not None and scale is not None:
            power_w = val * scale
    vm = _VF_RE.search(name)
    if vm:
        v_front = _parse_token(vm.group("val"))
    vm = _VB_RE.search(name)
    if vm:
        v_back = _parse_token(vm.group("val"))
    return power_w, v_front, v_back


def normalize_key(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return round(x, GRID_DECIMALS) if np.isfinite(x) else None


def parse_optional_float(meta: dict, key: str) -> Optional[float]:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def parse_int(meta: dict, key: str, default: int) -> int:
    val = meta.get(key)
    if val is None:
        return int(default)
    try:
        return int(float(val))
    except Exception:
        return int(default)


def parse_optional_int(meta: dict, key: str) -> Optional[int]:
    val = meta.get(key)
    if val is None:
        return None
    try:
        return int(float(val))
    except Exception:
        return None

# ── Data loading ───────────────────────────────────────────────────────────────

def load_trpl_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        rows = [row for row in reader if row and not row[0].startswith("#")]
    if not rows:
        raise ValueError("No data rows")
    arr = np.array([[float(v) for v in r] for r in rows], dtype=np.float64)
    return arr[:, 0], arr[:, 1:NUM_SPAD + 1]


def load_trpl_records(data_dir: Path) -> List[TRPLRecord]:
    records: List[TRPLRecord] = []
    for path in sorted(data_dir.glob("TRPL_*.csv")):
        try:
            time_ns, counts = load_trpl_csv(path)
        except Exception:
            continue
        src = path.name[len("TRPL_"):] if path.name.startswith("TRPL_") else path.name
        pw, vf, vb = parse_from_name(src)
        records.append(TRPLRecord(
            path=path, power_w=pw, v_front=vf, v_back=vb,
            time_ns=time_ns, counts=counts, mtime=float(path.stat().st_mtime)))
    return records


def load_andor_ascii(path: Path) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    meta: dict = {}
    data_lines: list = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                stripped = line[1:].strip()
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
            else:
                data_lines.append(line)
    arr = np.loadtxt(data_lines) if data_lines else np.zeros((1, 1))
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    cols = str(meta.get("columns", "")).lower()
    if arr.shape[1] > 1 and "wavelength" in cols:
        return meta, arr[:, 0], arr[:, 1:].T
    return meta, None, arr


def load_frames(data_dir: Path) -> Tuple[List[PLFrame], ViewerDefaults]:
    frames: List[PLFrame] = []
    first_meta: Optional[dict] = None
    for path in sorted(data_dir.glob("*.asc")):
        try:
            meta, wl, image = load_andor_ascii(path)
        except Exception:
            continue
        p_m = parse_optional_float(meta, "power_w")
        vf_m = parse_optional_float(meta, "front_v_set") or parse_optional_float(meta, "front_vset")
        vb_m = parse_optional_float(meta, "back_v_set") or parse_optional_float(meta, "back_vset")
        p_n, vf_n, vb_n = parse_from_name(path.name)
        frames.append(PLFrame(
            path=path,
            power_w=p_n if p_n is not None else p_m,
            v_front=vf_n if vf_n is not None else vf_m,
            v_back=vb_n if vb_n is not None else vb_m,
            wavelength_nm=wl, image=image, meta=meta,
            mtime=float(path.stat().st_mtime)))
        if first_meta is None:
            first_meta = meta
    if first_meta is None:
        defaults = ViewerDefaults(crop=DEFAULT_CROP, linecut_row=None, linecut_width=1)
    else:
        defaults = ViewerDefaults(
            crop=DEFAULT_CROP,
            linecut_row=parse_optional_int(first_meta, "linecut_row"),
            linecut_width=parse_int(first_meta, "linecut_width", 1))
    return frames, defaults

# ── TRPL alignment ─────────────────────────────────────────────────────────────

def compute_alignment(time_ns: np.ndarray, counts: np.ndarray,
                      smooth_window: int, noise_level: float,
                      align_t0: bool) -> AlignResult:
    N, P = counts.shape
    w = max(1, smooth_window)
    kernel = np.ones(w, dtype=np.float64) / w
    smoothed = np.apply_along_axis(
        lambda c: np.convolve(c.astype(np.float64), kernel, mode='same'), 0, counts)
    t0_idx = np.argmax(smoothed, axis=0).astype(int)
    if not align_t0:
        t0_idx[:] = int(t0_idx[11])
    elif noise_level > 0:
        peak_smooth = np.max(smoothed, axis=0)
        low_signal = peak_smooth <= noise_level
        low_signal[11] = False
        t0_idx[low_signal] = int(t0_idx[11])
    t0_ns = time_ns[t0_idx]
    t0_sorted = np.sort(t0_idx)
    ref_t0 = int(t0_sorted[min(5, P - 1)])
    n_aligned_bins = max(1, N - ref_t0)
    dt = float(time_ns[1] - time_ns[0]) if N > 1 else 1.0
    aligned_time_ns = np.arange(n_aligned_bins, dtype=np.float64) * dt
    k_idx = np.arange(n_aligned_bins)[:, np.newaxis]
    raw_idx = np.clip(t0_idx[np.newaxis, :] + k_idx, 0, N - 1)
    p_idx = np.arange(P)[np.newaxis, :]
    return AlignResult(
        t0_idx=t0_idx, t0_ns=t0_ns, smoothed_matrix=smoothed,
        aligned_counts=counts[raw_idx, p_idx].astype(np.float64),
        aligned_smoothed_counts=smoothed[raw_idx, p_idx],
        aligned_time_ns=aligned_time_ns, n_aligned_bins=n_aligned_bins)

# ── SPAD / Gaussian fitting helpers ───────────────────────────────────────────

def build_spad_fit_coords() -> Tuple[np.ndarray, np.ndarray]:
    max_cols = max(len(r) for r in SPAD_FIT_ROWS)
    coords: Dict[int, Tuple[float, float]] = {}
    for row_idx, row in enumerate(SPAD_FIT_ROWS):
        x0 = (PITCH_X_UM / 2.0) if len(row) < max_cols else 0.0
        for col_idx, pixel in enumerate(row):
            coords[pixel] = (x0 + col_idx * PITCH_X_UM, row_idx * PITCH_Y_UM)
    x = np.array([coords[i][0] for i in range(NUM_SPAD)], dtype=np.float64)
    y = np.array([coords[i][1] for i in range(NUM_SPAD)], dtype=np.float64)
    return x, y

_FIT_X_UM, _FIT_Y_UM = build_spad_fit_coords()


def fit_spad_gaussian_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[SpadFitResult]:
    if _curve_fit is None:
        return None
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size < 6 or np.allclose(z, z[0]):
        return None

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0) ** 2) / (2.0 * sx**2) +
                            ((yg - y0) ** 2) / (2.0 * sy**2))) + offset

    p0 = [max(float(np.max(z) - np.min(z)), 1e-6),
          float(np.mean(x)), float(np.mean(y)), 20.0, 20.0, float(np.min(z))]
    bounds = (
        [0.0, float(np.min(x) - PITCH_X_UM), float(np.min(y) - PITCH_Y_UM), 1e-3, 1e-3, -np.inf],
        [np.inf, float(np.max(x) + PITCH_X_UM), float(np.max(y) + PITCH_Y_UM), np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = _curve_fit(gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return SpadFitResult(popt=np.asarray(popt, dtype=np.float64), sigma_eq=sigma_eq)


def fit_gaussian_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[GaussFitResult]:
    if _curve_fit is None or z.size < 6 or np.allclose(z, z[0]):
        return None

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0) ** 2) / (2 * sx ** 2) +
                            ((yg - y0) ** 2) / (2 * sy ** 2))) + offset

    # Weighted-centroid initial guess (same as spad23 approach)
    z_pos = np.clip(np.asarray(z, dtype=float) - float(np.nanmin(z)), 0.0, None)
    z_sum = float(np.nansum(z_pos))
    if z_sum > 0:
        x0_est = float(np.nansum(x * z_pos) / z_sum)
        y0_est = float(np.nansum(y * z_pos) / z_sum)
        sigma_est = float(max(
            np.sqrt(np.nansum(z_pos * ((x - x0_est)**2 + (y - y0_est)**2)) / z_sum / 2),
            5.0))
    else:
        x0_est = float(np.mean(x))
        y0_est = float(np.mean(y))
        sigma_est = 20.0
    p0 = [max(float(np.max(z) - np.min(z)), 1e-6),
          x0_est, y0_est, sigma_est, sigma_est, float(np.min(z))]
    bounds = (
        [0, float(np.min(x) - PITCH_X_UM), float(np.min(y) - PITCH_Y_UM), 1e-3, 1e-3, -np.inf],
        [np.inf, float(np.max(x) + PITCH_X_UM), float(np.max(y) + PITCH_Y_UM), np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = _curve_fit(gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2))
    return GaussFitResult(popt=np.asarray(popt), sigma_eq=sigma_eq,
                          sigma_x=float(abs(popt[3])), sigma_y=float(abs(popt[4])))


def _fit_gaussian_2d_constrained(
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        prev_popt: np.ndarray, tightness: float = 1.0) -> Optional[GaussFitResult]:
    """Fit with bounds constrained around the previous frame's fit (from spad23 reference)."""
    if _curve_fit is None or np.allclose(z, z[0]):
        return None
    A_p, x0_p, y0_p, sx_p, sy_p, off_p = (float(v) for v in prev_popt)
    sx_p = max(sx_p, 1e-3);  sy_p = max(sy_p, 1e-3)
    dA  = max(abs(A_p)  * 0.18 * tightness, 1e-3)
    dx  = max(PITCH_X_UM * 0.18 * tightness, 0.8)
    dy  = max(PITCH_Y_UM * 0.18 * tightness, 0.8)
    dsx = max(abs(sx_p) * 0.18 * tightness, 0.5)
    dsy = max(abs(sy_p) * 0.18 * tightness, 0.5)
    doff = max(abs(off_p) * 0.25 * tightness, 0.5)

    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        xg, yg = coords
        return A * np.exp(-(((xg - x0)**2)/(2*sx**2) + ((yg - y0)**2)/(2*sy**2))) + offset

    try:
        popt, _ = _curve_fit(
            gauss2d, (x, y), z,
            p0=[A_p, x0_p, y0_p, sx_p, sy_p, off_p],
            bounds=([max(0.0, A_p-dA), x0_p-dx, y0_p-dy,
                     max(1e-3, sx_p-dsx), max(1e-3, sy_p-dsy), off_p-doff],
                    [A_p+dA, x0_p+dx, y0_p+dy, sx_p+dsx, sy_p+dsy, off_p+doff]),
            maxfev=20000)
    except Exception:
        return None
    return GaussFitResult(popt=np.asarray(popt),
                          sigma_eq=float(np.sqrt((popt[3]**2 + popt[4]**2) / 2)),
                          sigma_x=float(abs(popt[3])), sigma_y=float(abs(popt[4])))


# ── PL spectral helpers ────────────────────────────────────────────────────────

def apply_crop(image: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim != 2:
        return a
    h, w = a.shape
    top, bottom, left, right = crop
    top = max(0, min(int(top), h - 1))
    bottom = max(0, min(int(bottom), h - 1))
    left = max(0, min(int(left), w - 1))
    right = max(0, min(int(right), w - 1))
    y2 = max(top + 1, h - bottom)
    x2 = max(left + 1, w - right)
    return a[top:y2, left:x2]


def crop_axis(axis: Optional[np.ndarray], crop: Tuple[int, int, int, int], width: int) -> Optional[np.ndarray]:
    if axis is None:
        return None
    arr = np.asarray(axis, dtype=float).ravel()
    if arr.size != width:
        return arr
    left, right = crop[2], crop[3]
    if right > 0:
        arr = arr[: max(0, arr.size - right)]
    if left > 0:
        arr = arr[min(left, arr.size):]
    return arr


def linecut_horizontal(image: np.ndarray, row: int, width: int) -> Optional[np.ndarray]:
    a = np.asarray(image)
    if a.ndim != 2:
        return None
    h, _ = a.shape
    width = max(1, int(width))
    half = width // 2
    r1 = max(0, int(row) - half)
    r2 = min(h, int(row) + half + (1 if width % 2 else 0))
    if r2 <= r1:
        return None
    return a[r1:r2, :].sum(axis=0)


def resample_linecut(wl_src: Optional[np.ndarray], linecut: np.ndarray,
                     wl_dst: Optional[np.ndarray]) -> np.ndarray:
    if wl_src is None or wl_dst is None:
        return linecut
    src = np.asarray(wl_src, dtype=float).ravel()
    dst = np.asarray(wl_dst, dtype=float).ravel()
    if src.size == dst.size and np.allclose(src, dst):
        return linecut
    order = np.argsort(src)
    return np.interp(dst, src[order], np.asarray(linecut, dtype=float)[order],
                     left=np.nan, right=np.nan)


def derivative_vs_x_nan(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(data, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    n = x.size
    if y.ndim != 2 or n != y.shape[1] or n < 2:
        return out
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        prev = i - 1
        while prev >= 0 and (not np.isfinite(x[prev]) or x[prev] == x[i]):
            prev -= 1
        nxt = i + 1
        while nxt < n and (not np.isfinite(x[nxt]) or x[nxt] == x[i]):
            nxt += 1
        if prev >= 0 and nxt < n:
            denom = x[nxt] - x[prev]
            if denom != 0:
                out[:, i] = (y[:, nxt] - y[:, prev]) / denom
        elif prev >= 0:
            denom = x[i] - x[prev]
            if denom != 0:
                out[:, i] = (y[:, i] - y[:, prev]) / denom
        elif nxt < n:
            denom = x[nxt] - x[i]
            if denom != 0:
                out[:, i] = (y[:, nxt] - y[:, i]) / denom
    return out


def derivative_dlogp_nan(data: np.ndarray, powers_w: np.ndarray) -> np.ndarray:
    p = np.asarray(powers_w, dtype=float).ravel()
    logp = np.full_like(p, np.nan, dtype=float)
    mask = p > 0
    logp[mask] = np.log10(p[mask])
    return derivative_vs_x_nan(data, logp)


def lorentzian_component(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        return np.zeros_like(x, dtype=float)
    g = float(gamma)
    dx = x - float(center)
    return float(amplitude) * (g * g) / (dx * dx + g * g)


def estimate_baseline_from_tail(wavelength_nm: np.ndarray, intensity: np.ndarray,
                                 *, low_nm: float = 940.0, high_nm: float = 960.0) -> float:
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y) & (x >= float(low_nm)) & (x <= float(high_nm))
    if np.count_nonzero(mask) >= 3:
        return float(np.nanmean(y[mask]))
    finite = y[np.isfinite(y)]
    return float(np.nanpercentile(finite, 10.0)) if finite.size > 0 else 0.0


def _estimate_noise_floor(y: np.ndarray) -> float:
    vals = np.asarray(y, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return 0.0
    diffs = np.diff(vals)
    mad = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad / np.sqrt(2.0))
    std = np.nanstd(vals)
    return float(std) if np.isfinite(std) and std > 0 else 0.0


def fit_lorentzian_peak_local(wavelength_nm, intensity, center_guess_nm, *,
                               window_nm, min_points=7, min_snr=2.0,
                               min_gamma_nm=0.08, max_gamma_nm=10.0):
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size < min_points:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x - float(center_guess_nm)) <= float(window_nm))
    if np.count_nonzero(mask) < min_points:
        return None
    xs, ys = x[mask], y[mask]
    baseline = float(np.nanpercentile(ys, 10.0))
    yb = ys - baseline
    amp = float(np.nanmax(yb))
    if not np.isfinite(amp) or amp <= 0:
        return None
    noise = _estimate_noise_floor(ys)
    if noise > 0 and amp < float(min_snr) * noise:
        return None
    peak_idx = int(np.nanargmax(yb))
    center_fit = float(xs[peak_idx])
    y_half = 0.5 * amp
    left = np.where(yb[:peak_idx] <= y_half)[0]
    right = np.where(yb[peak_idx + 1:] <= y_half)[0]
    if left.size > 0 and right.size > 0:
        fwhm_guess = max(2.0 * float(min_gamma_nm), float(xs[peak_idx + 1 + right[0]]) - float(xs[left[-1]]))
    else:
        fwhm_guess = max(2.0 * float(min_gamma_nm), 0.25 * float(window_nm))
    gamma_fit = float(np.clip(0.5 * fwhm_guess, float(min_gamma_nm), float(max_gamma_nm)))
    amp_fit = amp
    if _curve_fit is not None:
        try:
            p0 = np.array([amp_fit, center_fit, gamma_fit], dtype=float)
            blo = np.array([0.0, float(center_guess_nm) - float(window_nm), float(min_gamma_nm)], dtype=float)
            bhi = np.array([max(1.0, 4.0 * amp_fit), float(center_guess_nm) + float(window_nm), float(max_gamma_nm)], dtype=float)
            p0 = np.clip(p0, blo + 1e-9, bhi - 1e-9)
            popt, _ = _curve_fit(lambda xx, a, m, g: lorentzian_component(xx, a, m, g),
                                 xs, yb, p0=p0, bounds=(blo, bhi), maxfev=8000)
            amp_fit, center_fit, gamma_fit = [float(v) for v in popt]
        except Exception:
            pass
    if gamma_fit < float(min_gamma_nm) or gamma_fit > float(max_gamma_nm):
        return None
    if abs(center_fit - float(center_guess_nm)) > float(window_nm) * 1.2:
        return None
    model = baseline + lorentzian_component(xs, amp_fit, center_fit, gamma_fit)
    resid = ys - model
    sst = float(np.nansum((ys - np.nanmean(ys)) ** 2))
    r2 = 1.0 - float(np.nansum(resid * resid)) / sst if sst > 0 else 1.0
    if not np.isfinite(r2) or r2 < -0.3:
        return None
    return {"center_nm": float(center_fit), "gamma_nm": float(gamma_fit),
            "fwhm_nm": float(2.0 * gamma_fit), "amplitude": float(max(0.0, amp_fit))}


def _estimate_peak_guess(x, y, low_nm, high_nm, fallback_nm):
    mask = np.isfinite(x) & np.isfinite(y) & (x >= float(low_nm)) & (x <= float(high_nm))
    if np.count_nonzero(mask) < 3:
        return 0.0, float(fallback_nm), 3.0
    xs, ys = x[mask], y[mask]
    idx = int(np.nanargmax(ys))
    center = float(xs[idx])
    baseline = float(np.nanpercentile(ys, 15.0))
    amp = float(max(0.0, ys[idx] - baseline))
    w = np.clip(ys - baseline, 0.0, None)
    wsum = float(np.nansum(w))
    sigma = float(np.sqrt(np.nansum(w * (xs - center) ** 2) / wsum)) if wsum > 0 else 3.0
    return amp, center, float(np.clip(0.5 * sigma, 0.5, 10.0))


def _one_lorentzian_model(x, a1, m1, g1):
    return lorentzian_component(x, a1, m1, g1)


def _two_lorentzian_model(x, a1, m1, g1, a2, m2, g2):
    return lorentzian_component(x, a1, m1, g1) + lorentzian_component(x, a2, m2, g2)


def fit_two_peak_linecut(wavelength_nm, intensity, *, peak1_guess_nm, peak2_guess_nm,
                          peak1_window_nm=10.0, peak2_window_nm=5.0) -> dict:
    x_all = np.asarray(wavelength_nm, dtype=float).ravel()
    y_all = np.asarray(intensity, dtype=float).ravel()
    if x_all.size != y_all.size or x_all.size < 8:
        return {"peak_1": None, "peak_2": None, "offset": 0.0}
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    if np.count_nonzero(valid) < 8:
        return {"peak_1": None, "peak_2": None, "offset": 0.0}
    x_all, y_all = x_all[valid], y_all[valid]
    order = np.argsort(x_all)
    x_all, y_all = x_all[order], y_all[order]
    baseline_value = estimate_baseline_from_tail(x_all, y_all)
    y_all_sub = y_all - baseline_value
    w1 = max(0.2, float(peak1_window_nm))
    w2 = max(0.2, float(peak2_window_nm))
    p1_lo, p1_hi = float(peak1_guess_nm) - w1, float(peak1_guess_nm) + w1
    p2_lo, p2_hi = float(peak2_guess_nm) - w2, float(peak2_guess_nm) + w2
    fit_lo = min(p2_lo, p1_lo) - 6.0
    fit_hi = max(p2_hi, p1_hi) + 30.0
    fit_mask = (x_all >= fit_lo) & (x_all <= fit_hi)
    x = x_all[fit_mask] if np.count_nonzero(fit_mask) >= 20 else x_all
    y = y_all_sub[fit_mask] if np.count_nonzero(fit_mask) >= 20 else y_all_sub
    if x.size < 8:
        return {"peak_1": None, "peak_2": None, "offset": float(baseline_value)}
    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    y_span = float(max(1.0, y_max - y_min))
    a1_g, m1_g, g1_g = _estimate_peak_guess(x, y, p1_lo, p1_hi, float(peak1_guess_nm))
    a2_g, m2_g, g2_g = _estimate_peak_guess(x, y, p2_lo, p2_hi, float(peak2_guess_nm))
    a1_g = float(max(0.08 * y_span, a1_g))
    a2_g = float(max(0.02 * y_span, a2_g))
    p0_two = np.clip(np.array([a1_g, m1_g, g1_g, a2_g, m2_g, g2_g], dtype=float),
                     np.array([0.0, p1_lo, 0.4, 0.0, p2_lo, 0.4]) + 1e-6,
                     np.array([4.0*y_span, p1_hi, 15.0, 4.0*y_span, p2_hi, 12.0]) - 1e-6)
    p0_one = np.clip(np.array([a1_g, m1_g, g1_g], dtype=float),
                     np.array([0.0, p1_lo, 0.4]) + 1e-6,
                     np.array([4.0*y_span, p1_hi, 15.0]) - 1e-6)
    low_two = np.array([0.0, p1_lo, 0.4, 0.0, p2_lo, 0.4])
    high_two = np.array([4.0*y_span, p1_hi, 15.0, 4.0*y_span, p2_hi, 12.0])
    low_one = np.array([0.0, p1_lo, 0.4])
    high_one = np.array([4.0*y_span, p1_hi, 15.0])
    popt_two = popt_one = None
    if _curve_fit is not None:
        try:
            popt_two, _ = _curve_fit(
                lambda xx, a1, m1, g1, a2, m2, g2: _two_lorentzian_model(xx, a1, m1, g1, a2, m2, g2),
                x, y, p0=p0_two, bounds=(low_two, high_two), maxfev=20000)
        except Exception:
            popt_two = None
        try:
            popt_one, _ = _curve_fit(
                lambda xx, a1, m1, g1: _one_lorentzian_model(xx, a1, m1, g1),
                x, y, p0=p0_one, bounds=(low_one, high_one), maxfev=12000)
        except Exception:
            popt_one = None
    if popt_two is None and popt_one is None:
        p1 = fit_lorentzian_peak_local(x_all, y_all_sub, float(peak1_guess_nm), window_nm=10.0, min_snr=1.2)
        residual = np.asarray(y_all_sub, dtype=float).copy()
        if p1 is not None:
            residual -= lorentzian_component(x_all, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
        p2 = fit_lorentzian_peak_local(x_all, residual, float(peak2_guess_nm), window_nm=5.0, min_snr=1.5)
        if p1 is not None and not (p1_lo <= p1["center_nm"] <= p1_hi):
            p1 = None
        if p2 is not None and not (p2_lo <= p2["center_nm"] <= p2_hi):
            p2 = None
        return {"peak_1": p1, "peak_2": p2, "offset": float(baseline_value)}
    use_second = False
    if popt_two is not None:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt_two]
        y_fit_one = (_one_lorentzian_model(x, *[float(v) for v in popt_one])
                     if popt_one is not None else _one_lorentzian_model(x, a1, m1, g1))
        y_fit_two = _two_lorentzian_model(x, a1, m1, g1, a2, m2, g2)
        noise = _estimate_noise_floor(y - y_fit_two)
        rss_one = float(np.nansum((y - y_fit_one) ** 2))
        rss_two = float(np.nansum((y - y_fit_two) ** 2))
        improvement = (rss_one - rss_two) / rss_one if rss_one > 0 else 0.0
        min_amp2 = float(max(2.0 * noise, 0.04 * y_span, 0.06 * max(a1, 1.0)))
        use_second = bool(a2 >= min_amp2 and m2 < m1 - 1.0 and p2_lo <= m2 <= p2_hi
                          and 0.4 <= g2 <= 12.0 and improvement > 0.03)
    if use_second:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt_two]
    elif popt_one is not None:
        a1, m1, g1 = [float(v) for v in popt_one]
        a2, m2, g2 = 0.0, float(peak2_guess_nm), 3.0
    else:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt_two]
        a2 = 0.0
    p1 = ({"center_nm": float(m1), "gamma_nm": float(g1), "fwhm_nm": float(2.0*g1), "amplitude": float(a1)}
          if a1 > 0 and 0.4 <= g1 <= 20.0 and p1_lo <= m1 <= p1_hi else None)
    p2 = ({"center_nm": float(m2), "gamma_nm": float(g2), "fwhm_nm": float(2.0*g2), "amplitude": float(a2)}
          if use_second and a2 > 0 and 0.4 <= g2 <= 20.0 and p2_lo <= m2 <= p2_hi else None)
    return {"peak_1": p1, "peak_2": p2, "offset": float(baseline_value)}


def fit_two_peak_map_sequential(wavelength_nm, map_data, *, ref_idx, peak1_guess_nm,
                                 peak2_guess_nm, max_shift_nm=2.0) -> dict:
    wl = np.asarray(wavelength_nm, dtype=float).ravel()
    data = np.asarray(map_data, dtype=float)
    n_x = data.shape[1] if data.ndim == 2 else 0
    center_1 = np.full(n_x, np.nan)
    fwhm_1 = np.full(n_x, np.nan)
    gamma_1 = np.full(n_x, np.nan)
    amp_1 = np.full(n_x, np.nan)
    center_2 = np.full(n_x, np.nan)
    fwhm_2 = np.full(n_x, np.nan)
    gamma_2 = np.full(n_x, np.nan)
    amp_2 = np.full(n_x, np.nan)
    offset = np.full(n_x, np.nan)

    def _result():
        return {"center_1_nm": center_1, "fwhm_1_nm": fwhm_1, "gamma_1_nm": gamma_1,
                "amp_1": amp_1, "center_2_nm": center_2, "fwhm_2_nm": fwhm_2,
                "gamma_2_nm": gamma_2, "amp_2": amp_2, "offset": offset}

    if data.ndim != 2 or wl.size != data.shape[0] or n_x == 0:
        return _result()
    ref = max(0, min(int(ref_idx), n_x - 1))
    max_step = max(0.2, float(max_shift_nm))

    def _store(j, fit):
        p1 = fit.get("peak_1")
        p2 = fit.get("peak_2")
        offset[j] = float(fit.get("offset", np.nan))
        if p1 is not None:
            center_1[j] = float(p1.get("center_nm", np.nan))
            fwhm_1[j] = float(p1.get("fwhm_nm", np.nan))
            gamma_1[j] = float(p1.get("gamma_nm", np.nan))
            amp_1[j] = float(p1.get("amplitude", np.nan))
        if p2 is not None:
            center_2[j] = float(p2.get("center_nm", np.nan))
            fwhm_2[j] = float(p2.get("fwhm_nm", np.nan))
            gamma_2[j] = float(p2.get("gamma_nm", np.nan))
            amp_2[j] = float(p2.get("amplitude", np.nan))

    _store(ref, fit_two_peak_linecut(wl, data[:, ref],
                                      peak1_guess_nm=float(peak1_guess_nm),
                                      peak2_guess_nm=float(peak2_guess_nm)))
    g1_ref = center_1[ref] if np.isfinite(center_1[ref]) else float(peak1_guess_nm)
    g2_ref = center_2[ref] if np.isfinite(center_2[ref]) else float(peak2_guess_nm)

    def _walk(start, stop, step, g1_init, g2_init):
        g1, g2 = float(g1_init), float(g2_init)
        for j in range(start, stop, step):
            _store(j, fit_two_peak_linecut(wl, data[:, j],
                                            peak1_guess_nm=g1, peak2_guess_nm=g2,
                                            peak1_window_nm=max_step, peak2_window_nm=max_step))
            if np.isfinite(center_1[j]):
                g1 = float(center_1[j])
            if np.isfinite(center_2[j]):
                g2 = float(center_2[j])

    _walk(ref + 1, n_x, +1, g1_ref, g2_ref)
    _walk(ref - 1, -1, -1, g1_ref, g2_ref)
    return _result()


def fwhm_nm_to_mev(center_nm: np.ndarray, fwhm_nm: np.ndarray) -> np.ndarray:
    c = np.asarray(center_nm, dtype=float)
    w = np.asarray(fwhm_nm, dtype=float)
    out = np.full(np.broadcast(c, w).shape, np.nan, dtype=float)
    c, w = np.broadcast_arrays(c, w)
    half = 0.5 * w
    valid = np.isfinite(c) & np.isfinite(w) & (w > 0) & (c > half) & (c + half > 0)
    if np.any(valid):
        out[valid] = np.abs(EV_NM / (c[valid] - half[valid]) - EV_NM / (c[valid] + half[valid])) * 1e3
    return out


def spectral_map_nm_to_energy(wavelength_nm: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wl = np.asarray(wavelength_nm, dtype=float).ravel()
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or wl.size != arr.shape[0]:
        return wl, arr
    with np.errstate(divide="ignore", invalid="ignore"):
        energy_ev = EV_NM / wl
    finite = np.isfinite(energy_ev)
    if np.count_nonzero(finite) < 2:
        return wl, arr
    energy_ev = energy_ev[finite]
    arr = arr[finite, :]
    order = np.argsort(energy_ev)
    return energy_ev[order], arr[order, :]

# ── pyqtgraph helpers ──────────────────────────────────────────────────────────

def _fix_axes(pi) -> None:
    for ax in ("bottom", "left"):
        pi.getAxis(ax).setPen(pg.mkPen("k"))
        pi.getAxis(ax).setTextPen(pg.mkPen("k"))


# ── Extra-trace colors (condition overlay, radial extras) ───────────────────────

_EXTRA_COLORS = ['#e85d04', '#7b2d8b', '#0f766e', '#b45309', '#1e40af',
                 '#dc2626', '#059669', '#7c3aed', '#d97706', '#0284c7']

# ── Rolling-median smoothing ────────────────────────────────────────────────────

def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if w <= 1 or n == 0:
        return x.copy()
    half = w // 2
    padded = np.pad(x, half, mode='edge')
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(np.median(padded[i:i + w]))
    return out

# ── EMG fit helpers ─────────────────────────────────────────────────────────────

try:
    from scipy.special import erfcx as _erfcx
except ImportError:
    _erfcx = None


def _emg_k(t: np.ndarray, t0: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially modified Gaussian (numerically stable via erfcx)."""
    sigma = max(float(abs(sigma)), 1e-12)
    tau = max(float(abs(tau)), 1e-12)
    lam = 1.0 / tau
    t = np.asarray(t, dtype=np.float64)
    arg = (lam * sigma ** 2 - (t - t0)) / (np.sqrt(2.0) * sigma)
    gauss = np.exp(np.clip(-((t - t0) ** 2) / (2.0 * sigma ** 2), -500.0, 0.0))
    if _erfcx is not None:
        ec = np.where(np.isfinite(arg), _erfcx(-arg), 0.0)
    else:
        try:
            from scipy.special import erfc as _erfc
            ec = np.exp(np.minimum(arg ** 2, 500.0)) * _erfc(-arg)
        except Exception:
            return np.zeros_like(t)
    return (lam / 2.0) * gauss * np.where(np.isfinite(ec), ec, 0.0)


def fit_emg_histogram(t: np.ndarray, y: np.ndarray, *,
                       sigma_ns: float = 0.010,
                       use_rise: bool = False,
                       use_2nd: bool = False,
                       use_3rd: bool = False) -> Optional[dict]:
    """Fit EMG model to a single-pixel TRPL histogram.  σ fixed at sigma_ns."""
    if _curve_fit is None:
        return None
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 10:
        return None
    t, y = t[mask], y[mask]
    peak_idx = int(np.argmax(y))
    bg_est = float(np.nanpercentile(y, 5))
    A_peak = max(float(y[peak_idx]) - bg_est, 1.0)
    half_val = 0.5 * A_peak + bg_est
    right_idx = np.where(t > t[peak_idx])[0]
    tau1_est = max(float(t[-1] - t[peak_idx]) * 0.3, 0.05)
    if right_idx.size > 0:
        below = right_idx[y[right_idx] <= half_val]
        if below.size > 0:
            tau1_est = max(float(t[below[0]]) - float(t[peak_idx]), 0.05)
    # A_est is the area (amplitude × lifetime) not peak amplitude, since _emg_k is a
    # normalised PDF whose peak height is ~1/τ. Multiply by τ so p0 is near the optimum.
    A_est = A_peak * tau1_est
    s = float(sigma_ns)

    if not use_rise and not use_2nd and not use_3rd:
        def model(tt, A1, tau1, bg):
            return A1 * _emg_k(tt, 0.0, s, tau1) + bg
        p0 = [A_est, tau1_est, bg_est]
        lo, hi = [0, 1e-4, -np.inf], [np.inf, 500.0, np.inf]
        names = ['A1', 'τ₁ (ns)', 'bg']
    elif use_rise and not use_2nd and not use_3rd:
        def model(tt, A1, tau1, tau_r, bg):
            return A1 * (_emg_k(tt, 0.0, s, tau1) - _emg_k(tt, 0.0, s, tau_r)) + bg
        p0 = [A_est, tau1_est, max(tau1_est * 0.1, 0.01), bg_est]
        lo, hi = [0, 1e-4, 1e-4, -np.inf], [np.inf, 500.0, 500.0, np.inf]
        names = ['A1', 'τ₁ (ns)', 'τ_r (ns)', 'bg']
    elif not use_rise and use_2nd and not use_3rd:
        def model(tt, A1, tau1, A2, tau2, bg):
            return A1 * _emg_k(tt, 0.0, s, tau1) + A2 * _emg_k(tt, 0.0, s, tau2) + bg
        p0 = [A_est * 0.7, tau1_est, A_est * 0.3, tau1_est * 3.0, bg_est]
        lo, hi = [0, 1e-4, 0, 1e-4, -np.inf], [np.inf, 500.0, np.inf, 500.0, np.inf]
        names = ['A1', 'τ₁ (ns)', 'A2', 'τ₂ (ns)', 'bg']
    elif not use_rise and use_2nd and use_3rd:
        def model(tt, A1, tau1, A2, tau2, A3, tau3, bg):
            return (A1 * _emg_k(tt, 0.0, s, tau1) + A2 * _emg_k(tt, 0.0, s, tau2)
                    + A3 * _emg_k(tt, 0.0, s, tau3) + bg)
        p0 = [A_est*0.6, tau1_est, A_est*0.3, tau1_est*3, A_est*0.1, tau1_est*10, bg_est]
        lo = [0,1e-4,0,1e-4,0,1e-4,-np.inf]; hi = [np.inf,500,np.inf,500,np.inf,500,np.inf]
        names = ['A1','τ₁ (ns)','A2','τ₂ (ns)','A3','τ₃ (ns)','bg']
    elif use_rise and use_2nd and not use_3rd:
        def model(tt, A1, tau1, tau_r, A2, tau2, bg):
            return (A1 * (_emg_k(tt,0.0,s,tau1) - _emg_k(tt,0.0,s,tau_r))
                    + A2 * _emg_k(tt, 0.0, s, tau2) + bg)
        p0 = [A_est*0.7, tau1_est, max(tau1_est*0.1,0.01), A_est*0.3, tau1_est*3, bg_est]
        lo = [0,1e-4,1e-4,0,1e-4,-np.inf]; hi = [np.inf,500,500,np.inf,500,np.inf]
        names = ['A1','τ₁ (ns)','τ_r (ns)','A2','τ₂ (ns)','bg']
    elif use_rise and use_2nd and use_3rd:
        def model(tt, A1, tau1, tau_r, A2, tau2, A3, tau3, bg):
            return (A1 * (_emg_k(tt,0.0,s,tau1) - _emg_k(tt,0.0,s,tau_r))
                    + A2*_emg_k(tt,0.0,s,tau2) + A3*_emg_k(tt,0.0,s,tau3) + bg)
        p0 = [A_est*0.6,tau1_est,max(tau1_est*0.1,0.01),A_est*0.3,tau1_est*3,A_est*0.1,tau1_est*10,bg_est]
        lo = [0,1e-4,1e-4,0,1e-4,0,1e-4,-np.inf]; hi = [np.inf,500,500,np.inf,500,np.inf,500,np.inf]
        names = ['A1','τ₁ (ns)','τ_r (ns)','A2','τ₂ (ns)','A3','τ₃ (ns)','bg']
    else:  # rise + no 2nd + 3rd
        def model(tt, A1, tau1, tau_r, A3, tau3, bg):
            return (A1 * (_emg_k(tt,0.0,s,tau1) - _emg_k(tt,0.0,s,tau_r))
                    + A3 * _emg_k(tt, 0.0, s, tau3) + bg)
        p0 = [A_est*0.7,tau1_est,max(tau1_est*0.1,0.01),A_est*0.3,tau1_est*10,bg_est]
        lo = [0,1e-4,1e-4,0,1e-4,-np.inf]; hi = [np.inf,500,500,np.inf,500,np.inf]
        names = ['A1','τ₁ (ns)','τ_r (ns)','A3','τ₃ (ns)','bg']

    try:
        lo_a = np.array(lo, dtype=float)
        hi_a = np.array(hi, dtype=float)
        p0_a = np.clip(np.array(p0, dtype=float), lo_a + 1e-9, hi_a - 1e-9)
        popt, _ = _curve_fit(model, t, y, p0=p0_a, bounds=(lo_a, hi_a), maxfev=50000)
    except Exception:
        return None
    return {'popt': np.asarray(popt), 'names': names, 'model': model, 't': t, 'y': y}

# ── SPAD Map Widget ────────────────────────────────────────────────────────────

class SpadMapWidget(QtWidgets.QWidget):
    pixel_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._pw = pg.PlotWidget()
        pi = self._pw.plotItem
        _fix_axes(pi)
        pi.hideAxis("bottom")
        pi.hideAxis("left")
        pi.setTitle("Count Map @ Selected Time Bin")
        pi.setAspectLocked(True)
        pi.vb.setRange(xRange=(-0.7, 4.7), yRange=(-0.7, 4.7), padding=0)
        layout.addWidget(self._pw, 1)

        self._cmap = pg.colormap.get("inferno")
        self._scatter = pg.ScatterPlotItem(
            x=SPAD_MAP_COORDS[:, 0], y=SPAD_MAP_COORDS[:, 1],
            size=30, pxMode=True,
            brush=[pg.mkBrush(self._cmap.map(0.0, mode='qcolor'))] * NUM_SPAD,
            pen=[pg.mkPen('#5f6368', width=2)] * NUM_SPAD,
            data=list(range(NUM_SPAD)))
        self._pw.addItem(self._scatter)

        self._sel = pg.ScatterPlotItem(
            symbol='o', size=35, pxMode=True,
            pen=pg.mkPen('#22d3ee', width=2.2), brush=pg.mkBrush(None))
        self._pw.addItem(self._sel)

        for i, (x, y) in SPAD_PIXEL_COORDS.items():
            t = pg.TextItem(str(i), anchor=(0.5, 0.5), color='#f7f7f7')
            f = t.textItem.font()
            f.setBold(True)
            f.setPointSize(8)
            t.textItem.setFont(f)
            t.setPos(x, y)
            self._pw.addItem(t)

        self._lbl = QtWidgets.QLabel("t = -- | Selected: --")
        layout.addWidget(self._lbl)
        self._pw.scene().sigMouseClicked.connect(self._on_click)

    def update(self, frame: np.ndarray, t_label: str, selected: Set[int]) -> None:
        vmin, vmax = float(np.min(frame)), float(np.max(frame))
        norm = ((frame - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(frame))
        colors = self._cmap.map(norm.astype(np.float32), mode='byte')
        self._scatter.setData(
            x=SPAD_MAP_COORDS[:, 0], y=SPAD_MAP_COORDS[:, 1],
            brush=[pg.mkBrush(*colors[i]) for i in range(NUM_SPAD)],
            pen=[pg.mkPen('#5f6368', width=2)] * NUM_SPAD,
            size=30, pxMode=True, data=list(range(NUM_SPAD)))
        if selected:
            pts = np.array([SPAD_PIXEL_COORDS[p] for p in sorted(selected)], dtype=float)
            self._sel.setData(x=pts[:, 0], y=pts[:, 1])
        else:
            self._sel.setData(x=[], y=[])
        sel_s = ", ".join(str(p) for p in sorted(selected)) if selected else "none"
        self._lbl.setText(f"{t_label} | Selected: {sel_s}")

    def _on_click(self, event) -> None:
        vb = self._pw.plotItem.vb
        pos = vb.mapSceneToView(event.scenePos())
        if not vb.viewRect().contains(pos):
            return
        d2 = (SPAD_MAP_COORDS[:, 0] - pos.x()) ** 2 + (SPAD_MAP_COORDS[:, 1] - pos.y()) ** 2
        nearest = int(np.argmin(d2))
        if float(np.sqrt(d2[nearest])) < 0.42:
            self.pixel_clicked.emit(nearest)

# ── TRPL Histogram Widget ──────────────────────────────────────────────────────

class TrplHistogramWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._pw = pg.PlotWidget()
        self._pi = self._pw.plotItem
        _fix_axes(self._pi)
        self._pi.setTitle("TRPL Histograms — aligned to t=0 per pixel")
        self._pi.setLabel("bottom", "Time relative to t₀ (ns)")
        self._pi.setLabel("left", "Counts per bin")
        self._pi.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._pi.addLegend()
        layout.addWidget(self._pw, 1)

        # EMG fit controls
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(6)
        self.chkFitEMG = QtWidgets.QCheckBox("Fit EMG")
        self.chkRise = QtWidgets.QCheckBox("Rise τ")
        self.chkRise.setEnabled(False)
        self.chk2nd = QtWidgets.QCheckBox("2nd τ")
        self.chk2nd.setEnabled(False)
        self.chk3rd = QtWidgets.QCheckBox("3rd τ")
        self.chk3rd.setEnabled(False)
        self._emg_label = QtWidgets.QLabel("")
        self._emg_label.setStyleSheet("font-size: 11px; color: #1d4ed8;")
        ctrl.addWidget(self.chkFitEMG)
        ctrl.addWidget(self.chkRise)
        ctrl.addWidget(self.chk2nd)
        ctrl.addWidget(self.chk3rd)
        ctrl.addWidget(self._emg_label, 1)
        layout.addLayout(ctrl)

        self._time_line = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#111827', style=QtCore.Qt.DashLine, width=1))
        self._zero_line = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#64748b', style=QtCore.Qt.DotLine, width=1))
        self._zero_line.setValue(0.0)
        self._pw.addItem(self._zero_line)
        self._pw.addItem(self._time_line)
        self._emg_fit_curve = self._pw.plot(pen=pg.mkPen('#dc2626', width=2.0))

        self._normalize = False
        self._raw: Dict[int, pg.PlotDataItem] = {}
        self._smooth: Dict[int, pg.PlotDataItem] = {}
        # stored for re-running fit
        self._last_time: Optional[np.ndarray] = None
        self._last_raw_y: Optional[np.ndarray] = None

        self.chkFitEMG.toggled.connect(self._on_emg_toggle)
        self.chkRise.toggled.connect(lambda _: self._rerun_emg())
        self.chk2nd.toggled.connect(lambda _: self._rerun_emg())
        self.chk3rd.toggled.connect(lambda _: self._rerun_emg())

    def _on_emg_toggle(self, checked: bool) -> None:
        self.chkRise.setEnabled(checked)
        self.chk2nd.setEnabled(checked)
        self.chk3rd.setEnabled(checked)
        if checked:
            self._rerun_emg()
        else:
            self._emg_fit_curve.setData([], [])
            self._emg_label.setText("")

    def _rerun_emg(self) -> None:
        if not self.chkFitEMG.isChecked() or self._last_time is None:
            return
        self._run_emg_fit(self._last_time, self._last_raw_y)

    def _run_emg_fit(self, t: np.ndarray, y: np.ndarray) -> None:
        result = fit_emg_histogram(
            t, y, sigma_ns=0.010,
            use_rise=self.chkRise.isChecked(),
            use_2nd=self.chk2nd.isChecked(),
            use_3rd=self.chk3rd.isChecked())
        if result is None:
            self._emg_fit_curve.setData([], [])
            self._emg_label.setText("fit failed")
            return
        t_dense = np.linspace(float(t[0]), float(t[-1]), 2000)
        y_fit = result['model'](t_dense, *result['popt'])
        self._emg_fit_curve.setData(t_dense, y_fit)
        parts = [f"{n}={v:.3g}" for n, v in zip(result['names'], result['popt'])
                 if 'τ' in n or 'tau' in n.lower()]
        self._emg_label.setText("  ".join(parts))

    def update(self, time_ns, counts_matrix, smoothed, t0_ns,
               current_aligned_ns, selected, log_y, normalize) -> None:
        self._time_line.setValue(current_aligned_ns)
        if normalize != self._normalize:
            self._normalize = normalize
            for pix in list(self._raw):
                self._pw.removeItem(self._raw.pop(pix))
                curve = self._smooth.pop(pix)
                self._legend.removeItem(curve)
                self._pw.removeItem(curve)
        for pix in list(self._raw):
            if pix not in selected:
                self._pw.removeItem(self._raw.pop(pix))
                curve = self._smooth.pop(pix)
                self._legend.removeItem(curve)
                self._pw.removeItem(curve)
        for pix in sorted(selected):
            aligned_t = time_ns - float(t0_ns[pix])
            raw_y = counts_matrix[:, pix].astype(float)
            smooth_y = smoothed[:, pix].copy()
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
                raw_item = pg.PlotDataItem(aligned_t, raw_y, pen=None,
                    symbol='o', symbolSize=3, symbolBrush=pg.mkBrush(color + '88'), symbolPen=None)
                self._pw.addItem(raw_item)
                self._raw[pix] = raw_item
                curve = self._pw.plot(aligned_t, smooth_y, pen=pg.mkPen(color, width=1.6), name=f"Px {pix}")
                self._smooth[pix] = curve
        self._pi.setLabel("left", "Normalized counts" if normalize else "Counts per bin")
        self._pw.setLogMode(y=log_y)
        # Store fit data in the same scale as displayed (normalized or raw)
        if selected:
            pix = min(selected)
            aligned_t = time_ns - float(t0_ns[pix])
            self._last_time = aligned_t
            fit_y = counts_matrix[:, pix].astype(float)
            if normalize:
                smooth_pix = smoothed[:, pix].copy()
                peak_val = float(np.max(smooth_pix))
                if peak_val > 0:
                    fit_y = fit_y / peak_val
            self._last_raw_y = fit_y
            if self.chkFitEMG.isChecked():
                self._run_emg_fit(self._last_time, self._last_raw_y)

    def clear(self) -> None:
        for pix in list(self._raw):
            self._pw.removeItem(self._raw.pop(pix))
        for pix in list(self._smooth):
            curve = self._smooth.pop(pix)
            try:
                self._legend.removeItem(curve)
            except Exception:
                pass
            self._pw.removeItem(curve)
        self._emg_fit_curve.setData([], [])
        self._emg_label.setText("")

# ── TRPL Map Widget ────────────────────────────────────────────────────────────

class TrplMapWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._pw = pg.PlotWidget()
        self._pi = self._pw.plotItem
        _fix_axes(self._pi)
        self._pi.setTitle("TRPL Map")
        self._pi.setLabel("bottom", "—")
        self._pi.setLabel("left", "Time (ns)")
        layout.addWidget(self._pw, 1)

        self._img = pg.ImageItem()
        self._img.setColorMap(pg.colormap.get("inferno"))
        self._pi.addItem(self._img)
        self._pi.setAspectLocked(False)

    def update(self, time_ns, x_vals, map_data, x_label, pixel, log_y, normalize) -> None:
        if map_data.size == 0 or x_vals.size == 0:
            self._img.clear()
            return
        data = map_data.astype(np.float64)
        if normalize:
            col_max = np.nanmax(data, axis=0)
            col_max[col_max <= 0] = 1.0
            data = data / col_max[np.newaxis, :]
        if log_y:
            with np.errstate(divide='ignore', invalid='ignore'):
                data = np.where(data > 0, np.log10(data), np.nan)
        finite = data[np.isfinite(data)]
        vmin, vmax = (float(np.nanmin(finite)), float(np.nanmax(finite))) if finite.size else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        dx = float(x_vals[1] - x_vals[0]) if x_vals.size > 1 else 1.0
        dt = float(time_ns[1] - time_ns[0]) if time_ns.size > 1 else 1.0
        tr = QtGui.QTransform()
        tr.translate(float(x_vals[0]) - dx / 2, float(time_ns[0]) - dt / 2)
        tr.scale(dx, dt)
        self._img.setTransform(tr)
        self._img.setImage(data, levels=(vmin, vmax))
        self._pi.setTitle(f"TRPL Map — pixel {pixel}")
        self._pi.setLabel("bottom", x_label)

# ── Last 5 Histograms Widget ───────────────────────────────────────────────────

class HistoryWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._pw = pg.PlotWidget()
        self._pi = self._pw.plotItem
        _fix_axes(self._pi)
        self._pi.setTitle("Last 5 Histograms (aligned to pixel t₀)")
        self._pi.setLabel("bottom", "Time (ns)")
        self._pi.setLabel("left", "Counts")
        self._pi.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._pi.addLegend()
        layout.addWidget(self._pw, 1)
        self._items: List[pg.PlotDataItem] = []

        # Condition-list overlay controls
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(4)
        ctrl.addWidget(QtWidgets.QLabel("Overlay conditions:"))
        self.condEdit = QtWidgets.QLineEdit()
        self.condEdit.setPlaceholderText("e.g. 0.0, 1.0, -1.5 (V or µW)")
        ctrl.addWidget(self.condEdit, 1)
        layout.addLayout(ctrl)

        self._pw2 = pg.PlotWidget()
        self._pi2 = self._pw2.plotItem
        _fix_axes(self._pi2)
        self._pi2.setTitle("Histograms at listed conditions (selected pixel)")
        self._pi2.setLabel("bottom", "Time (ns)")
        self._pi2.setLabel("left", "Counts")
        self._pi2.showGrid(x=True, y=True, alpha=0.25)
        self._legend2 = self._pi2.addLegend()
        layout.addWidget(self._pw2, 1)
        self._overlay_items: List[pg.PlotDataItem] = []

    def update(self, series: List[dict], log_y: bool, normalize: bool) -> None:
        for item in self._items:
            try:
                self._legend.removeItem(item)
            except Exception:
                pass
            self._pw.removeItem(item)
        self._items.clear()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, entry in enumerate(series[-5:]):
            t = np.asarray(entry["time_ns"], dtype=float)
            y = np.asarray(entry["counts"], dtype=float)
            if normalize:
                m = float(np.nanmax(y))
                if m > 0:
                    y = y / m
            item = self._pw.plot(t, y, pen=pg.mkPen(colors[i % 5], width=1.4), name=entry["label"])
            self._items.append(item)
        self._pi.setLabel("left", "Normalized counts" if normalize else "Counts")
        self._pw.setLogMode(y=log_y)

    def update_overlay(self, series: List[dict], log_y: bool, normalize: bool) -> None:
        for item in self._overlay_items:
            try:
                self._legend2.removeItem(item)
            except Exception:
                pass
            self._pw2.removeItem(item)
        self._overlay_items.clear()
        for i, entry in enumerate(series):
            t = np.asarray(entry["time_ns"], dtype=float)
            y = np.asarray(entry["counts"], dtype=float)
            if normalize:
                m = float(np.nanmax(y))
                if m > 0:
                    y = y / m
            color = _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
            item = self._pw2.plot(t, y, pen=pg.mkPen(color, width=1.4), name=entry["label"])
            self._overlay_items.append(item)
        self._pi2.setLabel("left", "Normalized counts" if normalize else "Counts")
        self._pw2.setLogMode(y=log_y)

# ── Gaussian Fit Widget ────────────────────────────────────────────────────────

_N_RADIAL_EXTRA = 5
_RADIAL_CURRENT_COLOR = '#111827'

class GaussianFitWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(6)
        ctrl.addWidget(QtWidgets.QLabel("Extra t (ns):"))
        self.extraTimesEdit = QtWidgets.QLineEdit()
        self.extraTimesEdit.setPlaceholderText("e.g. 0.5, 1.0, 2.0 (max 5)")
        self.extraTimesEdit.setFixedWidth(220)
        ctrl.addWidget(self.extraTimesEdit)
        ctrl.addWidget(QtWidgets.QLabel("Min pix-11 cts:"))
        self.spin_pix11_thresh = QtWidgets.QSpinBox()
        self.spin_pix11_thresh.setRange(0, 1_000_000)
        self.spin_pix11_thresh.setValue(20)
        self.spin_pix11_thresh.setFixedWidth(80)
        self.spin_pix11_thresh.setToolTip(
            "Skip time bins where SPAD pixel 11 count is below this value.\n"
            "Takes effect on the next 'Fit Current Map' / 'Fit All Maps' run.")
        ctrl.addWidget(self.spin_pix11_thresh)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw, 1)

        self.p_sigma = glw.addPlot(row=0, col=0)
        _fix_axes(self.p_sigma)
        self._sigma_type: str = "σ_eq²"
        self.p_sigma.setTitle("σ_eq² vs Time")
        self.p_sigma.setLabel("bottom", "Time (ns)")
        self.p_sigma.setLabel("left", "σ_eq² (µm²)")
        self.p_sigma.showGrid(x=True, y=True, alpha=0.25)

        self.p_radial = glw.addPlot(row=1, col=0)
        _fix_axes(self.p_radial)
        self.p_radial.setTitle("Radial profile")
        self.p_radial.setLabel("bottom", "r (µm)")
        self.p_radial.setLabel("left", "Counts")
        self.p_radial.showGrid(x=True, y=True, alpha=0.25)

        glw.ci.layout.setRowStretchFactor(0, 1)
        glw.ci.layout.setRowStretchFactor(1, 2)

        self._magnification: float = 25.0

        # σ² plot items
        self._sigma_reliable = self.p_sigma.plot(
            pen=pg.mkPen('#1d4ed8', width=1.1),
            symbol='o', symbolSize=5, symbolBrush=pg.mkBrush('#1d4ed8'), symbolPen=None)
        self._sigma_unreliable = self.p_sigma.plot(
            pen=None,
            symbol='o', symbolSize=5, symbolBrush=pg.mkBrush('#9ca3af'), symbolPen=None)
        self._sigma_vline = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#111827', style=QtCore.Qt.DashLine, width=0.95))
        self.p_sigma.addItem(self._sigma_vline)

        # Radial plot items — current trace (dark)
        self._radial_scatter = pg.ScatterPlotItem(
            symbol='o', size=5,
            brush=pg.mkBrush(_RADIAL_CURRENT_COLOR + 'aa'),
            pen=None)
        self.p_radial.addItem(self._radial_scatter)
        self._radial_fit = pg.PlotDataItem(pen=pg.mkPen(_RADIAL_CURRENT_COLOR, width=2.0))
        self.p_radial.addItem(self._radial_fit)

        # Extra traces (up to _N_RADIAL_EXTRA)
        self._radial_extra_scatter: List[pg.ScatterPlotItem] = []
        self._radial_extra: List[pg.PlotDataItem] = []
        for i in range(_N_RADIAL_EXTRA):
            c = _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
            sc = pg.ScatterPlotItem(symbol='o', size=4, brush=pg.mkBrush(c + 'aa'), pen=None)
            ln = pg.PlotDataItem(pen=pg.mkPen(c, width=1.6))
            self.p_radial.addItem(sc)
            self.p_radial.addItem(ln)
            self._radial_extra_scatter.append(sc)
            self._radial_extra.append(ln)

        self._radial_legend = self.p_radial.addLegend(offset=(10, 10))

    def set_magnification(self, M: float) -> None:
        self._magnification = max(1e-9, M)
        self._update_sigma_label()

    def set_sigma_type(self, sigma_type: str) -> None:
        self._sigma_type = sigma_type
        self._update_sigma_label()
        self.p_sigma.setTitle(f"{sigma_type} vs Time")

    def _update_sigma_label(self) -> None:
        M = self._magnification
        lbl = getattr(self, '_sigma_type', 'σ_eq²')
        self.p_sigma.setLabel("left", f"{lbl} (sample µm²,  M={M:.4g})")

    def update_sigma(self, t_arr, sigma2, current_t: float,
                     smooth_window: int = 1, reliable_mask=None) -> None:
        M2 = self._magnification ** 2
        if t_arr is not None and sigma2 is not None and len(t_arr) > 0:
            s2 = np.asarray(sigma2, dtype=float) / M2
            w = max(1, smooth_window)
            if w > 1:
                s2 = _rolling_median(s2, w)
            if reliable_mask is not None and len(reliable_mask) == len(s2):
                self._sigma_reliable.setData(t_arr[reliable_mask], s2[reliable_mask])
                self._sigma_unreliable.setData(t_arr[~reliable_mask], s2[~reliable_mask])
            else:
                self._sigma_reliable.setData(t_arr, s2)
                self._sigma_unreliable.setData([], [])
        else:
            self._sigma_reliable.setData([], [])
            self._sigma_unreliable.setData([], [])
        self._sigma_vline.setValue(current_t)

    def _clear_radial_legend(self) -> None:
        try:
            self._radial_legend.removeItem(self._radial_fit)
        except Exception:
            pass
        for ln in self._radial_extra:
            try:
                self._radial_legend.removeItem(ln)
            except Exception:
                pass

    def update_radial(self, frame: np.ndarray, fit,
                      extra_traces: list, normalize: bool = False) -> None:
        """Draw symmetric radial scatter + Gaussian fit line.

        extra_traces: list of (r_arr, z_arr, popt_1d, t_ns) tuples.
        popt_1d = [A, sigma, offset] for a 1D Gaussian.
        """
        self._clear_radial_legend()
        M = self._magnification
        if fit is None:
            self._radial_scatter.setData([], [])
            self._radial_fit.setData([], [])
            for sc, ln in zip(self._radial_extra_scatter, self._radial_extra):
                sc.setData([], [])
                ln.setData([], [])
            self.p_radial.setTitle("Radial profile — no fit")
            return

        popt = fit.popt  # [A, x0, y0, sx, sy, offset]
        A = float(popt[0])
        x0, y0 = float(popt[1]), float(popt[2])
        sx, sy = float(popt[3]), float(popt[4])
        offset = float(popt[5])
        sigma_eq = float(np.sqrt((sx**2 + sy**2) / 2.0))
        sigma_safe = max(sigma_eq, 1e-6) / M  # convert detector µm → sample µm

        # Compute radial distance for each pixel from fit center
        r_det = np.sqrt((_FIT_X_UM - x0) ** 2 + (_FIT_Y_UM - y0) ** 2)
        z_vals = np.asarray(frame, dtype=float)
        order = np.argsort(r_det)
        r_sorted = r_det[order] / M
        z_sorted = z_vals[order]

        rmax = float(np.max(r_sorted))
        r_line = np.linspace(-rmax, rmax, 401)
        fit_line = A * np.exp(-(r_line ** 2) / (2 * sigma_safe ** 2)) + offset

        if normalize and A > 0:
            z_sorted = (z_sorted - offset) / A
            fit_line = (fit_line - offset) / A

        # Symmetric scatter (mirror to negative r)
        r_sym = np.concatenate([-r_sorted[::-1], r_sorted])
        z_sym = np.concatenate([z_sorted[::-1], z_sorted])

        self._radial_scatter.setData(x=r_sym, y=z_sym)
        self._radial_fit.setData(r_line, fit_line)
        self._radial_legend.addItem(self._radial_fit, "current")

        self.p_radial.setTitle(f"Radial — σ_eq={sigma_safe:.2f} µm (sample)")
        self.p_radial.setLabel("left", "Normalized intensity" if normalize else "Counts")

        # Extra traces
        for i, (sc, ln) in enumerate(zip(self._radial_extra_scatter, self._radial_extra)):
            if i >= len(extra_traces):
                sc.setData([], [])
                ln.setData([], [])
                continue
            r_e, z_e, popt_e, t_ns = extra_traces[i]
            if popt_e is None or len(r_e) == 0:
                sc.setData([], [])
                ln.setData([], [])
                continue
            Ae = float(popt_e[0])
            sig_e = max(abs(float(popt_e[1])), 1e-6)
            off_e = float(popt_e[2])
            rmax_e = max(float(np.max(r_e)), rmax)
            r_le = np.linspace(-rmax_e, rmax_e, 401)
            if normalize and Ae > 0:
                fl_e = np.exp(-(r_le ** 2) / (2 * sig_e ** 2))
            else:
                fl_e = Ae * np.exp(-(r_le ** 2) / (2 * sig_e ** 2)) + off_e
            ln.setData(r_le, fl_e)
            self._radial_legend.addItem(ln, f"t = {t_ns:.3f} ns")
            if z_e is not None and len(z_e) > 0:
                r_sc = np.asarray(r_e, dtype=float)
                z_sc = np.asarray(z_e, dtype=float)
                r_sc_sym = np.concatenate([-r_sc[::-1], r_sc])
                z_sc_sym = np.concatenate([z_sc[::-1], z_sc])
                if normalize and Ae > 0:
                    z_sc_sym = (z_sc_sym - off_e) / Ae
                sc.setData(x=r_sc_sym, y=z_sc_sym)
            else:
                sc.setData([], [])

    def clear(self) -> None:
        self._sigma_reliable.setData([], [])
        self._sigma_unreliable.setData([], [])
        self._radial_scatter.setData([], [])
        self._radial_fit.setData([], [])
        for sc, ln in zip(self._radial_extra_scatter, self._radial_extra):
            sc.setData([], [])
            ln.setData([], [])
        self._clear_radial_legend()
        self.p_radial.setTitle("Radial profile")


# ── PL Image Widget ────────────────────────────────────────────────────────────

class PlImageWidget(QtWidgets.QWidget):
    linecut_changed = pyqtSignal(int)  # display row (0 = top)

    def __init__(self, title: str = "PL Image", default_linecut_width: int = 1, parent=None):
        super().__init__(parent)
        self._img_data: Optional[np.ndarray] = None
        self._wl:       Optional[np.ndarray] = None
        self._h = self._w = 0
        self._title = str(title)
        self._build_ui(int(default_linecut_width))

    def _build_ui(self, lc_width: int) -> None:
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(4)

        self._titleLbl = QtWidgets.QLabel(self._title)
        vlay.addWidget(self._titleLbl)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Linecut width:"))
        self.linecutWidthSpin = QtWidgets.QSpinBox()
        self.linecutWidthSpin.setRange(1, 1000)
        self.linecutWidthSpin.setValue(lc_width)
        self.linecutWidthSpin.setFixedWidth(60)
        ctrl.addWidget(self.linecutWidthSpin)
        ctrl.addStretch()
        vlay.addLayout(ctrl)

        self._glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self._glw, 1)

        # Image plot (col=0) — autoRange disabled so setXRange/setYRange always stick
        self._imgPlot = self._glw.addPlot(row=0, col=0)
        self._imgPlot.getViewBox().invertY(True)
        self._imgPlot.vb.disableAutoRange()
        self._imgPlot.setLabel("bottom", "Wavelength (nm)")
        self._imgPlot.setLabel("left", "Spatial pixel")
        _fix_axes(self._imgPlot)
        self._imgItem = pg.ImageItem()
        self._imgPlot.addItem(self._imgItem)

        # Movable horizontal line (display row → horizontal linecut)
        self._hline = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen("y", width=1.5),
            hoverPen=pg.mkPen("y", width=2.5),
        )
        self._hline.setValue(0)
        self._imgPlot.addItem(self._hline)
        self._hline.sigPositionChanged.connect(self._on_hline_moved)

        # Movable vertical line (wavelength column → vertical linecut)
        self._vline = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("c", width=1.5),
            hoverPen=pg.mkPen("c", width=2.5),
        )
        self._vline.setValue(0)
        self._imgPlot.addItem(self._vline)
        self._vline.sigPositionChanged.connect(self._refresh_vlinecut)

        # Vertical linecut (col=1): Y linked to image, left axis hidden (shared range)
        self._vlcPlot = self._glw.addPlot(row=0, col=1)
        self._vlcPlot.setLabel("bottom", "Intensity")
        self._vlcPlot.hideAxis("left")
        self._vlcPlot.getViewBox().invertY(True)
        self._vlcPlot.setYLink(self._imgPlot)
        _fix_axes(self._vlcPlot)
        self._vlcCurve = self._vlcPlot.plot(pen=pg.mkPen("#ff7f0e", width=1.2))

        # Horizontal linecut plot (spectrum at cursor row) — spans both columns
        self._lcPlot = self._glw.addPlot(row=1, col=0, colspan=2)
        self._lcPlot.setLabel("bottom", "Wavelength (nm)")
        self._lcPlot.setLabel("left", "Intensity")
        self._lcPlot.addLegend(offset=(-10, 10))
        _fix_axes(self._lcPlot)
        self._lcCurve  = self._lcPlot.plot(pen=pg.mkPen("#1f77b4", width=1.2), name="Linecut")
        self._fitTotal = self._lcPlot.plot(
            pen=pg.mkPen("#ffb300", width=1.2, style=QtCore.Qt.DashLine), name="Fit total")
        self._fitPeak1 = self._lcPlot.plot(
            pen=pg.mkPen("#1f77b4", width=1.0, style=QtCore.Qt.DotLine), name="Fit peak 1")
        self._fitPeak2 = self._lcPlot.plot(
            pen=pg.mkPen("#d62728", width=1.0, style=QtCore.Qt.DotLine), name="Fit peak 2")

        self._glw.ci.layout.setRowStretchFactor(0, 4)
        self._glw.ci.layout.setRowStretchFactor(1, 2)
        self._glw.ci.layout.setColumnStretchFactor(0, 4)   # image
        self._glw.ci.layout.setColumnStretchFactor(1, 2)   # vertical linecut

        self.linecutWidthSpin.valueChanged.connect(lambda _: self._refresh_linecut())

    # ── public API ─────────────────────────────────────────────────────────────

    def linecut_row(self) -> Optional[int]:
        """Display row (0 = top). Caller converts to raw row via h-1-row_d."""
        if self._h == 0:
            return None
        return int(np.clip(round(self._hline.value()), 0, self._h - 1))

    def linecut_width(self) -> int:
        return int(self.linecutWidthSpin.value())

    def update_frame(self, data: dict) -> None:
        """data["image"] = 2-D float array (rows × cols)."""
        img = np.asarray(data["image"], dtype=float)
        self._img_data = img
        self._h, self._w = img.shape
        # flipud: display row 0 (top) = raw row h-1; no transpose (row-major: row→Y, col→X)
        disp = np.flipud(img)
        wl = self._wl
        if wl is not None and wl.size == self._w:
            x0 = float(wl[0])
            x_span = float(wl[-1] - wl[0]) or 1.0
        else:
            x0, x_span = 0.0, float(self._w) or 1.0
        disp_safe = disp if np.isfinite(disp).any() else np.zeros_like(disp)
        self._imgItem.setImage(disp_safe, autoLevels=True)
        self._imgItem.setRect(QtCore.QRectF(x0, 0, x_span, float(self._h)))
        self._imgPlot.setXRange(x0, x0 + x_span, padding=0.01)
        self._imgPlot.setYRange(0, float(self._h), padding=0)
        self._vline.setValue(x0 + x_span / 2.0)
        # Place hline at centre if it's out of range (first load)
        if not (0 <= self._hline.value() <= self._h):
            self._hline.setValue(float(self._h) / 2.0)
        self._refresh_linecut()
        self._refresh_vlinecut()

    def set_wavelength_axis(self, wl: Optional[np.ndarray]) -> None:
        self._wl = None if wl is None else np.asarray(wl, dtype=float).ravel()

    def set_image_title(self, text: str) -> None:
        self._titleLbl.setText(str(text))

    def set_crop(self, *args) -> None:
        pass  # crop applied before calling update_frame

    def set_linecut_fit_overlay(self, x_axis, y_total, y1, y2) -> None:
        if x_axis is None or y_total is None:
            self.clear_linecut_fit_overlay()
            return
        x = np.asarray(x_axis, dtype=float).ravel()
        if x.size == 0:
            self.clear_linecut_fit_overlay()
            return
        def _v(y):
            if y is None:
                return None
            a = np.asarray(y, dtype=float).ravel()
            return a if (a.size == x.size and np.isfinite(a).any()) else None
        yt, y1v, y2v = _v(y_total), _v(y1), _v(y2)
        self._fitTotal.setData(x, yt)  if yt  is not None else self._fitTotal.setData([], [])
        self._fitPeak1.setData(x, y1v) if y1v is not None else self._fitPeak1.setData([], [])
        self._fitPeak2.setData(x, y2v) if y2v is not None else self._fitPeak2.setData([], [])

    def clear_linecut_fit_overlay(self) -> None:
        self._fitTotal.setData([], [])
        self._fitPeak1.setData([], [])
        self._fitPeak2.setData([], [])

    # ── internal ───────────────────────────────────────────────────────────────

    def _on_hline_moved(self) -> None:
        row_d = self.linecut_row()
        if row_d is not None:
            self.linecut_changed.emit(row_d)
        self._refresh_linecut()

    def _refresh_linecut(self) -> None:
        if self._img_data is None or self._h == 0:
            self._lcCurve.setData([], [])
            return
        row_d   = int(np.clip(round(self._hline.value()), 0, self._h - 1))
        row_raw = self._h - 1 - row_d  # display→raw (flipud was applied)
        lc = linecut_horizontal(self._img_data, row_raw, self.linecut_width())
        if lc is None:
            self._lcCurve.setData([], [])
            return
        x = (self._wl if (self._wl is not None and self._wl.size == lc.size)
             else np.arange(lc.size, dtype=float))
        self._lcCurve.setData(x, lc)

    def _refresh_vlinecut(self) -> None:
        if self._img_data is None or self._h == 0 or self._w == 0:
            self._vlcCurve.setData([], [])
            return
        pos = float(self._vline.value())
        if self._wl is not None and self._wl.size == self._w:
            col = int(np.argmin(np.abs(self._wl - pos)))
        else:
            col = int(np.clip(round(pos), 0, self._w - 1))
        col = max(0, min(col, self._w - 1))
        profile = self._img_data[::-1, col]  # flip to match display order
        self._vlcCurve.setData(profile, np.arange(self._h, dtype=float))


# ── Map Plot Widget V2 ─────────────────────────────────────────────────────────

class MapPlotWidgetV2(QtWidgets.QGroupBox):
    def __init__(self, title: str, *, cmap_name: str, line_label: str, parent=None):
        super().__init__(title, parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        br = QtWidgets.QHBoxLayout()
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setMaximumWidth(BTN_W)
        br.addWidget(self.exportBtn)
        br.addStretch()
        vbox.addLayout(br)

        glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(glw, 1)

        self.p_map = glw.addPlot(row=0, col=0)
        _fix_axes(self.p_map)
        self.p_map.getAxis("top").show()

        self.p_cut = glw.addPlot(row=1, col=0)
        _fix_axes(self.p_cut)
        self.p_cut.setLabel("left", line_label)
        self.p_cut.vb.setXLink(self.p_map.vb)

        glw.ci.layout.setRowStretchFactor(0, 4)
        glw.ci.layout.setRowStretchFactor(1, 1)

        try:
            self._cmap = pg.colormap.get(cmap_name)
        except Exception:
            self._cmap = pg.colormap.get("viridis")
        self._lut = self._cmap.getLookupTable(nPts=512)

        self._img_item = pg.ImageItem()
        self._img_item.setLookupTable(self._lut)
        self.p_map.addItem(self._img_item)

        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                      pen=pg.mkPen("r", width=1))
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen("#2563eb", width=1))
        self.p_map.addItem(self._hline)
        self.p_map.addItem(self._vline)

        self._cut_item = pg.PlotDataItem(pen=pg.mkPen("k", width=1))
        self.p_cut.addItem(self._cut_item)

        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None
        self._selected_y_idx: int = 0
        self._manual_sym_max: Optional[float] = None
        self._top_x: Optional[np.ndarray] = None
        self._top_label: str = ""
        self._line_label = line_label

        self.exportBtn.clicked.connect(self._on_export_csv)
        self.p_map.scene().sigMouseClicked.connect(self._on_map_click)

    def _on_map_click(self, event) -> None:
        if self._y is None or event.button() != QtCore.Qt.LeftButton:
            return
        if not self.p_map.vb.sceneBoundingRect().contains(event.scenePos()):
            return
        pos = self.p_map.vb.mapSceneToView(event.scenePos())
        idx = int(np.argmin(np.abs(self._y - float(pos.y()))))
        self.select_y_index(idx)

    def set_map(self, x, y, z, *, x_label: str = "x", y_label: str = "y",
                symmetric: bool = False) -> None:
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._z = np.asarray(z, dtype=float)
        self.p_map.setLabel("bottom", x_label)
        self.p_map.setLabel("left", y_label)
        self.p_cut.setLabel("bottom", x_label)
        nx, ny = self._x.size, self._y.size
        if nx < 1 or ny < 1:
            return
        self._selected_y_idx = int(np.clip(self._selected_y_idx, 0, ny - 1))
        dx = float(np.median(np.diff(self._x))) if nx > 1 else 1.0
        dy = float(np.median(np.diff(self._y))) if ny > 1 else 1.0
        if symmetric or self._manual_sym_max is not None:
            amax = (self._manual_sym_max if self._manual_sym_max is not None
                    else float(np.nanmax(np.abs(self._z))))
            zmin, zmax = -amax, amax
        else:
            zmin = float(np.nanmin(self._z))
            zmax = float(np.nanmax(self._z))
        if zmax <= zmin:
            zmax = zmin + 1e-9
        tr = QtGui.QTransform()
        tr.translate(float(self._x[0]) - dx / 2, float(self._y[0]) - dy / 2)
        tr.scale(dx, dy)
        self._img_item.setTransform(tr)
        self._img_item.setImage(self._z, levels=(zmin, zmax))
        self._update_top_axis()
        self._update_hline()
        self._draw_linecut()

    def _update_top_axis(self) -> None:
        if self._top_x is None or self._x is None or self._x.size < 2:
            return
        nticks = min(8, self._x.size, self._top_x.size)
        x_idxs   = np.round(np.linspace(0, self._x.size - 1,     nticks)).astype(int)
        top_idxs  = np.round(np.linspace(0, self._top_x.size - 1, nticks)).astype(int)
        ticks = [[(float(self._x[xi]), f"{self._top_x[ti]:.3g}")
                  for xi, ti in zip(x_idxs, top_idxs)]]
        self.p_map.getAxis("top").setTicks(ticks)
        self.p_map.getAxis("top").setLabel(self._top_label)

    def _update_hline(self) -> None:
        if self._y is not None and self._selected_y_idx < self._y.size:
            self._hline.setValue(float(self._y[self._selected_y_idx]))

    def _draw_linecut(self) -> None:
        if self._x is None or self._z is None:
            return
        idx = min(self._selected_y_idx, self._z.shape[0] - 1)
        self._cut_item.setData(self._x, self._z[idx, :])

    def set_x_marker(self, x: float) -> None:
        self._vline.setValue(float(x))

    def set_secondary_axis(self, x2, label: str = "") -> None:
        if x2 is None:
            self._top_x = None
            self.p_map.getAxis("top").setTicks(None)
            self.p_map.getAxis("top").setLabel("")
            return
        self._top_x = np.asarray(x2, dtype=float)
        self._top_label = str(label)
        self._update_top_axis()

    def set_manual_symmetric_max(self, value) -> None:
        self._manual_sym_max = float(value) if value is not None else None

    def select_y_index(self, idx: int) -> None:
        if self._y is None:
            return
        self._selected_y_idx = int(np.clip(idx, 0, self._y.size - 1))
        self._update_hline()
        self._draw_linecut()

    def _on_export_csv(self) -> None:
        if self._x is None or self._y is None or self._z is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(["y\\x"] + [f"{v:.6g}" for v in self._x])
                for i, yv in enumerate(self._y):
                    wr.writerow([f"{yv:.6g}"] + [f"{v:.6g}" for v in self._z[i, :]])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))


# ── Fit Trend Plot Widget V2 ───────────────────────────────────────────────────

class FitTrendPlotWidgetV2(QtWidgets.QGroupBox):
    def __init__(self, title: str, y_label: str, parent=None):
        super().__init__(title, parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        br = QtWidgets.QHBoxLayout()
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setMaximumWidth(BTN_W)
        br.addWidget(self.exportBtn)
        br.addStretch()
        vbox.addLayout(br)

        self._pw = pg.PlotWidget()
        pi = self._pw.plotItem
        _fix_axes(pi)
        self._pw.setLabel("left", y_label)
        vbox.addWidget(self._pw, 1)

        self._item1 = pg.PlotDataItem(pen=pg.mkPen("#1f77b4", width=2), name="")
        self._item2 = pg.PlotDataItem(pen=pg.mkPen("#d62728", width=2), name="")
        self._pw.addItem(self._item1)
        self._pw.addItem(self._item2)
        self._legend = pi.addLegend(offset=(10, 10))
        self._legend.addItem(self._item1, "")
        self._legend.addItem(self._item2, "")

        self._x: Optional[np.ndarray] = None
        self._y1: Optional[np.ndarray] = None
        self._y2: Optional[np.ndarray] = None
        self._label1: str = ""
        self._label2: str = ""
        self._text_item: Optional[pg.TextItem] = None

        self.exportBtn.clicked.connect(self._on_export_csv)

    def _set_legend_labels(self, l1: str, l2: str) -> None:
        try:
            self._legend.clear()
            if l1:
                self._legend.addItem(self._item1, l1)
            if l2:
                self._legend.addItem(self._item2, l2)
        except Exception:
            pass

    def set_data(self, x, y1, y2, *, x_label: str, label_1: str, label_2: str) -> None:
        self._x = np.asarray(x, dtype=float)
        self._y1 = np.asarray(y1, dtype=float)
        self._y2 = np.asarray(y2, dtype=float)
        self._label1 = label_1
        self._label2 = label_2
        if self._text_item is not None:
            self._pw.plotItem.removeItem(self._text_item)
            self._text_item = None
        self._pw.setLabel("bottom", x_label)
        self._item1.setData(self._x, self._y1)
        self._item2.setData(self._x, self._y2)
        self._set_legend_labels(label_1, label_2)

    def clear_data(self, message: str, *, x_label: str) -> None:
        self._x = self._y1 = self._y2 = None
        self._item1.setData([], [])
        self._item2.setData([], [])
        self._pw.setLabel("bottom", x_label)
        if self._text_item is not None:
            self._pw.plotItem.removeItem(self._text_item)
        self._text_item = pg.TextItem(message, color="k", anchor=(0.5, 0.5))
        self._pw.plotItem.addItem(self._text_item)
        self._text_item.setPos(0.5, 0.5)

    def _on_export_csv(self) -> None:
        if self._x is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(["x", self._label1, self._label2])
                for x, y1, y2 in zip(self._x, self._y1, self._y2):
                    wr.writerow([f"{x:.6g}", f"{y1:.6g}", f"{y2:.6g}"])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))


# ── Space-Time Map Widget ──────────────────────────────────────────────────────

class SpaceTimeMapWidget(QtWidgets.QWidget):
    """Space-time (r–t) colormap — populated by Fit Current Map."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        ctrl = QtWidgets.QHBoxLayout()
        self._normalize_chk = QtWidgets.QCheckBox("Norm. per time slice")
        self._normalize_chk.setToolTip(
            "Normalize intensity to [0,1] independently at each time bin")
        ctrl.addWidget(self._normalize_chk)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._pw = pg.PlotWidget()
        pi = self._pw.plotItem
        _fix_axes(pi)
        pi.setTitle("Space-Time Map (current condition)")
        pi.setLabel("bottom", "r (sample µm)")
        pi.setLabel("left", "Time (ns)")
        pi.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self._pw, 1)

        self._img = pg.ImageItem()
        try:
            self._img.setColorMap(pg.colormap.get("inferno"))
        except Exception:
            pass
        pi.addItem(self._img)

        self._xt_sigma_curve = pi.plot(
            pen=pg.mkPen('w', width=1.8, style=QtCore.Qt.DashLine))
        self._xt_sigma_curve_neg = pi.plot(
            pen=pg.mkPen('w', width=1.8, style=QtCore.Qt.DashLine))
        self._xt_vline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#aaaaaa', style=QtCore.Qt.DotLine, width=1.0))
        pi.addItem(self._xt_vline)

        self._magnification: float = 1.0
        self._lbl = QtWidgets.QLabel("Run 'Fit Current Map' to populate.")
        layout.addWidget(self._lbl)

        self._last_args: Optional[tuple] = None
        self._normalize_chk.toggled.connect(self._on_normalize_toggled)

    def set_magnification(self, M: float) -> None:
        self._magnification = max(1e-9, float(M))
        if self._last_args is not None:
            self.update(*self._last_args)

    def update(self, t_arr: np.ndarray, popt_list: list,
               r_max_det: float,
               sigma2_det: Optional[np.ndarray] = None,
               normalize: bool = False) -> None:
        """Synthetic dense-grid space-time map mirrored to ±r.
        t_arr: time values (ns), shape (n_t,)
        popt_list: list of 6-element [A,x0,y0,sx,sy,offset] arrays, length n_t
        r_max_det: max radial distance in detector µm
        sigma2_det: σ_eq² (detector µm²), same length as t_arr"""
        self._last_args = (t_arr, popt_list, r_max_det, sigma2_det, normalize)
        M = self._magnification
        t_arr = np.asarray(t_arr, dtype=float)
        n_t = len(t_arr)
        if n_t == 0 or not popt_list:
            self._img.clear()
            self._xt_sigma_curve.setData([], [])
            self._xt_sigma_curve_neg.setData([], [])
            self._lbl.setText("No data.")
            return
        r_max_s = float(r_max_det) / M
        r_half = np.linspace(0.0, r_max_s, R_GRID)  # 0 → r_max_s sample µm
        n_r_full = 2 * R_GRID - 1  # full symmetric: -r_max_s → +r_max_s
        d = np.full((n_t, n_r_full), np.nan, dtype=float)
        for k, popt in enumerate(popt_list):
            A, _x0, _y0, sx, sy, offset = (float(v) for v in popt)
            sx = max(abs(sx), 1e-6); sy = max(abs(sy), 1e-6)
            sigma_eq_s = max(np.sqrt((sx ** 2 + sy ** 2) / 2.0) / M, 1e-9)  # sample µm
            z_half = A * np.exp(-r_half ** 2 / (2.0 * sigma_eq_s ** 2)) + offset
            d[k] = np.concatenate([z_half[::-1][:-1], z_half])
        if normalize or self._normalize_chk.isChecked():
            row_min = np.nanmin(d, axis=1, keepdims=True)
            row_max = np.nanmax(d, axis=1, keepdims=True)
            span = row_max - row_min
            span[span <= 0] = 1.0
            d = (d - row_min) / span
        finite = d[np.isfinite(d)]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-9
        t_min = float(t_arr[0])
        t_max = float(t_arr[-1]) if n_t > 1 else t_min + 1.0
        dt = (t_max - t_min) / n_t if n_t > 1 else 1.0
        # row-major: data[t, r_col] → y=time, x=r; image spans -r_max_s to +r_max_s
        tr = QtGui.QTransform()
        tr.translate(-r_max_s, t_min)
        tr.scale(2.0 * r_max_s / n_r_full, dt)
        self._img.setTransform(tr)
        self._img.setImage(d, levels=(vmin, vmax))
        self._pw.plotItem.setLabel("bottom", f"r (sample µm,  M={M:.4g})")
        self._pw.plotItem.setXRange(-r_max_s, r_max_s, padding=0.02)
        self._pw.plotItem.setYRange(t_min, t_max, padding=0.02)
        if sigma2_det is not None and len(sigma2_det) == n_t:
            sigma_s = np.sqrt(np.maximum(np.asarray(sigma2_det, dtype=float), 0.0)) / M
            self._xt_sigma_curve.setData(sigma_s, t_arr)
            self._xt_sigma_curve_neg.setData(-sigma_s, t_arr)
        else:
            self._xt_sigma_curve.setData([], [])
            self._xt_sigma_curve_neg.setData([], [])
        self._lbl.setText(f"{n_t} time bins × {n_r_full} r points (R_GRID={R_GRID})")

    def update_time_marker(self, t_ns: float) -> None:
        self._xt_vline.setValue(t_ns)

    def _on_normalize_toggled(self, _=None) -> None:
        if self._last_args is not None:
            self.update(*self._last_args)

    def clear(self) -> None:
        self._img.clear()
        self._xt_sigma_curve.setData([], [])
        self._xt_sigma_curve_neg.setData([], [])
        self._last_args = None
        self._lbl.setText("Run 'Fit Current Map' to populate.")


# ── Sigma² Trace Overlay Widget (replaces DiffusionMapWidget) ──────────────────

class SigmaTraceOverlayWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("σ² Traces", parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(6)
        self.normalizeChk = QtWidgets.QCheckBox("Normalize by σ²(t=0)")
        ctrl.addWidget(self.normalizeChk)
        ctrl.addWidget(QtWidgets.QLabel("Show conditions:"))
        self.condEdit = QtWidgets.QLineEdit()
        self.condEdit.setPlaceholderText("all  – or e.g. 0.0, 1.0, -1.5")
        ctrl.addWidget(self.condEdit, 1)
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setMaximumWidth(BTN_W)
        ctrl.addWidget(self.exportBtn)
        vbox.addLayout(ctrl)

        self._pw = pg.PlotWidget()
        pi = self._pw.plotItem
        _fix_axes(pi)
        pi.setTitle("σ_eq²(t) per condition")
        pi.setLabel("bottom", "Time (ns)")
        pi.setLabel("left", "σ_eq² (µm²)")
        pi.showGrid(x=True, y=True, alpha=0.25)
        self._legend = pi.addLegend(offset=(10, 10))
        vbox.addWidget(self._pw, 1)

        self._per_cond: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._x_label: str = "Condition"
        self._sigma_type: str = "σ_eq²"
        self._items: List[pg.PlotDataItem] = []

        self.normalizeChk.toggled.connect(self._refresh)
        self.condEdit.editingFinished.connect(self._refresh)
        self.exportBtn.clicked.connect(self._on_export_csv)

    def set_data(self, per_cond: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                 x_label: str = "Condition", sigma_type: str = "σ_eq²") -> None:
        self._per_cond = dict(per_cond)
        self._x_label = str(x_label)
        self._sigma_type = sigma_type
        self._refresh()

    def set_sigma_type(self, sigma_type: str) -> None:
        self._sigma_type = sigma_type
        self._refresh()

    @staticmethod
    def _key_numeric(key: str) -> float:
        """Extract numeric value from a key like 'Front gate (V)=0.500'."""
        try:
            return float(key.split("=")[-1])
        except Exception:
            return float("nan")

    def _selected_keys(self) -> List[str]:
        txt = self.condEdit.text().strip()
        if not txt or txt.lower() == "all":
            return list(self._per_cond.keys())
        selected = []
        all_keys = list(self._per_cond.keys())
        all_nums = [self._key_numeric(k) for k in all_keys]
        for tok in txt.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = float(tok)
            except ValueError:
                continue
            finite_pairs = [(abs(n - v), k) for n, k in zip(all_nums, all_keys) if np.isfinite(n)]
            if not finite_pairs:
                continue
            best = min(finite_pairs)[1]
            if best not in selected:
                selected.append(best)
        return selected if selected else all_keys

    @staticmethod
    def _pick_s2(entry, sigma_type: str) -> np.ndarray:
        """Extract the right σ² column from a (t, s2_eq, s2_x, s2_y) tuple."""
        if entry is None or len(entry) < 4:
            return entry[1] if entry is not None and len(entry) >= 2 else np.array([])
        _t, s2_eq, s2_x, s2_y = entry
        return {'σ_eq²': s2_eq, 'σ_x²': s2_x, 'σ_y²': s2_y}.get(sigma_type, s2_eq)

    def _refresh(self) -> None:
        for item in self._items:
            try:
                self._legend.removeItem(item)
            except Exception:
                pass
            self._pw.removeItem(item)
        self._items.clear()
        keys = self._selected_keys()
        norm = self.normalizeChk.isChecked()
        for i, key in enumerate(keys):
            entry = self._per_cond.get(key)
            if entry is None or len(entry[0]) == 0:
                continue
            t_arr = entry[0]
            y = np.asarray(self._pick_s2(entry, self._sigma_type), dtype=float)
            if norm:
                finite = y[np.isfinite(y)]
                y0 = float(finite[0]) if finite.size > 0 else 1.0
                if y0 > 0:
                    y = y / y0
            color = _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
            item = self._pw.plot(t_arr, y, pen=pg.mkPen(color, width=1.6), name=key)
            self._items.append(item)
        lbl = self._sigma_type
        self._pw.plotItem.setTitle(f"{lbl}(t) per condition")
        self._pw.plotItem.setLabel("left",
            "σ²/σ₀²" if norm else f"{lbl} (µm²)")

    def clear(self) -> None:
        self._per_cond.clear()
        self._refresh()

    def _on_export_csv(self) -> None:
        if not self._per_cond:
            return
        col_idx = {'σ_eq²': 1, 'σ_x²': 2, 'σ_y²': 3}.get(self._sigma_type, 1)
        tag = {'σ_eq²': 'sigma_eq2', 'σ_x²': 'sigma_x2', 'σ_y²': 'sigma_y2'}.get(
            self._sigma_type, 'sigma_eq2')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export σ² traces CSV",
            f"sigma_traces_{tag}.csv", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            keys = self._selected_keys()
            data = {}
            for key in keys:
                entry = self._per_cond.get(key)
                if entry is not None and len(entry[0]) > 0:
                    t_arr = np.asarray(entry[0], dtype=float)
                    s2 = np.asarray(entry[col_idx] if len(entry) > col_idx else entry[1], dtype=float)
                    data[key] = (t_arr, s2)
            if not data:
                return
            all_t = np.sort(np.unique(np.concatenate([v[0] for v in data.values()])))
            s2_lookup = {key: {float(t): float(s2) for t, s2 in zip(*data[key])}
                         for key in data}
            ordered_keys = list(data.keys())
            x_lbl = self._x_label or "Condition"
            col_vals = [k.split("=")[-1] for k in ordered_keys]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow([f"time_ns / {x_lbl}"] + col_vals)
                for t in all_t:
                    row = [f"{t:.10g}"]
                    for key in ordered_keys:
                        row.append(f"{s2_lookup[key][t]:.10g}" if t in s2_lookup[key] else "")
                    wr.writerow(row)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))


# ── Main Window ────────────────────────────────────────────────────────────────

class PowerGateDepWindowV2(QtWidgets.QMainWindow):
    MODE_POWER = "Power"
    MODE_GATE = "Gate"

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__()
        # v1 state
        self._crop: Tuple[int, int, int, int] = DEFAULT_CROP
        self._grid: Dict[Tuple[float, float], PLFrame] = {}
        self._trpl_grid: Dict[Tuple[float, float], TRPLRecord] = {}
        self._powers = np.array([], dtype=float)
        self._gates = np.array([], dtype=float)
        self._backs_for_gate = np.array([], dtype=float)
        self._default_power_key: float = 0.0
        self._default_gate_key: float = 0.0
        self._can_power_mode: bool = False
        self._can_gate_mode: bool = False
        self._blank_image = np.zeros((16, 16), dtype=float)
        self._blank_wl = np.arange(16, dtype=float)
        self._cropped_image_cache: Dict[Path, np.ndarray] = {}
        self._cropped_wl_cache: Dict[Path, np.ndarray] = {}
        self._current_mode: str = self.MODE_POWER
        self._main_idx: int = 0
        self._fixed_idx: int = 0
        self._map_version: int = 0
        self._fit_map_version: int = -1
        self._fit_result: Optional[dict] = None
        self._last_x: Optional[np.ndarray] = None
        self._last_x_label: Optional[str] = None
        self._last_y_wl: Optional[np.ndarray] = None
        self._last_map_data: Optional[np.ndarray] = None
        self._selected_spad_pixel: int = 11
        self._spad_fit_x_um = _FIT_X_UM
        self._spad_fit_y_um = _FIT_Y_UM
        self._selection_history: List[Tuple[float, float, int]] = []
        # v2 state
        self._align_result: Optional[AlignResult] = None
        self._align_cache: Dict[tuple, AlignResult] = {}
        self._current_aligned_idx: int = 0
        self._smooth_window: int = 3
        self._noise_level: float = 0.05
        self._align_t0: bool = True
        self._magnification: float = 1.0
        self._sigma2_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._fit_end_ns: float = 100.0
        self._diffusion_data: Optional[dict] = None
        self._fit_current_data: Optional[dict] = None  # {t, sigma2, sigma2_x, sigma2_y, popt_list, r_max_det}
        # data
        self.frames: List[PLFrame] = []
        self.trpl_records: List[TRPLRecord] = []
        self.defaults = ViewerDefaults(crop=DEFAULT_CROP, linecut_row=None, linecut_width=1)
        self.data_dir = Path(".")
        self._initializing = True
        self._build_ui()
        self._initializing = False
        if data_dir is not None:
            self._load_data_dir(Path(data_dir))

    def _build_ui(self) -> None:
        self.setWindowTitle("Power + Gate Dependent PL/TRPL (v2)")
        self.resize(1920, 1000)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Controls ──────────────────────────────────────────────────────────
        controls = QtWidgets.QGroupBox("Controls")
        grid = QtWidgets.QGridLayout(controls)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # row 0: data folder
        grid.addWidget(QtWidgets.QLabel("Data folder:"), 0, 0)
        self.folderEdit = QtWidgets.QLineEdit()
        self.folderEdit.setPlaceholderText("Folder containing PL_*.asc and TRPL_*.csv")
        grid.addWidget(self.folderEdit, 0, 1, 1, 5)
        self.browseFolderBtn = QtWidgets.QPushButton("Browse")
        self.browseFolderBtn.setFixedWidth(80)
        grid.addWidget(self.browseFolderBtn, 0, 6)
        self.loadFolderBtn = QtWidgets.QPushButton("Load")
        self.loadFolderBtn.setFixedWidth(70)
        grid.addWidget(self.loadFolderBtn, 0, 7)
        self.folderStatusLbl = QtWidgets.QLabel("")
        grid.addWidget(self.folderStatusLbl, 0, 8, 1, 2)

        # row 1: mode + main slider
        grid.addWidget(QtWidgets.QLabel("Mode:"), 1, 0)
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItems([self.MODE_POWER, self.MODE_GATE])
        self.modeCombo.setFixedWidth(160)
        grid.addWidget(self.modeCombo, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Main:"), 1, 2)
        self.mainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mainSlider.setTracking(True)
        grid.addWidget(self.mainSlider, 1, 3, 1, 4)
        self.mainValueLbl = QtWidgets.QLabel("--")
        self.mainValueLbl.setMinimumWidth(240)
        grid.addWidget(self.mainValueLbl, 1, 7, 1, 3)

        # row 2: fixed slider
        grid.addWidget(QtWidgets.QLabel("Fixed:"), 2, 0)
        self.fixedSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fixedSlider.setTracking(True)
        grid.addWidget(self.fixedSlider, 2, 1, 1, 6)
        self.fixedValueLbl = QtWidgets.QLabel("--")
        grid.addWidget(self.fixedValueLbl, 2, 7, 1, 3)

        # row 3: crop + deriv max
        self.cropTopSpin = QtWidgets.QSpinBox(); self.cropTopSpin.setRange(0, 10000); self.cropTopSpin.setFixedWidth(68)
        self.cropBottomSpin = QtWidgets.QSpinBox(); self.cropBottomSpin.setRange(0, 10000); self.cropBottomSpin.setFixedWidth(68)
        self.cropLeftSpin = QtWidgets.QSpinBox(); self.cropLeftSpin.setRange(0, 10000); self.cropLeftSpin.setFixedWidth(68)
        self.cropRightSpin = QtWidgets.QSpinBox(); self.cropRightSpin.setRange(0, 10000); self.cropRightSpin.setFixedWidth(68)
        for s, v in zip((self.cropTopSpin, self.cropBottomSpin, self.cropLeftSpin, self.cropRightSpin), self._crop):
            s.setValue(int(v))
        self.cropApplyBtn = QtWidgets.QPushButton("Apply Crop"); self.cropApplyBtn.setFixedWidth(100)
        grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 3, 0)
        grid.addWidget(self.cropTopSpin, 3, 1); grid.addWidget(self.cropBottomSpin, 3, 2)
        grid.addWidget(self.cropLeftSpin, 3, 3); grid.addWidget(self.cropRightSpin, 3, 4)
        grid.addWidget(self.cropApplyBtn, 3, 5)
        self.derivMaxSpin = QtWidgets.QDoubleSpinBox()
        self.derivMaxSpin.setDecimals(4); self.derivMaxSpin.setRange(0.0, 1e12)
        self.derivMaxSpin.setSpecialValueText("Auto"); self.derivMaxSpin.setValue(0.0); self.derivMaxSpin.setFixedWidth(110)
        grid.addWidget(QtWidgets.QLabel("Deriv max:"), 3, 6)
        grid.addWidget(self.derivMaxSpin, 3, 7)
        self.reverseWlCheck = QtWidgets.QCheckBox("Reverse λ")
        self.reverseWlCheck.setToolTip(
            "Flip the wavelength axis of the PL image (for Andor files with inverted λ)")
        grid.addWidget(self.reverseWlCheck, 3, 8)

        # row 4: peak guesses + fit
        self.peak1GuessEdit = QtWidgets.QLineEdit("855"); self.peak1GuessEdit.setFixedWidth(80)
        self.peak2GuessEdit = QtWidgets.QLineEdit("835"); self.peak2GuessEdit.setFixedWidth(80)
        self.refIndexSpin = QtWidgets.QSpinBox(); self.refIndexSpin.setRange(0, 0); self.refIndexSpin.setFixedWidth(75)
        self.maxShiftSpin = QtWidgets.QDoubleSpinBox()
        self.maxShiftSpin.setDecimals(2); self.maxShiftSpin.setRange(0.2, 20.0)
        self.maxShiftSpin.setValue(2.0); self.maxShiftSpin.setSingleStep(0.2); self.maxShiftSpin.setFixedWidth(75)
        self.fitPeaksBtn = QtWidgets.QPushButton("Fit Peaks"); self.fitPeaksBtn.setFixedWidth(BTN_W)
        grid.addWidget(QtWidgets.QLabel("Peak1 (nm):"), 4, 0); grid.addWidget(self.peak1GuessEdit, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Peak2 (nm):"), 4, 2); grid.addWidget(self.peak2GuessEdit, 4, 3)
        grid.addWidget(QtWidgets.QLabel("Ref idx:"), 4, 4); grid.addWidget(self.refIndexSpin, 4, 5)
        grid.addWidget(QtWidgets.QLabel("Max shift:"), 4, 6); grid.addWidget(self.maxShiftSpin, 4, 7)
        grid.addWidget(self.fitPeaksBtn, 4, 8)

        # row 5: TRPL time range + histogram y range + normalize + log
        self.trplTminSpin = QtWidgets.QDoubleSpinBox()
        self.trplTminSpin.setDecimals(4); self.trplTminSpin.setRange(0.0, 1e9); self.trplTminSpin.setValue(0.0); self.trplTminSpin.setFixedWidth(95)
        self.trplTmaxSpin = QtWidgets.QDoubleSpinBox()
        self.trplTmaxSpin.setDecimals(4); self.trplTmaxSpin.setRange(0.0, 1e9); self.trplTmaxSpin.setValue(100.0); self.trplTmaxSpin.setFixedWidth(95)
        self.histYminEdit = QtWidgets.QLineEdit(); self.histYminEdit.setPlaceholderText("auto"); self.histYminEdit.setFixedWidth(75)
        self.histYmaxEdit = QtWidgets.QLineEdit(); self.histYmaxEdit.setPlaceholderText("auto"); self.histYmaxEdit.setFixedWidth(75)
        self.trplNormalizeChk = QtWidgets.QCheckBox("Normalize"); self.trplNormalizeChk.setChecked(True)
        self.trplLogChk = QtWidgets.QCheckBox("Log scale")
        grid.addWidget(QtWidgets.QLabel("t min/max ns:"), 5, 0)
        grid.addWidget(self.trplTminSpin, 5, 1); grid.addWidget(self.trplTmaxSpin, 5, 2)
        grid.addWidget(QtWidgets.QLabel("Hist y:"), 5, 3)
        grid.addWidget(self.histYminEdit, 5, 4); grid.addWidget(self.histYmaxEdit, 5, 5)
        grid.addWidget(self.trplNormalizeChk, 5, 6); grid.addWidget(self.trplLogChk, 5, 7)

        # row 6: alignment controls
        self.alignChk = QtWidgets.QCheckBox("Align t=0"); self.alignChk.setChecked(True)
        self.smoothSpin = QtWidgets.QSpinBox()
        self.smoothSpin.setRange(1, 500); self.smoothSpin.setValue(3); self.smoothSpin.setPrefix("Smooth "); self.smoothSpin.setFixedWidth(110)
        self.noiseSpin = QtWidgets.QDoubleSpinBox()
        self.noiseSpin.setDecimals(4); self.noiseSpin.setRange(0.0, 1e9); self.noiseSpin.setValue(0.05); self.noiseSpin.setPrefix("Noise "); self.noiseSpin.setFixedWidth(120)
        self.magnSpin = QtWidgets.QDoubleSpinBox()
        self.magnSpin.setDecimals(3); self.magnSpin.setRange(0.001, 1000.0); self.magnSpin.setValue(1.0); self.magnSpin.setPrefix("M "); self.magnSpin.setFixedWidth(100)
        self.spin_sigma_smooth = QtWidgets.QSpinBox()
        self.spin_sigma_smooth.setRange(1, 200); self.spin_sigma_smooth.setValue(1); self.spin_sigma_smooth.setPrefix("σ² smooth "); self.spin_sigma_smooth.setFixedWidth(120)
        self.sigmaTypeCombo = QtWidgets.QComboBox()
        self.sigmaTypeCombo.addItems(["σ_eq²", "σ_x²", "σ_y²"])
        self.sigmaTypeCombo.setFixedWidth(80)
        self.sigmaTypeCombo.setToolTip("Choose which σ² to display in Gaussian Fit, Space-Time Map, and σ² Traces tabs")
        grid.addWidget(self.alignChk, 6, 0)
        grid.addWidget(self.smoothSpin, 6, 1); grid.addWidget(self.noiseSpin, 6, 2)
        grid.addWidget(self.magnSpin, 6, 3); grid.addWidget(self.spin_sigma_smooth, 6, 4)
        grid.addWidget(QtWidgets.QLabel("Display:"), 6, 5); grid.addWidget(self.sigmaTypeCombo, 6, 6)

        # row 7: aligned bin slider
        grid.addWidget(QtWidgets.QLabel("Aligned bin:"), 7, 0)
        self.alignedBinSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alignedBinSlider.setRange(0, 0); self.alignedBinSlider.setTracking(True)
        grid.addWidget(self.alignedBinSlider, 7, 1, 1, 7)
        self.alignedTimeLabel = QtWidgets.QLabel("t = -- ns")
        self.alignedTimeLabel.setMinimumWidth(160)
        grid.addWidget(self.alignedTimeLabel, 7, 8, 1, 2)

        # row 8: Gaussian fit controls
        self.trplFitEndSpin = QtWidgets.QDoubleSpinBox()
        self.trplFitEndSpin.setDecimals(4); self.trplFitEndSpin.setRange(0.0, 1e9); self.trplFitEndSpin.setValue(100.0); self.trplFitEndSpin.setFixedWidth(100)
        self.fitCurrentBtn = QtWidgets.QPushButton("Fit Current Map"); self.fitCurrentBtn.setFixedWidth(BTN_W + 20)
        self.fitAllBtn = QtWidgets.QPushButton("Fit All Maps"); self.fitAllBtn.setFixedWidth(BTN_W)
        self.btnExportListedHist   = QtWidgets.QPushButton("Export Listed Histograms")
        self.btnExportHistMap      = QtWidgets.QPushButton("Export Histogram Map")
        self.btnExportThisSigma2   = QtWidgets.QPushButton("Export This σ²")
        self.btnExportListedSigma2 = QtWidgets.QPushButton("Export Listed σ²")
        self.btnExportSigma2Map    = QtWidgets.QPushButton("Export σ² Map")
        self.btnExportThisSigma2.setEnabled(False)
        self.btnExportListedSigma2.setEnabled(False)
        self.btnExportSigma2Map.setEnabled(False)
        grid.addWidget(QtWidgets.QLabel("Fit end (ns):"), 8, 0)
        grid.addWidget(self.trplFitEndSpin, 8, 1)
        grid.addWidget(self.fitCurrentBtn, 8, 2)
        grid.addWidget(self.fitAllBtn, 8, 3)
        grid.addWidget(self.btnExportListedHist,   9, 0, 1, 2)
        grid.addWidget(self.btnExportHistMap,      9, 2, 1, 2)
        grid.addWidget(self.btnExportThisSigma2,   9, 4)
        grid.addWidget(self.btnExportListedSigma2, 9, 5)
        grid.addWidget(self.btnExportSigma2Map,    9, 6)

        root.addWidget(controls)

        # ── Three-pane splitter ────────────────────────────────────────────────
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # left pane
        self.leftTabs = QtWidgets.QTabWidget()
        self.liveView = PlImageWidget(title="PL Image", default_linecut_width=self.defaults.linecut_width)
        self.leftTabs.addTab(self.liveView, "PL Image")
        self.spadMapPanel = SpadMapWidget()
        self.leftTabs.addTab(self.spadMapPanel, "SPAD Map")
        split.addWidget(self.leftTabs)

        # middle pane
        self.middleTabs = QtWidgets.QTabWidget()
        self.mapView = MapPlotWidgetV2("Linecut map", cmap_name="viridis", line_label="I")
        self.middleTabs.addTab(self.mapView, "PL Linecut Map")
        self.trplHistWidget = TrplHistogramWidget()
        self.middleTabs.addTab(self.trplHistWidget, "TRPL Histogram")
        self.trplMapPanel = TrplMapWidget()
        self.middleTabs.addTab(self.trplMapPanel, "TRPL Map")
        self.historyPanel = HistoryWidget()
        self.middleTabs.addTab(self.historyPanel, "Last Five Histograms")
        self.gaussianPanel = GaussianFitWidget()
        self.middleTabs.addTab(self.gaussianPanel, "Gaussian Fit")
        self.spaceTimePanel = SpaceTimeMapWidget()
        self.middleTabs.addTab(self.spaceTimePanel, "Space-Time Map")
        self.sigmaTracesPanel = SigmaTraceOverlayWidget()
        self.middleTabs.addTab(self.sigmaTracesPanel, "σ² Traces")
        split.addWidget(self.middleTabs)

        # right pane
        self.analysisTabs = QtWidgets.QTabWidget()
        self.derivView = MapPlotWidgetV2("Derivative map", cmap_name="coolwarm", line_label="dI/dx")
        self.analysisTabs.addTab(self.derivView, "Derivative")
        self.fitWidthView = FitTrendPlotWidgetV2("Lorentzian Width vs Sweep", "FWHM (meV)")
        self.analysisTabs.addTab(self.fitWidthView, "Fit Width")
        self.fitEnergyView = FitTrendPlotWidgetV2("Peak Position vs Sweep", "Energy (eV)")
        self.analysisTabs.addTab(self.fitEnergyView, "Peak Energy")
        self.fitIntensityView = FitTrendPlotWidgetV2("Peak Intensity vs Sweep", "Fitted Intensity (a.u.)")
        self.analysisTabs.addTab(self.fitIntensityView, "Peak Intensity")
        split.addWidget(self.analysisTabs)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

        self.setCentralWidget(central)
        self._status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self._status_bar)

        # ── Signal wiring ──────────────────────────────────────────────────────
        self.browseFolderBtn.clicked.connect(self._on_browse_folder)
        self.loadFolderBtn.clicked.connect(self._on_load_folder)
        self.folderEdit.returnPressed.connect(self._on_load_folder)
        self.modeCombo.currentTextChanged.connect(self._on_mode_changed)
        self.mainSlider.valueChanged.connect(self._on_main_changed)
        self.fixedSlider.valueChanged.connect(self._on_fixed_changed)
        self.cropApplyBtn.clicked.connect(self._on_apply_crop)
        self.reverseWlCheck.toggled.connect(self._on_reverse_wl_toggled)
        self.derivMaxSpin.valueChanged.connect(self._on_deriv_scale_changed)
        self.peak1GuessEdit.editingFinished.connect(self._refresh_all)
        self.peak2GuessEdit.editingFinished.connect(self._refresh_all)
        self.fitPeaksBtn.clicked.connect(self._on_fit_peaks_clicked)
        self.liveView.linecut_changed.connect(lambda _: self._refresh_all())
        self.liveView.linecutWidthSpin.valueChanged.connect(lambda _: self._refresh_all())
        self.trplTminSpin.valueChanged.connect(lambda _: self._refresh_trpl_views())
        self.trplTmaxSpin.valueChanged.connect(lambda _: self._refresh_trpl_views())
        self.histYminEdit.editingFinished.connect(self._update_history_plot)
        self.histYmaxEdit.editingFinished.connect(self._update_history_plot)
        self.trplNormalizeChk.toggled.connect(lambda _: self._refresh_trpl_views())
        self.trplLogChk.toggled.connect(lambda _: self._refresh_trpl_views())
        self.middleTabs.currentChanged.connect(lambda _: self._refresh_trpl_views())
        self.alignChk.toggled.connect(self._on_align_changed)
        self.smoothSpin.valueChanged.connect(self._on_smooth_changed)
        self.noiseSpin.valueChanged.connect(self._on_noise_changed)
        self.magnSpin.valueChanged.connect(self._on_magnification_changed)
        self.alignedBinSlider.valueChanged.connect(self._on_aligned_changed)
        self.fitCurrentBtn.clicked.connect(self._on_fit_current_map)
        self.fitAllBtn.clicked.connect(self._on_fit_all_maps)
        self.trplFitEndSpin.valueChanged.connect(lambda _: self._invalidate_sigma2())
        self.btnExportListedHist.clicked.connect(self._export_listed_hist_csv)
        self.btnExportHistMap.clicked.connect(self._export_hist_map_csv)
        self.btnExportThisSigma2.clicked.connect(self._export_this_sigma2_csv)
        self.btnExportListedSigma2.clicked.connect(self._export_listed_sigma2_csv)
        self.btnExportSigma2Map.clicked.connect(self._export_sigma2_map_csv)
        self.spadMapPanel.pixel_clicked.connect(self._on_spad_pixel_clicked)
        self.historyPanel.condEdit.editingFinished.connect(self._update_history_overlay)
        self.gaussianPanel.extraTimesEdit.editingFinished.connect(
            lambda: self._update_gaussian() if
            self.middleTabs.tabText(self.middleTabs.currentIndex()) == "Gaussian Fit" else None)
        self.sigmaTypeCombo.currentTextChanged.connect(self._on_sigma_type_changed)


    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_bar.showMessage(str(msg))

    def _default_trpl_tmax(self) -> float:
        for rec in self.trpl_records:
            if rec.time_ns.size:
                return float(np.nanmax(rec.time_ns))
        return 100.0

    @staticmethod
    def _parse_optional_edit(edit: QtWidgets.QLineEdit) -> Optional[float]:
        txt = edit.text().strip()
        if not txt:
            return None
        try:
            val = float(txt)
        except Exception:
            return None
        return val if np.isfinite(val) else None

    def _format_selection_label(self, p: float, vf: float, pix: int) -> str:
        p_uW = p * 1e6
        if self._can_power_mode and self._can_gate_mode:
            return f"P={p_uW:.4g} uW, Vf={vf:.4g} V, px {pix}"
        if self._can_power_mode:
            return f"P={p_uW:.4g} uW, px {pix}"
        if self._can_gate_mode:
            return f"Vf={vf:.4g} V, px {pix}"
        return f"px {pix}"

    def _trpl_time_bounds(self) -> Tuple[float, float]:
        tmin = float(self.trplTminSpin.value())
        tmax = float(self.trplTmaxSpin.value())
        if tmax < tmin:
            tmin, tmax = tmax, tmin
        return tmin, tmax

    def _choose_export_path(self, default_name: str) -> Optional[Path]:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", str(self.data_dir / default_name),
            "CSV files (*.csv);;All files (*)")
        if not path:
            return None
        out = Path(path)
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        return out

    # ── Folder loading ─────────────────────────────────────────────────────────

    def _on_browse_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select PL/TRPL data folder",
            str(Path(self.folderEdit.text()).expanduser()
                if self.folderEdit.text().strip() else self.data_dir))
        if not folder:
            return
        self.folderEdit.setText(folder)
        self._load_data_dir(Path(folder))

    def _on_load_folder(self) -> None:
        self._load_data_dir(Path(self.folderEdit.text().strip()).expanduser())

    def _load_data_dir(self, data_dir: Path) -> None:
        data_dir = data_dir.resolve()
        if not data_dir.is_dir():
            QtWidgets.QMessageBox.warning(self, "Load", f"Not a directory:\n{data_dir}")
            return
        try:
            frames, defaults = load_frames(data_dir)
            trpl_records = load_trpl_records(data_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load", f"Failed:\n{exc}")
            return
        if not frames:
            QtWidgets.QMessageBox.warning(self, "Load", f"No .asc PL files in:\n{data_dir}")
            return
        self._initializing = True
        try:
            self.data_dir = data_dir
            self.frames = frames
            self.trpl_records = trpl_records
            self.defaults = defaults
            self._crop = tuple(defaults.crop)
            self._cropped_image_cache.clear()
            self._cropped_wl_cache.clear()
            self._fit_result = None
            self._fit_map_version = -1
            self._map_version = 0
            self._selection_history.clear()
            self._main_idx = 0
            self._fixed_idx = 0
            self._align_result = None
            self._align_cache.clear()
            self._sigma2_cache = None
            self._diffusion_data = None
            self._fit_current_data = None
            self._current_aligned_idx = 0
            for spin, value in zip(
                    (self.cropTopSpin, self.cropBottomSpin, self.cropLeftSpin, self.cropRightSpin),
                    self._crop):
                spin.blockSignals(True); spin.setValue(int(value)); spin.blockSignals(False)
            self.trplTmaxSpin.blockSignals(True)
            self.trplTmaxSpin.setValue(self._default_trpl_tmax())
            self.trplFitEndSpin.setValue(self._default_trpl_tmax())
            self.trplTmaxSpin.blockSignals(False)
            self._build_grid()
            self.folderEdit.setText(str(data_dir))
            self.folderStatusLbl.setText(f"{len(frames)} PL, {len(trpl_records)} TRPL")
            self._init_mode()
            self._on_apply_crop()
        finally:
            self._initializing = False
        self._record_selection()
        self._refresh_trpl_views()
        self._set_status(f"Loaded {data_dir}")

    # ── Grid ───────────────────────────────────────────────────────────────────

    def _build_grid(self) -> None:
        power_keys: Set[float] = set()
        gate_keys: Set[float] = set()
        back_by_gate: Dict[float, List[float]] = {}
        for fr in list(self.frames) + list(self.trpl_records):
            p = normalize_key(fr.power_w)
            vf = normalize_key(fr.v_front)
            vb = normalize_key(fr.v_back)
            if p is not None:
                power_keys.add(p)
            if vf is not None:
                gate_keys.add(vf)
                if vb is not None:
                    back_by_gate.setdefault(vf, []).append(float(vb))
        if not power_keys:
            power_keys = {0.0}
        if not gate_keys:
            gate_keys = {0.0}
        self._powers = np.array(sorted(power_keys), dtype=float)
        self._gates = np.array(sorted(gate_keys), dtype=float)
        self._default_power_key = normalize_key(float(self._powers[0]))
        self._default_gate_key = normalize_key(float(self._gates[0]))
        self._grid = {}
        for fr in self.frames:
            k = (self._norm_power_key(fr.power_w), self._norm_gate_key(fr.v_front))
            old = self._grid.get(k)
            if old is None or fr.mtime > old.mtime:
                self._grid[k] = fr
        self._trpl_grid = {}
        for rec in self.trpl_records:
            k = (self._norm_power_key(rec.power_w), self._norm_gate_key(rec.v_front))
            old = self._trpl_grid.get(k)
            if old is None or rec.mtime > old.mtime:
                self._trpl_grid[k] = rec
        self._can_power_mode = self._powers.size > 1
        self._can_gate_mode = self._gates.size > 1
        backs = np.full(self._gates.shape, np.nan, dtype=float)
        for i, vf in enumerate(self._gates):
            vals = back_by_gate.get(normalize_key(vf), [])
            if vals:
                backs[i] = float(np.median(vals))
        self._backs_for_gate = backs
        ref = self.frames[0] if self.frames else None
        if ref is not None:
            self._blank_image = np.zeros_like(ref.image, dtype=float)
            self._blank_wl = (np.asarray(ref.wavelength_nm, dtype=float).copy()
                              if ref.wavelength_nm is not None
                              else np.arange(ref.image.shape[1], dtype=float))
        else:
            self._blank_image = np.zeros((16, 16), dtype=float)
            self._blank_wl = np.arange(16, dtype=float)

    def _norm_power_key(self, p: Optional[float]) -> float:
        k = normalize_key(p)
        return self._default_power_key if k is None else float(k)

    def _norm_gate_key(self, vf: Optional[float]) -> float:
        k = normalize_key(vf)
        return self._default_gate_key if k is None else float(k)

    # ── Mode / slider management ───────────────────────────────────────────────

    def _init_mode(self) -> None:
        if self._can_power_mode and not self._can_gate_mode:
            self._current_mode = self.MODE_POWER
            self.modeCombo.setCurrentText(self.MODE_POWER)
            self.modeCombo.setEnabled(False)
        elif self._can_gate_mode and not self._can_power_mode:
            self._current_mode = self.MODE_GATE
            self.modeCombo.setCurrentText(self.MODE_GATE)
            self.modeCombo.setEnabled(False)
        elif self._can_gate_mode and self._can_power_mode:
            self._current_mode = self.modeCombo.currentText()
            self.modeCombo.setEnabled(True)
        else:
            self._current_mode = self.MODE_POWER
            self.modeCombo.setCurrentText(self.MODE_POWER)
            self.modeCombo.setEnabled(False)
        self._configure_sliders_for_mode()

    def _axis_for_mode(self) -> Tuple[np.ndarray, np.ndarray, str, str]:
        if self._current_mode == self.MODE_GATE:
            return self._gates, self._powers, "Front gate (V)", "Power"
        return self._powers * 1e6, self._gates, "Power (uW)", "Front gate"

    def _configure_sliders_for_mode(self) -> None:
        x, fixed, _, fixed_name = self._axis_for_mode()
        for s, arr, idx_attr in (
                (self.mainSlider, x, "_main_idx"),
                (self.fixedSlider, fixed, "_fixed_idx"),
                (self.refIndexSpin, x, "_main_idx")):
            s.blockSignals(True)
            s.setRange(0, max(0, arr.size - 1))
            cur = int(np.clip(getattr(self, idx_attr), 0, max(0, arr.size - 1)))
            s.setValue(cur)
            s.blockSignals(False)
        self._main_idx = int(np.clip(self._main_idx, 0, max(0, x.size - 1)))
        self._fixed_idx = int(np.clip(self._fixed_idx, 0, max(0, fixed.size - 1)))
        self.fixedValueLbl.setText(f"{fixed_name}: --")
        self._refresh_all()

    # ── Key / frame lookup ─────────────────────────────────────────────────────

    def _get_key_for_display(self) -> Tuple[Optional[float], Optional[float]]:
        p = self._default_power_key
        vf = self._default_gate_key
        if self._current_mode == self.MODE_GATE:
            if self._powers.size:
                p = self._norm_power_key(float(self._powers[self._fixed_idx]))
            if self._gates.size:
                vf = self._norm_gate_key(float(self._gates[self._main_idx]))
        else:
            if self._powers.size:
                p = self._norm_power_key(float(self._powers[self._main_idx]))
            if self._gates.size:
                vf = self._norm_gate_key(float(self._gates[self._fixed_idx]))
        return p, vf

    def _get_frame_for_display(self) -> Optional[PLFrame]:
        key = self._get_key_for_display()
        fr = self._grid.get(key)
        if fr is not None:
            return fr
        if self._current_mode == self.MODE_POWER:
            p = key[0]
            cands = [(k, v) for k, v in self._grid.items() if k[0] == p]
            if not cands:
                return None
            tv = float(self._gates[self._fixed_idx]) if self._gates.size else 0.0
            cands.sort(key=lambda kv: abs(float(kv[0][1]) - tv))
            return cands[0][1]
        vf = key[1]
        cands = [(k, v) for k, v in self._grid.items() if k[1] == vf]
        if not cands:
            return None
        tp = float(self._powers[self._fixed_idx]) if self._powers.size else 0.0
        cands.sort(key=lambda kv: abs(float(kv[0][0]) - tp))
        return cands[0][1]

    def _get_trpl_record_for_display(self) -> Optional[TRPLRecord]:
        key = self._get_key_for_display()
        rec = self._trpl_grid.get(key)
        if rec is not None:
            return rec
        if self._current_mode == self.MODE_POWER:
            p = key[0]
            cands = [(k, v) for k, v in self._trpl_grid.items() if k[0] == p]
            if not cands:
                return None
            tv = float(self._gates[self._fixed_idx]) if self._gates.size else 0.0
            cands.sort(key=lambda kv: abs(float(kv[0][1]) - tv))
            return cands[0][1]
        vf = key[1]
        cands = [(k, v) for k, v in self._trpl_grid.items() if k[1] == vf]
        if not cands:
            return None
        tp = float(self._powers[self._fixed_idx]) if self._powers.size else 0.0
        cands.sort(key=lambda kv: abs(float(kv[0][0]) - tp))
        return cands[0][1]

    def _get_trpl_record_for_key(self, power_key: float, gate_key: float) -> Optional[TRPLRecord]:
        return self._trpl_grid.get((float(power_key), float(gate_key)))

    def _current_selection_key(self) -> Tuple[float, float, int]:
        p, vf = self._get_key_for_display()
        return (float(p if p is not None else 0.0),
                float(vf if vf is not None else 0.0),
                int(self._selected_spad_pixel))

    def _record_selection(self) -> None:
        if getattr(self, "_initializing", False):
            return
        item = self._current_selection_key()
        if self._selection_history and self._selection_history[-1] == item:
            return
        self._selection_history.append(item)
        self._selection_history = self._selection_history[-5:]
        self._update_history_plot()

    # ── Crop helpers ───────────────────────────────────────────────────────────

    def _cropped_frame_data(self, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        ci = self._cropped_image_cache.get(fr.path)
        cw = self._cropped_wl_cache.get(fr.path)
        if ci is not None and cw is not None:
            return ci, cw
        img = apply_crop(fr.image, self._crop)
        wl = fr.wavelength_nm if fr.wavelength_nm is not None else np.arange(fr.image.shape[1], dtype=float)
        wl_c = crop_axis(wl, self._crop, fr.image.shape[1])
        if self.reverseWlCheck.isChecked():
            img = img[:, ::-1]
        self._cropped_image_cache[fr.path] = img
        self._cropped_wl_cache[fr.path] = wl_c
        return img, wl_c

    def _blank_cropped(self) -> Tuple[np.ndarray, np.ndarray]:
        blank = apply_crop(self._blank_image, self._crop)
        wl = crop_axis(self._blank_wl, self._crop, int(self._blank_image.shape[1]))
        return blank, wl

    def _on_apply_crop(self) -> None:
        self._crop = (int(self.cropTopSpin.value()), int(self.cropBottomSpin.value()),
                      int(self.cropLeftSpin.value()), int(self.cropRightSpin.value()))
        self._cropped_image_cache.clear()
        self._cropped_wl_cache.clear()
        self.liveView.set_crop(*self._crop)
        self._refresh_all()

    def _on_reverse_wl_toggled(self) -> None:
        self._cropped_image_cache.clear()
        self._cropped_wl_cache.clear()
        self._refresh_all()

    def _current_linecut_row_raw(self, shape_hw: Tuple[int, int]) -> int:
        h = int(shape_hw[0])
        row = self.liveView.linecut_row()
        if row is None:
            return h // 2
        row_d = int(np.clip(row, 0, h - 1))
        return h - 1 - row_d  # display row 0=top → raw array row

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._update_shared_image()
        self._update_maps()
        self._refresh_trpl_views()

    def _refresh_trpl_views(self) -> None:
        rec = self._get_trpl_record_for_display()
        self._sync_aligned_slider(rec)
        self._update_spad_map()
        tab = self.middleTabs.tabText(self.middleTabs.currentIndex())
        if tab == "TRPL Histogram":
            self._update_histogram()
        elif tab == "TRPL Map":
            self._update_trpl_map()
        elif tab == "Last Five Histograms":
            self._update_history_plot()
            self._update_history_overlay()
        elif tab == "Gaussian Fit":
            self._update_gaussian()

    def _update_shared_image(self) -> None:
        def _fmt(v, spec=".4g"):
            if v is None:
                return "na"
            try:
                fv = float(v)
            except Exception:
                return "na"
            return "na" if not np.isfinite(fv) else format(fv, spec)

        fr = self._get_frame_for_display()
        if fr is None:
            blank_c = apply_crop(self._blank_image, self._crop)
            blank_wl_c = crop_axis(self._blank_wl, self._crop, self._blank_image.shape[1])
            if self.reverseWlCheck.isChecked():
                blank_c = blank_c[:, ::-1]
            self.liveView.set_wavelength_axis(blank_wl_c)
            self.liveView.update_frame({"image": blank_c})
            self.liveView.set_image_title("PL Image (missing)")
            self.mainValueLbl.setText("Missing frame")
            return
        # Display the CROPPED image so that liveView row coords match map row coords.
        img_c, wl_c = self._cropped_frame_data(fr)
        self.liveView.set_wavelength_axis(wl_c)
        self.liveView.update_frame({"image": img_c})
        p_uW = float(fr.power_w * 1e6) if fr.power_w is not None else np.nan
        self.liveView.set_image_title(
            f"PL Image (P={_fmt(p_uW, '.3f')} uW, Vf={_fmt(fr.v_front)} V, Vb={_fmt(fr.v_back)} V)")
        if self._current_mode == self.MODE_POWER:
            self.mainValueLbl.setText(f"Power: {_fmt(p_uW, '.3f')} uW")
            fv = float(self._gates[self._fixed_idx]) if self._gates.size else np.nan
            fb = (float(self._backs_for_gate[self._fixed_idx])
                  if self._backs_for_gate.size and self._fixed_idx < self._backs_for_gate.size
                  else np.nan)
            self.fixedValueLbl.setText(f"Front gate: {_fmt(fv)} V (Back ~ {_fmt(fb)} V)")
        else:
            self.mainValueLbl.setText(f"Front gate: {_fmt(fr.v_front)} V (Back {_fmt(fr.v_back)} V)")
            fp = float(self._powers[self._fixed_idx] * 1e6) if self._powers.size else np.nan
            self.fixedValueLbl.setText(f"Power: {_fmt(fp, '.3f')} uW")

    # ── Alignment ──────────────────────────────────────────────────────────────

    def _align_cache_key(self, rec: TRPLRecord) -> tuple:
        return (str(rec.path), self._smooth_window, self._noise_level, self._align_t0)

    def _apply_alignment(self, rec: TRPLRecord) -> AlignResult:
        key = self._align_cache_key(rec)
        if key in self._align_cache:
            return self._align_cache[key]
        ar = compute_alignment(rec.time_ns, rec.counts,
                               self._smooth_window, self._noise_level, self._align_t0)
        self._align_cache[key] = ar
        return ar

    def _sync_aligned_slider(self, rec: Optional[TRPLRecord]) -> None:
        if rec is None or rec.counts.size == 0:
            self.alignedBinSlider.blockSignals(True)
            self.alignedBinSlider.setRange(0, 0)
            self.alignedBinSlider.setValue(0)
            self.alignedBinSlider.blockSignals(False)
            self.alignedTimeLabel.setText("t = --")
            return
        ar = self._apply_alignment(rec)
        self._align_result = ar
        max_idx = ar.n_aligned_bins - 1
        val = int(np.clip(self._current_aligned_idx, 0, max_idx))
        self.alignedBinSlider.blockSignals(True)
        self.alignedBinSlider.setRange(0, max_idx)
        self.alignedBinSlider.setValue(val)
        self.alignedBinSlider.blockSignals(False)
        self._current_aligned_idx = val
        self._update_aligned_label()

    def _update_aligned_label(self) -> None:
        ar = self._align_result
        if ar is None:
            self.alignedTimeLabel.setText("t = --")
            return
        k = min(self._current_aligned_idx, ar.n_aligned_bins - 1)
        self.alignedTimeLabel.setText(f"t' = {ar.aligned_time_ns[k]:.4g} ns | px {self._selected_spad_pixel}")

    def _on_aligned_changed(self, val: int) -> None:
        self._current_aligned_idx = int(val)
        self._update_aligned_label()
        self._update_spad_map()
        tab = self.middleTabs.tabText(self.middleTabs.currentIndex())
        if tab == "TRPL Histogram":
            self._update_histogram()
        elif tab == "Gaussian Fit":
            self._update_gaussian()

    def _recompute_and_refresh(self) -> None:
        self._align_cache.clear()
        self._align_result = None
        self._refresh_trpl_views()

    def _invalidate_sigma2(self) -> None:
        self._sigma2_cache = None
        self._fit_current_data = None

    def _on_align_changed(self, state) -> None:
        self._align_t0 = bool(state)
        self._recompute_and_refresh()

    def _on_smooth_changed(self, val: int) -> None:
        self._smooth_window = max(1, int(val))
        self._recompute_and_refresh()

    def _on_noise_changed(self, val: float) -> None:
        self._noise_level = float(val)
        self._recompute_and_refresh()

    def _on_magnification_changed(self, val: float) -> None:
        self._magnification = max(1e-9, float(val))
        self.gaussianPanel.set_magnification(self._magnification)
        self.spaceTimePanel.set_magnification(self._magnification)
        tab = self.middleTabs.tabText(self.middleTabs.currentIndex())
        if tab == "Gaussian Fit":
            self._update_gaussian()
        elif tab == "Space-Time Map":
            if self._fit_current_data is not None:
                d = self._fit_current_data
                self.spaceTimePanel.update(
                    d['t'], d['popt_list'], d['r_max_det'],
                    sigma2_det=self._get_sigma2_for_display(d))

    # ── SPAD map ───────────────────────────────────────────────────────────────

    def _update_spad_map(self) -> None:
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            self.spadMapPanel.update(np.zeros(NUM_SPAD, dtype=float),
                                     "t = --", {self._selected_spad_pixel})
            return
        ar = self._apply_alignment(rec)
        k = min(self._current_aligned_idx, ar.n_aligned_bins - 1)
        frame = ar.aligned_counts[k, :].astype(float)
        t_ns = float(ar.aligned_time_ns[k])
        self.spadMapPanel.update(frame, f"t' = {t_ns:.4g} ns", {self._selected_spad_pixel})

    def _on_spad_pixel_clicked(self, pix: int) -> None:
        self._selected_spad_pixel = int(pix)
        self._record_selection()
        self._refresh_trpl_views()

    # ── PL maps ────────────────────────────────────────────────────────────────

    def _update_maps(self) -> None:
        fr = self._get_frame_for_display()
        if fr is None:
            shape = apply_crop(self._blank_image, self._crop).shape
            row_raw = self._current_linecut_row_raw(shape)
        else:
            img_c, _ = self._cropped_frame_data(fr)
            row_raw = self._current_linecut_row_raw(img_c.shape)
        width = int(self.liveView.linecut_width())
        if self._current_mode == self.MODE_GATE:
            x, y_nm, m_nm, d_nm = self._build_map_for_gate_mode(row_raw, width)
            self.mapView.set_secondary_axis(self._backs_for_gate, label="Back gate (V)")
            self.derivView.set_secondary_axis(self._backs_for_gate, label="Back gate (V)")
            self.mapView.setTitle("Linecut vs Front Gate")
            self.derivView.setTitle("dI/dVf")
            self.derivView._line_label = "dI/dVf"
            x_label = "Front gate (V)"
            marker = float(self._gates[self._main_idx]) if self._gates.size else 0.0
        else:
            x, y_nm, m_nm, d_nm = self._build_map_for_power_mode(row_raw, width)
            self.mapView.set_secondary_axis(None)
            self.derivView.set_secondary_axis(None)
            self.mapView.setTitle("Linecut vs Power")
            self.derivView.setTitle("dI/dlog10P")
            self.derivView._line_label = "dI/dlog10P"
            x_label = "Power (uW)"
            marker = float(self._powers[self._main_idx] * 1e6) if self._powers.size else 0.0
        self._map_version += 1
        self._last_x = np.asarray(x, dtype=float).copy()
        self._last_x_label = str(x_label)
        self._last_y_wl = np.asarray(y_nm, dtype=float).copy()
        self._last_map_data = np.asarray(m_nm, dtype=float).copy()
        y_energy, m_energy = spectral_map_nm_to_energy(y_nm, m_nm)
        _, d_energy = spectral_map_nm_to_energy(y_nm, d_nm)
        y_label = "Energy (eV)"
        max_abs = float(self.derivMaxSpin.value())
        self.derivView.set_manual_symmetric_max(max_abs if max_abs > 0 else None)
        self.mapView.set_map(x, y_energy, m_energy, x_label=x_label, y_label=y_label, symmetric=False)
        self.derivView.set_map(x, y_energy, d_energy, x_label=x_label, y_label=y_label, symmetric=True)
        self.mapView.set_x_marker(marker)
        self.derivView.set_x_marker(marker)
        self._update_fit_trend_tabs(x, x_label, y_nm, m_nm)
        self._update_live_linecut_fit_overlay()

    def _build_map_for_power_mode(self, row_raw: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_uW = self._powers * 1e6
        fv = self._norm_gate_key(float(self._gates[self._fixed_idx])) if self._gates.size else self._default_gate_key
        y_ref = None
        for p in self._powers:
            fr = self._grid.get((self._norm_power_key(float(p)), fv))
            if fr is not None:
                _, wl = self._cropped_frame_data(fr)
                y_ref = np.asarray(wl, dtype=float)
                break
        if y_ref is None:
            _, wl = self._blank_cropped()
            y_ref = np.asarray(wl, dtype=float)
        cols = []
        for p in self._powers:
            fr = self._grid.get((self._norm_power_key(float(p)), fv))
            if fr is None:
                cols.append(np.full(y_ref.shape, np.nan, dtype=float))
                continue
            img, wl = self._cropped_frame_data(fr)
            lc = linecut_horizontal(img, row_raw, width)
            cols.append(np.asarray(resample_linecut(wl, lc, y_ref), dtype=float)
                        if lc is not None else np.full(y_ref.shape, np.nan, dtype=float))
        m = np.stack(cols, axis=1) if cols else np.full((y_ref.size, x_uW.size), np.nan, dtype=float)
        d = derivative_dlogp_nan(m, self._powers)
        return x_uW, y_ref, m, d

    def _build_map_for_gate_mode(self, row_raw: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_vf = self._gates
        fp = self._norm_power_key(float(self._powers[self._fixed_idx])) if self._powers.size else self._default_power_key
        y_ref = None
        for vf in self._gates:
            fr = self._grid.get((fp, self._norm_gate_key(float(vf))))
            if fr is not None:
                _, wl = self._cropped_frame_data(fr)
                y_ref = np.asarray(wl, dtype=float)
                break
        if y_ref is None:
            _, wl = self._blank_cropped()
            y_ref = np.asarray(wl, dtype=float)
        cols = []
        for vf in self._gates:
            fr = self._grid.get((fp, self._norm_gate_key(float(vf))))
            if fr is None:
                cols.append(np.full(y_ref.shape, np.nan, dtype=float))
                continue
            img, wl = self._cropped_frame_data(fr)
            lc = linecut_horizontal(img, row_raw, width)
            cols.append(np.asarray(resample_linecut(wl, lc, y_ref), dtype=float)
                        if lc is not None else np.full(y_ref.shape, np.nan, dtype=float))
        m = np.stack(cols, axis=1) if cols else np.full((y_ref.size, x_vf.size), np.nan, dtype=float)
        d = derivative_vs_x_nan(m, x_vf)
        return x_vf, y_ref, m, d

    def _update_live_linecut_fit_overlay(self) -> None:
        fr = self._get_frame_for_display()
        if fr is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        img, wl = self._cropped_frame_data(fr)
        if img is None or wl is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        row_raw = self._current_linecut_row_raw(img.shape)
        linecut = linecut_horizontal(img, row_raw, int(self.liveView.linecut_width()))
        if linecut is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        wl = np.asarray(wl, dtype=float).ravel()
        linecut = np.asarray(linecut, dtype=float).ravel()
        if wl.size != linecut.size or wl.size < 8:
            self.liveView.clear_linecut_fit_overlay()
            return
        p1_g, p2_g = self._read_peak_guesses()
        fit = fit_two_peak_linecut(wl, linecut, peak1_guess_nm=p1_g, peak2_guess_nm=p2_g)
        p1 = fit.get("peak_1")
        p2 = fit.get("peak_2")
        if p1 is None and p2 is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        baseline = np.full_like(wl, float(fit.get("offset", 0.0)), dtype=float)
        g1 = np.zeros_like(wl, dtype=float)
        g2 = np.zeros_like(wl, dtype=float)
        y1 = y2 = None
        if p1 is not None:
            g1 = lorentzian_component(wl, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
            y1 = baseline + g1
        if p2 is not None:
            g2 = lorentzian_component(wl, p2["amplitude"], p2["center_nm"], p2["gamma_nm"])
            y2 = baseline + g2
        self.liveView.set_linecut_fit_overlay(wl, baseline + g1 + g2, y1, y2)

    # ── Fit trend ──────────────────────────────────────────────────────────────

    def _read_peak_guesses(self) -> Tuple[float, float]:
        p1, p2 = 855.0, 835.0
        for edit, attr in ((self.peak1GuessEdit, "p1"), (self.peak2GuessEdit, "p2")):
            try:
                v = float(edit.text().strip())
                if np.isfinite(v):
                    if attr == "p1":
                        p1 = v
                    else:
                        p2 = v
            except Exception:
                pass
        return p1, p2

    def _clear_fit_cache(self) -> None:
        self._fit_result = None
        self._fit_map_version = -1

    def _on_fit_peaks_clicked(self) -> None:
        if self._last_x is None or self._last_y_wl is None or self._last_map_data is None:
            xl = self._last_x_label or "X"
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
                w.clear_data("No map data", x_label=xl)
            return
        p1_g, p2_g = self._read_peak_guesses()
        ref_idx = int(self.refIndexSpin.value())
        max_shift = float(self.maxShiftSpin.value())
        fit = fit_two_peak_map_sequential(
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map_data, dtype=float),
            ref_idx=ref_idx, peak1_guess_nm=p1_g,
            peak2_guess_nm=p2_g, max_shift_nm=max_shift)
        self._fit_result = fit
        self._fit_map_version = int(self._map_version)
        self._update_fit_trend_tabs(
            np.asarray(self._last_x, dtype=float),
            str(self._last_x_label or "X"),
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map_data, dtype=float))

    def _update_fit_trend_tabs(self, x, x_label, y_wl, map_data) -> None:
        if map_data.ndim != 2 or y_wl.size != map_data.shape[0]:
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
                w.clear_data("No data", x_label=x_label)
            return
        if self._fit_result is None or int(self._fit_map_version) != int(self._map_version):
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
                w.clear_data("Click 'Fit Peaks' to run", x_label=x_label)
            return
        fit = self._fit_result
        c1 = np.asarray(fit.get("center_1_nm", []), dtype=float)
        w1 = np.asarray(fit.get("fwhm_1_nm", []), dtype=float)
        a1 = np.asarray(fit.get("amp_1", []), dtype=float)
        c2 = np.asarray(fit.get("center_2_nm", []), dtype=float)
        w2 = np.asarray(fit.get("fwhm_2_nm", []), dtype=float)
        a2 = np.asarray(fit.get("amp_2", []), dtype=float)
        w1_mev = fwhm_nm_to_mev(c1, w1)
        w2_mev = fwhm_nm_to_mev(c2, w2)
        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = EV_NM / c1
            e2 = EV_NM / c2
        e1[~np.isfinite(e1)] = np.nan
        e2[~np.isfinite(e2)] = np.nan
        self.fitWidthView.set_data(x, w1_mev, w2_mev, x_label=x_label,
                                   label_1="Peak 1", label_2="Peak 2 (optional)")
        self.fitEnergyView.set_data(x, e1, e2, x_label=x_label,
                                    label_1="Peak 1", label_2="Peak 2 (optional)")
        self.fitIntensityView.set_data(x, a1, a2, x_label=x_label,
                                       label_1="Peak 1", label_2="Peak 2 (optional)")

    # ── TRPL map / histogram views ─────────────────────────────────────────────

    def _records_along_current_axis(self) -> Tuple[np.ndarray, List[Optional[TRPLRecord]], str]:
        if self._current_mode == self.MODE_GATE:
            p = (self._norm_power_key(float(self._powers[self._fixed_idx]))
                 if self._powers.size else self._default_power_key)
            xvals = np.asarray(self._gates, dtype=float)
            records = [self._get_trpl_record_for_key(float(p), self._norm_gate_key(float(vf)))
                       for vf in xvals]
            return xvals, records, "Front gate (V)"
        vf = (self._norm_gate_key(float(self._gates[self._fixed_idx]))
              if self._gates.size else self._default_gate_key)
        pvals = np.asarray(self._powers, dtype=float)
        records = [self._get_trpl_record_for_key(self._norm_power_key(float(p)), float(vf))
                   for p in pvals]
        return pvals * 1e6, records, "Power (uW)"

    def _update_trpl_map(self) -> None:
        xvals, records, xlabel = self._records_along_current_axis()
        pixel = int(self._selected_spad_pixel)
        tmin, tmax = self._trpl_time_bounds()
        cols, x_used, time_axis = [], [], None
        for x, rec in zip(xvals, records):
            if rec is None or rec.counts.shape[1] <= pixel:
                continue
            t = np.asarray(rec.time_ns, dtype=float)
            y = np.asarray(rec.counts[:, pixel], dtype=float)
            mask = (t >= tmin) & (t <= tmax)
            if not np.any(mask):
                continue
            if time_axis is None:
                time_axis = t[mask]
                col = y[mask]
            else:
                col = np.interp(time_axis, t, y, left=np.nan, right=np.nan)
            cols.append(col)
            x_used.append(float(x))
        if not cols or time_axis is None:
            self.trplMapPanel.update(np.array([]), np.array([]),
                                     np.zeros((0, 0)), xlabel, pixel,
                                     self.trplLogChk.isChecked(),
                                     self.trplNormalizeChk.isChecked())
            return
        raw = np.column_stack(cols)
        self.trplMapPanel.update(time_axis, np.asarray(x_used, dtype=float), raw,
                                 xlabel, pixel,
                                 self.trplLogChk.isChecked(),
                                 self.trplNormalizeChk.isChecked())

    def _update_histogram(self) -> None:
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            self.trplHistWidget.clear()
            return
        ar = self._apply_alignment(rec)
        k = min(self._current_aligned_idx, ar.n_aligned_bins - 1)
        # Pass RAW time/counts so TrplHistogramWidget can apply per-pixel t0 shifts
        # itself (aligned_t = raw_time - t0_ns[pix]); passing pre-aligned data would
        # cause the widget to subtract t0 a second time, shifting traces off-screen.
        self.trplHistWidget.update(
            rec.time_ns, rec.counts, ar.smoothed_matrix, ar.t0_ns,
            float(ar.aligned_time_ns[k]),
            {self._selected_spad_pixel},
            self.trplLogChk.isChecked(), self.trplNormalizeChk.isChecked())

    def _history_series(self) -> List[dict]:
        out = []
        for p, vf, pix in self._selection_history[-5:]:
            rec = self._get_trpl_record_for_key(p, vf)
            if rec is None or rec.counts.shape[1] <= pix:
                continue
            ar = self._apply_alignment(rec)
            # Use per-pixel t0 shift on raw data (same as TrplHistogramWidget)
            t = rec.time_ns.astype(float) - float(ar.t0_ns[pix])
            y = rec.counts[:, pix].astype(float)
            out.append({"time_ns": t, "counts": y,
                        "label": self._format_selection_label(p, vf, pix)})
        return out

    def _update_history_plot(self) -> None:
        series = self._history_series()
        self.historyPanel.update(series,
                                 log_y=self.trplLogChk.isChecked(),
                                 normalize=self.trplNormalizeChk.isChecked())
        ymin = self._parse_optional_edit(self.histYminEdit)
        ymax = self._parse_optional_edit(self.histYmaxEdit)
        if ymin is not None and ymax is not None:
            self.historyPanel._pw.setYRange(ymin, ymax, padding=0)

    def _build_overlay_series(self) -> List[dict]:
        """Return histogram series for conditions typed in historyPanel.condEdit."""
        pix = int(self._selected_spad_pixel)
        cond_vals = self._parse_float_list(self.historyPanel.condEdit.text())
        series = []
        for v in cond_vals:
            if self._current_mode == self.MODE_GATE:
                nearest_key = float(self._gates[int(np.argmin(np.abs(self._gates - v)))])
                fp = (self._norm_power_key(float(self._powers[self._fixed_idx]))
                      if self._powers.size else self._default_power_key)
                rec = self._get_trpl_record_for_key(fp, self._norm_gate_key(nearest_key))
                label = f"Vf={nearest_key:.4g} V, px {pix}"
            else:
                v_w = v * 1e-6
                nearest_key = float(self._powers[int(np.argmin(np.abs(self._powers - v_w)))])
                fv = (self._norm_gate_key(float(self._gates[self._fixed_idx]))
                      if self._gates.size else self._default_gate_key)
                rec = self._get_trpl_record_for_key(self._norm_power_key(nearest_key), fv)
                label = f"P={nearest_key*1e6:.4g} µW, px {pix}"
            if rec is None or rec.counts.shape[1] <= pix:
                continue
            ar = self._apply_alignment(rec)
            t = rec.time_ns.astype(float) - float(ar.t0_ns[pix])
            y = rec.counts[:, pix].astype(float)
            series.append({"time_ns": t, "counts": y, "label": label})
        return series

    def _update_history_overlay(self) -> None:
        series = self._build_overlay_series()
        log_y = self.trplLogChk.isChecked()
        norm = self.trplNormalizeChk.isChecked()
        self.historyPanel.update_overlay(series, log_y, norm)

    # ── Gaussian fit ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_float_list(text: str) -> List[float]:
        vals = []
        for tok in text.replace(";", ",").replace("\n", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        return vals

    def _get_sigma2_for_display(self, fit_data: dict) -> np.ndarray:
        """Return the σ² array (detector µm²) matching the current combo selection."""
        key = {'σ_eq²': 'sigma2', 'σ_x²': 'sigma2_x', 'σ_y²': 'sigma2_y'}.get(
            self.sigmaTypeCombo.currentText(), 'sigma2')
        return fit_data.get(key, fit_data['sigma2'])

    def _on_sigma_type_changed(self, _text: str) -> None:
        sigma_type = self.sigmaTypeCombo.currentText()
        self.gaussianPanel.set_sigma_type(sigma_type)
        self.sigmaTracesPanel.set_sigma_type(sigma_type)
        if self._fit_current_data is not None:
            sigma2_display = self._get_sigma2_for_display(self._fit_current_data)
            self.spaceTimePanel.update(
                self._fit_current_data['t'],
                self._fit_current_data['popt_list'],
                self._fit_current_data['r_max_det'],
                sigma2_det=sigma2_display)
        self._update_gaussian()

    def _update_gaussian(self) -> None:
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            self.gaussianPanel.clear()
            return
        ar = self._apply_alignment(rec)
        self._align_result = ar
        self.gaussianPanel.set_magnification(self._magnification)
        k = min(self._current_aligned_idx, ar.n_aligned_bins - 1)
        current_t = float(ar.aligned_time_ns[k])

        # σ² plot — use stored fit data if available, pick the chosen variant
        t_arr = sigma2 = None
        if self._fit_current_data is not None:
            t_arr = self._fit_current_data['t']
            sigma2 = self._get_sigma2_for_display(self._fit_current_data)
        elif self._sigma2_cache is not None:
            t_arr, s2_eq, s2_x, s2_y = self._sigma2_cache
            chosen = self.sigmaTypeCombo.currentText()
            sigma2 = {'σ_eq²': s2_eq, 'σ_x²': s2_x, 'σ_y²': s2_y}.get(chosen, s2_eq)

        self.gaussianPanel.update_sigma(t_arr, sigma2, current_t,
                                        self.spin_sigma_smooth.value())

        # Radial fit for current time bin
        frame = ar.aligned_smoothed_counts[k, :]
        if self._fit_current_data is not None:
            # Use stored fit at closest time
            t_stored = self._fit_current_data['t']
            popt_list = self._fit_current_data.get('popt_list', [])
            ki = int(np.argmin(np.abs(t_stored - current_t))) if len(t_stored) > 0 else -1
            if ki >= 0 and ki < len(popt_list) and popt_list[ki] is not None:
                from dataclasses import dataclass as _dc
                _popt = popt_list[ki]
                fit_cur = GaussFitResult(popt=_popt,
                                         sigma_eq=float(np.sqrt((_popt[3]**2 + _popt[4]**2)/2)))
            else:
                fit_cur = fit_gaussian_2d(self._spad_fit_x_um, self._spad_fit_y_um, frame)
        else:
            fit_cur = fit_gaussian_2d(self._spad_fit_x_um, self._spad_fit_y_um, frame)

        # Extra traces from user-specified times
        extra_traces = []
        extra_times = self._parse_float_list(self.gaussianPanel.extraTimesEdit.text())
        M = self._magnification
        for t_extra in extra_times[:_N_RADIAL_EXTRA]:
            k_e = int(np.argmin(np.abs(ar.aligned_time_ns - t_extra)))
            t_actual = float(ar.aligned_time_ns[k_e])
            frame_e = ar.aligned_smoothed_counts[k_e, :]
            fit_e = fit_gaussian_2d(self._spad_fit_x_um, self._spad_fit_y_um, frame_e)
            if fit_e is None:
                continue
            popt_e = fit_e.popt
            x0_e, y0_e = float(popt_e[1]), float(popt_e[2])
            r_e = np.sqrt((_FIT_X_UM - x0_e)**2 + (_FIT_Y_UM - y0_e)**2)
            ord_e = np.argsort(r_e)
            r_sorted = r_e[ord_e] / M
            z_sorted = frame_e[ord_e]
            sigma_eq_e = float(np.sqrt((popt_e[3]**2 + popt_e[4]**2) / 2))
            popt_1d = np.array([float(popt_e[0]), sigma_eq_e / M, float(popt_e[5])])
            extra_traces.append((r_sorted, z_sorted, popt_1d, t_actual))

        normalize = False  # no normalize checkbox in this GUI's Gaussian panel
        self.gaussianPanel.update_radial(frame, fit_cur, extra_traces, normalize)
        self.spaceTimePanel.update_time_marker(current_t)

    def _on_fit_current_map(self) -> None:
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            return
        ar = self._apply_alignment(rec)
        self._align_result = ar
        self.fitCurrentBtn.setEnabled(False)
        self._set_status("Fitting all time bins for current condition...")
        t_end = float(self.trplFitEndSpin.value())
        x, y = self._spad_fit_x_um, self._spad_fit_y_um
        t_list: List[float] = []
        s2_list: List[float] = []
        s2x_list: List[float] = []
        s2y_list: List[float] = []
        popt_list: List[np.ndarray] = []
        frame_list: List[np.ndarray] = []
        prev_fit: Optional[GaussFitResult] = None
        failures = 0
        pix11_thresh = int(self.gaussianPanel.spin_pix11_thresh.value())
        for k in range(ar.n_aligned_bins):
            t = float(ar.aligned_time_ns[k])
            if t > t_end:
                break
            frame = ar.aligned_smoothed_counts[k, :]
            if float(frame[11]) < pix11_thresh:
                failures += 1
                continue
            if prev_fit is None:
                fit = fit_gaussian_2d(x, y, frame)
            else:
                fit = _fit_gaussian_2d_constrained(x, y, frame, prev_fit.popt, tightness=1.0)
                if fit is None:
                    fit = _fit_gaussian_2d_constrained(x, y, frame, prev_fit.popt, tightness=1.8)
            if fit is None:
                failures += 1
                continue  # keep prev_fit unchanged — don't break the chain
            prev_fit = fit
            t_list.append(t)
            s2_list.append(float(fit.sigma_eq) ** 2)
            s2x_list.append(float(fit.sigma_x) ** 2)
            s2y_list.append(float(fit.sigma_y) ** 2)
            popt_list.append(np.asarray(fit.popt).copy())
            frame_list.append(frame.copy())
            if k % 50 == 0:
                QtWidgets.QApplication.processEvents()
        if t_list:
            t_arr = np.array(t_list)
            s2_arr = np.array(s2_list)
            s2x_arr = np.array(s2x_list)
            s2y_arr = np.array(s2y_list)
            self._sigma2_cache = (t_arr, s2_arr, s2x_arr, s2y_arr)
            # r_max_det: max pixel distance from fit centroid (detector µm)
            x0_ref, y0_ref = float(popt_list[0][1]), float(popt_list[0][2])
            r_ref = np.sqrt((x - x0_ref)**2 + (y - y0_ref)**2)
            r_max_det = float(np.max(r_ref))
            self._fit_current_data = {
                't': t_arr, 'sigma2': s2_arr, 'sigma2_x': s2x_arr, 'sigma2_y': s2y_arr,
                'popt_list': popt_list, 'r_max_det': r_max_det}
            self.spaceTimePanel.set_magnification(self._magnification)
            sigma2_display = self._get_sigma2_for_display(self._fit_current_data)
            self.spaceTimePanel.update(t_arr, popt_list, r_max_det, sigma2_det=sigma2_display)
        else:
            self._sigma2_cache = None
            self._fit_current_data = None
        self.fitCurrentBtn.setEnabled(True)
        if self._fit_current_data is not None:
            self.btnExportThisSigma2.setEnabled(True)
        for i in range(self.middleTabs.count()):
            if self.middleTabs.tabText(i) == "Gaussian Fit":
                self.middleTabs.setCurrentIndex(i)
                break
        self._update_gaussian()
        self._set_status(f"Fit current map done: {len(t_list)} bins, {failures} failures")

    def _on_fit_all_maps(self) -> None:
        xvals, records, xlabel = self._records_along_current_axis()
        if not any(r is not None for r in records):
            self._set_status("No TRPL records for current sweep")
            return
        self.fitAllBtn.setEnabled(False)
        t_end = float(self.trplFitEndSpin.value())
        x, y = self._spad_fit_x_um, self._spad_fit_y_um
        pix11_thresh = int(self.gaussianPanel.spin_pix11_thresh.value())
        per_cond: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []
        total = len(xvals)
        for i, (xv, rec) in enumerate(zip(xvals, records)):
            self._set_status(f"Fitting condition {i + 1}/{total}...")
            QtWidgets.QApplication.processEvents()
            if rec is None or rec.counts.size == 0:
                per_cond.append(None)
                continue
            ar = self._apply_alignment(rec)
            t_list: List[float] = []
            s2_list: List[float] = []
            s2x_list: List[float] = []
            s2y_list: List[float] = []
            prev_fit: Optional[GaussFitResult] = None
            for k in range(ar.n_aligned_bins):
                t = float(ar.aligned_time_ns[k])
                if t > t_end:
                    break
                frame = ar.aligned_smoothed_counts[k, :]
                if float(frame[11]) < pix11_thresh:
                    continue
                if prev_fit is None:
                    fit = fit_gaussian_2d(x, y, frame)
                else:
                    fit = _fit_gaussian_2d_constrained(x, y, frame, prev_fit.popt, tightness=1.0)
                    if fit is None:
                        fit = _fit_gaussian_2d_constrained(x, y, frame, prev_fit.popt, tightness=1.8)
                if fit is None:
                    continue  # keep prev_fit unchanged — don't break the chain
                prev_fit = fit
                t_list.append(t)
                s2_list.append(float(fit.sigma_eq) ** 2)
                s2x_list.append(float(fit.sigma_x) ** 2)
                s2y_list.append(float(fit.sigma_y) ** 2)
                if k % 50 == 0:
                    QtWidgets.QApplication.processEvents()
            per_cond.append((np.array(t_list), np.array(s2_list),
                             np.array(s2x_list), np.array(s2y_list)) if t_list else None)
        # Update sigma2_cache with current condition
        cur_idx = int(np.clip(self._main_idx, 0, len(per_cond) - 1))
        if per_cond[cur_idx] is not None:
            self._sigma2_cache = per_cond[cur_idx]
        # Build dict for SigmaTraceOverlayWidget
        M2 = self._magnification ** 2
        per_cond_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for j, (xv, res) in enumerate(zip(xvals, per_cond)):
            if res is None:
                continue
            t_arr, s2_arr, s2x_arr, s2y_arr = res
            key = f"{xlabel}={xv:.6g}"
            per_cond_dict[key] = (t_arr, s2_arr / M2, s2x_arr / M2, s2y_arr / M2)
        self._diffusion_data = {"per_cond": per_cond_dict, "x_label": xlabel}
        self.sigmaTracesPanel.set_data(per_cond_dict, x_label=xlabel,
                                       sigma_type=self.sigmaTypeCombo.currentText())
        for i in range(self.middleTabs.count()):
            if self.middleTabs.tabText(i) == "σ² Traces":
                self.middleTabs.setCurrentIndex(i)
                break
        self.fitAllBtn.setEnabled(True)
        self.btnExportListedSigma2.setEnabled(True)
        self.btnExportSigma2Map.setEnabled(True)
        valid = sum(1 for r in per_cond if r is not None)
        self._set_status(f"Fit all done: {valid}/{total} conditions")
        self._update_gaussian()

    # ── Export ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _lbl_to_num_str(lbl: str) -> str:
        """Extract bare number from a label like 'Vf=-1.5 V, px 11' → '-1.5'."""
        m = re.search(r'=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lbl)
        return m.group(1) if m else lbl

    def _x_label_and_val(self) -> Tuple[str, str]:
        """Return (x_label_str, current_xval_str) for the active sweep axis."""
        p, vf = self._get_key_for_display()
        if self._current_mode == self.MODE_GATE:
            return "V_front", f"{vf:.6g}"
        return "Power_uW", f"{p * 1e6:.6g}"

    def _write_hist_wide(self, path: Path, series: List[dict]) -> None:
        """Write histogram series to a wide-format CSV (Origin-compatible)."""
        x_lbl = "V_front" if self._current_mode == self.MODE_GATE else "Power_uW"
        t_ref = np.asarray(series[0]["time_ns"], dtype=float)
        columns = []
        for item in series:
            ta = np.asarray(item["time_ns"], dtype=float)
            ya = np.asarray(item["counts"], dtype=float)
            if len(ta) == len(t_ref) and np.allclose(ta, t_ref, atol=1e-9):
                columns.append(ya.copy())
            else:
                columns.append(np.interp(t_ref, ta, ya, left=np.nan, right=np.nan))
        col_vals = [self._lbl_to_num_str(item["label"]) for item in series]
        with path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([f"time_ns / {x_lbl}"] + col_vals)
            for i, t in enumerate(t_ref):
                row = [f"{t:.12g}"]
                for col in columns:
                    v = col[i]
                    row.append(f"{v:.12g}" if np.isfinite(v) else "")
                wr.writerow(row)

    def _write_sigma2_wide(self, path: Path,
                           per_cond: Dict[str, tuple],
                           x_label: str,
                           sigma_type: str = "σ_eq²") -> None:
        """Write per-condition σ²(t) to a wide-format CSV (Origin-compatible).
        per_cond values are (t, s2_eq, s2_x, s2_y) tuples in sample µm².
        Only the column selected by sigma_type is written."""
        ordered_keys = sorted(
            (k for k, v in per_cond.items() if v is not None and len(v[0]) > 0),
            key=lambda k: (float(k.split("=")[-1]) if "=" in k else 0.0))
        if not ordered_keys:
            return
        col_idx = {'σ_eq²': 1, 'σ_x²': 2, 'σ_y²': 3}.get(sigma_type, 1)
        def _s2(v): return np.asarray(v[col_idx] if len(v) > col_idx else v[1], dtype=float)
        data = {k: (np.asarray(per_cond[k][0], dtype=float), _s2(per_cond[k]))
                for k in ordered_keys}
        all_t = np.unique(np.concatenate([v[0] for v in data.values()]))
        s2_lookup = {k: {float(t): float(s2) for t, s2 in zip(*data[k])}
                     for k in ordered_keys}
        col_vals = [k.split("=")[-1] for k in ordered_keys]
        with path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([f"time_ns / {x_label}"] + col_vals)
            for t in all_t:
                row = [f"{t:.10g}"]
                for k in ordered_keys:
                    row.append(f"{s2_lookup[k][t]:.10g}" if t in s2_lookup[k] else "")
                wr.writerow(row)

    # ── Export actions ──────────────────────────────────────────────────────────

    def _export_listed_hist_csv(self) -> None:
        """Button 1: histograms for conditions in overlay list, or current if empty."""
        series = self._build_overlay_series()
        pix = int(self._selected_spad_pixel)
        if not series:
            rec = self._get_trpl_record_for_display()
            if rec is None or rec.counts.shape[1] <= pix:
                QtWidgets.QMessageBox.warning(self, "Export", "No data to export.")
                return
            ar = self._apply_alignment(rec)
            t = rec.time_ns.astype(float) - float(ar.t0_ns[pix])
            y = rec.counts[:, pix].astype(float)
            _, xval = self._x_label_and_val()
            series = [{"time_ns": t, "counts": y, "label": xval}]
        path = self._choose_export_path("listed_histograms_export.csv")
        if path is None:
            return
        try:
            self._write_hist_wide(path, series)
            self._set_status(f"Exported listed histograms: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed:\n{exc}")

    def _export_hist_map_csv(self) -> None:
        """Button 2: full TRPL histogram map (all conditions × time) for selected pixel."""
        xvals, records, xlabel = self._records_along_current_axis()
        pixel = int(self._selected_spad_pixel)
        tmin, tmax = self._trpl_time_bounds()
        cols, x_used, time_axis = [], [], None
        for x, rec in zip(xvals, records):
            if rec is None or rec.counts.shape[1] <= pixel:
                continue
            t = np.asarray(rec.time_ns, dtype=float)
            y = np.asarray(rec.counts[:, pixel], dtype=float)
            mask = (t >= tmin) & (t <= tmax)
            if not np.any(mask):
                continue
            if time_axis is None:
                time_axis = t[mask]
                col = y[mask]
            else:
                col = np.interp(time_axis, t, y, left=np.nan, right=np.nan)
            cols.append(col)
            x_used.append(float(x))
        if not cols or time_axis is None:
            QtWidgets.QMessageBox.warning(self, "Export Histogram Map", "No data.")
            return
        raw = np.column_stack(cols)
        x_arr = np.asarray(x_used, dtype=float)
        path = self._choose_export_path("histogram_map_export.csv")
        if path is None:
            return
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow([f"time_ns / {xlabel}"] + [f"{v:.10g}" for v in x_arr])
                for i, t in enumerate(time_axis):
                    wr.writerow([f"{t:.10g}"] +
                                [f"{v:.10g}" if np.isfinite(v) else "" for v in raw[i, :]])
            self._set_status(f"Exported histogram map: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Histogram Map", f"Failed:\n{exc}")

    def _sigma_type_tag(self) -> str:
        return {'σ_eq²': 'sigma_eq2', 'σ_x²': 'sigma_x2', 'σ_y²': 'sigma_y2'}.get(
            self.sigmaTypeCombo.currentText(), 'sigma_eq2')

    def _export_this_sigma2_csv(self) -> None:
        """Button 3: σ²(t) for the current single condition (chosen σ² type)."""
        if self._fit_current_data is None and self._sigma2_cache is None:
            QtWidgets.QMessageBox.warning(self, "Export σ²", "Run 'Fit Current Map' first.")
            return
        M2 = self._magnification ** 2
        if self._fit_current_data is not None:
            t_arr = self._fit_current_data['t']
            s2_det = self._get_sigma2_for_display(self._fit_current_data)
        else:
            t_arr, s2_eq, s2_x, s2_y = self._sigma2_cache
            chosen = self.sigmaTypeCombo.currentText()
            s2_det = {'σ_eq²': s2_eq, 'σ_x²': s2_x, 'σ_y²': s2_y}.get(chosen, s2_eq)
        s2_sample = np.asarray(s2_det, dtype=float) / M2
        x_lbl, xval = self._x_label_and_val()
        tag = self._sigma_type_tag()
        path = self._choose_export_path(f"this_{tag}_export.csv")
        if path is None:
            return
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow([f"time_ns / {x_lbl}", xval])
                for t, s2 in zip(t_arr, s2_sample):
                    wr.writerow([f"{float(t):.12g}", f"{float(s2):.12g}"])
            self._set_status(f"Exported this σ²: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export σ²", f"Failed:\n{exc}")

    def _export_listed_sigma2_csv(self) -> None:
        """Button 4: σ²(t) for conditions in σ² Traces list (chosen σ² type)."""
        diff = getattr(self, '_diffusion_data', None)
        txt = self.sigmaTracesPanel.condEdit.text().strip()
        if diff is not None and txt:
            selected = self.sigmaTracesPanel._selected_keys()
            per_cond = {k: diff["per_cond"][k] for k in selected if k in diff["per_cond"]}
            x_label = diff.get("x_label", "condition")
        elif diff is not None and not txt:
            per_cond = None
            x_label = diff.get("x_label", "condition")
        else:
            per_cond = None
            x_label = ("V_front" if self._current_mode == self.MODE_GATE else "Power_uW")
        if not per_cond:
            if self._sigma2_cache is None:
                QtWidgets.QMessageBox.warning(self, "Export σ²", "No data available.")
                return
            t_arr, s2_eq, s2_x, s2_y = self._sigma2_cache
            M2 = self._magnification ** 2
            x_lbl, xval = self._x_label_and_val()
            per_cond = {f"{x_lbl}={xval}": (
                np.asarray(t_arr),
                np.asarray(s2_eq) / M2,
                np.asarray(s2_x) / M2,
                np.asarray(s2_y) / M2)}
            x_label = x_lbl
        sigma_type = self.sigmaTypeCombo.currentText()
        tag = self._sigma_type_tag()
        path = self._choose_export_path(f"listed_{tag}_export.csv")
        if path is None:
            return
        try:
            self._write_sigma2_wide(path, per_cond, x_label, sigma_type=sigma_type)
            self._set_status(f"Exported listed σ²: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export σ²", f"Failed:\n{exc}")

    def _export_sigma2_map_csv(self) -> None:
        """Button 5: σ²(t) for all conditions from Fit All Maps (chosen σ² type)."""
        diff = getattr(self, '_diffusion_data', None)
        if diff is None or not diff.get("per_cond"):
            QtWidgets.QMessageBox.warning(self, "Export σ² Map", "Run 'Fit All Maps' first.")
            return
        sigma_type = self.sigmaTypeCombo.currentText()
        tag = self._sigma_type_tag()
        path = self._choose_export_path(f"{tag}_map_export.csv")
        if path is None:
            return
        try:
            self._write_sigma2_wide(path, diff["per_cond"], diff.get("x_label", "condition"),
                                    sigma_type=sigma_type)
            self._set_status(f"Exported σ² map: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export σ² Map", f"Failed:\n{exc}")

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _on_mode_changed(self, text: str) -> None:
        self._current_mode = text
        self._main_idx = 0
        self._fixed_idx = 0
        self._configure_sliders_for_mode()
        self._record_selection()
        self._refresh_trpl_views()

    def _on_main_changed(self, value: int) -> None:
        self._main_idx = int(value)
        self._record_selection()
        self._refresh_all()

    def _on_fixed_changed(self, value: int) -> None:
        self._fixed_idx = int(value)
        self._record_selection()
        self._refresh_all()

    def _on_deriv_scale_changed(self, _value: float) -> None:
        if self._last_x is not None and self._last_y_wl is not None and self._last_map_data is not None:
            self._update_maps()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")
    app.setFont(QtGui.QFont("Segoe UI", 9))
    data_dir: Optional[Path] = None
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser().resolve()
        if p.is_dir():
            data_dir = p
    win = PowerGateDepWindowV2(data_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


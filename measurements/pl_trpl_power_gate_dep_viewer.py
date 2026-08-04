import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import colors, colormaps
try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

DEFAULT_CROP = (50, 50, 200, 200)
EV_NM = 1239.841984
GRID_DECIMALS = 9
NUM_SPAD_PIXELS = 23
SPAD_PIXEL_COORDS: Dict[int, Tuple[float, float]] = {
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
SPAD_FIT_ROWS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12, 13],
    [14, 15, 16, 17],
    [18, 19, 20, 21, 22],
]
SPAD_PITCH_X_UM = 23.0
SPAD_PITCH_Y_UM = 19.92

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from andor.gui.live_view_widget import AndorLiveViewWidget


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
class TRPLRecord:
    path: Path
    power_w: Optional[float]
    v_front: Optional[float]
    v_back: Optional[float]
    time_ns: np.ndarray
    counts: np.ndarray
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


def build_spad_fit_coords() -> Tuple[np.ndarray, np.ndarray]:
    coords: Dict[int, Tuple[float, float]] = {}
    max_cols = max(len(row) for row in SPAD_FIT_ROWS)
    for row_idx, row in enumerate(SPAD_FIT_ROWS):
        x0 = (SPAD_PITCH_X_UM / 2.0) if len(row) < max_cols else 0.0
        for col_idx, pixel in enumerate(row):
            coords[pixel] = (x0 + col_idx * SPAD_PITCH_X_UM, row_idx * SPAD_PITCH_Y_UM)
    x = np.array([coords[i][0] for i in range(NUM_SPAD_PIXELS)], dtype=np.float64)
    y = np.array([coords[i][1] for i in range(NUM_SPAD_PIXELS)], dtype=np.float64)
    return x, y


def fit_spad_gaussian_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[SpadFitResult]:
    if curve_fit is None:
        return None
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size < 6 or np.allclose(z, z[0]):
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
        [0.0, float(np.min(x) - SPAD_PITCH_X_UM), float(np.min(y) - SPAD_PITCH_Y_UM), 1e-3, 1e-3, -np.inf],
        [np.inf, float(np.max(x) + SPAD_PITCH_X_UM), float(np.max(y) + SPAD_PITCH_Y_UM), np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return SpadFitResult(popt=np.asarray(popt, dtype=np.float64), sigma_eq=sigma_eq)


def parse_header(path: Path) -> dict:
    meta = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if not line:
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip().lower()] = val.strip()
            else:
                meta.setdefault("_header", []).append(line)
    return meta


def load_andor_ascii(path: Path) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    meta = parse_header(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        data = np.loadtxt(handle, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    wl = None
    cols = str(meta.get("columns", "")).lower()
    if data.shape[1] > 1 and "wavelength" in cols:
        wl = data[:, 0]
        image = data[:, 1:].T
    else:
        image = data
    return meta, wl, image


def parse_optional_float(meta: dict, key: str) -> Optional[float]:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


_POWER_RE = re.compile(r"(?:^|_)P(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)\s*(?P<unit>nW|uW|mW|W)(?=_|\.|$)")
_VF_RE = re.compile(r"(?:^|_)Vf(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)(?=_|\.|$)")
_VB_RE = re.compile(r"(?:^|_)Vb(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)(?=_|\.|$)")


def _parse_token(token: str) -> Optional[float]:
    if not token:
        return None
    sign = 1.0
    if token.startswith("-"):
        sign = -1.0
        token = token[1:]
    elif token.startswith("+"):
        token = token[1:]
    if token.startswith("m"):
        sign *= -1.0
        token = token[1:]
    try:
        return sign * float(token.replace("p", "."))
    except Exception:
        return None


def parse_from_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    power_w = None
    v_front = None
    v_back = None

    pm = _POWER_RE.search(name)
    if pm:
        val = _parse_token(pm.group("val"))
        unit = pm.group("unit").lower()
        scale = {"w": 1.0, "mw": 1e-3, "uw": 1e-6, "nw": 1e-9}.get(unit)
        if val is not None and scale is not None:
            power_w = float(val) * float(scale)

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
    if not np.isfinite(x):
        return None
    return round(x, GRID_DECIMALS)


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


def apply_crop(image: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    if image is None:
        return image
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
        arr = arr[min(left, arr.size) :]
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


def resample_linecut(wl_src: Optional[np.ndarray], linecut: np.ndarray, wl_dst: Optional[np.ndarray]) -> np.ndarray:
    if wl_src is None or wl_dst is None:
        return linecut
    src = np.asarray(wl_src, dtype=float).ravel()
    dst = np.asarray(wl_dst, dtype=float).ravel()
    if src.size == dst.size and np.allclose(src, dst):
        return linecut
    order = np.argsort(src)
    src_sorted = src[order]
    line_sorted = np.asarray(linecut, dtype=float)[order]
    return np.interp(dst, src_sorted, line_sorted, left=np.nan, right=np.nan)


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


def estimate_baseline_from_tail(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    *,
    low_nm: float = 940.0,
    high_nm: float = 960.0,
) -> float:
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y) & (x >= float(low_nm)) & (x <= float(high_nm))
    if np.count_nonzero(mask) >= 3:
        return float(np.nanmean(y[mask]))
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, 10.0))


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
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(std)


def fit_lorentzian_peak_local(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    center_guess_nm: float,
    *,
    window_nm: float,
    min_points: int = 7,
    min_snr: float = 2.0,
    min_gamma_nm: float = 0.08,
    max_gamma_nm: float = 10.0,
):
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size < min_points:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x - float(center_guess_nm)) <= float(window_nm))
    if np.count_nonzero(mask) < min_points:
        return None

    xs = x[mask]
    ys = y[mask]
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
    right = np.where(yb[peak_idx + 1 :] <= y_half)[0]
    if left.size > 0 and right.size > 0:
        left_x = float(xs[left[-1]])
        right_x = float(xs[peak_idx + 1 + right[0]])
        fwhm_guess = max(2.0 * float(min_gamma_nm), right_x - left_x)
    else:
        fwhm_guess = max(2.0 * float(min_gamma_nm), 0.25 * float(window_nm))
    gamma_fit = 0.5 * float(fwhm_guess)
    gamma_fit = float(np.clip(gamma_fit, float(min_gamma_nm), float(max_gamma_nm)))
    amp_fit = amp

    if curve_fit is not None:
        try:
            p0 = np.array([amp_fit, center_fit, gamma_fit], dtype=float)
            bounds_low = np.array([0.0, float(center_guess_nm) - float(window_nm), float(min_gamma_nm)], dtype=float)
            bounds_high = np.array(
                [max(1.0, 4.0 * amp_fit), float(center_guess_nm) + float(window_nm), float(max_gamma_nm)],
                dtype=float,
            )
            p0 = np.clip(p0, bounds_low + 1e-9, bounds_high - 1e-9)
            popt, _ = curve_fit(
                lambda xx, a, m, g: lorentzian_component(xx, a, m, g),
                xs,
                yb,
                p0=p0,
                bounds=(bounds_low, bounds_high),
                maxfev=8000,
            )
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
    return {
        "center_nm": float(center_fit),
        "gamma_nm": float(gamma_fit),
        "fwhm_nm": float(2.0 * gamma_fit),
        "amplitude": float(max(0.0, amp_fit)),
    }


def _estimate_peak_guess(x: np.ndarray, y: np.ndarray, low_nm: float, high_nm: float, fallback_nm: float):
    mask = np.isfinite(x) & np.isfinite(y) & (x >= float(low_nm)) & (x <= float(high_nm))
    if np.count_nonzero(mask) < 3:
        return 0.0, float(fallback_nm), 3.0
    xs = x[mask]
    ys = y[mask]
    idx = int(np.nanargmax(ys))
    center = float(xs[idx])
    baseline = float(np.nanpercentile(ys, 15.0))
    amp = float(max(0.0, ys[idx] - baseline))
    w = np.clip(ys - baseline, 0.0, None)
    wsum = float(np.nansum(w))
    if wsum > 0:
        sigma = float(np.sqrt(np.nansum(w * (xs - center) ** 2) / wsum))
    else:
        sigma = 3.0
    gamma = float(np.clip(0.5 * sigma, 0.5, 10.0))
    return amp, center, gamma


def _one_lorentzian_model(x: np.ndarray, a1: float, m1: float, g1: float) -> np.ndarray:
    return lorentzian_component(x, a1, m1, g1)


def _two_lorentzian_model(x: np.ndarray, a1: float, m1: float, g1: float, a2: float, m2: float, g2: float) -> np.ndarray:
    return lorentzian_component(x, a1, m1, g1) + lorentzian_component(x, a2, m2, g2)


def fit_two_peak_linecut(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    *,
    peak1_guess_nm: float,
    peak2_guess_nm: float,
    peak1_window_nm: float = 10.0,
    peak2_window_nm: float = 5.0,
) -> dict:
    x_all = np.asarray(wavelength_nm, dtype=float).ravel()
    y_all = np.asarray(intensity, dtype=float).ravel()
    if x_all.size != y_all.size or x_all.size < 8:
        return {"peak_1": None, "peak_2": None, "offset": 0.0}

    valid = np.isfinite(x_all) & np.isfinite(y_all)
    if np.count_nonzero(valid) < 8:
        return {"peak_1": None, "peak_2": None, "offset": 0.0}
    x_all = x_all[valid]
    y_all = y_all[valid]
    order = np.argsort(x_all)
    x_all = x_all[order]
    y_all = y_all[order]

    baseline_value = estimate_baseline_from_tail(x_all, y_all)
    y_all_sub = y_all - baseline_value

    w1 = max(0.2, float(peak1_window_nm))
    w2 = max(0.2, float(peak2_window_nm))
    p1_lo = float(peak1_guess_nm) - w1
    p1_hi = float(peak1_guess_nm) + w1
    p2_lo = float(peak2_guess_nm) - w2
    p2_hi = float(peak2_guess_nm) + w2
    fit_lo = min(p2_lo, p1_lo) - 6.0
    fit_hi = max(p2_hi, p1_hi) + 30.0

    fit_mask = (x_all >= fit_lo) & (x_all <= fit_hi)
    if np.count_nonzero(fit_mask) >= 20:
        x = x_all[fit_mask]
        y = y_all_sub[fit_mask]
    else:
        x = x_all
        y = y_all_sub
    if x.size < 8:
        return {"peak_1": None, "peak_2": None, "offset": float(baseline_value)}

    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    y_span = float(max(1.0, y_max - y_min))

    a1_g, m1_g, g1_g = _estimate_peak_guess(x, y, p1_lo, p1_hi, float(peak1_guess_nm))
    a2_g, m2_g, g2_g = _estimate_peak_guess(x, y, p2_lo, p2_hi, float(peak2_guess_nm))
    a1_g = float(max(0.08 * y_span, a1_g))
    a2_g = float(max(0.02 * y_span, a2_g))

    p0_two = np.array([a1_g, m1_g, g1_g, a2_g, m2_g, g2_g], dtype=float)
    low_two = np.array([0.0, p1_lo, 0.4, 0.0, p2_lo, 0.4], dtype=float)
    high_two = np.array([4.0 * y_span, p1_hi, 15.0, 4.0 * y_span, p2_hi, 12.0], dtype=float)
    p0_two = np.clip(p0_two, low_two + 1e-6, high_two - 1e-6)

    p0_one = np.array([a1_g, m1_g, g1_g], dtype=float)
    low_one = np.array([0.0, p1_lo, 0.4], dtype=float)
    high_one = np.array([4.0 * y_span, p1_hi, 15.0], dtype=float)
    p0_one = np.clip(p0_one, low_one + 1e-6, high_one - 1e-6)

    popt_two = None
    popt_one = None
    if curve_fit is not None:
        try:
            popt_two, _ = curve_fit(
                lambda xx, a1, m1, g1, a2, m2, g2: _two_lorentzian_model(xx, a1, m1, g1, a2, m2, g2),
                x,
                y,
                p0=p0_two,
                bounds=(low_two, high_two),
                maxfev=20000,
            )
        except Exception:
            popt_two = None
        try:
            popt_one, _ = curve_fit(
                lambda xx, a1, m1, g1: _one_lorentzian_model(xx, a1, m1, g1),
                x,
                y,
                p0=p0_one,
                bounds=(low_one, high_one),
                maxfev=12000,
            )
        except Exception:
            popt_one = None

    if popt_two is None and popt_one is None:
        p1 = fit_lorentzian_peak_local(x_all, y_all_sub, float(peak1_guess_nm), window_nm=10.0, min_snr=1.2)
        residual = np.asarray(y_all_sub, dtype=float).copy()
        if p1 is not None:
            residual = residual - lorentzian_component(x_all, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
        p2 = fit_lorentzian_peak_local(x_all, residual, float(peak2_guess_nm), window_nm=5.0, min_snr=1.5)
        if p1 is not None and not (p1_lo <= p1["center_nm"] <= p1_hi):
            p1 = None
        if p2 is not None and not (p2_lo <= p2["center_nm"] <= p2_hi):
            p2 = None
        return {"peak_1": p1, "peak_2": p2, "offset": float(baseline_value)}

    use_second = False
    if popt_two is not None:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt_two]
        y_fit_one_like = _one_lorentzian_model(x, a1, m1, g1)
        if popt_one is not None:
            a1_1, m1_1, g1_1 = [float(v) for v in popt_one]
            y_fit_one_like = _one_lorentzian_model(x, a1_1, m1_1, g1_1)
        y_fit_two = _two_lorentzian_model(x, a1, m1, g1, a2, m2, g2)
        noise = _estimate_noise_floor(y - y_fit_two)
        rss_one = float(np.nansum((y - y_fit_one_like) ** 2))
        rss_two = float(np.nansum((y - y_fit_two) ** 2))
        improvement = (rss_one - rss_two) / rss_one if rss_one > 0 else 0.0
        min_amp2 = float(max(2.0 * noise, 0.04 * y_span, 0.06 * max(a1, 1.0)))
        use_second = bool(
            a2 >= min_amp2 and m2 < m1 - 1.0 and p2_lo <= m2 <= p2_hi and 0.4 <= g2 <= 12.0 and improvement > 0.03
        )

    if use_second:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt_two]
    elif popt_one is not None:
        a1, m1, g1 = [float(v) for v in popt_one]
        a2, m2, g2 = 0.0, float(peak2_guess_nm), 3.0
    else:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt_two]
        a2 = 0.0

    p1 = None
    if a1 > 0 and 0.4 <= g1 <= 20.0 and p1_lo <= m1 <= p1_hi:
        p1 = {"center_nm": float(m1), "gamma_nm": float(g1), "fwhm_nm": float(2.0 * g1), "amplitude": float(a1)}
    p2 = None
    if use_second and a2 > 0 and 0.4 <= g2 <= 20.0 and p2_lo <= m2 <= p2_hi:
        p2 = {"center_nm": float(m2), "gamma_nm": float(g2), "fwhm_nm": float(2.0 * g2), "amplitude": float(a2)}
    return {"peak_1": p1, "peak_2": p2, "offset": float(baseline_value)}


def fit_two_peak_map(
    wavelength_nm: np.ndarray,
    map_data: np.ndarray,
    *,
    peak1_guess_nm: float,
    peak2_guess_nm: float,
) -> dict:
    wl = np.asarray(wavelength_nm, dtype=float).ravel()
    data = np.asarray(map_data, dtype=float)
    n_x = data.shape[1] if data.ndim == 2 else 0
    center_1 = np.full(n_x, np.nan, dtype=float)
    fwhm_1 = np.full(n_x, np.nan, dtype=float)
    amp_1 = np.full(n_x, np.nan, dtype=float)
    center_2 = np.full(n_x, np.nan, dtype=float)
    fwhm_2 = np.full(n_x, np.nan, dtype=float)
    amp_2 = np.full(n_x, np.nan, dtype=float)

    if data.ndim != 2 or wl.size != data.shape[0]:
        return {
            "center_1_nm": center_1,
            "fwhm_1_nm": fwhm_1,
            "amp_1": amp_1,
            "center_2_nm": center_2,
            "fwhm_2_nm": fwhm_2,
            "amp_2": amp_2,
        }

    for j in range(n_x):
        fit = fit_two_peak_linecut(
            wl,
            data[:, j],
            peak1_guess_nm=float(peak1_guess_nm),
            peak2_guess_nm=float(peak2_guess_nm),
        )
        p1 = fit.get("peak_1")
        p2 = fit.get("peak_2")
        if p1 is not None:
            center_1[j] = float(p1["center_nm"])
            fwhm_1[j] = float(p1["fwhm_nm"])
            amp_1[j] = float(p1["amplitude"])
        if p2 is not None:
            center_2[j] = float(p2["center_nm"])
            fwhm_2[j] = float(p2["fwhm_nm"])
            amp_2[j] = float(p2["amplitude"])

    return {
        "center_1_nm": center_1,
        "fwhm_1_nm": fwhm_1,
        "amp_1": amp_1,
        "center_2_nm": center_2,
        "fwhm_2_nm": fwhm_2,
        "amp_2": amp_2,
    }


def fit_two_peak_map_sequential(
    wavelength_nm: np.ndarray,
    map_data: np.ndarray,
    *,
    ref_idx: int,
    peak1_guess_nm: float,
    peak2_guess_nm: float,
    max_shift_nm: float = 2.0,
) -> dict:
    wl = np.asarray(wavelength_nm, dtype=float).ravel()
    data = np.asarray(map_data, dtype=float)
    n_x = data.shape[1] if data.ndim == 2 else 0
    center_1 = np.full(n_x, np.nan, dtype=float)
    fwhm_1 = np.full(n_x, np.nan, dtype=float)
    gamma_1 = np.full(n_x, np.nan, dtype=float)
    amp_1 = np.full(n_x, np.nan, dtype=float)
    center_2 = np.full(n_x, np.nan, dtype=float)
    fwhm_2 = np.full(n_x, np.nan, dtype=float)
    gamma_2 = np.full(n_x, np.nan, dtype=float)
    amp_2 = np.full(n_x, np.nan, dtype=float)
    offset = np.full(n_x, np.nan, dtype=float)

    def _empty():
        return {
            "center_1_nm": center_1,
            "fwhm_1_nm": fwhm_1,
            "gamma_1_nm": gamma_1,
            "amp_1": amp_1,
            "center_2_nm": center_2,
            "fwhm_2_nm": fwhm_2,
            "gamma_2_nm": gamma_2,
            "amp_2": amp_2,
            "offset": offset,
        }

    if data.ndim != 2 or wl.size != data.shape[0] or n_x == 0:
        return _empty()

    ref = max(0, min(int(ref_idx), n_x - 1))
    max_step = max(0.2, float(max_shift_nm))

    def _store(j: int, fit: dict) -> None:
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

    fit_ref = fit_two_peak_linecut(
        wl,
        data[:, ref],
        peak1_guess_nm=float(peak1_guess_nm),
        peak2_guess_nm=float(peak2_guess_nm),
        peak1_window_nm=10.0,
        peak2_window_nm=5.0,
    )
    _store(ref, fit_ref)
    g1_ref = center_1[ref] if np.isfinite(center_1[ref]) else float(peak1_guess_nm)
    g2_ref = center_2[ref] if np.isfinite(center_2[ref]) else float(peak2_guess_nm)

    def _walk(start: int, stop: int, step: int, g1_init: float, g2_init: float) -> None:
        g1 = float(g1_init)
        g2 = float(g2_init)
        for j in range(start, stop, step):
            fit_j = fit_two_peak_linecut(
                wl,
                data[:, j],
                peak1_guess_nm=g1,
                peak2_guess_nm=g2,
                peak1_window_nm=max_step,
                peak2_window_nm=max_step,
            )
            _store(j, fit_j)
            if np.isfinite(center_1[j]):
                g1 = float(center_1[j])
            if np.isfinite(center_2[j]):
                g2 = float(center_2[j])

    _walk(ref + 1, n_x, +1, g1_ref, g2_ref)
    _walk(ref - 1, -1, -1, g1_ref, g2_ref)
    return _empty()


def fwhm_nm_to_mev(center_nm: np.ndarray, fwhm_nm: np.ndarray) -> np.ndarray:
    c = np.asarray(center_nm, dtype=float)
    w = np.asarray(fwhm_nm, dtype=float)
    out = np.full(np.broadcast(c, w).shape, np.nan, dtype=float)
    c, w = np.broadcast_arrays(c, w)
    half = 0.5 * w
    valid = np.isfinite(c) & np.isfinite(w) & (w > 0) & (c > half) & (c + half > 0)
    if not np.any(valid):
        return out
    e_hi = EV_NM / (c[valid] - half[valid])
    e_lo = EV_NM / (c[valid] + half[valid])
    out[valid] = np.abs(e_hi - e_lo) * 1e3
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


def load_frames(data_dir: Path) -> Tuple[List[PLFrame], ViewerDefaults]:
    frames: List[PLFrame] = []
    first_meta = None
    for path in sorted(data_dir.glob("*.asc")):
        try:
            meta, wl, image = load_andor_ascii(path)
        except Exception:
            continue
        p_m = parse_optional_float(meta, "power_w")
        vf_m = parse_optional_float(meta, "front_v_set")
        if vf_m is None:
            vf_m = parse_optional_float(meta, "front_vset")
        vb_m = parse_optional_float(meta, "back_v_set")
        if vb_m is None:
            vb_m = parse_optional_float(meta, "back_vset")
        p_n, vf_n, vb_n = parse_from_name(path.name)
        power_w = p_n if p_n is not None else p_m
        v_front = vf_n if vf_n is not None else vf_m
        v_back = vb_n if vb_n is not None else vb_m

        frames.append(
            PLFrame(
                path=path,
                power_w=power_w,
                v_front=v_front,
                v_back=v_back,
                wavelength_nm=wl,
                image=image,
                meta=meta,
                mtime=float(path.stat().st_mtime),
            )
        )
        if first_meta is None:
            first_meta = meta

    if first_meta is None:
        defaults = ViewerDefaults(crop=DEFAULT_CROP, linecut_row=None, linecut_width=1)
    else:
        defaults = ViewerDefaults(
            crop=DEFAULT_CROP,
            linecut_row=parse_optional_int(first_meta, "linecut_row"),
            linecut_width=parse_int(first_meta, "linecut_width", 1),
        )
    return frames, defaults


def load_trpl_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < NUM_SPAD_PIXELS + 1:
        raise ValueError(f"TRPL CSV needs time_ns plus {NUM_SPAD_PIXELS} pixel columns")
    time_ns = np.asarray(data[:, 0], dtype=np.float64)
    counts = np.asarray(data[:, 1 : NUM_SPAD_PIXELS + 1], dtype=np.float64)
    return time_ns, counts


def load_trpl_records(data_dir: Path) -> List[TRPLRecord]:
    records: List[TRPLRecord] = []
    for path in sorted(data_dir.glob("TRPL_*.csv")):
        try:
            time_ns, counts = load_trpl_csv(path)
        except Exception:
            continue
        source_name = path.name
        if source_name.startswith("TRPL_"):
            source_name = source_name[len("TRPL_") :]
        power_w, v_front, v_back = parse_from_name(source_name)
        records.append(
            TRPLRecord(
                path=path,
                power_w=power_w,
                v_front=v_front,
                v_back=v_back,
                time_ns=time_ns,
                counts=counts,
                mtime=float(path.stat().st_mtime),
            )
        )
    return records


class OfflineLiveView(AndorLiveViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fit_line_total = None
        self._fit_line_1 = None
        self._fit_line_2 = None
        self._fit_legend = None

    def set_wavelength_axis(self, wl: Optional[np.ndarray]) -> None:
        if wl is None:
            self._wl_axis = None
        else:
            try:
                self._wl_axis = np.asarray(wl, dtype=float).ravel()
            except Exception:
                self._wl_axis = None
        if self._last_frame is not None:
            h, w = self._last_frame.shape
            self._maybe_update_xaxis_label(w)
            self._update_image_wavelength_axis(w)
            self._update_cursor_overlays(h, w)
            self.canvas.draw_idle()

    def set_image_title(self, text: str) -> None:
        self.ax_img.set_title(text)
        self.canvas.draw_idle()

    def clear_linecut_fit_overlay(self) -> None:
        for attr in ("_fit_line_total", "_fit_line_1", "_fit_line_2"):
            line = getattr(self, attr, None)
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._fit_legend is not None:
            try:
                self._fit_legend.remove()
            except Exception:
                pass
            self._fit_legend = None
        self.canvas.draw_idle()

    def set_linecut_fit_overlay(
        self,
        x_axis: np.ndarray,
        y_total: Optional[np.ndarray],
        y1: Optional[np.ndarray],
        y2: Optional[np.ndarray],
    ) -> None:
        x = np.asarray(x_axis, dtype=float).ravel()
        if x.size == 0:
            self.clear_linecut_fit_overlay()
            return

        def _valid_arr(y):
            if y is None:
                return None
            arr = np.asarray(y, dtype=float).ravel()
            if arr.size != x.size or not np.isfinite(arr).any():
                return None
            return arr

        yt = _valid_arr(y_total)
        y1v = _valid_arr(y1)
        y2v = _valid_arr(y2)
        if yt is None and y1v is None and y2v is None:
            self.clear_linecut_fit_overlay()
            return

        if self._fit_line_total is None:
            self._fit_line_total, = self.ax_h.plot([], [], "--", lw=1.2, color="#ffb300", label="Fit total")
        if self._fit_line_1 is None:
            self._fit_line_1, = self.ax_h.plot([], [], "--", lw=1.0, color="#1f77b4", label="Fit peak 1")
        if self._fit_line_2 is None:
            self._fit_line_2, = self.ax_h.plot([], [], "--", lw=1.0, color="#d62728", label="Fit peak 2")

        if yt is None:
            self._fit_line_total.set_data([], [])
        else:
            self._fit_line_total.set_data(x, yt)
        if y1v is None:
            self._fit_line_1.set_data([], [])
        else:
            self._fit_line_1.set_data(x, y1v)
        if y2v is None:
            self._fit_line_2.set_data([], [])
        else:
            self._fit_line_2.set_data(x, y2v)

        handles = []
        labels = []
        if yt is not None:
            handles.append(self._fit_line_total)
            labels.append("Fit total")
        if y1v is not None:
            handles.append(self._fit_line_1)
            labels.append("Fit peak 1")
        if y2v is not None:
            handles.append(self._fit_line_2)
            labels.append("Fit peak 2")

        if self._fit_legend is not None:
            try:
                self._fit_legend.remove()
            except Exception:
                pass
            self._fit_legend = None
        if handles:
            self._fit_legend = self.ax_h.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
        self.canvas.draw_idle()


class MapPlotWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, *, cmap_name: str, line_label: str, parent=None):
        super().__init__(title, parent)
        self._line_label = line_label
        self._manual_sym_max = None
        self._x = None
        self._y = None
        self._z = None
        self._selected_y_idx = None
        self._top_x = None
        self._top_label = ""
        self._last_vline_x = None

        base_cmap = colormaps.get_cmap(cmap_name)
        self._cmap = colors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
        self._cmap.set_bad((1, 1, 1, 0))

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(110)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.toolbar)
        top_row.addStretch()
        top_row.addWidget(self.exportBtn)
        self.ax_map = self.fig.add_subplot(211)
        self.ax_line = self.fig.add_subplot(212, sharex=self.ax_map)
        self.ax_top = None
        self._mesh = None
        self._hline = None
        self._vline = None
        self._line = None

        layout.addLayout(top_row)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.exportBtn.clicked.connect(self._on_export_csv)

    @staticmethod
    def _edges(v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=float).ravel()
        if x.size == 0:
            return np.array([0.0, 1.0])
        if x.size == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5])
        mid = 0.5 * (x[:-1] + x[1:])
        first = x[0] - (mid[0] - x[0])
        last = x[-1] + (x[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    def set_manual_symmetric_max(self, value: Optional[float]) -> None:
        if value is None or value <= 0:
            self._manual_sym_max = None
        else:
            self._manual_sym_max = float(value)

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x = None if x2 is None else np.asarray(x2, dtype=float).ravel()
        self._top_label = str(label or "")

    def _apply_secondary_axis(self) -> None:
        if self.ax_top is not None:
            try:
                self.ax_top.remove()
            except Exception:
                pass
            self.ax_top = None

        if self._top_x is None or self._x is None:
            return
        if self._top_x.size != self._x.size or self._x.size == 0:
            return

        self.ax_top = self.ax_map.twiny()
        self.ax_top.set_xlim(self.ax_map.get_xlim())
        self.ax_top.set_xlabel(self._top_label)

        nticks = min(8, self._x.size)
        idx = np.linspace(0, self._x.size - 1, nticks).astype(int)
        xt = self._x[idx]
        x2 = self._top_x[idx]
        labels = []
        for v in x2:
            if np.isfinite(v):
                labels.append(f"{float(v):.3g}")
            else:
                labels.append("")
        self.ax_top.set_xticks(xt)
        self.ax_top.set_xticklabels(labels)

    def set_map(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, *, x_label: str, y_label: str, symmetric: bool = False) -> None:
        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        self._z = np.asarray(z, dtype=float)
        self._x_label = str(x_label)
        self._y_label = str(y_label)

        self.ax_map.clear()
        self.ax_line.clear()
        if self._x.size == 0 or self._y.size == 0 or self._z.size == 0:
            self.canvas.draw_idle()
            return

        xe = self._edges(self._x)
        ye = self._edges(self._y)
        self._mesh = self.ax_map.pcolormesh(xe, ye, self._z, shading="auto", cmap=self._cmap)

        finite = self._z[np.isfinite(self._z)]
        if finite.size:
            if symmetric:
                if self._manual_sym_max is not None:
                    vmax = float(self._manual_sym_max)
                else:
                    vmax = float(np.nanpercentile(np.abs(finite), 98))
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = float(np.nanmax(np.abs(finite)))
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = 1.0
                self._mesh.set_clim(-vmax, vmax)
            else:
                vmin = float(np.nanpercentile(finite, 1))
                vmax = float(np.nanpercentile(finite, 99))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin = float(np.nanmin(finite))
                    vmax = float(np.nanmax(finite))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = 0.0, 1.0
                self._mesh.set_clim(vmin, vmax)

        self.ax_map.set_xlabel(x_label)
        self.ax_map.set_ylabel(y_label)
        self.ax_line.set_xlabel(x_label)
        self.ax_line.set_ylabel(self._line_label)

        if self._selected_y_idx is None:
            self._selected_y_idx = int(self._y.size // 2)
        self._selected_y_idx = max(0, min(int(self._selected_y_idx), self._y.size - 1))
        yv = float(self._y[self._selected_y_idx])
        self._hline = self.ax_map.axhline(yv, color="w", lw=0.8, alpha=0.9)
        x0 = float(self._x[0])
        self._vline = self.ax_map.axvline(x0, color="w", lw=0.8, alpha=0.7)
        self._line, = self.ax_line.plot(self._x, self._z[self._selected_y_idx, :], color="#1565c0")
        self.ax_line.relim()
        self.ax_line.autoscale_view()
        self.ax_line.set_title(f"{yv:.4g}")

        self._apply_secondary_axis()
        if self._last_vline_x is not None:
            self.set_x_marker(self._last_vline_x)
        self.canvas.draw_idle()

    def set_x_marker(self, x: float) -> None:
        self._last_vline_x = float(x)
        if self._vline is None:
            return
        self._vline.set_xdata([x, x])
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax_map or self._y is None or self._z is None:
            return
        if event.ydata is None:
            return
        idx = int(np.argmin(np.abs(self._y - float(event.ydata))))
        self.select_y_index(idx)

    def select_y_index(self, idx: int) -> None:
        if self._y is None or self._z is None:
            return
        idx = max(0, min(int(idx), self._y.size - 1))
        self._selected_y_idx = idx
        yv = float(self._y[idx])
        if self._hline is not None:
            self._hline.set_ydata([yv, yv])
        if self._line is not None:
            self._line.set_data(self._x, self._z[idx, :])
        self.ax_line.relim()
        self.ax_line.autoscale_view()
        self.ax_line.set_title(f"{yv:.4g}")
        self.canvas.draw_idle()

    def _on_export_csv(self) -> None:
        if self._x is None or self._y is None or self._z is None:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data to export yet.")
            return
        x = np.asarray(self._x, dtype=float).ravel()
        y = np.asarray(self._y, dtype=float).ravel()
        z = np.asarray(self._z, dtype=float)
        if x.size == 0 or y.size == 0 or z.shape != (y.size, x.size):
            QtWidgets.QMessageBox.warning(self, "Export CSV", "Data shape mismatch.")
            return

        parent = self.window()
        default_dir = Path.cwd()
        if hasattr(parent, "data_dir"):
            try:
                default_dir = Path(getattr(parent, "data_dir"))
            except Exception:
                default_dir = Path.cwd()
        default_name = self.title().lower().replace(" ", "_").replace("/", "_")
        default_path = default_dir / f"{default_name}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", str(default_path), "CSV Files (*.csv)")
        if not path:
            return

        out = np.empty((y.size + 1, x.size + 1), dtype=float)
        out.fill(np.nan)
        out[0, 0] = 0.0
        out[0, 1:] = x
        out[1:, 0] = y
        out[1:, 1:] = z
        np.savetxt(path, out, delimiter=",", fmt="%.10g")


class FitTrendPlotWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, y_label: str, parent=None):
        super().__init__(title, parent)
        self._y_label = y_label
        self._x_last = None
        self._y1_last = None
        self._y2_last = None
        self._x_label_last = "X"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(110)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.toolbar)
        top_row.addStretch()
        top_row.addWidget(self.exportBtn)
        self.ax = self.fig.add_subplot(111)
        layout.addLayout(top_row)
        layout.addWidget(self.canvas)
        self.exportBtn.clicked.connect(self._on_export_csv)

    def set_data(
        self,
        x: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        *,
        x_label: str,
        label_1: str = "Peak 1",
        label_2: str = "Peak 2",
    ) -> None:
        x = np.asarray(x, dtype=float).ravel()
        y1 = np.asarray(y1, dtype=float).ravel()
        y2 = np.asarray(y2, dtype=float).ravel()
        n = min(x.size, y1.size, y2.size)
        x, y1, y2 = x[:n], y1[:n], y2[:n]
        self._x_last = x.copy()
        self._y1_last = y1.copy()
        self._y2_last = y2.copy()
        self._x_label_last = str(x_label)

        self.ax.clear()
        mask1 = np.isfinite(x) & np.isfinite(y1)
        mask2 = np.isfinite(x) & np.isfinite(y2)
        drew = False
        if np.count_nonzero(mask1):
            self.ax.plot(x[mask1], y1[mask1], "-o", ms=3, lw=1.2, color="#1f77b4", label=label_1)
            drew = True
        if np.count_nonzero(mask2):
            self.ax.plot(x[mask2], y2[mask2], "-o", ms=3, lw=1.2, color="#d62728", label=label_2)
            drew = True
        self.ax.set_xlabel(str(x_label))
        self.ax.set_ylabel(self._y_label)
        self.ax.grid(True, alpha=0.25)
        if drew:
            self.ax.legend(loc="best", frameon=False)
        else:
            self.ax.text(0.5, 0.5, "No valid fit points", transform=self.ax.transAxes, ha="center", va="center")
        self.canvas.draw_idle()

    def clear_data(self, message: str, *, x_label: str = "X") -> None:
        self._x_last = None
        self._y1_last = None
        self._y2_last = None
        self._x_label_last = str(x_label)
        self.ax.clear()
        self.ax.set_xlabel(str(x_label))
        self.ax.set_ylabel(self._y_label)
        self.ax.grid(True, alpha=0.25)
        self.ax.text(0.5, 0.5, message, transform=self.ax.transAxes, ha="center", va="center")
        self.canvas.draw_idle()

    def _on_export_csv(self) -> None:
        if self._x_last is None or self._y1_last is None or self._y2_last is None:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data to export yet.")
            return
        x = np.asarray(self._x_last, dtype=float).ravel()
        y1 = np.asarray(self._y1_last, dtype=float).ravel()
        y2 = np.asarray(self._y2_last, dtype=float).ravel()
        n = min(x.size, y1.size, y2.size)
        if n == 0:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data to export yet.")
            return
        arr = np.column_stack([x[:n], y1[:n], y2[:n]])

        parent = self.window()
        default_dir = Path.cwd()
        if hasattr(parent, "data_dir"):
            try:
                default_dir = Path(getattr(parent, "data_dir"))
            except Exception:
                default_dir = Path.cwd()
        default_name = self.title().lower().replace(" ", "_").replace("/", "_")
        default_path = default_dir / f"{default_name}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", str(default_path), "CSV Files (*.csv)")
        if not path:
            return
        header = f"{self._x_label_last},peak1,peak2"
        np.savetxt(path, arr, delimiter=",", fmt="%.10g", header=header, comments="")


class PowerGateDepWindow(QtWidgets.QMainWindow):
    MODE_POWER = "Power dependence"
    MODE_GATE = "Gate dependence"

    def __init__(self, frames: List[PLFrame], trpl_records: List[TRPLRecord], defaults: ViewerDefaults, data_dir: Path):
        super().__init__()
        self.frames = frames
        self.trpl_records = trpl_records
        self.defaults = defaults
        self.data_dir = data_dir

        self._crop = tuple(defaults.crop)
        self._grid: Dict[Tuple[Optional[float], Optional[float]], PLFrame] = {}
        self._trpl_grid: Dict[Tuple[Optional[float], Optional[float]], TRPLRecord] = {}
        self._powers: np.ndarray = np.array([], dtype=float)
        self._gates: np.ndarray = np.array([], dtype=float)
        self._backs_for_gate: np.ndarray = np.array([], dtype=float)
        self._default_power_key: float = 0.0
        self._default_gate_key: float = 0.0
        self._can_power_mode = False
        self._can_gate_mode = False
        self._blank_image = None
        self._blank_wl = None

        self._cropped_image_cache: Dict[Path, np.ndarray] = {}
        self._cropped_wl_cache: Dict[Path, np.ndarray] = {}
        self._build_grid()

        self._current_mode = self.MODE_POWER
        self._main_idx = 0
        self._fixed_idx = 0
        self._map_version = 0
        self._fit_map_version = -1
        self._fit_result = None
        self._last_x = None
        self._last_x_label = ""
        self._last_y_wl = None
        self._last_map_data = None
        self._selected_spad_pixel = 11
        self._spad_map_coords = np.array([SPAD_PIXEL_COORDS[i] for i in range(NUM_SPAD_PIXELS)], dtype=np.float64)
        self._spad_fit_x_um, self._spad_fit_y_um = build_spad_fit_coords()
        self._selection_history: List[Tuple[float, float, int]] = []
        self._initializing = True

        self._build_ui()
        self.folderStatusLbl.setText(f"{len(self.frames)} PL, {len(self.trpl_records)} TRPL")
        self._init_mode()
        self._on_apply_crop()
        self._initializing = False
        self._record_selection()
        self._refresh_trpl_views()

    def _build_grid(self) -> None:
        power_keys = set()
        gate_keys = set()
        back_by_gate = {}
        all_points = list(self.frames) + list(self.trpl_records)
        for fr in all_points:
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
        self._default_power_key = normalize_key(float(self._powers[0])) if self._powers.size else 0.0
        self._default_gate_key = normalize_key(float(self._gates[0])) if self._gates.size else 0.0
        self._grid = {}
        for fr in self.frames:
            p = self._norm_power_key(fr.power_w)
            vf = self._norm_gate_key(fr.v_front)
            key = (p, vf)
            old = self._grid.get(key)
            if old is None or fr.mtime > old.mtime:
                self._grid[key] = fr
        self._trpl_grid = {}
        for rec in self.trpl_records:
            p = self._norm_power_key(rec.power_w)
            vf = self._norm_gate_key(rec.v_front)
            key = (p, vf)
            old = self._trpl_grid.get(key)
            if old is None or rec.mtime > old.mtime:
                self._trpl_grid[key] = rec
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
            if ref.wavelength_nm is not None:
                self._blank_wl = np.asarray(ref.wavelength_nm, dtype=float).copy()
            else:
                self._blank_wl = np.arange(ref.image.shape[1], dtype=float)
        else:
            self._blank_image = np.zeros((16, 16), dtype=float)
            self._blank_wl = np.arange(16, dtype=float)

    def _norm_power_key(self, p: Optional[float]) -> float:
        k = normalize_key(p)
        return self._default_power_key if k is None else float(k)

    def _norm_gate_key(self, vf: Optional[float]) -> float:
        k = normalize_key(vf)
        return self._default_gate_key if k is None else float(k)

    def _build_ui(self) -> None:
        self.setWindowTitle("Power + Gate Dependent PL (offline)")
        self.resize(1900, 980)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        controls = QtWidgets.QGroupBox("Controls")
        grid = QtWidgets.QGridLayout(controls)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        grid.addWidget(QtWidgets.QLabel("Data folder:"), 0, 0)
        self.folderEdit = QtWidgets.QLineEdit(str(self.data_dir))
        self.folderEdit.setPlaceholderText("Folder containing PL_*.asc and TRPL_*.csv files")
        self.browseFolderBtn = QtWidgets.QPushButton("Browse")
        self.browseFolderBtn.setFixedWidth(90)
        self.loadFolderBtn = QtWidgets.QPushButton("Load")
        self.loadFolderBtn.setFixedWidth(80)
        self.folderStatusLbl = QtWidgets.QLabel("")
        grid.addWidget(self.folderEdit, 0, 1, 1, 4)
        grid.addWidget(self.browseFolderBtn, 0, 5)
        grid.addWidget(self.loadFolderBtn, 0, 6)
        grid.addWidget(self.folderStatusLbl, 0, 7)

        grid.addWidget(QtWidgets.QLabel("Mode:"), 1, 0)
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItems([self.MODE_POWER, self.MODE_GATE])
        self.modeCombo.setFixedWidth(180)
        grid.addWidget(self.modeCombo, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Main index:"), 1, 2)
        self.mainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mainSlider.setTracking(True)
        grid.addWidget(self.mainSlider, 1, 3, 1, 3)
        self.mainValueLbl = QtWidgets.QLabel("--")
        self.mainValueLbl.setMinimumWidth(280)
        grid.addWidget(self.mainValueLbl, 1, 6, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Fixed index:"), 2, 0)
        self.fixedSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fixedSlider.setTracking(True)
        grid.addWidget(self.fixedSlider, 2, 1, 1, 5)
        self.fixedValueLbl = QtWidgets.QLabel("--")
        grid.addWidget(self.fixedValueLbl, 2, 6, 1, 2)

        self.cropTopSpin = QtWidgets.QSpinBox()
        self.cropBottomSpin = QtWidgets.QSpinBox()
        self.cropLeftSpin = QtWidgets.QSpinBox()
        self.cropRightSpin = QtWidgets.QSpinBox()
        for s in (self.cropTopSpin, self.cropBottomSpin, self.cropLeftSpin, self.cropRightSpin):
            s.setRange(0, 10000)
            s.setFixedWidth(70)
        self.cropTopSpin.setValue(int(self._crop[0]))
        self.cropBottomSpin.setValue(int(self._crop[1]))
        self.cropLeftSpin.setValue(int(self._crop[2]))
        self.cropRightSpin.setValue(int(self._crop[3]))
        self.cropApplyBtn = QtWidgets.QPushButton("Apply Crop")
        self.cropApplyBtn.setFixedWidth(120)

        grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 3, 0)
        grid.addWidget(self.cropTopSpin, 3, 1)
        grid.addWidget(self.cropBottomSpin, 3, 2)
        grid.addWidget(self.cropLeftSpin, 3, 3)
        grid.addWidget(self.cropRightSpin, 3, 4)
        grid.addWidget(self.cropApplyBtn, 3, 5)

        self.derivMaxSpin = QtWidgets.QDoubleSpinBox()
        self.derivMaxSpin.setDecimals(4)
        self.derivMaxSpin.setRange(0.0, 1.0e12)
        self.derivMaxSpin.setSpecialValueText("Auto")
        self.derivMaxSpin.setValue(0.0)
        self.derivMaxSpin.setFixedWidth(120)
        grid.addWidget(QtWidgets.QLabel("Derivative max abs:"), 3, 6)
        grid.addWidget(self.derivMaxSpin, 3, 7)

        self.peak1GuessEdit = QtWidgets.QLineEdit("855")
        self.peak2GuessEdit = QtWidgets.QLineEdit("835")
        self.peak1GuessEdit.setFixedWidth(90)
        self.peak2GuessEdit.setFixedWidth(90)
        grid.addWidget(QtWidgets.QLabel("Peak 1 guess (nm):"), 4, 0)
        grid.addWidget(self.peak1GuessEdit, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Peak 2 guess (nm):"), 4, 2)
        grid.addWidget(self.peak2GuessEdit, 4, 3)
        self.refIndexSpin = QtWidgets.QSpinBox()
        self.refIndexSpin.setRange(0, 0)
        self.refIndexSpin.setValue(0)
        self.refIndexSpin.setFixedWidth(90)
        self.maxShiftSpin = QtWidgets.QDoubleSpinBox()
        self.maxShiftSpin.setDecimals(2)
        self.maxShiftSpin.setRange(0.2, 20.0)
        self.maxShiftSpin.setValue(2.0)
        self.maxShiftSpin.setSingleStep(0.2)
        self.maxShiftSpin.setFixedWidth(90)
        self.fitPeaksBtn = QtWidgets.QPushButton("Fit Peaks")
        self.fitPeaksBtn.setFixedWidth(120)
        grid.addWidget(QtWidgets.QLabel("Ref frame idx:"), 4, 4)
        grid.addWidget(self.refIndexSpin, 4, 5)
        grid.addWidget(QtWidgets.QLabel("Max shift/frame (nm):"), 4, 6)
        grid.addWidget(self.maxShiftSpin, 4, 7)
        grid.addWidget(self.fitPeaksBtn, 5, 7)

        self.trplTminSpin = QtWidgets.QDoubleSpinBox()
        self.trplTminSpin.setDecimals(4)
        self.trplTminSpin.setRange(0.0, 1.0e9)
        self.trplTminSpin.setValue(0.0)
        self.trplTminSpin.setFixedWidth(100)
        self.trplTmaxSpin = QtWidgets.QDoubleSpinBox()
        self.trplTmaxSpin.setDecimals(4)
        self.trplTmaxSpin.setRange(0.0, 1.0e9)
        self.trplTmaxSpin.setValue(self._default_trpl_tmax())
        self.trplTmaxSpin.setFixedWidth(100)
        self.histYminEdit = QtWidgets.QLineEdit()
        self.histYminEdit.setPlaceholderText("auto")
        self.histYminEdit.setFixedWidth(80)
        self.histYmaxEdit = QtWidgets.QLineEdit()
        self.histYmaxEdit.setPlaceholderText("auto")
        self.histYmaxEdit.setFixedWidth(80)
        self.trplNormalizeChk = QtWidgets.QCheckBox("Normalize max")
        self.trplNormalizeChk.setChecked(True)
        self.trplLogChk = QtWidgets.QCheckBox("Log scale")
        self.trplLogChk.setChecked(False)
        grid.addWidget(QtWidgets.QLabel("TRPL t min/max ns:"), 6, 0)
        grid.addWidget(self.trplTminSpin, 6, 1)
        grid.addWidget(self.trplTmaxSpin, 6, 2)
        grid.addWidget(QtWidgets.QLabel("Hist y min/max:"), 6, 3)
        grid.addWidget(self.histYminEdit, 6, 4)
        grid.addWidget(self.histYmaxEdit, 6, 5)
        grid.addWidget(self.trplNormalizeChk, 6, 6)
        grid.addWidget(self.trplLogChk, 6, 7)

        root.addWidget(controls)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.liveView = OfflineLiveView(title="PL Image", default_linecut_width=self.defaults.linecut_width)
        self.liveToolbar = NavigationToolbar(self.liveView.canvas, self.liveView)
        lv_layout = self.liveView.layout()
        if lv_layout is not None:
            lv_layout.addWidget(self.liveToolbar, 2, 0, 1, 2)
        self.leftTabs = QtWidgets.QTabWidget()
        self.leftTabs.addTab(self.liveView, "PL Image")
        self.spadPanel = self._build_spad_panel()
        self.leftTabs.addTab(self.spadPanel, "SPAD Map")
        split.addWidget(self.leftTabs)
        self.mapView = MapPlotWidget("Linecut map", cmap_name="viridis", line_label="I")
        self.middleTabs = QtWidgets.QTabWidget()
        self.middleTabs.addTab(self.mapView, "PL Linecut Map")
        self.trplMapPanel = self._build_trpl_map_panel()
        self.historyPanel = self._build_history_panel()
        self.gaussianPanel = self._build_gaussian_panel()
        self.middleTabs.addTab(self.trplMapPanel, "TRPL Map")
        self.middleTabs.addTab(self.historyPanel, "Last Five Histograms")
        self.middleTabs.addTab(self.gaussianPanel, "Gaussian Fit")
        self.derivView = MapPlotWidget("Derivative map", cmap_name="coolwarm", line_label="dI/dx")
        self.fitWidthView = FitTrendPlotWidget("Lorentzian Width vs Sweep", "FWHM (meV)")
        self.fitEnergyView = FitTrendPlotWidget("Peak Position vs Sweep", "Energy (eV)")
        self.fitIntensityView = FitTrendPlotWidget("Peak Intensity vs Sweep", "Fitted Intensity (a.u.)")
        self.analysisTabs = QtWidgets.QTabWidget()
        self.analysisTabs.addTab(self.derivView, "Derivative")
        self.analysisTabs.addTab(self.fitWidthView, "Fit Width")
        self.analysisTabs.addTab(self.fitEnergyView, "Peak Energy")
        self.analysisTabs.addTab(self.fitIntensityView, "Peak Intensity")
        split.addWidget(self.middleTabs)
        split.addWidget(self.analysisTabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

        self.setCentralWidget(central)

        self.modeCombo.currentTextChanged.connect(self._on_mode_changed)
        self.browseFolderBtn.clicked.connect(self._on_browse_folder)
        self.loadFolderBtn.clicked.connect(self._on_load_folder)
        self.folderEdit.returnPressed.connect(self._on_load_folder)
        self.mainSlider.valueChanged.connect(self._on_main_changed)
        self.fixedSlider.valueChanged.connect(self._on_fixed_changed)
        self.cropApplyBtn.clicked.connect(self._on_apply_crop)
        self.derivMaxSpin.valueChanged.connect(self._on_deriv_scale_changed)
        self.peak1GuessEdit.editingFinished.connect(self._refresh_all)
        self.peak2GuessEdit.editingFinished.connect(self._refresh_all)
        self.fitPeaksBtn.clicked.connect(self._on_fit_peaks_clicked)
        self.liveView.linecut_changed.connect(lambda _: self._refresh_all())
        self.liveView.linecutWidthSpin.valueChanged.connect(lambda _: self._refresh_all())
        self.trplTimeSlider.valueChanged.connect(self._on_trpl_time_changed)
        self.trplTminSpin.valueChanged.connect(lambda _: self._refresh_trpl_views())
        self.trplTmaxSpin.valueChanged.connect(lambda _: self._refresh_trpl_views())
        self.histYminEdit.editingFinished.connect(self._update_history_plot)
        self.histYmaxEdit.editingFinished.connect(self._update_history_plot)
        self.trplNormalizeChk.toggled.connect(lambda _: self._refresh_trpl_views())
        self.trplLogChk.toggled.connect(lambda _: self._refresh_trpl_views())
        self.middleTabs.currentChanged.connect(lambda _: self._refresh_trpl_views())
        self.exportTrplMapBtn.clicked.connect(self._export_trpl_map_csv)
        self.exportHistBtn.clicked.connect(self._export_history_csv)
        self.exportSigmaBtn.clicked.connect(self._export_sigma2_csv)

    def _default_trpl_tmax(self) -> float:
        for rec in self.trpl_records:
            if rec.time_ns.size:
                return float(np.nanmax(rec.time_ns))
        return 100.0

    def _on_browse_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select PL/TRPL data folder",
            str(Path(self.folderEdit.text()).expanduser() if self.folderEdit.text().strip() else self.data_dir),
        )
        if not folder:
            return
        self.folderEdit.setText(folder)
        self._load_data_dir(Path(folder))

    def _on_load_folder(self) -> None:
        self._load_data_dir(Path(self.folderEdit.text().strip()).expanduser())

    def _load_data_dir(self, data_dir: Path) -> None:
        data_dir = data_dir.resolve()
        if not data_dir.is_dir():
            QtWidgets.QMessageBox.warning(self, "Load folder", f"Folder does not exist:\n{data_dir}")
            return
        try:
            frames, defaults = load_frames(data_dir)
            trpl_records = load_trpl_records(data_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load folder", f"Failed to read data folder:\n{exc}")
            return
        if not frames:
            QtWidgets.QMessageBox.warning(self, "Load folder", f"No .asc PL files found in:\n{data_dir}")
            return

        self._initializing = True
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

        for spin, value in (
            (self.cropTopSpin, self._crop[0]),
            (self.cropBottomSpin, self._crop[1]),
            (self.cropLeftSpin, self._crop[2]),
            (self.cropRightSpin, self._crop[3]),
        ):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        self.trplTmaxSpin.blockSignals(True)
        self.trplTmaxSpin.setValue(self._default_trpl_tmax())
        self.trplTmaxSpin.blockSignals(False)

        self._build_grid()
        self.folderEdit.setText(str(data_dir))
        self.folderStatusLbl.setText(f"{len(frames)} PL, {len(trpl_records)} TRPL")
        self._init_mode()
        self._on_apply_crop()
        self._initializing = False
        self._record_selection()
        self._refresh_trpl_views()
        self.statusBar().showMessage(f"Loaded {data_dir}")

    def _build_spad_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        self.fig_spad = Figure(figsize=(4, 5), dpi=100)
        self.canvas_spad = FigureCanvas(self.fig_spad)
        self.ax_spad = self.fig_spad.add_subplot(111)
        layout.addWidget(NavigationToolbar(self.canvas_spad, panel))
        layout.addWidget(self.canvas_spad, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Time:"))
        self.trplTimeSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.trplTimeSlider.setRange(0, 0)
        self.trplTimeSlider.setTracking(True)
        self.trplTimeLabel = QtWidgets.QLabel("t = -- ns | Pixel 11")
        self.trplTimeLabel.setMinimumWidth(180)
        row.addWidget(self.trplTimeSlider, 1)
        row.addWidget(self.trplTimeLabel)
        layout.addLayout(row)
        self.canvas_spad.mpl_connect("button_press_event", self._on_spad_map_click)
        return panel

    def _build_trpl_map_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addStretch()
        self.exportTrplMapBtn = QtWidgets.QPushButton("Export TRPL Map CSV")
        ctrl.addWidget(self.exportTrplMapBtn)
        layout.addLayout(ctrl)
        self.fig_trpl_map = Figure(figsize=(5.5, 5), dpi=100)
        self.canvas_trpl_map = FigureCanvas(self.fig_trpl_map)
        self.ax_trpl_map = self.fig_trpl_map.add_subplot(111)
        layout.addWidget(NavigationToolbar(self.canvas_trpl_map, panel))
        layout.addWidget(self.canvas_trpl_map, 1)
        return panel

    def _build_history_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addStretch()
        self.exportHistBtn = QtWidgets.QPushButton("Export Histograms CSV")
        ctrl.addWidget(self.exportHistBtn)
        layout.addLayout(ctrl)
        self.fig_history = Figure(figsize=(5.5, 5), dpi=100)
        self.canvas_history = FigureCanvas(self.fig_history)
        self.ax_history = self.fig_history.add_subplot(111)
        layout.addWidget(NavigationToolbar(self.canvas_history, panel))
        layout.addWidget(self.canvas_history, 1)
        return panel

    def _build_gaussian_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addStretch()
        self.exportSigmaBtn = QtWidgets.QPushButton("Export Sigma^2 CSV")
        ctrl.addWidget(self.exportSigmaBtn)
        layout.addLayout(ctrl)
        self.fig_gaussian = Figure(figsize=(5.5, 5), dpi=100)
        self.canvas_gaussian = FigureCanvas(self.fig_gaussian)
        gs = self.fig_gaussian.add_gridspec(1, 2, wspace=0.35)
        self.ax_gauss_map = self.fig_gaussian.add_subplot(gs[0, 0])
        self.ax_sigma2 = self.fig_gaussian.add_subplot(gs[0, 1])
        layout.addWidget(NavigationToolbar(self.canvas_gaussian, panel))
        layout.addWidget(self.canvas_gaussian, 1)
        return panel

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
        self.mainSlider.blockSignals(True)
        self.fixedSlider.blockSignals(True)
        self.refIndexSpin.blockSignals(True)
        self.mainSlider.setRange(0, max(0, x.size - 1))
        self.fixedSlider.setRange(0, max(0, fixed.size - 1))
        self.refIndexSpin.setRange(0, max(0, x.size - 1))
        self._main_idx = max(0, min(self._main_idx, max(0, x.size - 1)))
        self._fixed_idx = max(0, min(self._fixed_idx, max(0, fixed.size - 1)))
        self.mainSlider.setValue(self._main_idx)
        self.fixedSlider.setValue(self._fixed_idx)
        self.refIndexSpin.setValue(self._main_idx)
        self.mainSlider.blockSignals(False)
        self.fixedSlider.blockSignals(False)
        self.refIndexSpin.blockSignals(False)
        self.fixedValueLbl.setText(f"{fixed_name}: --")
        self._refresh_all()

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
            candidates = [(k, v) for k, v in self._grid.items() if k[0] == p]
            if not candidates:
                return None
            target_v = float(self._gates[self._fixed_idx]) if self._gates.size else 0.0
            candidates.sort(key=lambda kv: abs(float(kv[0][1]) - target_v))
            return candidates[0][1]
        vf = key[1]
        candidates = [(k, v) for k, v in self._grid.items() if k[1] == vf]
        if not candidates:
            return None
        target_p = float(self._powers[self._fixed_idx]) if self._powers.size else 0.0
        candidates.sort(key=lambda kv: abs(float(kv[0][0]) - target_p))
        return candidates[0][1]

    def _get_trpl_record_for_display(self) -> Optional[TRPLRecord]:
        key = self._get_key_for_display()
        rec = self._trpl_grid.get(key)
        if rec is not None:
            return rec
        if self._current_mode == self.MODE_POWER:
            p = key[0]
            candidates = [(k, v) for k, v in self._trpl_grid.items() if k[0] == p]
            if not candidates:
                return None
            target_v = float(self._gates[self._fixed_idx]) if self._gates.size else 0.0
            candidates.sort(key=lambda kv: abs(float(kv[0][1]) - target_v))
            return candidates[0][1]
        vf = key[1]
        candidates = [(k, v) for k, v in self._trpl_grid.items() if k[1] == vf]
        if not candidates:
            return None
        target_p = float(self._powers[self._fixed_idx]) if self._powers.size else 0.0
        candidates.sort(key=lambda kv: abs(float(kv[0][0]) - target_p))
        return candidates[0][1]

    def _get_trpl_record_for_key(self, power_key: float, gate_key: float) -> Optional[TRPLRecord]:
        return self._trpl_grid.get((float(power_key), float(gate_key)))

    def _current_selection_key(self) -> Tuple[float, float, int]:
        p, vf = self._get_key_for_display()
        return float(p if p is not None else 0.0), float(vf if vf is not None else 0.0), int(self._selected_spad_pixel)

    def _record_selection(self) -> None:
        if getattr(self, "_initializing", False):
            return
        item = self._current_selection_key()
        if self._selection_history and self._selection_history[-1] == item:
            return
        self._selection_history.append(item)
        self._selection_history = self._selection_history[-5:]
        if hasattr(self, "canvas_history"):
            self._update_history_plot()

    def _spad_nearest_pixel(self, x: float, y: float, max_r: float = 0.42) -> Optional[int]:
        coords = self._spad_map_coords
        d2 = (coords[:, 0] - float(x)) ** 2 + (coords[:, 1] - float(y)) ** 2
        idx = int(np.argmin(d2))
        if float(d2[idx]) <= float(max_r) ** 2:
            return idx
        return None

    def _current_linecut_row_raw(self, shape_hw: Tuple[int, int]) -> int:
        h = int(shape_hw[0])
        row_disp = self.liveView.linecut_row()
        if row_disp is None:
            return int(h // 2)
        row_disp = max(0, min(int(row_disp), h - 1))
        return h - 1 - row_disp

    def _cropped_frame_data(self, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        cached_img = self._cropped_image_cache.get(fr.path)
        cached_wl = self._cropped_wl_cache.get(fr.path)
        if cached_img is not None and cached_wl is not None:
            return cached_img, cached_wl
        img = apply_crop(fr.image, self._crop)
        wl = fr.wavelength_nm
        if wl is None:
            wl = np.arange(fr.image.shape[1], dtype=float)
        wl_c = crop_axis(wl, self._crop, fr.image.shape[1])
        self._cropped_image_cache[fr.path] = img
        self._cropped_wl_cache[fr.path] = wl_c
        return img, wl_c

    def _blank_cropped(self) -> Tuple[np.ndarray, np.ndarray]:
        blank = apply_crop(self._blank_image, self._crop)
        wl = crop_axis(self._blank_wl, self._crop, int(self._blank_image.shape[1]))
        return blank, wl

    def _on_apply_crop(self) -> None:
        self._crop = (
            int(self.cropTopSpin.value()),
            int(self.cropBottomSpin.value()),
            int(self.cropLeftSpin.value()),
            int(self.cropRightSpin.value()),
        )
        self._cropped_image_cache.clear()
        self._cropped_wl_cache.clear()
        self.liveView.set_crop(*self._crop)
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._update_shared_image()
        self._update_maps()
        self._refresh_trpl_views()

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

    def _trpl_time_bounds(self) -> Tuple[float, float]:
        tmin = float(self.trplTminSpin.value())
        tmax = float(self.trplTmaxSpin.value())
        if tmax < tmin:
            tmin, tmax = tmax, tmin
        return tmin, tmax

    def _selected_time_index(self, rec: TRPLRecord) -> int:
        n = int(rec.time_ns.size)
        if n <= 0:
            return 0
        return int(np.clip(self.trplTimeSlider.value(), 0, n - 1))

    def _sync_time_slider(self, rec: Optional[TRPLRecord]) -> None:
        if rec is None or rec.time_ns.size == 0:
            self.trplTimeSlider.blockSignals(True)
            self.trplTimeSlider.setRange(0, 0)
            self.trplTimeSlider.setValue(0)
            self.trplTimeSlider.blockSignals(False)
            self.trplTimeLabel.setText(f"t = -- ns | Pixel {self._selected_spad_pixel}")
            return
        max_idx = int(rec.time_ns.size - 1)
        val = int(np.clip(self.trplTimeSlider.value(), 0, max_idx))
        self.trplTimeSlider.blockSignals(True)
        self.trplTimeSlider.setRange(0, max_idx)
        self.trplTimeSlider.setValue(val)
        self.trplTimeSlider.blockSignals(False)
        self.trplTimeLabel.setText(f"t = {float(rec.time_ns[val]):.4g} ns | Pixel {self._selected_spad_pixel}")

    def _format_selection_label(self, p: float, vf: float, pix: int) -> str:
        p_uW = p * 1e6
        if self._can_power_mode and self._can_gate_mode:
            return f"P={p_uW:.4g} uW, Vf={vf:.4g} V, px {pix}"
        if self._can_power_mode:
            return f"P={p_uW:.4g} uW, px {pix}"
        if self._can_gate_mode:
            return f"Vf={vf:.4g} V, px {pix}"
        return f"px {pix}"

    def _column_scale(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64).copy()
        if not self.trplNormalizeChk.isChecked():
            return arr
        for j in range(arr.shape[1]):
            m = float(np.nanmax(arr[:, j])) if np.any(np.isfinite(arr[:, j])) else 0.0
            if m > 0:
                arr[:, j] /= m
            else:
                arr[:, j] = np.nan
        return arr

    def _imshow_norm_kwargs(self, data: np.ndarray) -> dict:
        if not self.trplLogChk.isChecked():
            if self.trplNormalizeChk.isChecked():
                return {"vmin": 0.0, "vmax": 1.0}
            return {}
        positive = np.asarray(data, dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if positive.size == 0:
            return {}
        vmax = float(np.nanmax(positive))
        vmin = max(float(np.nanmin(positive)), vmax * 1e-4, 1e-12)
        if self.trplNormalizeChk.isChecked():
            vmax = 1.0
            vmin = max(vmin, 1e-4)
        return {"norm": colors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0))}

    def _on_trpl_time_changed(self, _value: int) -> None:
        rec = self._get_trpl_record_for_display()
        self._sync_time_slider(rec)
        self._update_spad_map()
        if hasattr(self, "middleTabs") and self.middleTabs.tabText(self.middleTabs.currentIndex()) == "Gaussian Fit":
            self._update_gaussian_plot()

    def _on_spad_map_click(self, event) -> None:
        if event.inaxes != self.ax_spad or event.xdata is None or event.ydata is None:
            return
        pix = self._spad_nearest_pixel(float(event.xdata), float(event.ydata))
        if pix is None:
            return
        self._selected_spad_pixel = int(pix)
        self._record_selection()
        self._refresh_trpl_views()

    def _refresh_trpl_views(self) -> None:
        if not hasattr(self, "canvas_spad"):
            return
        rec = self._get_trpl_record_for_display()
        self._sync_time_slider(rec)
        self._update_spad_map()
        tab = self.middleTabs.tabText(self.middleTabs.currentIndex()) if hasattr(self, "middleTabs") else ""
        if tab == "TRPL Map":
            self._update_trpl_map()
        elif tab == "Last Five Histograms":
            self._update_history_plot()
        elif tab == "Gaussian Fit":
            self._update_gaussian_plot()

    def _update_spad_map(self) -> None:
        rec = self._get_trpl_record_for_display()
        self.fig_spad.clear()
        self.ax_spad = self.fig_spad.add_subplot(111)
        if rec is None or rec.counts.size == 0:
            self.ax_spad.text(0.5, 0.5, "No TRPL CSV for current selection", ha="center", va="center", transform=self.ax_spad.transAxes)
            self.ax_spad.set_axis_off()
            self.canvas_spad.draw_idle()
            return
        idx = self._selected_time_index(rec)
        vals = np.asarray(rec.counts[idx, :], dtype=np.float64)
        sc = self.ax_spad.scatter(
            self._spad_map_coords[:, 0],
            self._spad_map_coords[:, 1],
            c=vals,
            s=620,
            cmap="viridis",
            edgecolors="#111827",
            linewidths=0.8,
        )
        sx, sy = SPAD_PIXEL_COORDS[int(self._selected_spad_pixel)]
        self.ax_spad.scatter([sx], [sy], s=980, facecolors="none", edgecolors="#22d3ee", linewidths=2.2)
        for pix, (x, y) in SPAD_PIXEL_COORDS.items():
            self.ax_spad.text(x, y, str(pix), ha="center", va="center", color="#f7f7f7", fontsize=8.5, fontweight="bold")
        self.ax_spad.set_title(f"SPAD counts @ {float(rec.time_ns[idx]):.4g} ns")
        self.ax_spad.set_aspect("equal")
        self.ax_spad.set_xlim(-0.7, 4.7)
        self.ax_spad.set_ylim(-0.7, 4.7)
        self.ax_spad.set_xticks([])
        self.ax_spad.set_yticks([])
        self.fig_spad.colorbar(sc, ax=self.ax_spad, fraction=0.05, pad=0.04)
        self.trplTimeLabel.setText(f"t = {float(rec.time_ns[idx]):.4g} ns | Pixel {self._selected_spad_pixel}")
        self.canvas_spad.draw_idle()

    def _records_along_current_axis(self) -> Tuple[np.ndarray, List[Optional[TRPLRecord]], str]:
        if self._current_mode == self.MODE_GATE:
            p = self._norm_power_key(float(self._powers[self._fixed_idx])) if self._powers.size else self._default_power_key
            xvals = np.asarray(self._gates, dtype=float)
            records = [self._get_trpl_record_for_key(float(p), self._norm_gate_key(float(vf))) for vf in xvals]
            return xvals, records, "Front gate (V)"
        vf = self._norm_gate_key(float(self._gates[self._fixed_idx])) if self._gates.size else self._default_gate_key
        pvals = np.asarray(self._powers, dtype=float)
        records = [self._get_trpl_record_for_key(self._norm_power_key(float(p)), float(vf)) for p in pvals]
        return pvals * 1e6, records, "Power (uW)"

    def _trpl_map_data(self) -> Optional[dict]:
        xvals, records, xlabel = self._records_along_current_axis()
        pixel = int(self._selected_spad_pixel)
        tmin, tmax = self._trpl_time_bounds()
        cols = []
        x_used = []
        time_axis = None
        for x, rec in zip(xvals, records):
            if rec is None or rec.counts.shape[1] <= pixel:
                continue
            t = np.asarray(rec.time_ns, dtype=np.float64)
            y = np.asarray(rec.counts[:, pixel], dtype=np.float64)
            mask = (t >= tmin) & (t <= tmax)
            if np.count_nonzero(mask) < 1:
                continue
            if time_axis is None:
                time_axis = t[mask]
                col = y[mask]
            else:
                col = np.interp(time_axis, t, y, left=np.nan, right=np.nan)
            cols.append(col)
            x_used.append(float(x))
        if not cols or time_axis is None:
            return None
        raw = np.column_stack(cols)
        return {
            "x": np.asarray(x_used, dtype=np.float64),
            "time_ns": np.asarray(time_axis, dtype=np.float64),
            "raw": raw,
            "plot": self._column_scale(raw),
            "xlabel": xlabel,
            "pixel": pixel,
        }

    def _update_trpl_map(self) -> None:
        self.fig_trpl_map.clear()
        self.ax_trpl_map = self.fig_trpl_map.add_subplot(111)
        map_data = self._trpl_map_data()
        if map_data is None:
            self.ax_trpl_map.text(0.5, 0.5, "No TRPL data for selected axis/range", ha="center", va="center", transform=self.ax_trpl_map.transAxes)
            self.canvas_trpl_map.draw_idle()
            return
        data = map_data["plot"]
        x_used = map_data["x"]
        time_axis = map_data["time_ns"]
        xlabel = str(map_data["xlabel"])
        pixel = int(map_data["pixel"])
        xmin = float(np.nanmin(x_used))
        xmax = float(np.nanmax(x_used))
        if xmax <= xmin:
            xmax = xmin + 1.0
        kwargs = self._imshow_norm_kwargs(data)
        im = self.ax_trpl_map.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            extent=(xmin, xmax, float(time_axis[0]), float(time_axis[-1])),
            **kwargs,
        )
        self.fig_trpl_map.colorbar(im, ax=self.ax_trpl_map, fraction=0.05, pad=0.04)
        self.ax_trpl_map.set_title(f"TRPL vs {xlabel} (pixel {pixel})")
        self.ax_trpl_map.set_xlabel(xlabel)
        self.ax_trpl_map.set_ylabel("Time (ns)")
        self.canvas_trpl_map.draw_idle()

    def _update_history_plot(self) -> None:
        self.ax_history.clear()
        ymin = self._parse_optional_edit(self.histYminEdit)
        ymax = self._parse_optional_edit(self.histYmaxEdit)
        plotted = 0
        for item in self._history_series():
            self.ax_history.plot(item["time_ns"], item["plot_counts"], linewidth=1.2, label=item["label"])
            plotted += 1
        if plotted == 0:
            self.ax_history.text(0.5, 0.5, "No selected TRPL histograms yet", ha="center", va="center", transform=self.ax_history.transAxes)
        else:
            self.ax_history.legend(loc="best", fontsize=8)
        self.ax_history.set_title("Last five selected histograms")
        self.ax_history.set_xlabel("Time (ns)")
        self.ax_history.set_ylabel("Normalized counts" if self.trplNormalizeChk.isChecked() else "Counts")
        self.ax_history.set_yscale("log" if self.trplLogChk.isChecked() else "linear")
        if ymin is not None or ymax is not None:
            self.ax_history.set_ylim(ymin, ymax)
        self.ax_history.grid(True, alpha=0.25)
        self.canvas_history.draw_idle()

    def _history_series(self) -> List[dict]:
        tmin, tmax = self._trpl_time_bounds()
        out = []
        for sel_idx, (p, vf, pix) in enumerate(self._selection_history[-5:], start=1):
            rec = self._get_trpl_record_for_key(p, vf)
            if rec is None or rec.counts.shape[1] <= pix:
                continue
            t = np.asarray(rec.time_ns, dtype=np.float64)
            y = np.asarray(rec.counts[:, pix], dtype=np.float64)
            mask = (t >= tmin) & (t <= tmax)
            if not np.any(mask):
                continue
            raw = y[mask].astype(np.float64)
            plotted = raw.copy()
            norm_factor = 1.0
            if self.trplNormalizeChk.isChecked():
                m = float(np.nanmax(plotted)) if np.any(np.isfinite(plotted)) else 0.0
                if m > 0:
                    norm_factor = m
                    plotted = plotted / m
            out.append(
                {
                    "selection_index": sel_idx,
                    "power_w": float(p),
                    "front_gate_v": float(vf),
                    "pixel": int(pix),
                    "time_ns": t[mask],
                    "raw_counts": raw,
                    "plot_counts": plotted,
                    "norm_factor": float(norm_factor),
                    "label": self._format_selection_label(p, vf, pix),
                }
            )
        return out

    def _fit_frame(self, frame: np.ndarray) -> Optional[SpadFitResult]:
        return fit_spad_gaussian_2d(self._spad_fit_x_um, self._spad_fit_y_um, np.asarray(frame, dtype=np.float64))

    def _sigma2_trace(self, *, sampled: bool) -> Tuple[Optional[TRPLRecord], np.ndarray, List[SpadFitResult]]:
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            return None, np.array([], dtype=np.float64), []
        tmin, tmax = self._trpl_time_bounds()
        idxs = np.flatnonzero((rec.time_ns >= tmin) & (rec.time_ns <= tmax))
        if sampled and idxs.size > 80:
            stride = int(np.ceil(idxs.size / 80.0))
            idxs = idxs[::stride]
        times = []
        fits = []
        for k in idxs:
            fit = self._fit_frame(rec.counts[int(k), :])
            if fit is None:
                continue
            times.append(float(rec.time_ns[int(k)]))
            fits.append(fit)
        return rec, np.asarray(times, dtype=np.float64), fits

    def _update_gaussian_plot(self) -> None:
        self.fig_gaussian.clear()
        gs = self.fig_gaussian.add_gridspec(1, 2, wspace=0.35)
        self.ax_gauss_map = self.fig_gaussian.add_subplot(gs[0, 0])
        self.ax_sigma2 = self.fig_gaussian.add_subplot(gs[0, 1])
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            self.ax_gauss_map.text(0.5, 0.5, "No TRPL CSV for current selection", ha="center", va="center", transform=self.ax_gauss_map.transAxes)
            self.canvas_gaussian.draw_idle()
            return
        idx = self._selected_time_index(rec)
        frame = np.asarray(rec.counts[idx, :], dtype=np.float64)
        t_sel = float(rec.time_ns[idx])
        fit = self._fit_frame(frame)
        if fit is None:
            sc = self.ax_gauss_map.scatter(self._spad_fit_x_um, self._spad_fit_y_um, c=frame, cmap="viridis", edgecolors="k", s=45)
            self.fig_gaussian.colorbar(sc, ax=self.ax_gauss_map, fraction=0.05, pad=0.04)
            self.ax_gauss_map.set_title(f"Gaussian fit unavailable @ {t_sel:.4g} ns")
        else:
            popt = fit.popt
            xpad = SPAD_PITCH_X_UM * 0.35
            ypad = SPAD_PITCH_Y_UM * 0.35
            gx = np.linspace(float(np.min(self._spad_fit_x_um) - xpad), float(np.max(self._spad_fit_x_um) + xpad), 220)
            gy = np.linspace(float(np.min(self._spad_fit_y_um) - ypad), float(np.max(self._spad_fit_y_um) + ypad), 220)
            gxx, gyy = np.meshgrid(gx, gy)
            zfit = popt[0] * np.exp(-(((gxx - popt[1]) ** 2) / (2.0 * popt[3] ** 2) + ((gyy - popt[2]) ** 2) / (2.0 * popt[4] ** 2))) + popt[5]
            zmin = float(np.nanmin(zfit))
            zmax = float(np.nanmax(zfit))
            if zmax <= zmin:
                zmax = zmin + 1.0
            levels = np.linspace(zmin, zmax, 80)
            cf = self.ax_gauss_map.contourf(gxx, gyy, zfit, levels=levels, cmap="viridis")
            self.ax_gauss_map.scatter(self._spad_fit_x_um, self._spad_fit_y_um, c=frame, cmap="viridis", edgecolors="k", s=45, vmin=zmin, vmax=zmax)
            self.ax_gauss_map.plot(float(popt[1]), float(popt[2]), "wx", markersize=8, markeredgewidth=2)
            self.fig_gaussian.colorbar(cf, ax=self.ax_gauss_map, fraction=0.05, pad=0.04)
            self.ax_gauss_map.set_title(f"Fit @ {t_sel:.4g} ns, sigma^2={fit.sigma_eq ** 2:.4g}")
        self.ax_gauss_map.set_xlabel("x (um)")
        self.ax_gauss_map.set_ylabel("y (um)")
        self.ax_gauss_map.set_aspect("equal")
        self.ax_gauss_map.invert_yaxis()

        _, ts, fits = self._sigma2_trace(sampled=True)
        if ts.size:
            sigma2 = [float(f.sigma_eq ** 2) for f in fits]
            self.ax_sigma2.plot(ts, sigma2, "-o", markersize=2.5, linewidth=1.1, color="#2563eb")
            self.ax_sigma2.axvline(t_sel, color="#111827", linestyle="--", linewidth=1.0)
        else:
            self.ax_sigma2.text(0.5, 0.5, "No valid sigma^2 fits in range", ha="center", va="center", transform=self.ax_sigma2.transAxes)
        self.ax_sigma2.set_title("sigma^2 vs time")
        self.ax_sigma2.set_xlabel("Time (ns)")
        self.ax_sigma2.set_ylabel("sigma^2 (um^2)")
        self.ax_sigma2.grid(True, alpha=0.25)
        self.canvas_gaussian.draw_idle()

    def _choose_export_path(self, default_name: str) -> Optional[Path]:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            str(self.data_dir / default_name),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return None
        out = Path(path)
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        return out

    def _export_trpl_map_csv(self) -> None:
        map_data = self._trpl_map_data()
        if map_data is None:
            QtWidgets.QMessageBox.warning(self, "Export TRPL map", "No TRPL map data to export.")
            return
        path = self._choose_export_path("trpl_map_export.csv")
        if path is None:
            return
        x = np.asarray(map_data["x"], dtype=np.float64)
        time_ns = np.asarray(map_data["time_ns"], dtype=np.float64)
        raw = np.asarray(map_data["raw"], dtype=np.float64)
        plotted = np.asarray(map_data["plot"], dtype=np.float64)
        xlabel = str(map_data["xlabel"])
        pixel = int(map_data["pixel"])
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["# type", "trpl_map"])
                writer.writerow(["# x_label", xlabel])
                writer.writerow(["# selected_pixel", pixel])
                writer.writerow(["# normalized_to_max", bool(self.trplNormalizeChk.isChecked())])
                writer.writerow(["# log_scale_display", bool(self.trplLogChk.isChecked())])
                writer.writerow([])
                writer.writerow(["time_ns"] + [f"plot_{xlabel}={v:.10g}" for v in x])
                for i, t in enumerate(time_ns):
                    writer.writerow([f"{t:.10g}"] + [f"{v:.10g}" if np.isfinite(v) else "" for v in plotted[i, :]])
                writer.writerow([])
                writer.writerow(["time_ns"] + [f"raw_{xlabel}={v:.10g}" for v in x])
                for i, t in enumerate(time_ns):
                    writer.writerow([f"{t:.10g}"] + [f"{v:.10g}" if np.isfinite(v) else "" for v in raw[i, :]])
            self.statusBar().showMessage(f"Exported TRPL map CSV: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export TRPL map", f"Failed to write CSV:\n{exc}")

    def _export_history_csv(self) -> None:
        series = self._history_series()
        if not series:
            QtWidgets.QMessageBox.warning(self, "Export histograms", "No selected histogram data to export.")
            return
        path = self._choose_export_path("selected_histograms_export.csv")
        if path is None:
            return
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["# type", "selected_histograms"])
                writer.writerow(["# normalized_to_max", bool(self.trplNormalizeChk.isChecked())])
                writer.writerow(["# log_scale_display", bool(self.trplLogChk.isChecked())])
                writer.writerow([])
                writer.writerow(
                    [
                        "selection_index",
                        "label",
                        "power_w",
                        "front_gate_v",
                        "pixel",
                        "norm_factor",
                        "time_ns",
                        "raw_counts",
                        "plot_counts",
                    ]
                )
                for item in series:
                    for t, raw, plotted in zip(item["time_ns"], item["raw_counts"], item["plot_counts"]):
                        writer.writerow(
                            [
                                item["selection_index"],
                                item["label"],
                                f"{float(item['power_w']):.12g}",
                                f"{float(item['front_gate_v']):.12g}",
                                int(item["pixel"]),
                                f"{float(item['norm_factor']):.12g}",
                                f"{float(t):.12g}",
                                f"{float(raw):.12g}",
                                f"{float(plotted):.12g}",
                            ]
                        )
            self.statusBar().showMessage(f"Exported selected histograms CSV: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export histograms", f"Failed to write CSV:\n{exc}")

    def _export_sigma2_csv(self) -> None:
        rec = self._get_trpl_record_for_display()
        if rec is None or rec.counts.size == 0:
            QtWidgets.QMessageBox.warning(self, "Export sigma^2", "No TRPL record selected.")
            return
        path = self._choose_export_path("sigma2_vs_time_export.csv")
        if path is None:
            return
        self.statusBar().showMessage("Computing sigma^2 export...")
        QtWidgets.QApplication.processEvents()
        _, times, fits = self._sigma2_trace(sampled=True)
        if times.size == 0:
            QtWidgets.QMessageBox.warning(self, "Export sigma^2", "No valid Gaussian fits in the selected time range.")
            return
        p, vf = self._get_key_for_display()
        selected_idx = self._selected_time_index(rec)
        selected_time = float(rec.time_ns[selected_idx]) if rec.time_ns.size else np.nan
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["# type", "sigma2_vs_time"])
                writer.writerow(["# power_w", f"{float(p if p is not None else 0.0):.12g}"])
                writer.writerow(["# front_gate_v", f"{float(vf if vf is not None else 0.0):.12g}"])
                writer.writerow(["# selected_time_ns", f"{selected_time:.12g}"])
                writer.writerow(["# source_file", str(rec.path)])
                writer.writerow([])
                writer.writerow(["time_ns", "sigma_um", "sigma2_um2", "amplitude", "x0_um", "y0_um", "sx_um", "sy_um", "offset"])
                for t, fit in zip(times, fits):
                    popt = fit.popt
                    writer.writerow(
                        [
                            f"{float(t):.12g}",
                            f"{float(fit.sigma_eq):.12g}",
                            f"{float(fit.sigma_eq ** 2):.12g}",
                            f"{float(popt[0]):.12g}",
                            f"{float(popt[1]):.12g}",
                            f"{float(popt[2]):.12g}",
                            f"{float(popt[3]):.12g}",
                            f"{float(popt[4]):.12g}",
                            f"{float(popt[5]):.12g}",
                        ]
                    )
            self.statusBar().showMessage(f"Exported sigma^2 CSV: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export sigma^2", f"Failed to write CSV:\n{exc}")

    def _update_shared_image(self) -> None:
        def _fmt(v: Optional[float], spec: str = ".4g") -> str:
            if v is None:
                return "na"
            try:
                fv = float(v)
            except Exception:
                return "na"
            if not np.isfinite(fv):
                return "na"
            return format(fv, spec)

        fr = self._get_frame_for_display()
        if fr is None:
            self.liveView.set_wavelength_axis(self._blank_wl)
            self.liveView.update_frame({"image": self._blank_image, "image8": None})
            self.liveView.set_image_title("PL Image (missing)")
            self.mainValueLbl.setText("Missing frame")
            return

        self.liveView.set_wavelength_axis(fr.wavelength_nm)
        self.liveView.update_frame({"image": fr.image, "image8": None})
        p_uW = float(fr.power_w * 1e6) if fr.power_w is not None else np.nan
        self.liveView.set_image_title(
            f"PL Image (P={_fmt(p_uW, '.3f')} uW, Vf={_fmt(fr.v_front)} V, Vb={_fmt(fr.v_back)} V)"
        )
        if self._current_mode == self.MODE_POWER:
            self.mainValueLbl.setText(f"Power: {_fmt(p_uW, '.3f')} uW")
            fixed_val = float(self._gates[self._fixed_idx]) if self._gates.size else np.nan
            fixed_back = np.nan
            if self._backs_for_gate.size and self._fixed_idx < self._backs_for_gate.size:
                fixed_back = float(self._backs_for_gate[self._fixed_idx])
            self.fixedValueLbl.setText(f"Front gate: {_fmt(fixed_val)} V (Back ~ {_fmt(fixed_back)} V)")
        else:
            self.mainValueLbl.setText(f"Front gate: {_fmt(fr.v_front)} V (Back {_fmt(fr.v_back)} V)")
            fixed_p = float(self._powers[self._fixed_idx] * 1e6) if self._powers.size else np.nan
            self.fixedValueLbl.setText(f"Power: {_fmt(fixed_p, '.3f')} uW")

    def _build_map_for_power_mode(self, row_raw: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_uW = self._powers * 1e6
        fixed_v = self._norm_gate_key(float(self._gates[self._fixed_idx])) if self._gates.size else self._default_gate_key
        y_ref = None
        for p in self._powers:
            fr = self._grid.get((self._norm_power_key(float(p)), fixed_v))
            if fr is not None:
                _, wl = self._cropped_frame_data(fr)
                y_ref = np.asarray(wl, dtype=float)
                break
        if y_ref is None:
            _, wl = self._blank_cropped()
            y_ref = np.asarray(wl, dtype=float)
        cols = []
        for p in self._powers:
            fr = self._grid.get((self._norm_power_key(float(p)), fixed_v))
            if fr is None:
                cols.append(np.full(y_ref.shape, np.nan, dtype=float))
                continue
            img, wl = self._cropped_frame_data(fr)
            lc = linecut_horizontal(img, row_raw, width)
            if lc is None:
                cols.append(np.full(y_ref.shape, np.nan, dtype=float))
                continue
            cols.append(np.asarray(resample_linecut(wl, lc, y_ref), dtype=float))

        map_data = np.stack(cols, axis=1) if cols else np.full((y_ref.size, x_uW.size), np.nan, dtype=float)
        deriv = derivative_dlogp_nan(map_data, self._powers)
        return x_uW, y_ref, map_data, deriv

    def _build_map_for_gate_mode(self, row_raw: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_vf = self._gates
        fixed_p = self._norm_power_key(float(self._powers[self._fixed_idx])) if self._powers.size else self._default_power_key
        y_ref = None
        for vf in self._gates:
            fr = self._grid.get((fixed_p, self._norm_gate_key(float(vf))))
            if fr is not None:
                _, wl = self._cropped_frame_data(fr)
                y_ref = np.asarray(wl, dtype=float)
                break
        if y_ref is None:
            _, wl = self._blank_cropped()
            y_ref = np.asarray(wl, dtype=float)
        cols = []
        for vf in self._gates:
            fr = self._grid.get((fixed_p, self._norm_gate_key(float(vf))))
            if fr is None:
                cols.append(np.full(y_ref.shape, np.nan, dtype=float))
                continue
            img, wl = self._cropped_frame_data(fr)
            lc = linecut_horizontal(img, row_raw, width)
            if lc is None:
                cols.append(np.full(y_ref.shape, np.nan, dtype=float))
                continue
            cols.append(np.asarray(resample_linecut(wl, lc, y_ref), dtype=float))

        map_data = np.stack(cols, axis=1) if cols else np.full((y_ref.size, x_vf.size), np.nan, dtype=float)
        deriv = derivative_vs_x_nan(map_data, x_vf)
        return x_vf, y_ref, map_data, deriv

    def _read_peak_guesses(self) -> Tuple[float, float]:
        p1 = 855.0
        p2 = 835.0
        try:
            val = float(self.peak1GuessEdit.text().strip())
            if np.isfinite(val):
                p1 = val
        except Exception:
            pass
        try:
            val = float(self.peak2GuessEdit.text().strip())
            if np.isfinite(val):
                p2 = val
        except Exception:
            pass
        return float(p1), float(p2)

    def _clear_fit_cache(self) -> None:
        self._fit_result = None
        self._fit_map_version = -1

    def _on_fit_peaks_clicked(self) -> None:
        if self._last_x is None or self._last_y_wl is None or self._last_map_data is None:
            self.fitWidthView.clear_data("No map data", x_label=self._last_x_label or "X")
            self.fitEnergyView.clear_data("No map data", x_label=self._last_x_label or "X")
            self.fitIntensityView.clear_data("No map data", x_label=self._last_x_label or "X")
            return

        p1_guess, p2_guess = self._read_peak_guesses()
        try:
            ref_idx = int(self.refIndexSpin.value())
        except Exception:
            ref_idx = 0
        try:
            max_shift_nm = float(self.maxShiftSpin.value())
        except Exception:
            max_shift_nm = 2.0

        fit = fit_two_peak_map_sequential(
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map_data, dtype=float),
            ref_idx=ref_idx,
            peak1_guess_nm=float(p1_guess),
            peak2_guess_nm=float(p2_guess),
            max_shift_nm=float(max_shift_nm),
        )
        self._fit_result = fit
        self._fit_map_version = int(self._map_version)
        self._update_fit_trend_tabs(
            np.asarray(self._last_x, dtype=float),
            str(self._last_x_label or "X"),
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map_data, dtype=float),
        )

    def _update_fit_trend_tabs(self, x: np.ndarray, x_label: str, y_wl: np.ndarray, map_data: np.ndarray) -> None:
        if map_data.ndim != 2 or y_wl.size != map_data.shape[0]:
            self.fitWidthView.clear_data("No data", x_label=x_label)
            self.fitEnergyView.clear_data("No data", x_label=x_label)
            self.fitIntensityView.clear_data("No data", x_label=x_label)
            return
        if self._fit_result is None or int(self._fit_map_version) != int(self._map_version):
            msg = "Click 'Fit Peaks' to run / refresh fits"
            self.fitWidthView.clear_data(msg, x_label=x_label)
            self.fitEnergyView.clear_data(msg, x_label=x_label)
            self.fitIntensityView.clear_data(msg, x_label=x_label)
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

        self.fitWidthView.set_data(
            x,
            w1_mev,
            w2_mev,
            x_label=x_label,
            label_1="Peak 1",
            label_2="Peak 2 (optional)",
        )
        self.fitEnergyView.set_data(
            x,
            e1,
            e2,
            x_label=x_label,
            label_1="Peak 1",
            label_2="Peak 2 (optional)",
        )
        self.fitIntensityView.set_data(
            x,
            a1,
            a2,
            x_label=x_label,
            label_1="Peak 1",
            label_2="Peak 2 (optional)",
        )

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
        width = int(self.liveView.linecut_width())
        linecut = linecut_horizontal(img, row_raw, width)
        if linecut is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        wl = np.asarray(wl, dtype=float).ravel()
        linecut = np.asarray(linecut, dtype=float).ravel()
        if wl.size != linecut.size or wl.size < 8:
            self.liveView.clear_linecut_fit_overlay()
            return
        p1_guess, p2_guess = self._read_peak_guesses()
        fit = fit_two_peak_linecut(wl, linecut, peak1_guess_nm=p1_guess, peak2_guess_nm=p2_guess)
        p1 = fit.get("peak_1")
        p2 = fit.get("peak_2")
        if p1 is None and p2 is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        baseline = np.full_like(wl, float(fit.get("offset", 0.0)), dtype=float)
        g1 = np.zeros_like(wl, dtype=float)
        g2 = np.zeros_like(wl, dtype=float)
        y1 = None
        y2 = None
        if p1 is not None:
            g1 = lorentzian_component(wl, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
            y1 = baseline + g1
        if p2 is not None:
            g2 = lorentzian_component(wl, p2["amplitude"], p2["center_nm"], p2["gamma_nm"])
            y2 = baseline + g2
        y_total = baseline + g1 + g2
        self.liveView.set_linecut_fit_overlay(wl, y_total, y1, y2)

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
        self._update_maps()


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).resolve()
    frames, defaults = load_frames(data_dir)
    trpl_records = load_trpl_records(data_dir)
    if not frames:
        defaults = ViewerDefaults(crop=DEFAULT_CROP, linecut_row=None, linecut_width=1)
        print(f"No .asc files found in {data_dir}; start the GUI and choose a data folder.")

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    win = PowerGateDepWindow(frames, trpl_records, defaults, data_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

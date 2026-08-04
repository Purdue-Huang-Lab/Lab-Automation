"""
PyQtGraph TAM Viewer — fast single-file replacement for the Matplotlib TAM GUI.

Usage:
    python tam_viewer_pg.py

Dependencies:
    pip install pyqt6 pyqtgraph numpy scipy pandas openpyxl
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
#  SETTINGS — edit these for your experiment (same as top of TAM code new.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TAMSettings:
    filename_sum: str = "ND140-6Ku 1 w 1 ND 1sum.txt"
    filename_para: str = "ND140-6Ku 1 w 1 ND 1para.txt"

    N1: int = 128
    N2: int = 480
    crop1: int = 0
    crop2: int = 0
    crop3: int = 20
    crop4: int = 100

    # wavelength calibration: wavelength = m * pixel + b
    m: float = 0.42
    b: float = 656+45-67-60

    pixelperum: float = 0.08

    TZERO: float = 91.1
    ETIME: float = 300.0
    numberofplots: int = 10

    chirp_coeff: np.ndarray = field(
        default_factory=lambda: np.array([-2.09211940e-04, 3.02271550e-01, -1.03070449e+02])
    )

    smthk: int = 1

    def _resolve(self, name: str) -> str:
        if os.path.isabs(name) or os.path.dirname(name):
            return name
        return os.path.join(self.base_dir, name)

    def __post_init__(self):
        try:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.base_dir = _SCRIPT_DIR
        self.filename_sum = self._resolve(self.filename_sum)
        self.filename_para = self._resolve(self.filename_para)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA I/O — load & reconstruct the (T, Y, X) stack
# ═══════════════════════════════════════════════════════════════════════════════

def load_time_axis(filename_para: str) -> np.ndarray:
    df = pd.read_csv(filename_para, delimiter="\t")
    df.columns = df.columns.astype(float)
    return np.array(df.columns, dtype=np.float64)


def load_stack(filename_sum, n_times, N1, N2, crop1, crop2, crop3, crop4, smthk):
    df = pd.read_csv(filename_sum, delimiter="\t", header=None, na_values=["--", "---"])
    images = df.values

    row_lo, row_hi = crop1, N2 - crop2
    col_lo, col_hi = N1 - crop4, N1 - crop3
    Y, X = col_hi - col_lo, row_hi - row_lo

    stack = np.empty((n_times, Y, X), dtype=np.float32)
    for i in range(n_times):
        leng = i * N2
        raw = np.rot90(images[leng + row_lo : leng + row_hi, col_lo : col_hi])
        stack[i] = uniform_filter(raw.astype(np.float32), size=smthk, mode="nearest")
    return stack


def build_smooth_stack(stack, smthk):
    return uniform_filter(stack, size=(1, smthk, smthk), mode="nearest")


def build_axes(settings, stack_shape):
    _, Y, X = stack_shape
    newwave = settings.m * (np.arange(X) + settings.crop1) + settings.b
    newpos = settings.pixelperum * (np.arange(Y) + (-Y / 2.0))
    return newwave, newpos


def load_all(settings):
    time_ps = load_time_axis(settings.filename_para)
    stack = load_stack(
        settings.filename_sum, len(time_ps),
        settings.N1, settings.N2,
        settings.crop1, settings.crop2, settings.crop3, settings.crop4,
        settings.smthk,
    )
    smooth_stack = build_smooth_stack(stack, settings.smthk)
    newwave, newpos = build_axes(settings, stack.shape)
    return dict(time_ps=time_ps, stack=stack, smooth_stack=smooth_stack,
                newwave=newwave, newpos=newpos)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTS — same output formats as the original script
# ═══════════════════════════════════════════════════════════════════════════════

def _avg_pixel(frame_2d, iy, ix, hw):
    """Average a (2*hw+1)x(2*hw+1) window around (iy, ix) in a 2D frame."""
    if hw <= 0:
        return float(frame_2d[iy, ix])
    Y, X = frame_2d.shape
    y_lo, y_hi = max(0, iy - hw), min(Y, iy + hw + 1)
    x_lo, x_hi = max(0, ix - hw), min(X, ix + hw + 1)
    return float(np.mean(frame_2d[y_lo:y_hi, x_lo:x_hi]))


def _avg_pixel_stack(stack_3d, iy, ix, hw):
    """Average a (2*hw+1)x(2*hw+1) window around (iy, ix) across all time steps."""
    if hw <= 0:
        return stack_3d[:, iy, ix].astype(np.float64, copy=False)
    _, Y, X = stack_3d.shape
    y_lo, y_hi = max(0, iy - hw), min(Y, iy + hw + 1)
    x_lo, x_hi = max(0, ix - hw), min(X, ix + hw + 1)
    return np.mean(stack_3d[:, y_lo:y_hi, x_lo:x_hi], axis=(1, 2))


def export_ta(stack, newwave, time_ps, iy, filename_sum, newpos):
    df = pd.DataFrame()
    df["Wavelength"] = newwave
    for t in range(stack.shape[0]):
        df[f"{time_ps[t]}"] = stack[t, iy, :]
    position = np.round(newpos[iy])
    base = os.path.splitext(os.path.basename(filename_sum))[0]
    path = os.path.join(os.path.dirname(filename_sum), f"{base}TAatpos{position}.xlsx")
    df.to_excel(path, index=False)
    return path


def export_tam(stack, newpos, time_ps, ix, filename_sum, newwave):
    df = pd.DataFrame()
    df["Space"] = newpos
    for t in range(stack.shape[0]):
        df[f"{time_ps[t]}"] = stack[t, :, ix]
    wavelength = np.round(newwave[ix])
    base = os.path.splitext(os.path.basename(filename_sum))[0]
    path = os.path.join(os.path.dirname(filename_sum), f"{base}TAMatwav{wavelength}.xlsx")
    df.to_excel(path, index=False)
    return path


def export_dynamics(stack, time_ps, selected_spots, corx, cory, newwave, newpos, filename_sum,
                    dyn_avg_hw=0):
    spots = selected_spots if selected_spots else [
        (corx, cory, np.round(newwave[corx]), np.round(newpos[cory]))
    ]
    df = pd.DataFrame()
    df["Time"] = time_ps
    for sx, sy, wav, pos in spots:
        df[f"{int(wav)}"] = _avg_pixel_stack(stack, int(sy), int(sx), dyn_avg_hw)
    base = os.path.splitext(os.path.basename(filename_sum))[0]
    out_dir = os.path.dirname(filename_sum)
    if len(spots) == 1:
        fname = f"{base}DYatwav{spots[0][2]}pos{spots[0][3]}.csv"
    else:
        fname = f"{base}DY_multiple_spots.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    return path


def export_gaussian_fit(time_ps, G_STD, G_STD_ERR, ix, newwave, filename_sum):
    df = pd.DataFrame()
    df["Time"] = time_ps
    df["W_squared_um2"] = G_STD ** 2
    df["W_squared_Err_um2"] = 2.0 * G_STD * G_STD_ERR
    wavelength = np.round(newwave[ix])
    base = os.path.splitext(os.path.basename(filename_sum))[0]
    path = os.path.join(os.path.dirname(filename_sum), f"{base}GaussianW_atwav{wavelength}.xlsx")
    df.to_excel(path, index=False)
    return path


def export_all_dynamics(selected_spots, corx, cory, newwave, newpos, filename_sum,
                        N1, N2, crop1, crop2, crop3, crop4, smthk,
                        dyn_avg_hw=0, progress_callback=None):
    spots = selected_spots if selected_spots else [
        (corx, cory, np.round(newwave[corx]), np.round(newpos[cory]))
    ]
    sum_files = sorted(glob.glob(os.path.join(os.path.dirname(filename_sum), "*sum.txt")))
    if not sum_files:
        return None

    all_columns: dict[str, list] = {}
    max_rows = 0

    for f_idx, f_path in enumerate(sum_files):
        try:
            base_name = os.path.basename(f_path)
            para_path = f_path.replace("sum.txt", "para.txt")
            if not os.path.exists(para_path):
                continue

            imgs_df = pd.read_csv(f_path, delimiter="\t", header=None, na_values=["--", "---"])
            images = imgs_df.values

            time_df = pd.read_csv(para_path, delimiter="\t")
            time_df.columns = time_df.columns.astype(float)
            curr_time = np.array(time_df.columns)

            curr_recon = {}
            for i in range(len(curr_time)):
                leng = i * N2
                if leng + N2 > images.shape[0]:
                    break
                raw = np.rot90(images[leng + crop1 : leng + N2 - crop2, N1 - crop4 : N1 - crop3])
                curr_recon[i] = uniform_filter(raw.astype(np.float32), size=smthk, mode="nearest")

            for sx, sy, wav, pos in spots:
                times, intensities = [], []
                for i in range(len(curr_time)):
                    if i in curr_recon:
                        times.append(curr_time[i])
                        intensities.append(_avg_pixel(curr_recon[i], int(sy), int(sx), dyn_avg_hw))
                prefix = base_name.replace("sum.txt", "")
                all_columns[f"{prefix}_Time_W{int(wav)}_P{int(pos)}"] = times
                all_columns[f"{prefix}_Intensity_W{int(wav)}_P{int(pos)}"] = intensities
                max_rows = max(max_rows, len(times))

            if progress_callback:
                progress_callback(f_idx + 1, len(sum_files), base_name)
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue

    if not all_columns:
        return None
    for col in all_columns:
        pad = max_rows - len(all_columns[col])
        if pad > 0:
            all_columns[col].extend([np.nan] * pad)

    first_wav = int(spots[0][2]) if spots else 0
    path = os.path.join(os.path.dirname(filename_sum), f"AllDynamics_Combined_W{first_wav}.csv")
    pd.DataFrame(all_columns).to_csv(path, index=False)
    return path


def export_all_fits(selected_spots, corx, cory, newwave, newpos, filename_sum,
                    N1, N2, crop1, crop2, crop3, crop4, smthk, fit_avg_hw, fit_start_index,
                    use_max_track=False, progress_callback=None):
    spots = selected_spots if selected_spots else [
        (corx, cory, np.round(newwave[corx]), np.round(newpos[cory]))
    ]
    sum_files = sorted(glob.glob(os.path.join(os.path.dirname(filename_sum), "*sum.txt")))
    if not sum_files:
        return None

    all_columns: dict[str, list] = {}
    max_rows = 0

    wav_first = int(spots[0][2]) if spots else 0
    kind = "MaxTrack" if use_max_track else "FixedWL"

    for f_idx, f_path in enumerate(sum_files):
        try:
            base_name = os.path.basename(f_path)
            para_path = f_path.replace("sum.txt", "para.txt")
            if not os.path.exists(para_path):
                continue

            imgs_df = pd.read_csv(f_path, delimiter="\t", header=None, na_values=["--", "---"])
            images = imgs_df.values

            time_df = pd.read_csv(para_path, delimiter="\t")
            time_df.columns = time_df.columns.astype(float)
            curr_time = np.array(time_df.columns)

            curr_recon = {}
            for i in range(len(curr_time)):
                leng = i * N2
                if leng + N2 > images.shape[0]:
                    break
                raw = np.rot90(images[leng + crop1 : leng + N2 - crop2, N1 - crop4 : N1 - crop3])
                curr_recon[i] = uniform_filter(raw.astype(np.float32), size=smthk, mode="nearest")

            for sx, sy, wav, pos in spots:
                times, stds, errs = [], [], []
                for i in range(len(curr_time)):
                    if i < fit_start_index or i not in curr_recon:
                        continue
                    times.append(curr_time[i])
                    
                    yx = curr_recon[i]
                    _, X = yx.shape
                    
                    if use_max_track:
                        wsc = 10
                        cory_local = int(np.clip(sy, 0, yx.shape[0] - 1))
                        ix_local = int(np.clip(sx, 0, yx.shape[1] - 1))
                        lo_b, hi_b = max(0, ix_local - wsc), min(yx.shape[1], ix_local + wsc)
                        midx = lo_b + np.argmax(np.abs(yx[cory_local, lo_b:hi_b]))
                        center_x = midx
                    else:
                        center_x = int(np.clip(sx, 0, X - 1))
                        
                    hw = fit_avg_hw
                    lo = max(0, center_x - hw)
                    hi = min(X, center_x + hw + 1)
                    if lo >= hi:
                        why = yx[:, center_x].astype(np.float64, copy=False)
                    else:
                        why = np.mean(yx[:, lo:hi], axis=1).astype(np.float64, copy=False)

                    amp0 = why[np.argmax(np.abs(why))]
                    mu0 = newpos[np.argmax(np.abs(why))]
                    sig0 = max((newpos[-1] - newpos[0]) / 10.0, 1e-3)
                    off0 = float(np.median(why))
                    
                    try:
                        p, cov = curve_fit(gaussian_with_offset, newpos, why,
                                           p0=[amp0, mu0, sig0, off0], maxfev=20000)
                        e = np.sqrt(np.diag(cov))
                        w = abs(p[2])
                        we = abs(e[2])
                        stds.append(w ** 2)
                        errs.append(2.0 * w * we)
                    except (RuntimeError, ValueError):
                        stds.append(np.nan)
                        errs.append(np.nan)

                prefix = base_name.replace("sum.txt", "")
                all_columns[f"{prefix}_Time_W{int(wav)}_P{int(pos)}"] = times
                all_columns[f"{prefix}_{kind}_W_sq_um2_W{int(wav)}_P{int(pos)}"] = stds
                all_columns[f"{prefix}_{kind}_W_sq_Err_W{int(wav)}_P{int(pos)}"] = errs
                max_rows = max(max_rows, len(times))

            if progress_callback:
                progress_callback(f_idx + 1, len(sum_files), base_name)
        except Exception as e:
            print(f"Error processing {f_path} for fits: {e}")
            continue

    if not all_columns:
        return None
    for col in all_columns:
        pad = max_rows - len(all_columns[col])
        if pad > 0:
            all_columns[col].extend([np.nan] * pad)

    path = os.path.join(os.path.dirname(filename_sum), f"AllFits_{kind}_Combined_W{wav_first}.csv")
    pd.DataFrame(all_columns).to_csv(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FITTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_with_offset(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / (2 * stddev)) ** 2) + offset


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int, str)


class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(str(result) if result else "Done")
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class TAMViewer(QtWidgets.QMainWindow):
    def __init__(self, settings: TAMSettings | None = None):
        super().__init__()
        self.setWindowTitle("TAM Viewer (pyqtgraph)")
        self.resize(1400, 900)
        self.settings = settings or TAMSettings()
        self.threadpool = QtCore.QThreadPool()

        self.corx: int = 10
        self.cory: int = 10
        self.selected_spots: list[tuple] = []
        self.show_spots: bool = False

        self.G_STD = self.G_STD_ERR = None
        self.stack = None
        self.time_ps = self.smooth_stack = self.newwave = self.newpos = None

        self._build_ui()
        self._connect_signals()
        self._load_data()
        if self.stack is not None:
            self.time_slider.setValue(0)
            self._on_time_changed(0)
        else:
            self.control_tabs.setCurrentIndex(0)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left column: image + lineout plots
        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.image_gw = pg.GraphicsLayoutWidget()
        self.image_plot = self.image_gw.addPlot(title="Camera Image")
        self.image_item = pg.ImageItem()
        self.image_item.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.image_plot.addItem(self.image_item)
        self.image_plot.setLabel("bottom", "pixel (wavelength)")
        self.image_plot.setLabel("left", "pixel (space)")
        dash = QtCore.Qt.PenStyle.DashLine
        self.crosshair_h = pg.InfiniteLine(angle=0, pen=pg.mkPen("k", width=1, style=dash))
        self.crosshair_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1, style=dash))
        self.image_plot.addItem(self.crosshair_h)
        self.image_plot.addItem(self.crosshair_v)
        self.scatter_spots = pg.ScatterPlotItem(
            pen=pg.mkPen("r", width=2), brush=pg.mkBrush(None), size=12, symbol="o"
        )
        self.image_plot.addItem(self.scatter_spots)
        self.scatter_spots.setVisible(False)
        left.addWidget(self.image_gw)

        self.ta_pw = pg.PlotWidget(title="TA (wavelength lineout)")
        self.ta_pw.setLabel("bottom", "Wavelength (nm)")
        self.ta_pw.setLabel("left", "Intensity")
        self.ta_curve = self.ta_pw.plot(pen=pg.mkPen((44, 160, 44), width=1.5))
        self.ta_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.ta_pw.addItem(self.ta_vline)
        left.addWidget(self.ta_pw)

        self.tam_pw = pg.PlotWidget(title="TAM (space lineout)")
        self.tam_pw.setLabel("bottom", "Space (um)")
        self.tam_pw.setLabel("left", "Intensity")
        self.tam_curve = self.tam_pw.plot(pen=None, symbol="o", symbolSize=5,
                                          symbolPen=(200, 50, 50), symbolBrush=(200, 50, 50))
        self.tam_fit_curve = self.tam_pw.plot(pen=pg.mkPen((0, 0, 0), width=2))
        self.tam_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.tam_pw.addItem(self.tam_vline)
        left.addWidget(self.tam_pw)

        root.addWidget(left, stretch=3)

        # Right column: dynamics, fit, controls — use a splitter so plots are large
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # ── Top: two plots stacked vertically ──
        plots_widget = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(4)

        self.dyn_pw = pg.PlotWidget(title="Dynamics")
        self.dyn_pw.setLabel("bottom", "Time (ps)")
        self.dyn_pw.setLabel("left", "Intensity")
        self.dyn_curve = self.dyn_pw.plot(pen=pg.mkPen((31, 119, 180), width=2))
        plots_layout.addWidget(self.dyn_pw, stretch=1)

        self.fit_pw = pg.PlotWidget(title="Gaussian W\u00b2 vs Time")
        self.fit_pw.setLabel("bottom", "Time (ps)")
        self.fit_pw.setLabel("left", "W\u00b2 (um\u00b2)")
        self.fit_curve = self.fit_pw.plot(pen=pg.mkPen((148, 103, 189), width=1.5),
                                         symbol="o", symbolSize=5,
                                         symbolPen=(148, 103, 189), symbolBrush=(148, 103, 189))
        self.fit_err_item = pg.ErrorBarItem(pen=pg.mkPen((148, 103, 189), width=1))
        self.fit_pw.addItem(self.fit_err_item)
        plots_layout.addWidget(self.fit_pw, stretch=1)

        right_splitter.addWidget(plots_widget)

        # ── Bottom: controls in a scroll area so they stay compact ──
        controls_container = QtWidgets.QWidget()
        cl_outer = QtWidgets.QVBoxLayout(controls_container)
        cl_outer.setContentsMargins(0, 0, 0, 0)
        cl_outer.setSpacing(4)

        # Controls group
        cg = QtWidgets.QGroupBox("Controls")
        cl = QtWidgets.QGridLayout(cg)

        cl.addWidget(QtWidgets.QLabel("Time:"), 0, 0)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        cl.addWidget(self.time_slider, 0, 1, 1, 3)
        self.time_label = QtWidgets.QLabel("0 ps (frame 0)")
        cl.addWidget(self.time_label, 0, 4)

        self.pos_label = QtWidgets.QLabel("Cursor: --")
        cl.addWidget(self.pos_label, 0, 5, 1, 2)

        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Min:"))
        self.min_edit = QtWidgets.QLineEdit("-3")
        self.min_edit.setFixedWidth(60)
        range_row.addWidget(self.min_edit)
        range_row.addSpacing(10)
        range_row.addWidget(QtWidgets.QLabel("Max:"))
        self.max_edit = QtWidgets.QLineEdit("5")
        self.max_edit.setFixedWidth(60)
        range_row.addWidget(self.max_edit)
        range_row.addSpacing(10)
        self.apply_levels_btn = QtWidgets.QPushButton("Apply")
        range_row.addWidget(self.apply_levels_btn)
        range_row.addStretch()
        cl.addLayout(range_row, 1, 0, 1, 7)

        self.control_tabs = QtWidgets.QTabWidget()

        tab_paths = QtWidgets.QWidget()
        tab_paths_l = QtWidgets.QVBoxLayout(tab_paths)
        tab_paths_l.setContentsMargins(4, 8, 4, 4)
        sum_row = QtWidgets.QHBoxLayout()
        sum_row.addWidget(QtWidgets.QLabel("sum.txt:"))
        self.path_sum_edit = QtWidgets.QLineEdit()
        self.path_sum_edit.setMinimumWidth(280)
        self.path_sum_edit.setPlaceholderText("Full path to *sum.txt")
        sum_row.addWidget(self.path_sum_edit, stretch=1)
        self.btn_browse_sum = QtWidgets.QPushButton("Browse\u2026")
        sum_row.addWidget(self.btn_browse_sum)
        tab_paths_l.addLayout(sum_row)

        para_row = QtWidgets.QHBoxLayout()
        para_row.addWidget(QtWidgets.QLabel("para.txt:"))
        self.path_para_edit = QtWidgets.QLineEdit()
        self.path_para_edit.setMinimumWidth(280)
        self.path_para_edit.setPlaceholderText("Full path to *para.txt")
        para_row.addWidget(self.path_para_edit, stretch=1)
        self.btn_browse_para = QtWidgets.QPushButton("Browse\u2026")
        para_row.addWidget(self.btn_browse_para)
        tab_paths_l.addLayout(para_row)

        hint = QtWidgets.QLabel(
            "Launch the app first, then choose files here (Browse or paste full paths) and click "
            "\u201cLoad data\u201d. Default paths point next to this script when they exist."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444;")
        tab_paths_l.addWidget(hint)

        load_row = QtWidgets.QHBoxLayout()
        self.btn_load_paths = QtWidgets.QPushButton("Load data")
        self.btn_load_paths.setStyleSheet("font-weight: bold;")
        load_row.addWidget(self.btn_load_paths)
        load_row.addStretch()
        tab_paths_l.addLayout(load_row)

        tab_fit_snr = QtWidgets.QWidget()
        tab_fit_snr_l = QtWidgets.QVBoxLayout(tab_fit_snr)
        tab_fit_snr_l.setContentsMargins(4, 8, 4, 4)
        fit_row = QtWidgets.QHBoxLayout()
        fit_row.addWidget(QtWidgets.QLabel("Fit start frame:"))
        self.fit_start_edit = QtWidgets.QLineEdit("0")
        self.fit_start_edit.setFixedWidth(48)
        fit_row.addWidget(self.fit_start_edit)
        self.fit_btn = QtWidgets.QPushButton("Fit Here")
        fit_row.addWidget(self.fit_btn)
        self.fitall_btn = QtWidgets.QPushButton("Fit All")
        fit_row.addWidget(self.fitall_btn)
        self.fitallmax_btn = QtWidgets.QPushButton("Fit Max (\u03bb-track)")
        fit_row.addWidget(self.fitallmax_btn)
        fit_row.addStretch()
        tab_fit_snr_l.addLayout(fit_row)

        avg_row = QtWidgets.QHBoxLayout()
        avg_row.addWidget(QtWidgets.QLabel("TAM lineout \u03bb avg \u00b1"))
        self.fit_avg_spin = QtWidgets.QSpinBox()
        self.fit_avg_spin.setRange(0, 50)
        self.fit_avg_spin.setValue(3)
        self.fit_avg_spin.setToolTip(
            "Along wavelength (camera X), average pixels ix\u00b1N when building the spatial lineout "
            "for Gaussian fits. Larger N improves SNR but can blur across sharp features."
        )
        avg_row.addWidget(self.fit_avg_spin)
        avg_row.addWidget(QtWidgets.QLabel("px"))
        avg_lbl = QtWidgets.QLabel("(0 = single pixel)")
        avg_lbl.setStyleSheet("color: #444;")
        avg_row.addWidget(avg_lbl)
        avg_row.addStretch()
        tab_fit_snr_l.addLayout(avg_row)

        dyn_avg_row = QtWidgets.QHBoxLayout()
        dyn_avg_row.addWidget(QtWidgets.QLabel("DY pixel avg \u00b1"))
        self.dyn_avg_spin = QtWidgets.QSpinBox()
        self.dyn_avg_spin.setRange(0, 50)
        self.dyn_avg_spin.setValue(0)
        self.dyn_avg_spin.setToolTip(
            "Average a (2N+1)\u00d7(2N+1) pixel window around the selected point "
            "for dynamics display, Export DY, and Export All DY. "
            "0 = single pixel (no averaging)."
        )
        dyn_avg_row.addWidget(self.dyn_avg_spin)
        dyn_avg_row.addWidget(QtWidgets.QLabel("px"))
        dyn_avg_lbl = QtWidgets.QLabel("(0 = single pixel)")
        dyn_avg_lbl.setStyleSheet("color: #444;")
        dyn_avg_row.addWidget(dyn_avg_lbl)
        dyn_avg_row.addStretch()
        tab_fit_snr_l.addLayout(dyn_avg_row)

        gpg = QtWidgets.QGroupBox("Last Gaussian (Fit Here)")
        gpl = QtWidgets.QGridLayout(gpg)
        self.gauss_amp_edit = QtWidgets.QLineEdit("")
        self.gauss_mean_edit = QtWidgets.QLineEdit("")
        self.gauss_sigma_edit = QtWidgets.QLineEdit("")
        self.gauss_off_edit = QtWidgets.QLineEdit("")
        for i, (lbl, w) in enumerate([
            ("Amplitude", self.gauss_amp_edit),
            ("\u03bc (\u00b5m)", self.gauss_mean_edit),
            ("\u03c3 (\u00b5m)", self.gauss_sigma_edit),
            ("offset", self.gauss_off_edit),
        ]):
            gpl.addWidget(QtWidgets.QLabel(lbl), i // 2, (i % 2) * 2)
            gpl.addWidget(w, i // 2, (i % 2) * 2 + 1)
        tab_fit_snr_l.addWidget(gpg)

        self.control_tabs.addTab(tab_paths, "Data files")
        self.control_tabs.addTab(tab_fit_snr, "Fit & SNR")
        cl.addWidget(self.control_tabs, 2, 0, 1, 7)

        self._sync_path_edits_from_settings()

        cl.addWidget(QtWidgets.QLabel("X:"), 3, 0)
        self.manual_x = QtWidgets.QLineEdit()
        cl.addWidget(self.manual_x, 3, 1)
        cl.addWidget(QtWidgets.QLabel("Y:"), 3, 2)
        self.manual_y = QtWidgets.QLineEdit()
        cl.addWidget(self.manual_y, 3, 3)
        self.add_spot_btn = QtWidgets.QPushButton("Add Spot")
        cl.addWidget(self.add_spot_btn, 3, 4)

        cl_outer.addWidget(cg)

        row1 = QtWidgets.QHBoxLayout()
        self.btn_export_ta = QtWidgets.QPushButton("Export TA")
        self.btn_export_tam = QtWidgets.QPushButton("Export TAM")
        self.btn_export_dy = QtWidgets.QPushButton("Export DY")
        self.btn_export_fit = QtWidgets.QPushButton("Export Fit")
        self.btn_export_all = QtWidgets.QPushButton("Export All DY")
        for b in (self.btn_export_ta, self.btn_export_tam, self.btn_export_dy,
                  self.btn_export_fit, self.btn_export_all):
            row1.addWidget(b)
        cl_outer.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Clear Selection")
        self.btn_clear.setStyleSheet("background-color: #e88; color: black;")
        self.btn_show = QtWidgets.QPushButton("Show Spots")
        self.btn_export_all_fits = QtWidgets.QPushButton("Export All Fits")
        self.btn_export_all_fits_max = QtWidgets.QPushButton("Export All Fits (Max)")
        row2.addWidget(self.btn_clear)
        row2.addWidget(self.btn_show)
        row2.addWidget(self.btn_export_all_fits)
        row2.addWidget(self.btn_export_all_fits_max)
        cl_outer.addLayout(row2)

        # Crop / image parameters
        crop_group = QtWidgets.QGroupBox("Crop && Image Params")
        crop_layout = QtWidgets.QGridLayout(crop_group)
        s = self.settings
        self.crop1_edit = QtWidgets.QLineEdit(str(s.crop1))
        self.crop2_edit = QtWidgets.QLineEdit(str(s.crop2))
        self.crop3_edit = QtWidgets.QLineEdit(str(s.crop3))
        self.crop4_edit = QtWidgets.QLineEdit(str(s.crop4))
        self.n1_edit = QtWidgets.QLineEdit(str(s.N1))
        self.n2_edit = QtWidgets.QLineEdit(str(s.N2))
        self.smthk_edit = QtWidgets.QLineEdit(str(s.smthk))
        for col, (lbl, w) in enumerate([
            ("crop1", self.crop1_edit), ("crop2", self.crop2_edit),
            ("crop3", self.crop3_edit), ("crop4", self.crop4_edit),
        ]):
            crop_layout.addWidget(QtWidgets.QLabel(lbl), 0, col * 2)
            crop_layout.addWidget(w, 0, col * 2 + 1)
        crop_layout.addWidget(QtWidgets.QLabel("N1"), 1, 0)
        crop_layout.addWidget(self.n1_edit, 1, 1)
        crop_layout.addWidget(QtWidgets.QLabel("N2"), 1, 2)
        crop_layout.addWidget(self.n2_edit, 1, 3)
        crop_layout.addWidget(QtWidgets.QLabel("smooth k"), 1, 4)
        crop_layout.addWidget(self.smthk_edit, 1, 5)
        self.reload_btn = QtWidgets.QPushButton("Reload Data")
        self.reload_btn.setStyleSheet("background-color: #8cf; color: black; font-weight: bold;")
        crop_layout.addWidget(self.reload_btn, 1, 6, 1, 2)
        cl_outer.addWidget(crop_group)

        self.status_label = QtWidgets.QLabel("Ready")
        cl_outer.addWidget(self.status_label)

        # Wrap controls in a scroll area so they compress gracefully
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right_splitter.addWidget(scroll)

        # Give plots ~60% of the vertical space, controls ~40%
        right_splitter.setSizes([500, 300])
        right_splitter.setStretchFactor(0, 3)  # plots stretch more
        right_splitter.setStretchFactor(1, 1)  # controls stay compact

        root.addWidget(right_splitter, stretch=2)

        self._set_data_widgets_enabled(False)

    # ── Data loading ─────────────────────────────────────────────────────

    def _load_data(self):
        self.status_label.setText("Loading data\u2026")
        QtWidgets.QApplication.processEvents()

        s = self.settings
        if not (s.filename_sum and s.filename_para):
            self._clear_data_state("Set sum and para paths, then Load data.")
            return
        if not os.path.isfile(s.filename_sum) or not os.path.isfile(s.filename_para):
            self._clear_data_state("Files not found — select sum and para, then Load data.")
            return

        try:
            data = load_all(s)
        except Exception as exc:
            self._clear_data_state(f"Load failed: {exc}")
            return

        self.time_ps = data["time_ps"]
        self.stack = data["stack"]
        self.smooth_stack = data["smooth_stack"]
        self.newwave = data["newwave"]
        self.newpos = data["newpos"]

        n = len(self.time_ps)
        self.time_slider.setMaximum(max(0, n - 1))

        self.G_STD = np.zeros(n)
        self.G_STD_ERR = np.zeros(n)
        self.G_AMP = np.zeros(n)
        self.G_MEAN = np.zeros(n)
        self.G_OFFSET = np.zeros(n)
        self.G_AMP_ERR = np.zeros(n)
        self.G_MEAN_ERR = np.zeros(n)
        self.G_OFFSET_ERR = np.zeros(n)

        _, Y, X = self.stack.shape
        self.corx = min(X - 1, max(0, X // 2))
        self.cory = min(Y - 1, max(0, Y // 2))
        self._set_data_widgets_enabled(n > 0)
        self.status_label.setText(f"Loaded {n} frames  ({Y}\u00d7{X})")

    def _clear_data_state(self, message: str):
        self.stack = None
        self.time_ps = self.smooth_stack = self.newwave = self.newpos = None
        self.G_STD = self.G_STD_ERR = None
        self.time_slider.setMaximum(0)
        self.time_slider.setValue(0)
        self._set_data_widgets_enabled(False)
        self._clear_plots_empty()
        self.status_label.setText(message)

    def _clear_plots_empty(self):
        z = np.zeros((64, 64), dtype=np.float32)
        self.image_item.setImage(z, autoLevels=False)
        self.image_item.setLevels([-1.0, 1.0])
        self.ta_curve.setData([], [])
        self.ta_vline.setValue(0.0)
        self.tam_curve.setData([], [])
        self.tam_fit_curve.setData([], [])
        self.tam_vline.setValue(0.0)
        self.dyn_curve.setData([], [])
        self.fit_curve.setData([], [])
        self.fit_err_item.setData(x=np.array([]), y=np.array([]), height=np.array([]))
        self.time_label.setText("No data")
        self.pos_label.setText("Cursor: --")

    def _set_data_widgets_enabled(self, on: bool):
        self.time_slider.setEnabled(on)
        self.fit_btn.setEnabled(on)
        self.fitall_btn.setEnabled(on)
        self.fitallmax_btn.setEnabled(on)
        self.add_spot_btn.setEnabled(on)
        self.btn_clear.setEnabled(on)
        self.btn_show.setEnabled(on)
        for b in (
            self.btn_export_ta,
            self.btn_export_tam,
            self.btn_export_dy,
            self.btn_export_fit,
            self.btn_export_all,
            self.btn_export_all_fits,
            self.btn_export_all_fits_max,
        ):
            b.setEnabled(on)

    def _reload_data(self):
        """Re-read crop/image params from the GUI fields, rebuild stack."""
        s = self.settings
        try:
            s.crop1 = int(self.crop1_edit.text())
            s.crop2 = int(self.crop2_edit.text())
            s.crop3 = int(self.crop3_edit.text())
            s.crop4 = int(self.crop4_edit.text())
            s.N1 = int(self.n1_edit.text())
            s.N2 = int(self.n2_edit.text())
            s.smthk = int(self.smthk_edit.text())
        except ValueError:
            self.status_label.setText("Invalid crop / image parameter")
            return
        self.selected_spots.clear()
        self._refresh_spots()
        self._load_data()
        self.time_slider.setValue(0)
        self._on_time_changed(0)

    # ── Signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self):
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.apply_levels_btn.clicked.connect(self._apply_levels)
        self.min_edit.returnPressed.connect(self._apply_levels)
        self.max_edit.returnPressed.connect(self._apply_levels)
        self.image_plot.scene().sigMouseClicked.connect(self._on_image_click)
        self.fit_btn.clicked.connect(self._fit_here)
        self.fitall_btn.clicked.connect(self._fit_all)
        self.fitallmax_btn.clicked.connect(self._fit_all_max)
        self.dyn_avg_spin.valueChanged.connect(lambda: self._update_lineouts())
        self.btn_browse_sum.clicked.connect(self._browse_sum)
        self.btn_browse_para.clicked.connect(self._browse_para)
        self.btn_load_paths.clicked.connect(self._apply_paths_and_load)
        self.add_spot_btn.clicked.connect(self._add_manual_spot)
        self.btn_clear.clicked.connect(self._clear_spots)
        self.btn_show.clicked.connect(self._toggle_spots)
        self.btn_export_ta.clicked.connect(self._export_ta)
        self.btn_export_tam.clicked.connect(self._export_tam)
        self.btn_export_dy.clicked.connect(self._export_dy)
        self.btn_export_fit.clicked.connect(self._export_fit)
        self.btn_export_all.clicked.connect(self._export_all_dy)
        self.btn_export_all_fits.clicked.connect(self._export_all_fits)
        self.btn_export_all_fits_max.clicked.connect(self._export_all_fits_max)
        self.reload_btn.clicked.connect(self._reload_data)

    def _sync_path_edits_from_settings(self):
        s = self.settings
        self.path_sum_edit.setText(os.path.normpath(s.filename_sum))
        self.path_para_edit.setText(os.path.normpath(s.filename_para))

    def _apply_paths_and_load(self):
        sum_p = self.path_sum_edit.text().strip()
        para_p = self.path_para_edit.text().strip()
        if not sum_p or not para_p:
            self.status_label.setText("Set both sum and para paths")
            return
        if not os.path.isfile(sum_p):
            self.status_label.setText("sum file not found")
            return
        if not os.path.isfile(para_p):
            self.status_label.setText("para file not found")
            return
        self.settings.filename_sum = os.path.abspath(sum_p)
        self.settings.filename_para = os.path.abspath(para_p)
        self.selected_spots.clear()
        self._refresh_spots()
        self._load_data()
        self.time_slider.setValue(0)
        self._on_time_changed(0)

    def _browse_sum(self):
        start = os.path.dirname(self.path_sum_edit.text().strip()) or _SCRIPT_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select sum.txt", start, "Text (*.txt);;All (*.*)"
        )
        if path:
            self.path_sum_edit.setText(os.path.normpath(path))

    def _browse_para(self):
        start = os.path.dirname(self.path_para_edit.text().strip()) or _SCRIPT_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select para.txt", start, "Text (*.txt);;All (*.*)"
        )
        if path:
            self.path_para_edit.setText(os.path.normpath(path))

    # ── Fast update callbacks ────────────────────────────────────────────

    def _on_time_changed(self, t_idx):
        if self.stack is None:
            return
        vmin, vmax = self._levels()
        self.image_item.setImage(self.stack[t_idx], autoLevels=False)
        self.image_item.setLevels([vmin, vmax])
        self.time_label.setText(f"{self.time_ps[t_idx]:.2f} ps  (frame {t_idx})")
        self._update_lineouts(t_idx)

    def _apply_levels(self):
        if self.stack is None:
            return
        vmin, vmax = self._levels()
        self.image_item.setLevels([vmin, vmax])

    def _levels(self):
        try:
            vmin = float(self.min_edit.text())
        except ValueError:
            vmin = -3.0
        try:
            vmax = float(self.max_edit.text())
        except ValueError:
            vmax = 5.0
        return vmin, vmax

    def _update_lineouts(self, t_idx=None):
        if self.stack is None or self.time_ps is None:
            return
        if t_idx is None:
            t_idx = self.time_slider.value()
        ix = int(np.clip(self.corx, 0, self.stack.shape[2] - 1))
        iy = int(np.clip(self.cory, 0, self.stack.shape[1] - 1))

        self.ta_curve.setData(self.newwave, self.stack[t_idx, iy, :])
        self.ta_vline.setValue(self.newwave[ix])

        tam_data = self._tam_lineout_for_fit(self.stack[t_idx], ix)
        self.tam_curve.setData(self.newpos, tam_data)
        self.tam_vline.setValue(self.newpos[iy])
        ymax = float(np.max(np.abs(tam_data)))
        if ymax < 1e-9:
            ymax = 1.0
        self.tam_pw.setYRange(-ymax, ymax)
        self._update_tam_fit(t_idx)

        dyn_hw = int(self.dyn_avg_spin.value())
        self.dyn_curve.setData(self.time_ps, _avg_pixel_stack(self.smooth_stack, iy, ix, dyn_hw))

        self.crosshair_h.setValue(iy)
        self.crosshair_v.setValue(ix)

        wav = self.newwave[ix]
        pos = self.newpos[iy]
        val = self.stack[t_idx, iy, ix]
        self.pos_label.setText(
            f"px({ix},{iy})  \u03bb={wav:.1f} nm  y={pos:.2f} \u00b5m  I={val:.3g}"
        )

    def _update_tam_fit(self, t_idx):
        """Draw Gaussian fit curve on the TAM space plot if fit params exist."""
        if self.newpos is None or self.G_STD is None:
            self.tam_fit_curve.setData([], [])
            return
        if self.G_STD[t_idx] > 0 and np.isfinite(self.G_STD[t_idx]):
            x_fine = np.linspace(self.newpos[0], self.newpos[-1], 300)
            y_fit = gaussian_with_offset(
                x_fine, self.G_AMP[t_idx], self.G_MEAN[t_idx],
                self.G_STD[t_idx], self.G_OFFSET[t_idx],
            )
            self.tam_fit_curve.setData(x_fine, y_fit)
        else:
            self.tam_fit_curve.setData([], [])

    # ── Mouse interaction ────────────────────────────────────────────────

    def _on_image_click(self, event):
        if self.stack is None:
            return
        pos = event.scenePos()
        if not self.image_plot.sceneBoundingRect().contains(pos):
            return
        mp = self.image_plot.vb.mapSceneToView(pos)
        ix = int(np.clip(np.round(mp.x()), 0, self.stack.shape[2] - 1))
        iy = int(np.clip(np.round(mp.y()), 0, self.stack.shape[1] - 1))

        mods = QtWidgets.QApplication.keyboardModifiers()
        is_shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        is_middle = event.button() == QtCore.Qt.MouseButton.MiddleButton

        if is_shift or is_middle:
            self._toggle_spot(ix, iy)
        else:
            self.corx = ix
            self.cory = iy
            self._update_lineouts()

    def _toggle_spot(self, ix, iy):
        if self.newwave is None or self.newpos is None:
            return
        wav = np.round(self.newwave[ix])
        pos = np.round(self.newpos[iy])
        info = (ix, iy, wav, pos)
        if info in self.selected_spots:
            self.selected_spots.remove(info)
            self.status_label.setText(f"Removed spot {wav:.0f} nm / {pos:.0f} um  ({len(self.selected_spots)} total)")
        else:
            self.selected_spots.append(info)
            self.status_label.setText(f"Added spot {wav:.0f} nm / {pos:.0f} um  ({len(self.selected_spots)} total)")
        self._refresh_spots()

    def _refresh_spots(self):
        if self.show_spots and self.selected_spots:
            self.scatter_spots.setData([s[0] for s in self.selected_spots],
                                       [s[1] for s in self.selected_spots])
            self.scatter_spots.setVisible(True)
        else:
            self.scatter_spots.setVisible(False)

    def _clear_spots(self):
        self.selected_spots.clear()
        self._refresh_spots()
        self.status_label.setText("Selection cleared")

    def _toggle_spots(self):
        self.show_spots = not self.show_spots
        self.btn_show.setText("Hide Spots" if self.show_spots else "Show Spots")
        self._refresh_spots()

    def _add_manual_spot(self):
        if self.stack is None:
            return
        try:
            mx, my = int(float(self.manual_x.text())), int(float(self.manual_y.text()))
        except ValueError:
            self.status_label.setText("Invalid X / Y")
            return
        _, Y, X = self.stack.shape
        if not (0 <= mx < X and 0 <= my < Y):
            self.status_label.setText("X / Y out of bounds")
            return
        self._toggle_spot(mx, my)

    # ── Keyboard ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if self.stack is None:
            super().keyPressEvent(event)
            return
        if event.key() == QtCore.Qt.Key.Key_Right:
            self.time_slider.setValue(min(self.time_slider.value() + 1, self.time_slider.maximum()))
        elif event.key() == QtCore.Qt.Key.Key_Left:
            self.time_slider.setValue(max(self.time_slider.value() - 1, 0))
        else:
            super().keyPressEvent(event)

    # ── Gaussian fitting ─────────────────────────────────────────────────

    def _tam_lineout_for_fit(self, yx: np.ndarray, ix: int) -> np.ndarray:
        """Spatial (TAM) lineout: mean over wavelength pixels ix\u00b1N at each Y."""
        _, X = yx.shape
        ix = int(np.clip(ix, 0, X - 1))
        hw = int(self.fit_avg_spin.value())
        lo = max(0, ix - hw)
        hi = min(X, ix + hw + 1)
        if lo >= hi:
            return yx[:, ix].astype(np.float64, copy=False)
        return np.mean(yx[:, lo:hi], axis=1)

    def _fit_here(self):
        if self.stack is None or self.newpos is None:
            return
        t = self.time_slider.value()
        ix = int(np.clip(self.corx, 0, self.stack.shape[2] - 1))
        why = self._tam_lineout_for_fit(self.stack[t], ix)
        amp0 = why[np.argmax(np.abs(why))]
        mu0 = self.newpos[np.argmax(np.abs(why))]
        sig0 = max((self.newpos[-1] - self.newpos[0]) / 10.0, 1e-3)
        off0 = float(np.median(why))
        try:
            p, cov = curve_fit(gaussian_with_offset, self.newpos, why,
                               p0=[amp0, mu0, sig0, off0], maxfev=20000)
            e = np.sqrt(np.diag(cov))
            self.G_AMP[t], self.G_MEAN[t] = p[0], p[1]
            self.G_STD[t], self.G_OFFSET[t] = abs(p[2]), p[3]
            self.G_AMP_ERR[t], self.G_MEAN_ERR[t] = e[0], e[1]
            self.G_STD_ERR[t], self.G_OFFSET_ERR[t] = abs(e[2]), e[3]
            self.gauss_amp_edit.setText(f"{p[0]:.3g}")
            self.gauss_mean_edit.setText(f"{p[1]:.3g}")
            self.gauss_sigma_edit.setText(f"{abs(p[2]):.3g}")
            self.gauss_off_edit.setText(f"{p[3]:.3g}")
            self._update_tam_fit(t)
            navg = 2 * int(self.fit_avg_spin.value()) + 1
            self.status_label.setText(
                f"Fit @ frame {t}: \u03c3={abs(p[2]):.4g} um  (\u03bb avg {navg} px)"
            )
        except RuntimeError:
            self.status_label.setText("Gaussian fit did not converge")

    def _fit_all(self):
        if self.stack is None or self.time_ps is None or self.newpos is None:
            return
        si = self._fit_start_index()
        for i in range(si, len(self.time_ps)):
            ix = int(np.clip(self.corx, 0, self.stack.shape[2] - 1))
            why = self._tam_lineout_for_fit(self.stack[i], ix)
            amp0 = why[np.argmax(np.abs(why))]
            mu0 = self.newpos[np.argmax(np.abs(why))]
            sig0 = max((self.newpos[-1] - self.newpos[0]) / 10.0, 1e-3)
            off0 = float(np.median(why))
            try:
                p, cov = curve_fit(gaussian_with_offset, self.newpos, why,
                                   p0=[amp0, mu0, sig0, off0], maxfev=20000)
                e = np.sqrt(np.diag(cov))
                self.G_AMP[i], self.G_MEAN[i] = p[0], p[1]
                self.G_STD[i], self.G_OFFSET[i] = abs(p[2]), p[3]
                self.G_AMP_ERR[i], self.G_MEAN_ERR[i] = e[0], e[1]
                self.G_STD_ERR[i], self.G_OFFSET_ERR[i] = abs(e[2]), e[3]
            except (RuntimeError, ValueError):
                self.G_STD[i] = self.G_STD_ERR[i] = np.nan
        self._plot_fit_results(si)
        navg = 2 * int(self.fit_avg_spin.value()) + 1
        self.status_label.setText(f"Fit All complete  (\u03bb avg {navg} px)")

    def _fit_all_max(self):
        """Gaussian fit along space at each time, after tracking peak wavelength (|signal| at cursor Y)."""
        if self.stack is None or self.time_ps is None or self.newpos is None:
            return
        si = self._fit_start_index()
        wsc = 10
        cory = int(np.clip(self.cory, 0, self.stack.shape[1] - 1))
        for i in range(si, len(self.time_ps)):
            ix = int(np.clip(self.corx, 0, self.stack.shape[2] - 1))
            lo, hi = max(0, ix - wsc), min(self.stack.shape[2], ix + wsc)
            midx = lo + np.argmax(np.abs(self.stack[i, cory, lo:hi]))
            why = self._tam_lineout_for_fit(self.stack[i], midx)
            amp0 = why[np.argmax(np.abs(why))]
            mu0 = self.newpos[np.argmax(np.abs(why))]
            sig0 = max((self.newpos[-1] - self.newpos[0]) / 10.0, 1e-3)
            off0 = float(np.median(why))
            try:
                p, cov = curve_fit(gaussian_with_offset, self.newpos, why,
                                   p0=[amp0, mu0, sig0, off0], maxfev=20000)
                e = np.sqrt(np.diag(cov))
                self.G_AMP[i], self.G_MEAN[i] = p[0], p[1]
                self.G_STD[i], self.G_OFFSET[i] = abs(p[2]), p[3]
                self.G_AMP_ERR[i], self.G_MEAN_ERR[i] = e[0], e[1]
                self.G_STD_ERR[i], self.G_OFFSET_ERR[i] = abs(e[2]), e[3]
            except (RuntimeError, ValueError):
                self.G_STD[i] = self.G_STD_ERR[i] = np.nan
        self._plot_fit_results(si)
        navg = 2 * int(self.fit_avg_spin.value()) + 1
        self.status_label.setText(f"Fit Max (\u03bb-track) complete  (\u03bb avg {navg} px)")

    def _plot_fit_results(self, si):
        if self.G_STD is None or self.time_ps is None:
            self.fit_curve.setData([], [])
            self.fit_err_item.setData(x=np.array([]), y=np.array([]), height=np.array([]))
            return
        gw, gwe = self.G_STD[si:], self.G_STD_ERR[si:]
        m = np.isfinite(gw) & (gw > 0)
        t = self.time_ps[si:][m]
        w2, w2e = gw[m] ** 2, 2.0 * gw[m] * gwe[m]
        self.fit_curve.setData(t, w2)
        self.fit_err_item.setData(x=t, y=w2, height=w2e)

    def _fit_start_index(self):
        if self.time_ps is None or len(self.time_ps) == 0:
            return 0
        try:
            v = int(float(self.fit_start_edit.text()))
        except ValueError:
            v = 0
        return max(0, min(v, len(self.time_ps) - 1))

    # ── Exports (threaded) ───────────────────────────────────────────────

    def _run_in_thread(self, fn, *args, **kwargs):
        if kwargs.get('progress_callback') == 'inject':
            kwargs.pop('progress_callback')
            w = Worker(fn, *args, **kwargs)
            w.kwargs['progress_callback'] = w.signals.progress.emit
            w.signals.progress.connect(lambda curr, tot, name: self.status_label.setText(f"Processing ({curr}/{tot}): {name}"))
        else:
            w = Worker(fn, *args, **kwargs)
            
        def on_finished(msg):
            if "." in msg and ("\\" in msg or "/" in msg):
                self.status_label.setText(f"Finished. Saved to: {os.path.basename(msg)}")
            else:
                self.status_label.setText(f"Finished: {msg}")
                
        w.signals.finished.connect(on_finished)
        w.signals.error.connect(lambda msg: self.status_label.setText(f"Error: {msg}"))
        self.threadpool.start(w)

    def _export_ta(self):
        self.status_label.setText("Exporting TA\u2026")
        self._run_in_thread(export_ta, self.stack, self.newwave, self.time_ps,
                            int(self.cory), self.settings.filename_sum, self.newpos)

    def _export_tam(self):
        self.status_label.setText("Exporting TAM\u2026")
        self._run_in_thread(export_tam, self.stack, self.newpos, self.time_ps,
                            int(self.corx), self.settings.filename_sum, self.newwave)

    def _export_dy(self):
        self.status_label.setText("Exporting dynamics\u2026")
        self._run_in_thread(export_dynamics, self.stack, self.time_ps,
                            list(self.selected_spots), int(self.corx), int(self.cory),
                            self.newwave, self.newpos, self.settings.filename_sum,
                            int(self.dyn_avg_spin.value()))

    def _export_fit(self):
        self.status_label.setText("Exporting Gaussian fit\u2026")
        self._run_in_thread(export_gaussian_fit, self.time_ps,
                            self.G_STD, self.G_STD_ERR,
                            int(self.corx), self.newwave, self.settings.filename_sum)

    def _export_all_dy(self):
        self.status_label.setText("Exporting all dynamics (batch)\u2026")
        s = self.settings
        self._run_in_thread(export_all_dynamics,
                            list(self.selected_spots), int(self.corx), int(self.cory),
                            self.newwave, self.newpos, s.filename_sum,
                            s.N1, s.N2, s.crop1, s.crop2, s.crop3, s.crop4, s.smthk,
                            int(self.dyn_avg_spin.value()),
                            progress_callback='inject')

    def _export_all_fits(self):
        self.status_label.setText("Exporting all fits (batch)\u2026")
        s = self.settings
        self._run_in_thread(export_all_fits,
                            list(self.selected_spots), int(self.corx), int(self.cory),
                            self.newwave, self.newpos, s.filename_sum,
                            s.N1, s.N2, s.crop1, s.crop2, s.crop3, s.crop4, s.smthk,
                            int(self.fit_avg_spin.value()), self._fit_start_index(), False,
                            progress_callback='inject')

    def _export_all_fits_max(self):
        self.status_label.setText("Exporting all fits max-track (batch)\u2026")
        s = self.settings
        self._run_in_thread(export_all_fits,
                            list(self.selected_spots), int(self.corx), int(self.cory),
                            self.newwave, self.newpos, s.filename_sum,
                            s.N1, s.N2, s.crop1, s.crop2, s.crop3, s.crop4, s.smthk,
                            int(self.fit_avg_spin.value()), self._fit_start_index(), True,
                            progress_callback='inject')


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.environ.setdefault("QT_LOGGING_RULES", "qt.core.qobject.connect.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = TAMViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

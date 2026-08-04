# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

- **GitHub**: [Purdue-Huang-Lab/Automation](https://github.com/Purdue-Huang-Lab/Automation) (transferred from `moonriver366/Automation` on 2026-06-16)
- **History note**: All commit authorship was unified to `moonriver366 <wang6560@purdue.edu>` on 2026-06-16 via `git filter-repo`. Any clone made before that date has stale SHAs — re-clone from the org URL above.

## Environment

- **Python**: 3.10 (strictly required — hardware SDKs are version-pinned)
- **Venv**: `.venv\Scripts\python.exe` (never use the system Python)
- **OS**: Windows only (hardware drivers, PowerShell launchers, DLL paths)

Activate: `.venv\Scripts\activate`
Run any GUI: `.\run_<name>.ps1` (all launchers live at repo root)
Or directly: `.venv\Scripts\python.exe -m <module.path>`

Update requirements after installing new packages:
```powershell
[System.IO.File]::WriteAllText("$PWD\requirements.txt", (pip freeze | Out-String), [System.Text.UTF8Encoding]::new($false))
```

No test suite exists. Verification is done by launching the relevant GUI.

## Data Directory

Measurement output goes to `DATA_DIR`, configured in [measurements/config.py](measurements/config.py):
- Default: `C:\Data\Minxue\260402gatedtrilayer`
- Override: set `AUTOMATION_DATA_DIR` environment variable

`ROOT_DIR` points to the repo root (one level above `measurements/`).

## Architecture

This is a PyQt5 lab automation framework. Code is organized in three layers:

### 1. Hardware Wrappers (`andor/`, `keithley/`, `ph300/`, `PM100A/`, `rot/`, `spad23/`)

Each device module exposes a single wrapper class with a consistent interface: `open()` / `close()`, configuration setters, acquisition methods, and optional `verbose` logging. Wrappers are thread-safe via `RLock`. They do not import Qt and have no GUI dependency.

Key classes: `AndorSystem`, `KeithleySMU`, `PicoHarp300`, `PM100ADevice`, `MotionController`.

### 2. Measurement Modules (`measurements/`)

Complex measurements follow the **Config → Workers → Widget → Entry** pattern:

| File | Role |
|------|------|
| `*_config.py` | Constants: device serials, default exposure, ROI coordinates, etc. |
| `*_workers.py` | `QThread` subclasses — acquire data, emit signals (`point_ready`, `frame_ready`, `done`) |
| `*_widget.py` | `QWidget` subclass — manages device connections, sweep control, live plotting, data saving |
| `*_gui.py` | Thin entry point: `QMainWindow` wraps the central widget |

Workers emit signals; widgets connect slots. All hardware access in measurements goes through the wrappers above. `measurements/config.py` provides shared defaults across measurement modules (device serials, Andor settings, ROI bounds) and imports device-level configs where available.

Submodule measurements (`pump_probe_gate_dep/`, `trpl_gate_power_dep/`) follow the same pattern but are launched via `gui_integrated.py` or `__main__.py` within the subdirectory.

Offline data viewers (`*_viewer*.py`) load `.csv`/`.txt`/`.npy` files from disk and do no hardware access.

### 3. Launch Scripts (`run_*.ps1`)

Every script activates the venv and calls either `python -m module.gui_entry` (simple) or `python path/to/gui_integrated.py` (submodule). They are the canonical way to start any GUI.

## Writing v2 GUIs

All new GUIs follow the v2 style. The reference implementations are [andor/gui_v2/](andor/gui_v2/), [keithley/gui_v2/](keithley/gui_v2/), and [ph300/gui_v2/](ph300/gui_v2/).

### Graphics engine: pyqtgraph only

Use `pyqtgraph` for all live and interactive plots. Never use `matplotlib` in a live-updating GUI (it is too slow and not thread-safe with Qt). matplotlib is acceptable only in offline viewers for one-shot publication-quality figures.

**Required initialization at the top of every entry-point file**, before any widget is created:

```python
from PyQt5 import QtWidgets  # PyQt5 must come before pyqtgraph
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k', useOpenGL=False)
```

HiDPI fix is handled globally by `.venv/Lib/site-packages/sitecustomize.py` — do not add DPI env vars or `AA_EnableHighDpiScaling` to entry-point files.

**Root cause**: `QScreen::devicePixelRatio()` returns the fractional display scale (e.g. 1.75 at 175%) regardless of Qt env vars. PyQtGraph reads it directly when allocating its graphics backing store, causing axes (logical pixels) and image items (physical pixels) to misalign. **Fix**: `sitecustomize.py` calls `ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_ssize_t(-1))` (`DPI_AWARENESS_CONTEXT_UNAWARE`) before Qt is imported, making Windows report DPR=1.0 and handle scaling via bitmap upscale. Combined with `QT_ENABLE_HIGHDPI_SCALING=0` and `QT_SCALE_FACTOR=1`. The newer `SetProcessDpiAwarenessContext` (Win10 1607+) is used instead of the older `shcore.SetProcessDpiAwareness` because the latter returns `E_ACCESSDENIED` silently when the VS Code launcher or Python launcher has already set a DPI awareness level — the newer API succeeds in that case. **If the venv is rebuilt, recreate `sitecustomize.py` from this note.**

For `PlotWidget` instances created later, also set axis pens explicitly so labels stay black on the white background:

```python
for ax in ("bottom", "left"):
    widget.pw.getAxis(ax).setPen(pg.mkPen("k"))
    widget.pw.getAxis(ax).setTextPen(pg.mkPen("k"))
```

### File layout for a v2 GUI package

Create a `gui_v2/` subdirectory under the device or measurement directory:

| File | Contents |
|------|----------|
| `config.py` | Width constants (`BTN_W`, `STATUS_W`), poll interval, all GUI defaults |
| `workers.py` | `QThread` subclasses for blocking device I/O |
| `plots.py` | Self-contained `QWidget` subclasses that wrap pyqtgraph plots |
| `widget.py` | Main `QWidget` — control panel + live display |
| `app.py` | `MainWindow(QMainWindow)` + `main()` entry point |

The entry-point module at the device root (e.g. `andor_gui_v2.py`) just imports from `gui_v2.app` and re-exports.

### Dashboard layout structure

`_build_ui()` builds the layout in a fixed three-zone order:

1. **Top bar** (`QHBoxLayout`): Connect / Disconnect / Start / Stop buttons, then `addStretch()`, then a persistent `QLabel` for state ("Connected", "FPS: 1.23").
2. **Middle** (`QHBoxLayout`): One or more `QGroupBox` panels using `QGridLayout` for settings. Use `setHorizontalSpacing(8)`, `setVerticalSpacing(6)`, `setContentsMargins(8, 8, 8, 8)` on every grid.
3. **Live display**: The plot/image widget added with a stretch factor of 1 so it fills remaining space.
4. **Status bar** (`QStatusBar`): Embedded as the last item in the outer `QVBoxLayout` — not taken from `QMainWindow`. Use it for transient messages ("Settings applied", "Saving…").

Button widths come from `config.BTN_W`. Disable buttons that are not valid in the current state (e.g. Start is disabled before connect, Stop is disabled when idle) and update them in every state transition.

### Polling and live acquisition

**Never call device I/O in the GUI (main) thread.**

Two patterns are used:

- **Continuous acquisition** (camera live view, sweep workers): `QThread` subclass with a `_stop: bool` flag. `run()` loops calling the device, emits `frame_ready` / `progress` / `done` signals. Rate-limit with:
  ```python
  elapsed = time.monotonic() - loop_start
  if (sleep_s := self._min_interval_s - elapsed) > 0:
      time.sleep(sleep_s)
  ```
  The widget calls `thread.stop()` then polls with `QTimer.singleShot(200, self._check_stop)` — never `thread.wait()` in the GUI thread.

- **Status polling** (temperature, count rates, instrument readbacks): `QTimer` started on connect, stopped on disconnect:
  ```python
  self._poll_timer = QtCore.QTimer(self)
  self._poll_timer.timeout.connect(self._on_poll)
  # on connect:
  self._poll_timer.start(POLL_MS)   # e.g. 500–1000 ms
  # on disconnect:
  self._poll_timer.stop()
  ```

### Live image display with linecuts

Use a `pg.GraphicsLayoutWidget` containing three sub-plots arranged as a 2×2 grid (image top-left, vertical linecut top-right, horizontal linecut bottom-left). Set stretch factors to keep the image dominant:

```python
glw = pg.GraphicsLayoutWidget()
img_plot  = glw.addPlot(row=0, col=0)
vcut_plot = glw.addPlot(row=0, col=1)
hcut_plot = glw.addPlot(row=1, col=0)
glw.ci.layout.setColumnStretchFactor(0, 4)
glw.ci.layout.setColumnStretchFactor(1, 1)
glw.ci.layout.setRowStretchFactor(0, 4)
glw.ci.layout.setRowStretchFactor(1, 1)
```

Image display details:
- `pg.ImageItem` with a grayscale LUT; call `img_plot.invertY(True)` so row 0 is at the top (matching the physical array convention).
- Display by `np.fliplr(np.flipud(raw))` before passing to `img_item.setImage()`. Keep the unflipped raw array for linecut arithmetic.
- Set levels explicitly: `img_item.setLevels([vmin, vmax])`; do not use `autoLevels=True` in hot loops.
- Crosshairs: two `pg.InfiniteLine` items (angle=0 and angle=90) hidden until the user clicks. Wire the click via `img_plot.scene().sigMouseClicked.connect(handler)` and convert scene → view coordinates with `img_plot.vb.mapSceneToView(pos)`.
- Linecut width: sum rows/cols within a band (`array[r1:r2, :].sum(axis=0)`), controlled by a `QSpinBox`.
- ROI rectangle: a single `PlotCurveItem` drawn as a closed polygon `[x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1]`; update with `setData()` on every frame rather than recreating it.
- Encapsulate all of the above in a dedicated `QGroupBox` subclass so it can be dropped into any measurement widget.

### Separate plot widgets

Extract each distinct chart into its own `QWidget` subclass in `plots.py`. The widget owns the `pg.PlotWidget` and exposes a minimal update API (`add_point`, `update_live`, `clear`). This keeps the main widget free of pyqtgraph internals. See [keithley/gui_v2/plots.py](keithley/gui_v2/plots.py) and [ph300/gui_v2/plots.py](ph300/gui_v2/plots.py) for examples.

## Power and Gate Dependence Patterns

These patterns are used by every measurement that sweeps power and/or gate voltage. Follow the existing implementations in [measurements/pl_power_voltage_widget.py](measurements/pl_power_voltage_widget.py) and [measurements/trpl_gate_power_dep/gui.py](measurements/trpl_gate_power_dep/gui.py).

### Power source: three modes

The power source mode is determined at load time by auto-detecting the calibration file format. Keep the single `_calib_mode` string (`"dual"`, `"hwp_nd"`, `"angle_list"`) to drive all downstream logic.

**Mode 1 — Dual wheel** (calibration file from `dual_wheel_intensity_calib_gui.py`)

CSV header: `a_deg, b_deg, power_<unit>` (unit embedded in column name, e.g. `power_nW`).
Loaded by `load_dual_wheel_calibration()` → `DualWheelPowerData.entries: List[DualWheelPowerEntry]`.
Sweep order follows file order exactly; no sorting. Both rotation stages (A and B) must be connected.

Detection: `is_dual_wheel_calibration(path)` — checks that the header has `a`/`b`/`power` columns.

**Mode 2 — One wheel + HWP** (calibration file from `wheel_hwp_power_calib_gui.py`)

CSV has two sections separated by `[HWP_SWEEP]` and `[ND_SWEEP]` markers. Each row: `angle_deg, power_w, std_w`.
Loaded by `load_hwp_nd_calibration()` → `HWPNDCalibData` with separate `hwp_angles`, `hwp_powers_w`, `nd_angles`, `nd_powers_w` lists.

After loading, show an ND-wheel-angle `QComboBox` (one item per `nd_angles` entry) and an HWP range `QDoubleSpinBox` pair (min/max). When either changes, call `compute_hwp_nd_entries(data, nd_angle)` to produce a fresh `List[DualWheelPowerEntry]` (HWP stage → `a_deg`, ND stage stays at fixed `b_deg`).

Detection: `is_hwp_nd_calibration(path)` — checks for a `[HWP_SWEEP]` line.

**Mode 3 — Manual angle + power table** (no calibration file)

Present a `QTableWidget` with two columns: **Angle (°)** and **Power (W)**. Provide Add Row and Delete Row buttons. Parse rows into a `List[DualWheelPowerEntry]` (set `b_deg=0`, `intensity=nan`) when the sweep starts. Only one rotation stage (A) is needed. No skip-N control — the user already chose the exact entries.

**Loading flow** (from `on_load_calib`):

```python
if is_hwp_nd_calibration(path):
    data = load_hwp_nd_calibration(path)
    self._load_hwp_nd_calibration(data)   # sets _calib_mode = "hwp_nd"
elif is_dual_wheel_calibration(path):
    data = load_dual_wheel_calibration(path)
    self._load_dual_calibration(data)     # sets _calib_mode = "dual"
```

Show/hide the ND-wheel-angle combo and HWP range spinboxes based on mode; hide them for dual-wheel.

### Skip N powers (modes 1 and 2 only)

Add a `QSpinBox` (range 0–100 000, default 0, label "Skip N power points") to the sweep options panel. Apply at sweep start:

```python
power_stride = max(1, skip_n + 1)
entries = list(self._dual_entries)[::power_stride]
```

`skip_n = 0` → all entries; `skip_n = 1` → every other; etc. Disable the spinbox for the manual table mode and when the sweep is running.

### Gate voltage (follow `trpl_gate_power_dep` exactly)

Use a `QGroupBox` with a `QStackedWidget` to switch between two input modes:

- **Manual list**: a `QTextEdit` accepting comma-, newline-, or semicolon-separated floats for Va. Parse with:
  ```python
  def _parse_float_list(text: str) -> List[float]:
      vals = []
      for tok in text.replace("\n", ",").replace(";", ",").split(","):
          tok = tok.strip()
          if tok:
              try: vals.append(float(tok))
              except ValueError: pass
      return vals
  ```
- **Range**: three `QDoubleSpinBox` fields (Start, Step, End) plus a "Generate" button; build the list with a `while` loop tolerating floating-point drift (add `abs(step) * 1e-6` to the end condition).

Below the stack, show a `QDoubleSpinBox` for **Ratio** (Vb = ratio × Va, default `1.8`, range `−1000`–`1000`, 5 decimals). After the user sets or generates the Va list, compute Vb and update two preview `QLabel`s showing the first six values of each.

Wire `QRadioButton`s to switch the stack page. Cache the parsed list in `self._cached_va_list`; at sweep start build `vb_list = [ratio * v for v in va_list]` and zip into pairs — **not** a Cartesian product.

## Special Dependencies

- **spinnaker-python** (FLIR camera): not in `requirements.txt`; install manually from the FLIR SDK `.whl` after running `pip install -r requirements.txt`.
- **PHLib DLL** (PicoHarp 300): must be present at `C:\Program Files\PicoQuant\PH300-PHLibv30\demos\64\c\TTTRmode\PHLib64.dll` or the path configured in `ph300/ph300_wrapper.py`.

## Git Rules

- Never commit `*.csv`, `*.sif`, `*.h5`, `*.npy`, `*.npz`, `*.mat`, `*.dat` — measurement outputs.
- Never commit `.venv/`, `__pycache__/`, `.claude/settings.local.json`.
- `ph300/740 irf 20260311.csv` is a calibration file — **do** track it; do not add `ph300/*.csv` to `.gitignore`.
- Use feature branches for experimental changes; `main` is production.

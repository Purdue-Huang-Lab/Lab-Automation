import itertools
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, ScalarFormatter

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

from andor.andor_wrapper import AndorSystem
from andor.gui.live_view_widget import AndorLiveViewWidget
from andor.gui.workers import LiveAcqThread
from rot.rot_wrapper import MotionController, ROT_RANGE_DEFAULT, RotationStage, list_kinesis_serials

from measurements.intensity_calib_config import (
    DEFAULT_ACQ_NUMBER,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_POWER_CALIB_PATH,
    DEFAULT_ROI_X1,
    DEFAULT_ROI_X2,
    DEFAULT_ROI_Y1,
    DEFAULT_ROI_Y2,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_STAGE_SCALE,
    DEFAULT_STAGE_SERIAL,
    POLL_MS,
    DATA_DIR,
)
from measurements.intensity_workers import IntensitySweepThread
from measurements.power_calibration import load_power_calibration

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None


class IntensityCalibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)
        self.stage: Optional[RotationStage] = None

        self._live_thread: Optional[LiveAcqThread] = None
        self._sweep_thread: Optional[IntensitySweepThread] = None
        self._stage_controller: Optional[MotionController] = None
        self._stage_busy = False
        self._busy = False
        self._live_active = False

        self._save_dir = None
        self._results: List[dict] = []
        self._base_positions = None
        self._base_powers_w = None
        self._power_unit = "nW"
        self._power_scale = 1e-9
        self._nd_ratios = {}
        self._calib_path = ""
        self._sweep_start = None
        self._sweep_stop = None
        self._fit_func = None
        self._fit_ctx = None
        self._fit_xs = None
        self._fit_ys = None

        self._crop = (DEFAULT_CROP_TOP, DEFAULT_CROP_BOTTOM, DEFAULT_CROP_LEFT, DEFAULT_CROP_RIGHT)

        self._build_ui()
        self._update_roi_info()

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

        line1 = QtWidgets.QHBoxLayout()
        line1.addWidget(self.camDot)
        line1.addWidget(QtWidgets.QLabel("Andor:"))
        line1.addWidget(self.camStatusLbl, 1)
        line1.addStretch()
        line1.addWidget(QtWidgets.QLabel("Exp (ms):"))
        line1.addWidget(self.exposureSpin)
        line1.addWidget(QtWidgets.QLabel("Acq N:"))
        line1.addWidget(self.accumSpin)

        grid.addLayout(line1, 0, 0, 1, 1)

        self.stageDot = self._make_dot()
        self.stageStatusLbl = QtWidgets.QLabel("Stage: disconnected")
        self.stageStatusLbl.setFont(mono)

        self.stageSerialCombo = QtWidgets.QComboBox()
        self.stageSerialCombo.setEditable(True)
        self.stageSerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_SERIAL:
            self.stageSerialCombo.addItem(DEFAULT_STAGE_SERIAL)

        self.stageDetectBtn = QtWidgets.QPushButton("Search")
        self.stageDetectBtn.setFixedWidth(80)
        self.stageHomeBtn = QtWidgets.QPushButton("Home")
        self.stageHomeBtn.setFixedWidth(80)

        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.stageDot)
        line2.addWidget(QtWidgets.QLabel("Stage:"))
        line2.addWidget(self.stageStatusLbl, 1)
        line2.addStretch()
        line2.addWidget(QtWidgets.QLabel("Serial:"))
        line2.addWidget(self.stageSerialCombo)
        line2.addWidget(self.stageDetectBtn)
        line2.addWidget(self.stageHomeBtn)

        grid.addLayout(line2, 1, 0, 1, 1)

        self.initBtn = QtWidgets.QPushButton("Initialize")
        self.initBtn.setFixedWidth(120)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setFixedWidth(120)
        self.liveBtn = QtWidgets.QPushButton("Live")
        self.liveBtn.setFixedWidth(90)
        self.stopLiveBtn = QtWidgets.QPushButton("Stop Live")
        self.stopLiveBtn.setFixedWidth(110)
        self.sweepBtn = QtWidgets.QPushButton("Start Sweep")
        self.sweepBtn.setFixedWidth(120)
        self.abortBtn = QtWidgets.QPushButton("Abort")
        self.abortBtn.setFixedWidth(100)

        line3 = QtWidgets.QHBoxLayout()
        line3.addStretch()
        line3.addWidget(self.initBtn)
        line3.addWidget(self.disconnectBtn)
        line3.addSpacing(10)
        line3.addWidget(self.liveBtn)
        line3.addWidget(self.stopLiveBtn)
        line3.addWidget(self.sweepBtn)
        line3.addWidget(self.abortBtn)

        grid.addLayout(line3, 2, 0, 1, 1)

        parent_layout.addWidget(box)

    def _build_controls(self, parent_layout):
        box = QtWidgets.QGroupBox("Calibration / ROI / Sweep")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.saveNameEdit = QtWidgets.QLineEdit(self._default_save_name())
        self.saveNameEdit.setPlaceholderText("subfolder name under measurements/")

        self.calibPathEdit = QtWidgets.QLineEdit(DEFAULT_POWER_CALIB_PATH)
        self.calibBrowseBtn = QtWidgets.QPushButton("Browse")
        self.calibBrowseBtn.setFixedWidth(90)
        self.calibLoadBtn = QtWidgets.QPushButton("Load Calib")
        self.calibLoadBtn.setFixedWidth(110)
        self.calibInfoLbl = QtWidgets.QLabel("Calib: --")

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

        self.sweepStartSpin = QtWidgets.QDoubleSpinBox()
        self.sweepStopSpin = QtWidgets.QDoubleSpinBox()
        self.sweepStepSpin = QtWidgets.QDoubleSpinBox()
        for spin in (self.sweepStartSpin, self.sweepStopSpin, self.sweepStepSpin):
            spin.setDecimals(3)
            spin.setRange(-360.0, 360.0)
        self.sweepStartSpin.setValue(0.0)
        self.sweepStopSpin.setValue(90.0)
        self.sweepStepSpin.setValue(5.0)

        self.saveBaseBtn = QtWidgets.QPushButton("Save Base CSV")
        self.saveBaseBtn.setFixedWidth(130)
        self.saveExtBtn = QtWidgets.QPushButton("Save Extended CSV")
        self.saveExtBtn.setFixedWidth(150)

        grid.addWidget(QtWidgets.QLabel("Save subfolder:"), 0, 0)
        grid.addWidget(self.saveNameEdit, 0, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Power calib file:"), 1, 0)
        grid.addWidget(self.calibPathEdit, 1, 1)
        grid.addWidget(self.calibBrowseBtn, 1, 2)
        grid.addWidget(self.calibLoadBtn, 1, 3)
        grid.addWidget(self.calibInfoLbl, 1, 4, 1, 2)

        grid.addWidget(QtWidgets.QLabel("ROI x1/x2:"), 2, 0)
        grid.addWidget(self.roiX1, 2, 1)
        grid.addWidget(self.roiX2, 2, 2)
        grid.addWidget(QtWidgets.QLabel("ROI y1/y2:"), 2, 3)
        grid.addWidget(self.roiY1, 2, 4)
        grid.addWidget(self.roiY2, 2, 5)
        grid.addWidget(self.roiApplyBtn, 2, 6)
        grid.addWidget(self.roiInfoLbl, 2, 7)

        grid.addWidget(QtWidgets.QLabel("Sweep start/stop/step (deg):"), 3, 0)
        grid.addWidget(self.sweepStartSpin, 3, 1)
        grid.addWidget(self.sweepStopSpin, 3, 2)
        grid.addWidget(self.sweepStepSpin, 3, 3)
        grid.addWidget(self.saveBaseBtn, 3, 4)
        grid.addWidget(self.saveExtBtn, 3, 5)

        parent_layout.addWidget(box)

    def _build_views(self, parent_layout):
        box = QtWidgets.QGroupBox("Intensity Calibration")
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.liveView = AndorLiveViewWidget(self.cam, title="Andor Live")
        self.liveView.set_crop(*self._crop)
        self.liveView.set_roi(self.roiX1.value(), self.roiX2.value(), self.roiY1.value(), self.roiY2.value())

        self.fig_int = Figure(figsize=(4, 4.5), dpi=100)
        self.canvas_int = FigureCanvas(self.fig_int)
        self.ax_int = self.fig_int.add_subplot(111)
        self.ax_power = self.ax_int.twinx()
        self.ax_int.set_title("ROI Intensity vs Position")
        self.ax_int.set_xlabel("Position (deg)")
        self.ax_int.set_ylabel("Integrated Intensity")
        self.line_int, = self.ax_int.plot([], [], marker="o", ls="-", ms=3, color="tab:blue")
        self.line_power, = self.ax_power.plot([], [], ls="--", lw=1.2, color="tab:orange")
        self.ax_power.set_ylabel("Power")
        self.ax_power.tick_params(axis="y", colors="tab:orange")

        layout.addWidget(self.liveView, 3)
        layout.addWidget(self.canvas_int, 2)
        parent_layout.addWidget(box, 1)

    def _wire_signals(self):
        self.initBtn.clicked.connect(self.on_initialize)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.liveBtn.clicked.connect(self.on_live)
        self.stopLiveBtn.clicked.connect(self.on_stop_live)
        self.sweepBtn.clicked.connect(self.on_sweep)
        self.abortBtn.clicked.connect(self.on_abort)
        self.calibBrowseBtn.clicked.connect(self.on_browse_calib)
        self.calibLoadBtn.clicked.connect(self.on_load_calib)
        self.stageDetectBtn.clicked.connect(self.on_stage_detect)
        self.stageHomeBtn.clicked.connect(self.on_stage_home)
        self.roiApplyBtn.clicked.connect(self.on_apply_roi)
        self.saveBaseBtn.clicked.connect(self.on_save_base)
        self.saveExtBtn.clicked.connect(self.on_save_extended)
        self.sweepStartSpin.valueChanged.connect(self._on_sweep_range_changed)
        self.sweepStopSpin.valueChanged.connect(self._on_sweep_range_changed)

    # -----------------
    # Helpers
    # -----------------
    def _make_dot(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setFixedSize(10, 10)
        lbl.setStyleSheet("border-radius:5px; background-color:#c62828;")
        return lbl

    def _set_dot(self, lbl: QtWidgets.QLabel, ok: bool) -> None:
        color = "#2e7d32" if ok else "#c62828"
        lbl.setStyleSheet(f"border-radius:5px; background-color:{color};")

    def _default_save_name(self) -> str:
        return time.strftime("roi_calib_%Y%m%d_%H%M%S")

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
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return live or sweep

    def _is_busy(self) -> bool:
        return self._busy or self._stage_busy or self._threads_running()

    def _sync_controls(self) -> None:
        cam_ok = self.cam.cam is not None
        stage_ok = self.stage is not None
        busy = self._is_busy()
        self.initBtn.setEnabled(not busy)
        self.disconnectBtn.setEnabled(not busy)
        self.liveBtn.setEnabled(cam_ok and not busy)
        self.stopLiveBtn.setEnabled(self._threads_running() and self._live_active)
        self.sweepBtn.setEnabled(cam_ok and stage_ok and self._base_positions is not None and not busy)
        self.abortBtn.setEnabled(busy)
        self.stageHomeBtn.setEnabled(stage_ok and not busy)

    def _roi_tuple(self) -> Optional[Tuple[int, int, int, int]]:
        x1 = int(self.roiX1.value())
        x2 = int(self.roiX2.value())
        y1 = int(self.roiY1.value())
        y2 = int(self.roiY2.value())
        if x1 == x2 or y1 == y2:
            return None
        return (x1, x2, y1, y2)

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

    def _interp_power_w(self, pos_deg: float) -> Optional[float]:
        ctx = self._fit_ctx
        if ctx is None or self._fit_xs is None or self._fit_ys is None:
            return None
        target = self._to_sweep_axis(float(pos_deg), ctx)
        try:
            if self._fit_func is not None:
                return float(self._fit_func(target))
            if self._fit_xs.size == 1:
                return float(self._fit_ys[0])
            return float(np.interp(target, self._fit_xs, self._fit_ys))
        except Exception:
            return None

    def _angle_range(self) -> Tuple[float, float]:
        if self.stage is not None and hasattr(self.stage, "angle_range"):
            lo, hi = self.stage.angle_range
            return float(lo), float(hi)
        return float(ROT_RANGE_DEFAULT[0]), float(ROT_RANGE_DEFAULT[1])

    def _sweep_context(self, *, use_current: bool = False) -> dict:
        if not use_current and self._sweep_start is not None and self._sweep_stop is not None:
            start = float(self._sweep_start)
            stop = float(self._sweep_stop)
        else:
            start = float(self.sweepStartSpin.value())
            stop = float(self.sweepStopSpin.value())
        lo, hi = self._angle_range()
        if lo > hi:
            lo, hi = hi, lo
        width = hi - lo
        start = max(lo, min(hi, start))
        stop = max(lo, min(hi, stop))
        wrap = width >= 360.0 - 1e-6 and start > stop
        return {
            "start": start,
            "stop": stop,
            "lo": lo,
            "hi": hi,
            "width": width,
            "wrap": wrap,
        }

    def _to_sweep_axis(self, angle: float, ctx: dict) -> float:
        start = float(ctx["start"])
        width = float(ctx["width"])
        wrap = bool(ctx["wrap"])
        val = float(angle)
        if wrap and val < start:
            val += width
        return val

    def _base_curve_sweep(self):
        if self._base_positions is None or self._base_powers_w is None:
            return None
        use_current = self._sweep_start is None
        if self._sweep_thread is not None and self._sweep_thread.isRunning():
            use_current = False
        elif not self._results:
            use_current = True
        ctx = self._sweep_context(use_current=use_current)
        start = float(ctx["start"])
        stop = float(ctx["stop"])
        wrap = bool(ctx["wrap"])

        def _in_path(pos: float) -> bool:
            if wrap:
                return pos >= start or pos <= stop
            if start <= stop:
                return start <= pos <= stop
            return stop <= pos <= start

        pos_map = {}
        for pos, power in zip(self._base_positions, self._base_powers_w):
            if not _in_path(float(pos)):
                continue
            sweep_pos = self._to_sweep_axis(float(pos), ctx)
            pos_map.setdefault(sweep_pos, []).append(float(power))
        xs = sorted(pos_map.keys())
        ys = [float(np.mean(pos_map[x])) for x in xs]
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), ctx

    def _apply_sweep_axis_format(self, ctx: dict) -> None:
        wrap = bool(ctx["wrap"])
        lo = float(ctx["lo"])
        width = float(ctx["width"])
        if wrap and width > 0:
            def _fmt(x, _pos=None):
                val = ((x - lo) % width) + lo
                return f"{val:g}"
            formatter = FuncFormatter(_fmt)
            self.ax_int.xaxis.set_major_formatter(formatter)
            self.ax_power.xaxis.set_major_formatter(formatter)
            self.ax_int.set_xlabel("Position (deg; 360=0)")
        else:
            fmt = ScalarFormatter()
            fmt.set_useOffset(False)
            self.ax_int.xaxis.set_major_formatter(fmt)
            self.ax_power.xaxis.set_major_formatter(fmt)
            self.ax_int.set_xlabel("Position (deg)")

    # -----------------
    # Status polling
    # -----------------
    def _poll_status(self) -> None:
        cam_ok = self.cam.cam is not None
        stage_ok = self.stage is not None

        self._set_dot(self.camDot, cam_ok)
        self._set_dot(self.stageDot, stage_ok)

        if cam_ok:
            temp = self.cam.get_temperature_c()
            temp_s = f"{temp:.1f} C" if temp is not None else "--"
            exp_ms = float(self.exposureSpin.value())
            acc = int(self.accumSpin.value())
            self.camStatusLbl.setText(f"T={temp_s} | Exp {exp_ms:g} ms x{acc}")
        else:
            self.camStatusLbl.setText("Disconnected")

        if stage_ok:
            try:
                pos = float(self.stage.get_position())
            except Exception:
                pos = None
            try:
                homed = self.stage.is_homed()
            except Exception:
                homed = None
            try:
                moving = bool(self.stage.is_moving())
            except Exception:
                moving = False
            pos_s = f"{pos:.3f} deg" if pos is not None else "--"
            home_s = "yes" if homed else ("no" if homed is False else "?")
            status = "Moving..." if (self._stage_busy or moving) else "Ready"
            self.stageStatusLbl.setText(f"SN {self.stage.serial} | Homed {home_s} | {status} | Angle {pos_s}")
        else:
            self.stageStatusLbl.setText("Disconnected")

    # -----------------
    # Actions
    # -----------------
    def on_initialize(self):
        if self._busy:
            return
        try:
            self.cam.connect()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Andor connect", str(exc))
            return
        self._apply_camera_defaults()

        serial = self.stageSerialCombo.currentText().strip()
        if serial:
            try:
                self.stage = RotationStage(serial, scale=DEFAULT_STAGE_SCALE)
                self.stage.open()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Stage connect", str(exc))
                self.stage = None

        self.statusBar().showMessage("Initialized")
        self._sync_controls()

    def on_disconnect(self):
        self.on_abort()
        try:
            if self.stage:
                self.stage.close()
        except Exception:
            pass
        self.stage = None
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
        current = self.stageSerialCombo.currentText().strip()
        self.stageSerialCombo.blockSignals(True)
        self.stageSerialCombo.clear()
        self.stageSerialCombo.addItems(serials)
        if current:
            self.stageSerialCombo.setCurrentText(current)
        self.stageSerialCombo.blockSignals(False)
        self.statusBar().showMessage(f"Detected {len(serials)} stage(s)")

    def on_stage_home(self):
        if self.stage is None or self._stage_busy:
            return
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        try:
            self.stage.home(controller=self._stage_controller)
            self.statusBar().showMessage("Home OK")
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
        self._stage_busy = False
        self._sync_controls()

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
        if self._sweep_thread is not None and self._sweep_thread.isRunning():
            self._sweep_thread.stop()
        if self._stage_controller is not None:
            try:
                self._stage_controller.abort()
            except Exception:
                pass
        try:
            if self.stage is not None:
                self.stage.stop()
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

    def on_browse_calib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select calibration file",
            os.path.join(DATA_DIR, "measurements"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.calibPathEdit.setText(path)

    def on_load_calib(self):
        path = self.calibPathEdit.text().strip()
        try:
            data = load_power_calibration(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        base = [e for e in data.entries if e.is_base]
        if not base:
            QtWidgets.QMessageBox.warning(self, "Load error", "No base entries found.")
            return
        pos_map = {}
        for e in base:
            pos_map.setdefault(e.position_deg, []).append(e.power_w)
        positions = sorted(pos_map.keys())
        powers = [float(np.mean(pos_map[p])) for p in positions]
        self._base_positions = np.asarray(positions, dtype=float)
        self._base_powers_w = np.asarray(powers, dtype=float)
        self._power_unit = data.unit
        self._power_scale = data.scale
        self._nd_ratios = dict(data.nd_ratios)
        self._calib_path = path
        self.calibInfoLbl.setText(f"Calib: {len(base)} base pts, {len(self._nd_ratios)} ND")
        self._update_power_curve()
        self.statusBar().showMessage(f"Loaded calibration: {os.path.basename(path)}")
        self._sync_controls()

    def on_apply_roi(self):
        roi = self._roi_tuple()
        if roi is None:
            self.liveView.clear_roi()
        else:
            self.liveView.set_roi(*roi)
        self._update_roi_info()

    def on_sweep(self):
        if self._busy or self.cam.cam is None or self.stage is None:
            return
        if self._base_positions is None:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Load a power calibration file first.")
            return
        start = float(self.sweepStartSpin.value())
        stop = float(self.sweepStopSpin.value())
        step = float(self.sweepStepSpin.value())
        if step == 0:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Step cannot be 0.")
            return
        positions = self._gen_points(start, stop, step)
        if not positions:
            QtWidgets.QMessageBox.warning(self, "Sweep", "No sweep points generated.")
            return

        self._results.clear()
        self._sweep_start = start
        self._sweep_stop = stop
        self._resolve_save_dir()
        roi = self._roi_tuple()
        self.on_apply_roi()
        self._update_power_curve()

        self._sweep_thread = IntensitySweepThread(
            self.cam,
            self.stage,
            positions,
            int(self.accumSpin.value()),
            float(self.exposureSpin.value()),
            float(DEFAULT_RAMP_STEP_DEG),
            self._crop,
            roi,
        )
        self._sweep_thread.frame_ready.connect(self._on_sweep_frame)
        self._sweep_thread.point_ready.connect(self._on_sweep_point)
        self._sweep_thread.status.connect(self.statusBar().showMessage)
        self._sweep_thread.done.connect(self._on_sweep_done)
        self._sweep_thread.start()
        self._set_busy(True)
        self.statusBar().showMessage("Sweep started")

    def _gen_points(self, start: float, stop: float, step: float) -> List[float]:
        start = float(start)
        stop = float(stop)
        step = float(step)
        if step == 0:
            return []
        ctx = self._sweep_context(use_current=True)
        start = float(ctx["start"])
        stop = float(ctx["stop"])
        lo = float(ctx["lo"])
        hi = float(ctx["hi"])
        wrap = bool(ctx["wrap"])

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

    def _on_live_frame(self, fr: dict):
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def _on_sweep_frame(self, fr: dict):
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def _on_sweep_point(self, pos_deg: float, intensity: float, image):
        power_w = self._interp_power_w(pos_deg)
        ctx = self._sweep_context()
        sweep_deg = self._to_sweep_axis(pos_deg, ctx)
        if power_w is None:
            power_w = float("nan")
        self._results.append({
            "position_deg": float(pos_deg),
            "sweep_deg": float(sweep_deg),
            "power_w": float(power_w),
            "series": "base",
            "intensity": float(intensity),
        })
        self._update_plot()

    def _on_sweep_done(self, status: str):
        self._set_busy(False)
        if status == "ok":
            self.statusBar().showMessage("Sweep complete")
            self._save_base_csv()
        elif status == "aborted":
            self.statusBar().showMessage("Sweep aborted")
        else:
            self.statusBar().showMessage(str(status))

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._is_busy():
            self._poll_timer.stop()
        else:
            if not self._poll_timer.isActive():
                self._poll_timer.start(int(POLL_MS))
        self._sync_controls()

    def _update_plot(self):
        if not self._results:
            self.line_int.set_data([], [])
        else:
            xs = [r.get("sweep_deg", r["position_deg"]) for r in self._results]
            ys = [r["intensity"] for r in self._results]
            self.line_int.set_data(xs, ys)
        self.ax_int.relim()
        self.ax_int.autoscale_view()
        self.canvas_int.draw_idle()

    def _update_power_curve(self) -> None:
        sweep = self._base_curve_sweep()
        if sweep is None:
            self._fit_func = None
            self._fit_ctx = None
            self._fit_xs = None
            self._fit_ys = None
            self.line_power.set_data([], [])
            self.ax_power.set_ylabel("Power")
            self.canvas_int.draw_idle()
            return
        xs, ys, ctx = sweep
        order = np.argsort(xs)
        xs = np.asarray(xs, dtype=float)[order]
        ys = np.asarray(ys, dtype=float)[order]
        self._fit_ctx = ctx
        self._fit_xs = xs
        self._fit_ys = ys
        self._fit_func = None
        if PchipInterpolator is not None and xs.size >= 2:
            try:
                self._fit_func = PchipInterpolator(xs, ys, extrapolate=True)
            except Exception:
                self._fit_func = None

        scale = float(self._power_scale or 1.0)
        if self._fit_func is not None:
            xgrid = np.linspace(xs[0], xs[-1], 400)
            powers = self._fit_func(xgrid) / scale
            self.line_power.set_data(xgrid, powers)
        else:
            powers = ys / scale
            self.line_power.set_data(xs, powers)

        self.ax_power.set_ylabel(f"Power ({self._power_unit})")
        self.ax_power.relim()
        self.ax_power.autoscale_view()
        self._apply_sweep_axis_format(ctx)
        self.canvas_int.draw_idle()

    def _on_sweep_range_changed(self):
        if not self._results:
            self._update_power_curve()

    def on_save_base(self):
        if not self._results:
            self.statusBar().showMessage("No data to save")
            return
        self._resolve_save_dir()
        self._save_base_csv()

    def _save_base_csv(self):
        if not self._results or not self._save_dir:
            return
        fname = os.path.join(self._save_dir, "roi_power_calib_base.csv")
        self._write_csv(fname, self._results, include_nd=True)
        self.statusBar().showMessage(f"Saved {os.path.basename(fname)}")

    def on_save_extended(self):
        if not self._results:
            self.statusBar().showMessage("No data to extend")
            return
        if not self._nd_ratios:
            self.statusBar().showMessage("No ND ratios loaded")
            return
        self._resolve_save_dir()
        fname = os.path.join(self._save_dir, "roi_power_calib_extended.csv")
        rows = list(self._results)
        combos = self._combo_keys()
        for entry in self._results:
            base_power = entry["power_w"]
            for combo in combos:
                ratio = self._combo_ratio(combo)
                if ratio is None or ratio <= 0:
                    continue
                rows.append({
                    "position_deg": entry["position_deg"],
                    "power_w": base_power / ratio,
                    "series": self._series_label(combo),
                    "intensity": entry["intensity"],
                })
        self._write_csv(fname, rows, include_nd=True)
        self.statusBar().showMessage(f"Saved {os.path.basename(fname)}")

    def _write_csv(self, path: str, rows: List[dict], include_nd: bool = True):
        lines = []
        if include_nd and self._nd_ratios:
            for idx in sorted(self._nd_ratios.keys()):
                ratio = self._nd_ratios[idx]
                lines.append(f"#nd_filter,nd #{idx},{ratio}")
        header = f"position_deg,power_{self._power_unit},series,intensity"
        lines.append(header)
        for row in rows:
            power = row.get("power_w", float("nan")) / float(self._power_scale or 1.0)
            pos = row.get("position_deg", 0.0)
            series = row.get("series", "base")
            intensity = row.get("intensity", 0.0)
            lines.append(f"{pos:.6f},{power:.9g},{series},{intensity:.9g}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}")

    def _combo_keys(self):
        ids = sorted(self._nd_ratios.keys())
        combos: List[Tuple[int, ...]] = []
        for r in range(1, len(ids) + 1):
            combos.extend(tuple(c) for c in itertools.combinations(ids, r))
        return combos

    def _combo_ratio(self, combo: Tuple[int, ...]) -> Optional[float]:
        if not combo:
            return 1.0
        ratio = 1.0
        for fid in combo:
            if fid not in self._nd_ratios:
                return None
            ratio *= float(self._nd_ratios[fid])
        return ratio

    def _series_label(self, combo: Tuple[int, ...]) -> str:
        if not combo:
            return "base"
        parts = [f"nd #{c}" for c in combo]
        return "+".join(parts)

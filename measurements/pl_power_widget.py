import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from andor.andor_wrapper import AndorSystem
from andor.gui.workers import LiveAcqThread, SingleAcqThread
from andor.gui.live_view_widget import AndorLiveViewWidget
from rot.rot_wrapper import MotionController, RotationStage, list_kinesis_serials
from rot.gui.workers import HomeThread, MoveThread

from measurements.config import (
    DEFAULT_ACQ_NUMBER,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_GRATING,
    DEFAULT_LINECUT_ROW,
    DEFAULT_LINECUT_WIDTH,
    DEFAULT_OUTPUT_AMP,
    DEFAULT_POWER_CALIB_PATH,
    DEFAULT_PREAMP_GAIN,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_READOUT_RATE,
    DEFAULT_SLIT_ID,
    DEFAULT_SLIT_UM,
    DEFAULT_SPEC_INDEX,
    DEFAULT_STAGE_ACCEL,
    DEFAULT_STAGE_SCALE,
    DEFAULT_STAGE_B_SERIAL,
    DEFAULT_STAGE_SERIAL,
    POLL_MS,
    DATA_DIR,
)
from measurements.dual_wheel_power_calibration import (
    DualWheelPowerEntry,
    DualWheelPowerData,
    is_dual_wheel_calibration,
    load_dual_wheel_calibration,
)
from measurements.power_calibration import (
    PowerCalibEntry,
    entry_priority,
    load_power_calibration,
    power_key,
)
from measurements.workers import DualWheelSweepThread, SweepThread

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None


class PLPowerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)
        self.stage: Optional[RotationStage] = None
        self.stage_b: Optional[RotationStage] = None

        self._live_thread: Optional[LiveAcqThread] = None
        self._accum_thread: Optional[SingleAcqThread] = None
        self._sweep_thread: Optional[SweepThread] = None
        self._home_thread: Optional[HomeThread] = None
        self._move_thread: Optional[MoveThread] = None

        self._stage_controller: Optional[MotionController] = None
        self._stage_busy = False
        self._busy = False
        self._live_active = False

        self._live_view = None
        self._linecut_row = int(DEFAULT_LINECUT_ROW)
        self._linecut_width = int(DEFAULT_LINECUT_WIDTH)
        self._wl_axis = None
        self._xaxis_values = None
        self._cursor_rc = None

        self._calib_data = None
        self._dual_calib_data: Optional[DualWheelPowerData] = None
        self._series_entries: Dict[str, List[PowerCalibEntry]] = {}
        self._dual_entries: List[DualWheelPowerEntry] = []
        self._power_display: Dict[float, dict] = {}
        self._power_list_keys: List[float] = []
        self._power_images: Dict[float, dict] = {}
        self._taken_powers = set()
        self._taken_entries = set()
        self._total_entries = 0
        self._session_saved = False
        self._file_counters: Dict[str, int] = {}
        self._save_dir = None
        self._calib_unit = "nW"
        self._calib_scale = 1e-9
        self._calib_mode = "single"

        self._crop_top = int(DEFAULT_CROP_TOP)
        self._crop_bottom = int(DEFAULT_CROP_BOTTOM)
        self._crop_left = int(DEFAULT_CROP_LEFT)
        self._crop_right = int(DEFAULT_CROP_RIGHT)

        self._readout_rate_label = str(DEFAULT_READOUT_RATE)
        self._preamp_gain_label = str(DEFAULT_PREAMP_GAIN)
        self._output_amp_label = str(DEFAULT_OUTPUT_AMP)

        self._build_ui()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.start(int(POLL_MS))

    def _threads_running(self) -> bool:
        live = self._live_thread is not None and self._live_thread.isRunning()
        accum = self._accum_thread is not None and self._accum_thread.isRunning()
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return live or accum or sweep

    def _is_busy(self) -> bool:
        return self._busy or self._stage_busy or self._threads_running()

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self._build_dashboard(vbox)
        self._build_session_controls(vbox)
        self._build_plots(vbox)

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

        # Andor line
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

        self.preampCombo = QtWidgets.QComboBox()
        self.preampCombo.setEditable(False)
        self.preampCombo.addItem(str(DEFAULT_PREAMP_GAIN))
        self.preampCombo.setFixedWidth(70)
        self.preampInfoLbl = QtWidgets.QLabel("Gain: --")
        self.preampInfoLbl.setFont(mono)

        line1 = QtWidgets.QHBoxLayout()
        line1.addWidget(self.camDot)
        line1.addWidget(QtWidgets.QLabel("Andor:"))
        line1.addWidget(self.camStatusLbl, 1)
        line1.addStretch()
        line1.addWidget(QtWidgets.QLabel("Exp (ms):"))
        line1.addWidget(self.exposureSpin)
        line1.addWidget(QtWidgets.QLabel("Acq N:"))
        line1.addWidget(self.accumSpin)
        line1.addWidget(QtWidgets.QLabel("Preamp:"))
        line1.addWidget(self.preampCombo)
        line1.addWidget(self.preampInfoLbl)

        grid.addLayout(line1, 0, 0, 1, 1)

        # Spectrograph line
        self.specDot = self._make_dot()
        self.specStatusLbl = QtWidgets.QLabel("Spectrograph: disconnected")
        self.specStatusLbl.setFont(mono)

        self.gratingSpin = QtWidgets.QSpinBox()
        self.gratingSpin.setRange(0, 10)
        self.gratingSpin.setValue(int(DEFAULT_GRATING))
        self.gratingSpin.setFixedWidth(60)

        self.centerSpin = QtWidgets.QDoubleSpinBox()
        self.centerSpin.setDecimals(3)
        self.centerSpin.setRange(0.0, 20000.0)
        self.centerSpin.setValue(float(DEFAULT_CENTER_WL_NM))
        self.centerSpin.setFixedWidth(90)

        self.slitSpin = QtWidgets.QDoubleSpinBox()
        self.slitSpin.setDecimals(2)
        self.slitSpin.setRange(0.0, 5000.0)
        self.slitSpin.setValue(float(DEFAULT_SLIT_UM))
        self.slitSpin.setFixedWidth(80)

        self.applySpecBtn = QtWidgets.QPushButton("Apply Spec")
        self.applySpecBtn.setFixedWidth(120)

        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.specDot)
        line2.addWidget(QtWidgets.QLabel("Spectrograph:"))
        line2.addWidget(self.specStatusLbl, 1)
        line2.addStretch()
        line2.addWidget(QtWidgets.QLabel("Grating:"))
        line2.addWidget(self.gratingSpin)
        line2.addWidget(QtWidgets.QLabel("Center (nm):"))
        line2.addWidget(self.centerSpin)
        line2.addWidget(QtWidgets.QLabel("Slit (um):"))
        line2.addWidget(self.slitSpin)
        line2.addWidget(self.applySpecBtn)

        grid.addLayout(line2, 1, 0, 1, 1)

        # Stage A line
        self.stageDot = self._make_dot()
        self.stageStatusLbl = QtWidgets.QLabel("Stage A: disconnected")
        self.stageStatusLbl.setFont(mono)

        self.stageSerialCombo = QtWidgets.QComboBox()
        self.stageSerialCombo.setEditable(True)
        self.stageSerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_SERIAL:
            self.stageSerialCombo.addItem(DEFAULT_STAGE_SERIAL)

        self.stageHomeBtn = QtWidgets.QPushButton("Home A")
        self.stageHomeBtn.setFixedWidth(90)
        self.stageMoveBtn = QtWidgets.QPushButton("Move A")
        self.stageMoveBtn.setFixedWidth(90)
        self.stageDetectBtn = QtWidgets.QPushButton("Search")
        self.stageDetectBtn.setFixedWidth(80)

        self.stageTargetSpin = QtWidgets.QDoubleSpinBox()
        self.stageTargetSpin.setDecimals(3)
        self.stageTargetSpin.setRange(0.0, 360.0)
        self.stageTargetSpin.setValue(0.0)
        self.stageTargetSpin.setFixedWidth(90)

        line3 = QtWidgets.QHBoxLayout()
        line3.addWidget(self.stageDot)
        line3.addWidget(QtWidgets.QLabel("Stage A:"))
        line3.addWidget(self.stageStatusLbl, 1)
        line3.addStretch()
        line3.addWidget(QtWidgets.QLabel("Serial:"))
        line3.addWidget(self.stageSerialCombo)
        line3.addWidget(self.stageDetectBtn)
        line3.addWidget(self.stageHomeBtn)
        line3.addWidget(QtWidgets.QLabel("Target (deg):"))
        line3.addWidget(self.stageTargetSpin)
        line3.addWidget(self.stageMoveBtn)

        grid.addLayout(line3, 2, 0, 1, 1)

        # Stage B line
        self.stageBDot = self._make_dot()
        self.stageBStatusLbl = QtWidgets.QLabel("Stage B: disconnected")
        self.stageBStatusLbl.setFont(mono)

        self.stageBSerialCombo = QtWidgets.QComboBox()
        self.stageBSerialCombo.setEditable(True)
        self.stageBSerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_B_SERIAL:
            self.stageBSerialCombo.addItem(DEFAULT_STAGE_B_SERIAL)

        self.stageBHomeBtn = QtWidgets.QPushButton("Home B")
        self.stageBHomeBtn.setFixedWidth(90)
        self.stageBMoveBtn = QtWidgets.QPushButton("Move B")
        self.stageBMoveBtn.setFixedWidth(90)
        self.stageBDetectBtn = QtWidgets.QPushButton("Search")
        self.stageBDetectBtn.setFixedWidth(80)

        self.stageBTargetSpin = QtWidgets.QDoubleSpinBox()
        self.stageBTargetSpin.setDecimals(3)
        self.stageBTargetSpin.setRange(0.0, 360.0)
        self.stageBTargetSpin.setValue(0.0)
        self.stageBTargetSpin.setFixedWidth(90)

        line4 = QtWidgets.QHBoxLayout()
        line4.addWidget(self.stageBDot)
        line4.addWidget(QtWidgets.QLabel("Stage B:"))
        line4.addWidget(self.stageBStatusLbl, 1)
        line4.addStretch()
        line4.addWidget(QtWidgets.QLabel("Serial:"))
        line4.addWidget(self.stageBSerialCombo)
        line4.addWidget(self.stageBDetectBtn)
        line4.addWidget(self.stageBHomeBtn)
        line4.addWidget(QtWidgets.QLabel("Target (deg):"))
        line4.addWidget(self.stageBTargetSpin)
        line4.addWidget(self.stageBMoveBtn)

        grid.addLayout(line4, 3, 0, 1, 1)

        # Action line
        self.initBtn = QtWidgets.QPushButton("Initialize")
        self.initBtn.setFixedWidth(120)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setFixedWidth(120)
        self.takeCurrentBtn = QtWidgets.QPushButton("Take Current Image")
        self.takeCurrentBtn.setFixedWidth(160)
        self.stopLiveBtn = QtWidgets.QPushButton("Stop Live")
        self.stopLiveBtn.setFixedWidth(120)
        self.sweepBtn = QtWidgets.QPushButton("Take All Powers")
        self.sweepBtn.setFixedWidth(140)
        self.abortBtn = QtWidgets.QPushButton("Abort")
        self.abortBtn.setFixedWidth(100)

        line5 = QtWidgets.QHBoxLayout()
        line5.addStretch()
        line5.addWidget(self.initBtn)
        line5.addWidget(self.disconnectBtn)
        line5.addSpacing(10)
        line5.addWidget(self.takeCurrentBtn)
        line5.addWidget(self.stopLiveBtn)
        line5.addWidget(self.sweepBtn)
        line5.addWidget(self.abortBtn)

        grid.addLayout(line5, 4, 0, 1, 1)

        parent_layout.addWidget(box)

    def _build_session_controls(self, parent_layout):
        box = QtWidgets.QGroupBox("Session / Calibration")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.saveNameEdit = QtWidgets.QLineEdit(self._default_save_name())
        self.saveNameEdit.setPlaceholderText("subfolder name under measurements/")

        self.calibPathEdit = QtWidgets.QLineEdit(DEFAULT_POWER_CALIB_PATH)
        self.calibBrowseBtn = QtWidgets.QPushButton("Browse")
        self.calibBrowseBtn.setFixedWidth(90)
        self.calibLoadBtn = QtWidgets.QPushButton("Load/Reload")
        self.calibLoadBtn.setFixedWidth(110)

        self.seriesCombo = QtWidgets.QComboBox()
        self.seriesInfoLbl = QtWidgets.QLabel("Series: --")
        self.calibModeLbl = QtWidgets.QLabel("Mode: --")

        self.cropTopSpin = QtWidgets.QSpinBox()
        self.cropBottomSpin = QtWidgets.QSpinBox()
        self.cropLeftSpin = QtWidgets.QSpinBox()
        self.cropRightSpin = QtWidgets.QSpinBox()
        for spin in (self.cropTopSpin, self.cropBottomSpin, self.cropLeftSpin, self.cropRightSpin):
            spin.setRange(0, 10000)
            spin.setFixedWidth(70)
        self.cropTopSpin.setValue(self._crop_top)
        self.cropBottomSpin.setValue(self._crop_bottom)
        self.cropLeftSpin.setValue(self._crop_left)
        self.cropRightSpin.setValue(self._crop_right)
        self.cropApplyBtn = QtWidgets.QPushButton("Apply Crop")
        self.cropApplyBtn.setFixedWidth(110)

        grid.addWidget(QtWidgets.QLabel("Save subfolder:"), 0, 0)
        grid.addWidget(self.saveNameEdit, 0, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Power calib file:"), 1, 0)
        grid.addWidget(self.calibPathEdit, 1, 1)
        grid.addWidget(self.calibBrowseBtn, 1, 2)
        grid.addWidget(self.calibLoadBtn, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Calibration mode:"), 2, 0)
        grid.addWidget(self.calibModeLbl, 2, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Current series:"), 3, 0)
        grid.addWidget(self.seriesCombo, 3, 1)
        grid.addWidget(self.seriesInfoLbl, 3, 2, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 4, 0)
        grid.addWidget(self.cropTopSpin, 4, 1)
        grid.addWidget(self.cropBottomSpin, 4, 2)
        grid.addWidget(self.cropLeftSpin, 4, 3)
        grid.addWidget(self.cropRightSpin, 4, 4)
        grid.addWidget(self.cropApplyBtn, 4, 5)

        parent_layout.addWidget(box)

    def _build_plots(self, parent_layout):
        box = QtWidgets.QGroupBox("PL Image + Linecut")
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # PL image (Andor live view widget)
        self.liveView = AndorLiveViewWidget(self.cam, title="PL Image")
        self._apply_crop_to_liveview()
        self.liveView.set_linecut_row(int(DEFAULT_LINECUT_ROW))
        self.liveView.linecut_changed.connect(lambda _: self._refresh_linecut_map())

        # Linecut vs power
        self.fig_map = Figure(figsize=(4, 4.5), dpi=100)
        self.canvas_map = FigureCanvas(self.fig_map)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("Linecut vs Power")
        self.ax_map.set_xlabel("Power")
        self.ax_map.set_ylabel("Wavelength (nm)")
        self.map_artist = self.ax_map.imshow(
            np.zeros((10, 10)),
            cmap="viridis",
            origin="lower",
            aspect="auto",
        )

        # Power list
        self.powerTable = QtWidgets.QTableWidget(0, 2)
        self.powerTable.setHorizontalHeaderLabels(["Power", "Taken"])
        self.powerTable.horizontalHeader().setStretchLastSection(True)
        self.powerTable.verticalHeader().setVisible(False)
        self.powerTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.powerTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.powerTable.setMinimumWidth(150)

        layout.addWidget(self.liveView, 4)
        layout.addWidget(self.canvas_map, 2)
        layout.addWidget(self.powerTable, 1)

        parent_layout.addWidget(box, 1)

    def _wire_signals(self):
        self.initBtn.clicked.connect(self.on_initialize)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.takeCurrentBtn.clicked.connect(self.on_take_current)
        self.sweepBtn.clicked.connect(self.on_sweep)
        self.abortBtn.clicked.connect(self.on_abort)
        self.calibBrowseBtn.clicked.connect(self.on_browse_calib)
        self.calibLoadBtn.clicked.connect(self.on_load_calib)
        self.applySpecBtn.clicked.connect(self.on_apply_spec)
        self.stageHomeBtn.clicked.connect(self.on_stage_home)
        self.stageMoveBtn.clicked.connect(self.on_stage_move)
        self.stageDetectBtn.clicked.connect(self.on_stage_detect)
        self.stageBHomeBtn.clicked.connect(self.on_stage_home_b)
        self.stageBMoveBtn.clicked.connect(self.on_stage_move_b)
        self.stageBDetectBtn.clicked.connect(self.on_stage_detect)
        self.stopLiveBtn.clicked.connect(self.on_stop_live)
        self.seriesCombo.currentIndexChanged.connect(self._update_series_info)
        self.cropApplyBtn.clicked.connect(self.on_apply_crop)
        self.preampCombo.currentIndexChanged.connect(self.on_preamp_change)

    # -----------------
    # Helpers
    # -----------------
    def _make_dot(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setFixedSize(10, 10)
        lbl.setStyleSheet("border-radius:5px; background-color:#c62828;")
        return lbl

    def _apply_crop_to_liveview(self) -> None:
        if self.liveView is None:
            return
        raw_top = int(self._crop_bottom)
        raw_bottom = int(self._crop_top)
        self.liveView.set_crop(raw_top, raw_bottom, int(self._crop_left), int(self._crop_right))

    def _set_dot(self, lbl: QtWidgets.QLabel, ok: bool) -> None:
        color = "#2e7d32" if ok else "#c62828"
        lbl.setStyleSheet(f"border-radius:5px; background-color:{color};")

    def _default_save_name(self) -> str:
        return time.strftime("pl_%Y%m%d_%H%M%S")

    def _resolve_save_dir(self) -> str:
        name = self.saveNameEdit.text().strip()
        if not name:
            name = self._default_save_name()
            self.saveNameEdit.setText(name)
        base = os.path.join(DATA_DIR, "measurements", name)
        os.makedirs(base, exist_ok=True)
        self._save_dir = base
        return base

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

    def _refresh_amp_choices(self) -> None:
        if self.cam is None or self.cam.cam is None:
            return
        current = self.preampCombo.currentText().strip()
        gains = []
        try:
            _, _, gains = self.cam.get_amp_mode_choices()
        except Exception:
            gains = []
        if not gains:
            return
        self.preampCombo.blockSignals(True)
        self.preampCombo.clear()
        self.preampCombo.addItems(gains)
        if str(DEFAULT_PREAMP_GAIN) in gains:
            self.preampCombo.setCurrentText(str(DEFAULT_PREAMP_GAIN))
        elif current and current in gains:
            self.preampCombo.setCurrentText(current)
        else:
            self.preampCombo.setCurrentIndex(0)
        self.preampCombo.blockSignals(False)

    def _apply_amp_settings(self) -> None:
        if self.cam is None or self.cam.cam is None:
            return
        req_gain = self.preampCombo.currentText().strip()
        req_rate = self._readout_rate_label
        req_amp = self._output_amp_label

        def _gain_value(label: str):
            m = re.search(r"(\\d+(?:\\.\\d+)?)", str(label))
            if m:
                return float(m.group(1))
            return None

        want_gain = _gain_value(req_gain)
        ok = True
        try:
            ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
        except Exception as exc:
            ok = False
            self.statusBar().showMessage(f"Preamp set failed: {exc}")

        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None

        if want_gain is not None:
            cur_gain = None
            if info is not None:
                cur_gain = info.get("preamp_gain")
                if cur_gain is None:
                    cur_gain = _gain_value(info.get("gain_label"))
            if cur_gain is None or abs(float(cur_gain) - float(want_gain)) > 0.05:
                try:
                    ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
                except Exception:
                    ok = False
                try:
                    info = self.cam.get_amp_mode_info()
                except Exception:
                    info = None

        if info:
            self._readout_rate_label = info.get("rate_label") or self._readout_rate_label
            self._preamp_gain_label = info.get("gain_label") or self._preamp_gain_label
            self._output_amp_label = info.get("amp_label") or self._output_amp_label
            gain_label = info.get("gain_label")
            if gain_label:
                self.preampCombo.blockSignals(True)
                self.preampCombo.setCurrentText(str(gain_label))
                self.preampCombo.blockSignals(False)

        self._update_preamp_label()
        if not ok:
            self.statusBar().showMessage("Preamp gain not applied")

    def _update_preamp_label(self) -> None:
        if self.cam is None or self.cam.cam is None:
            self.preampInfoLbl.setText("Gain: --")
            return
        label = self._preamp_gain_label or self.preampCombo.currentText().strip()
        self.preampInfoLbl.setText(f"Gain: {label}")

    def _apply_spec_settings(self) -> None:
        if self.cam.spec is None:
            return
        try:
            self.cam.spec_set_grating(int(self.gratingSpin.value()))
        except Exception:
            pass
        try:
            self.cam.set_center_wavelength_nm(float(self.centerSpin.value()))
        except Exception:
            pass
        try:
            self.cam.spec_set_slit_width_um(DEFAULT_SLIT_ID, float(self.slitSpin.value()))
        except Exception:
            pass
        if self.liveView is not None:
            center = float(self.centerSpin.value())
            if abs(center) <= 1e-9:
                self.liveView.set_wavelength_axis_enabled(False)
            else:
                self.liveView.set_wavelength_axis_enabled(True)

    def _prepare_live_mode(self) -> None:
        try:
            self.cam.set_frame_api("Stream+Buffer")
        except Exception:
            pass
        try:
            self.cam.set_acquisition_mode("Run till abort")
        except Exception:
            pass
        try:
            self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            self.cam.set_exposure_ms(float(self.exposureSpin.value()))
        except Exception:
            pass

    def _prepare_single_mode(self) -> None:
        try:
            if hasattr(self.cam, "stop_stream"):
                self.cam.stop_stream()
        except Exception:
            pass
        try:
            self.cam.set_frame_api("Snap")
        except Exception:
            pass
        try:
            self.cam.set_acquisition_mode("Single")
        except Exception:
            pass
        try:
            self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            self.cam.set_exposure_ms(float(self.exposureSpin.value()))
        except Exception:
            pass

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._is_busy():
            self._poll_timer.stop()
        else:
            if not self._poll_timer.isActive():
                self._poll_timer.start(int(POLL_MS))
        self._sync_controls()

    def _sync_controls(self) -> None:
        cam_ok = self.cam.cam is not None
        stage_a_ok = self.stage is not None
        stage_b_ok = self.stage_b is not None
        spec_ok = self.cam.spec is not None
        busy = self._is_busy()

        need_dual = self._calib_mode == "dual"
        calib_ok = self._dual_calib_data is not None if need_dual else self._calib_data is not None
        stages_ok = stage_a_ok and (stage_b_ok if need_dual else True)

        self.initBtn.setEnabled(not busy)
        self.disconnectBtn.setEnabled(not busy)
        self.takeCurrentBtn.setEnabled(cam_ok and not busy)
        self.stopLiveBtn.setEnabled(self._threads_running() and self._live_active)
        self.sweepBtn.setEnabled(cam_ok and stages_ok and calib_ok and not busy)
        self.abortBtn.setEnabled(busy)
        self.applySpecBtn.setEnabled(spec_ok and not busy)
        self.stageHomeBtn.setEnabled(stage_a_ok and not busy)
        self.stageMoveBtn.setEnabled(stage_a_ok and not busy)
        self.stageBHomeBtn.setEnabled(stage_b_ok and not busy)
        self.stageBMoveBtn.setEnabled(stage_b_ok and not busy)
        self.preampCombo.setEnabled(cam_ok and not busy)
        self.seriesCombo.setEnabled(not need_dual and calib_ok and not busy)

    def _entry_key(self, entry) -> Tuple:
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            return (
                "dual",
                power_key(entry.power_w, scale=self._calib_scale),
                float(entry.a_deg),
                float(entry.b_deg),
                int(getattr(entry, "index", 0)),
            )
        return (entry.series, power_key(entry.power_w, scale=self._calib_scale), entry.position_deg)

    def _entry_rank(self, entry):
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            return int(getattr(entry, "index", 0))
        return entry_priority(entry)

    def _format_power(self, power_w: float) -> str:
        val = power_w / float(self._calib_scale or 1e-9)
        return f"{val:.6g} {self._calib_unit}"

    def _format_power_key(self, key: float) -> str:
        return f"{key:.6g} {self._calib_unit}"

    def _safe_series_label(self, series: str) -> str:
        clean = series.strip().lower().replace(" ", "_").replace("#", "")
        clean = "".join(ch for ch in clean if ch.isalnum() or ch in ("_", "+"))
        return clean or "series"

    def _next_filename(self, base: str, ext: str = ".asc") -> str:
        idx = self._file_counters.get(base, 0)
        while True:
            suffix = "" if idx == 0 else f"_{idx}"
            name = f"{base}{suffix}{ext}"
            path = os.path.join(self._save_dir or ".", name)
            if not os.path.exists(path):
                self._file_counters[base] = idx + 1
                return path
            idx += 1

    def _build_metadata_base(self) -> dict:
        return {
            "exposure_ms": float(self.exposureSpin.value()),
            "accum_n": int(self.accumSpin.value()),
            "readout_rate": self._readout_rate_label,
            "preamp_gain": self._preamp_gain_label,
            "output_amp": self._output_amp_label,
            "grating": int(self.gratingSpin.value()),
            "center_wl_nm": float(self.centerSpin.value()),
            "slit_id": DEFAULT_SLIT_ID,
            "slit_um": float(self.slitSpin.value()),
            "stage_serial": self.stage.serial if self.stage else "",
            "stage_serial_a": self.stage.serial if self.stage else "",
            "stage_serial_b": self.stage_b.serial if self.stage_b else "",
            "calib_mode": self._calib_mode,
            "power_calib_file": self.calibPathEdit.text().strip(),
            "crop_top": self._crop_top,
            "crop_bottom": self._crop_bottom,
            "crop_left": self._crop_left,
            "crop_right": self._crop_right,
            "linecut_row": int(self.liveView.linecut_row() or 0) if self.liveView is not None else 0,
            "linecut_width": int(self.liveView.linecut_width()) if self.liveView is not None else 1,
        }

    def _reset_sweep_data(self) -> None:
        self._power_images.clear()
        self._taken_powers.clear()
        self._taken_entries.clear()
        self._session_saved = False
        self._file_counters.clear()
        self._refresh_power_table()
        self._refresh_linecut_map()

    # -----------------
    # Status polling
    # -----------------
    def _poll_status(self) -> None:
        cam_ok = self.cam.cam is not None
        spec_ok = self.cam.spec is not None
        stage_ok = self.stage is not None
        stage_b_ok = self.stage_b is not None

        self._set_dot(self.camDot, cam_ok)
        self._set_dot(self.specDot, spec_ok)
        self._set_dot(self.stageDot, stage_ok)
        self._set_dot(self.stageBDot, stage_b_ok)

        if cam_ok:
            temp = self.cam.get_temperature_c()
            temp_s = f"{temp:.1f} C" if temp is not None else "--"
            exp_ms = float(self.exposureSpin.value())
            acc = int(self.accumSpin.value())
            ro = self._readout_rate_label or "--"
            self.camStatusLbl.setText(f"T={temp_s} | Exp {exp_ms:g} ms x{acc} | RO {ro}")
            self._update_preamp_label()
        else:
            self.camStatusLbl.setText("Disconnected")
            self._update_preamp_label()

        if spec_ok:
            gr = self.cam.spec_get_grating()
            cw = self.cam.get_center_wavelength_nm()
            slit = self.cam.spec_get_slit_width_um(DEFAULT_SLIT_ID)
            gr_s = str(gr) if gr is not None else "--"
            cw_s = f"{cw:.3f}" if cw is not None else "--"
            slit_s = f"{slit:.1f}" if slit is not None else "--"
            self.specStatusLbl.setText(f"Gr {gr_s} | Center {cw_s} nm | Slit {slit_s} um")
        else:
            self.specStatusLbl.setText("Disconnected")

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

        if stage_b_ok:
            try:
                pos_b = float(self.stage_b.get_position())
            except Exception:
                pos_b = None
            try:
                homed_b = self.stage_b.is_homed()
            except Exception:
                homed_b = None
            try:
                moving_b = bool(self.stage_b.is_moving())
            except Exception:
                moving_b = False
            pos_b_s = f"{pos_b:.3f} deg" if pos_b is not None else "--"
            home_b_s = "yes" if homed_b else ("no" if homed_b is False else "?")
            status_b = "Moving..." if (self._stage_busy or moving_b) else "Ready"
            self.stageBStatusLbl.setText(f"SN {self.stage_b.serial} | Homed {home_b_s} | {status_b} | Angle {pos_b_s}")
        else:
            self.stageBStatusLbl.setText("Disconnected")

    # -----------------
    # Calibration
    # -----------------
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
            if is_dual_wheel_calibration(path):
                data = load_dual_wheel_calibration(path)
                self._load_dual_calibration(data)
            else:
                data = load_power_calibration(path)
                self._load_single_calibration(data)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        self.statusBar().showMessage(f"Loaded calibration: {os.path.basename(path)}")
        self._sync_controls()

    def _load_single_calibration(self, data):
        self._calib_mode = "single"
        self._calib_data = data
        self._dual_calib_data = None
        self._dual_entries = []
        self._calib_unit = data.unit
        self._calib_scale = data.scale
        self._series_entries.clear()
        for entry in data.entries:
            self._series_entries.setdefault(entry.series, []).append(entry)
        for series in self._series_entries:
            self._series_entries[series].sort(key=lambda e: (e.power_w, e.position_deg))

        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItems(data.series_names)
        if "base" in data.series_names:
            self.seriesCombo.setCurrentText("base")
        self.seriesCombo.blockSignals(False)
        self.calibModeLbl.setText("Mode: single wheel + ND filters")

        self._build_power_display()
        self._total_entries = len(data.entries)
        self._reset_sweep_data()
        self._update_series_info()

    def _load_dual_calibration(self, data: DualWheelPowerData):
        self._calib_mode = "dual"
        self._dual_calib_data = data
        self._calib_data = None
        self._series_entries.clear()
        self._dual_entries = list(data.entries)
        self._calib_unit = data.unit
        self._calib_scale = data.scale

        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItem("dual_wheel")
        self.seriesCombo.setCurrentIndex(0)
        self.seriesCombo.blockSignals(False)
        self.calibModeLbl.setText("Mode: dual wheel (file order sweep)")

        self._build_power_display()
        self._total_entries = len(data.entries)
        self._reset_sweep_data()
        self._update_series_info()

    def _build_power_display(self):
        self._power_display.clear()
        if self._calib_mode == "dual":
            if not self._dual_entries:
                return
            for entry in self._dual_entries:
                key = power_key(entry.power_w, scale=self._calib_scale)
                rank = int(getattr(entry, "index", 0))
                cur = self._power_display.get(key)
                if cur is None or rank < cur["rank"]:
                    self._power_display[key] = {"entry": entry, "rank": rank}
        else:
            if self._calib_data is None:
                return
            for entry in self._calib_data.entries:
                key = power_key(entry.power_w, scale=self._calib_scale)
                rank = entry_priority(entry)
                cur = self._power_display.get(key)
                if cur is None or rank < cur["rank"]:
                    self._power_display[key] = {"entry": entry, "rank": rank}

        self._power_list_keys = sorted(self._power_display.keys())
        self._refresh_power_table()

    def _update_series_info(self):
        if self._calib_mode == "dual":
            n = len(self._dual_entries)
            self.seriesInfoLbl.setText(f"dual_wheel ({n} entries)")
            return
        series = self.seriesCombo.currentText().strip()
        n = len(self._series_entries.get(series, []))
        self.seriesInfoLbl.setText(f"{series} ({n} powers)")

    def _refresh_power_table(self):
        unit = self._calib_unit
        self.powerTable.setRowCount(len(self._power_list_keys))
        self.powerTable.setHorizontalHeaderLabels([f"Power ({unit})", "Taken"])
        for row, key in enumerate(self._power_list_keys):
            power_item = QtWidgets.QTableWidgetItem(f"{key:.6g}")
            taken_item = QtWidgets.QTableWidgetItem("")
            taken_item.setFlags(QtCore.Qt.ItemIsEnabled)
            taken_item.setCheckState(QtCore.Qt.Checked if key in self._taken_powers else QtCore.Qt.Unchecked)
            self.powerTable.setItem(row, 0, power_item)
            self.powerTable.setItem(row, 1, taken_item)
        self.powerTable.resizeColumnsToContents()

    def _mark_power_taken(self, key: float) -> None:
        self._taken_powers.add(key)
        for row, k in enumerate(self._power_list_keys):
            if k == key:
                item = self.powerTable.item(row, 1)
                if item is not None:
                    item.setCheckState(QtCore.Qt.Checked)
                break

    def on_apply_spec(self):
        self._apply_spec_settings()
        self.statusBar().showMessage("Spectrograph settings applied")

    def on_preamp_change(self):
        if self.cam is None or self.cam.cam is None:
            self._preamp_gain_label = self.preampCombo.currentText().strip()
            self._update_preamp_label()
            return
        self._apply_amp_settings()

    def on_apply_crop(self):
        self._crop_top = int(self.cropTopSpin.value())
        self._crop_bottom = int(self.cropBottomSpin.value())
        self._crop_left = int(self.cropLeftSpin.value())
        self._crop_right = int(self.cropRightSpin.value())
        self._apply_crop_to_liveview()
        self._refresh_linecut_map()
        self.statusBar().showMessage("Crop updated")

    def on_stage_detect(self):
        try:
            serials = list_kinesis_serials()
        except Exception as exc:
            self.statusBar().showMessage(f"Detect failed: {exc}")
            return
        if not serials:
            self.statusBar().showMessage("No stages detected")
            return
        current_a = self.stageSerialCombo.currentText().strip()
        current_b = self.stageBSerialCombo.currentText().strip()
        self.stageSerialCombo.blockSignals(True)
        self.stageBSerialCombo.blockSignals(True)
        self.stageSerialCombo.clear()
        self.stageBSerialCombo.clear()
        self.stageSerialCombo.addItems(serials)
        self.stageBSerialCombo.addItems(serials)
        if current_a:
            self.stageSerialCombo.setCurrentText(current_a)
        if current_b:
            self.stageBSerialCombo.setCurrentText(current_b)
        self.stageSerialCombo.blockSignals(False)
        self.stageBSerialCombo.blockSignals(False)
        self.statusBar().showMessage(f"Detected {len(serials)} stage(s)")

    # -----------------
    # Initialize / Disconnect
    # -----------------
    def on_initialize(self):
        if self._busy:
            return
        try:
            self.cam.connect()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Andor connect", str(exc))
            return

        try:
            self.cam.connect_spectrograph(int(DEFAULT_SPEC_INDEX))
        except Exception:
            pass

        self._apply_camera_defaults()
        self._apply_spec_settings()

        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None
        if info:
            self._readout_rate_label = info.get("rate_label") or self._readout_rate_label
            self._preamp_gain_label = info.get("gain_label") or self._preamp_gain_label
            self._output_amp_label = info.get("amp_label") or self._output_amp_label

        self._refresh_amp_choices()
        self._apply_amp_settings()

        serial = self.stageSerialCombo.currentText().strip()
        if serial:
            try:
                self.stage = RotationStage(serial, scale=DEFAULT_STAGE_SCALE)
                self.stage.open()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Stage connect", str(exc))
                self.stage = None

        serial_b = self.stageBSerialCombo.currentText().strip()
        if serial_b and serial_b != serial:
            try:
                self.stage_b = RotationStage(serial_b, scale=DEFAULT_STAGE_SCALE)
                self.stage_b.open()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Stage B connect", str(exc))
                self.stage_b = None

        if self.stage is not None:
            self.on_stage_home()
        if self.stage_b is not None:
            self.on_stage_home_b()

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
            if self.stage_b:
                self.stage_b.close()
        except Exception:
            pass
        self.stage_b = None
        self._live_active = False
        try:
            self.cam.disconnect_spectrograph()
        except Exception:
            pass
        try:
            self.cam.disconnect()
        except Exception:
            pass
        self.statusBar().showMessage("Disconnected")
        self._sync_controls()

    # -----------------
    # Stage control
    # -----------------
    def on_stage_home(self):
        if self.stage is None or self._stage_busy:
            return
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        self._home_thread = HomeThread(self.stage, self._stage_controller)
        self._home_thread.status.connect(self.statusBar().showMessage)
        self._home_thread.done.connect(lambda s: self._on_stage_done(s, "Stage A"))
        self._home_thread.start()

    def on_stage_move(self):
        if self.stage is None or self._stage_busy:
            return
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        target = float(self.stageTargetSpin.value())
        self._move_thread = MoveThread(
            self.stage,
            target,
            DEFAULT_RAMP_STEP_DEG,
            DEFAULT_STAGE_ACCEL,
            self._stage_controller,
        )
        self._move_thread.status.connect(self.statusBar().showMessage)
        self._move_thread.done.connect(lambda s: self._on_stage_done(s, "Stage A"))
        self._move_thread.start()

    def on_stage_home_b(self):
        if self.stage_b is None or self._stage_busy:
            return
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        self._home_thread = HomeThread(self.stage_b, self._stage_controller)
        self._home_thread.status.connect(self.statusBar().showMessage)
        self._home_thread.done.connect(lambda s: self._on_stage_done(s, "Stage B"))
        self._home_thread.start()

    def on_stage_move_b(self):
        if self.stage_b is None or self._stage_busy:
            return
        self._stage_busy = True
        self._sync_controls()
        self._stage_controller = MotionController()
        target = float(self.stageBTargetSpin.value())
        self._move_thread = MoveThread(
            self.stage_b,
            target,
            DEFAULT_RAMP_STEP_DEG,
            DEFAULT_STAGE_ACCEL,
            self._stage_controller,
        )
        self._move_thread.status.connect(self.statusBar().showMessage)
        self._move_thread.done.connect(lambda s: self._on_stage_done(s, "Stage B"))
        self._move_thread.start()

    def _on_stage_done(self, status: str, tag: str = "Stage"):
        self._stage_busy = False
        if status == "ok":
            self.statusBar().showMessage(f"{tag} ready")
        elif status == "aborted":
            self.statusBar().showMessage(f"{tag} motion aborted")
        else:
            self.statusBar().showMessage(str(status))
        self._sync_controls()

    # -----------------
    # Acquisition
    # -----------------
    def on_take_current(self):
        if self._busy:
            return
        if self.cam.cam is None:
            QtWidgets.QMessageBox.warning(self, "Camera", "Camera not connected.")
            return

        accum_n = int(self.accumSpin.value())
        if accum_n <= 1:
            self._prepare_live_mode()
            self._live_thread = LiveAcqThread(self.cam)
            self._live_thread.frame_ready.connect(self.on_live_frame)
            self._live_thread.status.connect(self.statusBar().showMessage)
            self._live_thread.finished.connect(self._on_live_done)
            self._live_thread.start()
            self._live_active = True
            self._set_busy(True)
            self.statusBar().showMessage("Live started")
        else:
            self._prepare_single_mode()
            self._accum_thread = SingleAcqThread(self.cam, accum_n)
            self._accum_thread.frame_ready.connect(self.on_accum_frame)
            self._accum_thread.status.connect(self.statusBar().showMessage)
            self._accum_thread.finished.connect(self.on_accum_done)
            self._accum_thread.start()
            self._set_busy(True)
            self.statusBar().showMessage("Accumulation started")

    def on_sweep(self):
        if self._busy:
            return
        if self._calib_mode == "dual":
            if self._dual_calib_data is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Load a dual-wheel calibration file first.")
                return
            if self.cam.cam is None or self.stage is None or self.stage_b is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Camera and both stages must be connected.")
                return
            entries = list(self._dual_entries)
            if not entries:
                QtWidgets.QMessageBox.warning(self, "Sweep", "No entries in dual-wheel calibration file.")
                return

            self._prepare_single_mode()
            self._resolve_save_dir()

            plan = []
            for entry in entries:
                power_val = power_key(entry.power_w, scale=self._calib_scale)
                power_label = f"{power_val:.6g}".replace(".", "p")
                a_label = f"{entry.a_deg:.3f}".replace(".", "p")
                b_label = f"{entry.b_deg:.3f}".replace(".", "p")
                base = f"PL_A{a_label}_B{b_label}_P{power_label}{self._calib_unit}"
                filepath = self._next_filename(base)
                plan.append({"entry": entry, "filepath": filepath})

            meta = self._build_metadata_base()
            wl_axis = self.liveView.get_wavelength_axis() if self.liveView is not None else None

            self._sweep_thread = DualWheelSweepThread(
                self.cam,
                self.stage,
                self.stage_b,
                plan,
                int(self.accumSpin.value()),
                float(self.exposureSpin.value()),
                float(DEFAULT_RAMP_STEP_DEG),
                meta,
                wl_axis,
            )
            self._sweep_thread.frame_ready.connect(self.on_sweep_frame)
            self._sweep_thread.power_done.connect(self.on_sweep_power_done)
            self._sweep_thread.status.connect(self.statusBar().showMessage)
            self._sweep_thread.done.connect(self.on_sweep_done)
            self._sweep_thread.start()
            self._set_busy(True)
            self.statusBar().showMessage("Sweeping dual-wheel powers (file order)")
            return

        if self._calib_data is None:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Load a power calibration file first.")
            return
        if self.cam.cam is None or self.stage is None:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Camera and stage must be connected.")
            return

        series = self.seriesCombo.currentText().strip()
        entries = sorted(self._series_entries.get(series, []), key=lambda e: e.power_w)
        if not entries:
            QtWidgets.QMessageBox.warning(self, "Sweep", f"No entries for series '{series}'.")
            return

        self._prepare_single_mode()
        self._resolve_save_dir()

        plan = []
        for entry in entries:
            power_val = power_key(entry.power_w, scale=self._calib_scale)
            power_label = f"{power_val:.6g}".replace(".", "p")
            series_label = self._safe_series_label(entry.series)
            base = f"PL_{series_label}_{power_label}{self._calib_unit}"
            filepath = self._next_filename(base)
            plan.append({"entry": entry, "filepath": filepath})

        meta = self._build_metadata_base()
        wl_axis = self.liveView.get_wavelength_axis() if self.liveView is not None else None

        self._sweep_thread = SweepThread(
            self.cam,
            self.stage,
            plan,
            int(self.accumSpin.value()),
            float(self.exposureSpin.value()),
            float(DEFAULT_RAMP_STEP_DEG),
            meta,
            wl_axis,
        )
        self._sweep_thread.frame_ready.connect(self.on_sweep_frame)
        self._sweep_thread.power_done.connect(self.on_sweep_power_done)
        self._sweep_thread.status.connect(self.statusBar().showMessage)
        self._sweep_thread.done.connect(self.on_sweep_done)
        self._sweep_thread.start()
        self._set_busy(True)
        self.statusBar().showMessage(f"Sweeping series '{series}'")

    def on_abort(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
        if self._accum_thread is not None and self._accum_thread.isRunning():
            self._accum_thread.stop()
        if self._sweep_thread is not None and self._sweep_thread.isRunning():
            self._sweep_thread.stop()
        if self._stage_controller is not None:
            try:
                self._stage_controller.abort()
            except Exception:
                pass
        for stage in (self.stage, self.stage_b):
            try:
                if stage is not None:
                    stage.stop()
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

    def on_stop_live(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
            self.statusBar().showMessage("Stopping live...")
            self._live_thread.wait(2000)
        if self._live_thread is not None and not self._live_thread.isRunning():
            self._on_live_done()
        try:
            if self.cam is not None:
                self.cam.stop_stream()
        except Exception:
            pass
        if self._live_thread is None:
            self.statusBar().showMessage("Live stopped")

    def on_live_frame(self, fr: dict):
        raw = fr.get("image")
        if raw is None:
            return
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def on_accum_frame(self, fr: dict):
        raw = fr.get("image")
        if raw is None:
            return
        accum_idx = fr.get("accum_idx")
        accum_n = fr.get("accum_n")
        if accum_idx and accum_n:
            self.statusBar().showMessage(f"Accumulated {accum_idx}/{accum_n}")
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def on_accum_done(self):
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Accumulation done")

    def on_sweep_frame(self, fr: dict):
        raw = fr.get("image")
        if raw is None:
            return
        accum_idx = fr.get("accum_idx")
        accum_n = fr.get("accum_n")
        if accum_idx and accum_n:
            self.statusBar().showMessage(f"Accumulated {accum_idx}/{accum_n}")
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def _on_live_done(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            return
        self._live_thread = None
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Live stopped")

    def on_sweep_power_done(self, payload: dict):
        entry = payload.get("entry")
        raw = payload.get("image")
        if entry is None or raw is None:
            return

        key = power_key(entry.power_w, scale=self._calib_scale)
        self._mark_power_taken(key)
        self._taken_entries.add(self._entry_key(entry))

        if self.liveView is None:
            return
        disp = self.liveView.prepare_display_image(raw)
        rank = self._entry_rank(entry)
        cur = self._power_images.get(key)
        if cur is None or rank < cur["rank"]:
            self._power_images[key] = {"rank": rank, "image": disp}
        elif key not in self._power_images:
            self._power_images[key] = {"rank": rank, "image": disp}

        self._refresh_linecut_map()

        if self._total_entries and len(self._taken_entries) >= self._total_entries:
            self._write_session_metadata()

    def on_sweep_done(self, status: str):
        self._set_busy(False)
        self._sweep_thread = None
        if status == "ok":
            self.statusBar().showMessage("Sweep complete")
        elif status == "aborted":
            self.statusBar().showMessage("Sweep aborted")
        else:
            self.statusBar().showMessage(str(status))

    # -----------------
    # Image handling
    # -----------------
    def _prepare_display_image(self, raw):
        disp = np.flipud(np.asarray(raw))
        h, w = disp.shape
        top = max(0, min(self._crop_top, h - 1))
        bottom = max(0, min(self._crop_bottom, h - 1))
        left = max(0, min(self._crop_left, w - 1))
        right = max(0, min(self._crop_right, w - 1))
        y1 = max(top, 0)
        y2 = max(y1 + 1, h - bottom)
        x1 = max(left, 0)
        x2 = max(x1 + 1, w - right)
        return disp[y1:y2, x1:x2]

    def _update_wavelength_axis(self, force: bool = False) -> None:
        center = float(self.centerSpin.value())
        if abs(center) <= 1e-9:
            self._wl_axis = None
            return
        try:
            wl = self.cam.get_wavelength_axis(force=bool(force))
        except Exception:
            wl = None
        if wl is None:
            self._wl_axis = None
            return
        arr = np.asarray(wl, dtype=float).ravel()
        if arr.size == 0 or np.allclose(arr, 0.0):
            self._wl_axis = None
            return
        self._wl_axis = arr

    def _update_xaxis_values(self, raw_width: int, disp_width: int) -> None:
        if self._wl_axis is None or abs(float(self.centerSpin.value())) <= 1e-9:
            self._xaxis_values = np.arange(disp_width)
            return
        arr = np.asarray(self._wl_axis, dtype=float).ravel()
        if arr.size != raw_width:
            self._xaxis_values = np.arange(disp_width)
            return
        left = max(0, self._crop_left)
        right = max(0, self._crop_right)
        if right > 0:
            arr = arr[: arr.size - right]
        if left > 0:
            arr = arr[left:]
        if arr.size != disp_width:
            self._xaxis_values = np.arange(disp_width)
            return
        self._xaxis_values = arr

    def _update_image(self, raw, raw8):
        disp = self._prepare_display_image(raw)
        h, w = disp.shape
        self._update_wavelength_axis(force=False)
        self._update_xaxis_values(int(np.asarray(raw).shape[1]), w)

        if self._linecut_row < 0 or self._linecut_row >= h:
            self._linecut_row = int(h // 2)

        x_axis = self._xaxis_values if self._xaxis_values is not None else np.arange(w)
        x0 = float(x_axis[0]) if len(x_axis) else 0.0
        x1 = float(x_axis[-1]) if len(x_axis) else float(w)

        self.img_artist.set_data(disp)
        self.img_artist.set_extent((x0, x1, h, 0))
        self.ax_img.set_xlim(x0, x1)
        self.ax_img.set_ylim(h, 0)
        self.ax_img.set_xlabel("Wavelength (nm)" if self._wl_axis is not None and abs(float(self.centerSpin.value())) > 1e-9 else "Pixel")
        self._line_sel.set_ydata([self._linecut_row, self._linecut_row])
        self._line_sel.set_visible(True)
        if raw8 is not None:
            disp8 = self._prepare_display_image(raw8)
            self.img_artist.set_data(disp8)
            self.img_artist.set_clim(0, 255)
        else:
            try:
                vmin = float(np.nanmin(disp))
                vmax = float(np.nanmax(disp))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.img_artist.set_clim(vmin, vmax)
            except Exception:
                pass

        self._update_cursor_overlays(disp, x_axis)
        self.canvas_img.draw_idle()

    def on_image_click(self, event):
        if event.inaxes != self.ax_img:
            return
        if self.img_artist is None:
            return
        if event.ydata is None or event.xdata is None:
            return
        try:
            y = int(round(float(event.ydata)))
        except Exception:
            return
        img = self.img_artist.get_array()
        if img is None:
            return
        h, w = img.shape
        if not (0 <= y < h):
            return
        x_axis = self._xaxis_values if self._xaxis_values is not None else np.arange(w)
        try:
            x = int(np.argmin(np.abs(x_axis - float(event.xdata))))
        except Exception:
            try:
                x = int(round(float(event.xdata)))
            except Exception:
                return
        x = max(0, min(w - 1, x))
        self._cursor_rc = (y, x)
        self._linecut_row = y
        self._update_cursor_overlays(img, x_axis)
        self.canvas_img.draw_idle()
        self._refresh_linecut_map()

    def _linecut_from_display(self, img, row: int) -> Optional[np.ndarray]:
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim != 2:
            return None
        h, _ = a.shape
        width = 1
        if self.liveView is not None:
            width = max(1, int(self.liveView.linecut_width()))
        half = width // 2
        r1 = max(0, int(row) - half)
        r2 = min(h, int(row) + half + (1 if width % 2 else 0))
        sl = a[r1:r2, :]
        return sl.sum(axis=0)

    def _linecut_vertical(self, img, col: int) -> Optional[np.ndarray]:
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim != 2:
            return None
        _, w = a.shape
        width = 1
        if self.liveView is not None:
            width = max(1, int(self.liveView.linecut_width()))
        half = width // 2
        c1 = max(0, int(col) - half)
        c2 = min(w, int(col) + half + (1 if width % 2 else 0))
        sl = a[:, c1:c2]
        return sl.sum(axis=1)

    def _update_cursor_overlays(self, img, x_axis):
        if img is None:
            return
        h, w = img.shape
        if self._cursor_rc is None:
            y = self._linecut_row
            x = int(w // 2)
        else:
            y, x = self._cursor_rc
        y = max(0, min(h - 1, int(y)))
        x = max(0, min(w - 1, int(x)))
        self._cursor_rc = (y, x)
        self._line_sel.set_ydata([y, y])
        self._line_sel.set_visible(True)
        self._cursor_h.set_ydata([y, y])
        x_pos = float(x_axis[x]) if len(x_axis) else float(x)
        self._cursor_v.set_xdata([x_pos, x_pos])
        self._cursor_h.set_visible(True)
        self._cursor_v.set_visible(True)

        try:
            val = float(img[y, x])
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}, I = {val:g}")
        except Exception:
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}")

        hcut = self._linecut_from_display(img, y)
        vcut = self._linecut_vertical(img, x)
        if hcut is not None:
            self.line_h.set_data(x_axis, hcut)
            self.ax_h.relim()
            self.ax_h.autoscale_view()
            xlabel = "Wavelength (nm)" if self._wl_axis is not None and abs(float(self.centerSpin.value())) > 1e-9 else "Pixel"
            self.ax_h.set_xlabel(xlabel)
        if vcut is not None:
            self.line_v.set_data(vcut, np.arange(vcut.size))
            self.ax_v.relim()
            self.ax_v.autoscale_view()

    def _refresh_linecut_map(self):
        if not self._power_list_keys or not self._power_images:
            self.map_artist.set_data(np.zeros((10, 10)))
            self.canvas_map.draw_idle()
            return
        if self.liveView is None:
            return
        first = next(iter(self._power_images.values()))
        img0 = first.get("image")
        if img0 is None:
            return
        _, width = img0.shape

        data = np.full((width, len(self._power_list_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return
        for col, key in enumerate(self._power_list_keys):
            info = self._power_images.get(key)
            if not info:
                continue
            img = info.get("image")
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        powers = self._power_list_keys
        pmin = float(powers[0]) if powers else 0.0
        pmax = float(powers[-1]) if powers else 1.0
        if pmin == pmax:
            pmax = pmin + 1.0

        x_axis = self.liveView.get_linecut_axis()
        if x_axis is not None and len(x_axis) == width:
            y0 = float(x_axis[0])
            y1 = float(x_axis[-1])
            ylabel = "Wavelength (nm)" if self.liveView.has_wavelength_axis() else "Pixel"
        else:
            y0 = 0.0
            y1 = float(width)
            ylabel = "Pixel"

        self.map_artist.set_data(data)
        self.map_artist.set_extent((pmin, pmax, y0, y1))
        self.ax_map.set_xlabel(f"Power ({self._calib_unit})")
        self.ax_map.set_ylabel(ylabel)
        try:
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
            if vmax <= vmin:
                vmax = vmin + 1.0
            self.map_artist.set_clim(vmin, vmax)
        except Exception:
            pass
        self.canvas_map.draw_idle()

    # -----------------
    # Session metadata
    # -----------------
    def _write_session_metadata(self) -> None:
        if self._session_saved or not self._save_dir:
            return
        path = os.path.join(self._save_dir, "session.md")
        meta = self._build_metadata_base()
        lines = [
            "# PL power sweep session",
            "",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Save folder: {self._save_dir}",
            "",
            "## Hardware",
            f"- Stage A serial: {meta.get('stage_serial_a', '')}",
            f"- Stage B serial: {meta.get('stage_serial_b', '')}",
            f"- Exposure (ms): {meta.get('exposure_ms', '')}",
            f"- Accum N: {meta.get('accum_n', '')}",
            f"- Readout rate: {meta.get('readout_rate', '')}",
            f"- Preamp gain: {meta.get('preamp_gain', '')}",
            f"- Output amp: {meta.get('output_amp', '')}",
            f"- Grating: {meta.get('grating', '')}",
            f"- Center (nm): {meta.get('center_wl_nm', '')}",
            f"- Slit ({meta.get('slit_id', '')}) um: {meta.get('slit_um', '')}",
            "",
            "## Calibration",
            f"- File: {meta.get('power_calib_file', '')}",
            f"- Mode: {meta.get('calib_mode', '')}",
            "",
            "## Display Crop",
            f"- Top: {meta.get('crop_top', '')}",
            f"- Bottom: {meta.get('crop_bottom', '')}",
            f"- Left: {meta.get('crop_left', '')}",
            f"- Right: {meta.get('crop_right', '')}",
            "",
            "## Linecut",
            f"- Row: {meta.get('linecut_row', '')}",
            f"- Width: {meta.get('linecut_width', '')}",
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._session_saved = True
            self.statusBar().showMessage("Session metadata saved")
        except Exception as exc:
            self.statusBar().showMessage(f"Metadata save failed: {exc}")

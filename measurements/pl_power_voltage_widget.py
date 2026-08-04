import os
import re
import time
import types as _types
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

from keithley.gui.plots import PlotWidget
from keithley.keithley_wrapper import KEITHLEY_V_LIMIT, KeithleySMU, SweepController, make_resource_manager

from measurements.pl_power_voltage_config import (
    DEFAULT_ACQ_NUMBER,
    DEFAULT_BACK_RESOURCE,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_FRONT_RESOURCE,
    DEFAULT_GATE_ICOMP_NA,
    DEFAULT_GATE_VSET,
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
    DEFAULT_SWEEP_ICOMP_NA,
    DEFAULT_SWEEP_STEP,
    DEFAULT_SWEEP_STEPS,
    DEFAULT_SWEEP_V0,
    POLL_ERROR_LIMIT,
    POLL_GATE_MS,
    POLL_MS,
    POLL_TIMEOUT_MS,
    RAMP_DWELL_S,
    RAMP_STEP_V,
    READOUT_UNKNOWN,
    DATA_DIR,
    SWEEP_SETTLE_MS,
    ZERO_V_EPS,
    ZERO_V_EXTRA_SETTLE_MS,
    VISA_DLL,
)
from measurements.dual_wheel_power_calibration import (
    DualWheelPowerEntry,
    DualWheelPowerData,
    HWPNDCalibData,
    is_dual_wheel_calibration,
    is_hwp_nd_calibration,
    load_dual_wheel_calibration,
    load_hwp_nd_calibration,
    compute_hwp_nd_entries,
)
from measurements.power_calibration import (
    PowerCalibEntry,
    entry_priority,
    load_power_calibration,
    power_key,
)
from measurements.pl_power_voltage_workers import GateRampThread, PowerVoltageSweepThread

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None


class PLPowerVoltageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)
        self.stage: Optional[RotationStage] = None
        self.stage_b: Optional[RotationStage] = None

        self._live_thread: Optional[LiveAcqThread] = None
        self._accum_thread: Optional[SingleAcqThread] = None
        self._sweep_thread: Optional[PowerVoltageSweepThread] = None
        self._home_thread: Optional[HomeThread] = None
        self._move_thread: Optional[MoveThread] = None

        self._stage_controller: Optional[MotionController] = None
        self._stage_busy = False
        self._busy = False
        self._aborting = False
        self._live_active = False

        self._live_view = None
        self._linecut_row = int(DEFAULT_LINECUT_ROW)
        self._linecut_width = int(DEFAULT_LINECUT_WIDTH)
        self._wl_axis = None
        self._xaxis_values = None
        self._cursor_rc = None

        self._calib_data = None
        self._dual_calib_data: Optional[DualWheelPowerData] = None
        self._hwp_nd_calib_data: Optional[HWPNDCalibData] = None
        self._series_entries: Dict[str, List[PowerCalibEntry]] = {}
        self._dual_entries: List[DualWheelPowerEntry] = []
        self._power_display: Dict[float, dict] = {}
        self._power_list_keys: List[float] = []
        self._power_view_keys: List[float] = []
        self._pv_images: Dict[float, Dict[float, dict]] = {}
        self._voltage_keys: List[float] = []
        self._voltage_labels: List[str] = []
        self._voltage_pairs: Dict[float, Tuple[Optional[float], Optional[float]]] = {}
        self._taken_powers = set()
        self._taken_entries = set()
        self._power_done_counts: Dict[float, int] = {}
        self._power_target_counts: Dict[float, int] = {}
        self._total_entries = 0
        self._angle_list_mode = False
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

        self.rm = None
        try:
            self.rm = make_resource_manager(VISA_DLL)
        except Exception:
            self.rm = None

        self.front_dev: Optional[KeithleySMU] = None
        self.back_dev: Optional[KeithleySMU] = None
        self._front_output_on = False
        self._back_output_on = False
        self._front_poll_failures = 0
        self._back_poll_failures = 0
        self._front_ramp_thread: Optional[GateRampThread] = None
        self._back_ramp_thread: Optional[GateRampThread] = None
        self._front_ramp_controller: Optional[SweepController] = None
        self._back_ramp_controller: Optional[SweepController] = None
        self._gate_auto_reconnect_armed = False
        self._front_reconnect_next_ts = 0.0
        self._back_reconnect_next_ts = 0.0

        self._build_ui()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.start(int(POLL_MS))

        self._gate_poll_timer = QtCore.QTimer(self)
        self._gate_poll_timer.timeout.connect(self._poll_gates)
        self._gate_poll_timer.start(int(POLL_GATE_MS))

    def _threads_running(self) -> bool:
        live = self._live_thread is not None and self._live_thread.isRunning()
        accum = self._accum_thread is not None and self._accum_thread.isRunning()
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return live or accum or sweep

    def _gate_threads_running(self) -> bool:
        front = self._front_ramp_thread is not None and self._front_ramp_thread.isRunning()
        back = self._back_ramp_thread is not None and self._back_ramp_thread.isRunning()
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return front or back or sweep

    def _is_busy(self) -> bool:
        return self._busy or self._stage_busy or self._threads_running() or self._gate_threads_running()

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self._build_dashboard(vbox)
        self._build_session_controls(vbox)
        self._build_gate_controls(vbox)
        self._build_plots(vbox)

        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)
        self.statusBar().showMessage("Idle")

        self._wire_signals()
        self._update_gate_sweep_controls()

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

        self.readoutRateCombo = QtWidgets.QComboBox()
        self.readoutRateCombo.setEditable(False)
        self.readoutRateCombo.addItem(str(DEFAULT_READOUT_RATE))
        self.readoutRateCombo.setFixedWidth(150)

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
        line1.addWidget(QtWidgets.QLabel("Rate:"))
        line1.addWidget(self.readoutRateCombo)
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

        # Front gate line
        self.frontGateDot = self._make_dot()
        self.frontGateStatusLbl = QtWidgets.QLabel("Front gate: disconnected")
        self.frontGateStatusLbl.setFont(mono)

        self.frontVreadVal = QtWidgets.QLabel(READOUT_UNKNOWN)
        self.frontVreadVal.setFont(mono)
        self.frontVreadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.frontVreadVal.setMinimumWidth(90)

        self.frontIreadVal = QtWidgets.QLabel(READOUT_UNKNOWN)
        self.frontIreadVal.setFont(mono)
        self.frontIreadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.frontIreadVal.setMinimumWidth(90)

        self.frontOutputLbl = QtWidgets.QLabel("Output: --")
        self.frontOutputLbl.setFont(mono)

        self.frontResourceEdit = QtWidgets.QLineEdit(DEFAULT_FRONT_RESOURCE)
        self.frontResourceEdit.setMinimumWidth(340)

        line5 = QtWidgets.QHBoxLayout()
        line5.addWidget(self.frontGateDot)
        line5.addWidget(QtWidgets.QLabel("Front gate:"))
        line5.addWidget(self.frontGateStatusLbl, 1)
        line5.addWidget(QtWidgets.QLabel("V-read (V):"))
        line5.addWidget(self.frontVreadVal)
        line5.addWidget(QtWidgets.QLabel("I-read (nA):"))
        line5.addWidget(self.frontIreadVal)
        line5.addWidget(self.frontOutputLbl)
        line5.addSpacing(12)
        line5.addWidget(QtWidgets.QLabel("Resource:"))
        line5.addWidget(self.frontResourceEdit)

        self.frontVsetSpin = QtWidgets.QDoubleSpinBox()
        self.frontVsetSpin.setDecimals(6)
        self.frontVsetSpin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.frontVsetSpin.setValue(DEFAULT_GATE_VSET)
        self.frontVsetSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.frontVsetSpin.setFixedWidth(110)

        self.frontIcompSpin = QtWidgets.QDoubleSpinBox()
        self.frontIcompSpin.setDecimals(3)
        self.frontIcompSpin.setRange(0.0, 1e9)
        self.frontIcompSpin.setValue(DEFAULT_GATE_ICOMP_NA)
        self.frontIcompSpin.setFixedWidth(110)

        self.frontApplyBtn = QtWidgets.QPushButton("Apply Front")
        self.frontApplyBtn.setFixedWidth(120)

        line6 = QtWidgets.QHBoxLayout()
        line6.addStretch()
        line6.addWidget(QtWidgets.QLabel("V set (V):"))
        line6.addWidget(self.frontVsetSpin)
        line6.addWidget(QtWidgets.QLabel("I comp (nA):"))
        line6.addWidget(self.frontIcompSpin)
        line6.addWidget(self.frontApplyBtn)

        grid.addLayout(line5, 4, 0, 1, 1)
        grid.addLayout(line6, 5, 0, 1, 1)

        # Back gate line
        self.backGateDot = self._make_dot()
        self.backGateStatusLbl = QtWidgets.QLabel("Back gate: disconnected")
        self.backGateStatusLbl.setFont(mono)

        self.backVreadVal = QtWidgets.QLabel(READOUT_UNKNOWN)
        self.backVreadVal.setFont(mono)
        self.backVreadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.backVreadVal.setMinimumWidth(90)

        self.backIreadVal = QtWidgets.QLabel(READOUT_UNKNOWN)
        self.backIreadVal.setFont(mono)
        self.backIreadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.backIreadVal.setMinimumWidth(90)

        self.backOutputLbl = QtWidgets.QLabel("Output: --")
        self.backOutputLbl.setFont(mono)

        self.backResourceEdit = QtWidgets.QLineEdit(DEFAULT_BACK_RESOURCE)
        self.backResourceEdit.setMinimumWidth(340)

        line7 = QtWidgets.QHBoxLayout()
        line7.addWidget(self.backGateDot)
        line7.addWidget(QtWidgets.QLabel("Back gate:"))
        line7.addWidget(self.backGateStatusLbl, 1)
        line7.addWidget(QtWidgets.QLabel("V-read (V):"))
        line7.addWidget(self.backVreadVal)
        line7.addWidget(QtWidgets.QLabel("I-read (nA):"))
        line7.addWidget(self.backIreadVal)
        line7.addWidget(self.backOutputLbl)
        line7.addSpacing(12)
        line7.addWidget(QtWidgets.QLabel("Resource:"))
        line7.addWidget(self.backResourceEdit)

        self.backVsetSpin = QtWidgets.QDoubleSpinBox()
        self.backVsetSpin.setDecimals(6)
        self.backVsetSpin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.backVsetSpin.setValue(DEFAULT_GATE_VSET)
        self.backVsetSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.backVsetSpin.setFixedWidth(110)

        self.backIcompSpin = QtWidgets.QDoubleSpinBox()
        self.backIcompSpin.setDecimals(3)
        self.backIcompSpin.setRange(0.0, 1e9)
        self.backIcompSpin.setValue(DEFAULT_GATE_ICOMP_NA)
        self.backIcompSpin.setFixedWidth(110)

        self.backApplyBtn = QtWidgets.QPushButton("Apply Back")
        self.backApplyBtn.setFixedWidth(120)

        line8 = QtWidgets.QHBoxLayout()
        line8.addStretch()
        line8.addWidget(QtWidgets.QLabel("V set (V):"))
        line8.addWidget(self.backVsetSpin)
        line8.addWidget(QtWidgets.QLabel("I comp (nA):"))
        line8.addWidget(self.backIcompSpin)
        line8.addWidget(self.backApplyBtn)

        grid.addLayout(line7, 6, 0, 1, 1)
        grid.addLayout(line8, 7, 0, 1, 1)

        # Action line
        self.initBtn = QtWidgets.QPushButton("Initialize")
        self.initBtn.setFixedWidth(120)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setFixedWidth(120)
        self.takeCurrentBtn = QtWidgets.QPushButton("Take Current Image")
        self.takeCurrentBtn.setFixedWidth(160)
        self.stopLiveBtn = QtWidgets.QPushButton("Stop Live")
        self.stopLiveBtn.setFixedWidth(120)
        self.sweepBtn = QtWidgets.QPushButton("Start Sweep")
        self.sweepBtn.setFixedWidth(140)
        self.abortBtn = QtWidgets.QPushButton("Abort")
        self.abortBtn.setFixedWidth(100)
        self.homeBothBtn = QtWidgets.QPushButton("Home Both Gates")
        self.homeBothBtn.setFixedWidth(150)
        self.homeBothBtn.setEnabled(False)

        line5 = QtWidgets.QHBoxLayout()
        line5.addStretch()
        line5.addWidget(self.initBtn)
        line5.addWidget(self.disconnectBtn)
        line5.addSpacing(10)
        line5.addWidget(self.homeBothBtn)
        line5.addSpacing(10)
        line5.addWidget(self.takeCurrentBtn)
        line5.addWidget(self.stopLiveBtn)
        line5.addWidget(self.sweepBtn)
        line5.addWidget(self.abortBtn)

        grid.addLayout(line5, 8, 0, 1, 1)

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

        self.ndWheelAngleLbl = QtWidgets.QLabel("ND wheel angle:")
        self.ndWheelAngleCombo = QtWidgets.QComboBox()
        self.ndWheelAngleCombo.setMinimumWidth(100)

        self.hwpAngleLbl = QtWidgets.QLabel("HWP range:")
        self.hwpMinSpin = QtWidgets.QDoubleSpinBox()
        self.hwpMinSpin.setRange(0, 360); self.hwpMinSpin.setDecimals(2)
        self.hwpMinSpin.setSingleStep(0.5); self.hwpMinSpin.setSuffix(" °")
        self.hwpMinSpin.setFixedWidth(90)
        self.hwpMaxSpin = QtWidgets.QDoubleSpinBox()
        self.hwpMaxSpin.setRange(0, 360); self.hwpMaxSpin.setDecimals(2)
        self.hwpMaxSpin.setSingleStep(0.5); self.hwpMaxSpin.setSuffix(" °")
        self.hwpMaxSpin.setFixedWidth(90)

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

        grid.addWidget(self.ndWheelAngleLbl,   4, 0)
        grid.addWidget(self.ndWheelAngleCombo, 4, 1)
        grid.addWidget(self.hwpAngleLbl,       4, 2)
        grid.addWidget(self.hwpMinSpin,        4, 3)
        grid.addWidget(QtWidgets.QLabel("–"),  4, 4)
        grid.addWidget(self.hwpMaxSpin,        4, 5)

        grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 5, 0)
        grid.addWidget(self.cropTopSpin, 5, 1)
        grid.addWidget(self.cropBottomSpin, 5, 2)
        grid.addWidget(self.cropLeftSpin, 5, 3)
        grid.addWidget(self.cropRightSpin, 5, 4)
        grid.addWidget(self.cropApplyBtn, 5, 5)

        parent_layout.addWidget(box)

    def _build_gate_controls(self, parent_layout):
        box = QtWidgets.QGroupBox("Sweep Options")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.sweepPowerChk = QtWidgets.QCheckBox("Sweep power")
        self.sweepPowerChk.setChecked(True)
        self.sweepVoltageChk = QtWidgets.QCheckBox("Sweep voltage")
        self.sweepVoltageChk.setChecked(False)

        self.sweepGateCombo = QtWidgets.QComboBox()
        self.sweepGateCombo.addItems(["Dual gate", "Front gate", "Back gate"])
        self.sweepGateCombo.setEnabled(False)

        self.frontV0Spin = QtWidgets.QDoubleSpinBox()
        self.frontV0Spin.setDecimals(6)
        self.frontV0Spin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.frontV0Spin.setValue(DEFAULT_SWEEP_V0)
        self.frontV0Spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.frontStepSpin = QtWidgets.QDoubleSpinBox()
        self.frontStepSpin.setDecimals(6)
        self.frontStepSpin.setRange(-50.0, 50.0)
        self.frontStepSpin.setValue(DEFAULT_SWEEP_STEP)
        self.frontStepSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.backV0Spin = QtWidgets.QDoubleSpinBox()
        self.backV0Spin.setDecimals(6)
        self.backV0Spin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.backV0Spin.setValue(DEFAULT_SWEEP_V0)
        self.backV0Spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.backStepSpin = QtWidgets.QDoubleSpinBox()
        self.backStepSpin.setDecimals(6)
        self.backStepSpin.setRange(-50.0, 50.0)
        self.backStepSpin.setValue(DEFAULT_SWEEP_STEP)
        self.backStepSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.sweepStepsSpin = QtWidgets.QSpinBox()
        self.sweepStepsSpin.setRange(1, 100000)
        self.sweepStepsSpin.setValue(int(DEFAULT_SWEEP_STEPS))
        self.sweepStepsSpin.setFixedWidth(90)

        self.powerSkipSpin = QtWidgets.QSpinBox()
        self.powerSkipSpin.setRange(0, 100000)
        self.powerSkipSpin.setValue(0)
        self.powerSkipSpin.setFixedWidth(90)

        grid.addWidget(self.sweepPowerChk, 0, 0)
        grid.addWidget(self.sweepVoltageChk, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Sweep gate:"), 0, 2)
        grid.addWidget(self.sweepGateCombo, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Skip N power points:"), 0, 4)
        grid.addWidget(self.powerSkipSpin, 0, 5)

        # Angle list row (stage A, no calibration file needed)
        self.angleListChk = QtWidgets.QCheckBox("Use angle list (stage A)")
        self.angleListEdit = QtWidgets.QLineEdit()
        self.angleListEdit.setPlaceholderText("e.g.  0, 15.5, 30  (comma-separated °)")
        self.angleListPreviewLbl = QtWidgets.QLabel("(0 angles)")
        self.angleListPreviewLbl.setStyleSheet("color: #555; font-size: 8pt;")
        grid.addWidget(self.angleListChk,        1, 0, 1, 2)
        grid.addWidget(self.angleListEdit,        1, 2, 1, 3)
        grid.addWidget(self.angleListPreviewLbl,  1, 5, 1, 2)

        self.frontVendLbl = QtWidgets.QLabel("End: 0.000000 V")
        self.backVendLbl = QtWidgets.QLabel("End: 0.000000 V")

        grid.addWidget(QtWidgets.QLabel("Front V0 (V):"), 2, 0)
        grid.addWidget(self.frontV0Spin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Front step (V):"), 2, 2)
        grid.addWidget(self.frontStepSpin, 2, 3)
        grid.addWidget(QtWidgets.QLabel("Steps:"), 2, 4)
        grid.addWidget(self.sweepStepsSpin, 2, 5)
        grid.addWidget(self.frontVendLbl, 2, 6)

        grid.addWidget(QtWidgets.QLabel("Back V0 (V):"), 3, 0)
        grid.addWidget(self.backV0Spin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Back step (V):"), 3, 2)
        grid.addWidget(self.backStepSpin, 3, 3)
        grid.addWidget(self.backVendLbl, 3, 6)

        parent_layout.addWidget(box)
    def _build_plots(self, parent_layout):
        tabs = QtWidgets.QTabWidget()

        # Live tab
        live_tab = QtWidgets.QWidget()
        live_layout = QtWidgets.QHBoxLayout(live_tab)
        live_layout.setContentsMargins(8, 8, 8, 8)
        live_layout.setSpacing(8)

        self.liveView = AndorLiveViewWidget(self.cam, title="PL Image")
        self._apply_crop_to_liveview()
        self.liveView.set_linecut_row(int(DEFAULT_LINECUT_ROW))
        self.liveView.linecut_changed.connect(lambda _: self._refresh_linecut_maps())

        gate_plot_box = QtWidgets.QWidget()
        gate_plot_layout = QtWidgets.QVBoxLayout(gate_plot_box)
        gate_plot_layout.setContentsMargins(0, 0, 0, 0)
        gate_plot_layout.setSpacing(6)

        front_plot_box = QtWidgets.QGroupBox("Front gate V/I")
        front_plot_layout = QtWidgets.QVBoxLayout(front_plot_box)
        front_plot_layout.setContentsMargins(6, 6, 6, 6)
        self.frontPlot = PlotWidget(front_plot_box)
        front_plot_layout.addWidget(self.frontPlot)

        back_plot_box = QtWidgets.QGroupBox("Back gate V/I")
        back_plot_layout = QtWidgets.QVBoxLayout(back_plot_box)
        back_plot_layout.setContentsMargins(6, 6, 6, 6)
        self.backPlot = PlotWidget(back_plot_box)
        back_plot_layout.addWidget(self.backPlot)

        gate_plot_layout.addWidget(front_plot_box, 1)
        gate_plot_layout.addWidget(back_plot_box, 1)

        live_layout.addWidget(self.liveView, 4)
        live_layout.addWidget(gate_plot_box, 2)

        # Power/Voltage tab
        power_tab = QtWidgets.QWidget()
        power_layout = QtWidgets.QHBoxLayout(power_tab)
        power_layout.setContentsMargins(8, 8, 8, 8)
        power_layout.setSpacing(8)

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

        self.voltageSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.voltageSlider.setRange(0, 0)
        self.voltageSlider.setEnabled(False)
        self.voltageSelLbl = QtWidgets.QLabel("Voltage: --")

        power_plot_box = QtWidgets.QWidget()
        power_plot_layout = QtWidgets.QVBoxLayout(power_plot_box)
        power_plot_layout.setContentsMargins(0, 0, 0, 0)
        power_plot_layout.setSpacing(4)
        power_plot_layout.addWidget(self.canvas_map)
        power_slider_row = QtWidgets.QHBoxLayout()
        power_slider_row.addWidget(QtWidgets.QLabel("Select V:"))
        power_slider_row.addWidget(self.voltageSlider, 1)
        power_slider_row.addWidget(self.voltageSelLbl)
        power_plot_layout.addLayout(power_slider_row)
        power_clim_row = QtWidgets.QHBoxLayout()
        power_clim_row.addWidget(QtWidgets.QLabel("Color min:"))
        self.mapCminEdit = QtWidgets.QLineEdit()
        self.mapCminEdit.setPlaceholderText("auto")
        self.mapCminEdit.setFixedWidth(90)
        power_clim_row.addWidget(self.mapCminEdit)
        power_clim_row.addWidget(QtWidgets.QLabel("max:"))
        self.mapCmaxEdit = QtWidgets.QLineEdit()
        self.mapCmaxEdit.setPlaceholderText("auto")
        self.mapCmaxEdit.setFixedWidth(90)
        power_clim_row.addWidget(self.mapCmaxEdit)
        power_clim_row.addStretch()
        power_plot_layout.addLayout(power_clim_row)

        # Linecut vs voltage
        self.fig_vmap = Figure(figsize=(4, 4.5), dpi=100)
        self.canvas_vmap = FigureCanvas(self.fig_vmap)
        self.ax_vmap = self.fig_vmap.add_subplot(111)
        self.ax_vmap.set_title("Linecut vs Voltage")
        self.ax_vmap.set_xlabel("Voltage (V)")
        self.ax_vmap.set_ylabel("Wavelength (nm)")
        self.vmap_artist = self.ax_vmap.imshow(
            np.zeros((10, 10)),
            cmap="viridis",
            origin="lower",
            aspect="auto",
        )

        self.powerSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.powerSlider.setRange(0, 0)
        self.powerSlider.setEnabled(False)
        self.powerSelLbl = QtWidgets.QLabel("Power: --")

        voltage_plot_box = QtWidgets.QWidget()
        voltage_plot_layout = QtWidgets.QVBoxLayout(voltage_plot_box)
        voltage_plot_layout.setContentsMargins(0, 0, 0, 0)
        voltage_plot_layout.setSpacing(4)
        voltage_plot_layout.addWidget(self.canvas_vmap)
        voltage_slider_row = QtWidgets.QHBoxLayout()
        voltage_slider_row.addWidget(QtWidgets.QLabel("Select power:"))
        voltage_slider_row.addWidget(self.powerSlider, 1)
        voltage_slider_row.addWidget(self.powerSelLbl)
        voltage_plot_layout.addLayout(voltage_slider_row)
        voltage_clim_row = QtWidgets.QHBoxLayout()
        voltage_clim_row.addWidget(QtWidgets.QLabel("Color min:"))
        self.vmapCminEdit = QtWidgets.QLineEdit()
        self.vmapCminEdit.setPlaceholderText("auto")
        self.vmapCminEdit.setFixedWidth(90)
        voltage_clim_row.addWidget(self.vmapCminEdit)
        voltage_clim_row.addWidget(QtWidgets.QLabel("max:"))
        self.vmapCmaxEdit = QtWidgets.QLineEdit()
        self.vmapCmaxEdit.setPlaceholderText("auto")
        self.vmapCmaxEdit.setFixedWidth(90)
        voltage_clim_row.addWidget(self.vmapCmaxEdit)
        voltage_clim_row.addStretch()
        voltage_plot_layout.addLayout(voltage_clim_row)

        # Power list
        self.powerTable = QtWidgets.QTableWidget(0, 2)
        self.powerTable.setHorizontalHeaderLabels(["Power", "Taken"])
        self.powerTable.horizontalHeader().setStretchLastSection(True)
        self.powerTable.verticalHeader().setVisible(False)
        self.powerTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.powerTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.powerTable.setMinimumWidth(150)

        power_layout.addWidget(power_plot_box, 2)
        power_layout.addWidget(voltage_plot_box, 2)
        power_layout.addWidget(self.powerTable, 1)

        tabs.addTab(live_tab, "Live")
        tabs.addTab(power_tab, "Power/Voltage")

        parent_layout.addWidget(tabs, 1)

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
        self.ndWheelAngleCombo.currentIndexChanged.connect(self._on_hwp_nd_angle_changed)
        self.hwpMinSpin.valueChanged.connect(self._on_hwp_nd_angle_changed)
        self.hwpMaxSpin.valueChanged.connect(self._on_hwp_nd_angle_changed)
        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpMaxSpin):
            w.setVisible(False)
        self.cropApplyBtn.clicked.connect(self.on_apply_crop)
        self.readoutRateCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.preampCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.homeBothBtn.clicked.connect(self.on_home_both_gates)
        self.frontApplyBtn.clicked.connect(self.on_front_apply)
        self.backApplyBtn.clicked.connect(self.on_back_apply)
        self.sweepVoltageChk.toggled.connect(self._update_gate_sweep_controls)
        self.sweepPowerChk.toggled.connect(self._sync_controls)
        self.angleListChk.toggled.connect(self._on_angle_list_changed)
        self.angleListEdit.textChanged.connect(self._on_angle_list_changed)
        self.sweepGateCombo.currentIndexChanged.connect(self._update_gate_sweep_controls)
        self.frontV0Spin.valueChanged.connect(self._update_vend_labels)
        self.frontStepSpin.valueChanged.connect(self._update_vend_labels)
        self.backV0Spin.valueChanged.connect(self._update_vend_labels)
        self.backStepSpin.valueChanged.connect(self._update_vend_labels)
        self.sweepStepsSpin.valueChanged.connect(self._update_vend_labels)
        self._update_vend_labels()
        self.voltageSlider.valueChanged.connect(self._on_voltage_slider)
        self.powerSlider.valueChanged.connect(self._on_power_slider)
        self.mapCminEdit.editingFinished.connect(self._on_map_clim_changed)
        self.mapCmaxEdit.editingFinished.connect(self._on_map_clim_changed)
        self.vmapCminEdit.editingFinished.connect(self._on_vmap_clim_changed)
        self.vmapCmaxEdit.editingFinished.connect(self._on_vmap_clim_changed)

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
        base = os.path.join(DATA_DIR, name)
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
        current_gain = self.preampCombo.currentText().strip()
        current_rate = self.readoutRateCombo.currentText().strip()
        _, rates, gains = [], [], []
        try:
            _, rates, gains = self.cam.get_amp_mode_choices()
        except Exception:
            pass
        if gains:
            self.preampCombo.blockSignals(True)
            self.preampCombo.clear()
            self.preampCombo.addItems(gains)
            if str(DEFAULT_PREAMP_GAIN) in gains:
                self.preampCombo.setCurrentText(str(DEFAULT_PREAMP_GAIN))
            elif current_gain and current_gain in gains:
                self.preampCombo.setCurrentText(current_gain)
            else:
                self.preampCombo.setCurrentIndex(0)
            self.preampCombo.blockSignals(False)
        if rates:
            self.readoutRateCombo.blockSignals(True)
            self.readoutRateCombo.clear()
            self.readoutRateCombo.addItems(rates)
            if str(DEFAULT_READOUT_RATE) in rates:
                self.readoutRateCombo.setCurrentText(str(DEFAULT_READOUT_RATE))
            elif current_rate and current_rate in rates:
                self.readoutRateCombo.setCurrentText(current_rate)
            else:
                self.readoutRateCombo.setCurrentIndex(0)
            self.readoutRateCombo.blockSignals(False)

    def _apply_amp_settings(self) -> None:
        if self.cam is None or self.cam.cam is None:
            return
        req_gain = self.preampCombo.currentText().strip()
        req_rate = self.readoutRateCombo.currentText().strip() or self._readout_rate_label
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
            rate_label = info.get("rate_label")
            if rate_label:
                self.readoutRateCombo.blockSignals(True)
                self.readoutRateCombo.setCurrentText(str(rate_label))
                self.readoutRateCombo.blockSignals(False)
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

    def _parse_angle_list(self) -> List[float]:
        """Parse the angle list text field into a sorted list of unique floats."""
        vals: List[float] = []
        for tok in self.angleListEdit.text().replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        seen: set = set()
        unique: List[float] = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def _on_angle_list_changed(self) -> None:
        angles = self._parse_angle_list()
        if angles:
            preview = f"{len(angles)} angles: {[f'{a:.3g}' for a in angles[:5]]}"
            if len(angles) > 5:
                preview += " …"
        else:
            preview = "(0 angles)"
        self.angleListPreviewLbl.setText(preview)
        self._sync_controls()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._is_busy():
            self._poll_timer.stop()
            self._gate_poll_timer.stop()
        else:
            if not self._poll_timer.isActive():
                self._poll_timer.start(int(POLL_MS))
            if not self._gate_poll_timer.isActive():
                self._gate_poll_timer.start(int(POLL_GATE_MS))
        self._sync_controls()

    def _sync_controls(self) -> None:
        cam_ok = self.cam.cam is not None
        stage_a_ok = self.stage is not None
        stage_b_ok = self.stage_b is not None
        spec_ok = self.cam.spec is not None
        busy = self._is_busy()
        hard_busy = busy and not self._aborting

        sweep_power = self.sweepPowerChk.isChecked()
        sweep_voltage = self.sweepVoltageChk.isChecked()
        need_dual = self._calib_mode in ("dual", "hwp_nd")

        angle_list_mode = sweep_power and self.angleListChk.isChecked()
        calib_ok = True
        stages_ok = True
        if sweep_power:
            if angle_list_mode:
                calib_ok = bool(self._parse_angle_list())
                stages_ok = stage_a_ok
            else:
                calib_ok = bool(self._dual_entries) if need_dual else self._calib_data is not None
                stages_ok = stage_a_ok and (stage_b_ok if need_dual else True)

        gate_ok = True
        if sweep_voltage:
            mode = self._sweep_gate_mode()
            gate_ok = True
            if mode in ("front", "dual"):
                gate_ok = gate_ok and (self.front_dev is not None)
            if mode in ("back", "dual"):
                gate_ok = gate_ok and (self.back_dev is not None)
        sweep_ok = sweep_power or sweep_voltage

        self.initBtn.setEnabled(not busy)
        self.disconnectBtn.setEnabled(not hard_busy)
        self.takeCurrentBtn.setEnabled(cam_ok and not busy)
        self.stopLiveBtn.setEnabled(self._threads_running() and self._live_active)
        self.sweepBtn.setEnabled(cam_ok and sweep_ok and stages_ok and calib_ok and gate_ok and not busy)
        self.abortBtn.setEnabled(busy)
        self.applySpecBtn.setEnabled(spec_ok and not busy)
        self.stageHomeBtn.setEnabled(stage_a_ok and not busy)
        self.stageMoveBtn.setEnabled(stage_a_ok and not busy)
        self.stageBHomeBtn.setEnabled(stage_b_ok and not busy)
        self.stageBMoveBtn.setEnabled(stage_b_ok and not busy)
        self.readoutRateCombo.setEnabled(cam_ok and not busy)
        self.preampCombo.setEnabled(cam_ok and not busy)
        self.seriesCombo.setEnabled(sweep_power and (not need_dual) and (not angle_list_mode) and calib_ok and not busy)
        self.powerSkipSpin.setEnabled(sweep_power and (not angle_list_mode) and not busy)
        self.angleListEdit.setEnabled(sweep_power and self.angleListChk.isChecked() and not busy)

        front_busy = self._front_ramp_thread is not None and self._front_ramp_thread.isRunning()
        back_busy = self._back_ramp_thread is not None and self._back_ramp_thread.isRunning()
        self.frontApplyBtn.setEnabled(self.front_dev is not None and (not hard_busy) and (not front_busy))
        self.backApplyBtn.setEnabled(self.back_dev is not None and (not hard_busy) and (not back_busy))
        either_gate = (self.front_dev is not None) or (self.back_dev is not None)
        self.homeBothBtn.setEnabled(either_gate and (not hard_busy) and (not front_busy) and (not back_busy))

    def _update_gate_sweep_controls(self) -> None:
        enabled = self.sweepVoltageChk.isChecked()
        self.sweepGateCombo.setEnabled(enabled)
        self.sweepStepsSpin.setEnabled(enabled)
        mode = self._sweep_gate_mode()
        self.frontV0Spin.setEnabled(enabled and mode in ("front", "dual"))
        self.frontStepSpin.setEnabled(enabled and mode in ("front", "dual"))
        self.backV0Spin.setEnabled(enabled and mode in ("back", "dual"))
        self.backStepSpin.setEnabled(enabled and mode in ("back", "dual"))
        self._sync_controls()

    def _current_voltage_key(self) -> Optional[float]:
        if not self._voltage_keys:
            return None
        idx = int(self.voltageSlider.value())
        idx = max(0, min(idx, len(self._voltage_keys) - 1))
        return self._voltage_keys[idx]

    def _current_power_key(self) -> Optional[float]:
        if not self._power_view_keys:
            return None
        idx = int(self.powerSlider.value())
        idx = max(0, min(idx, len(self._power_view_keys) - 1))
        return self._power_view_keys[idx]

    def _voltage_label_for_index(self, idx: int) -> str:
        if not self._voltage_keys:
            return "--"
        idx = max(0, min(idx, len(self._voltage_keys) - 1))
        if idx < len(self._voltage_labels):
            return self._voltage_labels[idx]
        key = self._voltage_keys[idx]
        return f"{key:.6g} V"

    def _power_label_for_index(self, idx: int) -> str:
        if (not self._power_view_keys) or (not self.sweepPowerChk.isChecked()):
            return "--"
        idx = max(0, min(idx, len(self._power_view_keys) - 1))
        key = self._power_view_keys[idx]
        if key is None or not np.isfinite(float(key)):
            return "--"
        if self._angle_list_mode:
            return f"{key:.3f} °"
        return f"{key:.6g} {self._calib_unit}"

    @staticmethod
    def _normalize_voltage_key(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return round(v, 9)

    def _voltage_label_for_key(self, key: Optional[float]) -> str:
        if key is None:
            return "--"
        try:
            k = float(key)
        except Exception:
            return "--"
        for idx, v in enumerate(self._voltage_keys):
            try:
                if abs(float(v) - k) <= 1e-9:
                    return self._voltage_label_for_index(idx)
            except Exception:
                continue
        return f"{k:.6g} V"

    def _update_voltage_slider(self) -> None:
        n = len(self._voltage_keys)
        if n == 0:
            self.voltageSlider.setRange(0, 0)
            self.voltageSlider.setEnabled(False)
            self.voltageSelLbl.setText("Voltage: --")
            return
        self.voltageSlider.setRange(0, max(0, n - 1))
        enable = self.sweepPowerChk.isChecked() and self.sweepVoltageChk.isChecked() and (n > 1)
        self.voltageSlider.setEnabled(enable)
        idx = min(int(self.voltageSlider.value()), n - 1)
        self.voltageSlider.setValue(idx)
        self.voltageSelLbl.setText(f"Voltage: {self._voltage_label_for_index(idx)}")

    def _update_power_slider(self) -> None:
        n = len(self._power_view_keys)
        if n == 0:
            self.powerSlider.setRange(0, 0)
            self.powerSlider.setEnabled(False)
            self.powerSelLbl.setText("Power: --")
            return
        self.powerSlider.setRange(0, max(0, n - 1))
        enable = self.sweepPowerChk.isChecked() and self.sweepVoltageChk.isChecked() and (n > 1)
        self.powerSlider.setEnabled(enable)
        idx = min(int(self.powerSlider.value()), n - 1)
        self.powerSlider.setValue(idx)
        self.powerSelLbl.setText(f"Power: {self._power_label_for_index(idx)}")

    def _on_voltage_slider(self, _value: int) -> None:
        self._update_voltage_slider()
        self._refresh_linecut_power_map()

    def _on_power_slider(self, _value: int) -> None:
        self._update_power_slider()
        self._refresh_linecut_voltage_map()

    @staticmethod
    def _parse_optional_float(text: str) -> Optional[float]:
        s = str(text).strip()
        if not s:
            return None
        try:
            v = float(s)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return float(v)

    def _apply_map_clim(self, data: np.ndarray, artist, cmin_edit: QtWidgets.QLineEdit, cmax_edit: QtWidgets.QLineEdit) -> None:
        try:
            auto_min = float(np.nanmin(data))
            auto_max = float(np.nanmax(data))
        except Exception:
            return
        if not np.isfinite(auto_min) or not np.isfinite(auto_max):
            return
        if auto_max <= auto_min:
            auto_max = auto_min + 1.0

        vmin_user = self._parse_optional_float(cmin_edit.text())
        vmax_user = self._parse_optional_float(cmax_edit.text())
        vmin = auto_min if vmin_user is None else vmin_user
        vmax = auto_max if vmax_user is None else vmax_user
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin, vmax = auto_min, auto_max
        artist.set_clim(float(vmin), float(vmax))

    def _on_map_clim_changed(self) -> None:
        self._refresh_linecut_power_map()

    def _on_vmap_clim_changed(self) -> None:
        self._refresh_linecut_voltage_map()

    def _entry_key(self, entry) -> Tuple:
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            return (
                "dual",
                power_key(entry.power_w, scale=self._calib_scale),
                float(entry.a_deg),
                float(entry.b_deg),
                int(getattr(entry, "index", 0)),
            )
        if entry.power_w is None:  # angle-list SimpleNamespace entries
            return (entry.series, entry.position_deg)
        return (entry.series, power_key(entry.power_w, scale=self._calib_scale), entry.position_deg)

    def _entry_rank(self, entry):
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            return int(getattr(entry, "index", 0))
        if not hasattr(entry, "is_base"):  # angle-list SimpleNamespace entries
            return 0
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

    def _sweep_gate_mode(self) -> str:
        text = self.sweepGateCombo.currentText().strip().lower()
        if "front" in text:
            return "front"
        if "back" in text:
            return "back"
        return "dual"

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
            "front_resource": self.frontResourceEdit.text().strip(),
            "back_resource": self.backResourceEdit.text().strip(),
            "front_vset": float(self.frontVsetSpin.value()),
            "back_vset": float(self.backVsetSpin.value()),
            "front_icomp_nA": float(self.frontIcompSpin.value()),
            "back_icomp_nA": float(self.backIcompSpin.value()),
            "front_output_on": bool(self._front_output_on),
            "back_output_on": bool(self._back_output_on),
            "sweep_power": bool(self.sweepPowerChk.isChecked()),
            "sweep_voltage": bool(self.sweepVoltageChk.isChecked()),
            "sweep_gate_mode": self._sweep_gate_mode(),
            "front_v0": float(self.frontV0Spin.value()),
            "front_step": float(self.frontStepSpin.value()),
            "back_v0": float(self.backV0Spin.value()),
            "back_step": float(self.backStepSpin.value()),
            "sweep_steps": int(self.sweepStepsSpin.value()),
            "power_skip_n": int(self.powerSkipSpin.value()),
        }

    def _reset_sweep_data(self) -> None:
        self._pv_images.clear()
        self._voltage_keys = []
        self._voltage_labels = []
        self._voltage_pairs = {}
        self._taken_powers.clear()
        self._taken_entries.clear()
        self._power_done_counts.clear()
        self._power_target_counts.clear()
        self._power_view_keys = list(self._power_list_keys)
        self._session_saved = False
        self._file_counters.clear()
        self._refresh_power_table()
        self._update_voltage_slider()
        self._update_power_slider()
        self._refresh_linecut_maps()

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

    def _disconnect_gate(self, which: str, reason: str) -> None:
        if which == "front":
            dev = self.front_dev
        else:
            dev = self.back_dev
        try:
            if dev is not None:
                try:
                    dev.set_output(False)
                except Exception:
                    pass
                dev.close()
        except Exception:
            pass
        if which == "front":
            self.front_dev = None
            self._front_output_on = False
            self._front_poll_failures = 0
            self.frontVreadVal.setText(READOUT_UNKNOWN)
            self.frontIreadVal.setText(READOUT_UNKNOWN)
            self.frontGateStatusLbl.setText(reason)
            self.frontOutputLbl.setText("Output: OFF")
            self._set_dot(self.frontGateDot, False)
        else:
            self.back_dev = None
            self._back_output_on = False
            self._back_poll_failures = 0
            self.backVreadVal.setText(READOUT_UNKNOWN)
            self.backIreadVal.setText(READOUT_UNKNOWN)
            self.backGateStatusLbl.setText(reason)
            self.backOutputLbl.setText("Output: OFF")
            self._set_dot(self.backGateDot, False)

    def _connect_gate_device(self, which: str, resource: str, *, quiet: bool = False) -> bool:
        resource = (resource or "").strip()
        if not resource:
            return False
        if self.rm is None:
            try:
                self.rm = make_resource_manager(VISA_DLL)
            except Exception as exc:
                if not quiet:
                    self.statusBar().showMessage(f"VISA init failed: {exc}")
                return False
        try:
            dev = KeithleySMU(self.rm, resource, timeout_ms=20000, query_delay_s=0.0, verbose=False)
            idn = dev.open()
            icomp_spin = self.frontIcompSpin if which == "front" else self.backIcompSpin
            try:
                dev.set_compliance(float(icomp_spin.value()) * 1e-9)
            except Exception:
                pass
            # Sync the Vset spinbox to the instrument's actual source voltage
            vset_spin = self.frontVsetSpin if which == "front" else self.backVsetSpin
            try:
                v_now = float(dev.query(":SOUR:VOLT?").strip())
                vset_spin.blockSignals(True)
                vset_spin.setValue(v_now)
                vset_spin.blockSignals(False)
            except Exception:
                pass
            output_on = False
            try:
                dev.set_output(True)
                output_on = True
            except Exception:
                output_on = False
            if which == "front":
                self.front_dev = dev
                self._front_output_on = output_on
                self._front_poll_failures = 0
                self.frontGateStatusLbl.setText((idn or "Connected")[:60])
                self.frontOutputLbl.setText("Output: ON" if output_on else "Output: OFF")
                self._set_dot(self.frontGateDot, True)
            else:
                self.back_dev = dev
                self._back_output_on = output_on
                self._back_poll_failures = 0
                self.backGateStatusLbl.setText((idn or "Connected")[:60])
                self.backOutputLbl.setText("Output: ON" if output_on else "Output: OFF")
                self._set_dot(self.backGateDot, True)
            return True
        except Exception as exc:
            if which == "front":
                self.front_dev = None
                self._front_output_on = False
                self.frontGateStatusLbl.setText("Conn failed" if not quiet else "Reconnect failed")
                self._set_dot(self.frontGateDot, False)
            else:
                self.back_dev = None
                self._back_output_on = False
                self.backGateStatusLbl.setText("Conn failed" if not quiet else "Reconnect failed")
                self._set_dot(self.backGateDot, False)
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Keithley connect", f"{which} gate: {exc}")
            return False

    def _auto_reconnect_gate(self, which: str) -> None:
        if not self._gate_auto_reconnect_armed:
            return
        now = time.monotonic()
        if which == "front":
            if self.front_dev is not None:
                return
            if now < float(self._front_reconnect_next_ts):
                return
            ok = self._connect_gate_device("front", self.frontResourceEdit.text(), quiet=True)
            self._front_reconnect_next_ts = now + 3.0
            if ok:
                self.statusBar().showMessage("Front gate reconnected")
        else:
            if self.back_dev is not None:
                return
            if now < float(self._back_reconnect_next_ts):
                return
            ok = self._connect_gate_device("back", self.backResourceEdit.text(), quiet=True)
            self._back_reconnect_next_ts = now + 3.0
            if ok:
                self.statusBar().showMessage("Back gate reconnected")

    def _apply_gate_readout(self, which: str, v: Optional[float], iA: Optional[float]) -> None:
        if v is None or iA is None:
            return
        if which == "front":
            self.frontVreadVal.setText(f"{v:.6g}")
            self.frontIreadVal.setText(f"{iA*1e9:.6g}")
            self.frontGateStatusLbl.setText("Connected")
            self.frontOutputLbl.setText("Output: ON" if self._front_output_on else "Output: OFF")
            self.frontPlot.add_point(v, iA)
            self._front_poll_failures = 0
            self._set_dot(self.frontGateDot, True)
        else:
            self.backVreadVal.setText(f"{v:.6g}")
            self.backIreadVal.setText(f"{iA*1e9:.6g}")
            self.backGateStatusLbl.setText("Connected")
            self.backOutputLbl.setText("Output: ON" if self._back_output_on else "Output: OFF")
            self.backPlot.add_point(v, iA)
            self._back_poll_failures = 0
            self._set_dot(self.backGateDot, True)

    def _poll_gates(self) -> None:
        if self._gate_threads_running():
            return

        for which, dev in (("front", self.front_dev), ("back", self.back_dev)):
            if dev is None:
                self._auto_reconnect_gate(which)
                dev = self.front_dev if which == "front" else self.back_dev
            if dev is None:
                if which == "front":
                    self._set_dot(self.frontGateDot, False)
                    self.frontGateStatusLbl.setText("Disconnected")
                    self.frontOutputLbl.setText("Output: OFF" if not self._front_output_on else "Output: ON")
                else:
                    self._set_dot(self.backGateDot, False)
                    self.backGateStatusLbl.setText("Disconnected")
                    self.backOutputLbl.setText("Output: OFF" if not self._back_output_on else "Output: ON")
                continue

            try:
                vi = dev.read_vi_with_timeout(POLL_TIMEOUT_MS)
                v = float(vi.v)
                iA = float(vi.i)
                self._apply_gate_readout(which, v, iA)
            except Exception:
                if which == "front":
                    self._front_poll_failures += 1
                    self.frontGateStatusLbl.setText("Poll error")
                    if self._front_poll_failures >= POLL_ERROR_LIMIT:
                        self._disconnect_gate("front", "Device not responding")
                else:
                    self._back_poll_failures += 1
                    self.backGateStatusLbl.setText("Poll error")
                    if self._back_poll_failures >= POLL_ERROR_LIMIT:
                        self._disconnect_gate("back", "Device not responding")

    # -----------------
    # Calibration
    # -----------------
    def on_browse_calib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select calibration file",
            DATA_DIR,
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.calibPathEdit.setText(path)

    def on_load_calib(self):
        path = self.calibPathEdit.text().strip()
        try:
            if is_hwp_nd_calibration(path):
                data = load_hwp_nd_calibration(path)
                self._load_hwp_nd_calibration(data)
            elif is_dual_wheel_calibration(path):
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
        self._hwp_nd_calib_data = None
        self._dual_entries = []
        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpMaxSpin):
            w.setVisible(False)
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
        self._hwp_nd_calib_data = None
        self._calib_data = None
        self._series_entries.clear()
        self._dual_entries = list(data.entries)
        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpMaxSpin):
            w.setVisible(False)
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

    def _load_hwp_nd_calibration(self, data: HWPNDCalibData) -> None:
        self._calib_mode = "hwp_nd"
        self._hwp_nd_calib_data = data
        self._dual_calib_data = None
        self._calib_data = None
        self._series_entries.clear()

        # Populate ND angle selector
        self.ndWheelAngleCombo.blockSignals(True)
        self.ndWheelAngleCombo.clear()
        for a in data.nd_angles:
            self.ndWheelAngleCombo.addItem(f"{a:.2f}°")
        self.ndWheelAngleCombo.blockSignals(False)
        # Set HWP angle limits to full range of measured angles
        if data.hwp_angles:
            lo, hi = min(data.hwp_angles), max(data.hwp_angles)
            self.hwpMinSpin.blockSignals(True)
            self.hwpMaxSpin.blockSignals(True)
            self.hwpMinSpin.setValue(lo)
            self.hwpMaxSpin.setValue(hi)
            self.hwpMinSpin.blockSignals(False)
            self.hwpMaxSpin.blockSignals(False)

        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpMaxSpin):
            w.setVisible(True)

        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItem("hwp_nd")
        self.seriesCombo.setCurrentIndex(0)
        self.seriesCombo.blockSignals(False)

        self.calibModeLbl.setText(
            f"Mode: HWP+ND  |  {len(data.hwp_angles)} HWP steps"
            + (f", {len(data.nd_angles)} ND angles" if data.nd_angles else "")
        )

        # Compute entries for first ND angle (or default to first)
        self._apply_hwp_nd_angle()

    def _on_hwp_nd_angle_changed(self) -> None:
        if self._calib_mode == "hwp_nd":
            self._apply_hwp_nd_angle()

    def _apply_hwp_nd_angle(self) -> None:
        data = self._hwp_nd_calib_data
        if data is None:
            return
        idx = self.ndWheelAngleCombo.currentIndex()
        if idx < 0:
            return
        nd_angle = data.nd_angles[idx] if data.nd_angles else 0.0
        entries, unit, scale = compute_hwp_nd_entries(data, nd_angle)

        # Apply HWP angle range filter
        lo = self.hwpMinSpin.value()
        hi = self.hwpMaxSpin.value()
        entries = [e for e in entries if lo - 1e-9 <= e.a_deg <= hi + 1e-9]

        self._dual_entries = entries
        self._calib_unit = unit
        self._calib_scale = scale

        self._build_power_display()
        self._total_entries = len(entries)
        self._reset_sweep_data()
        self._update_series_info()
        self._sync_controls()

    def _build_power_display(self):
        self._power_display.clear()
        if self._calib_mode in ("dual", "hwp_nd"):
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
        self._power_view_keys = list(self._power_list_keys)
        self._refresh_power_table()
        self._update_power_slider()

    def _update_series_info(self):
        if self._calib_mode in ("dual", "hwp_nd"):
            n = len(self._dual_entries)
            label = "hwp_nd" if self._calib_mode == "hwp_nd" else "dual_wheel"
            self.seriesInfoLbl.setText(f"{label} ({n} entries)")
            return
        series = self.seriesCombo.currentText().strip()
        n = len(self._series_entries.get(series, []))
        self.seriesInfoLbl.setText(f"{series} ({n} powers)")

    def _refresh_power_table(self):
        unit = self._calib_unit
        self.powerTable.setRowCount(len(self._power_list_keys))
        self.powerTable.setHorizontalHeaderLabels([f"Power ({unit})", "Taken"])
        for row, key in enumerate(self._power_list_keys):
            if self.sweepPowerChk.isChecked():
                power_text = f"{key:.6g}"
            else:
                power_text = "--"
            power_item = QtWidgets.QTableWidgetItem(power_text)
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
        self._refresh_linecut_maps()
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
            rate_label = info.get("rate_label")
            if rate_label:
                self.readoutRateCombo.blockSignals(True)
                self.readoutRateCombo.setCurrentText(str(rate_label))
                self.readoutRateCombo.blockSignals(False)

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

        self._gate_auto_reconnect_armed = True
        self._front_reconnect_next_ts = 0.0
        self._back_reconnect_next_ts = 0.0
        self._connect_gate_device("front", self.frontResourceEdit.text())
        self._connect_gate_device("back", self.backResourceEdit.text())

        self.statusBar().showMessage("Initialized")
        self._sync_controls()

    def on_disconnect(self):
        self.on_abort()
        self._gate_auto_reconnect_armed = False
        self._disconnect_gate("front", "Disconnected")
        self._disconnect_gate("back", "Disconnected")
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

    def _on_gate_ramp_progress(self, which: str, v: float, iA: float) -> None:
        if which == "front":
            self.frontVreadVal.setText(f"{v:.6g}")
            self.frontIreadVal.setText(f"{iA*1e9:.6g}")
            self.frontPlot.add_point(v, iA)
        else:
            self.backVreadVal.setText(f"{v:.6g}")
            self.backIreadVal.setText(f"{iA*1e9:.6g}")
            self.backPlot.add_point(v, iA)

    def _on_gate_ramp_done(self, which: str, status: str) -> None:
        if which == "front":
            self._front_ramp_thread = None
            if status == "ok":
                self.frontGateStatusLbl.setText("Applied (ramped)")
            elif status == "aborted":
                self.frontGateStatusLbl.setText("Ramp aborted")
            else:
                self.frontGateStatusLbl.setText(status)
        else:
            self._back_ramp_thread = None
            if status == "ok":
                self.backGateStatusLbl.setText("Applied (ramped)")
            elif status == "aborted":
                self.backGateStatusLbl.setText("Ramp aborted")
            else:
                self.backGateStatusLbl.setText(status)
        self._sync_controls()

    def _start_gate_ramp(self, which: str, target_v: Optional[float] = None) -> None:
        dev = self.front_dev if which == "front" else self.back_dev
        if dev is None:
            if self._gate_auto_reconnect_armed:
                self._auto_reconnect_gate(which)
                dev = self.front_dev if which == "front" else self.back_dev
        if dev is None:
            QtWidgets.QMessageBox.warning(self, "Gate", f"{which} gate not connected.")
            return

        vset = target_v if target_v is not None else float(self.frontVsetSpin.value() if which == "front" else self.backVsetSpin.value())
        icomp_nA = float(self.frontIcompSpin.value() if which == "front" else self.backIcompSpin.value())
        icomp_a = icomp_nA * 1e-9

        controller = SweepController()
        thread = GateRampThread(dev, vset, icomp_a, controller, RAMP_STEP_V, RAMP_DWELL_S)
        if which == "front":
            self._front_ramp_controller = controller
            self._front_ramp_thread = thread
            thread.progress.connect(lambda v, i: self._on_gate_ramp_progress("front", v, i))
            thread.status.connect(lambda msg: self.frontGateStatusLbl.setText(msg[:60]))
            thread.done.connect(lambda status: self._on_gate_ramp_done("front", status))
            self._front_output_on = True
        else:
            self._back_ramp_controller = controller
            self._back_ramp_thread = thread
            thread.progress.connect(lambda v, i: self._on_gate_ramp_progress("back", v, i))
            thread.status.connect(lambda msg: self.backGateStatusLbl.setText(msg[:60]))
            thread.done.connect(lambda status: self._on_gate_ramp_done("back", status))
            self._back_output_on = True

        thread.start()
        self._sync_controls()

    def _update_vend_labels(self) -> None:
        n = int(self.sweepStepsSpin.value())
        front_end = float(self.frontV0Spin.value()) + (n - 1) * float(self.frontStepSpin.value())
        back_end = float(self.backV0Spin.value()) + (n - 1) * float(self.backStepSpin.value())
        self.frontVendLbl.setText(f"End: {front_end:.6g} V")
        self.backVendLbl.setText(f"End: {back_end:.6g} V")

    def on_home_both_gates(self):
        if self.front_dev is not None:
            self._start_gate_ramp("front", target_v=0.0)
        if self.back_dev is not None:
            self._start_gate_ramp("back", target_v=0.0)

    def on_front_apply(self):
        self._start_gate_ramp("front")

    def on_back_apply(self):
        self._start_gate_ramp("back")

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

        sweep_power = self.sweepPowerChk.isChecked()
        sweep_voltage = self.sweepVoltageChk.isChecked()
        if not (sweep_power or sweep_voltage):
            QtWidgets.QMessageBox.warning(self, "Sweep", "Enable sweep power and/or sweep voltage.")
            return
        if self.cam.cam is None:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Camera must be connected.")
            return

        self._reset_sweep_data()

        mode = self._sweep_gate_mode()
        nsteps = int(self.sweepStepsSpin.value())
        skip_n = int(self.powerSkipSpin.value())
        power_stride = max(1, skip_n + 1)
        front_v0 = float(self.frontV0Spin.value())
        front_step = float(self.frontStepSpin.value())
        back_v0 = float(self.backV0Spin.value())
        back_step = float(self.backStepSpin.value())

        if sweep_voltage:
            if mode in ("front", "dual") and self.front_dev is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Front gate must be connected.")
                return
            if mode in ("back", "dual") and self.back_dev is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Back gate must be connected.")
                return
            if nsteps < 1:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Voltage steps must be >= 1.")
                return

            def _check_v_limits(v0: float, step: float, n: int, label: str):
                vend = v0 + (n - 1) * step
                if abs(v0) > KEITHLEY_V_LIMIT or abs(vend) > KEITHLEY_V_LIMIT:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Voltage limit",
                        f"{label} start/end exceed +/-{KEITHLEY_V_LIMIT:.0f} V",
                    )
                    return False
                return True

            if mode in ("front", "dual") and not _check_v_limits(front_v0, front_step, nsteps, "Front gate"):
                return
            if mode in ("back", "dual") and not _check_v_limits(back_v0, back_step, nsteps, "Back gate"):
                return

        self._angle_list_mode = sweep_power and self.angleListChk.isChecked()

        plan = []
        entries: List[PowerCalibEntry] = []
        if sweep_power and self._angle_list_mode:
            angles = self._parse_angle_list()
            if not angles:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Enter at least one angle in the angle list.")
                return
            if self.stage is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Stage A must be connected for angle list sweep.")
                return
            for angle in angles:
                a_label = f"{angle:.3f}".replace(".", "p").replace("-", "m")
                base = f"PL_A{a_label}deg"
                entry = _types.SimpleNamespace(position_deg=angle, power_w=None, series="angle_list")
                plan.append({"entry": entry, "base": base})
            # use angles as pseudo-power keys for map display
            self._power_target_counts = {a: 1 for a in angles}
            self._power_done_counts = {a: 0 for a in angles}
            self._total_entries = len(angles)
            self._power_view_keys = list(angles)
        elif sweep_power:
            if self._calib_mode in ("dual", "hwp_nd"):
                if not self._dual_entries:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "Load a calibration file first.")
                    return
                if self.stage is None or self.stage_b is None:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "Both stages must be connected for dual sweep.")
                    return
                entries = list(self._dual_entries)
                if power_stride > 1:
                    entries = entries[::power_stride]
                if not entries:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "No entries in dual-wheel calibration file.")
                    return
                for entry in entries:
                    power_val = power_key(entry.power_w, scale=self._calib_scale)
                    power_label = f"{power_val:.6g}".replace(".", "p")
                    a_label = f"{entry.a_deg:.3f}".replace(".", "p")
                    b_label = f"{entry.b_deg:.3f}".replace(".", "p")
                    base = f"PL_A{a_label}_B{b_label}_P{power_label}{self._calib_unit}"
                    plan.append({"entry": entry, "base": base})
            else:
                if self._calib_data is None:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "Load a power calibration file first.")
                    return
                if self.stage is None:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "Stage must be connected.")
                    return
                series = self.seriesCombo.currentText().strip()
                entries = sorted(self._series_entries.get(series, []), key=lambda e: e.power_w)
                if power_stride > 1:
                    entries = entries[::power_stride]
                if not entries:
                    QtWidgets.QMessageBox.warning(self, "Sweep", f"No entries for series '{series}'.")
                    return
                for entry in entries:
                    power_val = power_key(entry.power_w, scale=self._calib_scale)
                    power_label = f"{power_val:.6g}".replace(".", "p")
                    series_label = self._safe_series_label(entry.series)
                    base = f"PL_{series_label}_{power_label}{self._calib_unit}"
                    plan.append({"entry": entry, "base": base})
        else:
            plan = [{"entry": None, "base": "PL_manual"}]
            self._power_view_keys = [0.0]

        if not self._angle_list_mode:
            self._power_target_counts.clear()
            if sweep_power:
                for entry in entries:
                    key = power_key(entry.power_w, scale=self._calib_scale)
                    self._power_target_counts[key] = self._power_target_counts.get(key, 0) + 1
            self._power_done_counts = {k: 0 for k in self._power_target_counts}
            self._total_entries = len(entries) if sweep_power else 0
            if sweep_power:
                self._power_view_keys = sorted(self._power_target_counts.keys())


        self._voltage_keys = []
        self._voltage_labels = []
        self._voltage_pairs = {}
        if sweep_voltage:
            for k in range(nsteps):
                vf = front_v0 + k * front_step if mode in ("front", "dual") else None
                vb = back_v0 + k * back_step if mode in ("back", "dual") else None
                key_raw = vf if mode in ("front", "dual") else vb
                key = self._normalize_voltage_key(key_raw)
                if key is None:
                    continue
                if mode == "dual":
                    label = f"Vf={vf:.6g}, Vb={vb:.6g} V"
                elif mode == "front":
                    label = f"{vf:.6g} V"
                else:
                    label = f"{vb:.6g} V"
                if key in self._voltage_pairs:
                    continue
                self._voltage_keys.append(key)
                self._voltage_labels.append(label)
                self._voltage_pairs[key] = (vf, vb)
        else:
            self._voltage_keys = [0.0]
            self._voltage_labels = ["--"]
            self._voltage_pairs = {0.0: (None, None)}

        self._update_voltage_slider()
        self._update_power_slider()

        self._prepare_single_mode()
        self._resolve_save_dir()

        meta = self._build_metadata_base()
        wl_axis = self.liveView.get_wavelength_axis() if self.liveView is not None else None

        front_icomp_a = float(self.frontIcompSpin.value()) * 1e-9 if sweep_voltage and mode in ("front", "dual") else None
        back_icomp_a = float(self.backIcompSpin.value()) * 1e-9 if sweep_voltage and mode in ("back", "dual") else None

        self._sweep_thread = PowerVoltageSweepThread(
            self.cam,
            self.stage,
            self.stage_b,
            plan,
            int(self.accumSpin.value()),
            float(self.exposureSpin.value()),
            float(DEFAULT_RAMP_STEP_DEG),
            meta,
            wl_axis,
            sweep_power=sweep_power,
            sweep_voltage=sweep_voltage,
            sweep_mode=mode,
            front_dev=self.front_dev,
            back_dev=self.back_dev,
            front_v0=front_v0,
            front_step=front_step,
            back_v0=back_v0,
            back_step=back_step,
            nsteps=nsteps,
            front_icomp_a=front_icomp_a,
            back_icomp_a=back_icomp_a,
            ramp_step_v=float(RAMP_STEP_V),
            ramp_dwell_s=float(RAMP_DWELL_S),
            settle_ms=float(SWEEP_SETTLE_MS),
            zero_v_eps=float(ZERO_V_EPS),
            zero_v_extra_settle_ms=float(ZERO_V_EXTRA_SETTLE_MS),
            save_dir=self._save_dir,
            readout_rate=self._readout_rate_label,
            preamp_gain=self._preamp_gain_label,
            output_amp=self._output_amp_label,
        )
        self._sweep_thread.frame_ready.connect(self.on_sweep_frame)
        self._sweep_thread.vi_ready.connect(self.on_sweep_vi)
        self._sweep_thread.point_done.connect(self.on_sweep_point)
        self._sweep_thread.power_done.connect(self.on_sweep_power_done)
        self._sweep_thread.status.connect(self.statusBar().showMessage)
        self._sweep_thread.done.connect(self.on_sweep_done)
        self._sweep_thread.start()
        self._set_busy(True)
        self.statusBar().showMessage("Sweeping...")

    def on_abort(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
        if self._accum_thread is not None and self._accum_thread.isRunning():
            self._accum_thread.stop()
        if self._sweep_thread is not None and self._sweep_thread.isRunning():
            self._sweep_thread.stop()
        if self._front_ramp_controller is not None:
            try:
                self._front_ramp_controller.abort()
            except Exception:
                pass
        if self._back_ramp_controller is not None:
            try:
                self._back_ramp_controller.abort()
            except Exception:
                pass
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
        self._aborting = True
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

    def on_sweep_vi(self, payload: dict):
        v_front = payload.get("v_front")
        i_front = payload.get("i_front")
        v_back = payload.get("v_back")
        i_back = payload.get("i_back")
        if v_front is not None and i_front is not None:
            self._apply_gate_readout("front", float(v_front), float(i_front))
        if v_back is not None and i_back is not None:
            self._apply_gate_readout("back", float(v_back), float(i_back))

    def _on_live_done(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            return
        self._live_thread = None
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Live stopped")

    def on_sweep_point(self, payload: dict):
        entry = payload.get("entry")
        raw = payload.get("image")
        if raw is None:
            return

        if self.liveView is None:
            return

        power_key_val = 0.0
        if entry is not None:
            try:
                if self._angle_list_mode and hasattr(entry, "position_deg"):
                    power_key_val = float(entry.position_deg)
                else:
                    power_key_val = power_key(entry.power_w, scale=self._calib_scale)
            except Exception:
                power_key_val = 0.0

        voltage_key = self._normalize_voltage_key(payload.get("voltage_key"))
        if voltage_key is None:
            voltage_key = 0.0

        if voltage_key not in self._voltage_keys:
            self._voltage_keys.append(voltage_key)
            self._voltage_labels.append(f"{voltage_key:.6g} V")
            self._update_voltage_slider()

        try:
            raw_store = np.asarray(raw).copy()
        except Exception:
            raw_store = np.asarray(raw)
        rank = self._entry_rank(entry) if entry is not None else 0
        entry_map = self._pv_images.setdefault(power_key_val, {})
        cur = entry_map.get(voltage_key)
        if cur is None or rank <= cur.get("rank", rank):
            entry_map[voltage_key] = {"rank": rank, "raw": raw_store}

        self._refresh_linecut_maps()

    def on_sweep_power_done(self, payload: dict):
        entry = payload.get("entry")
        if entry is None:
            return

        if self._angle_list_mode and hasattr(entry, "position_deg"):
            key = float(entry.position_deg)
        else:
            key = power_key(entry.power_w, scale=self._calib_scale)
        self._taken_entries.add(self._entry_key(entry))
        count = self._power_done_counts.get(key, 0) + 1
        self._power_done_counts[key] = count
        if count >= self._power_target_counts.get(key, 1):
            self._mark_power_taken(key)

        if self._total_entries and len(self._taken_entries) >= self._total_entries:
            self._write_session_metadata()

    def on_sweep_done(self, status: str):
        self._aborting = False
        self._set_busy(False)
        self._sweep_thread = None
        if status == "ok":
            self.statusBar().showMessage("Sweep complete")
        elif status == "aborted":
            self.statusBar().showMessage("Sweep aborted")
        elif status == "trip":
            self.statusBar().showMessage("Sweep TRIPPED (current limit exceeded)")
        else:
            self.statusBar().showMessage(str(status))

    # -----------------
    # Image handling
    # -----------------
    def _prepare_display_image(self, raw):
        disp = np.fliplr(np.flipud(np.asarray(raw)))
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

    def _display_image_from_entry(self, info: Optional[dict]) -> Optional[np.ndarray]:
        if not info:
            return None
        raw = info.get("raw")
        if raw is not None:
            try:
                return self._prepare_display_image(raw)
            except Exception:
                return None
        img = info.get("image")
        if img is None:
            return None
        try:
            arr = np.asarray(img)
        except Exception:
            return None
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr

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
        self._refresh_linecut_maps()

    def _refresh_linecut_maps(self) -> None:
        self._refresh_linecut_power_map()
        self._refresh_linecut_voltage_map()

    def _refresh_linecut_power_map(self) -> None:
        if self.liveView is None:
            return
        if not self._power_view_keys or not self._pv_images:
            self.map_artist.set_data(np.zeros((10, 10)))
            self.canvas_map.draw_idle()
            return

        selected_voltage_key = self._normalize_voltage_key(self._current_voltage_key())
        voltage_key = selected_voltage_key

        def _find_first_image(vkey: Optional[float]):
            if vkey is None:
                return None
            for pkey in self._power_view_keys:
                entry = self._pv_images.get(pkey, {})
                img = self._display_image_from_entry(entry.get(vkey) if entry else None)
                if img is not None:
                    return img
            return None

        first_image = _find_first_image(voltage_key)
        if first_image is None:
            for vkey in self._voltage_keys:
                first_image = _find_first_image(vkey)
                if first_image is not None:
                    voltage_key = vkey
                    break

        if first_image is None:
            self.map_artist.set_data(np.zeros((10, 10)))
            self.canvas_map.draw_idle()
            return

        _, width = first_image.shape
        data = np.full((width, len(self._power_view_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return

        for col, power_key in enumerate(self._power_view_keys):
            info = self._pv_images.get(power_key, {}).get(voltage_key)
            if not info:
                continue
            img = self._display_image_from_entry(info)
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        powers = self._power_view_keys
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

        x_label = "Stage A angle (°)" if self._angle_list_mode else (f"Power ({self._calib_unit})" if self._calib_unit else "Power")
        title = f"Linecut vs {x_label.split(' (')[0]}"
        vlabel = self._voltage_label_for_key(voltage_key)
        if vlabel and vlabel != "--":
            title = f"{title} at {vlabel}"

        self.map_artist.set_data(data)
        self.map_artist.set_extent((pmin, pmax, y0, y1))
        self.ax_map.set_title(title)
        self.ax_map.set_xlabel(x_label)
        self.ax_map.set_ylabel(ylabel)
        self._apply_map_clim(data, self.map_artist, self.mapCminEdit, self.mapCmaxEdit)
        self.canvas_map.draw_idle()

    def _refresh_linecut_voltage_map(self) -> None:
        if self.liveView is None:
            return
        if not self._voltage_keys or not self._pv_images:
            self.vmap_artist.set_data(np.zeros((10, 10)))
            self.canvas_vmap.draw_idle()
            return

        power_key = self._current_power_key()
        if power_key is None and self._power_view_keys:
            power_key = self._power_view_keys[0]
        if power_key is None:
            self.vmap_artist.set_data(np.zeros((10, 10)))
            self.canvas_vmap.draw_idle()
            return

        entry = self._pv_images.get(power_key, {})
        first_image = None
        for vkey in self._voltage_keys:
            first_image = self._display_image_from_entry(entry.get(vkey) if entry else None)
            if first_image is not None:
                break
        if first_image is None:
            for pkey in self._power_view_keys:
                candidate = self._pv_images.get(pkey, {})
                for vkey in self._voltage_keys:
                    first_image = self._display_image_from_entry(candidate.get(vkey) if candidate else None)
                    if first_image is not None:
                        power_key = pkey
                        entry = candidate
                        break
                if first_image is not None:
                    break
        if first_image is None:
            self.vmap_artist.set_data(np.zeros((10, 10)))
            self.canvas_vmap.draw_idle()
            return

        _, width = first_image.shape
        data = np.full((width, len(self._voltage_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return

        for col, vkey in enumerate(self._voltage_keys):
            info = entry.get(vkey)
            if not info:
                continue
            img = self._display_image_from_entry(info)
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        vmin = float(self._voltage_keys[0]) if self._voltage_keys else 0.0
        vmax = float(self._voltage_keys[-1]) if self._voltage_keys else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        x_axis = self.liveView.get_linecut_axis()
        if x_axis is not None and len(x_axis) == width:
            y0 = float(x_axis[0])
            y1 = float(x_axis[-1])
            ylabel = "Wavelength (nm)" if self.liveView.has_wavelength_axis() else "Pixel"
        else:
            y0 = 0.0
            y1 = float(width)
            ylabel = "Pixel"

        power_label = self._format_power_key(float(power_key))
        title = "Linecut vs Voltage"
        if power_label and power_label != "--" and self.sweepPowerChk.isChecked():
            title = f"Linecut vs Voltage at {power_label}"

        self.vmap_artist.set_data(data)
        self.vmap_artist.set_extent((vmin, vmax, y0, y1))
        self.ax_vmap.set_title(title)
        self.ax_vmap.set_xlabel("Voltage (V)")
        self.ax_vmap.set_ylabel(ylabel)
        self._apply_map_clim(data, self.vmap_artist, self.vmapCminEdit, self.vmapCmaxEdit)
        self.canvas_vmap.draw_idle()

    # -----------------
    # Session metadata
    # -----------------
    def _write_session_metadata(self) -> None:
        if self._session_saved or not self._save_dir:
            return
        path = os.path.join(self._save_dir, "session.md")
        meta = self._build_metadata_base()
        lines = [
            "# PL power/voltage sweep session",
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
            "## Gates",
            f"- Front resource: {meta.get('front_resource', '')}",
            f"- Back resource: {meta.get('back_resource', '')}",
            f"- Front Vset (V): {meta.get('front_vset', '')}",
            f"- Back Vset (V): {meta.get('back_vset', '')}",
            f"- Front Icomp (nA): {meta.get('front_icomp_nA', '')}",
            f"- Back Icomp (nA): {meta.get('back_icomp_nA', '')}",
            f"- Front output: {meta.get('front_output_on', '')}",
            f"- Back output: {meta.get('back_output_on', '')}",
            "",
            "## Sweep",
            f"- Sweep power: {meta.get('sweep_power', '')}",
            f"- Sweep voltage: {meta.get('sweep_voltage', '')}",
            f"- Gate mode: {meta.get('sweep_gate_mode', '')}",
            f"- Front V0/step: {meta.get('front_v0', '')}, {meta.get('front_step', '')}",
            f"- Skip N power points: {meta.get('power_skip_n', 0)}",
            f"- Back V0/step: {meta.get('back_v0', '')}, {meta.get('back_step', '')}",
            f"- Steps: {meta.get('sweep_steps', '')}",
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

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from PM100A.pm100a_wrapper import PM100A, PM100AError
from rot.rot_wrapper import RotationStage, list_kinesis_serials

from measurements.wheel_hwp_power_calib_config import (
    DATA_DIR,
    DEFAULT_HWP_FIXED_ANGLE,
    DEFAULT_HWP_SERIAL,
    DEFAULT_HWP_START,
    DEFAULT_HWP_STEP,
    DEFAULT_HWP_STOP,
    DEFAULT_N_READINGS,
    DEFAULT_ND_ANGLE,
    DEFAULT_ND_SERIAL,
    DEFAULT_ND_START,
    DEFAULT_ND_STEP,
    DEFAULT_ND_STOP,
    DEFAULT_PM_AVERAGING,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_STAGE_SCALE,
)
from measurements.wheel_hwp_power_calib_workers import StageSweepWorker

BTN_W = 90


# ------------------------------------------------------------------ #
# Background move worker                                               #
# ------------------------------------------------------------------ #

class _MoveWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(str)   # "ok" or "error: <msg>"

    def __init__(self, stage, target_deg: float, ramp_step_deg: float, parent=None):
        super().__init__(parent)
        self._stage      = stage
        self._target     = target_deg
        self._ramp_step  = ramp_step_deg
        self._controller = None

    def stop(self) -> None:
        if self._controller is not None:
            self._controller.abort()

    def run(self) -> None:
        from rot.rot_wrapper import MotionController
        self._controller = MotionController()
        try:
            self._stage.move_to(self._target, step_deg=self._ramp_step,
                                controller=self._controller)
            self.done.emit("ok")
        except Exception as exc:
            self.done.emit(f"error: {exc}")


# ------------------------------------------------------------------ #
# Main widget                                                          #
# ------------------------------------------------------------------ #

class WheelHWPCalibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._nd_stage:  Optional[RotationStage]   = None
        self._hwp_stage: Optional[RotationStage]   = None
        self._pm:        Optional[PM100A]           = None
        self._worker:    Optional[StageSweepWorker] = None
        self._nd_mover:  Optional[_MoveWorker]      = None
        self._hwp_mover: Optional[_MoveWorker]      = None

        self._running    = False
        self._sweep_type: Optional[str] = None   # "hwp" or "nd"
        self._queue_nd   = False                 # chain ND after HWP in "Run both"

        # HWP sweep results
        self._hwp_angles: List[float] = []
        self._hwp_powers: List[float] = []
        self._hwp_stds:   List[float] = []

        # ND sweep results
        self._nd_angles: List[float] = []
        self._nd_powers: List[float] = []
        self._nd_stds:   List[float] = []

        self._pos_timer = QtCore.QTimer(self)
        self._pos_timer.setInterval(500)
        self._pos_timer.timeout.connect(self._update_positions)
        self._pos_timer.start()

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)
        root.addLayout(left, stretch=0)

        left.addWidget(self._build_connection_box())
        left.addWidget(self._build_stage_control_box())
        left.addWidget(self._build_pm_box())
        left.addWidget(self._build_sweep_params_box())
        left.addWidget(self._build_action_row())
        left.addWidget(self._build_progress_row())
        left.addStretch()

        self._right_tabs = QtWidgets.QTabWidget()
        self._right_tabs.addTab(self._build_hwp_tab(),      "HWP calibration")
        self._right_tabs.addTab(self._build_nd_tab(),       "ND calibration")
        self._right_tabs.addTab(self._build_analysis_tab(), "Analysis")
        root.addWidget(self._right_tabs, stretch=1)

    # --- Devices box ---

    def _build_connection_box(self) -> QtWidgets.QGroupBox:
        box    = QtWidgets.QGroupBox("Devices")
        layout = QtWidgets.QGridLayout(box)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(8)

        def _status_lbl():
            lbl = QtWidgets.QLabel("Disconnected")
            lbl.setFont(mono)
            return lbl

        def _serial_combo(default):
            cb = QtWidgets.QComboBox()
            cb.setEditable(True)
            cb.setMinimumWidth(110)
            cb.addItem(default)
            cb.setCurrentText(default)
            return cb

        # ND row
        self._ndCombo     = _serial_combo(DEFAULT_ND_SERIAL)
        self._ndScanBtn   = QtWidgets.QPushButton("Scan");       self._ndScanBtn.setFixedWidth(50)
        self._ndConnBtn   = QtWidgets.QPushButton("Connect");    self._ndConnBtn.setFixedWidth(BTN_W)
        self._ndDiscBtn   = QtWidgets.QPushButton("Disconnect"); self._ndDiscBtn.setFixedWidth(BTN_W); self._ndDiscBtn.setEnabled(False)
        self._ndStatusLbl = _status_lbl()
        layout.addWidget(QtWidgets.QLabel("ND wheel:"), 0, 0)
        layout.addWidget(self._ndCombo,                 0, 1)
        layout.addWidget(self._ndScanBtn,               0, 2)
        layout.addWidget(self._ndConnBtn,               0, 3)
        layout.addWidget(self._ndDiscBtn,               0, 4)
        layout.addWidget(self._ndStatusLbl,             0, 5)

        # HWP row
        self._hwpCombo     = _serial_combo(DEFAULT_HWP_SERIAL)
        self._hwpScanBtn   = QtWidgets.QPushButton("Scan");       self._hwpScanBtn.setFixedWidth(50)
        self._hwpConnBtn   = QtWidgets.QPushButton("Connect");    self._hwpConnBtn.setFixedWidth(BTN_W)
        self._hwpDiscBtn   = QtWidgets.QPushButton("Disconnect"); self._hwpDiscBtn.setFixedWidth(BTN_W); self._hwpDiscBtn.setEnabled(False)
        self._hwpStatusLbl = _status_lbl()
        layout.addWidget(QtWidgets.QLabel("HWP:"),  1, 0)
        layout.addWidget(self._hwpCombo,            1, 1)
        layout.addWidget(self._hwpScanBtn,          1, 2)
        layout.addWidget(self._hwpConnBtn,          1, 3)
        layout.addWidget(self._hwpDiscBtn,          1, 4)
        layout.addWidget(self._hwpStatusLbl,        1, 5)

        # PM row
        self._pmResourceEdit = QtWidgets.QLineEdit(PM100A.DEFAULT_RESOURCE)
        self._pmResourceEdit.setMinimumWidth(110)
        self._pmScanBtn   = QtWidgets.QPushButton("Scan");       self._pmScanBtn.setFixedWidth(50)
        self._pmConnBtn   = QtWidgets.QPushButton("Connect");    self._pmConnBtn.setFixedWidth(BTN_W)
        self._pmDiscBtn   = QtWidgets.QPushButton("Disconnect"); self._pmDiscBtn.setFixedWidth(BTN_W); self._pmDiscBtn.setEnabled(False)
        self._pmStatusLbl = _status_lbl()
        layout.addWidget(QtWidgets.QLabel("Power meter:"), 2, 0)
        layout.addWidget(self._pmResourceEdit,             2, 1)
        layout.addWidget(self._pmScanBtn,                  2, 2)
        layout.addWidget(self._pmConnBtn,                  2, 3)
        layout.addWidget(self._pmDiscBtn,                  2, 4)
        layout.addWidget(self._pmStatusLbl,                2, 5)

        # Current positions
        self._ndPosLbl  = QtWidgets.QLabel("---")
        self._hwpPosLbl = QtWidgets.QLabel("---")
        self._ndPosLbl.setFont(mono)
        self._hwpPosLbl.setFont(mono)
        layout.addWidget(QtWidgets.QLabel("Current pos:"), 3, 0)
        layout.addWidget(QtWidgets.QLabel("ND:"),          3, 1)
        layout.addWidget(self._ndPosLbl,                   3, 2)
        layout.addWidget(QtWidgets.QLabel("HWP:"),         3, 3)
        layout.addWidget(self._hwpPosLbl,                  3, 4)

        layout.setColumnStretch(5, 1)

        self._ndScanBtn.clicked.connect(self._scan_stages)
        self._hwpScanBtn.clicked.connect(self._scan_stages)
        self._ndConnBtn.clicked.connect(self._connect_nd)
        self._ndDiscBtn.clicked.connect(self._disconnect_nd)
        self._hwpConnBtn.clicked.connect(self._connect_hwp)
        self._hwpDiscBtn.clicked.connect(self._disconnect_hwp)
        self._pmScanBtn.clicked.connect(self._scan_pm)
        self._pmConnBtn.clicked.connect(self._connect_pm)
        self._pmDiscBtn.clicked.connect(self._disconnect_pm)
        return box

    # --- Stage control box ---

    def _build_stage_control_box(self) -> QtWidgets.QGroupBox:
        box    = QtWidgets.QGroupBox("Stage control")
        layout = QtWidgets.QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        def _angle_spin(default=0.0):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(0, 360); s.setDecimals(2); s.setSingleStep(1.0)
            s.setSuffix(" °"); s.setValue(default)
            return s

        self._ndTargetSpin  = _angle_spin(DEFAULT_ND_ANGLE)
        self._hwpTargetSpin = _angle_spin(0.0)
        self._ndMoveBtn     = QtWidgets.QPushButton("Move to"); self._ndMoveBtn.setFixedWidth(BTN_W); self._ndMoveBtn.setEnabled(False)
        self._hwpMoveBtn    = QtWidgets.QPushButton("Move to"); self._hwpMoveBtn.setFixedWidth(BTN_W); self._hwpMoveBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("ND wheel:"), 0, 0)
        layout.addWidget(self._ndTargetSpin,            0, 1)
        layout.addWidget(self._ndMoveBtn,               0, 2)
        layout.addWidget(QtWidgets.QLabel("HWP:"),      1, 0)
        layout.addWidget(self._hwpTargetSpin,           1, 1)
        layout.addWidget(self._hwpMoveBtn,              1, 2)
        layout.setColumnStretch(1, 1)

        self._ndMoveBtn.clicked.connect(self._move_nd)
        self._hwpMoveBtn.clicked.connect(self._move_hwp)
        return box

    # --- PM settings box ---

    def _build_pm_box(self) -> QtWidgets.QGroupBox:
        box    = QtWidgets.QGroupBox("Power meter settings")
        layout = QtWidgets.QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        self._pmWaveSpin = QtWidgets.QDoubleSpinBox()
        self._pmWaveSpin.setRange(300, 1100); self._pmWaveSpin.setDecimals(1)
        self._pmWaveSpin.setSingleStep(1.0); self._pmWaveSpin.setSuffix(" nm")
        self._pmWaveSpin.setValue(532.0)

        self._pmAvgSpin = QtWidgets.QSpinBox()
        self._pmAvgSpin.setRange(1, 5000)
        self._pmAvgSpin.setValue(DEFAULT_PM_AVERAGING)

        self._pmSetWaveBtn = QtWidgets.QPushButton("Set"); self._pmSetWaveBtn.setFixedWidth(BTN_W)
        self._pmSetAvgBtn  = QtWidgets.QPushButton("Set"); self._pmSetAvgBtn.setFixedWidth(BTN_W)
        self._pmReadBtn    = QtWidgets.QPushButton("Read power"); self._pmReadBtn.setFixedWidth(BTN_W)
        self._pmReadLbl    = QtWidgets.QLabel("---")
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self._pmReadLbl.setFont(mono)

        layout.addWidget(QtWidgets.QLabel("Wavelength:"), 0, 0)
        layout.addWidget(self._pmWaveSpin,                0, 1)
        layout.addWidget(self._pmSetWaveBtn,              0, 2)
        layout.addWidget(QtWidgets.QLabel("Averaging:"),  1, 0)
        layout.addWidget(self._pmAvgSpin,                 1, 1)
        layout.addWidget(self._pmSetAvgBtn,               1, 2)
        layout.addWidget(self._pmReadBtn,                 2, 0)
        layout.addWidget(self._pmReadLbl,                 2, 1, 1, 2)
        layout.setColumnStretch(1, 1)

        self._pmSetWaveBtn.clicked.connect(self._pm_set_wavelength)
        self._pmSetAvgBtn.clicked.connect(self._pm_set_averaging)
        self._pmReadBtn.clicked.connect(self._pm_read_power)
        return box

    # --- Sweep parameters box ---

    def _build_sweep_params_box(self) -> QtWidgets.QGroupBox:
        box   = QtWidgets.QGroupBox("Sweep parameters")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setSpacing(4)

        sweep_tabs = QtWidgets.QTabWidget()
        sweep_tabs.setDocumentMode(True)
        outer.addWidget(sweep_tabs)

        def _dspin(val, lo, hi, dec=1, step=1.0, suffix=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(dec)
            s.setSingleStep(step); s.setValue(val)
            if suffix: s.setSuffix(suffix)
            return s

        # --- HWP sweep tab ---
        hwp_w  = QtWidgets.QWidget()
        hwp_gl = QtWidgets.QGridLayout(hwp_w)
        hwp_gl.setHorizontalSpacing(8); hwp_gl.setVerticalSpacing(4)

        self._ndFixedAngleSpin = _dspin(DEFAULT_ND_ANGLE,  0, 360, 1, 1.0, " °")
        self._hwpStartSpin     = _dspin(DEFAULT_HWP_START, 0, 360, 1, 1.0, " °")
        self._hwpStopSpin      = _dspin(DEFAULT_HWP_STOP,  0, 360, 1, 1.0, " °")
        self._hwpStepSpin      = _dspin(DEFAULT_HWP_STEP,  0.1, 90, 2, 0.1, " °")
        self._hwpTotalLbl      = QtWidgets.QLabel("")

        for r, (lbl, w) in enumerate([
            ("ND fixed angle:", self._ndFixedAngleSpin),
            ("HWP start:",      self._hwpStartSpin),
            ("HWP stop:",       self._hwpStopSpin),
            ("HWP step:",       self._hwpStepSpin),
        ]):
            hwp_gl.addWidget(QtWidgets.QLabel(lbl), r, 0)
            hwp_gl.addWidget(w,                      r, 1)
        hwp_gl.addWidget(self._hwpTotalLbl, 4, 0, 1, 2)
        hwp_gl.setColumnStretch(1, 1)
        sweep_tabs.addTab(hwp_w, "HWP sweep")

        # --- ND sweep tab ---
        nd_w  = QtWidgets.QWidget()
        nd_gl = QtWidgets.QGridLayout(nd_w)
        nd_gl.setHorizontalSpacing(8); nd_gl.setVerticalSpacing(4)

        self._hwpFixedAngleSpin = _dspin(DEFAULT_HWP_FIXED_ANGLE, 0, 360, 1, 1.0, " °")
        self._ndStartSpin       = _dspin(DEFAULT_ND_START, 0, 360, 1, 1.0, " °")
        self._ndStopSpin        = _dspin(DEFAULT_ND_STOP,  0, 360, 1, 1.0, " °")
        self._ndStepSpin        = _dspin(DEFAULT_ND_STEP,  0.1, 90, 2, 0.5, " °")
        self._ndTotalLbl        = QtWidgets.QLabel("")

        for r, (lbl, w) in enumerate([
            ("HWP fixed angle:", self._hwpFixedAngleSpin),
            ("ND start:",        self._ndStartSpin),
            ("ND stop:",         self._ndStopSpin),
            ("ND step:",         self._ndStepSpin),
        ]):
            nd_gl.addWidget(QtWidgets.QLabel(lbl), r, 0)
            nd_gl.addWidget(w,                      r, 1)
        nd_gl.addWidget(self._ndTotalLbl, 4, 0, 1, 2)
        nd_gl.setColumnStretch(1, 1)
        sweep_tabs.addTab(nd_w, "ND sweep")

        # --- Shared params ---
        shared = QtWidgets.QGridLayout()
        shared.setHorizontalSpacing(8); shared.setVerticalSpacing(4)

        self._nReadingsSpin = QtWidgets.QSpinBox()
        self._nReadingsSpin.setRange(1, 10000); self._nReadingsSpin.setValue(DEFAULT_N_READINGS)

        self._rampSpin = _dspin(DEFAULT_RAMP_STEP_DEG, 0.1, 20, 1, 0.5, " °")

        shared.addWidget(QtWidgets.QLabel("Readings/step:"), 0, 0)
        shared.addWidget(self._nReadingsSpin,                0, 1)
        shared.addWidget(QtWidgets.QLabel("Ramp step:"),     1, 0)
        shared.addWidget(self._rampSpin,                     1, 1)
        shared.setColumnStretch(1, 1)
        outer.addLayout(shared)

        self._update_hwp_total()
        self._update_nd_total()
        for spin in (self._hwpStartSpin, self._hwpStopSpin, self._hwpStepSpin):
            spin.valueChanged.connect(self._update_hwp_total)
        for spin in (self._ndStartSpin, self._ndStopSpin, self._ndStepSpin):
            spin.valueChanged.connect(self._update_nd_total)

        return box

    def _update_hwp_total(self) -> None:
        self._hwpTotalLbl.setText(f"{len(self._build_hwp_angles())} steps")

    def _update_nd_total(self) -> None:
        self._ndTotalLbl.setText(f"{len(self._build_nd_angles())} steps")

    def _build_hwp_angles(self) -> np.ndarray:
        start, stop, step = (self._hwpStartSpin.value(),
                             self._hwpStopSpin.value(),
                             self._hwpStepSpin.value())
        if step <= 0 or stop <= start: return np.array([])
        return np.arange(start, stop + step * 0.5, step)

    def _build_nd_angles(self) -> np.ndarray:
        start, stop, step = (self._ndStartSpin.value(),
                             self._ndStopSpin.value(),
                             self._ndStepSpin.value())
        if step <= 0 or stop <= start: return np.array([])
        return np.arange(start, stop + step * 0.5, step)

    # --- Action row ---

    def _build_action_row(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        gl = QtWidgets.QGridLayout(w)
        gl.setContentsMargins(0, 0, 0, 0); gl.setSpacing(4)

        self._runHwpBtn  = QtWidgets.QPushButton("Run HWP sweep"); self._runHwpBtn.setFixedWidth(110)
        self._runNdBtn   = QtWidgets.QPushButton("Run ND sweep");  self._runNdBtn.setFixedWidth(110)
        self._runBothBtn = QtWidgets.QPushButton("Run both");      self._runBothBtn.setFixedWidth(80)
        self._stopBtn    = QtWidgets.QPushButton("Stop");          self._stopBtn.setFixedWidth(60); self._stopBtn.setEnabled(False)
        self._saveBtn    = QtWidgets.QPushButton("Save CSV");      self._saveBtn.setEnabled(False)
        self._loadBtn    = QtWidgets.QPushButton("Load CSV")

        gl.addWidget(self._runHwpBtn,  0, 0)
        gl.addWidget(self._runNdBtn,   0, 1)
        gl.addWidget(self._runBothBtn, 0, 2)
        gl.addWidget(self._stopBtn,    0, 3)
        gl.addWidget(self._saveBtn,    1, 0, 1, 2)
        gl.addWidget(self._loadBtn,    1, 2, 1, 2)

        self._runHwpBtn.clicked.connect(self._on_start_hwp)
        self._runNdBtn.clicked.connect(self._on_start_nd)
        self._runBothBtn.clicked.connect(self._on_start_both)
        self._stopBtn.clicked.connect(self._on_stop)
        self._saveBtn.clicked.connect(self._on_save)
        self._loadBtn.clicked.connect(self._on_load)
        return w

    # --- Progress row ---

    def _build_progress_row(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0); vl.setSpacing(4)
        self._progressBar = QtWidgets.QProgressBar(); self._progressBar.setValue(0)
        self._statusLbl   = QtWidgets.QLabel("Idle.")
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(8)
        self._statusLbl.setFont(mono); self._statusLbl.setWordWrap(True)
        vl.addWidget(self._progressBar)
        vl.addWidget(self._statusLbl)
        return w

    # --- Right panel tabs ---

    def _build_hwp_tab(self) -> QtWidgets.QWidget:
        w  = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4); vl.setSpacing(6)

        self._hwpFig    = Figure(figsize=(5, 3), tight_layout=True)
        self._hwpAx     = self._hwpFig.add_subplot(111)
        self._hwpAx.set_xlabel("HWP angle (°)")
        self._hwpAx.set_ylabel("Power")
        self._hwpAx.set_title("HWP calibration")
        self._hwpCanvas = FigureCanvas(self._hwpFig)
        self._hwpCanvas.setMinimumHeight(200)

        self._hwpTable = QtWidgets.QTableWidget(0, 3)
        self._hwpTable.setHorizontalHeaderLabels(["#", "HWP angle (°)", "Mean power"])
        self._hwpTable.horizontalHeader().setStretchLastSection(True)
        self._hwpTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._hwpTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._hwpTable.setMinimumHeight(100)

        vl.addWidget(self._hwpCanvas, stretch=3)
        vl.addWidget(self._hwpTable,  stretch=2)
        return w

    def _build_nd_tab(self) -> QtWidgets.QWidget:
        w  = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4); vl.setSpacing(6)

        self._ndFig    = Figure(figsize=(5, 3), tight_layout=True)
        self._ndAx     = self._ndFig.add_subplot(111)
        self._ndAx.set_xlabel("ND angle (°)")
        self._ndAx.set_ylabel("Power")
        self._ndAx.set_title("ND calibration")
        self._ndCanvas = FigureCanvas(self._ndFig)
        self._ndCanvas.setMinimumHeight(200)

        self._ndTable = QtWidgets.QTableWidget(0, 3)
        self._ndTable.setHorizontalHeaderLabels(["#", "ND angle (°)", "Mean power"])
        self._ndTable.horizontalHeader().setStretchLastSection(True)
        self._ndTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._ndTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._ndTable.setMinimumHeight(100)

        vl.addWidget(self._ndCanvas, stretch=3)
        vl.addWidget(self._ndTable,  stretch=2)
        return w

    def _build_analysis_tab(self) -> QtWidgets.QWidget:
        w  = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4); vl.setSpacing(6)

        # ND selector row
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("ND wheel angle:"))
        self._ndSelCombo = QtWidgets.QComboBox()
        self._ndSelCombo.setMinimumWidth(100)
        sel_row.addWidget(self._ndSelCombo)
        sel_row.addSpacing(16)
        sel_row.addWidget(QtWidgets.QLabel("Max power at selection:"))
        self._analysisMaxLbl = QtWidgets.QLabel("---")
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self._analysisMaxLbl.setFont(mono)
        sel_row.addWidget(self._analysisMaxLbl)
        sel_row.addStretch()
        vl.addLayout(sel_row)

        self._analysisFig    = Figure(figsize=(5, 3), tight_layout=True)
        self._analysisAx     = self._analysisFig.add_subplot(111)
        self._analysisAx.set_xlabel("HWP angle (°)")
        self._analysisAx.set_ylabel("Converted power")
        self._analysisAx.set_title("Expected power vs HWP angle")
        self._analysisCanvas = FigureCanvas(self._analysisFig)
        self._analysisCanvas.setMinimumHeight(250)
        vl.addWidget(self._analysisCanvas, stretch=1)

        self._ndSelCombo.currentIndexChanged.connect(self._update_analysis)
        return w

    # ------------------------------------------------------------------ #
    # Device connection                                                    #
    # ------------------------------------------------------------------ #

    def _scan_stages(self) -> None:
        try:
            serials = list_kinesis_serials()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Scan", f"Scan failed: {e}")
            return
        if not serials:
            QtWidgets.QMessageBox.information(self, "Scan", "No Kinesis devices found.")
            return
        for combo in (self._ndCombo, self._hwpCombo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear(); combo.addItems(serials)
            if current in serials: combo.setCurrentText(current)
            combo.blockSignals(False)

    def _connect_nd(self) -> None:
        serial = self._ndCombo.currentText().strip()
        if not serial: return
        try:
            stage = RotationStage(serial, scale=DEFAULT_STAGE_SCALE)
            stage.open()
            self._nd_stage = stage
            self._ndStatusLbl.setText(f"OK ({serial})")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "ND connect failed", str(e))
            self._nd_stage = None; self._ndStatusLbl.setText("Error")
        self._sync_controls()

    def _disconnect_nd(self) -> None:
        try:
            if self._nd_stage: self._nd_stage.close()
        except Exception: pass
        self._nd_stage = None
        self._ndStatusLbl.setText("Disconnected")
        self._sync_controls()

    def _connect_hwp(self) -> None:
        serial = self._hwpCombo.currentText().strip()
        if not serial: return
        try:
            stage = RotationStage(serial, scale=DEFAULT_STAGE_SCALE)
            stage.open()
            self._hwp_stage = stage
            self._hwpStatusLbl.setText(f"OK ({serial})")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "HWP connect failed", str(e))
            self._hwp_stage = None; self._hwpStatusLbl.setText("Error")
        self._sync_controls()

    def _disconnect_hwp(self) -> None:
        try:
            if self._hwp_stage: self._hwp_stage.close()
        except Exception: pass
        self._hwp_stage = None
        self._hwpStatusLbl.setText("Disconnected")
        self._sync_controls()

    def _scan_pm(self) -> None:
        resources = PM100A.list_resources()
        if not resources:
            QtWidgets.QMessageBox.information(self, "PM scan", "No power meters found.")
            return
        if len(resources) == 1:
            self._pmResourceEdit.setText(resources[0])
        else:
            choice, ok = QtWidgets.QInputDialog.getItem(
                self, "Select power meter", "Resource:", resources, 0, False)
            if ok and choice: self._pmResourceEdit.setText(choice)

    def _connect_pm(self) -> None:
        resource = self._pmResourceEdit.text().strip()
        if not resource: return
        try:
            self._pm = PM100A(resource)
            self._pmStatusLbl.setText("OK")
        except PM100AError as e:
            QtWidgets.QMessageBox.critical(self, "PM connect failed", str(e))
            self._pm = None; self._pmStatusLbl.setText("Error")
        self._sync_controls()

    def _disconnect_pm(self) -> None:
        try:
            if self._pm: self._pm.close()
        except Exception: pass
        self._pm = None
        self._pmStatusLbl.setText("Disconnected")
        self._sync_controls()

    # ------------------------------------------------------------------ #
    # Stage control                                                        #
    # ------------------------------------------------------------------ #

    def _move_nd(self) -> None:
        if self._nd_stage is None or self._nd_mover is not None: return
        self._nd_mover = _MoveWorker(
            self._nd_stage, self._ndTargetSpin.value(), self._rampSpin.value(), self)
        self._nd_mover.done.connect(self._on_nd_move_done)
        self._nd_mover.start()
        self._sync_controls()

    def _on_nd_move_done(self, result: str) -> None:
        self._nd_mover = None
        if result != "ok": QtWidgets.QMessageBox.critical(self, "ND move failed", result)
        self._sync_controls()

    def _move_hwp(self) -> None:
        if self._hwp_stage is None or self._hwp_mover is not None: return
        self._hwp_mover = _MoveWorker(
            self._hwp_stage, self._hwpTargetSpin.value(), self._rampSpin.value(), self)
        self._hwp_mover.done.connect(self._on_hwp_move_done)
        self._hwp_mover.start()
        self._sync_controls()

    def _on_hwp_move_done(self, result: str) -> None:
        self._hwp_mover = None
        if result != "ok": QtWidgets.QMessageBox.critical(self, "HWP move failed", result)
        self._sync_controls()

    # ------------------------------------------------------------------ #
    # PM settings                                                          #
    # ------------------------------------------------------------------ #

    def _pm_set_wavelength(self) -> None:
        if self._pm is None: return
        try:
            self._pm.set_wavelength(self._pmWaveSpin.value())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "PM wavelength", str(e))

    def _pm_set_averaging(self) -> None:
        if self._pm is None: return
        try:
            self._pm.set_averaging(self._pmAvgSpin.value())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "PM averaging", str(e))

    def _pm_read_power(self) -> None:
        if self._pm is None: return
        try:
            self._pmReadLbl.setText(_fmt_power(self._pm.measure_power()))
        except Exception as e:
            self._pmReadLbl.setText("err")
            QtWidgets.QMessageBox.critical(self, "PM read", str(e))

    # ------------------------------------------------------------------ #
    # Position readout                                                     #
    # ------------------------------------------------------------------ #

    def _update_positions(self) -> None:
        for stage, lbl in ((self._nd_stage, self._ndPosLbl),
                           (self._hwp_stage, self._hwpPosLbl)):
            if stage is not None:
                try:
                    lbl.setText(f"{stage.get_position_cached():.2f}°")
                except Exception:
                    lbl.setText("err")
            else:
                lbl.setText("---")

    # ------------------------------------------------------------------ #
    # Control state                                                        #
    # ------------------------------------------------------------------ #

    def _sync_controls(self) -> None:
        nd_ok  = self._nd_stage  is not None
        hwp_ok = self._hwp_stage is not None
        pm_ok  = self._pm        is not None
        busy   = self._running
        nd_moving  = self._nd_mover  is not None
        hwp_moving = self._hwp_mover is not None

        self._ndConnBtn.setEnabled(not nd_ok and not busy)
        self._ndDiscBtn.setEnabled(nd_ok and not busy and not nd_moving)
        self._ndScanBtn.setEnabled(not busy)
        self._ndMoveBtn.setEnabled(nd_ok and not busy and not nd_moving)

        self._hwpConnBtn.setEnabled(not hwp_ok and not busy)
        self._hwpDiscBtn.setEnabled(hwp_ok and not busy and not hwp_moving)
        self._hwpScanBtn.setEnabled(not busy)
        self._hwpMoveBtn.setEnabled(hwp_ok and not busy and not hwp_moving)

        self._pmConnBtn.setEnabled(not pm_ok and not busy)
        self._pmDiscBtn.setEnabled(pm_ok and not busy)
        self._pmScanBtn.setEnabled(not busy)
        self._pmResourceEdit.setEnabled(not pm_ok and not busy)
        self._pmSetWaveBtn.setEnabled(pm_ok and not busy)
        self._pmSetAvgBtn.setEnabled(pm_ok and not busy)
        self._pmReadBtn.setEnabled(pm_ok and not busy)

        all_ready = nd_ok and hwp_ok and pm_ok
        self._runHwpBtn.setEnabled(all_ready and not busy)
        self._runNdBtn.setEnabled(all_ready and not busy)
        self._runBothBtn.setEnabled(all_ready and not busy)
        self._stopBtn.setEnabled(busy)

        has_data = bool(self._hwp_angles or self._nd_angles)
        self._saveBtn.setEnabled(has_data and not busy)
        self._loadBtn.setEnabled(not busy)

        for spin in (
            self._ndFixedAngleSpin, self._hwpStartSpin, self._hwpStopSpin, self._hwpStepSpin,
            self._hwpFixedAngleSpin, self._ndStartSpin, self._ndStopSpin, self._ndStepSpin,
            self._nReadingsSpin, self._rampSpin, self._pmWaveSpin, self._pmAvgSpin,
        ):
            spin.setEnabled(not busy)

    # ------------------------------------------------------------------ #
    # Measurement                                                          #
    # ------------------------------------------------------------------ #

    def _on_start_hwp(self) -> None:
        angles = self._build_hwp_angles()
        if len(angles) == 0:
            QtWidgets.QMessageBox.warning(self, "Bad sweep", "No HWP steps — check start/stop/step.")
            return
        self._hwp_angles.clear(); self._hwp_powers.clear(); self._hwp_stds.clear()
        self._hwpTable.setRowCount(0)
        self._hwpAx.clear()
        self._hwpAx.set_xlabel("HWP angle (°)"); self._hwpAx.set_ylabel("Power")
        self._hwpAx.set_title("HWP calibration")
        self._hwpCanvas.draw_idle()
        self._progressBar.setRange(0, len(angles)); self._progressBar.setValue(0)
        self._right_tabs.setCurrentIndex(0)
        self._start_sweep("hwp",
                          fixed_stage=self._nd_stage,  sweep_stage=self._hwp_stage,
                          fixed_angle=self._ndFixedAngleSpin.value(), sweep_angles=angles,
                          fixed_label="ND wheel", sweep_label="HWP")

    def _on_start_nd(self) -> None:
        angles = self._build_nd_angles()
        if len(angles) == 0:
            QtWidgets.QMessageBox.warning(self, "Bad sweep", "No ND steps — check start/stop/step.")
            return
        self._nd_angles.clear(); self._nd_powers.clear(); self._nd_stds.clear()
        self._ndTable.setRowCount(0)
        self._ndAx.clear()
        self._ndAx.set_xlabel("ND angle (°)"); self._ndAx.set_ylabel("Power")
        self._ndAx.set_title("ND calibration")
        self._ndCanvas.draw_idle()
        self._progressBar.setRange(0, len(angles)); self._progressBar.setValue(0)
        self._right_tabs.setCurrentIndex(1)
        self._start_sweep("nd",
                          fixed_stage=self._hwp_stage, sweep_stage=self._nd_stage,
                          fixed_angle=self._hwpFixedAngleSpin.value(), sweep_angles=angles,
                          fixed_label="HWP", sweep_label="ND wheel")

    def _on_start_both(self) -> None:
        self._queue_nd = True
        self._on_start_hwp()

    def _start_sweep(self, sweep_type: str, *, fixed_stage, sweep_stage,
                     fixed_angle, sweep_angles, fixed_label, sweep_label) -> None:
        pm_resource = self._pmResourceEdit.text().strip()
        self._disconnect_pm()

        self._sweep_type = sweep_type
        self._worker = StageSweepWorker(
            fixed_stage, sweep_stage, pm_resource,
            fixed_angle   = fixed_angle,
            sweep_angles  = sweep_angles,
            n_readings    = self._nReadingsSpin.value(),
            fixed_label   = fixed_label,
            sweep_label   = sweep_label,
            ramp_step_deg = self._rampSpin.value(),
            pm_averaging  = self._pmAvgSpin.value(),
            parent        = self,
        )
        self._worker.point_done.connect(self._on_point_done)
        self._worker.status.connect(self._statusLbl.setText)
        self._worker.done.connect(self._on_done)
        self._running = True
        self._sync_controls()
        self._worker.start()

    def _on_stop(self) -> None:
        self._queue_nd = False
        if self._worker: self._worker.stop()
        self._statusLbl.setText("Stopping…")

    def _on_point_done(self, step: int, total: int,
                       angle: float, mean_power: float, std_power: float) -> None:
        self._progressBar.setValue(step + 1)

        if self._sweep_type == "hwp":
            self._hwp_angles.append(angle)
            self._hwp_powers.append(mean_power)
            self._hwp_stds.append(std_power)
            row = self._hwpTable.rowCount()
            self._hwpTable.insertRow(row)
            self._hwpTable.setItem(row, 0, QtWidgets.QTableWidgetItem(str(step + 1)))
            self._hwpTable.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{angle:.2f}"))
            self._hwpTable.setItem(row, 2, QtWidgets.QTableWidgetItem(
                f"{_fmt_power(mean_power)} ± {_fmt_power(std_power)}"))
            self._hwpTable.scrollToBottom()
            self._redraw_hwp_plot()
        else:
            self._nd_angles.append(angle)
            self._nd_powers.append(mean_power)
            self._nd_stds.append(std_power)
            row = self._ndTable.rowCount()
            self._ndTable.insertRow(row)
            self._ndTable.setItem(row, 0, QtWidgets.QTableWidgetItem(str(step + 1)))
            self._ndTable.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{angle:.2f}"))
            self._ndTable.setItem(row, 2, QtWidgets.QTableWidgetItem(
                f"{_fmt_power(mean_power)} ± {_fmt_power(std_power)}"))
            self._ndTable.scrollToBottom()
            self._redraw_nd_plot()

    def _on_done(self, result: str, data: object) -> None:
        self._running = False
        self._connect_pm()

        if result == "ok":
            label = "HWP" if self._sweep_type == "hwp" else "ND"
            self._statusLbl.setText(f"{label} sweep done.")

            if self._sweep_type == "hwp" and self._queue_nd:
                self._queue_nd = False
                QtCore.QTimer.singleShot(500, self._on_start_nd)
                return   # _sync_controls called when ND sweep starts

            self._populate_analysis()
        elif result == "aborted":
            self._statusLbl.setText("Aborted.")
            self._queue_nd = False
        else:
            self._statusLbl.setText(result)
            QtWidgets.QMessageBox.critical(self, "Sweep error", result)
            self._queue_nd = False

        self._sync_controls()

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    def _populate_analysis(self) -> None:
        """Repopulate ND selector combo and refresh analysis plot."""
        if not self._nd_angles:
            return
        self._ndSelCombo.blockSignals(True)
        self._ndSelCombo.clear()
        for a in self._nd_angles:
            self._ndSelCombo.addItem(f"{a:.2f}°")
        self._ndSelCombo.blockSignals(False)
        self._update_analysis()
        if self._hwp_angles:
            self._right_tabs.setCurrentIndex(2)

    def _update_analysis(self) -> None:
        """Recompute and plot converted power for the selected ND angle."""
        idx = self._ndSelCombo.currentIndex()
        if idx < 0 or not self._hwp_angles or not self._nd_angles:
            self._analysisMaxLbl.setText("---")
            self._analysisAx.clear()
            self._analysisAx.set_xlabel("HWP angle (°)")
            self._analysisAx.set_ylabel("Converted power")
            self._analysisAx.set_title("Expected power vs HWP angle")
            self._analysisCanvas.draw_idle()
            return

        hwp_p    = np.asarray(self._hwp_powers)
        hwp_std  = np.asarray(self._hwp_stds)
        p_max    = float(np.nanmax(hwp_p))
        if p_max <= 0:
            return

        norm     = hwp_p   / p_max
        norm_std = hwp_std / p_max

        p_sel         = self._nd_powers[idx]
        conv_powers   = norm     * p_sel
        conv_stds     = norm_std * p_sel
        nd_angle_sel  = self._nd_angles[idx]

        self._analysisMaxLbl.setText(_fmt_power(p_sel))

        scale, unit = _best_unit(conv_powers)
        self._analysisAx.clear()
        self._analysisAx.errorbar(
            self._hwp_angles, conv_powers * scale, yerr=conv_stds * scale,
            fmt="o-", ms=3, lw=1, capsize=3, elinewidth=0.8,
        )
        self._analysisAx.set_xlabel("HWP angle (°)")
        self._analysisAx.set_ylabel(f"Converted power ({unit})")
        self._analysisAx.set_title(f"Expected power  (ND = {nd_angle_sel:.2f}°)")
        self._analysisFig.tight_layout()
        self._analysisCanvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Redraw helpers                                                       #
    # ------------------------------------------------------------------ #

    def _redraw_hwp_plot(self) -> None:
        powers = np.asarray(self._hwp_powers)
        stds   = np.asarray(self._hwp_stds)
        scale, unit = _best_unit(powers)
        self._hwpAx.clear()
        self._hwpAx.errorbar(self._hwp_angles, powers * scale, yerr=stds * scale,
                             fmt="o-", ms=3, lw=1, capsize=3, elinewidth=0.8)
        self._hwpAx.set_xlabel("HWP angle (°)")
        self._hwpAx.set_ylabel(f"Power ({unit})")
        self._hwpAx.set_title("HWP calibration")
        self._hwpFig.tight_layout()
        self._hwpCanvas.draw_idle()

    def _redraw_nd_plot(self) -> None:
        powers = np.asarray(self._nd_powers)
        stds   = np.asarray(self._nd_stds)
        scale, unit = _best_unit(powers)
        self._ndAx.clear()
        self._ndAx.errorbar(self._nd_angles, powers * scale, yerr=stds * scale,
                            fmt="o-", ms=3, lw=1, capsize=3, elinewidth=0.8)
        self._ndAx.set_xlabel("ND angle (°)")
        self._ndAx.set_ylabel(f"Power ({unit})")
        self._ndAx.set_title("ND calibration")
        self._ndFig.tight_layout()
        self._ndCanvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Save / Load                                                          #
    # ------------------------------------------------------------------ #

    def _on_save(self) -> None:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.join(DATA_DIR, f"wheel_hwp_calib_{ts}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration CSV", default, "CSV files (*.csv)")
        if not path: return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow([f"# Wheel+HWP power calibration  {datetime.now().isoformat()}"])
                w.writerow([f"# ND serial:      {self._ndCombo.currentText()}"])
                w.writerow([f"# HWP serial:     {self._hwpCombo.currentText()}"])
                w.writerow([f"# PM averaging:   {self._pmAvgSpin.value()}"])
                w.writerow([f"# Readings/step:  {self._nReadingsSpin.value()}"])
                w.writerow([f"# Wavelength (nm):{self._pmWaveSpin.value()}"])

                if self._hwp_angles:
                    hwp_p = np.asarray(self._hwp_powers)
                    p_max = float(np.nanmax(hwp_p)) if len(hwp_p) else 1.0
                    w.writerow(["[HWP_SWEEP]"])
                    w.writerow(["hwp_angle_deg", "mean_power_w", "std_power_w", "norm_power"])
                    for ang, pw, sd in zip(self._hwp_angles, self._hwp_powers, self._hwp_stds):
                        norm = pw / p_max if p_max > 0 else float("nan")
                        w.writerow([f"{ang:.4f}", f"{pw:.6e}", f"{sd:.6e}", f"{norm:.6f}"])

                if self._nd_angles:
                    w.writerow(["[ND_SWEEP]"])
                    w.writerow(["nd_angle_deg", "mean_power_w", "std_power_w"])
                    for ang, pw, sd in zip(self._nd_angles, self._nd_powers, self._nd_stds):
                        w.writerow([f"{ang:.4f}", f"{pw:.6e}", f"{sd:.6e}"])

            self._statusLbl.setText(f"Saved → {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration CSV", DATA_DIR, "CSV files (*.csv)")
        if not path: return
        try:
            hwp_rows, nd_rows = _parse_calib_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return

        if hwp_rows:
            self._hwp_angles = [r[0] for r in hwp_rows]
            self._hwp_powers = [r[1] for r in hwp_rows]
            self._hwp_stds   = [r[2] for r in hwp_rows]
            self._hwpTable.setRowCount(0)
            for i, (ang, pw, sd) in enumerate(zip(
                    self._hwp_angles, self._hwp_powers, self._hwp_stds)):
                self._hwpTable.insertRow(i)
                self._hwpTable.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
                self._hwpTable.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{ang:.2f}"))
                self._hwpTable.setItem(i, 2, QtWidgets.QTableWidgetItem(
                    f"{_fmt_power(pw)} ± {_fmt_power(sd)}"))
            self._redraw_hwp_plot()

        if nd_rows:
            self._nd_angles = [r[0] for r in nd_rows]
            self._nd_powers = [r[1] for r in nd_rows]
            self._nd_stds   = [r[2] for r in nd_rows]
            self._ndTable.setRowCount(0)
            for i, (ang, pw, sd) in enumerate(zip(
                    self._nd_angles, self._nd_powers, self._nd_stds)):
                self._ndTable.insertRow(i)
                self._ndTable.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
                self._ndTable.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{ang:.2f}"))
                self._ndTable.setItem(i, 2, QtWidgets.QTableWidgetItem(
                    f"{_fmt_power(pw)} ± {_fmt_power(sd)}"))
            self._redraw_nd_plot()

        self._populate_analysis()
        self._statusLbl.setText(f"Loaded ← {os.path.basename(path)}")
        self._sync_controls()


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #

def _parse_calib_csv(path: str):
    """Return (hwp_rows, nd_rows); each row is [angle, power_w, std_w]."""
    hwp_rows: list = []
    nd_rows:  list = []
    section        = None
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row: continue
            cell = row[0].strip()
            if cell.startswith("#"):            continue
            if cell == "[HWP_SWEEP]":          section = "hwp"; continue
            if cell == "[ND_SWEEP]":           section = "nd";  continue
            if cell in ("hwp_angle_deg", "nd_angle_deg"): continue
            try:
                vals = [float(c) for c in row[:3]]
            except (ValueError, IndexError):
                continue
            if   section == "hwp": hwp_rows.append(vals)
            elif section == "nd":  nd_rows.append(vals)
    return hwp_rows, nd_rows


def _best_unit(powers):
    mx = float(np.nanmax(np.abs(powers))) if len(powers) else 0.0
    if mx < 1e-6: return 1e9, "nW"
    if mx < 1e-3: return 1e6, "µW"
    if mx < 1.0:  return 1e3, "mW"
    return 1.0, "W"


def _fmt_power(watts: float) -> str:
    if np.isnan(watts):        return "NaN"
    if abs(watts) < 1e-9:  return f"{watts * 1e12:.2f} pW"
    if abs(watts) < 1e-6:  return f"{watts * 1e9:.2f} nW"
    if abs(watts) < 1e-3:  return f"{watts * 1e6:.2f} µW"
    if abs(watts) < 1.0:   return f"{watts * 1e3:.2f} mW"
    return f"{watts:.4f} W"

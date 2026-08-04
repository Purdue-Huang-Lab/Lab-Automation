from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from rot.rot_wrapper import RotationStage, MotionController, SweepConfig, ROT_RANGE_DEFAULT

from .config import (
    BTN_W,
    READ_W,
    STATUS_W,
    DEFAULT_STEP_DEG,
    DEFAULT_ACCEL,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_SWEEP_STEP_DEG,
    READOUT_UNKNOWN,
    DEFAULT_STAGE_SCALE,
)
from .workers import MoveThread, HomeThread, SweepThread
from .power_calibration import PowerCalibrationWidget


class RotationStagePanel(QtWidgets.QGroupBox):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.stage: Optional[RotationStage] = None

        self._move_thread: Optional[MoveThread] = None
        self._home_thread: Optional[HomeThread] = None
        self._sweep_thread: Optional[SweepThread] = None

        self._move_controller: Optional[MotionController] = None
        self._home_controller: Optional[MotionController] = None
        self._sweep_controller: Optional[MotionController] = None

        self._moving = False
        self._homing = False
        self._sweeping = False

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Connection
        self.serialCombo = QtWidgets.QComboBox()
        self.serialCombo.setEditable(True)
        self.serialCombo.setMinimumWidth(160)

        self.scaleCombo = QtWidgets.QComboBox()
        self.scaleCombo.setEditable(True)
        self.scaleCombo.setMinimumWidth(120)
        self.scaleCombo.addItems([DEFAULT_STAGE_SCALE, "stage", "PRM1-Z8", "step"])
        self.scaleCombo.setCurrentText(DEFAULT_STAGE_SCALE)

        self.connectBtn = QtWidgets.QPushButton("Connect"); self.connectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect"); self.disconnectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Serial:"), 0, 0)
        layout.addWidget(self.serialCombo, 0, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Scale:"), 0, 3)
        layout.addWidget(self.scaleCombo, 0, 4)
        layout.addWidget(self.connectBtn, 0, 5)
        layout.addWidget(self.disconnectBtn, 0, 6)

        # Status
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        self.posLbl = QtWidgets.QLabel(READOUT_UNKNOWN); self.posLbl.setFont(mono)
        self.posLbl.setMinimumWidth(READ_W)
        self.homedLbl = QtWidgets.QLabel("Homed: ?"); self.homedLbl.setFont(mono)
        self.homedLbl.setMinimumWidth(READ_W)
        self.statusLbl = QtWidgets.QLabel("Disconnected."); self.statusLbl.setFont(mono)
        self.statusLbl.setMinimumWidth(STATUS_W)

        layout.addWidget(QtWidgets.QLabel("Position (deg):"), 1, 0)
        layout.addWidget(self.posLbl, 1, 1)
        layout.addWidget(self.homedLbl, 1, 2)
        layout.addWidget(self.statusLbl, 1, 3, 1, 4)

        # Move controls
        self.targetSpin = QtWidgets.QDoubleSpinBox()
        self.targetSpin.setDecimals(3)
        self.targetSpin.setRange(ROT_RANGE_DEFAULT[0], ROT_RANGE_DEFAULT[1])
        self.targetSpin.setValue(0.0)

        self.stepSpin = QtWidgets.QDoubleSpinBox()
        self.stepSpin.setDecimals(3)
        self.stepSpin.setRange(0.001, 360.0)
        self.stepSpin.setValue(DEFAULT_STEP_DEG)

        self.accelSpin = QtWidgets.QDoubleSpinBox()
        self.accelSpin.setDecimals(3)
        self.accelSpin.setRange(0.0, 200.0)
        self.accelSpin.setValue(DEFAULT_ACCEL)

        self.moveBtn = QtWidgets.QPushButton("Move")
        self.moveBtn.setFixedWidth(BTN_W); self.moveBtn.setEnabled(False)

        self.jogNegBtn = QtWidgets.QPushButton("<"); self.jogNegBtn.setFixedWidth(40); self.jogNegBtn.setEnabled(False)
        self.jogPosBtn = QtWidgets.QPushButton(">"); self.jogPosBtn.setFixedWidth(40); self.jogPosBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Target (deg):"), 2, 0)
        layout.addWidget(self.targetSpin, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Step (deg):"), 2, 2)
        layout.addWidget(self.stepSpin, 2, 3)
        layout.addWidget(QtWidgets.QLabel("Accel:"), 2, 4)
        layout.addWidget(self.accelSpin, 2, 5)

        layout.addWidget(self.jogNegBtn, 3, 0)
        layout.addWidget(self.moveBtn, 3, 1)
        layout.addWidget(self.jogPosBtn, 3, 2)

        # Home/Stop
        self.homeBtn = QtWidgets.QPushButton("Home")
        self.homeBtn.setFixedWidth(BTN_W); self.homeBtn.setEnabled(False)
        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.stopBtn.setFixedWidth(BTN_W); self.stopBtn.setEnabled(False)

        layout.addWidget(self.homeBtn, 3, 4)
        layout.addWidget(self.stopBtn, 3, 5)

        # Sweep controls
        self.sweepStartSpin = QtWidgets.QDoubleSpinBox()
        self.sweepStartSpin.setDecimals(3)
        self.sweepStartSpin.setRange(ROT_RANGE_DEFAULT[0], ROT_RANGE_DEFAULT[1])
        self.sweepStartSpin.setValue(0.0)

        self.sweepStopSpin = QtWidgets.QDoubleSpinBox()
        self.sweepStopSpin.setDecimals(3)
        self.sweepStopSpin.setRange(ROT_RANGE_DEFAULT[0], ROT_RANGE_DEFAULT[1])
        self.sweepStopSpin.setValue(90.0)

        self.sweepStepSpin = QtWidgets.QDoubleSpinBox()
        self.sweepStepSpin.setDecimals(3)
        self.sweepStepSpin.setRange(0.001, 360.0)
        self.sweepStepSpin.setValue(DEFAULT_SWEEP_STEP_DEG)

        self.sweepRampSpin = QtWidgets.QDoubleSpinBox()
        self.sweepRampSpin.setDecimals(3)
        self.sweepRampSpin.setRange(0.001, 360.0)
        self.sweepRampSpin.setValue(DEFAULT_RAMP_STEP_DEG)

        self.sweepBtn = QtWidgets.QPushButton("Start Sweep")
        self.sweepBtn.setFixedWidth(BTN_W); self.sweepBtn.setEnabled(False)
        self.sweepStopBtn = QtWidgets.QPushButton("Stop Sweep")
        self.sweepStopBtn.setFixedWidth(BTN_W); self.sweepStopBtn.setEnabled(False)
        self.sweepProgLbl = QtWidgets.QLabel("Sweep: idle")

        layout.addWidget(QtWidgets.QLabel("Sweep start:"), 4, 0)
        layout.addWidget(self.sweepStartSpin, 4, 1)
        layout.addWidget(QtWidgets.QLabel("Sweep stop:"), 4, 2)
        layout.addWidget(self.sweepStopSpin, 4, 3)
        layout.addWidget(QtWidgets.QLabel("Step:"), 4, 4)
        layout.addWidget(self.sweepStepSpin, 4, 5)

        layout.addWidget(QtWidgets.QLabel("Ramp step:"), 5, 0)
        layout.addWidget(self.sweepRampSpin, 5, 1)
        layout.addWidget(self.sweepBtn, 5, 4)
        layout.addWidget(self.sweepStopBtn, 5, 5)
        layout.addWidget(self.sweepProgLbl, 6, 0, 1, 6)

        layout.setColumnStretch(3, 1)

        # Power calibration
        self.calibWidget = PowerCalibrationWidget(position_provider=self._get_positions_for_calib)
        layout.addWidget(self.calibWidget, 7, 0, 1, 6)
        layout.setRowStretch(7, 1)

        # Signals
        self.connectBtn.clicked.connect(self.on_connect)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.moveBtn.clicked.connect(self.on_move)
        self.jogNegBtn.clicked.connect(lambda: self.on_jog(-1))
        self.jogPosBtn.clicked.connect(lambda: self.on_jog(1))
        self.homeBtn.clicked.connect(self.on_home)
        self.stopBtn.clicked.connect(self.on_stop_any)
        self.sweepBtn.clicked.connect(self.on_sweep)
        self.sweepStopBtn.clicked.connect(self.on_stop_any)

        pos_font = QtGui.QFont(mono)
        pos_font.setPointSize(max(1, (mono.pointSize() + 3) * 5))
        self.posLbl.setFont(pos_font)

    def set_serials(self, serials: list[str]) -> None:
        current = self.serialCombo.currentText().strip()
        self.serialCombo.blockSignals(True)
        self.serialCombo.clear()
        self.serialCombo.addItems(serials)
        if current:
            self.serialCombo.setCurrentText(current)
        self.serialCombo.blockSignals(False)

    def _sync_controls(self) -> None:
        connected = self.stage is not None
        busy = self._moving or self._homing or self._sweeping
        self.connectBtn.setEnabled(not connected and not busy)
        self.disconnectBtn.setEnabled(connected and not busy)
        self.moveBtn.setEnabled(connected and not busy)
        self.jogNegBtn.setEnabled(connected and not busy)
        self.jogPosBtn.setEnabled(connected and not busy)
        self.homeBtn.setEnabled(connected and not busy)
        self.stopBtn.setEnabled(connected and busy)
        self.sweepBtn.setEnabled(connected and not busy)
        self.sweepStopBtn.setEnabled(connected and busy)

    def _get_positions_for_calib(self):
        if not self.stage:
            raise RuntimeError("Not connected")
        panels = self._find_panel_group()
        if not panels:
            return float(self.stage.get_position())
        connected = []
        for panel in panels:
            if getattr(panel, "stage", None) is None:
                continue
            connected.append(panel)
        if len(connected) <= 1:
            return float(self.stage.get_position())
        positions = {}
        for panel in connected:
            label = panel.title()
            try:
                pos = float(panel.stage.get_position())
            except Exception:
                continue
            positions[label] = pos
        if not positions:
            return float(self.stage.get_position())
        return (self.title(), positions)

    def _find_panel_group(self):
        widget = self.parent()
        while widget is not None:
            panels = getattr(widget, "panels", None)
            if isinstance(panels, list):
                return panels
            widget = widget.parent()
        return None

    # ---- Connection ----

    def on_connect(self):
        serial = self.serialCombo.currentText().strip()
        if not serial:
            QtWidgets.QMessageBox.warning(self, "Serial", "Enter a motor serial number.")
            return
        scale = self.scaleCombo.currentText().strip() or DEFAULT_STAGE_SCALE
        try:
            stage = RotationStage(serial, scale=scale)
            stage.open()
            self.stage = stage
            self.statusLbl.setText(f"Connected: {serial}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Connect failed", str(e))
            self.stage = None
        self._sync_controls()

    def on_disconnect(self):
        self.on_stop_any()
        try:
            if self.stage:
                self.stage.close()
        except Exception:
            pass
        self.stage = None
        self.posLbl.setText(READOUT_UNKNOWN)
        self.homedLbl.setText("Homed: ?")
        self.statusLbl.setText("Disconnected.")
        self.sweepProgLbl.setText("Sweep: idle")
        self._sync_controls()

    # ---- Poll ----

    def poll_once(self):
        if self.stage is None:
            return
        if self._moving or self._homing or self._sweeping:
            return
        try:
            pos = float(self.stage.get_position())
            self.posLbl.setText(f"{pos:.3f}")
            homed = self.stage.is_homed()
            if homed is None:
                self.homedLbl.setText("Homed: ?")
            else:
                self.homedLbl.setText("Homed: " + ("yes" if homed else "no"))
            self.statusLbl.setText("Ready")
        except Exception as e:
            self.statusLbl.setText(f"Poll error: {e}")

    # ---- Motion ----

    def on_move(self):
        if not self.stage:
            return
        self._moving = True
        self._sync_controls()
        target = float(self.targetSpin.value())
        step = float(self.stepSpin.value())
        accel = float(self.accelSpin.value())

        self._move_controller = MotionController()
        self._move_thread = MoveThread(self.stage, target, step, accel, self._move_controller)
        self._move_thread.progress.connect(self._on_motion_progress)
        self._move_thread.status.connect(self.statusLbl.setText)
        self._move_thread.done.connect(self._on_motion_done)
        self._move_thread.start()

    def on_jog(self, direction: int):
        if not self.stage:
            return
        self._moving = True
        self._sync_controls()
        step = float(self.stepSpin.value())
        accel = float(self.accelSpin.value())

        self._move_controller = MotionController()
        target = self.stage.get_position() + (step if direction >= 0 else -step)
        self._move_thread = MoveThread(self.stage, target, step, accel, self._move_controller)
        self._move_thread.progress.connect(self._on_motion_progress)
        self._move_thread.status.connect(self.statusLbl.setText)
        self._move_thread.done.connect(self._on_motion_done)
        self._move_thread.start()

    def _on_motion_progress(self, angle: float):
        self.posLbl.setText(f"{angle:.3f}")

    def _on_motion_done(self, status: str):
        self._moving = False
        self._sync_controls()
        if status == "ok":
            self.statusLbl.setText("Move OK")
        elif status == "aborted":
            self.statusLbl.setText("Move aborted")
        else:
            self.statusLbl.setText("Move error")

    def on_home(self):
        if not self.stage:
            return
        self._homing = True
        self._sync_controls()

        self._home_controller = MotionController()
        self._home_thread = HomeThread(self.stage, self._home_controller)
        self._home_thread.progress.connect(self._on_motion_progress)
        self._home_thread.status.connect(self.statusLbl.setText)
        self._home_thread.done.connect(self._on_home_done)
        self._home_thread.start()

    def _on_home_done(self, status: str):
        self._homing = False
        self._sync_controls()
        if status == "ok":
            self.statusLbl.setText("Home OK")
        elif status == "aborted":
            self.statusLbl.setText("Home aborted")
        else:
            self.statusLbl.setText("Home error")

    def on_sweep(self):
        if not self.stage:
            return
        self._sweeping = True
        self._sync_controls()

        cfg = SweepConfig(
            start_deg=float(self.sweepStartSpin.value()),
            stop_deg=float(self.sweepStopSpin.value()),
            step_deg=float(self.sweepStepSpin.value()),
            ramp_step_deg=float(self.sweepRampSpin.value()),
        )

        self._sweep_controller = MotionController()
        self._sweep_thread = SweepThread(self.stage, cfg, self._sweep_controller)
        self._sweep_thread.ramp_progress.connect(self._on_motion_progress)
        self._sweep_thread.progress.connect(self._on_sweep_progress)
        self._sweep_thread.status.connect(self.statusLbl.setText)
        self._sweep_thread.done.connect(self._on_sweep_done)
        self._sweep_thread.start()

    def _on_sweep_progress(self, idx: int, total: int, angle: float):
        self.posLbl.setText(f"{angle:.3f}")
        self.sweepProgLbl.setText(f"Sweep: {idx}/{total}")

    def _on_sweep_done(self, status: str):
        self._sweeping = False
        self._sync_controls()
        if status == "ok":
            self.sweepProgLbl.setText("Sweep: OK")
            self.statusLbl.setText("Sweep OK")
        elif status == "aborted":
            self.sweepProgLbl.setText("Sweep: aborted")
            self.statusLbl.setText("Sweep aborted")
        else:
            self.sweepProgLbl.setText("Sweep: error")
            self.statusLbl.setText("Sweep error")

    def on_stop_any(self):
        if self._move_controller:
            self._move_controller.abort()
        if self._home_controller:
            self._home_controller.abort()
        if self._sweep_controller:
            self._sweep_controller.abort()
        try:
            if self.stage:
                self.stage.stop()
        except Exception:
            pass

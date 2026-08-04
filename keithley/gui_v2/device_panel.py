import math
from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from ..keithley_wrapper import (
    KeithleySMU,
    SweepController,
    SweepConfig,
    VIReading,
    KEITHLEY_V_LIMIT,
)

from .config import (
    BTN_W,
    READ_W,
    STATUS_W,
    DEFAULT_ICOMP_NA,
    DEFAULT_SWEEP_ICOMP_NA,
    RAMP_STEP_V,
    RAMP_DWELL_S,
    SWEEP_SETTLE_MS,
    READOUT_UNKNOWN,
    POLL_ERROR_LIMIT,
    POLL_TIMEOUT_MS,
)
from .plots import PlotWidget
from .workers import RampThread, SingleSweepThread, HomeThread


class DevicePanel(QtWidgets.QGroupBox):
    def __init__(self, title: str, rm, default_resource: str, main_ref_callable):
        super().__init__(title)
        self.rm = rm
        self._get_main = main_ref_callable

        self.dev: Optional[KeithleySMU] = None

        self.sweep_thread: Optional[SingleSweepThread] = None
        self.home_thread: Optional[HomeThread] = None
        self.ramp_thread: Optional[RampThread] = None

        self.sweeping = False
        self.homing = False
        self.ramping = False

        self._sweep_controller: Optional[SweepController] = None
        self._home_controller: Optional[SweepController] = None
        self._ramp_controller: Optional[SweepController] = None
        self._controls_locked = False
        self._poll_failures = 0

        self._build_ui(default_resource)

    def _build_ui(self, default_resource: str):
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self.resourceEdit = QtWidgets.QLineEdit(default_resource)
        self.resourceEdit.setMinimumWidth(340)

        self.connectBtn = QtWidgets.QPushButton("Connect"); self.connectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect"); self.disconnectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("VISA Resource:"), 0, 0)
        layout.addWidget(self.resourceEdit, 0, 1, 1, 4)
        layout.addWidget(self.connectBtn, 0, 5)
        layout.addWidget(self.disconnectBtn, 0, 6)

        # Set controls
        self.vsetSpin = QtWidgets.QDoubleSpinBox()
        self.vsetSpin.setDecimals(6)
        self.vsetSpin.setRange(-200.0, 200.0)
        self.vsetSpin.setSingleStep(0.01)
        self.vsetSpin.setValue(0.0)
        self.vsetSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.icompSpin_nA = QtWidgets.QDoubleSpinBox()
        self.icompSpin_nA.setDecimals(3)
        self.icompSpin_nA.setRange(0.0, 1e9)
        self.icompSpin_nA.setSingleStep(0.1)
        self.icompSpin_nA.setValue(DEFAULT_ICOMP_NA)

        self.outputChk = QtWidgets.QCheckBox("Output ON")
        self.outputChk.setEnabled(False)

        self.applyBtn = QtWidgets.QPushButton("Apply (ramped)")
        self.applyBtn.setFixedWidth(BTN_W)
        self.applyBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("V set (V):"), 1, 0)
        layout.addWidget(self.vsetSpin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("I comp (nA):"), 1, 2)
        layout.addWidget(self.icompSpin_nA, 1, 3)
        layout.addWidget(self.outputChk, 1, 4)
        layout.addWidget(self.applyBtn, 1, 5, 1, 2)

        # Readouts
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        self.vreadVal = QtWidgets.QLabel(READOUT_UNKNOWN); self.vreadVal.setFont(mono)
        self.vreadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.vreadVal.setMinimumWidth(READ_W)

        self.ireadVal = QtWidgets.QLabel(READOUT_UNKNOWN); self.ireadVal.setFont(mono)
        self.ireadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.ireadVal.setMinimumWidth(READ_W)

        self.statusLbl = QtWidgets.QLabel("Disconnected.")
        self.statusLbl.setStyleSheet("color:#666;")
        self.statusLbl.setMinimumWidth(STATUS_W)

        layout.addWidget(QtWidgets.QLabel("V-read (V):"), 2, 0)
        layout.addWidget(self.vreadVal, 2, 1)
        layout.addWidget(QtWidgets.QLabel("I-read (nA):"), 2, 2)
        layout.addWidget(self.ireadVal, 2, 3)
        layout.addWidget(self.statusLbl, 2, 4, 1, 3)

        # Sweep controls (manual stepping)
        line = 3
        self.v0Spin = QtWidgets.QDoubleSpinBox()
        self.v0Spin.setDecimals(6); self.v0Spin.setRange(-210.0, 210.0); self.v0Spin.setValue(0.0)
        self.v0Spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.stepSpin = QtWidgets.QDoubleSpinBox()
        self.stepSpin.setDecimals(6); self.stepSpin.setRange(-50.0, 50.0); self.stepSpin.setValue(0.01)
        self.stepSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.nSpin = QtWidgets.QSpinBox()
        self.nSpin.setRange(1, 100000); self.nSpin.setValue(10)

        self.vEndLbl = QtWidgets.QLabel("End: 0.000000 V")
        self.vEndLbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.sweepItripSpin_nA = QtWidgets.QDoubleSpinBox()
        self.sweepItripSpin_nA.setDecimals(3)
        self.sweepItripSpin_nA.setRange(0.0, 1e9)
        self.sweepItripSpin_nA.setSingleStep(0.1)
        self.sweepItripSpin_nA.setValue(DEFAULT_SWEEP_ICOMP_NA)

        layout.addWidget(QtWidgets.QLabel("Sweep V0 (V):"), line, 0)
        layout.addWidget(self.v0Spin, line, 1)
        layout.addWidget(QtWidgets.QLabel("Step (V):"), line, 2)
        layout.addWidget(self.stepSpin, line, 3)
        layout.addWidget(QtWidgets.QLabel("Steps:"), line, 4)
        layout.addWidget(self.nSpin, line, 5)
        layout.addWidget(self.vEndLbl, line, 6)

        line2 = line + 1
        self.startSweepBtn = QtWidgets.QPushButton("Start Sweep")
        self.startSweepBtn.setFixedWidth(BTN_W); self.startSweepBtn.setEnabled(False)

        self.nextStepBtn = QtWidgets.QPushButton("Next step")
        self.nextStepBtn.setFixedWidth(BTN_W); self.nextStepBtn.setEnabled(False)

        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.stopBtn.setFixedWidth(BTN_W); self.stopBtn.setEnabled(False)

        self.homeBtn = QtWidgets.QPushButton("Home -> 0 V")
        self.homeBtn.setFixedWidth(BTN_W); self.homeBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Sweep trip |I| (nA):"), line2, 0)
        layout.addWidget(self.sweepItripSpin_nA, line2, 1)
        layout.addWidget(self.homeBtn, line2, 4)
        layout.addWidget(self.startSweepBtn, line2, 5)
        layout.addWidget(self.stopBtn, line2, 6)

        line3 = line2 + 1
        self.bannerLbl = QtWidgets.QLabel("")
        self.bannerLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.bannerLbl.setStyleSheet("color: red; font-weight: 700; font-size: 12pt;")
        layout.addWidget(self.bannerLbl, line3, 0, 1, 7)

        # Plot
        line4 = line3 + 1
        self.plot = PlotWidget(self)
        self.clearPlotBtn = QtWidgets.QPushButton("Clear Plot"); self.clearPlotBtn.setFixedWidth(BTN_W)
        layout.addWidget(self.plot, line4, 0, 1, 6)
        layout.addWidget(self.clearPlotBtn, line4, 6)

        layout.addWidget(self.nextStepBtn, line4 + 1, 6)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(6, 1)
        layout.setRowStretch(line4, 1)

        # Signals
        self.connectBtn.clicked.connect(self.on_connect)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.applyBtn.clicked.connect(self.on_apply_threaded)
        self.outputChk.toggled.connect(self.on_output_toggled)

        self.v0Spin.valueChanged.connect(self._update_vend)
        self.stepSpin.valueChanged.connect(self._update_vend)
        self.nSpin.valueChanged.connect(self._update_vend)

        self.startSweepBtn.clicked.connect(self.on_start_sweep)
        self.stopBtn.clicked.connect(self.on_stop_any)
        self.nextStepBtn.clicked.connect(self.on_next_step)

        self.homeBtn.clicked.connect(self.on_home)
        self.clearPlotBtn.clicked.connect(self.plot.clear)

        self._update_vend()

    def _vend_value(self) -> float:
        n = int(self.nSpin.value())
        return float(self.v0Spin.value() + (n - 1) * self.stepSpin.value())

    def _vend_ok(self, v_end: Optional[float] = None) -> bool:
        if v_end is None:
            v_end = self._vend_value()
        return (
            abs(float(self.v0Spin.value())) <= KEITHLEY_V_LIMIT + 1e-12
            and abs(v_end) <= KEITHLEY_V_LIMIT + 1e-12
        )

    def _set_start_sweep_enabled(self, ok: Optional[bool] = None) -> None:
        if self._controls_locked:
            self.startSweepBtn.setEnabled(False)
            return
        if ok is None:
            ok = self._vend_ok()
        self.startSweepBtn.setEnabled(self.dev is not None and ok and not self.sweeping and not self.homing and not self.ramping)

    def _update_vend(self):
        v_end = self._vend_value()
        self.vEndLbl.setText(f"End: {v_end:.6f} V")
        ok = self._vend_ok(v_end)
        self.vEndLbl.setStyleSheet("" if ok else "color: red; font-weight: 600;")
        self._set_start_sweep_enabled(ok)

    def _set_readouts(self, v: Optional[float], iA: Optional[float], *, status: Optional[str] = None, plot: bool = False) -> None:
        if (v is None) or (iA is None):
            self.vreadVal.setText(READOUT_UNKNOWN)
            self.ireadVal.setText(READOUT_UNKNOWN)
        else:
            self.vreadVal.setText(f"{v:.6g}")
            self.ireadVal.setText(f"{iA*1e9:.6g}")
            if plot:
                self.plot.add_point(v, iA)
        if status is not None:
            self.statusLbl.setText(status)

    def _sync_controls(self) -> None:
        if self._controls_locked:
            return
        connected = self.dev is not None
        busy = self.sweeping or self.homing or self.ramping

        self.connectBtn.setEnabled((not connected) and (not busy))
        self.disconnectBtn.setEnabled(connected and (not busy))
        self.applyBtn.setEnabled(connected and (not busy))
        self.outputChk.setEnabled(connected and (not busy))
        self.homeBtn.setEnabled(connected and (not busy))
        self.stopBtn.setEnabled(busy)
        self.nextStepBtn.setEnabled(self.sweeping and busy)
        self._set_start_sweep_enabled()

    def set_controls_locked(self, locked: bool) -> None:
        self._controls_locked = locked
        if locked:
            for w in (
                self.connectBtn, self.disconnectBtn, self.applyBtn, self.outputChk,
                self.homeBtn, self.startSweepBtn, self.stopBtn, self.nextStepBtn,
            ):
                w.setEnabled(False)
        else:
            self._sync_controls()

    # ---- Connection / control ----

    def on_connect(self):
        res = self.resourceEdit.text().strip()
        if not res:
            QtWidgets.QMessageBox.critical(self, "Error", "Enter a VISA resource string.")
            return

        self._get_main().pause_polling()
        try:
            dev = KeithleySMU(self.rm, res, timeout_ms=20000, query_delay_s=0.0, verbose=False)
            idn = dev.open()

            self.dev = dev
            self._controls_locked = False
            self._poll_failures = 0

            try:
                self.dev.set_compliance(float(self.icompSpin_nA.value()) * 1e-9)
            except Exception:
                pass

            try:
                self.dev.set_output(True)
                self.outputChk.blockSignals(True)
                self.outputChk.setChecked(True)
                self.outputChk.blockSignals(False)
            except Exception:
                self.outputChk.blockSignals(True)
                self.outputChk.setChecked(False)
                self.outputChk.blockSignals(False)

            self.bannerLbl.setText("")
            self.statusLbl.setText(("Connected: " + (idn or "(unknown)"))[:60])

        except Exception as e:
            self.statusLbl.setText("Conn failed")
            QtWidgets.QMessageBox.critical(self, "Connection failed", str(e))
            self.dev = None
        finally:
            self._sync_controls()
            self._update_vend()
            self._get_main().update_dual_controls()
            self._get_main().resume_polling_if_idle()

    def on_disconnect(self):
        self._disconnect_device(reason="Disconnected.", attempt_output_off=True)

    def on_output_toggled(self, checked: bool):
        if not self.dev:
            return
        try:
            self.dev.set_output(bool(checked))
            self.statusLbl.setText("Output " + ("ON" if checked else "OFF"))
            if not checked:
                self._set_readouts(None, None)
        except Exception as e:
            self.outputChk.blockSignals(True)
            self.outputChk.setChecked(not checked)
            self.outputChk.blockSignals(False)
            self._handle_device_lost("Device not responding (disconnected or local mode).")
            QtWidgets.QMessageBox.critical(self, "SCPI error", f"Failed to toggle output:\n{e}")

    # ---- Polling ----

    def poll_once(self, oc_limit_a: Optional[float], oc_trip_samples: int, allow_poll: bool) -> Tuple[Optional[float], Optional[float], bool]:
        if not self.dev or not allow_poll:
            return (None, None, False)
        if self.sweeping or self.homing or self.ramping:
            return (None, None, False)

        try:
            vi = self.dev.read_vi_with_timeout(POLL_TIMEOUT_MS)
            v = float(vi.v)
            iA = float(vi.i)

            self._set_readouts(v, iA, plot=True)
            self._poll_failures = 0

            tripped = False
            if oc_limit_a is not None and (not math.isnan(iA)) and abs(iA) > oc_limit_a:
                self._get_main().panel_oc_hit(self, iA, oc_limit_a, oc_trip_samples)
                tripped = True
            return (v, iA, tripped)
        except Exception:
            self._poll_failures += 1
            if self._poll_failures >= POLL_ERROR_LIMIT:
                self._handle_device_lost("Device not responding (disconnected or local mode).")
            else:
                self.statusLbl.setText("Poll error")
            return (None, None, False)

    # ---- Apply (threaded ramp) ----

    def on_apply_threaded(self):
        if not self.dev:
            return

        vset = float(self.vsetSpin.value())
        icomp_nA = float(self.icompSpin_nA.value())

        try:
            self.dev.set_compliance(icomp_nA * 1e-9)
        except Exception as e:
            self._handle_device_lost("Device not responding (disconnected or local mode).")
            QtWidgets.QMessageBox.critical(self, "SCPI error", f"Failed to set compliance:\n{e}")
            return

        try:
            self.dev.set_output(True)
            self.outputChk.blockSignals(True)
            self.outputChk.setChecked(True)
            self.outputChk.blockSignals(False)
        except Exception as e:
            self._handle_device_lost("Device not responding (disconnected or local mode).")
            QtWidgets.QMessageBox.critical(self, "SCPI error", f"Failed to enable output:\n{e}")
            return

        self.bannerLbl.setText("")
        self.ramping = True
        self._get_main().pause_polling()
        self._sync_controls()

        self._ramp_controller = SweepController()
        self.ramp_thread = RampThread(self.dev, vset, self._ramp_controller)
        self.ramp_thread.status.connect(self.statusLbl.setText)
        self.ramp_thread.progress.connect(self._on_ramp_progress)
        self.ramp_thread.done.connect(self._on_ramp_done)
        self.ramp_thread.start()

    def _on_ramp_progress(self, v: float, iA: float):
        self._set_readouts(v, iA, plot=True)

    def _on_ramp_done(self, status: str):
        self.ramping = False
        self._get_main().resume_polling_if_idle()

        self._sync_controls()
        self._update_vend()

        if status == "ok":
            self.statusLbl.setText("Applied (ramped)")
        elif status == "aborted":
            self.statusLbl.setText("Ramp aborted")
        else:
            self.bannerLbl.setText(status)
            self.statusLbl.setText("Ramp error")

        self._get_main().update_dual_controls()

    # ---- Single sweep ----

    def on_start_sweep(self):
        if not self.dev:
            QtWidgets.QMessageBox.warning(self, "Not connected", "Connect the device first.")
            return
        if self.ramping or self.homing:
            QtWidgets.QMessageBox.warning(self, "Busy", "Stop ramp/home first.")
            return

        v0 = float(self.v0Spin.value())
        step = float(self.stepSpin.value())
        n = int(self.nSpin.value())
        vend = self._vend_value()
        if abs(v0) > KEITHLEY_V_LIMIT or abs(vend) > KEITHLEY_V_LIMIT:
            QtWidgets.QMessageBox.critical(self, "Voltage limit", f"Start/End exceed +/-{KEITHLEY_V_LIMIT:.0f} V")
            return

        icomp_trip_a = float(self.sweepItripSpin_nA.value()) * 1e-9

        self.bannerLbl.setText("")
        self.sweeping = True
        self._get_main().pause_polling()
        self._sync_controls()

        self._sweep_controller = SweepController()

        cfg = SweepConfig(
            v0=v0,
            step=step,
            nsteps=n,
            icomp_limit_a=icomp_trip_a,
            manual=True,
            dwell_ms=0,
            settle_ms=SWEEP_SETTLE_MS,
            ramp_step_v=RAMP_STEP_V,
            ramp_dwell_s=RAMP_DWELL_S,
        )

        self.sweep_thread = SingleSweepThread(self.dev, cfg, self._sweep_controller)
        self.sweep_thread.ramp_progress.connect(self._on_sweep_ramp_progress)
        self.sweep_thread.progress.connect(self._on_sweep_progress)
        self.sweep_thread.status.connect(self._on_sweep_status)
        self.sweep_thread.done.connect(self._on_sweep_done)
        self.sweep_thread.start()

    def on_next_step(self):
        if self._sweep_controller and self.sweeping:
            self._sweep_controller.next_step()

    def _on_sweep_status(self, msg: str):
        self.statusLbl.setText(msg[:60])

    def _on_sweep_ramp_progress(self, v: float, iA: float):
        self._set_readouts(v, iA, plot=True)

    def _on_sweep_progress(self, idx: int, v: float, iA: float):
        self._set_readouts(v, iA, status=f"Sweep step {idx}", plot=True)

    def _on_sweep_done(self, status: str):
        self.sweeping = False
        self._get_main().resume_polling_if_idle()

        self._sync_controls()
        self._update_vend()

        if status == "ok":
            self.statusLbl.setText("Sweep OK")
        elif status == "aborted":
            self.statusLbl.setText("Sweep aborted")
        elif status == "trip":
            self.bannerLbl.setText("SWEEP TRIPPED (current limit exceeded)")
            self.statusLbl.setText("Sweep TRIPPED")
        else:
            self.bannerLbl.setText(status)
            self.statusLbl.setText("Sweep error")

        self._get_main().update_dual_controls()

    # ---- Home ----

    def on_home(self):
        if not self.dev:
            return
        if self.sweeping or self.ramping:
            QtWidgets.QMessageBox.warning(self, "Busy", "Stop sweep/ramp first.")
            return

        self.bannerLbl.setText("")
        self.homing = True
        self._get_main().pause_polling()
        self._sync_controls()

        self._home_controller = SweepController()
        self.home_thread = HomeThread(self.dev, self._home_controller)
        self.home_thread.progress.connect(self._on_home_progress)
        self.home_thread.status.connect(self.statusLbl.setText)
        self.home_thread.done.connect(self._on_home_done)
        self.home_thread.start()

    def _on_home_progress(self, v: float, iA: float):
        self._set_readouts(v, iA, plot=True)

    def _on_home_done(self, status: str):
        self.homing = False
        self._get_main().resume_polling_if_idle()

        self._sync_controls()
        self._update_vend()
        self.statusLbl.setText("Home: " + status)
        self._get_main().update_dual_controls()

    # ---- Stop ----

    def on_stop_any(self):
        if self._ramp_controller:
            self._ramp_controller.abort()
        if self._sweep_controller:
            self._sweep_controller.abort()
        if self._home_controller:
            self._home_controller.abort()

        self.stopBtn.setEnabled(False)
        self.nextStepBtn.setEnabled(False)

    def _handle_device_lost(self, reason: str):
        self._disconnect_device(reason=reason, attempt_output_off=False)

    def _disconnect_device(self, *, reason: str, attempt_output_off: bool):
        self.on_stop_any()

        try:
            if self.dev:
                if attempt_output_off:
                    try:
                        self.dev.set_output(False)
                    except Exception:
                        pass
                self.dev.close()
        except Exception:
            pass

        self.dev = None
        self._controls_locked = False
        self._poll_failures = 0
        self._set_readouts(None, None)
        self.statusLbl.setText(reason)
        self.bannerLbl.setText("")

        self._sync_controls()
        self._update_vend()
        self._get_main().update_dual_controls()

import csv
from datetime import datetime
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from ..keithley_wrapper import (
    make_resource_manager,
    SweepController,
    SweepConfig,
    DualSweepConfig,
    KEITHLEY_V_LIMIT,
)

from .config import (
    VISA_DLL,
    DEFAULT_A_RESOURCE,
    DEFAULT_B_RESOURCE,
    POLL_MS_DEFAULT,
    BTN_W,
    RAMP_STEP_V,
    RAMP_DWELL_S,
    SWEEP_SETTLE_MS,
)
from .device_panel import DevicePanel
from .workers import DualSweepThread


class DualKeithleyWidget(QtWidgets.QWidget):
    def __init__(
        self,
        visa_dll: str = VISA_DLL,
        default_a_resource: str = DEFAULT_A_RESOURCE,
        default_b_resource: str = DEFAULT_B_RESOURCE,
        parent=None,
    ):
        super().__init__(parent)

        try:
            self.rm = make_resource_manager(visa_dll)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "VISA Error", f"Could not create ResourceManager:\n{e}")
            raise

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Top bar
        top = QtWidgets.QHBoxLayout()
        vbox.addLayout(top)

        self.ocSpin = QtWidgets.QDoubleSpinBox()
        self.ocSpin.setDecimals(9)
        self.ocSpin.setRange(0.0, 10.0)
        self.ocSpin.setSingleStep(0.001)
        self.ocSpin.setValue(0.01)

        self.ocTripSamplesSpin = QtWidgets.QSpinBox()
        self.ocTripSamplesSpin.setRange(1, 10)
        self.ocTripSamplesSpin.setValue(2)

        self.pollSpin = QtWidgets.QSpinBox()
        self.pollSpin.setRange(100, 2000)
        self.pollSpin.setSingleStep(50)
        self.pollSpin.setValue(POLL_MS_DEFAULT)

        self.logChk = QtWidgets.QCheckBox("Log CSV")
        self.logBtn = QtWidgets.QPushButton("Choose file..."); self.logBtn.setFixedWidth(BTN_W)
        self.logLbl = QtWidgets.QLabel("Logging: OFF"); self.logLbl.setMinimumWidth(260)

        self.dualSweepBtn = QtWidgets.QPushButton("Start Dual Sweep"); self.dualSweepBtn.setFixedWidth(150)
        self.dualNextBtn = QtWidgets.QPushButton("Next (Dual)"); self.dualNextBtn.setFixedWidth(BTN_W)
        self.dualStopBtn = QtWidgets.QPushButton("Stop Dual"); self.dualStopBtn.setFixedWidth(BTN_W)
        self.dualSweepBtn.setEnabled(False)
        self.dualNextBtn.setEnabled(False)
        self.dualStopBtn.setEnabled(False)

        self.panicBtn = QtWidgets.QPushButton("All OFF (Panic)"); self.panicBtn.setFixedWidth(150)

        top.addWidget(QtWidgets.QLabel("Software OC (A):"))
        top.addWidget(self.ocSpin)
        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("Trip samples:"))
        top.addWidget(self.ocTripSamplesSpin)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Poll (ms):"))
        top.addWidget(self.pollSpin)
        top.addStretch()
        top.addWidget(self.logChk); top.addWidget(self.logBtn); top.addWidget(self.logLbl)
        top.addSpacing(10)
        top.addWidget(self.dualSweepBtn); top.addWidget(self.dualNextBtn); top.addWidget(self.dualStopBtn)
        top.addSpacing(10)
        top.addWidget(self.panicBtn)

        # Panels
        panels = QtWidgets.QHBoxLayout()
        vbox.addLayout(panels)

        self.panelA = DevicePanel("Source A", self.rm, default_a_resource, main_ref_callable=lambda: self)
        self.panelB = DevicePanel("Source B", self.rm, default_b_resource, main_ref_callable=lambda: self)
        panels.addWidget(self.panelA, 1)
        panels.addWidget(self.panelB, 1)

        # Status bar
        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)
        self.statusBar().showMessage("Idle")

        # Timer for polling
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_poll)
        self.timer.start(self.pollSpin.value())

        # Logging
        self.log_path: Optional[str] = None
        self.log_file = None
        self.log_writer = None

        # Dual sweep
        self.dual_thread: Optional[DualSweepThread] = None
        self._dual_controller: Optional[SweepController] = None

        # Signals
        self.panicBtn.clicked.connect(self.on_panic)
        self.pollSpin.valueChanged.connect(self.on_poll_interval_changed)

        self.logChk.toggled.connect(self.on_toggle_logging)
        self.logBtn.clicked.connect(self.choose_log_file)

        self.dualSweepBtn.clicked.connect(self.on_start_dual_sweep)
        self.dualNextBtn.clicked.connect(self.on_next_dual_step)
        self.dualStopBtn.clicked.connect(self.on_stop_dual_sweep)

        # update dual controls whenever user changes sweep settings
        for w in (self.panelA.nSpin, self.panelB.nSpin, self.panelA.v0Spin, self.panelA.stepSpin, self.panelB.v0Spin, self.panelB.stepSpin):
            w.valueChanged.connect(self.update_dual_controls)

        self.update_dual_controls()

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    # ---- polling pause/resume ----

    def _panel_pollable(self, panel: DevicePanel) -> bool:
        return panel.dev is not None and not (panel.sweeping or panel.homing or panel.ramping)

    def _should_poll_timer(self) -> bool:
        if self.dual_thread is not None:
            return False
        return self._panel_pollable(self.panelA) or self._panel_pollable(self.panelB)

    def _refresh_polling_state(self):
        should_run = self._should_poll_timer()
        if should_run and not self.timer.isActive():
            self.timer.start(self.pollSpin.value())
            self.statusBar().showMessage("Polling resumed.")
        elif (not should_run) and self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("Polling paused...")

    def pause_polling(self):
        self._refresh_polling_state()

    def resume_polling_if_idle(self):
        self._refresh_polling_state()

    # ---- software OC helper ----

    def panel_oc_hit(self, panel: DevicePanel, iA: float, limit_a: float, samples_needed: int):
        try:
            if panel.dev:
                panel.dev.set_output(False)
        except Exception:
            pass
        panel.outputChk.blockSignals(True)
        panel.outputChk.setChecked(False)
        panel.outputChk.blockSignals(False)
        panel.bannerLbl.setText(f"SOFTWARE OC: |I|={abs(iA)*1e9:.6g} nA > {abs(limit_a)*1e9:.6g} nA")
        panel.statusLbl.setText("OC trip: output OFF")

    # ---- logging ----

    def choose_log_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose log CSV", filter="CSV Files (*.csv);;All Files (*)")
        if path:
            self.log_path = path
            if self.logChk.isChecked():
                self.open_log()

    def open_log(self):
        self.close_log()
        if not self.log_path:
            QtWidgets.QMessageBox.information(self, "Logging", "Choose a CSV file first.")
            self.logChk.setChecked(False)
            return
        try:
            self.log_file = open(self.log_path, "a", newline="")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(["timestamp", "panel", "Vset(V)", "Icomp(A)", "Vread(V)", "Iread(A)", "output"])
            self.logLbl.setText("Logging: ON")
            self.statusBar().showMessage(f"Logging -> {self.log_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Log error", f"Cannot open log file:\n{e}")
            self.logChk.setChecked(False)

    def close_log(self):
        try:
            if self.log_file:
                self.log_file.flush()
                self.log_file.close()
        except Exception:
            pass
        self.log_file = None
        self.log_writer = None
        self.logLbl.setText("Logging: OFF")

    def on_toggle_logging(self, checked: bool):
        if checked:
            if not self.log_path:
                self.choose_log_file()
                if not self.log_path:
                    self.logChk.setChecked(False)
                    return
            self.open_log()
        else:
            self.close_log()

    # ---- panic ----

    def on_panic(self):
        n = 0
        for panel in (self.panelA, self.panelB):
            try:
                if panel.dev:
                    panel.dev.set_output(False)
                    panel.outputChk.blockSignals(True)
                    panel.outputChk.setChecked(False)
                    panel.outputChk.blockSignals(False)
                    n += 1
            except Exception:
                pass
        QtWidgets.QMessageBox.information(self, "Panic", f"Outputs disabled on {n} connected device(s).")

    # ---- polling ----

    def on_poll_interval_changed(self, ms: int):
        if self.timer.isActive():
            self.timer.setInterval(ms)
        self.statusBar().showMessage(f"Polling @ {ms} ms")

    def on_poll(self):
        oc_limit_a = float(self.ocSpin.value())
        trip_samples = int(self.ocTripSamplesSpin.value())
        ms = int(self.pollSpin.value())

        allow_poll = (self.dual_thread is None)

        for tag, panel in (("A", self.panelA), ("B", self.panelB)):
            v, iA, _ = panel.poll_once(oc_limit_a, trip_samples, allow_poll=allow_poll)

            if (v is not None) and (iA is not None) and self.log_writer and panel.dev:
                vset = float(panel.vsetSpin.value())
                icomp_A = float(panel.icompSpin_nA.value()) * 1e-9
                out = 1 if panel.outputChk.isChecked() else 0
                self.log_writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    tag, vset, icomp_A, v, iA, out
                ])
                try:
                    self.log_file.flush()
                except Exception:
                    pass

        self.statusBar().showMessage(f"Polling @ {ms} ms")
        self.update_dual_controls()

    # ---- dual sweep controls ----

    def update_dual_controls(self):
        a_ok = (self.panelA.dev is not None)
        b_ok = (self.panelB.dev is not None)
        same_n = int(self.panelA.nSpin.value()) == int(self.panelB.nSpin.value())

        within = lambda p: (abs(float(p.v0Spin.value())) <= KEITHLEY_V_LIMIT + 1e-12) and (abs(float(p._vend_value())) <= KEITHLEY_V_LIMIT + 1e-12)

        enable = (
            a_ok and b_ok and same_n and within(self.panelA) and within(self.panelB)
            and not self.panelA.sweeping and not self.panelB.sweeping
            and not self.panelA.homing and not self.panelB.homing
            and not self.panelA.ramping and not self.panelB.ramping
            and self.dual_thread is None
        )
        self.dualSweepBtn.setEnabled(enable)

    def on_start_dual_sweep(self):
        if not (self.panelA.dev and self.panelB.dev):
            QtWidgets.QMessageBox.warning(self, "Not connected", "Connect both devices first.")
            return
        if self.panelA.ramping or self.panelB.ramping or self.panelA.homing or self.panelB.homing:
            QtWidgets.QMessageBox.warning(self, "Busy", "Stop ramp/home first.")
            return

        n = int(self.panelA.nSpin.value())
        if n != int(self.panelB.nSpin.value()):
            QtWidgets.QMessageBox.warning(self, "Steps mismatch", "Set the same number of steps for both devices.")
            return

        v0A, stepA = float(self.panelA.v0Spin.value()), float(self.panelA.stepSpin.value())
        v0B, stepB = float(self.panelB.v0Spin.value()), float(self.panelB.stepSpin.value())

        vendA = v0A + (n - 1) * stepA
        vendB = v0B + (n - 1) * stepB
        if any(abs(v) > KEITHLEY_V_LIMIT for v in (v0A, vendA, v0B, vendB)):
            QtWidgets.QMessageBox.critical(self, "Voltage limit", f"Start/End exceed +/-{KEITHLEY_V_LIMIT:.0f} V")
            return

        self.panelA.bannerLbl.setText("")
        self.panelB.bannerLbl.setText("")
        self.panelA.sweeping = True
        self.panelB.sweeping = True

        self.pause_polling()

        for p in (self.panelA, self.panelB):
            p.set_controls_locked(True)

        self.dualSweepBtn.setEnabled(False)
        self.dualNextBtn.setEnabled(True)
        self.dualStopBtn.setEnabled(True)

        self._dual_controller = SweepController()

        cfgA = SweepConfig(
            v0=v0A, step=stepA, nsteps=n,
            icomp_limit_a=float(self.panelA.sweepItripSpin_nA.value()) * 1e-9,
            manual=True, dwell_ms=0, settle_ms=SWEEP_SETTLE_MS,
            ramp_step_v=RAMP_STEP_V, ramp_dwell_s=RAMP_DWELL_S
        )
        cfgB = SweepConfig(
            v0=v0B, step=stepB, nsteps=n,
            icomp_limit_a=float(self.panelB.sweepItripSpin_nA.value()) * 1e-9,
            manual=True, dwell_ms=0, settle_ms=SWEEP_SETTLE_MS,
            ramp_step_v=RAMP_STEP_V, ramp_dwell_s=RAMP_DWELL_S
        )
        dcfg = DualSweepConfig(
            a=cfgA, b=cfgB, nsteps=n,
            manual=True, dwell_ms=0,
            ramp_step_v=RAMP_STEP_V, ramp_dwell_s=RAMP_DWELL_S
        )

        self.dual_thread = DualSweepThread(self.panelA.dev, self.panelB.dev, dcfg, self._dual_controller)
        self.dual_thread.status.connect(self.statusBar().showMessage)
        self.dual_thread.ramp_progress.connect(self._on_dual_ramp_progress)
        self.dual_thread.progress.connect(self._on_dual_progress)
        self.dual_thread.done.connect(self._on_dual_done)
        self.dual_thread.start()

    def on_next_dual_step(self):
        if self._dual_controller:
            self._dual_controller.next_step()

    def on_stop_dual_sweep(self):
        if self._dual_controller:
            self._dual_controller.abort()
        self.dualStopBtn.setEnabled(False)
        self.dualNextBtn.setEnabled(False)

    def _on_dual_ramp_progress(self, tag: str, v: float, iA: float):
        if tag == "A":
            panel = self.panelA
        else:
            panel = self.panelB
        panel._set_readouts(v, iA, plot=True)

    def _on_dual_progress(self, idx: int, vA: float, iA: float, vB: float, iB: float):
        self.panelA._set_readouts(vA, iA, status=f"A step {idx}", plot=True)
        self.panelB._set_readouts(vB, iB, status=f"B step {idx}", plot=True)

        self.statusBar().showMessage(f"Dual sweep step {idx}")

    def _on_dual_done(self, status: str):
        self.panelA.sweeping = False
        self.panelB.sweeping = False

        self.dualNextBtn.setEnabled(False)
        self.dualStopBtn.setEnabled(False)

        for p in (self.panelA, self.panelB):
            p.set_controls_locked(False)
            p._update_vend()

        self.dual_thread = None
        self._dual_controller = None

        self.update_dual_controls()
        self.resume_polling_if_idle()

        if status == "ok":
            self.statusBar().showMessage("Dual sweep complete.")
        elif status == "aborted":
            self.statusBar().showMessage("Dual sweep aborted.")
        elif status == "trip":
            self.panelA.bannerLbl.setText("DUAL SWEEP TRIPPED")
            self.panelB.bannerLbl.setText("DUAL SWEEP TRIPPED")
            self.statusBar().showMessage("Dual sweep TRIPPED")
        else:
            self.statusBar().showMessage("Dual sweep " + status)

    # ---- close ----

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if self.dual_thread and self.dual_thread.isRunning():
                self.on_stop_dual_sweep()
        except Exception:
            pass

        try:
            self.timer.stop()
        except Exception:
            pass

        self.close_log()

        for panel in (self.panelA, self.panelB):
            try:
                panel.on_disconnect()
            except Exception:
                pass

        event.accept()

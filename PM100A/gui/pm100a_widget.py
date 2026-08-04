import math
from datetime import datetime
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from PM100A.pm100a_wrapper import PM100A, PM100AError, PM100A as _PM100A

BTN_W = 100
POLL_INTERVAL_MS = 500


class ZeroThread(QtCore.QThread):
    done = QtCore.pyqtSignal(str)  # "ok" or "error: <msg>"

    def __init__(self, pm: _PM100A, parent=None):
        super().__init__(parent)
        self._pm = pm

    def run(self):
        try:
            self._pm.zero()
            self.done.emit("ok")
        except Exception as e:
            self.done.emit(f"error: {e}")


class PM100AWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str = "PM100A Power Meter", parent=None):
        super().__init__(title, parent)
        self._pm: Optional[PM100A] = None
        self._zeroing = False
        self._zero_thread: Optional[ZeroThread] = None

        # Statistics state
        self._stats_values: List[float] = []
        self._stats_start: Optional[datetime] = None

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        # Row 0: resource string + scan/connect/disconnect
        self._resourceEdit = QtWidgets.QLineEdit(PM100A.DEFAULT_RESOURCE)
        self._resourceEdit.setMinimumWidth(220)
        self._scanBtn = QtWidgets.QPushButton("Scan")
        self._scanBtn.setFixedWidth(BTN_W)
        self._connectBtn = QtWidgets.QPushButton("Connect")
        self._connectBtn.setFixedWidth(BTN_W)
        self._disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self._disconnectBtn.setFixedWidth(BTN_W)
        self._disconnectBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Resource:"), 0, 0)
        layout.addWidget(self._resourceEdit, 0, 1, 1, 2)
        layout.addWidget(self._scanBtn, 0, 3)
        layout.addWidget(self._connectBtn, 0, 4)
        layout.addWidget(self._disconnectBtn, 0, 5)

        # Row 1: status
        self._statusLbl = QtWidgets.QLabel("Disconnected.")
        self._statusLbl.setFont(mono)
        layout.addWidget(QtWidgets.QLabel("Status:"), 1, 0)
        layout.addWidget(self._statusLbl, 1, 1, 1, 5)

        # Row 2: power reading (large)
        self._powerLbl = QtWidgets.QLabel("---")
        power_font = QtGui.QFont(mono)
        power_font.setPointSize(mono.pointSize() * 3)
        power_font.setBold(True)
        self._powerLbl.setFont(power_font)
        self._powerLbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self._powerLbl, 2, 0, 1, 6)

        # Row 3: wavelength
        self._wavSpin = QtWidgets.QDoubleSpinBox()
        self._wavSpin.setRange(400.0, 1100.0)
        self._wavSpin.setDecimals(1)
        self._wavSpin.setSingleStep(1.0)
        self._wavSpin.setValue(532.0)
        self._wavSpin.setSuffix(" nm")
        self._setWavBtn = QtWidgets.QPushButton("Set")
        self._setWavBtn.setFixedWidth(BTN_W)
        self._setWavBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Wavelength:"), 3, 0)
        layout.addWidget(self._wavSpin, 3, 1)
        layout.addWidget(self._setWavBtn, 3, 2)

        # Row 4: range + auto-range
        self._rangeSpin = QtWidgets.QDoubleSpinBox()
        self._rangeSpin.setRange(0.1, 9999.0)
        self._rangeSpin.setDecimals(1)
        self._rangeSpin.setSingleStep(1.0)
        self._rangeSpin.setValue(1.0)
        self._rangeUnitCombo = QtWidgets.QComboBox()
        self._rangeUnitCombo.addItems(["nW", "\u03bcW", "mW", "W"])
        self._rangeUnitCombo.setCurrentText("mW")
        self._setRangeBtn = QtWidgets.QPushButton("Set Range")
        self._setRangeBtn.setFixedWidth(BTN_W)
        self._setRangeBtn.setEnabled(False)
        self._autoRangeChk = QtWidgets.QCheckBox("Auto Range")
        self._autoRangeChk.setChecked(True)
        self._autoRangeChk.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Range:"), 4, 0)
        layout.addWidget(self._rangeSpin, 4, 1)
        layout.addWidget(self._rangeUnitCombo, 4, 2)
        layout.addWidget(self._setRangeBtn, 4, 3)
        layout.addWidget(self._autoRangeChk, 4, 4)

        # Row 5: zero + averaging
        self._zeroBtn = QtWidgets.QPushButton("Zero Sensor")
        self._zeroBtn.setFixedWidth(BTN_W)
        self._zeroBtn.setEnabled(False)

        self._avgSpin = QtWidgets.QSpinBox()
        self._avgSpin.setRange(1, 5000)
        self._avgSpin.setValue(10)
        self._avgSpin.setEnabled(False)
        self._setAvgBtn = QtWidgets.QPushButton("Set Avg")
        self._setAvgBtn.setFixedWidth(BTN_W)
        self._setAvgBtn.setEnabled(False)

        layout.addWidget(self._zeroBtn, 5, 0)
        layout.addWidget(QtWidgets.QLabel("Averaging:"), 5, 2)
        layout.addWidget(self._avgSpin, 5, 3)
        layout.addWidget(self._setAvgBtn, 5, 4)

        # Row 6: beam diameter
        self._beamDiaSpin = QtWidgets.QDoubleSpinBox()
        self._beamDiaSpin.setRange(0.01, 100.0)
        self._beamDiaSpin.setDecimals(2)
        self._beamDiaSpin.setSingleStep(0.1)
        self._beamDiaSpin.setValue(1.0)
        self._beamDiaSpin.setSuffix(" mm")
        self._setBeamDiaBtn = QtWidgets.QPushButton("Set")
        self._setBeamDiaBtn.setFixedWidth(BTN_W)
        self._setBeamDiaBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Beam diameter:"), 6, 0)
        layout.addWidget(self._beamDiaSpin, 6, 1)
        layout.addWidget(self._setBeamDiaBtn, 6, 2)

        layout.setColumnStretch(5, 1)

        # Row 7: statistics group box
        stats_box = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QGridLayout(stats_box)
        stats_layout.setHorizontalSpacing(16)
        stats_layout.setVerticalSpacing(4)

        val_font = QtGui.QFont(mono)
        val_font.setPointSize(mono.pointSize() + 1)
        val_font.setBold(True)

        def _stat_pair(label_text):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setFont(mono)
            val = QtWidgets.QLabel("---")
            val.setFont(val_font)
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            val.setMinimumWidth(90)
            return lbl, val

        lbl0, self._statCurrentLbl = _stat_pair("Most Recent")
        lbl1, self._statStartLbl   = _stat_pair("Start")
        lbl2, self._statSamplesLbl = _stat_pair("Samples")
        lbl3, self._statMinLbl     = _stat_pair("Minimum")
        lbl4, self._statMaxLbl     = _stat_pair("Maximum")
        lbl5, self._statMeanLbl    = _stat_pair("Mean")
        lbl6, self._statStdLbl     = _stat_pair("Std Dev")

        self._resetStatsBtn = QtWidgets.QPushButton("Reset")
        self._resetStatsBtn.setFixedWidth(60)
        self._resetStatsBtn.setEnabled(False)

        # Row 0: Most Recent | Start | Samples
        stats_layout.addWidget(lbl0, 0, 0)
        stats_layout.addWidget(self._statCurrentLbl, 0, 1)
        stats_layout.addWidget(lbl1, 0, 2)
        stats_layout.addWidget(self._statStartLbl, 0, 3)
        stats_layout.addWidget(lbl2, 0, 4)
        stats_layout.addWidget(self._statSamplesLbl, 0, 5)
        # Row 1: Min | Max | Mean | Std Dev | Reset
        stats_layout.addWidget(lbl3, 1, 0)
        stats_layout.addWidget(self._statMinLbl, 1, 1)
        stats_layout.addWidget(lbl4, 1, 2)
        stats_layout.addWidget(self._statMaxLbl, 1, 3)
        stats_layout.addWidget(lbl5, 1, 4)
        stats_layout.addWidget(self._statMeanLbl, 1, 5)
        stats_layout.addWidget(lbl6, 1, 6)
        stats_layout.addWidget(self._statStdLbl, 1, 7)
        stats_layout.addWidget(self._resetStatsBtn, 1, 8, QtCore.Qt.AlignRight)

        layout.addWidget(stats_box, 7, 0, 1, 6)

        # Signals
        self._scanBtn.clicked.connect(self._on_scan)
        self._connectBtn.clicked.connect(self._on_connect)
        self._disconnectBtn.clicked.connect(self._on_disconnect)
        self._setWavBtn.clicked.connect(self._on_set_wavelength)
        self._setRangeBtn.clicked.connect(self._on_set_range)
        self._autoRangeChk.stateChanged.connect(self._on_auto_range_changed)
        self._zeroBtn.clicked.connect(self._on_zero)
        self._setAvgBtn.clicked.connect(self._on_set_averaging)
        self._setBeamDiaBtn.clicked.connect(self._on_set_beam_diameter)
        self._resetStatsBtn.clicked.connect(self._reset_stats)

    # ---- Control state ----

    def _sync_controls(self):
        connected = self._pm is not None
        busy = self._zeroing
        self._connectBtn.setEnabled(not connected and not busy)
        self._scanBtn.setEnabled(not connected and not busy)
        self._resourceEdit.setEnabled(not connected and not busy)
        self._disconnectBtn.setEnabled(connected and not busy)
        self._setWavBtn.setEnabled(connected and not busy)
        manual = connected and not busy and not self._autoRangeChk.isChecked()
        self._setRangeBtn.setEnabled(manual)
        self._rangeUnitCombo.setEnabled(manual)
        self._autoRangeChk.setEnabled(connected and not busy)
        self._zeroBtn.setEnabled(connected and not busy)
        self._avgSpin.setEnabled(connected and not busy)
        self._setAvgBtn.setEnabled(connected and not busy)
        self._beamDiaSpin.setEnabled(connected and not busy)
        self._setBeamDiaBtn.setEnabled(connected and not busy)
        self._resetStatsBtn.setEnabled(connected)

    # ---- Connection ----

    def _on_scan(self):
        resources = PM100A.list_resources()
        if not resources:
            QtWidgets.QMessageBox.information(self, "Scan", "No VISA resources found.")
            return
        if len(resources) == 1:
            self._resourceEdit.setText(resources[0])
            return
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Select resource", "Available VISA resources:", resources, 0, False
        )
        if ok and choice:
            self._resourceEdit.setText(choice)

    def _on_connect(self):
        resource = self._resourceEdit.text().strip()
        if not resource:
            QtWidgets.QMessageBox.warning(self, "Resource", "Enter a VISA resource string.")
            return
        try:
            self._pm = PM100A(resource)
            self._statusLbl.setText(f"Connected: {resource}")
            try:
                wav = self._pm.get_wavelength()
                self._wavSpin.setValue(wav)
                auto = self._pm.get_auto_range()
                self._autoRangeChk.setChecked(auto)
                if not auto:
                    rng = self._pm.get_range()
                    val, unit = _watts_to_display(rng)
                    self._rangeSpin.setValue(val)
                    self._rangeUnitCombo.setCurrentText(unit)
                dia = self._pm.get_beam_diameter()
                self._beamDiaSpin.setValue(dia)
            except Exception:
                pass
            self._reset_stats()
            self._poll_timer.start()
        except PM100AError as e:
            QtWidgets.QMessageBox.critical(self, "Connect failed", str(e))
            self._pm = None
        self._sync_controls()

    def _on_disconnect(self):
        self._poll_timer.stop()
        try:
            if self._pm:
                self._pm.close()
        except Exception:
            pass
        self._pm = None
        self._powerLbl.setText("---")
        self._statusLbl.setText("Disconnected.")
        self._reset_stats()
        self._sync_controls()

    # ---- Poll ----

    def _poll(self):
        if self._pm is None or self._zeroing:
            return
        try:
            power_w = self._pm.measure_power()
            self._powerLbl.setText(_fmt_power(power_w))
            self._accumulate_stat(power_w)
        except Exception as e:
            self._statusLbl.setText(f"Poll error: {e}")

    # ---- Wavelength ----

    def _on_set_wavelength(self):
        if not self._pm:
            return
        try:
            self._pm.set_wavelength(self._wavSpin.value())
            self._statusLbl.setText(f"Wavelength set to {self._wavSpin.value():.1f} nm")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Set wavelength failed", str(e))

    # ---- Range ----

    def _on_set_range(self):
        if not self._pm:
            return
        try:
            watts = self._rangeSpin.value() * _UNIT_MULTIPLIERS[self._rangeUnitCombo.currentText()]
            self._pm.set_range(watts)
            self._statusLbl.setText(
                f"Range set to {self._rangeSpin.value():.1f} {self._rangeUnitCombo.currentText()}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Set range failed", str(e))

    def _on_auto_range_changed(self, state: int):
        enabled = state == QtCore.Qt.Checked
        manual = self._pm is not None and not self._zeroing and not enabled
        self._setRangeBtn.setEnabled(manual)
        self._rangeUnitCombo.setEnabled(manual)
        if not self._pm:
            return
        try:
            self._pm.set_auto_range(enabled)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Auto range failed", str(e))

    # ---- Averaging ----

    def _on_set_averaging(self):
        if not self._pm:
            return
        try:
            self._pm.set_averaging(self._avgSpin.value())
            self._statusLbl.setText(f"Averaging set to {self._avgSpin.value()}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Set averaging failed", str(e))

    # ---- Beam diameter ----

    def _on_set_beam_diameter(self):
        if not self._pm:
            return
        try:
            self._pm.set_beam_diameter(self._beamDiaSpin.value())
            self._statusLbl.setText(f"Beam diameter set to {self._beamDiaSpin.value():.2f} mm")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Set beam diameter failed", str(e))

    # ---- Zero ----

    def _on_zero(self):
        if not self._pm:
            return
        self._zeroing = True
        self._statusLbl.setText("Zeroing sensor...")
        self._poll_timer.stop()
        self._powerLbl.setText("---")
        self._sync_controls()

        self._zero_thread = ZeroThread(self._pm, parent=self)
        self._zero_thread.done.connect(self._on_zero_done)
        self._zero_thread.start()

    def _on_zero_done(self, result: str):
        self._zeroing = False
        if result == "ok":
            self._statusLbl.setText("Zero calibration complete.")
        else:
            self._statusLbl.setText(result)
            QtWidgets.QMessageBox.critical(self, "Zero failed", result)
        self._poll_timer.start()
        self._sync_controls()

    # ---- Statistics ----

    def _reset_stats(self):
        self._stats_values = []
        self._stats_start = None
        self._statCurrentLbl.setText("---")
        self._statStartLbl.setText("---")
        self._statSamplesLbl.setText("0")
        self._statMinLbl.setText("---")
        self._statMaxLbl.setText("---")
        self._statMeanLbl.setText("---")
        self._statStdLbl.setText("---")

    def _accumulate_stat(self, power_w: float):
        if self._stats_start is None:
            self._stats_start = datetime.now()
            self._statStartLbl.setText(self._stats_start.strftime("%H:%M:%S"))
        self._stats_values.append(power_w)
        n = len(self._stats_values)
        mn = min(self._stats_values)
        mx = max(self._stats_values)
        mean = sum(self._stats_values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in self._stats_values) / n
            std = math.sqrt(variance)
        else:
            std = 0.0
        self._statCurrentLbl.setText(_fmt_power(power_w))
        self._statSamplesLbl.setText(str(n))
        self._statMinLbl.setText(_fmt_power(mn))
        self._statMaxLbl.setText(_fmt_power(mx))
        self._statMeanLbl.setText(_fmt_power(mean))
        self._statStdLbl.setText(_fmt_power(std))


# ---- Helpers ----

_UNIT_MULTIPLIERS = {
    "nW": 1e-9,
    "\u03bcW": 1e-6,
    "mW": 1e-3,
    "W": 1.0,
}


def _watts_to_display(watts: float):
    """Return (value, unit_str) choosing the most readable unit."""
    if watts < 0.5e-6:
        return watts / 1e-9, "nW"
    if watts < 0.5e-3:
        return watts / 1e-6, "\u03bcW"
    if watts < 0.5:
        return watts / 1e-3, "mW"
    return watts, "W"


def _fmt_power(watts: float) -> str:
    """Format power with appropriate unit prefix."""
    if abs(watts) < 1e-9:
        return f"{watts * 1e12:.2f} pW"
    if abs(watts) < 1e-6:
        return f"{watts * 1e9:.2f} nW"
    if abs(watts) < 1e-3:
        return f"{watts * 1e6:.2f} \u03bcW"
    if abs(watts) < 1.0:
        return f"{watts * 1e3:.2f} mW"
    return f"{watts:.4f} W"

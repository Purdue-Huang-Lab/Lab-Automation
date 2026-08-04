import os
import re
import time
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

from andor.andor_wrapper import AndorSystem

from .config import (
    BTN_W,
    READ_W,
    STATUS_W,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_EM_GAIN,
    DEFAULT_HBIN,
    DEFAULT_VBIN,
    DEFAULT_TRIGGER,
    DEFAULT_ACQ_MODE,
    DEFAULT_FRAME_API,
    DEFAULT_READ_MODE,
    DEFAULT_OUTPUT_AMP,
    DEFAULT_READOUT_RATE,
    DEFAULT_PREAMP_GAIN,
    DEFAULT_BASELINE_CLAMP,
    DEFAULT_VSHIFT_US,
    DEFAULT_VCLOCK_AMP,
    DEFAULT_COOLER_ON,
    DEFAULT_SETPOINT_C,
    DEFAULT_SPEC_INDEX,
    DEFAULT_GRATING,
    DEFAULT_SLIT_ID,
    DEFAULT_SLIT_UM,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_LINECUT_WIDTH,
)
from .workers import LiveAcqThread, SingleAcqThread


class AndorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)
        self._live_thread: Optional[LiveAcqThread] = None
        self._accum_thread: Optional[SingleAcqThread] = None
        self._last_frame_raw = None
        self._last_frame = None
        self._last_frame8 = None
        self._cursor_rc = None
        self._wl_axis = None
        self._xaxis_mode = "pixel"
        self._ax_img_top = None

        self._build_ui()
        self._temp_timer = QtCore.QTimer(self)
        self._temp_timer.timeout.connect(self._update_temp_readout)

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        vbox.addLayout(top)

        self.camIndexSpin = QtWidgets.QSpinBox()
        self.camIndexSpin.setRange(0, 8)
        self.camIndexSpin.setValue(0)
        self.camIndexSpin.setFixedWidth(60)

        self.connectBtn = QtWidgets.QPushButton("Connect")
        self.connectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn.setEnabled(False)

        self.startBtn = QtWidgets.QPushButton("Start")
        self.startBtn.setFixedWidth(BTN_W)
        self.startBtn.setEnabled(False)
        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.stopBtn.setFixedWidth(BTN_W)
        self.stopBtn.setEnabled(False)

        self.saveBtn = QtWidgets.QPushButton("Save .asc")
        self.saveBtn.setFixedWidth(BTN_W)
        self.saveBtn.setEnabled(False)

        self.statusLbl = QtWidgets.QLabel("Disconnected")
        self.statusLbl.setMinimumWidth(STATUS_W)

        top.addWidget(QtWidgets.QLabel("Cam idx:"))
        top.addWidget(self.camIndexSpin)
        top.addWidget(self.connectBtn)
        top.addWidget(self.disconnectBtn)
        top.addSpacing(10)
        top.addWidget(self.startBtn)
        top.addWidget(self.stopBtn)
        top.addWidget(self.saveBtn)
        top.addStretch()
        top.addWidget(self.statusLbl)

        # Main content
        mid = QtWidgets.QHBoxLayout()
        vbox.addLayout(mid, 0)

        self._build_camera_group(mid)
        self._build_spectrograph_group(mid)

        # Live view
        self._build_live_view(vbox)

        # Status bar
        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)
        self.statusBar().showMessage("Idle")

        # Signals
        self.connectBtn.clicked.connect(self.on_connect)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.saveBtn.clicked.connect(self.on_save)

        self.readBtn.clicked.connect(self.on_read_settings)
        self.applyBtn.clicked.connect(self.on_apply_settings)
        self.coolerBtn.clicked.connect(self.on_toggle_cooler)
        self.diagBtn.clicked.connect(self.on_diag_readout)

        self.specConnectBtn.clicked.connect(self.on_spec_connect)
        self.specDisconnectBtn.clicked.connect(self.on_spec_disconnect)
        self.specRefreshAxisBtn.clicked.connect(self.on_refresh_axis)
        self.gratingSetBtn.clicked.connect(self.on_set_grating)
        self.gratingReadBtn.clicked.connect(self.on_read_grating)
        self.slitSetBtn.clicked.connect(self.on_set_slit)
        self.centerSetBtn.clicked.connect(self.on_set_center_wl)
        self.centerReadBtn.clicked.connect(self.on_read_center_wl)

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    def _build_camera_group(self, parent_layout):
        box = QtWidgets.QGroupBox("Andor Camera")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)

        row = 0
        self.nameLbl = QtWidgets.QLabel("Name: --")
        self.modelLbl = QtWidgets.QLabel("Model: --")
        self.serialLbl = QtWidgets.QLabel("Serial: --")
        self.fwLbl = QtWidgets.QLabel("Firmware: --")
        self.roiLbl = QtWidgets.QLabel("ROI: --")

        grid.addWidget(self.nameLbl, row, 0, 1, 2); row += 1
        grid.addWidget(self.modelLbl, row, 0, 1, 2); row += 1
        grid.addWidget(self.serialLbl, row, 0, 1, 2); row += 1
        grid.addWidget(self.fwLbl, row, 0, 1, 2); row += 1
        grid.addWidget(self.roiLbl, row, 0, 1, 2); row += 1

        self.exposureSpin = QtWidgets.QDoubleSpinBox()
        self.exposureSpin.setDecimals(3)
        self.exposureSpin.setRange(0.001, 1.0e7)
        self.exposureSpin.setValue(DEFAULT_EXPOSURE_MS)

        self.accumSpin = QtWidgets.QSpinBox()
        self.accumSpin.setRange(1, 100000)
        self.accumSpin.setValue(1)

        grid.addWidget(QtWidgets.QLabel("Exposure (ms):"), row, 0)
        grid.addWidget(self.exposureSpin, row, 1)
        grid.addWidget(QtWidgets.QLabel("Accum N:"), row, 2)
        grid.addWidget(self.accumSpin, row, 3)
        row += 1

        self.emGainSpin = QtWidgets.QSpinBox()
        self.emGainSpin.setRange(0, 3000)
        self.emGainSpin.setValue(DEFAULT_EM_GAIN)
        grid.addWidget(QtWidgets.QLabel("EM gain:"), row, 0)
        grid.addWidget(self.emGainSpin, row, 1)

        self.hbinSpin = QtWidgets.QSpinBox()
        self.hbinSpin.setRange(1, 1024)
        self.hbinSpin.setValue(DEFAULT_HBIN)
        self.vbinSpin = QtWidgets.QSpinBox()
        self.vbinSpin.setRange(1, 1024)
        self.vbinSpin.setValue(DEFAULT_VBIN)

        grid.addWidget(QtWidgets.QLabel("H bin:"), row, 2)
        grid.addWidget(self.hbinSpin, row, 3)
        row += 1
        grid.addWidget(QtWidgets.QLabel("V bin:"), row, 0)
        grid.addWidget(self.vbinSpin, row, 1)
        row += 1

        self.triggerCombo = QtWidgets.QComboBox()
        self.triggerCombo.addItems(["Internal", "External", "External Start", "External Exposure", "Software"])
        self.triggerCombo.setCurrentText(DEFAULT_TRIGGER)

        self.acqCombo = QtWidgets.QComboBox()
        self.acqCombo.addItems(["Single", "Kinetic", "Run till abort"])
        self.acqCombo.setCurrentText(DEFAULT_ACQ_MODE)

        grid.addWidget(QtWidgets.QLabel("Trigger:"), row, 0)
        grid.addWidget(self.triggerCombo, row, 1)
        grid.addWidget(QtWidgets.QLabel("Acq mode:"), row, 2)
        grid.addWidget(self.acqCombo, row, 3)
        row += 1

        self.frameApiCombo = QtWidgets.QComboBox()
        self.frameApiCombo.addItems(["Snap", "Stream+Buffer"])
        self.frameApiCombo.setCurrentText(DEFAULT_FRAME_API)

        self.readModeCombo = QtWidgets.QComboBox()
        self.readModeCombo.addItems(["Image", "FVB"])
        self.readModeCombo.setCurrentText(DEFAULT_READ_MODE)

        grid.addWidget(QtWidgets.QLabel("Frame API:"), row, 0)
        grid.addWidget(self.frameApiCombo, row, 1)
        grid.addWidget(QtWidgets.QLabel("Read mode:"), row, 2)
        grid.addWidget(self.readModeCombo, row, 3)
        row += 1

        self.outputAmpCombo = QtWidgets.QComboBox()
        self.outputAmpCombo.setEditable(False)
        self.outputAmpCombo.addItem(DEFAULT_OUTPUT_AMP)

        self.readoutRateCombo = QtWidgets.QComboBox()
        self.readoutRateCombo.setEditable(False)
        self.readoutRateCombo.addItem(DEFAULT_READOUT_RATE)

        self.preampCombo = QtWidgets.QComboBox()
        self.preampCombo.setEditable(False)
        self.preampCombo.addItem(DEFAULT_PREAMP_GAIN)

        grid.addWidget(QtWidgets.QLabel("Output amp:"), row, 0)
        grid.addWidget(self.outputAmpCombo, row, 1)
        grid.addWidget(QtWidgets.QLabel("Readout rate:"), row, 2)
        grid.addWidget(self.readoutRateCombo, row, 3)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Preamp gain:"), row, 0)
        grid.addWidget(self.preampCombo, row, 1)
        row += 1

        self.baselineChk = QtWidgets.QCheckBox("Baseline clamp")
        self.baselineChk.setChecked(DEFAULT_BASELINE_CLAMP)
        grid.addWidget(self.baselineChk, row, 0, 1, 2)

        self.vshiftCombo = QtWidgets.QComboBox()
        self.vshiftCombo.addItems([str(x) for x in (4.88, 9.68, 19.28, 38.47, 57.68)])
        self.vshiftCombo.setCurrentText(str(DEFAULT_VSHIFT_US))
        self.vclockCombo = QtWidgets.QComboBox()
        self.vclockCombo.addItems(["Normal", "+1", "+2", "+3", "+4"])
        self.vclockCombo.setCurrentText(DEFAULT_VCLOCK_AMP)

        grid.addWidget(QtWidgets.QLabel("V shift (us):"), row, 2)
        grid.addWidget(self.vshiftCombo, row, 3)
        row += 1
        grid.addWidget(QtWidgets.QLabel("V clock amp:"), row, 2)
        grid.addWidget(self.vclockCombo, row, 3)
        row += 1

        self.coolerChk = QtWidgets.QCheckBox("Cooler")
        self.coolerChk.setChecked(DEFAULT_COOLER_ON)
        self.setpointSpin = QtWidgets.QDoubleSpinBox()
        self.setpointSpin.setRange(-120.0, 40.0)
        self.setpointSpin.setValue(DEFAULT_SETPOINT_C)
        self.tempLbl = QtWidgets.QLabel("--")
        self.tempStatusLbl = QtWidgets.QLabel("")

        grid.addWidget(self.coolerChk, row, 0)
        grid.addWidget(QtWidgets.QLabel("Setpoint (C):"), row, 2)
        grid.addWidget(self.setpointSpin, row, 3)
        row += 1
        grid.addWidget(QtWidgets.QLabel("CCD Temp (C):"), row, 0)
        grid.addWidget(self.tempLbl, row, 1)
        grid.addWidget(self.tempStatusLbl, row, 2, 1, 2)
        row += 1

        self.readBtn = QtWidgets.QPushButton("Read from camera")
        self.readBtn.setFixedWidth(BTN_W)
        self.applyBtn = QtWidgets.QPushButton("Apply settings")
        self.applyBtn.setFixedWidth(BTN_W)
        self.coolerBtn = QtWidgets.QPushButton("Cooler ON/OFF")
        self.coolerBtn.setFixedWidth(BTN_W)
        self.diagBtn = QtWidgets.QPushButton("Diag readout")
        self.diagBtn.setFixedWidth(BTN_W)

        grid.addWidget(self.readBtn, row, 0)
        grid.addWidget(self.applyBtn, row, 1)
        grid.addWidget(self.coolerBtn, row, 2)
        grid.addWidget(self.diagBtn, row, 3)

        parent_layout.addWidget(box, 1)

    def _build_spectrograph_group(self, parent_layout):
        box = QtWidgets.QGroupBox("Spectrograph (Shamrock)")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)

        row = 0
        self.specIdxSpin = QtWidgets.QSpinBox()
        self.specIdxSpin.setRange(0, 8)
        self.specIdxSpin.setValue(DEFAULT_SPEC_INDEX)

        self.specConnectBtn = QtWidgets.QPushButton("Connect")
        self.specConnectBtn.setFixedWidth(BTN_W)
        self.specDisconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.specDisconnectBtn.setFixedWidth(BTN_W)
        self.specDisconnectBtn.setEnabled(False)
        self.specRefreshAxisBtn = QtWidgets.QPushButton("Refresh axis")
        self.specRefreshAxisBtn.setFixedWidth(BTN_W)

        grid.addWidget(QtWidgets.QLabel("Idx:"), row, 0)
        grid.addWidget(self.specIdxSpin, row, 1)
        grid.addWidget(self.specConnectBtn, row, 2)
        grid.addWidget(self.specRefreshAxisBtn, row, 3)
        row += 1
        grid.addWidget(self.specDisconnectBtn, row, 2)

        row += 1
        self.specStatusLbl = QtWidgets.QLabel("Spectrograph: not connected")
        grid.addWidget(self.specStatusLbl, row, 0, 1, 4)

        row += 1
        self.gratingCombo = QtWidgets.QComboBox()
        self.gratingCombo.setEditable(False)
        self.gratingSetBtn = QtWidgets.QPushButton("Set")
        self.gratingSetBtn.setFixedWidth(BTN_W)
        self.gratingReadBtn = QtWidgets.QPushButton("Read")
        self.gratingReadBtn.setFixedWidth(BTN_W)
        grid.addWidget(QtWidgets.QLabel("Grating:"), row, 0)
        grid.addWidget(self.gratingCombo, row, 1)
        grid.addWidget(self.gratingSetBtn, row, 2)
        grid.addWidget(self.gratingReadBtn, row, 3)

        row += 1
        self.gratingInfoLbl = QtWidgets.QLabel("Grating info: --")
        grid.addWidget(self.gratingInfoLbl, row, 0, 1, 4)

        row += 1
        self.slitCombo = QtWidgets.QComboBox()
        self.slitCombo.addItems(["input_side", "input_direct", "output_side", "output_direct"])
        self.slitCombo.setCurrentText(DEFAULT_SLIT_ID)
        self.slitSpin = QtWidgets.QDoubleSpinBox()
        self.slitSpin.setRange(0.0, 5000.0)
        self.slitSpin.setValue(DEFAULT_SLIT_UM)
        self.slitSetBtn = QtWidgets.QPushButton("Set slit (um)")
        self.slitSetBtn.setFixedWidth(BTN_W)
        grid.addWidget(QtWidgets.QLabel("Slit:"), row, 0)
        grid.addWidget(self.slitCombo, row, 1)
        grid.addWidget(self.slitSpin, row, 2)
        grid.addWidget(self.slitSetBtn, row, 3)

        row += 1
        self.centerSpin = QtWidgets.QDoubleSpinBox()
        self.centerSpin.setDecimals(3)
        self.centerSpin.setRange(0.0, 20000.0)
        self.centerSpin.setValue(DEFAULT_CENTER_WL_NM)
        self.centerSetBtn = QtWidgets.QPushButton("Set")
        self.centerSetBtn.setFixedWidth(BTN_W)
        self.centerReadBtn = QtWidgets.QPushButton("Read")
        self.centerReadBtn.setFixedWidth(BTN_W)
        grid.addWidget(QtWidgets.QLabel("Center (nm):"), row, 0)
        grid.addWidget(self.centerSpin, row, 1)
        grid.addWidget(self.centerSetBtn, row, 2)
        grid.addWidget(self.centerReadBtn, row, 3)

        parent_layout.addWidget(box, 1)

    def _build_live_view(self, parent_layout):
        panel = QtWidgets.QGroupBox("Live View")
        layout = QtWidgets.QGridLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.fig = Figure(figsize=(7, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[4, 1.2], wspace=0.25, hspace=0.25)
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self._ax_img_top = self.ax_img.twiny()
        self._ax_img_top.set_visible(False)
        self._ax_img_top.set_xlabel("Wavelength (nm)")
        self._ax_img_top.tick_params(axis="x", labelsize=8)
        self.ax_v = self.fig.add_subplot(gs[0, 1])
        self.ax_h = self.fig.add_subplot(gs[1, 0])
        self.ax_h.set_xlabel("X (px)")
        self.ax_v.set_ylabel("Y (px)")

        self.im = self.ax_img.imshow(np.zeros((10, 10)), cmap="gray", origin="upper", interpolation="nearest", aspect="auto")
        self._crosshair_h = self.ax_img.axhline(0, color="white", lw=0.6, alpha=0.9)
        self._crosshair_v = self.ax_img.axvline(0, color="white", lw=0.6, alpha=0.9)
        self._crosshair_h.set_visible(False)
        self._crosshair_v.set_visible(False)
        self.line_v, = self.ax_v.plot([], [])
        self.line_h, = self.ax_h.plot([], [])
        self.ax_img.set_title("Live")

        layout.addWidget(self.canvas, 0, 0, 2, 1)

        side = QtWidgets.QVBoxLayout()
        self.cursorLbl = QtWidgets.QLabel("Cursor: (row, col) = --, --")
        self.linecutWidthSpin = QtWidgets.QSpinBox()
        self.linecutWidthSpin.setRange(1, 999)
        self.linecutWidthSpin.setValue(DEFAULT_LINECUT_WIDTH)
        side.addWidget(self.cursorLbl)
        side.addWidget(QtWidgets.QLabel("Linecut width (px):"))
        side.addWidget(self.linecutWidthSpin)
        side.addStretch()
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        layout.addWidget(side_widget, 0, 1)

        parent_layout.addWidget(panel, 1)

        self.canvas.mpl_connect("button_press_event", self.on_image_click)

    # -----------------
    # Camera actions
    # -----------------
    def on_connect(self):
        idx = int(self.camIndexSpin.value())
        self.cam.cam_index = idx
        try:
            self.cam.connect()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Connect failed", str(exc))
            return

        self.connectBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(True)
        self.startBtn.setEnabled(True)
        self.saveBtn.setEnabled(True)
        self.statusLbl.setText("Connected")

        self._refresh_ampmode_choices()
        self.on_read_settings()
        self._apply_connect_defaults()
        self._update_temp_readout()
        self._temp_timer.start(1000)

    def on_disconnect(self):
        self.on_stop()
        try:
            self.cam.disconnect()
        except Exception:
            pass
        try:
            self._temp_timer.stop()
        except Exception:
            pass
        self.connectBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
        self.startBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.statusLbl.setText("Disconnected")

    def on_start(self):
        if self.cam.cam is None:
            return
        if int(self.accumSpin.value()) > 1:
            self._run_accum_once()
            return
        if self._live_thread is not None and self._live_thread.isRunning():
            return
        self._apply_settings_to_camera()
        self._live_thread = LiveAcqThread(self.cam)
        self._live_thread.frame_ready.connect(self.on_new_frame)
        self._live_thread.status.connect(self.statusBar().showMessage)
        self._live_thread.start()
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.statusBar().showMessage("Live started")

    def on_stop(self):
        if self._accum_thread is not None and self._accum_thread.isRunning():
            self._accum_thread.stop()
            self.stopBtn.setEnabled(False)
            self.statusBar().showMessage("Stopping accumulation...")
            return
        if self._live_thread is not None:
            self._live_thread.stop()
            try:
                self.cam.stop_stream()
            except Exception:
                pass
            self.stopBtn.setEnabled(False)
            self.statusBar().showMessage("Stopping...")
            exp_ms = None
            try:
                exp_ms = float(self.exposureSpin.value())
            except Exception:
                exp_ms = None
            if exp_ms is None:
                try:
                    exp_ms = float(self.cam.get_exposure_ms())
                except Exception:
                    exp_ms = None
            timeout_ms = 5000
            if exp_ms is not None and exp_ms > 0:
                timeout_ms = int(max(timeout_ms, exp_ms * 1.5 + 2000))
            self._live_stop_pending = True
            self._live_stop_t0 = time.monotonic()
            self._live_stop_timeout_ms = timeout_ms
            QtCore.QTimer.singleShot(200, self._check_live_stop)
            return
        try:
            self.cam.stop_stream()
        except Exception:
            pass
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def _run_accum_once(self):
        if self._accum_thread is not None and self._accum_thread.isRunning():
            return
        self._apply_settings_to_camera()
        n_accum = int(self.accumSpin.value())
        self.statusBar().showMessage(f"Accumulating {n_accum} frames...")
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        self._accum_thread = SingleAcqThread(self.cam, n_accum)
        self._accum_thread.frame_ready.connect(self._on_accum_frame)
        self._accum_thread.status.connect(self.statusBar().showMessage)
        self._accum_thread.finished.connect(self._on_accum_finished)
        self._accum_thread.start()
        self.stopBtn.setEnabled(True)

    def _on_accum_frame(self, fr: dict):
        if not fr.get("ok"):
            self.statusBar().showMessage(f"Accum failed: {fr.get('err')}")
            self.startBtn.setEnabled(True)
            self.stopBtn.setEnabled(False)
            return

        self.on_new_frame(fr)
        idx = fr.get("accum_idx")
        total = fr.get("accum_n")
        if (idx is not None) and (total is not None) and (int(idx) < int(total)):
            return
        self.statusBar().showMessage("Accumulation done")
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self._accum_thread = None

    def _on_accum_finished(self):
        if self._accum_thread is not None and self._accum_thread.isRunning():
            return
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self._accum_thread = None

    def _check_live_stop(self):
        if not getattr(self, "_live_stop_pending", False):
            return
        if self._live_thread is None or (not self._live_thread.isRunning()):
            self._live_stop_pending = False
            self._live_thread = None
            self.startBtn.setEnabled(True)
            self.stopBtn.setEnabled(False)
            self.statusBar().showMessage("Stopped")
            return
        elapsed_ms = (time.monotonic() - float(getattr(self, "_live_stop_t0", 0.0))) * 1000.0
        timeout_ms = float(getattr(self, "_live_stop_timeout_ms", 5000))
        if elapsed_ms >= timeout_ms:
            self.statusBar().showMessage("Stopping... camera busy")
            self._live_stop_t0 = time.monotonic()
        QtCore.QTimer.singleShot(200, self._check_live_stop)

    def on_save(self):
        if self._last_frame is None:
            QtWidgets.QMessageBox.information(self, "Save", "No frame to save yet.")
            return

        exp_ms = float(self.exposureSpin.value())
        slit_um = float(self.slitSpin.value())
        gr = self._current_grating_number()

        base = f"andor_exp{exp_ms:g}ms_slit{slit_um:g}um_gr{gr}"
        default_name = base + ".asc"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ASCII", default_name, "ASC Files (*.asc)")
        if not path:
            return

        meta = {
            "Exposure_ms": f"{exp_ms:g}",
            "Accumulations": str(int(self.accumSpin.value())),
            "Slit_um": f"{slit_um:g}",
            "Grating": str(gr),
            "Center_nm": f"{float(self.centerSpin.value()):g}",
        }
        try:
            self.cam.save_ascii(path, self._last_frame, metadata=meta)
            self.statusBar().showMessage(f"Saved: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    # -----------------
    # Spectrograph actions
    # -----------------
    def on_spec_connect(self):
        idx = int(self.specIdxSpin.value())
        ok = self.cam.connect_spectrograph(idx)
        if ok:
            self.specStatusLbl.setText("Spectrograph: connected")
            self.specConnectBtn.setEnabled(False)
            self.specDisconnectBtn.setEnabled(True)
            self.on_read_grating()
            self._update_wavelength_axis_cache(force=True)
        else:
            self.specStatusLbl.setText("Spectrograph: connect failed")

    def on_spec_disconnect(self):
        try:
            self.cam.disconnect_spectrograph()
        except Exception:
            pass
        self.specStatusLbl.setText("Spectrograph: not connected")
        self.specConnectBtn.setEnabled(True)
        self.specDisconnectBtn.setEnabled(False)

    def on_refresh_axis(self):
        try:
            self.cam.get_wavelength_axis(force=True)
            self._update_wavelength_axis_cache(force=True)
            self.statusBar().showMessage("Axis refreshed")
        except Exception as exc:
            self.statusBar().showMessage(f"Axis refresh failed: {exc}")

    def on_set_grating(self):
        val = self._current_grating_number()
        ok = self.cam.spec_set_grating(val)
        if ok:
            self.statusBar().showMessage(f"Grating set to {val}")
            self.on_read_grating()
            self._update_wavelength_axis_cache(force=True)
        else:
            self.statusBar().showMessage("Set grating failed")

    def on_read_grating(self):
        n = self.cam.spec_get_gratings_number()
        if n is None:
            self.gratingInfoLbl.setText("Grating info: --")
            return
        choices = []
        for g in range(1, n + 1):
            info = self.cam.spec_get_grating_info(g)
            if info is None:
                choices.append(str(g))
            else:
                choices.append(f"{g}: {info}")
        self.gratingCombo.blockSignals(True)
        self.gratingCombo.clear()
        self.gratingCombo.addItems(choices)
        self.gratingCombo.blockSignals(False)

        cur = self.cam.spec_get_grating()
        if cur is not None:
            for idx, c in enumerate(choices):
                if c.startswith(f"{cur}:") or c == str(cur):
                    self.gratingCombo.setCurrentIndex(idx)
                    break
            info = self.cam.spec_get_grating_info(int(cur))
            self.gratingInfoLbl.setText(f"Grating info: {info}")
        else:
            self.gratingInfoLbl.setText("Grating info: --")

    def on_set_slit(self):
        slit = self.slitCombo.currentText().strip()
        w_um = float(self.slitSpin.value())
        ok = self.cam.spec_set_slit_width_um(slit, w_um)
        if ok:
            self.statusBar().showMessage(f"Slit {slit} set to {w_um:g} um")
        else:
            self.statusBar().showMessage("Set slit failed")

    def on_set_center_wl(self):
        wl = float(self.centerSpin.value())
        ok = self.cam.set_center_wavelength_nm(wl)
        if ok:
            self.statusBar().showMessage(f"Center wavelength set to {wl:g} nm")
            self._update_wavelength_axis_cache(force=True)
        else:
            self.statusBar().showMessage("Set center wavelength failed")

    def on_read_center_wl(self):
        wl = self.cam.get_center_wavelength_nm()
        if wl is not None:
            self.centerSpin.setValue(float(wl))
            self.statusBar().showMessage(f"Center wavelength {wl:g} nm")
        else:
            self.statusBar().showMessage("Center wavelength not available")

    # -----------------
    # Settings IO
    # -----------------
    def on_read_settings(self):
        info = self.cam.get_camera_info()
        if info is not None:
            self.nameLbl.setText(f"Name: {info.camera_name}")
            self.modelLbl.setText(f"Model: {info.model_name}")
            self.serialLbl.setText(f"Serial: {info.serial_number}")
            self.fwLbl.setText(f"Firmware: {info.firmware_version}")

        exp = self.cam.get_exposure_ms()
        if exp is not None:
            self.exposureSpin.setValue(float(exp))

        em = self.cam.get_em_gain()
        if em is not None:
            self.emGainSpin.setValue(int(em))

        bins = self.cam.get_binning()
        if bins:
            self.hbinSpin.setValue(int(bins[0]))
            self.vbinSpin.setValue(int(bins[1]))

        trig = self.cam.get_trigger_mode()
        if trig:
            self._set_combo_by_text(self.triggerCombo, trig)

        acq = self.cam.get_acquisition_mode()
        if acq:
            if "accum" in str(acq).lower():
                acq = "Single"
            self._set_combo_by_text(self.acqCombo, acq)

        read_mode = self.cam.get_read_mode()
        if read_mode:
            self._set_combo_by_text(self.readModeCombo, read_mode)

        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None
        if info is not None:
            self._sync_amp_mode_from_info(info)

        self._update_temp_readout()

    def on_apply_settings(self):
        self._apply_settings_to_camera()

    def _apply_settings_to_camera(self):
        try:
            self.cam.stop_stream()
        except Exception:
            pass
        # Clear any pending acquisition so the first frame after applying
        # settings doesn't timeout (same issue as first frame after connect).
        try:
            self.cam._recover_after_timeout()
        except Exception:
            pass
        try:
            self._wait_idle(timeout_s=3.0)
        except Exception:
            pass
        cur_info = None
        try:
            cur_info = self.cam.get_amp_mode_info()
        except Exception:
            cur_info = None
        frame_api = self.frameApiCombo.currentText().strip()
        acq_mode = self.acqCombo.currentText().strip()
        if frame_api.lower().startswith("snap"):
            acq_mode = "Single"
            self.acqCombo.setCurrentText(acq_mode)
        else:
            acq_mode = "Run till abort"
            self.acqCombo.setCurrentText(acq_mode)
        self.cam.set_exposure_ms(float(self.exposureSpin.value()))
        self.cam.set_em_gain(int(self.emGainSpin.value()))
        self.cam.set_binning(int(self.hbinSpin.value()), int(self.vbinSpin.value()))
        self.cam.set_trigger_mode(self.triggerCombo.currentText())
        self.cam.set_frame_api(frame_api)
        self.cam.set_acquisition_mode(acq_mode)
        self.cam.set_shutter("auto")
        self.cam.set_read_mode(self.readModeCombo.currentText())
        req_out_amp = self.outputAmpCombo.currentText()
        req_rate = self.readoutRateCombo.currentText()
        req_gain = self.preampCombo.currentText()
        want_mhz = None
        want_gain = None
        try:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(khz|mhz|hz)?", str(req_rate).lower())
            if m:
                v = float(m.group(1))
                unit = m.group(2) or ""
                if unit == "khz":
                    want_mhz = v / 1000.0
                elif unit == "hz":
                    want_mhz = v / 1e6
                else:
                    want_mhz = v
        except Exception:
            want_mhz = None
        try:
            mg = re.search(r"(\d+(?:\.\d+)?)", str(req_gain).lower())
            if mg:
                want_gain = float(mg.group(1))
        except Exception:
            want_gain = None

        need_amp = True
        if cur_info is not None:
            try:
                cur_amp = str(cur_info.get("amp_label", "")).strip().lower()
                req_amp = str(req_out_amp).strip().lower()
                cur_mhz = cur_info.get("hsspeed_mhz")
                cur_gain = cur_info.get("preamp_gain")
                amp_ok = (not req_amp) or (req_amp in cur_amp) or (cur_amp in req_amp)
                rate_ok = (want_mhz is None) or (cur_mhz is None) or (abs(float(cur_mhz) - float(want_mhz)) <= 0.05)
                gain_ok = (want_gain is None) or (cur_gain is None) or (abs(float(cur_gain) - float(want_gain)) <= 0.05)
                if amp_ok and rate_ok and gain_ok:
                    need_amp = False
            except Exception:
                need_amp = True

        if need_amp:
            amp_ok = self.cam.set_amp_mode_by_labels(req_out_amp, req_rate, req_gain, force_preamp=True)
            info = None
            try:
                info = self.cam.get_amp_mode_info()
            except Exception:
                info = None
            if not amp_ok:
                self.statusBar().showMessage("Amp mode set failed; check readout rate")
            else:
                try:
                    got_mhz = info.get("hsspeed_mhz") if info else None
                    if (want_mhz is not None) and (got_mhz is not None) and (abs(got_mhz - want_mhz) > 0.05):
                        self.statusBar().showMessage("Readout rate mismatch; check amp mode table")
                except Exception:
                    pass
            if info is not None:
                self._sync_amp_mode_from_info(info)
        else:
            if cur_info is not None:
                self._sync_amp_mode_from_info(cur_info)

        if want_gain is not None:
            info_gain = None
            try:
                info_gain = self.cam.get_amp_mode_info()
            except Exception:
                info_gain = None
            cur_gain = None
            if info_gain is not None:
                cur_gain = info_gain.get("preamp_gain")
            if cur_gain is None or abs(float(cur_gain) - float(want_gain)) > 0.05:
                gain_ok = self.cam.set_amp_mode_by_labels(req_out_amp, req_rate, req_gain, force_preamp=True)
                try:
                    info_gain = self.cam.get_amp_mode_info()
                except Exception:
                    info_gain = None
                if info_gain is not None:
                    self._sync_amp_mode_from_info(info_gain)
                if not gain_ok:
                    self.statusBar().showMessage("Preamp gain set failed at selected readout rate")
        self.cam.set_baseline_clamp(bool(self.baselineChk.isChecked()))
        self.cam.set_vshift_us(float(self.vshiftCombo.currentText()))
        self.cam.set_vclock_amp(self.vclockCombo.currentText())
        self.cam.set_temperature_setpoint(float(self.setpointSpin.value()))
        self.cam.set_cooler(bool(self.coolerChk.isChecked()))
        if str(acq_mode).strip().lower().startswith("accum"):
            self.cam.set_accumulations(int(self.accumSpin.value()), switch_mode=True)
        self._update_temp_readout()
        self.statusBar().showMessage("Settings applied")

    def on_toggle_cooler(self):
        on = bool(self.coolerChk.isChecked())
        ok = self.cam.set_cooler(not on)
        if ok:
            self.coolerChk.setChecked(not on)
            self.statusBar().showMessage(f"Cooler {'ON' if not on else 'OFF'}")
        else:
            self.statusBar().showMessage("Cooler toggle failed")

    def on_diag_readout(self):
        if self.cam.cam is None:
            QtWidgets.QMessageBox.information(self, "Diag readout", "Camera not connected.")
            return
        lines = []

        def add(label, val):
            lines.append(f"{label}: {val}")

        add("UI frame_api", self.frameApiCombo.currentText())
        add("UI acq_mode", self.acqCombo.currentText())
        add("UI trigger", self.triggerCombo.currentText())
        add("UI read_mode", self.readModeCombo.currentText())
        add("UI out_amp", self.outputAmpCombo.currentText())
        add("UI readout_rate", self.readoutRateCombo.currentText())
        add("UI preamp_gain", self.preampCombo.currentText())
        add("UI exposure_ms", f"{float(self.exposureSpin.value()):g}")
        add("UI binning", f"{int(self.hbinSpin.value())}x{int(self.vbinSpin.value())}")

        try:
            add("SDK frame_api", self.cam.get_frame_api())
        except Exception as exc:
            add("SDK frame_api", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK acq_mode", self.cam.get_acquisition_mode())
        except Exception as exc:
            add("SDK acq_mode", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK trigger", self.cam.get_trigger_mode())
        except Exception as exc:
            add("SDK trigger", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK read_mode", self.cam.get_read_mode())
        except Exception as exc:
            add("SDK read_mode", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK exposure_ms", self.cam.get_exposure_ms())
        except Exception as exc:
            add("SDK exposure_ms", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK binning", self.cam.get_binning())
        except Exception as exc:
            add("SDK binning", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK amp_mode", self.cam.get_amp_mode(full=True))
        except Exception as exc:
            add("SDK amp_mode", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK amp_mode_info", self.cam.get_amp_mode_info())
        except Exception as exc:
            add("SDK amp_mode_info", f"ERR {type(exc).__name__}: {exc}")
        try:
            amps, rates, gains = self.cam.get_amp_mode_choices()
            add("SDK avail_rates", rates)
            add("SDK avail_gains", gains)
        except Exception as exc:
            add("SDK avail_rates", f"ERR {type(exc).__name__}: {exc}")
        try:
            modes = self.cam.get_all_amp_modes(full=True)
            combos = []
            if modes:
                for m in modes:
                    ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain = self.cam._parse_amp_mode(m)
                    try:
                        mhz = float(hs_mhz) if hs_mhz is not None else None
                    except Exception:
                        mhz = None
                    try:
                        gain = float(pre_gain) if pre_gain is not None else None
                    except Exception:
                        gain = None
                    combos.append((str(oamp_kind), mhz, gain, ch, oa, hs, pa))
            combos = sorted({(c[0], c[1], c[2]) for c in combos})
            add("SDK amp_mode_table", combos)
        except Exception as exc:
            add("SDK amp_mode_table", f"ERR {type(exc).__name__}: {exc}")
        try:
            add("SDK shutter", self.cam.get_shutter_parameters())
        except Exception as exc:
            add("SDK shutter", f"ERR {type(exc).__name__}: {exc}")

        inner = getattr(self.cam, "cam", None)
        if inner is not None:
            try:
                if hasattr(inner, "get_status"):
                    add("SDK status", inner.get_status())
            except Exception as exc:
                add("SDK status", f"ERR {type(exc).__name__}: {exc}")
            try:
                if hasattr(inner, "is_acquiring"):
                    add("SDK is_acquiring", inner.is_acquiring())
            except Exception as exc:
                add("SDK is_acquiring", f"ERR {type(exc).__name__}: {exc}")
            try:
                if hasattr(inner, "get_frame_timings"):
                    add("SDK frame_timings", inner.get_frame_timings())
            except Exception as exc:
                add("SDK frame_timings", f"ERR {type(exc).__name__}: {exc}")
            try:
                if hasattr(inner, "get_cycle_timings"):
                    add("SDK cycle_timings", inner.get_cycle_timings())
            except Exception as exc:
                add("SDK cycle_timings", f"ERR {type(exc).__name__}: {exc}")
            try:
                if hasattr(inner, "get_readout_time"):
                    add("SDK readout_time", inner.get_readout_time())
            except Exception as exc:
                add("SDK readout_time", f"ERR {type(exc).__name__}: {exc}")

        text = "\n".join(lines) if lines else "No data."
        QtWidgets.QMessageBox.information(self, "Diag readout", text)
        self.statusBar().showMessage("Diag readout captured")

    def _update_temp_readout(self):
        t = self.cam.get_temperature_c()
        st = self.cam.get_temperature_status()
        if t is not None:
            self.tempLbl.setText(f"{t:.1f}")
        if st is not None:
            self.tempStatusLbl.setText(str(st))

    # -----------------
    # Live display
    # -----------------
    def on_new_frame(self, fr: dict):
        img = fr.get("image")
        img8 = fr.get("image8")
        if img is None:
            return
        h, w = img.shape
        self._maybe_update_xaxis_label(w)
        self._update_image_wavelength_axis(w)
        raw = img
        disp = np.fliplr(np.flipud(raw))
        disp8 = np.fliplr(np.flipud(img8)) if img8 is not None else None
        self._last_frame_raw = raw
        self._last_frame = disp
        self._last_frame8 = disp8
        if disp8 is not None:
            self.im.set_data(disp8)
            self.im.set_clim(0, 255)
        else:
            self.im.set_data(disp)
            try:
                vmin = float(np.nanmin(disp))
                vmax = float(np.nanmax(disp))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.im.set_clim(vmin, vmax)
            except Exception:
                pass
        self.im.set_extent((0, w, h, 0))
        self.ax_img.set_xlim(0, w)
        self.ax_img.set_ylim(h, 0)
        self.ax_h.set_xlim(0, w)
        self.ax_v.set_ylim(h, 0)
        self._update_cursor_overlays(h, w)
        self.canvas.draw_idle()

        self.roiLbl.setText(f"ROI: 0,0 {w}x{h}")
        self.statusLbl.setText(f"FPS: {fr.get('fps_smooth', 0.0):.2f}")

    def on_image_click(self, event):
        if event.inaxes not in (self.ax_img, self._ax_img_top):
            return
        if self._last_frame is None or self._last_frame_raw is None:
            return
        try:
            if event.inaxes == self.ax_img:
                xdata, ydata = event.xdata, event.ydata
            else:
                xdata, ydata = self.ax_img.transData.inverted().transform((event.x, event.y))
            x = int(round(xdata))
            y = int(round(ydata))
        except Exception:
            return
        h, w = self._last_frame.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        y_raw = h - 1 - y
        if not (0 <= y_raw < h):
            return
        self._cursor_rc = (y, x)
        self._update_cursor_overlays(h, w)
        self.canvas.draw_idle()

    def _update_cursor_overlays(self, h: int, w: int) -> None:
        if self._cursor_rc is None or self._last_frame_raw is None:
            self._crosshair_h.set_visible(False)
            self._crosshair_v.set_visible(False)
            return
        y, x = self._cursor_rc
        if not ((0 <= x < w) and (0 <= y < h)):
            self._crosshair_h.set_visible(False)
            self._crosshair_v.set_visible(False)
            return

        y_raw = h - 1 - y
        if not (0 <= y_raw < h):
            self._crosshair_h.set_visible(False)
            self._crosshair_v.set_visible(False)
            return

        try:
            val = float(self._last_frame_raw[y_raw, x])
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}, I = {val:g}")
        except Exception:
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}")

        self._crosshair_h.set_ydata([y, y])
        self._crosshair_v.set_xdata([x, x])
        self._crosshair_h.set_visible(True)
        self._crosshair_v.set_visible(True)

        width = int(self.linecutWidthSpin.value())
        x_raw = w - 1 - x  # display col x maps to physical col x_raw after fliplr
        hcut = self.cam.linecut_horizontal(self._last_frame_raw, y_raw, width=width, mode="sum")
        vcut = self.cam.linecut_vertical(self._last_frame_raw, x_raw, width=width, mode="sum")
        if hcut is not None:
            x_axis = self._get_x_axis_data(w)
            self.line_h.set_data(x_axis, hcut[::-1])  # reverse to match fliplr display
            self.ax_h.relim()
            self.ax_h.autoscale_view()
        if vcut is not None:
            self.line_v.set_data(vcut, np.arange(vcut.size))
            self.ax_v.relim()
            self.ax_v.autoscale_view()

    def _update_wavelength_axis_cache(self, force: bool = False):
        try:
            wl = self.cam.get_wavelength_axis(force=bool(force))
        except Exception:
            wl = None
        if wl is None:
            self._wl_axis = None
            return None
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            self._wl_axis = None
            return None
        if arr.size == 0 or np.allclose(arr, 0.0):
            self._wl_axis = None
            return None
        self._wl_axis = arr
        try:
            if self._last_frame_raw is not None:
                self._update_image_wavelength_axis(self._last_frame_raw.shape[1])
                self.canvas.draw_idle()
        except Exception:
            pass
        return arr

    def _get_x_axis_data(self, width: int):
        wl = getattr(self, "_wl_axis", None)
        if wl is None:
            return np.arange(width)
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            return np.arange(width)
        if arr.size != width or np.allclose(arr, 0.0):
            return np.arange(width)
        return arr

    def _maybe_update_xaxis_label(self, width: int) -> None:
        wl = getattr(self, "_wl_axis", None)
        has_wl = False
        if wl is not None:
            try:
                arr = np.asarray(wl, dtype=float).ravel()
                has_wl = arr.size == width and (not np.allclose(arr, 0.0))
            except Exception:
                has_wl = False
        mode = "wavelength" if has_wl else "pixel"
        if mode == self._xaxis_mode:
            return
        self._xaxis_mode = mode
        if mode == "wavelength":
            self.ax_h.set_xlabel("Wavelength (nm)")
        else:
            self.ax_h.set_xlabel("X (px)")

    def _update_image_wavelength_axis(self, width: int) -> None:
        if self._ax_img_top is None:
            return
        wl = getattr(self, "_wl_axis", None)
        if wl is None:
            self._ax_img_top.set_visible(False)
            return
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            self._ax_img_top.set_visible(False)
            return
        if arr.size != width or np.allclose(arr, 0.0):
            self._ax_img_top.set_visible(False)
            return

        n_ticks = 6
        pix = np.linspace(0, max(1, width - 1), n_ticks)
        pix_int = np.unique(np.clip(np.round(pix).astype(int), 0, width - 1))
        labels = [f"{arr[i]:.1f}" for i in pix_int]
        self._ax_img_top.set_xlim(self.ax_img.get_xlim())
        self._ax_img_top.set_xticks(pix_int)
        self._ax_img_top.set_xticklabels(labels)
        self._ax_img_top.set_visible(True)

    # -----------------
    # Helpers
    # -----------------
    def _apply_connect_defaults(self):
        try:
            self.frameApiCombo.setCurrentText("Snap")
            self.acqCombo.setCurrentText("Single")
            self.triggerCombo.setCurrentText("Internal")
        except Exception:
            pass

        try:
            for i in range(self.outputAmpCombo.count()):
                label = self.outputAmpCombo.itemText(i).strip().lower()
                if "conventional" in label:
                    self.outputAmpCombo.setCurrentIndex(i)
                    break
        except Exception:
            pass

        try:
            best_idx = None
            best_mhz = -1.0
            for i in range(self.readoutRateCombo.count()):
                label = self.readoutRateCombo.itemText(i)
                m = re.search(r"(\d+(?:\.\d+)?)\s*(mhz|khz|hz)?", label.lower())
                if not m:
                    continue
                v = float(m.group(1))
                unit = m.group(2) or ""
                if unit == "khz":
                    v = v / 1000.0
                elif unit == "hz":
                    v = v / 1e6
                if v > best_mhz:
                    best_mhz = v
                    best_idx = i
            if best_idx is not None:
                self.readoutRateCombo.setCurrentIndex(best_idx)
        except Exception:
            pass

        try:
            best_idx = None
            for i in range(self.preampCombo.count()):
                if self.preampCombo.itemText(i).strip().lower() == "1x":
                    best_idx = i
                    break
            if best_idx is None:
                best_x = None
                for i in range(self.preampCombo.count()):
                    label = self.preampCombo.itemText(i).strip().lower()
                    m = re.search(r"(\d+(?:\.\d+)?)\s*x", label)
                    if not m:
                        continue
                    gx = float(m.group(1))
                    if best_x is None or gx < best_x:
                        best_x = gx
                        best_idx = i
            if best_idx is not None:
                self.preampCombo.setCurrentIndex(best_idx)
        except Exception:
            pass

        try:
            self._apply_settings_to_camera()
        except Exception:
            pass

    def _wait_idle(self, timeout_s: float = 2.0, poll_s: float = 0.05) -> bool:
        cam_wrap = self.cam
        if cam_wrap is None:
            return True
        inner = getattr(cam_wrap, "cam", cam_wrap)
        t0 = time.time()
        while time.time() - t0 < float(timeout_s):
            try:
                if hasattr(inner, "is_acquiring"):
                    if not inner.is_acquiring():
                        return True
                elif hasattr(inner, "get_status"):
                    st = inner.get_status()
                    s = str(st).lower()
                    if ("acq" not in s) and ("acquir" not in s) and ("run" not in s):
                        return True
                else:
                    return True
            except Exception:
                pass
            time.sleep(float(poll_s))
        return False

    def _refresh_ampmode_choices(self):
        amps, rates, gains = self.cam.get_amp_mode_choices()
        if amps:
            self.outputAmpCombo.clear()
            self.outputAmpCombo.addItems(amps)
            if DEFAULT_OUTPUT_AMP in amps:
                self.outputAmpCombo.setCurrentText(DEFAULT_OUTPUT_AMP)
        if rates:
            self.readoutRateCombo.clear()
            self.readoutRateCombo.addItems(rates)
            try:
                def to_mhz(label: str):
                    s = label.strip().lower()
                    if "khz" in s:
                        return float(s.replace("khz at 16-bit", "").replace("khz", "")) / 1000.0
                    if "mhz" in s:
                        return float(s.replace("mhz at 16-bit", "").replace("mhz", ""))
                    return 0.0
                best = max(rates, key=to_mhz)
                self.readoutRateCombo.setCurrentText(best)
            except Exception:
                pass
        if gains:
            self.preampCombo.clear()
            self.preampCombo.addItems(gains)
            if DEFAULT_PREAMP_GAIN in gains:
                self.preampCombo.setCurrentText(DEFAULT_PREAMP_GAIN)
            else:
                try:
                    def to_gain(label: str):
                        return float(label.lower().replace("x", ""))
                    best = min(gains, key=to_gain)
                    self.preampCombo.setCurrentText(best)
                except Exception:
                    pass

    def _set_rate_combo_by_mhz(self, mhz: float) -> None:
        try:
            target = float(mhz)
        except Exception:
            return
        best_idx = None
        best_diff = None
        for i in range(self.readoutRateCombo.count()):
            label = self.readoutRateCombo.itemText(i).strip().lower()
            m = re.search(r"(\d+(?:\.\d+)?)\s*(khz|mhz|hz)?", label)
            if not m:
                continue
            v = float(m.group(1))
            unit = m.group(2) or ""
            if unit == "khz":
                v = v / 1000.0
            elif unit == "hz":
                v = v / 1e6
            diff = abs(v - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx is not None:
            self.readoutRateCombo.setCurrentIndex(best_idx)

    def _sync_amp_mode_from_info(self, info: dict) -> None:
        if not info:
            return
        if info.get("amp_label"):
            self._set_combo_by_text(self.outputAmpCombo, str(info.get("amp_label")))
        mhz = info.get("hsspeed_mhz")
        if mhz is not None:
            self._set_rate_combo_by_mhz(mhz)
        if info.get("gain_label"):
            self._set_combo_by_text(self.preampCombo, str(info.get("gain_label")))

    def _current_grating_number(self) -> int:
        txt = self.gratingCombo.currentText().strip()
        if not txt:
            return int(DEFAULT_GRATING)
        if txt.isdigit():
            return int(txt)
        try:
            return int(txt.split(":")[0].strip())
        except Exception:
            return int(DEFAULT_GRATING)

    @staticmethod
    def _set_combo_by_text(combo: QtWidgets.QComboBox, text: str) -> None:
        s = str(text).strip().lower()
        for i in range(combo.count()):
            t = combo.itemText(i).strip().lower()
            if s == t or s in t:
                combo.setCurrentIndex(i)
                return

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.on_stop()
        except Exception:
            pass
        try:
            self.on_disconnect()
        except Exception:
            pass
        event.accept()

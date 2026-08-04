import csv
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ..ph300_wrapper import PicoHarp300
from .config import (
    DEFAULT_DLL_PATH,
    DEFAULT_TACQ_MS, DEFAULT_TARGET_BIN_PS, DEFAULT_POLL_HZ,
    DEFAULT_SYNC_OFFSET_PS, DEFAULT_HIST_OFFSET_PS,
    DEFAULT_CH0_LEVEL, DEFAULT_CH0_ZC, DEFAULT_CH1_LEVEL, DEFAULT_CH1_ZC,
    DEFAULT_XMIN_PS, DEFAULT_XMAX_PS, DEFAULT_YMIN, DEFAULT_YMAX,
    BTN_W, POLL_MS_MIN,
)
from .plots import HistogramPlotWidget


@dataclass
class Trace:
    label: str
    time_ps: np.ndarray
    counts: np.ndarray
    visible: bool = True


class PH300Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ph: Optional[PicoHarp300] = None
        self.connected = False
        self.measuring = False

        self.live_time_ps: Optional[np.ndarray] = None
        self.live_counts: Optional[np.ndarray] = None

        self.traces: Dict[str, Trace] = {}

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_poll)

        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 4, 4)
        outer.setSpacing(4)

        main = QtWidgets.QHBoxLayout()
        main.setSpacing(8)
        outer.addLayout(main, 1)

        self._status_bar = QtWidgets.QStatusBar()
        outer.addWidget(self._status_bar)
        self._status_bar.showMessage("Idle")

        # ---- Left: plot + plot controls ----
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)

        self.plot = HistogramPlotWidget(self)
        left.addWidget(self.plot, 1)

        plot_box = QtWidgets.QGroupBox("Plot Controls")
        plot_grid = QtWidgets.QGridLayout(plot_box)

        self.xmin_ps = QtWidgets.QDoubleSpinBox()
        self.xmin_ps.setRange(0, 1e12); self.xmin_ps.setDecimals(3)
        self.xmin_ps.setValue(DEFAULT_XMIN_PS)

        self.xmax_ps = QtWidgets.QDoubleSpinBox()
        self.xmax_ps.setRange(0, 1e12); self.xmax_ps.setDecimals(3)
        self.xmax_ps.setValue(DEFAULT_XMAX_PS)

        self.ymin = QtWidgets.QDoubleSpinBox()
        self.ymin.setRange(0, 1e12); self.ymin.setDecimals(0)
        self.ymin.setValue(DEFAULT_YMIN)

        self.ymax = QtWidgets.QDoubleSpinBox()
        self.ymax.setRange(0, 1e12); self.ymax.setDecimals(0)
        self.ymax.setValue(DEFAULT_YMAX)

        self.logy = QtWidgets.QCheckBox("Log Y (counts)")
        self.logy.setChecked(False)

        self.btn_apply_plot = QtWidgets.QPushButton("Apply")
        self.btn_apply_plot.setFixedWidth(BTN_W)
        self.btn_apply_plot.clicked.connect(self.update_plot_range)

        plot_grid.addWidget(QtWidgets.QLabel("X min (ps):"), 0, 0)
        plot_grid.addWidget(self.xmin_ps, 0, 1)
        plot_grid.addWidget(QtWidgets.QLabel("X max (ps):"), 0, 2)
        plot_grid.addWidget(self.xmax_ps, 0, 3)
        plot_grid.addWidget(QtWidgets.QLabel("Y min:"), 1, 0)
        plot_grid.addWidget(self.ymin, 1, 1)
        plot_grid.addWidget(QtWidgets.QLabel("Y max:"), 1, 2)
        plot_grid.addWidget(self.ymax, 1, 3)
        plot_grid.addWidget(self.logy, 2, 0, 1, 2)
        plot_grid.addWidget(self.btn_apply_plot, 2, 2, 1, 2)

        left.addWidget(plot_box)
        main.addLayout(left, 3)

        # ---- Right: control panels ----
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)
        main.addLayout(right, 2)

        # Connection
        conn_box = QtWidgets.QGroupBox("Connection")
        conn_grid = QtWidgets.QGridLayout(conn_box)

        self.dll_path = QtWidgets.QLineEdit(DEFAULT_DLL_PATH)
        self.device_index = QtWidgets.QSpinBox()
        self.device_index.setRange(0, 7); self.device_index.setValue(0)

        self.verbose = QtWidgets.QCheckBox("Verbose")
        self.verbose.setChecked(True)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.setFixedWidth(BTN_W)
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.setFixedWidth(BTN_W)
        self.btn_disconnect.setEnabled(False)

        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_disconnect.clicked.connect(self.on_disconnect)

        conn_grid.addWidget(QtWidgets.QLabel("PHLib DLL:"), 0, 0)
        conn_grid.addWidget(self.dll_path, 0, 1, 1, 3)
        conn_grid.addWidget(QtWidgets.QLabel("Device idx:"), 1, 0)
        conn_grid.addWidget(self.device_index, 1, 1)
        conn_grid.addWidget(self.verbose, 1, 2)
        conn_grid.addWidget(self.btn_connect, 2, 1)
        conn_grid.addWidget(self.btn_disconnect, 2, 2)

        right.addWidget(conn_box)

        # Acquisition / Timing
        acq_box = QtWidgets.QGroupBox("Acquisition / Timing")
        acq_grid = QtWidgets.QGridLayout(acq_box)

        self.target_bin_ps = QtWidgets.QDoubleSpinBox()
        self.target_bin_ps.setRange(0.1, 1e6); self.target_bin_ps.setDecimals(3)
        self.target_bin_ps.setValue(DEFAULT_TARGET_BIN_PS)

        self.tacq_ms = QtWidgets.QSpinBox()
        self.tacq_ms.setRange(1, 10_000_000); self.tacq_ms.setValue(DEFAULT_TACQ_MS)

        self.poll_hz = QtWidgets.QDoubleSpinBox()
        self.poll_hz.setRange(0.1, 20.0); self.poll_hz.setDecimals(2)
        self.poll_hz.setValue(DEFAULT_POLL_HZ)

        self.sync_div = QtWidgets.QComboBox()
        self.sync_div.addItems(["1", "2", "4", "8"])
        self.sync_div.setCurrentText("1")

        self.sync_offset_ps = QtWidgets.QSpinBox()
        self.sync_offset_ps.setRange(-999999999, 999999999)
        self.sync_offset_ps.setValue(DEFAULT_SYNC_OFFSET_PS)

        self.hist_offset_ps = QtWidgets.QSpinBox()
        self.hist_offset_ps.setRange(-999999999, 999999999)
        self.hist_offset_ps.setValue(DEFAULT_HIST_OFFSET_PS)

        acq_grid.addWidget(QtWidgets.QLabel("Target bin (ps):"), 0, 0)
        acq_grid.addWidget(self.target_bin_ps, 0, 1)
        acq_grid.addWidget(QtWidgets.QLabel("Acq time (ms):"), 0, 2)
        acq_grid.addWidget(self.tacq_ms, 0, 3)
        acq_grid.addWidget(QtWidgets.QLabel("Poll rate (Hz):"), 1, 0)
        acq_grid.addWidget(self.poll_hz, 1, 1)
        acq_grid.addWidget(QtWidgets.QLabel("Sync divider:"), 1, 2)
        acq_grid.addWidget(self.sync_div, 1, 3)
        acq_grid.addWidget(QtWidgets.QLabel("Sync offset (ps):"), 2, 0)
        acq_grid.addWidget(self.sync_offset_ps, 2, 1)
        acq_grid.addWidget(QtWidgets.QLabel("Hist offset (ps):"), 2, 2)
        acq_grid.addWidget(self.hist_offset_ps, 2, 3)

        right.addWidget(acq_box)

        # Inputs (CFD)
        inp_box = QtWidgets.QGroupBox("Inputs (CFD)")
        inp_grid = QtWidgets.QGridLayout(inp_box)

        self.ch0_level = QtWidgets.QSpinBox()
        self.ch0_level.setRange(0, 800); self.ch0_level.setValue(DEFAULT_CH0_LEVEL)

        self.ch0_zc = QtWidgets.QSpinBox()
        self.ch0_zc.setRange(0, 20); self.ch0_zc.setValue(DEFAULT_CH0_ZC)

        self.ch1_level = QtWidgets.QSpinBox()
        self.ch1_level.setRange(0, 800); self.ch1_level.setValue(DEFAULT_CH1_LEVEL)

        self.ch1_zc = QtWidgets.QSpinBox()
        self.ch1_zc.setRange(0, 20); self.ch1_zc.setValue(DEFAULT_CH1_ZC)

        inp_grid.addWidget(QtWidgets.QLabel("Ch0 (Sync) level (mV):"), 0, 0)
        inp_grid.addWidget(self.ch0_level, 0, 1)
        inp_grid.addWidget(QtWidgets.QLabel("ZC (mV):"), 0, 2)
        inp_grid.addWidget(self.ch0_zc, 0, 3)
        inp_grid.addWidget(QtWidgets.QLabel("Ch1 (Det) level (mV):"), 1, 0)
        inp_grid.addWidget(self.ch1_level, 1, 1)
        inp_grid.addWidget(QtWidgets.QLabel("ZC (mV):"), 1, 2)
        inp_grid.addWidget(self.ch1_zc, 1, 3)

        right.addWidget(inp_box)

        # Control buttons
        ctrl_box = QtWidgets.QGroupBox("Control")
        ctrl_row = QtWidgets.QHBoxLayout(ctrl_box)

        self.btn_arm = QtWidgets.QPushButton("Apply Settings")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_snapshot = QtWidgets.QPushButton("Snapshot")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

        self.btn_arm.clicked.connect(self.on_apply_settings)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_snapshot.clicked.connect(self.on_snapshot)

        ctrl_row.addWidget(self.btn_arm)
        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addWidget(self.btn_snapshot)

        right.addWidget(ctrl_box)

        # Rates / Warnings
        stat_box = QtWidgets.QGroupBox("Rates / Warnings")
        stat_grid = QtWidgets.QGridLayout(stat_box)

        self.lbl_sync = QtWidgets.QLabel("Sync rate: —")
        self.lbl_ch0 = QtWidgets.QLabel("Ch0 rate:  —")
        self.lbl_ch1 = QtWidgets.QLabel("Ch1 rate:  —")
        self.lbl_warn = QtWidgets.QLabel("Warnings: —")
        self.lbl_warn.setWordWrap(True)

        stat_grid.addWidget(self.lbl_sync, 0, 0, 1, 2)
        stat_grid.addWidget(self.lbl_ch0, 1, 0, 1, 2)
        stat_grid.addWidget(self.lbl_ch1, 2, 0, 1, 2)
        stat_grid.addWidget(self.lbl_warn, 3, 0, 1, 2)

        right.addWidget(stat_box)

        # Recorded Traces
        trace_box = QtWidgets.QGroupBox("Recorded Traces")
        trace_vbox = QtWidgets.QVBoxLayout(trace_box)

        self.trace_list = QtWidgets.QListWidget()
        self.trace_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.trace_list.itemChanged.connect(self.on_trace_toggle)

        trace_btn_row = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_save = QtWidgets.QPushButton("Save CSV")
        self.btn_clear = QtWidgets.QPushButton("Clear All")

        self.btn_delete.clicked.connect(self.on_delete_selected)
        self.btn_save.clicked.connect(self.on_save_selected)
        self.btn_clear.clicked.connect(self.on_clear_all)

        trace_btn_row.addWidget(self.btn_delete)
        trace_btn_row.addWidget(self.btn_save)
        trace_btn_row.addWidget(self.btn_clear)

        trace_vbox.addWidget(self.trace_list)
        trace_vbox.addLayout(trace_btn_row)

        right.addWidget(trace_box, 1)

        # Apply initial plot range
        self.update_plot_range()

    # ---- Status bar ----

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    # ---- Connection ----

    def on_connect(self):
        try:
            self.ph = PicoHarp300(
                self.dll_path.text().strip(),
                device_index=int(self.device_index.value()),
                verbose=bool(self.verbose.isChecked()),
            )
            ver = self.ph.get_library_version()
            serial = self.ph.open()
            self.ph.initialize_histogramming()
            self.ph.calibrate()

            hw = self.ph.get_hardware_info()
            self.statusBar().showMessage(
                f"Connected: PHLib {ver}, serial {serial}, {hw['model']} v{hw['version']}"
            )

            self.connected = True
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_arm.setEnabled(True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Connect failed", str(e))
            self.connected = False
            self.ph = None

    def on_disconnect(self):
        self.on_stop()
        try:
            if self.ph is not None:
                self.ph.close()
        except Exception:
            pass

        self.ph = None
        self.connected = False
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.statusBar().showMessage("Disconnected.")

    # ---- Apply settings ----

    def on_apply_settings(self):
        if not self.connected or self.ph is None:
            return
        try:
            self.ph.set_sync_div(int(self.sync_div.currentText()))
            self.ph.set_sync_offset_ps(int(self.sync_offset_ps.value()))
            self.ph.set_hist_offset_ps(int(self.hist_offset_ps.value()))

            self.ph.set_input_cfd(0, int(self.ch0_level.value()), int(self.ch0_zc.value()))
            self.ph.set_input_cfd(1, int(self.ch1_level.value()), int(self.ch1_zc.value()))

            _, res_ps = self.ph.set_target_resolution_ps(float(self.target_bin_ps.value()))

            if self.xmax_ps.value() <= 0 or abs(self.xmax_ps.value() - 65536 * 8.0) < 1:
                self.xmax_ps.setValue(65536 * res_ps)

            time.sleep(0.2)

            self.live_time_ps = np.asarray(self.ph.make_time_axis_ps(), dtype=np.float64)

            self.statusBar().showMessage(f"Settings applied. Resolution = {res_ps:.3f} ps")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Apply settings failed", str(e))

    # ---- Measurement controls ----

    def on_start(self):
        if not self.connected or self.ph is None:
            return
        if self.measuring:
            return
        try:
            self.on_apply_settings()

            self.ph.clear_hist_mem(block=0)
            self.ph.set_stop_overflow(enable=False)
            self.ph.start_meas(int(self.tacq_ms.value()))

            self.measuring = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_snapshot.setEnabled(True)

            interval_ms = int(round(1000.0 / float(self.poll_hz.value())))
            interval_ms = max(POLL_MS_MIN, interval_ms)
            self.timer.start(interval_ms)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start failed", str(e))
            self.measuring = False

    def on_stop(self):
        if self.ph is None:
            return
        if self.timer.isActive():
            self.timer.stop()
        if self.measuring:
            try:
                self.ph.stop_meas()
            except Exception:
                pass
        self.measuring = False
        self.plot.remove_live()
        self.btn_start.setEnabled(self.connected)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

    def on_snapshot(self):
        if self.live_time_ps is None or self.live_counts is None:
            return
        label = time.strftime("snap_%Y%m%d_%H%M%S")
        self._add_trace(label, self.live_time_ps.copy(), self.live_counts.copy())
        self.statusBar().showMessage(f"Snapshot saved: {label}")

    def on_poll(self):
        if self.ph is None or not self.measuring:
            return
        try:
            rs = self.ph.get_rates_and_warnings()
            self.lbl_sync.setText(f"Sync rate: {rs.sync_rate_hz} Hz")
            self.lbl_ch0.setText(f"Ch0 rate:  {rs.ch0_rate_hz} Hz")
            self.lbl_ch1.setText(f"Ch1 rate:  {rs.ch1_rate_hz} Hz")

            if rs.warnings_bitfield != 0:
                txt = rs.warnings_text if rs.warnings_text else f"(bitfield=0x{rs.warnings_bitfield:08X})"
                self.lbl_warn.setText(f"Warnings:\n{txt}")
            else:
                self.lbl_warn.setText("Warnings: (none)")

            hist = np.asarray(self.ph.read_histogram(block=0), dtype=np.uint32)
            self.live_counts = hist

            if self.live_time_ps is None:
                self.live_time_ps = np.asarray(self.ph.make_time_axis_ps(), dtype=np.float64)

            self.plot.update_live(self.live_time_ps, self.live_counts)

            if self.ph.ctc_done():
                label = time.strftime("meas_%Y%m%d_%H%M%S")
                self._add_trace(label, self.live_time_ps.copy(), self.live_counts.copy())
                self.statusBar().showMessage(f"Measurement complete. Stored: {label}")
                self.on_stop()

        except Exception as e:
            self.statusBar().showMessage(f"Poll error: {e}")
            self.on_stop()

    # ---- Trace management ----

    def _add_trace(self, label: str, time_ps: np.ndarray, counts: np.ndarray):
        base = label
        i = 1
        while label in self.traces:
            label = f"{base}_{i}"
            i += 1

        self.traces[label] = Trace(label=label, time_ps=time_ps, counts=counts, visible=True)
        self.plot.add_trace(label, time_ps, counts)

        item = QtWidgets.QListWidgetItem(label)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.Checked)
        self.trace_list.addItem(item)

    def on_trace_toggle(self, item: QtWidgets.QListWidgetItem):
        label = item.text()
        if label in self.traces:
            visible = (item.checkState() == QtCore.Qt.Checked)
            self.traces[label].visible = visible
            self.plot.set_visible(label, visible)

    def on_delete_selected(self):
        items = self.trace_list.selectedItems()
        if not items:
            return
        for it in items:
            label = it.text()
            if label in self.traces:
                del self.traces[label]
                self.plot.remove_trace(label)
            self.trace_list.takeItem(self.trace_list.row(it))

    def on_clear_all(self):
        self.traces.clear()
        self.trace_list.clear()
        self.plot.clear_all()

    def on_save_selected(self):
        items = self.trace_list.selectedItems()
        if not items:
            QtWidgets.QMessageBox.information(self, "Save CSV", "Select one or more traces to save.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "trpl_traces.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        labels = [it.text() for it in items if it.text() in self.traces]
        if not labels:
            return

        t = self.traces[labels[0]].time_ps
        n = len(t)

        for lab in labels[1:]:
            if len(self.traces[lab].time_ps) != n or len(self.traces[lab].counts) != n:
                QtWidgets.QMessageBox.warning(
                    self, "Save CSV",
                    f"Trace '{lab}' has a different length. Record traces with consistent settings."
                )
                return

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ps"] + labels)
            for i in range(n):
                row = [float(t[i])]
                for lab in labels:
                    row.append(int(self.traces[lab].counts[i]))
                w.writerow(row)

        self.statusBar().showMessage(f"Saved CSV: {path}")

    # ---- Plot range ----

    def update_plot_range(self):
        self.plot.apply_range(
            xmin=float(self.xmin_ps.value()),
            xmax=float(self.xmax_ps.value()),
            ymin=float(self.ymin.value()),
            ymax=float(self.ymax.value()),
            logy=self.logy.isChecked(),
        )

    # ---- Close ----

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        try:
            if self.ph is not None and self.measuring:
                self.ph.stop_meas()
        except Exception:
            pass
        try:
            if self.ph is not None:
                self.ph.close()
        except Exception:
            pass
        event.accept()

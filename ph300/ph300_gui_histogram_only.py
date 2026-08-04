import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QLineEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ph300.ph300_wrapper import PicoHarp300


# -----------------------------
# User config
# -----------------------------
DEFAULT_DLL_PATH = r"C:\Program Files\PicoQuant\PH300-PHLibv30\demos\64\c\TTTRmode\PHLib64.dll"


# -----------------------------
# Simple plotting widget
# -----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


@dataclass
class Trace:
    label: str
    time_ps: np.ndarray
    counts: np.ndarray
    visible: bool = True


# -----------------------------
# Main GUI
# -----------------------------
class PH300Gui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PicoHarp 300 TRPL (Histogramming) - Minimal GUI")

        # State
        self.ph: Optional[PicoHarp300] = None
        self.connected = False
        self.measuring = False

        self.live_time_ps: Optional[np.ndarray] = None
        self.live_counts: Optional[np.ndarray] = None

        self.traces: Dict[str, Trace] = {}
        self.live_label = "__LIVE__"

        # Timer for polling
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_poll)

        # Build UI
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: plot
        left = QVBoxLayout()
        self.canvas = MplCanvas(self)
        left.addWidget(self.canvas)

        # Plot controls
        plot_ctrl = QGroupBox("Plot Controls")
        plot_grid = QGridLayout(plot_ctrl)

        self.xmin_ps = QDoubleSpinBox()
        self.xmin_ps.setRange(0, 1e12)
        self.xmin_ps.setDecimals(3)
        self.xmin_ps.setValue(0.0)

        self.xmax_ps = QDoubleSpinBox()
        self.xmax_ps.setRange(0, 1e12)
        self.xmax_ps.setDecimals(3)
        # self.xmax_ps.setValue(65536 * 8.0)  # initial guess for 8 ps
        self.xmax_ps.setValue(70000)

        self.ymin = QDoubleSpinBox()
        self.ymin.setRange(0, 1e12)
        self.ymin.setDecimals(0)
        self.ymin.setValue(0.0)

        self.ymax = QDoubleSpinBox()
        self.ymax.setRange(0, 1e12)
        self.ymax.setDecimals(0)
        # self.ymax.setValue(1e4)
        self.ymax.setValue(150)

        self.logy = QCheckBox("Log Y (counts)")
        self.logy.setChecked(False)

        self.btn_apply_plot = QPushButton("Apply")
        self.btn_apply_plot.clicked.connect(self.update_plot)

        plot_grid.addWidget(QLabel("X min (ps)"), 0, 0)
        plot_grid.addWidget(self.xmin_ps, 0, 1)
        plot_grid.addWidget(QLabel("X max (ps)"), 0, 2)
        plot_grid.addWidget(self.xmax_ps, 0, 3)

        plot_grid.addWidget(QLabel("Y min"), 1, 0)
        plot_grid.addWidget(self.ymin, 1, 1)
        plot_grid.addWidget(QLabel("Y max"), 1, 2)
        plot_grid.addWidget(self.ymax, 1, 3)

        plot_grid.addWidget(self.logy, 2, 0, 1, 2)
        plot_grid.addWidget(self.btn_apply_plot, 2, 2, 1, 2)

        left.addWidget(plot_ctrl)

        root.addLayout(left, 3)

        # Right: controls + trace list
        right = QVBoxLayout()
        root.addLayout(right, 2)

        # Connection group
        conn_box = QGroupBox("Connection")
        conn_grid = QGridLayout(conn_box)

        self.dll_path = QLineEdit(DEFAULT_DLL_PATH)
        self.device_index = QSpinBox()
        self.device_index.setRange(0, 7)
        self.device_index.setValue(0)

        self.verbose = QCheckBox("Verbose wrapper logs")
        self.verbose.setChecked(True)

        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)

        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_disconnect.clicked.connect(self.on_disconnect)

        conn_grid.addWidget(QLabel("PHLib DLL path"), 0, 0)
        conn_grid.addWidget(self.dll_path, 0, 1, 1, 3)
        conn_grid.addWidget(QLabel("Device index"), 1, 0)
        conn_grid.addWidget(self.device_index, 1, 1)
        conn_grid.addWidget(self.verbose, 1, 2, 1, 2)
        conn_grid.addWidget(self.btn_connect, 2, 1)
        conn_grid.addWidget(self.btn_disconnect, 2, 2)

        right.addWidget(conn_box)

        # Acquisition settings
        acq_box = QGroupBox("Acquisition / Timing")
        acq_grid = QGridLayout(acq_box)

        self.target_bin_ps = QDoubleSpinBox()
        self.target_bin_ps.setRange(0.1, 1e6)
        self.target_bin_ps.setDecimals(3)
        self.target_bin_ps.setValue(8.0)

        self.tacq_ms = QSpinBox()
        self.tacq_ms.setRange(1, 10_000_000)
        self.tacq_ms.setValue(5000)

        self.poll_hz = QDoubleSpinBox()
        self.poll_hz.setRange(0.1, 20.0)
        self.poll_hz.setDecimals(2)
        self.poll_hz.setValue(1.0)

        self.sync_div = QComboBox()
        self.sync_div.addItems(["1", "2", "4", "8"])
        self.sync_div.setCurrentText("1")

        self.sync_offset_ps = QSpinBox()
        self.sync_offset_ps.setRange(-999999999, 999999999)
        self.sync_offset_ps.setValue(30000)

        self.hist_offset_ps = QSpinBox()
        self.hist_offset_ps.setRange(-999999999, 999999999)
        self.hist_offset_ps.setValue(30000)

        acq_grid.addWidget(QLabel("Target bin width (ps)"), 0, 0)
        acq_grid.addWidget(self.target_bin_ps, 0, 1)
        acq_grid.addWidget(QLabel("Acq time (ms)"), 0, 2)
        acq_grid.addWidget(self.tacq_ms, 0, 3)

        acq_grid.addWidget(QLabel("Poll rate (Hz)"), 1, 0)
        acq_grid.addWidget(self.poll_hz, 1, 1)
        acq_grid.addWidget(QLabel("Sync divider"), 1, 2)
        acq_grid.addWidget(self.sync_div, 1, 3)

        acq_grid.addWidget(QLabel("Sync offset (ps)"), 2, 0)
        acq_grid.addWidget(self.sync_offset_ps, 2, 1)
        acq_grid.addWidget(QLabel("Hist offset (ps)"), 2, 2)
        acq_grid.addWidget(self.hist_offset_ps, 2, 3)

        right.addWidget(acq_box)

        # Input settings
        inp_box = QGroupBox("Inputs (CFD)")
        inp_grid = QGridLayout(inp_box)

        self.ch0_level = QSpinBox()
        self.ch0_level.setRange(0, 800)
        self.ch0_level.setValue(80)

        self.ch0_zc = QSpinBox()
        self.ch0_zc.setRange(0, 20)
        self.ch0_zc.setValue(20)

        self.ch1_level = QSpinBox()
        self.ch1_level.setRange(0, 800)
        self.ch1_level.setValue(300)

        self.ch1_zc = QSpinBox()
        self.ch1_zc.setRange(0, 20)
        self.ch1_zc.setValue(20)

        inp_grid.addWidget(QLabel("Ch0 (Sync) CFD level (mV)"), 0, 0)
        inp_grid.addWidget(self.ch0_level, 0, 1)
        inp_grid.addWidget(QLabel("Ch0 (Sync) ZC (mV)"), 0, 2)
        inp_grid.addWidget(self.ch0_zc, 0, 3)

        inp_grid.addWidget(QLabel("Ch1 (Det) CFD level (mV)"), 1, 0)
        inp_grid.addWidget(self.ch1_level, 1, 1)
        inp_grid.addWidget(QLabel("Ch1 (Det) ZC (mV)"), 1, 2)
        inp_grid.addWidget(self.ch1_zc, 1, 3)

        right.addWidget(inp_box)

        # Control buttons
        btn_box = QGroupBox("Control")
        btn_row = QHBoxLayout(btn_box)

        self.btn_arm = QPushButton("Apply Settings")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_snapshot = QPushButton("Snapshot Trace")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

        self.btn_arm.clicked.connect(self.on_apply_settings)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_snapshot.clicked.connect(self.on_snapshot)

        btn_row.addWidget(self.btn_arm)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_snapshot)

        right.addWidget(btn_box)

        # Rates + warnings
        stat_box = QGroupBox("Rates / Warnings")
        stat_grid = QGridLayout(stat_box)

        self.lbl_sync = QLabel("Sync rate: —")
        self.lbl_ch0 = QLabel("Ch0 rate: —")
        self.lbl_ch1 = QLabel("Ch1 rate: —")
        self.lbl_warn = QLabel("Warnings: —")
        self.lbl_warn.setWordWrap(True)

        stat_grid.addWidget(self.lbl_sync, 0, 0, 1, 2)
        stat_grid.addWidget(self.lbl_ch0, 1, 0, 1, 2)
        stat_grid.addWidget(self.lbl_ch1, 2, 0, 1, 2)
        stat_grid.addWidget(self.lbl_warn, 3, 0, 1, 2)

        right.addWidget(stat_box)

        # Trace list + actions
        trace_box = QGroupBox("Recorded Traces")
        trace_layout = QVBoxLayout(trace_box)

        self.trace_list = QListWidget()
        from PySide6.QtWidgets import QAbstractItemView
        self.trace_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.trace_list.itemChanged.connect(self.on_trace_toggle)

        trace_btn_row = QHBoxLayout()
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_save = QPushButton("Save Selected CSV")
        self.btn_clear = QPushButton("Clear All")

        self.btn_delete.clicked.connect(self.on_delete_selected)
        self.btn_save.clicked.connect(self.on_save_selected)
        self.btn_clear.clicked.connect(self.on_clear_all)

        trace_btn_row.addWidget(self.btn_delete)
        trace_btn_row.addWidget(self.btn_save)
        trace_btn_row.addWidget(self.btn_clear)

        trace_layout.addWidget(self.trace_list)
        trace_layout.addLayout(trace_btn_row)

        right.addWidget(trace_box)

        # Initial plot
        self.canvas.ax.set_xlabel("Time (ps)")
        self.canvas.ax.set_ylabel("Counts")
        self.canvas.draw()

    # -----------------------------
    # Connection
    # -----------------------------
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
            self.statusBar().showMessage(f"Connected: PHLib {ver}, serial {serial}, {hw['model']} v{hw['version']}")

            self.connected = True
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_arm.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Connect failed", str(e))
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

    # -----------------------------
    # Apply settings
    # -----------------------------
    def on_apply_settings(self):
        if not self.connected or self.ph is None:
            return
        try:
            # Sync divider + offsets
            self.ph.set_sync_div(int(self.sync_div.currentText()))
            self.ph.set_sync_offset_ps(int(self.sync_offset_ps.value()))
            self.ph.set_hist_offset_ps(int(self.hist_offset_ps.value()))

            # CFD
            self.ph.set_input_cfd(0, int(self.ch0_level.value()), int(self.ch0_zc.value()))
            self.ph.set_input_cfd(1, int(self.ch1_level.value()), int(self.ch1_zc.value()))

            # Set timing resolution (via binning search)
            _, res_ps = self.ph.set_target_resolution_ps(float(self.target_bin_ps.value()))
            # Update plot xmax default if user hasn't customized too much
            if self.xmax_ps.value() <= 0 or abs(self.xmax_ps.value() - 65536 * 8.0) < 1:
                self.xmax_ps.setValue(65536 * res_ps)

            # Let rate meter settle
            time.sleep(0.2)

            # Precompute time axis
            self.live_time_ps = np.asarray(self.ph.make_time_axis_ps(), dtype=np.float64)

            self.statusBar().showMessage(f"Settings applied. Current resolution = {res_ps:.3f} ps")
        except Exception as e:
            QMessageBox.critical(self, "Apply settings failed", str(e))

    # -----------------------------
    # Measurement controls
    # -----------------------------
    def on_start(self):
        if not self.connected or self.ph is None:
            return
        if self.measuring:
            return
        try:
            # Apply settings first (to ensure time axis exists)
            self.on_apply_settings()

            self.ph.clear_hist_mem(block=0)
            self.ph.set_stop_overflow(enable=False)
            self.ph.start_meas(int(self.tacq_ms.value()))

            self.measuring = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_snapshot.setEnabled(True)

            interval_ms = int(round(1000.0 / float(self.poll_hz.value())))
            interval_ms = max(50, interval_ms)
            self.timer.start(interval_ms)

            # Ensure plot shows live
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Start failed", str(e))
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
        self.btn_start.setEnabled(self.connected)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

    def on_snapshot(self):
        """Store the current live trace as a recorded trace."""
        if self.live_time_ps is None or self.live_counts is None:
            return
        label = time.strftime("snap_%Y%m%d_%H%M%S")
        self.add_trace(label, self.live_time_ps.copy(), self.live_counts.copy())
        self.statusBar().showMessage(f"Snapshot saved: {label}")

    def on_poll(self):
        """Timer tick: update rates/warnings + live histogram + completion."""
        if self.ph is None or not self.measuring:
            return
        try:
            rs = self.ph.get_rates_and_warnings()
            self.lbl_sync.setText(f"Sync rate: {rs.sync_rate_hz} Hz")
            self.lbl_ch0.setText(f"Ch0 rate:   {rs.ch0_rate_hz} Hz")
            self.lbl_ch1.setText(f"Ch1 rate:   {rs.ch1_rate_hz} Hz")

            if rs.warnings_bitfield != 0:
                txt = rs.warnings_text if rs.warnings_text else f"(bitfield=0x{rs.warnings_bitfield:08X})"
                self.lbl_warn.setText(f"Warnings:\n{txt}")
            else:
                self.lbl_warn.setText("Warnings: (none)")

            hist = np.asarray(self.ph.read_histogram(block=0), dtype=np.uint32)
            self.live_counts = hist

            # If time axis not set for some reason, compute now
            if self.live_time_ps is None:
                self.live_time_ps = np.asarray(self.ph.make_time_axis_ps(), dtype=np.float64)

            # Auto-stop detection
            if self.ph.ctc_done():
                # Store final trace automatically
                label = time.strftime("meas_%Y%m%d_%H%M%S")
                self.add_trace(label, self.live_time_ps.copy(), self.live_counts.copy())
                self.statusBar().showMessage(f"Measurement complete. Stored trace: {label}")
                self.on_stop()
                return

            self.update_plot()

        except Exception as e:
            self.statusBar().showMessage(f"Poll error: {e}")
            # If polling fails repeatedly, stop to avoid leaving the device in an unknown state
            self.on_stop()

    # -----------------------------
    # Trace list management
    # -----------------------------
    def add_trace(self, label: str, time_ps: np.ndarray, counts: np.ndarray):
        # Ensure unique label
        base = label
        i = 1
        while label in self.traces:
            label = f"{base}_{i}"
            i += 1

        self.traces[label] = Trace(label=label, time_ps=time_ps, counts=counts, visible=True)

        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Checked)
        self.trace_list.addItem(item)

        self.update_plot()

    def on_trace_toggle(self, item: QListWidgetItem):
        label = item.text()
        if label in self.traces:
            self.traces[label].visible = (item.checkState() == Qt.Checked)
            self.update_plot()

    def on_delete_selected(self):
        items = self.trace_list.selectedItems()
        if not items:
            return
        for it in items:
            label = it.text()
            if label in self.traces:
                del self.traces[label]
            row = self.trace_list.row(it)
            self.trace_list.takeItem(row)
        self.update_plot()

    def on_clear_all(self):
        self.traces.clear()
        self.trace_list.clear()
        self.update_plot()

    def on_save_selected(self):
        items = self.trace_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Save CSV", "Select one or more traces to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "trpl_traces.csv", "CSV Files (*.csv)")
        if not path:
            return

        labels = [it.text() for it in items if it.text() in self.traces]
        if not labels:
            return

        # Use time axis from the first selected trace
        t = self.traces[labels[0]].time_ps
        n = len(t)

        # Basic consistency check
        for lab in labels[1:]:
            if len(self.traces[lab].time_ps) != n or len(self.traces[lab].counts) != n:
                QMessageBox.warning(
                    self,
                    "Save CSV",
                    f"Trace '{lab}' has a different length. Please snapshot/record traces with consistent settings."
                )
                return

        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ps"] + labels)
            for i in range(n):
                row = [float(t[i])]
                for lab in labels:
                    row.append(int(self.traces[lab].counts[i]))
                w.writerow(row)

        self.statusBar().showMessage(f"Saved CSV: {path}")


    # -----------------------------
    # Plot update
    # -----------------------------
    def update_plot(self):
        ax = self.canvas.ax
        ax.clear()

        # Plot recorded traces
        for tr in self.traces.values():
            if tr.visible:
                ax.plot(tr.time_ps, tr.counts, linewidth=1.0, label=tr.label)

        # Plot live trace (not recorded)
        if self.live_time_ps is not None and self.live_counts is not None and self.measuring:
            ax.plot(self.live_time_ps, self.live_counts, linewidth=1.5, label="LIVE")

        # Axes / scaling
        xmin = float(self.xmin_ps.value())
        xmax = float(self.xmax_ps.value())
        if xmax > xmin:
            ax.set_xlim(xmin, xmax)

        ymin = float(self.ymin.value())
        ymax = float(self.ymax.value())
        if self.logy.isChecked():
            ax.set_yscale("log")
            # Avoid log(0) issues by enforcing lower bound > 0
            ax.set_ylim(max(1.0, ymin if ymin > 0 else 1.0), ymax if ymax > 0 else 1.0)
        else:
            ax.set_yscale("linear")
            if ymax > ymin:
                ax.set_ylim(ymin, ymax)

        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Counts")
        if self.traces or (self.measuring and self.live_counts is not None):
            ax.legend(loc="best", fontsize=8)

        ax.grid(True, alpha=0.25)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = PH300Gui()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

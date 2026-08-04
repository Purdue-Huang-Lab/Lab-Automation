"""
TRPL Gate- and Power-dependent measurement GUI
Hardware: ND-wheel (Thorlabs rotation stage) + 2× Keithley SMU (gates) + PicoHarp 300
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

import pyqtgraph as pg

from keithley.keithley_wrapper import make_resource_manager, KeithleySMU, SweepController
from keithley.gui_v2.plots import PlotWidget as VIPlotWidget
from keithley.gui_v2.workers import RampThread as GateRampThread
from keithley.gui_v2.workers import HomeThread as GateHomeThread
from keithley.gui_v2.config import POLL_TIMEOUT_MS

from ph300.ph300_wrapper import PicoHarp300
from ph300.gui_v2.plots import HistogramPlotWidget

from rot.rot_wrapper import RotationStage, MotionController, list_kinesis_serials
from rot.gui.workers import MoveThread as WheelMoveThread
from rot.gui.workers import HomeThread as WheelHomeThread

from .config import (
    VISA_DLL,
    DEFAULT_DLL_PATH, DEFAULT_GATE_A_RESOURCE, DEFAULT_GATE_B_RESOURCE, DEFAULT_ROT_SERIAL,
    DEFAULT_TARGET_BIN_PS, DEFAULT_TACQ_MS,
    DEFAULT_SYNC_DIV, DEFAULT_SYNC_OFFSET_PS, DEFAULT_HIST_OFFSET_PS,
    DEFAULT_CH0_LEVEL, DEFAULT_CH0_ZC, DEFAULT_CH1_LEVEL, DEFAULT_CH1_ZC,
    DEFAULT_ICOMP_NA, DEFAULT_GATE_V,
    DEFAULT_GATE_SETTLE_S, DEFAULT_WHEEL_SETTLE_S,
    DEFAULT_XMIN_PS, DEFAULT_XMAX_PS,
    POLL_MS, BTN_W,
)
from .workers import HomeGatesWorker, SweepWorker

try:
    from measurements.config import DATA_DIR
except Exception:
    DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop")


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class MeasPoint:
    idx:     int
    angle:   float
    va:      float
    vb:      float
    time_ps: np.ndarray
    counts:  np.ndarray


# ---------------------------------------------------------------------------
# Helper: parse a comma/newline-separated text field → list[float]
# ---------------------------------------------------------------------------

def _parse_float_list(text: str) -> List[float]:
    vals = []
    for tok in text.replace("\n", ",").replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return vals


# ---------------------------------------------------------------------------
# Compact status LED label
# ---------------------------------------------------------------------------

def _make_status_lbl(text: str = "—") -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setMinimumWidth(200)
    return lbl


def _set_connected(lbl: QtWidgets.QLabel, connected: bool, detail: str = ""):
    if connected:
        lbl.setText(f"● {detail}" if detail else "● Connected")
        lbl.setStyleSheet("color: #2ca02c; font-weight: 600;")
    else:
        lbl.setText("○ Disconnected")
        lbl.setStyleSheet("color: #d62728; font-weight: 600;")


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class TRPLGatePowerWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- device handles ----
        self.rm:        Optional[object]        = None
        self.ph:        Optional[PicoHarp300]   = None
        self.dev_a:     Optional[KeithleySMU]   = None
        self.dev_b:     Optional[KeithleySMU]   = None
        self.rot_stage: Optional[RotationStage] = None

        # ---- state ----
        self._ph_res_ps: float = DEFAULT_TARGET_BIN_PS
        self._ph_time_ps: Optional[np.ndarray] = None

        self._poll_va: float = 0.0
        self._poll_vb: float = 0.0
        self._poll_ia: float = 0.0
        self._poll_ib: float = 0.0

        self._sweep_worker:   Optional[SweepWorker]     = None
        self._home_worker:    Optional[HomeGatesWorker] = None

        self._acq_running:    bool  = False
        self._acq_t0:         float = 0.0
        self._acq_timeout_s:  float = 0.0
        self._gate_a_ramp:    Optional[GateRampThread]           = None
        self._gate_b_ramp:    Optional[GateRampThread]           = None
        self._gate_a_home:    Optional[GateHomeThread]           = None
        self._gate_b_home:    Optional[GateHomeThread]           = None
        self._wheel_move:     Optional[WheelMoveThread]          = None
        self._wheel_home:     Optional[WheelHomeThread]          = None

        self._meas_data: List[MeasPoint] = []
        self._cached_va_list: List[float] = []
        self._live_hist_history: deque = deque(maxlen=5)

        # ---- poll timer ----
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._on_poll)

        # ---- build UI ----
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 4)
        outer.setSpacing(4)

        outer.addWidget(self._build_dashboard())
        outer.addWidget(self._build_action_bar())
        outer.addWidget(self._build_tabs(), 1)

        self._status_bar = QtWidgets.QStatusBar()
        outer.addWidget(self._status_bar)
        self._status_bar.showMessage("Idle — connect devices to begin.")

        self._sync_buttons()

    # =======================================================================
    # Dashboard
    # =======================================================================

    def _build_dashboard(self) -> QtWidgets.QWidget:
        dash = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(dash)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        hl.addWidget(self._build_status_panel(), 2)
        hl.addWidget(self._build_quick_panel(), 5)
        return dash

    # ---- left: status ----

    def _build_status_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Status")
        grid = QtWidgets.QGridLayout(box)
        grid.setVerticalSpacing(6)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        def readout(txt="—"):
            lbl = QtWidgets.QLabel(txt)
            lbl.setFont(mono)
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lbl.setMinimumWidth(90)
            return lbl

        # PH300
        self.lbl_ph_conn    = _make_status_lbl()
        self.lbl_ph_bin     = QtWidgets.QLabel("—  ps")
        self.lbl_ph_acq     = QtWidgets.QLabel("—  s")
        self.lbl_ph_sync    = QtWidgets.QLabel("—")
        self.lbl_ph_ch0     = QtWidgets.QLabel("—")
        self.lbl_ph_ch1     = QtWidgets.QLabel("—")
        self.lbl_ph_warn    = QtWidgets.QLabel("—")
        self.lbl_ph_warn.setWordWrap(True)
        self.lbl_ph_warn.setStyleSheet("color: #d62728; font-size: 8pt;")
        grid.addWidget(QtWidgets.QLabel("<b>PicoHarp 300</b>"), 0, 0, 1, 4)
        grid.addWidget(self.lbl_ph_conn,           1, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Bin:"),   1, 2)
        grid.addWidget(self.lbl_ph_bin,            1, 3)
        grid.addWidget(QtWidgets.QLabel("Acq:"),   2, 2)
        grid.addWidget(self.lbl_ph_acq,            2, 3)
        grid.addWidget(QtWidgets.QLabel("Sync:"),  3, 0)
        grid.addWidget(self.lbl_ph_sync,           3, 1)
        grid.addWidget(QtWidgets.QLabel("Ch0:"),   3, 2)
        grid.addWidget(self.lbl_ph_ch0,            3, 3)
        grid.addWidget(QtWidgets.QLabel("Ch1:"),   4, 2)
        grid.addWidget(self.lbl_ph_ch1,            4, 3)
        grid.addWidget(self.lbl_ph_warn,           5, 0, 1, 4)

        # Gate A
        self.lbl_a_v = readout()
        self.lbl_a_i = readout()
        grid.addWidget(QtWidgets.QLabel("<b>Gate A</b>"), 6, 0, 1, 4)
        grid.addWidget(QtWidgets.QLabel("V:"),  7, 0)
        grid.addWidget(self.lbl_a_v, 7, 1)
        grid.addWidget(QtWidgets.QLabel("I:"),  7, 2)
        grid.addWidget(self.lbl_a_i, 7, 3)

        # Gate B
        self.lbl_b_v = readout()
        self.lbl_b_i = readout()
        grid.addWidget(QtWidgets.QLabel("<b>Gate B</b>"), 8, 0, 1, 4)
        grid.addWidget(QtWidgets.QLabel("V:"),  9, 0)
        grid.addWidget(self.lbl_b_v, 9, 1)
        grid.addWidget(QtWidgets.QLabel("I:"),  9, 2)
        grid.addWidget(self.lbl_b_i, 9, 3)

        # ND Wheel
        self.lbl_wheel_conn  = _make_status_lbl()
        self.lbl_wheel_angle = QtWidgets.QLabel("—  °")
        grid.addWidget(QtWidgets.QLabel("<b>ND Wheel</b>"), 10, 0, 1, 4)
        grid.addWidget(self.lbl_wheel_conn,        11, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Angle:"), 11, 2)
        grid.addWidget(self.lbl_wheel_angle,       11, 3)

        grid.setRowStretch(12, 1)
        return box

    # ---- right: quick controls ----

    def _build_quick_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)

        vbox.addWidget(self._build_conn_box())

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._build_ph300_quick())
        row.addWidget(self._build_gate_quick("A"))
        row.addWidget(self._build_gate_quick("B"))
        row.addWidget(self._build_wheel_quick())
        vbox.addLayout(row, 1)
        return panel

    def _build_conn_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Device Connections")
        grid = QtWidgets.QGridLayout(box)
        grid.setVerticalSpacing(3)

        self.edit_dll    = QtWidgets.QLineEdit(DEFAULT_DLL_PATH)
        self.edit_visa_a = QtWidgets.QLineEdit(DEFAULT_GATE_A_RESOURCE)
        self.edit_visa_b = QtWidgets.QLineEdit(DEFAULT_GATE_B_RESOURCE)
        self.edit_serial = QtWidgets.QLineEdit(DEFAULT_ROT_SERIAL)
        self.edit_serial.setPlaceholderText("Thorlabs serial, e.g. 55000001")

        btn_scan = QtWidgets.QPushButton("Scan")
        btn_scan.setFixedWidth(55)
        btn_scan.clicked.connect(self._on_scan_serials)

        grid.addWidget(QtWidgets.QLabel("PHLib DLL:"),    0, 0)
        grid.addWidget(self.edit_dll,                      0, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Gate A VISA:"),  1, 0)
        grid.addWidget(self.edit_visa_a,                   1, 1)
        grid.addWidget(QtWidgets.QLabel("Gate B VISA:"),  1, 2)
        grid.addWidget(self.edit_visa_b,                   1, 3)
        grid.addWidget(QtWidgets.QLabel("Wheel serial:"), 2, 0)
        grid.addWidget(self.edit_serial,                   2, 1)
        grid.addWidget(btn_scan,                           2, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return box

    def _build_ph300_quick(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("PicoHarp")
        grid = QtWidgets.QGridLayout(box)

        self.spin_bin_ps = QtWidgets.QDoubleSpinBox()
        self.spin_bin_ps.setRange(0.1, 1e6); self.spin_bin_ps.setDecimals(3)
        self.spin_bin_ps.setValue(DEFAULT_TARGET_BIN_PS)

        self.spin_tacq = QtWidgets.QDoubleSpinBox()
        self.spin_tacq.setRange(0.001, 10000); self.spin_tacq.setDecimals(3)
        self.spin_tacq.setValue(DEFAULT_TACQ_MS / 1000)

        self.spin_poll_ms = QtWidgets.QSpinBox()
        self.spin_poll_ms.setRange(100, 10000); self.spin_poll_ms.setSingleStep(100)
        self.spin_poll_ms.setValue(POLL_MS); self.spin_poll_ms.setSuffix(" ms")
        self.spin_poll_ms.valueChanged.connect(
            lambda ms: self._poll_timer.setInterval(ms) if self._poll_timer.isActive() else None)

        btn = QtWidgets.QPushButton("Apply")
        btn.setFixedWidth(BTN_W)
        btn.clicked.connect(self._on_apply_ph300_quick)

        grid.addWidget(QtWidgets.QLabel("Bin (ps):"),   0, 0)
        grid.addWidget(self.spin_bin_ps,                 0, 1)
        grid.addWidget(QtWidgets.QLabel("Acq (s):"),    1, 0)
        grid.addWidget(self.spin_tacq,                   1, 1)
        grid.addWidget(QtWidgets.QLabel("Poll (ms):"),  2, 0)
        grid.addWidget(self.spin_poll_ms,                2, 1)
        grid.addWidget(btn,                              3, 0, 1, 2)
        grid.setRowStretch(4, 1)
        return box

    def _build_gate_quick(self, tag: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(f"Gate {tag}")
        grid = QtWidgets.QGridLayout(box)

        icomp = QtWidgets.QDoubleSpinBox()
        icomp.setRange(0, 1e9); icomp.setDecimals(3); icomp.setValue(DEFAULT_ICOMP_NA)
        icomp.setSuffix(" nA")

        vset = QtWidgets.QDoubleSpinBox()
        vset.setRange(-210, 210); vset.setDecimals(4); vset.setValue(DEFAULT_GATE_V)
        vset.setSuffix(" V")

        btn_set  = QtWidgets.QPushButton("Set (ramp)")
        btn_home = QtWidgets.QPushButton("Home → 0 V")
        btn_set.setFixedWidth(BTN_W)
        btn_home.setFixedWidth(BTN_W)

        grid.addWidget(QtWidgets.QLabel("Icomp:"), 0, 0)
        grid.addWidget(icomp,                       0, 1)
        grid.addWidget(QtWidgets.QLabel("Vset:"),  1, 0)
        grid.addWidget(vset,                        1, 1)
        grid.addWidget(btn_set,                     2, 0, 1, 2)
        grid.addWidget(btn_home,                    3, 0, 1, 2)
        grid.setRowStretch(4, 1)

        if tag == "A":
            self.spin_icomp_a, self.spin_v_a = icomp, vset
            self.btn_set_a,  self.btn_home_a = btn_set, btn_home
            btn_set.clicked.connect(self._on_set_gate_a)
            btn_home.clicked.connect(self._on_home_gate_a)
        else:
            self.spin_icomp_b, self.spin_v_b = icomp, vset
            self.btn_set_b,  self.btn_home_b = btn_set, btn_home
            btn_set.clicked.connect(self._on_set_gate_b)
            btn_home.clicked.connect(self._on_home_gate_b)
        return box

    def _build_wheel_quick(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("ND Wheel")
        grid = QtWidgets.QGridLayout(box)

        self.spin_wheel_angle = QtWidgets.QDoubleSpinBox()
        self.spin_wheel_angle.setRange(0, 360); self.spin_wheel_angle.setDecimals(3)
        self.spin_wheel_angle.setSuffix(" °")

        self.btn_wheel_move = QtWidgets.QPushButton("Move")
        self.btn_wheel_home = QtWidgets.QPushButton("Home")
        self.btn_wheel_move.setFixedWidth(BTN_W)
        self.btn_wheel_home.setFixedWidth(BTN_W)

        self.btn_wheel_move.clicked.connect(self._on_wheel_move)
        self.btn_wheel_home.clicked.connect(self._on_wheel_home)

        grid.addWidget(QtWidgets.QLabel("Target (°):"), 0, 0)
        grid.addWidget(self.spin_wheel_angle,            0, 1)
        grid.addWidget(self.btn_wheel_move,              1, 0, 1, 2)
        grid.addWidget(self.btn_wheel_home,              2, 0, 1, 2)
        grid.setRowStretch(3, 1)
        return box

    # ---- action bar ----

    def _build_action_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        hl  = QtWidgets.QHBoxLayout(bar)
        hl.setContentsMargins(0, 0, 0, 0)

        self.btn_init      = QtWidgets.QPushButton("Initialize")
        self.btn_disconn   = QtWidgets.QPushButton("Disconnect")
        self.btn_take_hist = QtWidgets.QPushButton("Take Histogram")
        self.btn_stop_acq  = QtWidgets.QPushButton("Stop Acq")
        self.btn_sweep     = QtWidgets.QPushButton("Start Sweep")
        self.btn_home_all  = QtWidgets.QPushButton("Home Both Gates")
        self.btn_abort     = QtWidgets.QPushButton("Abort")

        self.btn_abort.setStyleSheet("background: #d62728; color: white; font-weight: 700;")
        self.btn_stop_acq.setStyleSheet("background: #ff7f0e; color: white; font-weight: 700;")

        for btn in (self.btn_init, self.btn_disconn, self.btn_take_hist, self.btn_stop_acq,
                    self.btn_sweep, self.btn_home_all, self.btn_abort):
            btn.setMinimumWidth(120)
            hl.addWidget(btn)

        hl.addStretch(1)

        self.btn_init.clicked.connect(self.on_initialize)
        self.btn_disconn.clicked.connect(self.on_disconnect)
        self.btn_take_hist.clicked.connect(self.on_take_histogram)
        self.btn_stop_acq.clicked.connect(self.on_stop_acquisition)
        self.btn_sweep.clicked.connect(self.on_start_sweep)
        self.btn_home_all.clicked.connect(self.on_home_both_gates)
        self.btn_abort.clicked.connect(self.on_abort)

        return bar

    # =======================================================================
    # Tabs
    # =======================================================================

    def _build_tabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_tab_monitor(),  "1 · Live Monitor")
        tabs.addTab(self._build_tab_params(),   "2 · Sweep Parameters")
        tabs.addTab(self._build_tab_browser(),  "3 · Data Browser")
        return tabs

    # ---- Tab 1: Live Monitor ----

    def _build_tab_monitor(self) -> QtWidgets.QWidget:
        w   = QtWidgets.QWidget()
        hl  = QtWidgets.QHBoxLayout(w)
        hl.setSpacing(6)

        # Left: V&I plots for Gate A and Gate B stacked
        left = QtWidgets.QVBoxLayout()
        self.plot_vi_a = VIPlotWidget()
        self.plot_vi_b = VIPlotWidget()
        self.plot_vi_a.pw.setTitle("Gate A — V & I vs Samples")
        self.plot_vi_b.pw.setTitle("Gate B — V & I vs Samples")
        left.addWidget(self.plot_vi_a)
        left.addWidget(self.plot_vi_b)
        hl.addLayout(left, 1)

        # Right: current PicoHarp histogram + plot controls
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(4)

        hist_header = QtWidgets.QHBoxLayout()
        lbl_hist = QtWidgets.QLabel("<b>Current Histogram  (last 5)</b>")
        btn_clear_live = QtWidgets.QPushButton("Clear")
        btn_clear_live.setFixedWidth(60)
        btn_clear_live.clicked.connect(self._clear_live_hist)
        hist_header.addWidget(lbl_hist)
        hist_header.addStretch()
        hist_header.addWidget(btn_clear_live)
        right.addLayout(hist_header)

        self.live_hist_plot = HistogramPlotWidget()
        right.addWidget(self.live_hist_plot, 1)

        # Plot controls (mirrors ph300/gui_v2/widget.py)
        plot_ctrl = QtWidgets.QGroupBox("Plot Controls")
        pg_grid   = QtWidgets.QGridLayout(plot_ctrl)

        self.live_xmin = QtWidgets.QDoubleSpinBox()
        self.live_xmin.setRange(0, 1e12); self.live_xmin.setDecimals(1)
        self.live_xmin.setValue(DEFAULT_XMIN_PS)

        self.live_xmax = QtWidgets.QDoubleSpinBox()
        self.live_xmax.setRange(0, 1e12); self.live_xmax.setDecimals(1)
        self.live_xmax.setValue(DEFAULT_XMAX_PS)

        self.live_ymin = QtWidgets.QDoubleSpinBox()
        self.live_ymin.setRange(0, 1e12); self.live_ymin.setDecimals(0)
        self.live_ymin.setValue(0.0)

        self.live_ymax = QtWidgets.QDoubleSpinBox()
        self.live_ymax.setRange(0, 1e12); self.live_ymax.setDecimals(0)
        self.live_ymax.setValue(1000.0)

        self.live_logy = QtWidgets.QCheckBox("Log Y")
        self.live_logy.setChecked(False)

        btn_apply_live = QtWidgets.QPushButton("Apply")
        btn_apply_live.setFixedWidth(BTN_W)
        btn_apply_live.clicked.connect(self._apply_live_plot_range)

        pg_grid.addWidget(QtWidgets.QLabel("X min (ps):"), 0, 0)
        pg_grid.addWidget(self.live_xmin,                   0, 1)
        pg_grid.addWidget(QtWidgets.QLabel("X max (ps):"), 0, 2)
        pg_grid.addWidget(self.live_xmax,                   0, 3)
        pg_grid.addWidget(QtWidgets.QLabel("Y min:"),      1, 0)
        pg_grid.addWidget(self.live_ymin,                   1, 1)
        pg_grid.addWidget(QtWidgets.QLabel("Y max:"),      1, 2)
        pg_grid.addWidget(self.live_ymax,                   1, 3)
        pg_grid.addWidget(self.live_logy,                   2, 0, 1, 2)
        pg_grid.addWidget(btn_apply_live,                   2, 2, 1, 2)

        right.addWidget(plot_ctrl)
        hl.addLayout(right, 1)

        return w

    # ---- Tab 2: Sweep Parameters ----

    def _build_tab_params(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QtWidgets.QWidget()
        vbox  = QtWidgets.QVBoxLayout(inner)
        vbox.setSpacing(8)

        # --- sweep lists ---
        lists_row = QtWidgets.QHBoxLayout()
        lists_row.addWidget(self._build_list_editor("ND Wheel Angles (°)", "angles"), 1)
        lists_row.addWidget(self._build_gate_voltage_box(), 2)
        vbox.addLayout(lists_row)

        # --- PH300 full settings ---
        ph_box = QtWidgets.QGroupBox("PicoHarp 300 Full Settings")
        ph_grid = QtWidgets.QGridLayout(ph_box)

        self.p2_sync_div = QtWidgets.QComboBox()
        self.p2_sync_div.addItems(["1", "2", "4", "8"])
        self.p2_sync_div.setCurrentText(str(DEFAULT_SYNC_DIV))

        self.p2_sync_off = QtWidgets.QSpinBox()
        self.p2_sync_off.setRange(-999999999, 999999999)
        self.p2_sync_off.setValue(DEFAULT_SYNC_OFFSET_PS)

        self.p2_hist_off = QtWidgets.QSpinBox()
        self.p2_hist_off.setRange(-999999999, 999999999)
        self.p2_hist_off.setValue(DEFAULT_HIST_OFFSET_PS)

        self.p2_ch0_lvl = QtWidgets.QSpinBox()
        self.p2_ch0_lvl.setRange(0, 800); self.p2_ch0_lvl.setValue(DEFAULT_CH0_LEVEL)
        self.p2_ch0_zc  = QtWidgets.QSpinBox()
        self.p2_ch0_zc.setRange(0, 20);   self.p2_ch0_zc.setValue(DEFAULT_CH0_ZC)
        self.p2_ch1_lvl = QtWidgets.QSpinBox()
        self.p2_ch1_lvl.setRange(0, 800); self.p2_ch1_lvl.setValue(DEFAULT_CH1_LEVEL)
        self.p2_ch1_zc  = QtWidgets.QSpinBox()
        self.p2_ch1_zc.setRange(0, 20);   self.p2_ch1_zc.setValue(DEFAULT_CH1_ZC)

        ph_grid.addWidget(QtWidgets.QLabel("Sync divider:"),      0, 0); ph_grid.addWidget(self.p2_sync_div, 0, 1)
        ph_grid.addWidget(QtWidgets.QLabel("Sync offset (ps):"),  0, 2); ph_grid.addWidget(self.p2_sync_off, 0, 3)
        ph_grid.addWidget(QtWidgets.QLabel("Hist offset (ps):"),  0, 4); ph_grid.addWidget(self.p2_hist_off, 0, 5)
        ph_grid.addWidget(QtWidgets.QLabel("Ch0 (Sync) level:"),  1, 0); ph_grid.addWidget(self.p2_ch0_lvl,  1, 1)
        ph_grid.addWidget(QtWidgets.QLabel("Ch0 ZC (mV):"),       1, 2); ph_grid.addWidget(self.p2_ch0_zc,   1, 3)
        ph_grid.addWidget(QtWidgets.QLabel("Ch1 (Det) level:"),   2, 0); ph_grid.addWidget(self.p2_ch1_lvl,  2, 1)
        ph_grid.addWidget(QtWidgets.QLabel("Ch1 ZC (mV):"),       2, 2); ph_grid.addWidget(self.p2_ch1_zc,   2, 3)
        vbox.addWidget(ph_box)

        # --- settle times + output path ---
        misc_box = QtWidgets.QGroupBox("Timing & Output")
        misc_grid = QtWidgets.QGridLayout(misc_box)

        self.p2_gate_settle = QtWidgets.QDoubleSpinBox()
        self.p2_gate_settle.setRange(0, 60); self.p2_gate_settle.setDecimals(2)
        self.p2_gate_settle.setValue(DEFAULT_GATE_SETTLE_S); self.p2_gate_settle.setSuffix(" s")

        self.p2_wheel_settle = QtWidgets.QDoubleSpinBox()
        self.p2_wheel_settle.setRange(0, 60); self.p2_wheel_settle.setDecimals(2)
        self.p2_wheel_settle.setValue(DEFAULT_WHEEL_SETTLE_S); self.p2_wheel_settle.setSuffix(" s")

        self.p2_out_dir = QtWidgets.QLineEdit(str(DATA_DIR))
        btn_browse = QtWidgets.QPushButton("Browse…"); btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse_out_dir)

        self.p2_folder_name = QtWidgets.QLineEdit(
            time.strftime("trpl_%Y%m%d_%H%M%S"))
        self.p2_folder_name.setPlaceholderText("subfolder name")
        btn_reset_name = QtWidgets.QPushButton("↺ Now"); btn_reset_name.setFixedWidth(60)
        btn_reset_name.setToolTip("Reset folder name to current timestamp")
        btn_reset_name.clicked.connect(
            lambda: self.p2_folder_name.setText(time.strftime("trpl_%Y%m%d_%H%M%S")))

        misc_grid.addWidget(QtWidgets.QLabel("Gate settle:"),   0, 0); misc_grid.addWidget(self.p2_gate_settle,  0, 1)
        misc_grid.addWidget(QtWidgets.QLabel("Wheel settle:"),  0, 2); misc_grid.addWidget(self.p2_wheel_settle, 0, 3)
        misc_grid.addWidget(QtWidgets.QLabel("Output dir:"),    1, 0); misc_grid.addWidget(self.p2_out_dir,      1, 1, 1, 3)
        misc_grid.addWidget(btn_browse,                          1, 4)
        misc_grid.addWidget(QtWidgets.QLabel("Folder name:"),   2, 0); misc_grid.addWidget(self.p2_folder_name,  2, 1, 1, 3)
        misc_grid.addWidget(btn_reset_name,                      2, 4)
        vbox.addWidget(misc_box)

        vbox.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_list_editor(self, title: str, key: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        vb  = QtWidgets.QVBoxLayout(box)

        te = QtWidgets.QTextEdit()
        te.setPlaceholderText("e.g.  0, 15.5, 30\nor one per line")
        te.setMaximumHeight(90)

        lbl_preview = QtWidgets.QLabel("—")
        lbl_preview.setWordWrap(True)
        lbl_preview.setStyleSheet("color: #555; font-size: 8pt;")

        btn_set = QtWidgets.QPushButton("Set list")
        btn_set.setFixedWidth(80)

        def _parse():
            vals = _parse_float_list(te.toPlainText())
            if vals:
                lbl_preview.setText(f"→ {len(vals)} values: {vals[:6]}" +
                                    (" …" if len(vals) > 6 else ""))
            else:
                lbl_preview.setText("(empty)")
            return vals

        btn_set.clicked.connect(_parse)

        vb.addWidget(te)
        vb.addWidget(btn_set)
        vb.addWidget(lbl_preview)

        if key == "angles":
            self._te_angles = te; self._parse_angles = _parse
        return box

    def _build_gate_voltage_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Gate Voltages  (Vb = ratio × Va)")
        vb  = QtWidgets.QVBoxLayout(box)
        vb.setSpacing(6)

        # Mode radio buttons
        mode_row = QtWidgets.QHBoxLayout()
        self._rb_manual = QtWidgets.QRadioButton("Manual list")
        self._rb_range  = QtWidgets.QRadioButton("Range  start : step : end")
        self._rb_manual.setChecked(True)
        mode_row.addWidget(self._rb_manual)
        mode_row.addWidget(self._rb_range)
        mode_row.addStretch()
        vb.addLayout(mode_row)

        # Ratio
        ratio_row = QtWidgets.QHBoxLayout()
        self._spin_vb_ratio = QtWidgets.QDoubleSpinBox()
        self._spin_vb_ratio.setRange(-1000, 1000)
        self._spin_vb_ratio.setDecimals(5)
        self._spin_vb_ratio.setValue(1.8)
        self._spin_vb_ratio.setFixedWidth(120)
        ratio_row.addWidget(QtWidgets.QLabel("Ratio  (Vb = ratio × Va):"))
        ratio_row.addWidget(self._spin_vb_ratio)
        ratio_row.addStretch()
        vb.addLayout(ratio_row)

        # Stacked pages
        self._gate_stack = QtWidgets.QStackedWidget()

        # Page 0 — manual list
        page_manual = QtWidgets.QWidget()
        pm = QtWidgets.QVBoxLayout(page_manual)
        pm.setContentsMargins(0, 0, 0, 0)
        self._te_va = QtWidgets.QTextEdit()
        self._te_va.setPlaceholderText("Top gate Va values, e.g.  -2, -1, 0, 1, 2\nor one per line")
        self._te_va.setMaximumHeight(72)
        self._lbl_manual_preview = QtWidgets.QLabel("—")
        self._lbl_manual_preview.setWordWrap(True)
        self._lbl_manual_preview.setStyleSheet("color: #555; font-size: 8pt;")
        btn_set = QtWidgets.QPushButton("Set list"); btn_set.setFixedWidth(80)
        btn_set.clicked.connect(self._parse_gate_manual)
        pm.addWidget(self._te_va)
        pm.addWidget(btn_set)
        pm.addWidget(self._lbl_manual_preview)
        self._gate_stack.addWidget(page_manual)

        # Page 1 — range
        page_range = QtWidgets.QWidget()
        pr = QtWidgets.QGridLayout(page_range)
        pr.setContentsMargins(0, 0, 0, 0)
        self._spin_va_start = QtWidgets.QDoubleSpinBox()
        self._spin_va_start.setRange(-210, 210); self._spin_va_start.setDecimals(5)
        self._spin_va_start.setValue(-1.0)
        self._spin_va_step = QtWidgets.QDoubleSpinBox()
        self._spin_va_step.setRange(-100, 100); self._spin_va_step.setDecimals(5)
        self._spin_va_step.setValue(0.5)
        self._spin_va_end = QtWidgets.QDoubleSpinBox()
        self._spin_va_end.setRange(-210, 210); self._spin_va_end.setDecimals(5)
        self._spin_va_end.setValue(1.0)
        self._lbl_range_preview = QtWidgets.QLabel("—")
        self._lbl_range_preview.setWordWrap(True)
        self._lbl_range_preview.setStyleSheet("color: #555; font-size: 8pt;")
        btn_gen = QtWidgets.QPushButton("Generate"); btn_gen.setFixedWidth(80)
        btn_gen.clicked.connect(self._parse_gate_range)
        pr.addWidget(QtWidgets.QLabel("Start (V):"), 0, 0)
        pr.addWidget(self._spin_va_start,             0, 1)
        pr.addWidget(QtWidgets.QLabel("Step (V):"),  0, 2)
        pr.addWidget(self._spin_va_step,              0, 3)
        pr.addWidget(QtWidgets.QLabel("End (V):"),   0, 4)
        pr.addWidget(self._spin_va_end,               0, 5)
        pr.addWidget(btn_gen,                         1, 0)
        pr.addWidget(self._lbl_range_preview,         1, 1, 1, 5)
        self._gate_stack.addWidget(page_range)

        vb.addWidget(self._gate_stack)

        # Vb computed preview
        self._lbl_vb_preview = QtWidgets.QLabel("Vb: —")
        self._lbl_vb_preview.setWordWrap(True)
        self._lbl_vb_preview.setStyleSheet("color: #1f77b4; font-size: 8pt;")
        vb.addWidget(self._lbl_vb_preview)

        # Wire mode switch
        self._rb_manual.toggled.connect(
            lambda chk: self._gate_stack.setCurrentIndex(0) if chk else None)
        self._rb_range.toggled.connect(
            lambda chk: self._gate_stack.setCurrentIndex(1) if chk else None)

        return box

    # ---- gate voltage parsing ----

    def _parse_gate_manual(self):
        vals = _parse_float_list(self._te_va.toPlainText())
        self._cached_va_list = vals
        self._update_gate_preview(vals, self._lbl_manual_preview)

    def _parse_gate_range(self):
        start = float(self._spin_va_start.value())
        step  = float(self._spin_va_step.value())
        end   = float(self._spin_va_end.value())
        vals: List[float] = []
        if step != 0:
            v = start
            if step > 0:
                while v <= end + abs(step) * 1e-6:
                    vals.append(round(v, 8)); v += step
            else:
                while v >= end - abs(step) * 1e-6:
                    vals.append(round(v, 8)); v += step
        self._cached_va_list = vals
        self._update_gate_preview(vals, self._lbl_range_preview)

    def _update_gate_preview(self, va_list: List[float],
                              va_lbl: QtWidgets.QLabel):
        if va_list:
            short = [f"{v:.3f}" for v in va_list[:6]]
            va_lbl.setText(f"Va ({len(va_list)}): {short}" +
                           (" …" if len(va_list) > 6 else ""))
            ratio  = float(self._spin_vb_ratio.value())
            vb_short = [f"{ratio * v:.3f}" for v in va_list[:6]]
            self._lbl_vb_preview.setText(
                f"Vb ({len(va_list)}): {vb_short}" +
                (" …" if len(va_list) > 6 else ""))
        else:
            va_lbl.setText("(empty)")
            self._lbl_vb_preview.setText("Vb: —")

    # ---- Tab 3: Data Browser ----

    def _build_tab_browser(self) -> QtWidgets.QWidget:
        w  = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(w)
        hl.setSpacing(6)

        # Left: histogram plot + display controls
        left = QtWidgets.QVBoxLayout()

        self.browser_plot = HistogramPlotWidget()
        left.addWidget(self.browser_plot, 1)

        ctrl = QtWidgets.QGroupBox("Display Controls")
        cg   = QtWidgets.QGridLayout(ctrl)

        self.chk_normalize = QtWidgets.QCheckBox("Normalize")
        self.chk_log_y     = QtWidgets.QCheckBox("Log Y")
        self.chk_normalize.toggled.connect(self._refresh_browser_plot)
        self.chk_log_y.toggled.connect(self._refresh_browser_plot)

        self.spin_xmin3 = QtWidgets.QDoubleSpinBox()
        self.spin_xmin3.setRange(0, 1e12); self.spin_xmin3.setDecimals(1)
        self.spin_xmin3.setValue(DEFAULT_XMIN_PS)

        self.spin_xmax3 = QtWidgets.QDoubleSpinBox()
        self.spin_xmax3.setRange(0, 1e12); self.spin_xmax3.setDecimals(1)
        self.spin_xmax3.setValue(DEFAULT_XMAX_PS)

        self.spin_ymin3 = QtWidgets.QDoubleSpinBox()
        self.spin_ymin3.setRange(0, 1e12); self.spin_ymin3.setDecimals(0)
        self.spin_ymin3.setValue(0.0)

        self.spin_ymax3 = QtWidgets.QDoubleSpinBox()
        self.spin_ymax3.setRange(0, 1e12); self.spin_ymax3.setDecimals(0)
        self.spin_ymax3.setValue(1e6)

        btn_apply3 = QtWidgets.QPushButton("Apply range")
        btn_apply3.setFixedWidth(BTN_W)
        btn_apply3.clicked.connect(self._apply_browser_range)

        cg.addWidget(self.chk_normalize, 0, 0)
        cg.addWidget(self.chk_log_y,     0, 1)
        cg.addWidget(QtWidgets.QLabel("X min (ps):"), 1, 0)
        cg.addWidget(self.spin_xmin3, 1, 1)
        cg.addWidget(QtWidgets.QLabel("X max (ps):"), 1, 2)
        cg.addWidget(self.spin_xmax3, 1, 3)
        cg.addWidget(QtWidgets.QLabel("Y min:"),      2, 0)
        cg.addWidget(self.spin_ymin3, 2, 1)
        cg.addWidget(QtWidgets.QLabel("Y max:"),      2, 2)
        cg.addWidget(self.spin_ymax3, 2, 3)
        cg.addWidget(btn_apply3,      2, 4)

        left.addWidget(ctrl)
        hl.addLayout(left, 3)

        # Right: data table + actions
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Acquired Data</b>"))

        self.data_table = QtWidgets.QTableWidget(0, 5)
        self.data_table.setHorizontalHeaderLabels(["#", "Angle (°)", "Va (V)", "Vb (V)", "Show"])
        self.data_table.horizontalHeader().setStretchLastSection(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.data_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.data_table.itemChanged.connect(self._on_table_item_changed)

        btn_row = QtWidgets.QHBoxLayout()
        btn_save_csv  = QtWidgets.QPushButton("Save Selected CSV")
        btn_clear_data = QtWidgets.QPushButton("Clear All Data")
        btn_save_csv.clicked.connect(self._on_save_selected_csv)
        btn_clear_data.clicked.connect(self._on_clear_all_data)
        btn_row.addWidget(btn_save_csv)
        btn_row.addWidget(btn_clear_data)

        right.addWidget(self.data_table, 1)
        right.addLayout(btn_row)
        hl.addLayout(right, 2)

        return w

    # =======================================================================
    # Initialize / disconnect
    # =======================================================================

    def on_initialize(self):
        errors = []

        # --- PH300 ---
        try:
            dll = self.edit_dll.text().strip()
            self.ph = PicoHarp300(dll, device_index=0, verbose=False)
            self.ph.get_library_version()
            self.ph.open()
            self.ph.initialize_histogramming()
            self.ph.calibrate()
            self._apply_ph300_settings()
            hw = self.ph.get_hardware_info()
            _set_connected(self.lbl_ph_conn, True, f"{hw.get('model','')} s/n {hw.get('serial','')}")
        except Exception as e:
            errors.append(f"PH300: {e}")
            self.ph = None
            _set_connected(self.lbl_ph_conn, False)

        # --- Keithley gates ---
        try:
            self.rm = make_resource_manager(VISA_DLL)
        except Exception as e:
            errors.append(f"VISA RM: {e}")
            self.rm = None

        for tag, attr, edit in [("A", "dev_a", self.edit_visa_a),
                                  ("B", "dev_b", self.edit_visa_b)]:
            if self.rm is None:
                break
            try:
                dev = KeithleySMU(self.rm, edit.text().strip(),
                                  timeout_ms=20000, query_delay_s=0.0, verbose=False)
                dev.open()
                dev.set_compliance(float(self.spin_icomp_a.value() if tag == "A"
                                         else self.spin_icomp_b.value()) * 1e-9)
                dev.set_output(True)
                setattr(self, attr, dev)
            except Exception as e:
                errors.append(f"Gate {tag}: {e}")
                setattr(self, attr, None)

        # --- ND Wheel ---
        serial = self.edit_serial.text().strip()
        if serial:
            try:
                stage = RotationStage(serial)
                stage.open()
                self.rot_stage = stage
                angle = stage.get_position()
                _set_connected(self.lbl_wheel_conn, True, serial)
                self.lbl_wheel_angle.setText(f"{angle:.3f} °")
            except Exception as e:
                errors.append(f"ND Wheel: {e}")
                self.rot_stage = None
                _set_connected(self.lbl_wheel_conn, False)
        else:
            self.rot_stage = None
            _set_connected(self.lbl_wheel_conn, False, "no serial")

        # --- start polling ---
        self._poll_timer.start(int(self.spin_poll_ms.value()))

        msg = "Initialized. " + (("Errors: " + "; ".join(errors)) if errors else "All devices OK.")
        self._status_bar.showMessage(msg)
        self._update_status_labels()
        self._sync_buttons()

    def on_disconnect(self):
        self.on_abort()
        self._poll_timer.stop()

        for attr in ("ph", "dev_a", "dev_b", "rot_stage"):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
            setattr(self, attr, None)

        _set_connected(self.lbl_ph_conn,    False)
        _set_connected(self.lbl_wheel_conn, False)
        self._status_bar.showMessage("Disconnected.")
        self._sync_buttons()

    # =======================================================================
    # Quick controls — PH300
    # =======================================================================

    def _apply_ph300_settings(self):
        if self.ph is None:
            return
        try:
            self.ph.set_sync_div(int(self.p2_sync_div.currentText()))
            self.ph.set_sync_offset_ps(int(self.p2_sync_off.value()))
            self.ph.set_hist_offset_ps(int(self.p2_hist_off.value()))
            self.ph.set_input_cfd(0, int(self.p2_ch0_lvl.value()), int(self.p2_ch0_zc.value()))
            self.ph.set_input_cfd(1, int(self.p2_ch1_lvl.value()), int(self.p2_ch1_zc.value()))

            _, res_ps = self.ph.set_target_resolution_ps(float(self.spin_bin_ps.value()))
            self._ph_res_ps = res_ps
            time.sleep(0.1)
            self._ph_time_ps = np.asarray(self.ph.make_time_axis_ps(), dtype=np.float64)

            self.lbl_ph_bin.setText(f"{res_ps:.3f} ps")
            self.lbl_ph_acq.setText(f"{self.spin_tacq.value():.3g} s")
        except Exception as e:
            self._status_bar.showMessage(f"PH300 settings error: {e}")

    def _on_apply_ph300_quick(self):
        self._apply_ph300_settings()

    # =======================================================================
    # Quick controls — Gates
    # =======================================================================

    def _on_set_gate(self, tag: str):
        dev = self.dev_a if tag == "A" else self.dev_b
        if dev is None:
            return
        target_v = float(self.spin_v_a.value() if tag == "A" else self.spin_v_b.value())
        icomp_a  = float(self.spin_icomp_a.value() if tag == "A" else self.spin_icomp_b.value())
        try:
            dev.set_compliance(icomp_a * 1e-9)
        except Exception as e:
            self._status_bar.showMessage(f"Gate {tag} compliance error: {e}"); return

        ctrl = SweepController()
        thread = GateRampThread(dev, target_v, ctrl)
        thread.status.connect(self._status_bar.showMessage)
        thread.progress.connect(lambda v, i_a, _t=tag: self._on_sweep_vi(_t, v, i_a))
        thread.done.connect(lambda s, _t=tag: (
            self._status_bar.showMessage(f"Gate {_t} ramp: {s}"),
            self._poll_timer.start(int(self.spin_poll_ms.value())),
        ))
        thread.start()

        if tag == "A":
            self._gate_a_ramp = thread
        else:
            self._gate_b_ramp = thread

    def _on_set_gate_a(self):
        self._poll_timer.stop()   # avoid device race; progress signal covers live updates
        self._on_set_gate("A")

    def _on_set_gate_b(self):
        self._poll_timer.stop()
        self._on_set_gate("B")

    def _on_home_gate(self, tag: str):
        dev = self.dev_a if tag == "A" else self.dev_b
        if dev is None:
            return
        self._poll_timer.stop()
        ctrl   = SweepController()
        thread = GateHomeThread(dev, ctrl)
        thread.status.connect(self._status_bar.showMessage)
        thread.done.connect(lambda s, _t=tag: (
            self._status_bar.showMessage(f"Gate {_t} home: {s}"),
            self._poll_timer.start(int(self.spin_poll_ms.value())),
        ))
        thread.start()
        if tag == "A":
            self._gate_a_home = thread
        else:
            self._gate_b_home = thread

    def _on_home_gate_a(self): self._on_home_gate("A")
    def _on_home_gate_b(self): self._on_home_gate("B")

    # =======================================================================
    # Quick controls — ND Wheel
    # =======================================================================

    def _on_scan_serials(self):
        serials = list_kinesis_serials()
        if not serials:
            self._status_bar.showMessage("No Thorlabs Kinesis devices found.")
            return
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Thorlabs Device",
            f"Found {len(serials)} device(s) — select one:",
            serials, 0, False,
        )
        if ok and item:
            self.edit_serial.setText(item)
            self._status_bar.showMessage(f"Selected: {item}")

    def _on_wheel_move(self):
        if self.rot_stage is None:
            return
        target = float(self.spin_wheel_angle.value())
        ctrl   = MotionController()
        thread = WheelMoveThread(self.rot_stage, target, step_deg=5.0,
                                 accel=None, controller=ctrl)
        thread.progress.connect(lambda a: self.lbl_wheel_angle.setText(f"{a:.3f} °"))
        thread.status.connect(self._status_bar.showMessage)
        thread.done.connect(lambda s: self._status_bar.showMessage(f"Wheel move: {s}"))
        thread.start()
        self._wheel_move = thread

    def _on_wheel_home(self):
        if self.rot_stage is None:
            return
        ctrl   = MotionController()
        thread = WheelHomeThread(self.rot_stage, ctrl)
        thread.progress.connect(lambda a: self.lbl_wheel_angle.setText(f"{a:.3f} °"))
        thread.status.connect(self._status_bar.showMessage)
        thread.done.connect(lambda s: self._status_bar.showMessage(f"Wheel home: {s}"))
        thread.start()
        self._wheel_home = thread

    # =======================================================================
    # Action bar handlers
    # =======================================================================

    def on_take_histogram(self):
        if self.ph is None:
            self._status_bar.showMessage("PicoHarp not connected."); return
        if self._acq_running:
            return
        self._apply_ph300_settings()
        try:
            tacq_ms = int(self.spin_tacq.value() * 1000)
            self.ph.clear_hist_mem(block=0)
            self.ph.set_stop_overflow(enable=False)
            self.ph.start_meas(tacq_ms)
            self._acq_running   = True
            self._acq_t0        = time.time()
            self._acq_timeout_s = tacq_ms / 1000.0 + 10.0
            self._status_bar.showMessage(f"Acquiring histogram ({tacq_ms} ms)…")
            self.btn_stop_acq.setEnabled(True)
            # Guarantee the poll timer is running so _on_poll fires during acquisition
            self._poll_timer.start(int(self.spin_poll_ms.value()))
        except Exception as e:
            self._status_bar.showMessage(f"Acq error: {e}")

    def on_stop_acquisition(self):
        if not self._acq_running:
            return
        try:
            self.ph.stop_meas()
        except Exception:
            pass
        try:
            counts  = np.asarray(self.ph.read_histogram(block=0), dtype=np.uint32)
            time_ps = np.asarray(self.ph.make_time_axis_ps(),      dtype=np.float64)
        except Exception:
            counts = time_ps = None
        self.live_hist_plot.remove_live()
        self._acq_running = False
        self.btn_stop_acq.setEnabled(False)
        self._on_single_hist_done(time_ps, counts)
        self._status_bar.showMessage("Acquisition stopped (partial).")

    def _on_single_hist_done(self, time_ps, counts):
        if time_ps is None:
            self._status_bar.showMessage("Acquisition aborted or failed."); return
        self._push_live_hist(time.strftime("snap %H:%M:%S"), time_ps, counts)
        self._status_bar.showMessage("Histogram acquired.")

    def on_start_sweep(self):
        if self._sweep_worker and self._sweep_worker.isRunning():
            return

        angles  = self._parse_angles()
        va_list = list(self._cached_va_list)
        ratio   = float(self._spin_vb_ratio.value())
        vb_list = [ratio * v for v in va_list]

        if not angles:
            angles = [None]
        if not va_list:
            va_list = [None]
            vb_list = [None]

        if self.ph is None and self.dev_a is None and self.dev_b is None:
            self._status_bar.showMessage("No devices connected."); return

        self._apply_ph300_settings()
        self._set_sweep_buttons(sweeping=True)

        self._sweep_worker = SweepWorker(
            ph          = self.ph,
            dev_a       = self.dev_a,
            dev_b       = self.dev_b,
            rot_stage   = self.rot_stage,
            angles      = angles,
            va_list     = va_list,
            vb_list     = vb_list,
            tacq_ms     = int(self.spin_tacq.value() * 1000),
            icomp_a     = float(self.spin_icomp_a.value()) * 1e-9,
            icomp_b     = float(self.spin_icomp_b.value()) * 1e-9,
            gate_settle_s  = float(self.p2_gate_settle.value()),
            wheel_settle_s = float(self.p2_wheel_settle.value()),
        )
        self._sweep_worker.progress.connect(self._on_sweep_progress)
        self._sweep_worker.vi_update.connect(self._on_sweep_vi)
        self._sweep_worker.point_done.connect(self._on_sweep_point_done)
        self._sweep_worker.done.connect(self._on_sweep_done)
        self._sweep_worker.start()

    def on_home_both_gates(self):
        if self.dev_a is None and self.dev_b is None:
            return
        self._poll_timer.stop()
        self._home_worker = HomeGatesWorker(self.dev_a, self.dev_b)
        self._home_worker.status.connect(self._status_bar.showMessage)
        self._home_worker.done.connect(self._on_home_all_done)
        self._home_worker.start()

    def _on_home_all_done(self, status: str):
        self._poll_timer.start(int(self.spin_poll_ms.value()))
        self._status_bar.showMessage(f"Home both gates: {status}")

    def on_abort(self):
        if self._acq_running:
            try:
                self.ph.stop_meas()
            except Exception:
                pass
            self.live_hist_plot.remove_live()
            self._acq_running = False
            self.btn_stop_acq.setEnabled(False)
        if self._sweep_worker and self._sweep_worker.isRunning():
            self._sweep_worker.abort()
        if self._home_worker and self._home_worker.isRunning():
            self._home_worker.abort()
        if self._wheel_move and self._wheel_move.isRunning():
            self._wheel_move.wait(200)
        if self._wheel_home and self._wheel_home.isRunning():
            self._wheel_home.wait(200)
        self._status_bar.showMessage("Aborted.")

    # =======================================================================
    # Sweep signal handlers
    # =======================================================================

    def _on_sweep_progress(self, msg: str, done: int, total: int):
        self._status_bar.showMessage(f"[{done}/{total}] {msg}")

    def _on_sweep_vi(self, tag: str, v: float, i_a: float):
        if tag == "A":
            self.plot_vi_a.add_point(v, i_a)
            self.lbl_a_v.setText(f"{v:.5g} V")
            self.lbl_a_i.setText(f"{i_a * 1e9:.5g} nA")
        else:
            self.plot_vi_b.add_point(v, i_a)
            self.lbl_b_v.setText(f"{v:.5g} V")
            self.lbl_b_i.setText(f"{i_a * 1e9:.5g} nA")

    def _on_sweep_point_done(self, idx: int, angle: float, va: float, vb: float,
                              time_ps, counts):
        # Store
        pt = MeasPoint(idx=idx, angle=angle, va=va, vb=vb,
                       time_ps=time_ps, counts=counts)
        self._meas_data.append(pt)

        # Update live histogram (Tab 1)
        self._push_live_hist(self._point_label(pt), time_ps, counts)

        # Add row to table (Tab 3)
        self._add_table_row(pt)

        # Auto-save
        self._autosave_point(pt)

    def _on_sweep_done(self, status: str):
        self._poll_timer.start(int(self.spin_poll_ms.value()))
        self._set_sweep_buttons(sweeping=False)
        self._status_bar.showMessage(f"Sweep {status}. {len(self._meas_data)} points acquired.")

    # =======================================================================
    # Polling
    # =======================================================================

    def _on_poll(self):
        sweeping = self._sweep_worker is not None and self._sweep_worker.isRunning()

        # Gates (skip during sweep — sweep worker handles its own VI reads)
        if not sweeping:
            for tag, dev, v_lbl, i_lbl, plot in [
                ("A", self.dev_a, self.lbl_a_v, self.lbl_a_i, self.plot_vi_a),
                ("B", self.dev_b, self.lbl_b_v, self.lbl_b_i, self.plot_vi_b),
            ]:
                if dev is None:
                    continue
                try:
                    vi = dev.read_vi_with_timeout(POLL_TIMEOUT_MS)
                    v, i_a = float(vi.v), float(vi.i)
                    v_lbl.setText(f"{v:.5g} V")
                    i_lbl.setText(f"{i_a * 1e9:.5g} nA")
                    plot.add_point(v, i_a)
                except Exception:
                    pass

        # PH300 — poll rates from the main thread at all times (wrapper lock serialises
        # concurrent access with the sweep worker thread).
        if self.ph is not None:
            try:
                rs = self.ph.get_rates_and_warnings()
                self.lbl_ph_sync.setText(f"{rs.sync_rate_hz:,} Hz")
                self.lbl_ph_ch0.setText(f"{rs.ch0_rate_hz:,} Hz")
                self.lbl_ph_ch1.setText(f"{rs.ch1_rate_hz:,} Hz")
                if rs.warnings_bitfield != 0:
                    self.lbl_ph_warn.setText(rs.warnings_text or f"0x{rs.warnings_bitfield:08X}")
                else:
                    self.lbl_ph_warn.setText("")
            except Exception:
                pass

            if self._acq_running:
                # Live histogram — mirrors ph300/gui_v2 on_poll behaviour
                try:
                    hist = np.asarray(self.ph.read_histogram(block=0), dtype=np.uint32)
                    if self._ph_time_ps is not None:
                        self.live_hist_plot.update_live(self._ph_time_ps, hist)
                except Exception:
                    pass
                # Check completion
                try:
                    timed_out = (time.time() - self._acq_t0) > self._acq_timeout_s
                    if self.ph.ctc_done() or timed_out:
                        self.ph.stop_meas()
                        counts  = np.asarray(self.ph.read_histogram(block=0), dtype=np.uint32)
                        time_ps = np.asarray(self.ph.make_time_axis_ps(),      dtype=np.float64)
                        self.live_hist_plot.remove_live()
                        self._acq_running = False
                        self.btn_stop_acq.setEnabled(False)
                        self._on_single_hist_done(time_ps, counts)
                except Exception as e:
                    self.live_hist_plot.remove_live()
                    self._acq_running = False
                    self.btn_stop_acq.setEnabled(False)
                    self._status_bar.showMessage(f"Acq error: {e}")

        # ND Wheel angle
        if self.rot_stage is not None:
            try:
                angle = self.rot_stage.get_position()
                self.lbl_wheel_angle.setText(f"{angle:.3f} °")
            except Exception:
                pass

    def _push_live_hist(self, label: str, time_ps, counts):
        self._live_hist_history.append((label, time_ps, counts))
        self.live_hist_plot.clear_all()
        for lbl, t, c in self._live_hist_history:
            self.live_hist_plot.add_trace(lbl, t, c)
        self._apply_live_plot_range()

    def _clear_live_hist(self):
        self._live_hist_history.clear()
        self.live_hist_plot.clear_all()

    def _apply_live_plot_range(self):
        self.live_hist_plot.apply_range(
            xmin=float(self.live_xmin.value()),
            xmax=float(self.live_xmax.value()),
            ymin=float(self.live_ymin.value()),
            ymax=float(self.live_ymax.value()),
            logy=self.live_logy.isChecked(),
        )

    def _update_status_labels(self):
        _set_connected(self.lbl_ph_conn,    self.ph is not None)
        _set_connected(self.lbl_wheel_conn, self.rot_stage is not None)

    # =======================================================================
    # Tab 3: data table + browser plot
    # =======================================================================

    def _add_table_row(self, pt: MeasPoint):
        self.data_table.blockSignals(True)
        row = self.data_table.rowCount()
        self.data_table.insertRow(row)

        def _item(txt, editable=False):
            it = QtWidgets.QTableWidgetItem(txt)
            if not editable:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            return it

        self.data_table.setItem(row, 0, _item(str(pt.idx)))
        self.data_table.setItem(row, 1, _item(f"{pt.angle:.3f}"))
        self.data_table.setItem(row, 2, _item(f"{pt.va:.4f}"))
        self.data_table.setItem(row, 3, _item(f"{pt.vb:.4f}"))

        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk.setCheckState(QtCore.Qt.Unchecked)
        self.data_table.setItem(row, 4, chk)

        self.data_table.blockSignals(False)

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if item.column() != 4:
            return
        row   = item.row()
        show  = item.checkState() == QtCore.Qt.Checked
        if row >= len(self._meas_data):
            return
        pt    = self._meas_data[row]
        label = self._point_label(pt)
        if show:
            counts = self._maybe_normalize(pt.counts)
            self.browser_plot.add_trace(label, pt.time_ps, counts)
        else:
            self.browser_plot.remove_trace(label)

    def _point_label(self, pt: MeasPoint) -> str:
        return f"θ={pt.angle:.1f}° Va={pt.va:.2f}V Vb={pt.vb:.2f}V"

    def _maybe_normalize(self, counts: np.ndarray) -> np.ndarray:
        c = counts.astype(float)
        if self.chk_normalize.isChecked():
            mx = c.max()
            return c / mx if mx > 0 else c
        return c

    def _refresh_browser_plot(self):
        """Redraw all visible traces (after normalize or log toggle)."""
        self.browser_plot.clear_all()
        for row in range(self.data_table.rowCount()):
            chk = self.data_table.item(row, 4)
            if chk and chk.checkState() == QtCore.Qt.Checked and row < len(self._meas_data):
                pt = self._meas_data[row]
                counts = self._maybe_normalize(pt.counts)
                self.browser_plot.add_trace(self._point_label(pt), pt.time_ps, counts)
        self._apply_browser_range()

    def _apply_browser_range(self):
        self.browser_plot.apply_range(
            xmin=float(self.spin_xmin3.value()),
            xmax=float(self.spin_xmax3.value()),
            ymin=float(self.spin_ymin3.value()),
            ymax=float(self.spin_ymax3.value()),
            logy=self.chk_log_y.isChecked(),
        )

    def _on_save_selected_csv(self):
        rows = [r for r in range(self.data_table.rowCount())
                if (self.data_table.item(r, 4) and
                    self.data_table.item(r, 4).checkState() == QtCore.Qt.Checked)]
        if not rows:
            QtWidgets.QMessageBox.information(self, "Save CSV",
                                              "Check rows in the table to select traces.")
            return
        pts = [self._meas_data[r] for r in rows if r < len(self._meas_data)]
        if not pts:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "trpl_selection.csv", "CSV Files (*.csv)")
        if not path:
            return

        n = len(pts[0].time_ps)
        for pt in pts[1:]:
            if len(pt.time_ps) != n:
                QtWidgets.QMessageBox.warning(self, "Save CSV",
                    "Selected traces have different lengths — cannot combine."); return

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ps"] + [self._point_label(pt) for pt in pts])
            for i in range(n):
                w.writerow([pts[0].time_ps[i]] + [int(pt.counts[i]) for pt in pts])
        self._status_bar.showMessage(f"Saved: {path}")

    def _on_clear_all_data(self):
        if QtWidgets.QMessageBox.question(
                self, "Clear All", "Delete all acquired data?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        ) != QtWidgets.QMessageBox.Yes:
            return
        self._meas_data.clear()
        self.data_table.setRowCount(0)
        self.browser_plot.clear_all()
        self.live_hist_plot.clear_all()

    # =======================================================================
    # Auto-save
    # =======================================================================

    def _autosave_point(self, pt: MeasPoint):
        try:
            base = os.path.join(self.p2_out_dir.text(), self.p2_folder_name.text())
            os.makedirs(base, exist_ok=True)
            fname = f"pt{pt.idx:04d}_ang{pt.angle:.2f}_Va{pt.va:.4f}_Vb{pt.vb:.4f}.csv"
            fpath = os.path.join(base, fname)
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_ps", "counts",
                             f"# angle={pt.angle:.4f} va={pt.va:.6f} vb={pt.vb:.6f}"])
                for t, c in zip(pt.time_ps, pt.counts):
                    w.writerow([float(t), int(c)])
        except Exception as e:
            self._status_bar.showMessage(f"Auto-save error: {e}")

    # =======================================================================
    # Misc helpers
    # =======================================================================

    def _on_browse_out_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", self.p2_out_dir.text())
        if d:
            self.p2_out_dir.setText(d)

    def _set_sweep_buttons(self, sweeping: bool):
        self.btn_sweep.setEnabled(not sweeping)
        self.btn_init.setEnabled(not sweeping)
        self.btn_disconn.setEnabled(not sweeping)
        self.btn_abort.setEnabled(sweeping)

    def _sync_buttons(self):
        connected = (self.ph is not None or self.dev_a is not None
                     or self.dev_b is not None)
        self.btn_disconn.setEnabled(connected)
        self.btn_take_hist.setEnabled(self.ph is not None)
        self.btn_stop_acq.setEnabled(False)
        self.btn_sweep.setEnabled(True)
        self.btn_home_all.setEnabled(self.dev_a is not None or self.dev_b is not None)
        self.btn_abort.setEnabled(False)
        self.btn_set_a.setEnabled(self.dev_a is not None)
        self.btn_home_a.setEnabled(self.dev_a is not None)
        self.btn_set_b.setEnabled(self.dev_b is not None)
        self.btn_home_b.setEnabled(self.dev_b is not None)
        self.btn_wheel_move.setEnabled(self.rot_stage is not None)
        self.btn_wheel_home.setEnabled(self.rot_stage is not None)

    # =======================================================================
    # Close
    # =======================================================================

    def closeEvent(self, event):
        self.on_abort()
        self._poll_timer.stop()
        for attr in ("ph", "dev_a", "dev_b", "rot_stage"):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        event.accept()

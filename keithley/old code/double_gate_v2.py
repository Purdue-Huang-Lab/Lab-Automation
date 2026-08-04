#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keithley 2450 (LAN) + 2400 (GPIB) Controller
- Verified sweeps (auto/manual), simultaneous sweeps (Next/Stop)
- Live plots (rolling 100 points), compact UI
- nA throughout GUI; internal SCPI uses A
- Skips polling/plotting when Output is OFF
- Home (ramp both to 0 V) measures after each dwell step and updates plots
- 2400 front panel set to display measured values: DISP:VIEW MEAS
- NEW: "measure-first" step logic to keep both sources in sync with the slower one

Python 3.7–3.9, PyQt5 + matplotlib, PyVISA.
"""

import sys, os, csv, math, threading
from datetime import datetime
from collections import deque
from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

# ------------------- CONFIG -------------------
VISA_DLL = r"C:\Windows\System32\visa64.dll"  # leave "" to use env/default backend
DEFAULT_A_RESOURCE = "TCPIP0::169.254.153.76::5025::SOCKET"  # 2450 (LAN socket)
DEFAULT_B_RESOURCE = "TCPIP0::10.164.14.233::5025::SOCKET"                     # 2400 (GPIB)
POLL_MS_DEFAULT = 200
OC_TRIP_SAMPLES = 2
KEITHLEY_V_LIMIT = 210.0       # absolute max |V| enforced by GUI
VERIFY_TOL_FRACTION = 0.005    # 0.5% of setpoint
VERIFY_TOL_ABS      = 2e-3     # 0.002 V = 2 mV absolute floor
VERIFY_ATTEMPTS = 5
VERIFY_SLEEP_MS = 200
PLOT_MAX_POINTS = 100          # last N points on plots
BTN_W = 120                    # fixed button width
STATUS_W = 300                 # fixed status label width
READ_W = 110                   # fixed width for readout labels
DEFAULT_ICOMP_NA = 100.0         # default per-device compliance (nA), easy to change

# Home (ramp to 0 V) parameters
HOME_STEP_V    = 0.01          # volts per step toward 0
HOME_DWELL_MS  = 150           # ms between steps
HOME_TOL_V     = 2e-3          # stop when |V| <= this (2 mV)
# ---------------------------------------------

import pyvisa
from pyvisa import constants as pvconst  # noqa: F401

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def make_rm():
    if VISA_DLL and os.path.exists(VISA_DLL):
        return pyvisa.ResourceManager(VISA_DLL)
    return pyvisa.ResourceManager()


# ==============================
# Low-level instrument wrapper
# ==============================

class Kx4xx:
    """Minimal Keithley SMU helper (2450/2400 family as voltage source)."""
    def __init__(self, rm: "pyvisa.ResourceManager", resource: str, timeout_ms: int = 4000):
        self.rm = rm
        self.resource = resource
        self.inst = None  # type: Optional[pyvisa.resources.Resource]
        self.timeout_ms = timeout_ms
        self.model = ""
        self.is_2400 = False
        self._io_lock = threading.RLock()  # serialize all VISA I/O

    def open(self):
        self.close()
        self.inst = self.rm.open_resource(self.resource)
        try: self.inst.clear()
        except Exception: pass
        try: self.inst.read_termination = '\n'
        except Exception: pass
        try: self.inst.write_termination = '\n'
        except Exception: pass
        try: self.inst.timeout = self.timeout_ms
        except Exception: pass

        self._safe_write("*CLS")
        try: self.drain_errors()
        except Exception: pass

        try:
            self.model = self._safe_query("*IDN?").strip()
            self.is_2400 = ("2400" in self.model.upper())
        except Exception:
            self.model = ""
            self.is_2400 = False

        self.init_voltage_source()
        try: self.drain_errors()
        except Exception: pass

    def close(self):
        try:
            if self.inst is not None:
                self.inst.close()
        finally:
            self.inst = None

    def _safe_query(self, cmd: str) -> str:
        if self.inst is None:
            raise RuntimeError("Instrument not open")
        with self._io_lock:
            return self.inst.query(cmd)

    def _safe_write(self, cmd: str):
        if self.inst is None:
            raise RuntimeError("Instrument not open")
        with self._io_lock:
            self.inst.write(cmd)

    def drain_errors(self, max_reads: int = 8) -> list:
        errs = []
        for _ in range(max_reads):
            try:
                s = self._safe_query(":SYST:ERR?").strip()
            except Exception:
                break
            errs.append(s)
            if s.startswith("0"):
                break
        return errs

    def init_voltage_source(self):
        try: self._safe_write(":SOUR:FUNC VOLT")
        except Exception: self._safe_write("SOUR:FUNC VOLT")

        if self.is_2400:
            for cmd in (":SOUR:VOLT 0", "SOUR:VOLT 0"):
                try: self._safe_write(cmd); break
                except Exception: pass
            for cmd in (':SENS:FUNC "CURR"', 'SENS:FUNC "CURR"'):
                try: self._safe_write(cmd); break
                except Exception: pass
            for cmd in (":FORM:ELEM VOLT,CURR", "FORM:ELEM VOLT,CURR"):
                try: self._safe_write(cmd); break
                except Exception: pass
            try: self._safe_write(":TRIG:COUN 1")
            except Exception: pass
            # Show measured values on the 2400 front panel (reduces "one step ahead" confusion)
            try: self._safe_write(":DISP:VIEW MEAS")
            except Exception: pass
            for cmd in (":SENS:CURR:PROT 0.01", "SENS:CURR:PROT 0.01"):
                try: self._safe_write(cmd); break
                except Exception: pass
        else:
            try: self._safe_write(":SOUR:VOLT 0")
            except Exception: pass
            try: self._safe_write(":SOUR:VOLT:RANG:AUTO ON")
            except Exception: pass
            try: self._safe_write(":SENS:VOLT:RANG:AUTO ON")
            except Exception: pass
            try: self._safe_write(":SENS:CURR:RANG:AUTO ON")
            except Exception: pass
            try:
                self._safe_write(":SOUR:VOLT:ILIM 0.01")
            except Exception:
                try: self._safe_write(":SENS:CURR:PROT 0.01")
                except Exception: pass

    def set_output(self, on: bool):
        for cmd in (":OUTP {}".format("ON" if on else "OFF"),
                    "OUTP {}".format("ON" if on else "OFF")):
            try: self._safe_write(cmd); break
            except Exception: pass

    def set_compliance(self, limit_amps: float):
        if self.is_2400:
            for cmd in (":SENS:CURR:PROT {:.6g}".format(limit_amps),
                        "SENS:CURR:PROT {:.6g}".format(limit_amps)):
                try: self._safe_write(cmd); return
                except Exception: pass
        else:
            for cmd in (":SOUR:VOLT:ILIM {:.6g}".format(limit_amps),
                        ":SENS:CURR:PROT {:.6g}".format(limit_amps)):
                try: self._safe_write(cmd); return
                except Exception: pass

    def set_voltage(self, volts: float):
        for cmd in (":SOUR:VOLT {:.9g}".format(volts),
                    "SOUR:VOLT {:.9g}".format(volts)):
            try: self._safe_write(cmd); break
            except Exception: pass

    def meas_vi(self) -> Tuple[float, float]:
        """Returns (V, I) where I is in A."""
        if self.is_2400:
            ans = self._safe_query("READ?").strip()
            parts = [p.strip() for p in ans.split(",")]
            v = float(parts[0]) if parts and parts[0] else float('nan')
            i = float(parts[1]) if len(parts) > 1 else float('nan')
            return v, i
        else:
            v = float(self._safe_query(":MEAS:VOLT?").strip())
            i = float(self._safe_query(":MEAS:CURR?").strip())
            return v, i


# ==============================
# Sweep workers (QThread) — measure-first sync logic
# ==============================

def _verify_ok(v_set: float, v_read: float) -> bool:
    tol = max(VERIFY_TOL_ABS, VERIFY_TOL_FRACTION * abs(v_set))
    return (not math.isnan(v_read)) and (abs(v_read - v_set) <= tol)


class SweepWorker(QtCore.QThread):
    """
    Per-device sweep. For each step k:
      - In auto: dwell, then MEASURE; in manual: wait for click, then MEASURE.
      - If measured V is at current target V_k -> advance to V_{k+1}
        (unless k is last step).
      - If not at V_k -> set to V_k and verify; do NOT advance.
    This prevents the instrument from getting ahead due to UI/display quirks.
    """
    progress = QtCore.pyqtSignal(int, float, float)     # step_idx (1-based), Vread(V), Iread(A)
    done = QtCore.pyqtSignal(str)                       # "ok" | "aborted" | "error: ..." | "trip"
    tripped = QtCore.pyqtSignal(str)

    def __init__(self, dev: Kx4xx, v0: float, step: float, nsteps: int,
                 dwell_ms: int, icomp_limit_A: float, manual_mode: bool, parent=None):
        super().__init__(parent)
        self.dev = dev
        self.v0 = v0
        self.step = step
        self.nsteps = nsteps
        self.dwell_ms = dwell_ms
        self.icomp_limit_A = abs(icomp_limit_A)
        self.manual_mode = manual_mode
        self._abort = False
        self._step_event = threading.Event()

    def abort(self):
        self._abort = True
        self._step_event.set()

    def next_step(self):
        self._step_event.set()

    def _measure_vi(self) -> Tuple[float, float]:
        v, i = self.dev.meas_vi()
        if not math.isnan(i) and abs(i) > self.icomp_limit_A + 1e-15:
            msg = f"Compliance TRIP: |I|={abs(i)*1e9:.6g} nA > {self.icomp_limit_A*1e9:.6g} nA"
            self.tripped.emit(msg)
            self.done.emit("trip")
            raise RuntimeError("trip")
        return v, i

    def run(self):
        try:
            k = 0
            # Make sure we start at V0 (don’t assume)
            try:
                self.dev.set_voltage(self.v0)
            except Exception:
                pass

            while k < self.nsteps:
                if self._abort:
                    self.done.emit("aborted"); return

                # 1) Wait (auto dwell) or wait for user click (manual)
                if self.manual_mode:
                    self._step_event.clear()
                    while not self._abort and not self._step_event.is_set():
                        self.msleep(20)
                else:
                    self.msleep(max(0, int(self.dwell_ms)))
                if self._abort:
                    self.done.emit("aborted"); return

                # 2) Measure first, check if we’re at current step target V_k
                V_k = self.v0 + k * self.step
                if abs(V_k) > KEITHLEY_V_LIMIT + 1e-9:
                    self.done.emit("error: voltage limit exceeded"); return

                ok_now = False
                v_read = float('nan'); i_read = float('nan')
                for _ in range(VERIFY_ATTEMPTS):
                    v_read, i_read = self._measure_vi()
                    if _verify_ok(V_k, v_read):
                        ok_now = True
                        break
                    self.msleep(VERIFY_SLEEP_MS)

                if not ok_now:
                    # 3a) Not at current step -> set to V_k and verify; do NOT advance
                    try:
                        self.dev.set_voltage(V_k)
                    except Exception as e:
                        self.done.emit(f"error: set_voltage failed: {e}"); return

                    ok2 = False
                    for _ in range(VERIFY_ATTEMPTS):
                        v_read, i_read = self._measure_vi()
                        if _verify_ok(V_k, v_read):
                            ok2 = True
                            break
                        self.msleep(VERIFY_SLEEP_MS)

                    if not ok2:
                        self.done.emit(f"error: cannot reach current step (Vset {V_k:.6g}, Vread {v_read:.6g})")
                        return

                    # Report staying at current step; don’t increment k, repeat loop
                    self.progress.emit(k + 1, v_read, i_read)
                    continue

                # 3b) Already at current step -> advance to next (unless last)
                if k == self.nsteps - 1:
                    self.progress.emit(k + 1, v_read, i_read)
                    self.done.emit("ok"); return

                V_next = self.v0 + (k + 1) * self.step
                if abs(V_next) > KEITHLEY_V_LIMIT + 1e-9:
                    self.done.emit("error: voltage limit exceeded"); return

                try:
                    self.dev.set_voltage(V_next)
                except Exception as e:
                    self.done.emit(f"error: set_voltage failed: {e}"); return

                # Verify next setpoint reached (still measure-first style safety)
                ok_next = False
                v_read2 = float('nan'); i_read2 = float('nan')
                for _ in range(VERIFY_ATTEMPTS):
                    v_read2, i_read2 = self._measure_vi()
                    if _verify_ok(V_next, v_read2):
                        ok_next = True
                        break
                    self.msleep(VERIFY_SLEEP_MS)

                if not ok_next:
                    self.done.emit(f"error: setpoint not reached after advance (Vset {V_next:.6g}, Vread {v_read2:.6g})")
                    return

                k += 1
                self.progress.emit(k + 0, v_read2, i_read2)  # step index shown as 1..n

            self.done.emit("ok")
        except RuntimeError:
            # trip already signalled
            return
        except Exception as e:
            self.done.emit(f"error: {e}")


class DualSweepWorker(QtCore.QThread):
    """
    Dual sweep sync-to-slower, measure-first.
    Loop invariant for step k:
      - Ensure BOTH are at V_k (measure-first; if not, set lagging device to V_k and verify; do not advance)
      - When BOTH are confirmed at V_k, then (on dwell/next) advance BOTH to V_{k+1} together and verify.
    """
    progress = QtCore.pyqtSignal(int, float, float, float, float)  # step_idx (1-based), (VA,IA), (VB,IB)
    done = QtCore.pyqtSignal(str)
    tripped = QtCore.pyqtSignal(str)

    def __init__(self, devA: Kx4xx, v0A: float, stepA: float, nsteps: int,
                 dwell_ms: int, icompA_A: float,
                 devB: Kx4xx, v0B: float, stepB: float, dwell_ms_B: int, icompB_A: float,
                 manual_mode: bool, parent=None):
        super().__init__(parent)
        self.devA = devA; self.v0A = v0A; self.stepA = stepA; self.nsteps = nsteps
        self.dwell_ms = dwell_ms; self.icompA_A = abs(icompA_A)
        self.devB = devB; self.v0B = v0B; self.stepB = stepB
        self.dwell_ms_B = dwell_ms_B; self.icompB_A = abs(icompB_A)
        self.manual_mode = manual_mode
        self._abort = False
        self._step_event = threading.Event()

    def abort(self):
        self._abort = True
        self._step_event.set()

    def next_step(self):
        self._step_event.set()

    def _measure_vi_A(self) -> Tuple[float, float]:
        v, i = self.devA.meas_vi()
        if not math.isnan(i) and abs(i) > self.icompA_A + 1e-15:
            msg = f"A TRIP: |I|={abs(i)*1e9:.6g} nA > {self.icompA_A*1e9:.6g} nA"
            self.tripped.emit(msg); self.done.emit("trip")
            raise RuntimeError("tripA")
        return v, i

    def _measure_vi_B(self) -> Tuple[float, float]:
        v, i = self.devB.meas_vi()
        if not math.isnan(i) and abs(i) > self.icompB_A + 1e-15:
            msg = f"B TRIP: |I|={abs(i)*1e9:.6g} nA > {self.icompB_A*1e9:.6g} nA"
            self.tripped.emit(msg); self.done.emit("trip")
            raise RuntimeError("tripB")
        return v, i

    def run(self):
        try:
            k = 0
            # Ensure initial setpoints are written
            try: self.devA.set_voltage(self.v0A)
            except Exception: pass
            try: self.devB.set_voltage(self.v0B)
            except Exception: pass

            while k < self.nsteps:
                if self._abort:
                    self.done.emit("aborted"); return

                # 1) Wait dwell or wait for click
                if self.manual_mode:
                    self._step_event.clear()
                    while not self._abort and not self._step_event.is_set():
                        self.msleep(20)
                else:
                    self.msleep(max(0, int(self.dwell_ms)))
                if self._abort:
                    self.done.emit("aborted"); return

                # Targets for current k and next
                VA_k = self.v0A + k * self.stepA
                VB_k = self.v0B + k * self.stepB
                if any(abs(v) > KEITHLEY_V_LIMIT + 1e-9 for v in (VA_k, VB_k)):
                    self.done.emit("error: voltage limit exceeded"); return

                # 2) Measure-first: are both already at V_k?
                goodA = False; goodB = False
                vA_r = float('nan'); iA_r = float('nan')
                vB_r = float('nan'); iB_r = float('nan')

                for _ in range(VERIFY_ATTEMPTS):
                    vA_r, iA_r = self._measure_vi_A()
                    vB_r, iB_r = self._measure_vi_B()
                    goodA = _verify_ok(VA_k, vA_r)
                    goodB = _verify_ok(VB_k, vB_r)
                    if goodA and goodB:
                        break
                    self.msleep(VERIFY_SLEEP_MS)

                if not (goodA and goodB):
                    # 3a) At least one is not at current step: set the laggers to V_k and verify; do NOT advance.
                    if not goodA:
                        try: self.devA.set_voltage(VA_k)
                        except Exception as e:
                            self.done.emit(f"error: set A failed: {e}"); return
                    if not goodB:
                        try: self.devB.set_voltage(VB_k)
                        except Exception as e:
                            self.done.emit(f"error: set B failed: {e}"); return

                    okA = goodA; okB = goodB
                    for _ in range(VERIFY_ATTEMPTS):
                        vA_r, iA_r = self._measure_vi_A()
                        vB_r, iB_r = self._measure_vi_B()
                        okA = okA or _verify_ok(VA_k, vA_r)
                        okB = okB or _verify_ok(VB_k, vB_r)
                        if okA and okB:
                            break
                        self.msleep(VERIFY_SLEEP_MS)

                    if not (okA and okB):
                        self.done.emit("error: cannot reach current step for both devices")
                        return

                    # Report staying at current k (1-based), do not increment; loop again
                    self.progress.emit(k + 1, vA_r, iA_r, vB_r, iB_r)
                    continue

                # 3b) Both are at V_k: advance together to V_{k+1} (unless last step)
                if k == self.nsteps - 1:
                    # final report at current step
                    self.progress.emit(k + 1, vA_r, iA_r, vB_r, iB_r)
                    self.done.emit("ok"); return

                VA_next = self.v0A + (k + 1) * self.stepA
                VB_next = self.v0B + (k + 1) * self.stepB
                if any(abs(v) > KEITHLEY_V_LIMIT + 1e-9 for v in (VA_next, VB_next)):
                    self.done.emit("error: voltage limit exceeded"); return

                # Issue both sets back-to-back for best sync
                try:
                    self.devA.set_voltage(VA_next)
                    self.devB.set_voltage(VB_next)
                except Exception as e:
                    self.done.emit(f"error: set_voltage failed: {e}"); return

                # Verify both reached new step
                okA2 = False; okB2 = False
                vA2 = float('nan'); iA2 = float('nan')
                vB2 = float('nan'); iB2 = float('nan')
                for _ in range(VERIFY_ATTEMPTS):
                    vA2, iA2 = self._measure_vi_A()
                    vB2, iB2 = self._measure_vi_B()
                    okA2 = _verify_ok(VA_next, vA2)
                    okB2 = _verify_ok(VB_next, vB2)
                    if okA2 and okB2:
                        break
                    self.msleep(VERIFY_SLEEP_MS)

                if not (okA2 and okB2):
                    self.done.emit("error: setpoint not reached after advance"); return

                k += 1
                self.progress.emit(k, vA2, iA2, vB2, iB2)

            self.done.emit("ok")
        except RuntimeError:
            return
        except Exception as e:
            self.done.emit(f"error: {e}")


# ============ Dual Home (ramp both to 0 V) with measurements ============
class DualHomeWorker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)  # status text for status bar
    progress = QtCore.pyqtSignal(float, float, float, float)  # vA, iA, vB, iB (I in A)
    done = QtCore.pyqtSignal(str)    # "ok"/"aborted"/"error:..."

    def __init__(self, devA: Optional[Kx4xx], actA: bool,
                 devB: Optional[Kx4xx], actB: bool,
                 step_v: float, dwell_ms: int, tol_v: float, parent=None):
        super().__init__(parent)
        self.devA = devA; self.actA = actA
        self.devB = devB; self.actB = actB
        self.step_v = abs(step_v)
        self.dwell_ms = max(0, int(dwell_ms))
        self.tol_v = abs(tol_v)
        self._abort = False

    def abort(self): self._abort = True

    def _toward_zero(self, v: float) -> float:
        if abs(v) <= self.tol_v:
            return 0.0
        dv = min(self.step_v, abs(v))
        return v - math.copysign(dv, v)

    def _read_or_nan(self, dev: Optional[Kx4xx]) -> Tuple[float, float]:
        if not dev: return float('nan'), float('nan')
        try: return dev.meas_vi()
        except Exception: return float('nan'), float('nan')

    def run(self):
        try:
            while not self._abort:
                need_more = False

                vA_read, _ = self._read_or_nan(self.devA) if self.actA else (float('nan'), float('nan'))
                vB_read, _ = self._read_or_nan(self.devB) if self.actB else (float('nan'), float('nan'))

                if self.devA and self.actA and not math.isnan(vA_read) and abs(vA_read) > self.tol_v:
                    need_more = True
                    try: self.devA.set_voltage(self._toward_zero(vA_read))
                    except Exception as e:
                        self.done.emit(f"error: set A failed: {e}"); return

                if self.devB and self.actB and not math.isnan(vB_read) and abs(vB_read) > self.tol_v:
                    need_more = True
                    try: self.devB.set_voltage(self._toward_zero(vB_read))
                    except Exception as e:
                        self.done.emit(f"error: set B failed: {e}"); return

                self.status.emit("Homing…")
                self.msleep(self.dwell_ms)

                vA, iA = self._read_or_nan(self.devA if self.actA else None)
                vB, iB = self._read_or_nan(self.devB if self.actB else None)
                self.progress.emit(vA, iA, vB, iB)

                a_ok = (not self.actA) or (not math.isnan(vA) and abs(vA) <= self.tol_v)
                b_ok = (not self.actB) or (not math.isnan(vB) and abs(vB) <= self.tol_v)
                if a_ok and b_ok:
                    self.done.emit("ok"); return

                if not need_more:
                    # Neither needed a step, but outside tol (rare due to NaNs); keep looping
                    continue

            self.done.emit("aborted")
        except Exception as e:
            self.done.emit(f"error: {e}")


# =========================
# Plot widget (rolling, I in nA)
# =========================

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(5, 2.6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        self.axV = self.fig.add_subplot(111)
        self.axI = self.axV.twinx()
        self.axV.set_xlabel("Samples")
        self.axV.set_ylabel("V (V)")
        self.axI.set_ylabel("I (nA)")
        self.lineV, = self.axV.plot([], [], lw=1.5)
        self.lineI, = self.axI.plot([], [], lw=1.2, linestyle='--')

        self.vs = deque(maxlen=PLOT_MAX_POINTS)
        self.is_nA = deque(maxlen=PLOT_MAX_POINTS)

        self.fig.tight_layout()

    def add_point(self, v_volts: float, i_amps: float):
        self.vs.append(v_volts)
        self.is_nA.append(i_amps * 1e9)
        self._redraw()

    def clear(self):
        self.vs.clear()
        self.is_nA.clear()
        self._redraw()

    def _redraw(self):
        n = len(self.vs)
        self.lineV.set_data(range(n), list(self.vs))
        self.lineI.set_data(range(n), list(self.is_nA))
        self.axV.relim(); self.axV.autoscale_view()
        self.axI.relim(); self.axI.autoscale_view()
        self.axV.set_xlim(0, max(1, n - 1))
        self.canvas.draw_idle()


# =========================
# Device panel & MainWindow
# =========================

class DevicePanel(QtWidgets.QGroupBox):
    def __init__(self, title: str, rm: "pyvisa.ResourceManager", default_resource: str, main_ref_callable):
        super().__init__(title)
        self.rm = rm
        self.dev: Optional[Kx4xx] = None
        self.consecutive_oc = 0
        self.sweep_thread: Optional[SweepWorker] = None
        self.sweeping = False
        self._get_main = main_ref_callable
        self._build_ui(default_resource)

    def _build_ui(self, default_resource: str):
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self.resourceEdit = QtWidgets.QLineEdit(default_resource)
        self.resourceEdit.setMinimumWidth(320)
        self.connectBtn = QtWidgets.QPushButton("Connect"); self.connectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect"); self.disconnectBtn.setFixedWidth(BTN_W)
        self.disconnectBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("VISA Resource:"), 0, 0)
        layout.addWidget(self.resourceEdit, 0, 1, 1, 4)
        layout.addWidget(self.connectBtn, 0, 5)
        layout.addWidget(self.disconnectBtn, 0, 6)

        self.vsetSpin = QtWidgets.QDoubleSpinBox()
        self.vsetSpin.setDecimals(6); self.vsetSpin.setRange(-200.0, 200.0)
        self.vsetSpin.setSingleStep(0.01); self.vsetSpin.setValue(0.0); self.vsetSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.icompSpin_nA = QtWidgets.QDoubleSpinBox()
        self.icompSpin_nA.setDecimals(3)
        sheb = 1e9
        self.icompSpin_nA.setRange(0.0, sheb)
        self.icompSpin_nA.setSingleStep(0.1)
        self.icompSpin_nA.setValue(DEFAULT_ICOMP_NA)

        self.outputChk = QtWidgets.QCheckBox("Output ON")
        self.outputChk.setEnabled(False)

        self.applyBtn = QtWidgets.QPushButton("Apply V & Comp"); self.applyBtn.setFixedWidth(BTN_W); self.applyBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("V set (V):"), 1, 0)
        layout.addWidget(self.vsetSpin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("I comp (nA):"), 1, 2)
        layout.addWidget(self.icompSpin_nA, 1, 3)
        layout.addWidget(self.outputChk, 1, 4)
        layout.addWidget(self.applyBtn, 1, 5, 1, 2)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        self.vreadVal = QtWidgets.QLabel("—"); self.vreadVal.setFont(mono)
        self.vreadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.vreadVal.setFixedWidth(READ_W)

        self.ireadVal = QtWidgets.QLabel("—"); self.ireadVal.setFont(mono)
        self.ireadVal.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.ireadVal.setFixedWidth(READ_W)

        self.statusLbl = QtWidgets.QLabel("Disconnected.")
        self.statusLbl.setStyleSheet("color:#666;")
        self.statusLbl.setFixedWidth(STATUS_W)
        self.statusLbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        layout.addWidget(QtWidgets.QLabel("V-read (V):"), 2, 0)
        layout.addWidget(self.vreadVal, 2, 1)
        layout.addWidget(QtWidgets.QLabel("I-read (nA):"), 2, 2)
        layout.addWidget(self.ireadVal, 2, 3)
        layout.addWidget(self.statusLbl, 2, 4, 1, 3)

        # Sweep controls
        sweepLine = 3
        self.v0Spin = QtWidgets.QDoubleSpinBox(); self.v0Spin.setDecimals(6); self.v0Spin.setRange(-210.0, 210.0); self.v0Spin.setValue(0.0)
        self.v0Spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.stepSpin = QtWidgets.QDoubleSpinBox(); self.stepSpin.setDecimals(6); self.stepSpin.setRange(-50.0, 50.0); self.stepSpin.setValue(0.01)
        self.stepSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.nSpin = QtWidgets.QSpinBox(); self.nSpin.setRange(1, 100000); self.nSpin.setValue(10)
        self.vEndLbl = QtWidgets.QLabel("End: 0.000000 V"); self.vEndLbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.useDwellChk = QtWidgets.QCheckBox("Use dwell time")  # default OFF (manual)
        self.dwellSpin = QtWidgets.QSpinBox(); self.dwellSpin.setRange(10, 5000); self.dwellSpin.setValue(1000)
        self.dwellSpin.setEnabled(False)
        self.nextStepBtn = QtWidgets.QPushButton("Next step"); self.nextStepBtn.setFixedWidth(BTN_W)
        self.nextStepBtn.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel("Sweep V0 (V):"), sweepLine, 0)
        layout.addWidget(self.v0Spin, sweepLine, 1)
        layout.addWidget(QtWidgets.QLabel("Step (V):"), sweepLine, 2)
        layout.addWidget(self.stepSpin, sweepLine, 3)
        layout.addWidget(QtWidgets.QLabel("Steps:"), sweepLine, 4)
        layout.addWidget(self.nSpin, sweepLine, 5)
        layout.addWidget(self.vEndLbl, sweepLine, 6)

        sweepLine2 = sweepLine + 1
        self.sweepBtn = QtWidgets.QPushButton("Start Sweep"); self.sweepBtn.setFixedWidth(BTN_W)
        self.stopSweepBtn = QtWidgets.QPushButton("Stop"); self.stopSweepBtn.setFixedWidth(BTN_W)
        self.sweepBtn.setEnabled(False); self.stopSweepBtn.setEnabled(False)

        layout.addWidget(self.useDwellChk, sweepLine2, 0)
        layout.addWidget(QtWidgets.QLabel("Dwell (ms):"), sweepLine2, 1)
        layout.addWidget(self.dwellSpin, sweepLine2, 2)
        layout.addWidget(self.sweepBtn, sweepLine2, 5)
        layout.addWidget(self.stopSweepBtn, sweepLine2, 6)

        self.bannerLbl = QtWidgets.QLabel("")
        self.bannerLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.bannerLbl.setStyleSheet("color: red; font-weight: 700; font-size: 14pt;")
        layout.addWidget(self.bannerLbl, sweepLine2, 3, 1, 2)

        plotLine = sweepLine2 + 1
        self.plot = PlotWidget(self)
        self.clearPlotBtn = QtWidgets.QPushButton("Clear Plot"); self.clearPlotBtn.setFixedWidth(BTN_W)
        layout.addWidget(self.plot, plotLine, 0, 1, 6)
        layout.addWidget(self.clearPlotBtn, plotLine, 6)

        layout.addWidget(self.nextStepBtn, plotLine + 1, 6)

        layout.setColumnStretch(1, 1); layout.setColumnStretch(3, 1); layout.setColumnStretch(6, 1)

        # Signals
        self.connectBtn.clicked.connect(self.on_connect)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.applyBtn.clicked.connect(self.on_apply)
        self.outputChk.toggled.connect(self.on_output_toggled)

        self.v0Spin.valueChanged.connect(self._update_vend)
        self.stepSpin.valueChanged.connect(self._update_vend)
        self.nSpin.valueChanged.connect(self._update_vend)

        self.useDwellChk.toggled.connect(self._on_use_dwell_toggled)

        self.sweepBtn.clicked.connect(self.on_start_sweep)
        self.stopSweepBtn.clicked.connect(self.on_stop_sweep)
        self.nextStepBtn.clicked.connect(self.on_next_step)
        self.clearPlotBtn.clicked.connect(self.plot.clear)

        self._update_vend()

    def _vend_value(self) -> float:
        n = int(self.nSpin.value())
        return float(self.v0Spin.value() + (n - 1) * self.stepSpin.value())

    def _update_vend(self):
        v_end = self._vend_value()
        self.vEndLbl.setText("End: {:.6f} V".format(v_end))
        ok = (abs(float(self.v0Spin.value())) <= KEITHLEY_V_LIMIT + 1e-12) and (abs(v_end) <= KEITHLEY_V_LIMIT + 1e-12)
        self.vEndLbl.setStyleSheet("" if ok else "color: red; font-weight: 600;")
        self.sweepBtn.setEnabled(self.dev is not None and ok)

    def _on_use_dwell_toggled(self, checked: bool):
        self.dwellSpin.setEnabled(checked)

    def on_connect(self):
        res = self.resourceEdit.text().strip()
        if not res:
            QtWidgets.QMessageBox.critical(self, "Error", "Enter a VISA resource string.")
            return
        try:
            dev = Kx4xx(self.rm, res); dev.open()

            if "2400" in (dev.model or "").upper():
                try:
                    dev._safe_write(":SYST:REM")
                    dev._safe_write(":FORM:ELEM VOLT,CURR")
                    dev._safe_write(":TRIG:COUN 1")
                    dev._safe_write(":DISP:VIEW MEAS")  # make panel show MEAS
                    dev.drain_errors()
                except Exception:
                    pass

            self.dev = dev
            self.connectBtn.setEnabled(False)
            self.disconnectBtn.setEnabled(True)
            self.applyBtn.setEnabled(True)
            self.outputChk.setEnabled(True)
            self.sweepBtn.setEnabled(True)
            self.stopSweepBtn.setEnabled(False)

            try:
                dev.set_compliance(float(self.icompSpin_nA.value()) * 1e-9)
                dev.set_voltage(float(self.vsetSpin.value()))
            except Exception:
                pass
            try: dev.set_output(True)
            except Exception: pass

            self.outputChk.blockSignals(True); self.outputChk.setChecked(True); self.outputChk.blockSignals(False)

            self.bannerLbl.setText("")
            errs = []
            try: errs = dev.drain_errors()
            except Exception: pass
            if errs and not errs[-1].startswith("0"):
                self.statusLbl.setText(("Connected (warn): " + errs[-1])[:40])
            else:
                self.statusLbl.setText(("Connected: " + (dev.model or "(unknown)"))[:40])
        except Exception as e:
            self.statusLbl.setText("Conn failed")
            QtWidgets.QMessageBox.critical(self, "Connection failed", str(e))

        self._update_vend()

    def on_disconnect(self):
        try:
            if self.dev:
                try: self.dev._safe_write(":ABOR")
                except Exception: pass
                self.dev.close()
        except Exception:
            pass
        self.dev = None
        self.connectBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
        self.applyBtn.setEnabled(False)
        self.outputChk.setEnabled(False)
        self.sweepBtn.setEnabled(False)
        self.stopSweepBtn.setEnabled(False)
        self.nextStepBtn.setEnabled(False)
        self.vreadVal.setText("—"); self.ireadVal.setText("—")
        self.statusLbl.setText("Disconnected.")

    def on_apply(self):
        if not self.dev: return
        vset = float(self.vsetSpin.value())
        icomp_nA = float(self.icompSpin_nA.value())
        try:
            self.dev.set_compliance(icomp_nA * 1e-9)
            self.dev.set_voltage(vset)
            errs = []
            try: errs = self.dev.drain_errors()
            except Exception: pass
            if errs and not errs[-1].startswith("0"):
                self.statusLbl.setText(("Applied (warn): " + errs[-1])[:40])
            else:
                self.statusLbl.setText("Applied V={:.6g}".format(vset)[:40])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SCPI error", "Failed to apply settings:\n{}".format(e))

    def on_output_toggled(self, checked: bool):
        if not self.dev: return
        try:
            self.dev.set_output(checked)
            try: self.dev.drain_errors()
            except Exception: pass
            self.statusLbl.setText("Output " + ("ON" if checked else "OFF"))
            if not checked:
                self.vreadVal.setText("—")
                self.ireadVal.setText("—")
        except Exception as e:
            self.outputChk.blockSignals(True)
            self.outputChk.setChecked(not checked)
            self.outputChk.blockSignals(False)
            QtWidgets.QMessageBox.critical(self, "SCPI error", "Failed to toggle output:\n{}".format(e))

    def poll_once(self, oc_limit_A: Optional[float], oc_trip_samples: int = OC_TRIP_SAMPLES) -> Tuple[Optional[float], Optional[float], bool]:
        if not self.dev:
            return (None, None, False)
        if not self.outputChk.isChecked():
            self.vreadVal.setText("—")
            self.ireadVal.setText("—")
            return (None, None, False)
        if self.sweeping:
            return (None, None, False)

        tripped = False
        try:
            v, iA = self.dev.meas_vi()
            self.vreadVal.setText("{:.6g}".format(v))
            self.ireadVal.setText("{:.6g}".format(iA * 1e9))
            self.plot.add_point(v, iA)

            if oc_limit_A is not None:
                if abs(iA) > oc_limit_A:
                    self.consecutive_oc += 1
                    if self.consecutive_oc >= oc_trip_samples:
                        try:
                            self.dev.set_output(False)
                        finally:
                            self.outputChk.blockSignals(True)
                            self.outputChk.setChecked(False)
                            self.outputChk.blockSignals(False)
                            self.ireadVal.setStyleSheet("color:red;")
                            self.bannerLbl.setText("OC TRIP: |I|={:.6g} nA > {:.6g} nA".format(abs(iA)*1e9, oc_limit_A*1e9))
                            self.statusLbl.setText("OC trip.")
                            tripped = True
                else:
                    self.consecutive_oc = 0
                    self.ireadVal.setStyleSheet("")
            else:
                self.ireadVal.setStyleSheet("")
            return (v, iA, tripped)
        except Exception:
            self.statusLbl.setText("Poll error")
            return (None, None, False)

    # ---- sweep controls ----
    def _pause_global_polling(self, pause: bool):
        main = self._get_main()
        if pause:
            main.pause_polling()
        else:
            main.resume_polling_if_idle()

    def on_start_sweep(self):
        if not self.dev:
            QtWidgets.QMessageBox.warning(self, "Not connected", "Connect the device first.")
            return
        v0 = float(self.v0Spin.value())
        step = float(self.stepSpin.value())
        n = int(self.nSpin.value())
        vend = self._vend_value()
        if abs(v0) > KEITHLEY_V_LIMIT or abs(vend) > KEITHLEY_V_LIMIT:
            QtWidgets.QMessageBox.critical(self, "Voltage limit", "Start/End exceed ±{:.0f} V".format(KEITHLEY_V_LIMIT))
            return
        icomp_A = float(self.icompSpin_nA.value()) * 1e-9
        dwell_ms = int(self.dwellSpin.value())
        manual_mode = (not self.useDwellChk.isChecked())

        self.bannerLbl.setText("")
        self.sweeping = True
        self._pause_global_polling(True)

        self.sweepBtn.setEnabled(False)
        self.stopSweepBtn.setEnabled(True)
        self.applyBtn.setEnabled(False)
        self.outputChk.setEnabled(False)
        self.connectBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(False)
        self.nextStepBtn.setEnabled(manual_mode)

        self.sweep_thread = SweepWorker(self.dev, v0, step, n, dwell_ms, icomp_A, manual_mode)
        self.sweep_thread.progress.connect(self._on_sweep_progress)
        self.sweep_thread.tripped.connect(self._on_sweep_tripped)
        self.sweep_thread.done.connect(self._on_sweep_done)
        self.sweep_thread.start()

    def on_stop_sweep(self):
        if self.sweep_thread and self.sweep_thread.isRunning():
            self.sweep_thread.abort()
        self.stopSweepBtn.setEnabled(False)
        self.nextStepBtn.setEnabled(False)

    def on_next_step(self):
        if self.sweep_thread and self.sweep_thread.isRunning():
            self.sweep_thread.next_step()

    def _on_sweep_progress(self, idx: int, v: float, iA: float):
        self.vreadVal.setText("{:.6g}".format(v))
        self.ireadVal.setText("{:.6g}".format(iA * 1e9))
        self.statusLbl.setText("Sweep step {}".format(idx))
        self.plot.add_point(v, iA)

    def _on_sweep_tripped(self, msg: str):
        self.bannerLbl.setText(msg)
        self.statusLbl.setText("SWEEP TRIPPED")

    def _on_sweep_done(self, status: str):
        self.sweeping = False
        self.stopSweepBtn.setEnabled(False)
        self.nextStepBtn.setEnabled(False)
        self.applyBtn.setEnabled(True)
        self.outputChk.setEnabled(True)
        self.connectBtn.setEnabled(self.dev is None)
        self.disconnectBtn.setEnabled(self.dev is not None)
        self.sweepBtn.setEnabled(self.dev is not None and abs(self._vend_value()) <= KEITHLEY_V_LIMIT)
        if status == "ok":
            self.statusLbl.setText("Sweep OK")
        elif status == "aborted":
            self.statusLbl.setText("Sweep aborted")
        elif status.startswith("error"):
            self.bannerLbl.setText(status); self.statusLbl.setText("Sweep error")
        self._pause_global_polling(False)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keithley 2450 + 2400 Controller (Sync-to-Slower Sweeps, nA)")
        self.resize(1300, 740)

        try:
            self.rm = make_rm()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "VISA Error", "Could not create ResourceManager:\n{}".format(e))
            sys.exit(1)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8); vbox.setSpacing(8)

        top = QtWidgets.QHBoxLayout(); vbox.addLayout(top); top.setSpacing(8)

        self.ocSpin = QtWidgets.QDoubleSpinBox()
        self.ocSpin.setDecimals(6); self.ocSpin.setRange(0.0, 10.0)
        self.ocSpin.setSingleStep(0.001); self.ocSpin.setValue(0.01)

        self.pollSpin = QtWidgets.QSpinBox()
        self.pollSpin.setRange(100, 2000); self.pollSpin.setSingleStep(50)
        self.pollSpin.setValue(POLL_MS_DEFAULT)

        self.logChk = QtWidgets.QCheckBox("Log CSV")
        self.logBtn = QtWidgets.QPushButton("Choose file…"); self.logBtn.setFixedWidth(BTN_W)
        self.logLbl = QtWidgets.QLabel("Logging: OFF"); self.logLbl.setFixedWidth(220)

        self.simulBtn = QtWidgets.QPushButton("Sweep Simultaneously"); self.simulBtn.setFixedWidth(BTN_W)
        self.simulBtn.setEnabled(False)
        self.nextSimulBtn = QtWidgets.QPushButton("Next step (Simul)"); self.nextSimulBtn.setFixedWidth(BTN_W)
        self.nextSimulBtn.setEnabled(False)
        self.stopSimulBtn = QtWidgets.QPushButton("Stop Simul"); self.stopSimulBtn.setFixedWidth(BTN_W)
        self.stopSimulBtn.setEnabled(False)

        self.homeBtn = QtWidgets.QPushButton("Home (ramp to 0 V)"); self.homeBtn.setFixedWidth(150)
        self.panicBtn = QtWidgets.QPushButton("All OFF (Panic)"); self.panicBtn.setFixedWidth(BTN_W)

        top.addWidget(QtWidgets.QLabel("Software OC (A):")); top.addWidget(self.ocSpin)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Poll (ms):")); top.addWidget(self.pollSpin)
        top.addStretch()
        top.addWidget(self.logChk); top.addWidget(self.logBtn); top.addWidget(self.logLbl)
        top.addSpacing(12)
        top.addWidget(self.simulBtn); top.addWidget(self.nextSimulBtn); top.addWidget(self.stopSimulBtn)
        top.addSpacing(12)
        top.addWidget(self.homeBtn)
        top.addSpacing(12)
        top.addWidget(self.panicBtn)

        panels = QtWidgets.QHBoxLayout(); vbox.addLayout(panels); panels.setSpacing(10)
        self.panelA = DevicePanel("Source A (2450 @ LAN SOCKET)", self.rm, DEFAULT_A_RESOURCE, main_ref_callable=lambda: self)
        self.panelB = DevicePanel("Source B (2400 @ GPIB)", self.rm, DEFAULT_B_RESOURCE, main_ref_callable=lambda: self)
        panels.addWidget(self.panelA, 1); panels.addWidget(self.panelB, 1)

        self.statusBar().showMessage("Idle")

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.on_poll); self.timer.start(self.pollSpin.value())

        self.log_path = None; self.log_file = None; self.log_writer = None

        self.panicBtn.clicked.connect(self.on_panic)
        self.pollSpin.valueChanged.connect(self.on_poll_interval_changed)
        self.logChk.toggled.connect(self.on_toggle_logging)
        self.logBtn.clicked.connect(self.choose_log_file)
        self.simulBtn.clicked.connect(self.on_simul_sweep)
        self.nextSimulBtn.clicked.connect(self.on_next_simul_step)
        self.stopSimulBtn.clicked.connect(self.on_stop_simul_sweep)
        self.homeBtn.clicked.connect(self.on_home)

        for w in (self.panelA.nSpin, self.panelB.nSpin, self.panelA.dwellSpin, self.panelB.dwellSpin,
                  self.panelA.v0Spin, self.panelA.stepSpin, self.panelB.v0Spin, self.panelB.stepSpin,
                  self.panelA.useDwellChk, self.panelB.useDwellChk):
            if hasattr(w, "valueChanged"): w.valueChanged.connect(self._update_simul_enabled)
            else: w.toggled.connect(self._update_simul_enabled)
        self.panelA.connectBtn.clicked.connect(self._update_simul_enabled)
        self.panelB.connectBtn.clicked.connect(self._update_simul_enabled)
        self.panelA.disconnectBtn.clicked.connect(self._update_simul_enabled)
        self.panelB.disconnectBtn.clicked.connect(self._update_simul_enabled)

        self.dual_thread: Optional[DualSweepWorker] = None
        self.home_thread: Optional[DualHomeWorker] = None
        self._sweeps_active = 0

    # polling pause/resume
    def pause_polling(self):
        self._sweeps_active += 1
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("Polling paused…")

    def resume_polling_if_idle(self):
        self._sweeps_active = max(0, self._sweeps_active - 1)
        if self._sweeps_active == 0 and not self.timer.isActive():
            self.timer.start(self.pollSpin.value())
            self.statusBar().showMessage("Polling resumed.")

    # logging
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
            self.logChk.setChecked(False); return
        try:
            self.log_file = open(self.log_path, "a", newline="")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(["timestamp", "panel", "Vset(V)", "Icomp(A)", "Vread(V)", "Iread(A)", "output"])
            self.statusBar().showMessage("Logging → {}".format(self.log_path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Log error", "Cannot open log file:\n{}".format(e))
            self.logChk.setChecked(False)

    def close_log(self):
        try:
            if self.log_file:
                self.log_file.flush(); self.log_file.close()
        except Exception: pass
        self.log_file = None; self.log_writer = None

    def on_toggle_logging(self, checked: bool):
        if checked:
            if not self.log_path:
                self.choose_log_file()
                if not self.log_path:
                    self.logChk.setChecked(False); return
            self.open_log()
        else:
            self.close_log()

    # panic
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
            except Exception: pass
        QtWidgets.QMessageBox.information(self, "Panic", "Outputs disabled on {} connected device(s).".format(n))

    # polling
    def on_poll_interval_changed(self, ms: int):
        if self.timer.isActive(): self.timer.setInterval(ms)
        self.statusBar().showMessage("Polling @ {} ms".format(ms))

    def on_poll(self):
        oc_limit_A = float(self.ocSpin.value()); ms = self.pollSpin.value()
        for tag, panel in (("A", self.panelA), ("B", self.panelB)):
            v, iA, tripped = panel.poll_once(oc_limit_A, OC_TRIP_SAMPLES)
            if (v is not None) and (iA is not None) and self.log_writer and panel.dev:
                vset = float(panel.vsetSpin.value()); icomp_A = float(panel.icompSpin_nA.value()) * 1e-9
                out = 1 if panel.outputChk.isChecked() else 0
                self.log_writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    tag, vset, icomp_A, v, iA, out
                ])
                try: self.log_file.flush()
                except Exception: pass
        self.statusBar().showMessage("Polling @ {} ms".format(ms))
        self._update_simul_enabled()

    # sim sweep
    def _update_simul_enabled(self):
        a_ok = (self.panelA.dev is not None); b_ok = (self.panelB.dev is not None)
        same_n = (int(self.panelA.nSpin.value()) == int(self.panelB.nSpin.value()))
        same_dwell = (int(self.panelA.dwellSpin.value()) == int(self.panelB.dwellSpin.value()))
        within = lambda p: (abs(float(p.v0Spin.value())) <= KEITHLEY_V_LIMIT + 1e-12) and (abs(float(p._vend_value())) <= KEITHLEY_V_LIMIT + 1e-12)
        enable_simul = a_ok and b_ok and same_n and same_dwell and within(self.panelA) and within(self.panelB) \
                       and not (self.panelA.sweeping or self.panelB.sweeping) and self.dual_thread is None and self.home_thread is None
        self.simulBtn.setEnabled(enable_simul)

    def on_simul_sweep(self):
        if not (self.panelA.dev and self.panelB.dev):
            QtWidgets.QMessageBox.warning(self, "Not connected", "Connect both devices first."); return
        nA = int(self.panelA.nSpin.value()); nB = int(self.panelB.nSpin.value())
        if nA != nB:
            QtWidgets.QMessageBox.warning(self, "Steps mismatch", "Set the same number of steps for both devices."); return
        dwellA = int(self.panelA.dwellSpin.value()); dwellB = int(self.panelB.dwellSpin.value())
        if dwellA != dwellB:
            QtWidgets.QMessageBox.warning(self, "Dwell mismatch", "Set the same dwell (ms) for both devices."); return
        v0A = float(self.panelA.v0Spin.value()); stepA = float(self.panelA.stepSpin.value())
        v0B = float(self.panelB.v0Spin.value()); stepB = float(self.panelB.stepSpin.value())
        vendA = v0A + (nA - 1) * stepA; vendB = v0B + (nB - 1) * stepB
        if any(abs(v) > KEITHLEY_V_LIMIT for v in (v0A, vendA, v0B, vendB)):
            QtWidgets.QMessageBox.critical(self, "Voltage limit", "Start/End exceed ±{:.0f} V".format(KEITHLEY_V_LIMIT)); return

        icompA_A = float(self.panelA.icompSpin_nA.value()) * 1e-9
        icompB_A = float(self.panelB.icompSpin_nA.value()) * 1e-9

        manual_mode = (not self.panelA.useDwellChk.isChecked()) and (not self.panelB.useDwellChk.isChecked())

        self.panelA.bannerLbl.setText(""); self.panelB.bannerLbl.setText("")
        self.panelA.sweeping = True; self.panelB.sweeping = True
        self.panelA.applyBtn.setEnabled(False); self.panelB.applyBtn.setEnabled(False)
        self.panelA.outputChk.setEnabled(False); self.panelB.outputChk.setEnabled(False)
        self.panelA.sweepBtn.setEnabled(False); self.panelB.sweepBtn.setEnabled(False)
        self.simulBtn.setEnabled(False)
        self.pause_polling()

        self.panelA.nextStepBtn.setEnabled(False); self.panelB.nextStepBtn.setEnabled(False)
        self.nextSimulBtn.setEnabled(manual_mode); self.stopSimulBtn.setEnabled(True)

        self.dual_thread = DualSweepWorker(self.panelA.dev, v0A, stepA, nA, dwellA, icompA_A,
                                           self.panelB.dev, v0B, stepB, dwellB, icompB_A,
                                           manual_mode=manual_mode)
        self.dual_thread.progress.connect(self._on_dual_progress)
        self.dual_thread.tripped.connect(self._on_dual_tripped)
        self.dual_thread.done.connect(self._on_dual_done)
        self.dual_thread.start()

    def on_next_simul_step(self):
        if self.dual_thread and self.dual_thread.isRunning():
            self.dual_thread.next_step()

    def on_stop_simul_sweep(self):
        if self.dual_thread and self.dual_thread.isRunning():
            self.dual_thread.abort()
        self.stopSimulBtn.setEnabled(False)
        self.nextSimulBtn.setEnabled(False)

    def _on_dual_progress(self, idx, vA, iA, vB, iB):
        self.panelA.vreadVal.setText("{:.6g}".format(vA))
        self.panelA.ireadVal.setText("{:.6g}".format(iA * 1e9))
        self.panelA.statusLbl.setText("A step {}".format(idx))
        self.panelA.plot.add_point(vA, iA)

        self.panelB.vreadVal.setText("{:.6g}".format(vB))
        self.panelB.ireadVal.setText("{:.6g}".format(iB * 1e9))
        self.panelB.statusLbl.setText("B step {}".format(idx))
        self.panelB.plot.add_point(vB, iB)

        self.statusBar().showMessage("Dual sweep… step {}".format(idx))

    def _on_dual_tripped(self, msg: str):
        self.panelA.bannerLbl.setText(msg); self.panelB.bannerLbl.setText(msg)
        self.statusBar().showMessage("Dual sweep TRIPPED: " + msg)

    def _on_dual_done(self, status: str):
        self.panelA.sweeping = False; self.panelB.sweeping = False
        for p in (self.panelA, self.panelB):
            vend_ok = (abs(p.v0Spin.value()) <= KEITHLEY_V_LIMIT + 1e-12) and (abs(p._vend_value()) <= KEITHLEY_V_LIMIT + 1e-12)
            p.sweepBtn.setEnabled(p.dev is not None and vend_ok)
            p.stopSweepBtn.setEnabled(False)
            p.applyBtn.setEnabled(True)
            p.outputChk.setEnabled(True)
            p.nextStepBtn.setEnabled(False)
        self.nextSimulBtn.setEnabled(False); self.stopSimulBtn.setEnabled(False)
        self.dual_thread = None
        self._update_simul_enabled()
        self.resume_polling_if_idle()

        if status == "ok": self.statusBar().showMessage("Dual sweep complete.")
        elif status == "aborted": self.statusBar().showMessage("Dual sweep aborted.")
        else: self.statusBar().showMessage("Dual sweep " + status)

    # Home (ramp both to 0 V)
    def on_home(self):
        if self.dual_thread and self.dual_thread.isRunning():
            QtWidgets.QMessageBox.warning(self, "Busy", "Stop the simultaneous sweep first."); return
        if (self.panelA.sweeping or self.panelB.sweeping):
            QtWidgets.QMessageBox.warning(self, "Busy", "Stop any ongoing per-device sweep first."); return

        actA = bool(self.panelA.dev and self.panelA.outputChk.isChecked())
        actB = bool(self.panelB.dev and self.panelB.outputChk.isChecked())

        # Devices that are connected but OFF → just set setpoint to 0
        if self.panelA.dev and not actA:
            try: self.panelA.dev.set_voltage(0.0)
            except Exception: pass
            self.panelA.vsetSpin.setValue(0.0)
        if self.panelB.dev and not actB:
            try: self.panelB.dev.set_voltage(0.0)
            except Exception: pass
            self.panelB.vsetSpin.setValue(0.0)

        if not (actA or actB):
            QtWidgets.QMessageBox.information(self, "Home", "Nothing to ramp (both outputs are OFF).")
            return

        self.pause_polling()
        self.homeBtn.setEnabled(False)
        self.simulBtn.setEnabled(False)
        self.nextSimulBtn.setEnabled(False)
        self.stopSimulBtn.setEnabled(False)

        self.home_thread = DualHomeWorker(self.panelA.dev if actA else None, actA,
                                          self.panelB.dev if actB else None, actB,
                                          HOME_STEP_V, HOME_DWELL_MS, HOME_TOL_V)
        self.home_thread.progress.connect(self._on_home_progress)
        self.home_thread.status.connect(self.statusBar().showMessage)
        self.home_thread.done.connect(self._on_home_done)
        self.home_thread.start()

    def _on_home_progress(self, vA, iA, vB, iB):
        if not math.isnan(vA):
            self.panelA.vreadVal.setText("{:.6g}".format(vA))
            self.panelA.ireadVal.setText("{:.6g}".format(iA * 1e9))
            self.panelA.plot.add_point(vA, iA)
        if not math.isnan(vB):
            self.panelB.vreadVal.setText("{:.6g}".format(vB))
            self.panelB.ireadVal.setText("{:.6g}".format(iB * 1e9))
            self.panelB.plot.add_point(vB, iB)

    def _on_home_done(self, status: str):
        self.home_thread = None
        if self.panelA.dev: self.panelA.vsetSpin.setValue(0.0)
        if self.panelB.dev: self.panelB.vsetSpin.setValue(0.0)
        self.statusBar().showMessage("Home: " + status)
        self.homeBtn.setEnabled(True)
        self._update_simul_enabled()
        self.resume_polling_if_idle()

    def closeEvent(self, event):  # type: (QtGui.QCloseEvent) -> None
        try:
            if self.panelA.sweep_thread and self.panelA.sweep_thread.isRunning(): self.panelA.sweep_thread.abort()
            if self.panelB.sweep_thread and self.panelB.sweep_thread.isRunning(): self.panelB.sweep_thread.abort()
            if self.dual_thread and self.dual_thread.isRunning(): self.dual_thread.abort()
            if self.home_thread and self.home_thread.isRunning(): self.home_thread.abort()
        except Exception: pass

        if self.timer.isActive(): self.timer.stop()
        try: self.log_file and self.log_file.close()
        except Exception: pass
        for panel in (self.panelA, self.panelB):
            try:
                if panel.dev:
                    try: panel.dev._safe_write(":ABOR")
                    except Exception: pass
                    panel.dev.close()
            except Exception: pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

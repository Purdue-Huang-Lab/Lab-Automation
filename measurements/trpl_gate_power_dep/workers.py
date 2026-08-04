import time
from typing import List, Optional

import numpy as np
from PyQt5 import QtCore

from keithley.keithley_wrapper import SweepController
from rot.rot_wrapper import MotionController
from .config import RAMP_STEP_V, RAMP_DWELL_S


class AcquireHistogramWorker(QtCore.QThread):
    """Single histogram acquisition — used by 'Take Histogram' button."""
    done         = QtCore.pyqtSignal(object, object)   # time_ps, counts (or None, None)
    error        = QtCore.pyqtSignal(str)
    rates_update = QtCore.pyqtSignal(int, int, int, str)  # sync_hz, ch0_hz, ch1_hz, warn_text

    def __init__(self, ph, tacq_ms: int, poll_interval_s: float = 1.0, parent=None):
        super().__init__(parent)
        self.ph = ph
        self.tacq_ms = int(tacq_ms)
        self.poll_interval_s = float(poll_interval_s)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            ph = self.ph
            ph.clear_hist_mem(block=0)
            ph.set_stop_overflow(enable=False)
            ph.start_meas(self.tacq_ms)
            timeout_s      = self.tacq_ms / 1000.0 + 10.0
            t0             = time.time()
            last_rate_poll = t0
            while not ph.ctc_done():
                if self._abort or (time.time() - t0 > timeout_s):
                    ph.stop_meas()
                    self.done.emit(None, None)
                    return
                now = time.time()
                if now - last_rate_poll >= self.poll_interval_s:
                    try:
                        rs = ph.get_rates_and_warnings()
                        warn = rs.warnings_text if rs.warnings_bitfield != 0 else ""
                        self.rates_update.emit(
                            rs.sync_rate_hz, rs.ch0_rate_hz, rs.ch1_rate_hz, warn)
                    except Exception:
                        pass
                    last_rate_poll = now
                time.sleep(0.05)
            ph.stop_meas()
            counts  = np.asarray(ph.read_histogram(block=0), dtype=np.uint32)
            time_ps = np.asarray(ph.make_time_axis_ps(),      dtype=np.float64)
            self.done.emit(time_ps, counts)
        except Exception as e:
            self.error.emit(str(e))
            self.done.emit(None, None)


class HomeGatesWorker(QtCore.QThread):
    """Ramp both gates to 0 V sequentially."""
    status = QtCore.pyqtSignal(str)
    done   = QtCore.pyqtSignal(str)   # ok / error: ...

    def __init__(self, dev_a, dev_b, parent=None):
        super().__init__(parent)
        self.dev_a = dev_a
        self.dev_b = dev_b
        self._ctrl_a = SweepController()
        self._ctrl_b = SweepController()

    def abort(self):
        self._ctrl_a.abort()
        self._ctrl_b.abort()

    def run(self):
        errors = []
        for label, dev, ctrl in [("A", self.dev_a, self._ctrl_a),
                                  ("B", self.dev_b, self._ctrl_b)]:
            if dev is None:
                continue
            try:
                self.status.emit(f"Homing Gate {label} → 0 V …")
                dev.home(
                    controller=ctrl,
                    ramp_step_v=RAMP_STEP_V,
                    ramp_dwell_s=RAMP_DWELL_S,
                    on_progress=None,
                    verify=False,
                )
            except Exception as e:
                errors.append(f"Gate {label}: {e}")
        self.done.emit("error: " + "; ".join(errors) if errors else "ok")


class SweepWorker(QtCore.QThread):
    """Full grid sweep: angle × Va × Vb — acquires one histogram per point."""
    progress   = QtCore.pyqtSignal(str, int, int)              # msg, done_count, total
    vi_update  = QtCore.pyqtSignal(str, float, float)          # "A"/"B", V, I(A)
    point_done = QtCore.pyqtSignal(int, float, float, float,
                                   object, object)             # idx, angle, va, vb, t_ps, counts
    done       = QtCore.pyqtSignal(str)

    def __init__(
        self,
        ph,
        dev_a, dev_b,
        rot_stage,
        angles:  List[float],
        va_list: List[float],
        vb_list: List[float],
        tacq_ms: int,
        icomp_a: float,          # amps
        icomp_b: float,
        gate_settle_s:  float,
        wheel_settle_s: float,
        parent=None,
    ):
        super().__init__(parent)
        self.ph          = ph
        self.dev_a       = dev_a
        self.dev_b       = dev_b
        self.rot_stage   = rot_stage
        self.angles      = list(angles)  if angles  else [None]
        self.va_list     = list(va_list) if va_list else [None]
        self.vb_list     = list(vb_list) if vb_list else [None]
        self.tacq_ms     = int(tacq_ms)
        self.icomp_a     = float(icomp_a)
        self.icomp_b     = float(icomp_b)
        self.gate_settle_s  = float(gate_settle_s)
        self.wheel_settle_s = float(wheel_settle_s)
        self._abort = False

    def abort(self):
        self._abort = True

    # ---- helpers ----

    def _ramp(self, dev, label: str, target_v: float, icomp: float):
        ctrl = SweepController()
        dev.set_compliance(icomp)
        dev.set_output(True)
        vi = dev.ramp_to_voltage(
            target_v,
            ramp_step_v=RAMP_STEP_V,
            ramp_dwell_s=RAMP_DWELL_S,
            controller=ctrl,
            micro_read=True,
            on_microstep=lambda vi_: self.vi_update.emit(label, vi_.v, vi_.i),
            verify=True,
        )
        self.vi_update.emit(label, vi.v, vi.i)

    def _acquire(self):
        ph = self.ph
        ph.clear_hist_mem(block=0)
        ph.set_stop_overflow(enable=False)
        ph.start_meas(self.tacq_ms)
        timeout_s = self.tacq_ms / 1000.0 + 10.0
        t0 = time.time()
        while not ph.ctc_done():
            if self._abort or (time.time() - t0 > timeout_s):
                ph.stop_meas()
                return None, None
            time.sleep(0.05)
        ph.stop_meas()
        counts  = np.asarray(ph.read_histogram(block=0), dtype=np.uint32)
        time_ps = np.asarray(ph.make_time_axis_ps(),      dtype=np.float64)
        return time_ps, counts

    # ---- main loop ----

    def run(self):
        # va_list and vb_list are paired (same length); gates step together, not as a grid
        gate_pairs = list(zip(self.va_list, self.vb_list))
        total = len(self.angles) * len(gate_pairs)
        idx = 0
        try:
            for angle in self.angles:
                if self._abort:
                    break
                if angle is not None and self.rot_stage is not None:
                    self.progress.emit(f"Wheel → {angle:.2f}°", idx, total)
                    ctrl = MotionController()
                    self.rot_stage.move_to(angle, step_deg=5.0, controller=ctrl)
                    time.sleep(self.wheel_settle_s)

                for va, vb in gate_pairs:
                    if self._abort:
                        break
                    if va is not None and self.dev_a is not None:
                        self.progress.emit(f"Gate A → {va:.3f} V", idx, total)
                        self._ramp(self.dev_a, "A", va, self.icomp_a)
                    if vb is not None and self.dev_b is not None:
                        self.progress.emit(f"Gate B → {vb:.3f} V", idx, total)
                        self._ramp(self.dev_b, "B", vb, self.icomp_b)

                    time.sleep(self.gate_settle_s)
                    self.progress.emit(
                        f"Acquiring [{idx + 1}/{total}]  "
                        f"θ={angle or 0:.1f}°  Va={va or 0:.3f} V  Vb={vb or 0:.3f} V",
                        idx, total,
                    )
                    t_ps, cts = self._acquire()
                    if t_ps is not None:
                        self.point_done.emit(
                            idx,
                            float(angle or 0.0),
                            float(va or 0.0),
                            float(vb or 0.0),
                            t_ps, cts,
                        )
                    idx += 1

        except Exception as e:
            self.done.emit(f"error: {e}")
            return

        self.done.emit("aborted" if self._abort else "ok")

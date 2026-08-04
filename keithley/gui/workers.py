from PyQt5 import QtCore

from ..keithley_wrapper import (
    SweepConfig,
    DualSweepConfig,
    VIReading,
    ComplianceTrip,
    sweep_dual,
)

from .config import RAMP_STEP_V, RAMP_DWELL_S


class RampThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, float)  # V, I(A) at every microstep
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)  # ok/aborted/error:...

    def __init__(self, dev, target_v: float, controller, parent=None):
        super().__init__(parent)
        self.dev = dev
        self.target_v = float(target_v)
        self.controller = controller

    def run(self):
        try:
            self.status.emit(f"Ramping to {self.target_v:.6g} V...")

            def on_microstep(vi: VIReading):
                self.progress.emit(vi.v, vi.i)

            vi_last = self.dev.ramp_to_voltage(
                self.target_v,
                ramp_step_v=RAMP_STEP_V,
                ramp_dwell_s=RAMP_DWELL_S,
                controller=self.controller,
                micro_read=True,
                on_microstep=on_microstep,
                verify=True,
            )

            if self.controller.is_aborted():
                self.done.emit("aborted")
            else:
                self.progress.emit(vi_last.v, vi_last.i)
                self.done.emit("ok")

        except Exception as e:
            self.done.emit(f"error: {e}")


class SingleSweepThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, float, float)  # step_idx, V, I(A) macro steps
    ramp_progress = QtCore.pyqtSignal(float, float)  # V, I(A) during initial ramp to V0
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)  # ok/aborted/trip/error:...

    def __init__(self, dev, cfg: SweepConfig, controller, parent=None):
        super().__init__(parent)
        self.dev = dev
        self.cfg = cfg
        self.controller = controller

    def run(self):
        try:
            def on_status(msg: str):
                self.status.emit(msg)

            def on_ramp_progress(vi: VIReading):
                self.ramp_progress.emit(vi.v, vi.i)

            def on_progress(idx: int, vi: VIReading):
                self.progress.emit(idx, vi.v, vi.i)

            out = self.dev.sweep_single(
                self.cfg,
                self.controller,
                on_progress=on_progress,
                on_status=on_status,
                on_ramp_progress=on_ramp_progress,
            )
            self.done.emit(out)
        except ComplianceTrip:
            self.done.emit("trip")
        except Exception as e:
            self.done.emit(f"error: {e}")


class DualSweepThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, float, float, float, float)  # idx, vA,iA, vB,iB macro steps
    ramp_progress = QtCore.pyqtSignal(str, float, float)  # "A"/"B", V, I(A) during initial ramp to V0
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(self, devA, devB, cfg: DualSweepConfig, controller, parent=None):
        super().__init__(parent)
        self.devA = devA
        self.devB = devB
        self.cfg = cfg
        self.controller = controller

    def run(self):
        try:
            def on_status(msg: str):
                self.status.emit(msg)

            def on_ramp_progress_a(vi: VIReading):
                self.ramp_progress.emit("A", vi.v, vi.i)

            def on_ramp_progress_b(vi: VIReading):
                self.ramp_progress.emit("B", vi.v, vi.i)

            def on_progress(idx: int, viA: VIReading, viB: VIReading):
                self.progress.emit(idx, viA.v, viA.i, viB.v, viB.i)

            out = sweep_dual(
                self.devA,
                self.devB,
                self.cfg,
                self.controller,
                on_progress=on_progress,
                on_status=on_status,
                on_ramp_progress_a=on_ramp_progress_a,
                on_ramp_progress_b=on_ramp_progress_b,
            )
            self.done.emit(out)
        except ComplianceTrip:
            self.done.emit("trip")
        except Exception as e:
            self.done.emit(f"error: {e}")


class HomeThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, float)  # V, I (microsteps)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(self, dev, controller, parent=None):
        super().__init__(parent)
        self.dev = dev
        self.controller = controller

    def run(self):
        try:
            def on_progress(vi: VIReading):
                self.progress.emit(vi.v, vi.i)

            out = self.dev.home(
                controller=self.controller,
                ramp_step_v=RAMP_STEP_V,
                ramp_dwell_s=RAMP_DWELL_S,
                on_progress=on_progress,
                verify=False,
            )
            self.done.emit(out)
        except Exception as e:
            self.done.emit(f"error: {e}")

from PyQt5 import QtCore

from rot.rot_wrapper import MotionController, SweepConfig


class MoveThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float)  # angle
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(self, stage, target_deg: float, step_deg: float, accel: float, controller: MotionController, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.target_deg = float(target_deg)
        self.step_deg = float(step_deg)
        self.accel = float(accel) if accel is not None else None
        self.controller = controller

    def run(self):
        try:
            self.status.emit(f"Moving to {self.target_deg:.3f} deg")

            def on_step(angle: float):
                self.progress.emit(angle)

            self.stage.move_to(
                self.target_deg,
                step_deg=self.step_deg,
                accel=self.accel,
                controller=self.controller,
                on_step=on_step,
            )
            if self.controller.is_aborted():
                self.done.emit("aborted")
            else:
                self.progress.emit(self.stage.get_position())
                self.done.emit("ok")
        except Exception as e:
            self.done.emit(f"error: {e}")


class HomeThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(self, stage, controller: MotionController, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.controller = controller

    def run(self):
        try:
            self.status.emit("Homing...")

            def on_step(angle: float):
                self.progress.emit(angle)

            self.stage.home(controller=self.controller, on_progress=on_step)
            if self.controller.is_aborted():
                self.done.emit("aborted")
            else:
                self.progress.emit(self.stage.get_position())
                self.done.emit("ok")
        except Exception as e:
            self.done.emit(f"error: {e}")


class SweepThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, float)  # idx, total, angle
    ramp_progress = QtCore.pyqtSignal(float)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(
        self,
        stage,
        cfg: SweepConfig,
        controller: MotionController,
        parent=None,
    ):
        super().__init__(parent)
        self.stage = stage
        self.cfg = cfg
        self.controller = controller

    def run(self):
        try:
            self.status.emit("Sweeping...")

            def on_step(idx: int, total: int, angle: float):
                self.progress.emit(idx, total, angle)

            def on_ramp(angle: float):
                self.ramp_progress.emit(angle)

            self.stage.sweep(
                self.cfg,
                controller=self.controller,
                on_step=on_step,
                on_ramp_step=on_ramp,
            )
            if self.controller.is_aborted():
                self.done.emit("aborted")
            else:
                self.done.emit("ok")
        except Exception as e:
            self.done.emit(f"error: {e}")

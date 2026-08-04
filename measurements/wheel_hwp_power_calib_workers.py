from __future__ import annotations

import numpy as np
from PyQt5 import QtCore

from rot.rot_wrapper import MotionController
from PM100A.pm100a_wrapper import PM100A


class StageSweepWorker(QtCore.QThread):
    """
    Move *fixed_stage* to *fixed_angle*, then sweep *sweep_stage* through
    *sweep_angles*, collecting *n_readings* power samples at each step.

    The PM100A session is opened inside run() so it lives on this thread —
    TLPM sessions have thread affinity and cannot be called from a different
    thread than the one that opened them.

    Signals
    -------
    point_done(step, total, sweep_angle_deg, mean_power_w, std_power_w)
    status(message)
    done(result_str, data)
        result_str : "ok" | "aborted" | "error: <msg>"
        data       : float ndarray shape (N, 2) — [sweep_angle_deg, mean_power_w]
    """

    point_done = QtCore.pyqtSignal(int, int, float, float, float)
    status     = QtCore.pyqtSignal(str)
    done       = QtCore.pyqtSignal(str, object)

    def __init__(
        self,
        fixed_stage,
        sweep_stage,
        pm_resource: str,
        *,
        fixed_angle: float,
        sweep_angles: np.ndarray,
        n_readings: int,
        fixed_label: str = "stage",
        sweep_label: str = "stage",
        ramp_step_deg: float = 5.0,
        pm_averaging: int = 100,
        parent=None,
    ):
        super().__init__(parent)
        self.fixed_stage   = fixed_stage
        self.sweep_stage   = sweep_stage
        self.pm_resource   = pm_resource
        self.fixed_angle   = float(fixed_angle)
        self.sweep_angles  = np.asarray(sweep_angles, dtype=float)
        self.n_readings    = int(n_readings)
        self.fixed_label   = fixed_label
        self.sweep_label   = sweep_label
        self.ramp_step_deg = float(ramp_step_deg)
        self.pm_averaging  = int(pm_averaging)
        self._stop         = False
        self._controller   = MotionController()

    def stop(self) -> None:
        self._stop = True
        self._controller.abort()
        for stage in (self.fixed_stage, self.sweep_stage):
            try:
                if stage is not None:
                    stage.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------

    def run(self) -> None:
        total   = len(self.sweep_angles)
        results = np.full((total, 2), np.nan)

        self.status.emit("Opening power meter…")
        try:
            pm = PM100A(self.pm_resource)
            pm.set_averaging(self.pm_averaging)
            pm.set_auto_range(True)
        except Exception as exc:
            self.done.emit(f"error: Could not open power meter: {exc}", results)
            return

        try:
            # --- 1. Move fixed stage ---
            self.status.emit(
                f"Moving {self.fixed_label} to {self.fixed_angle:.1f}°…"
            )
            self.fixed_stage.move_to(
                self.fixed_angle,
                step_deg=self.ramp_step_deg,
                controller=self._controller,
            )
            if self._stop:
                self.done.emit("aborted", results)
                return

            # --- 2. Sweep ---
            for i, angle in enumerate(self.sweep_angles):
                if self._stop:
                    self.done.emit("aborted", results)
                    return

                self.status.emit(
                    f"Step {i + 1}/{total}  {self.sweep_label} → {angle:.2f}°  (moving…)"
                )
                self.sweep_stage.move_to(
                    angle,
                    step_deg=self.ramp_step_deg,
                    controller=self._controller,
                )
                if self._stop:
                    self.done.emit("aborted", results)
                    return

                # --- 3. Collect N readings ---
                samples    = []
                fail_count = 0
                last_error = None
                for k in range(self.n_readings):
                    if self._stop:
                        break
                    self.status.emit(
                        f"Step {i + 1}/{total}  {self.sweep_label} = {angle:.2f}°  "
                        f"(reading {k + 1}/{self.n_readings})"
                    )
                    try:
                        samples.append(pm.measure_power())
                    except Exception as exc:
                        fail_count += 1
                        last_error = exc
                        if fail_count == 1:
                            self.status.emit(f"PM read error: {exc}")

                if not samples:
                    self.status.emit(
                        f"Step {i + 1}: all {fail_count} reads failed. "
                        f"Last error: {last_error}"
                    )

                mean_p = float(np.mean(samples)) if samples else float("nan")
                std_p  = float(np.std(samples))  if samples else float("nan")
                results[i] = [angle, mean_p]
                self.point_done.emit(i, total, angle, mean_p, std_p)

            self.done.emit("ok", results)

        except Exception as exc:
            self.done.emit(f"error: {exc}", results)
        finally:
            try:
                pm.close()
            except Exception:
                pass

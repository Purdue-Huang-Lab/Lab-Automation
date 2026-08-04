from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5 import QtCore

from rot.rot_wrapper import MotionController
from measurements.power_calibration import PowerCalibEntry
from measurements.dual_wheel_power_calibration import DualWheelPowerEntry


class SweepThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    power_done = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(
        self,
        cam,
        stage,
        plan: List[dict],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        metadata_base: dict,
        wavelength_axis_nm,
        parent=None,
    ):
        super().__init__(parent)
        self.cam = cam
        self.stage = stage
        self.plan = plan
        self.accum_n = int(accum_n)
        self.exposure_ms = float(exposure_ms)
        self.ramp_step_deg = float(ramp_step_deg)
        self.metadata_base = metadata_base or {}
        self.wavelength_axis_nm = wavelength_axis_nm
        self._stop = False
        self._controller = MotionController()

    def stop(self):
        self._stop = True
        try:
            self._controller.abort()
        except Exception:
            pass
        try:
            if self.stage is not None:
                self.stage.stop()
        except Exception:
            pass

    def run(self):
        prev_timeout_scale = None
        try:
            if self.cam is None or self.stage is None:
                self.status.emit("Sweep error: device not connected")
                self.done.emit("error: device not connected")
                return

            try:
                if hasattr(self.cam, "stop_stream"):
                    self.cam.stop_stream()
                self.cam.set_frame_api("Snap")
                self.cam.set_acquisition_mode("Single")
                self.cam.set_trigger_mode("Internal")
                self.cam.set_shutter("auto")
                self.cam.set_exposure_ms(self.exposure_ms)
            except Exception:
                pass

            try:
                prev_timeout_scale = getattr(self.cam, "_timeout_scale", None)
            except Exception:
                prev_timeout_scale = None
            try:
                timeout_scale = 2.0
                if self.exposure_ms > 10000.0:
                    timeout_scale = max(timeout_scale, (self.exposure_ms / 10000.0) * 2.0)
                self.cam._timeout_scale = timeout_scale
            except Exception:
                pass

            for idx, item in enumerate(self.plan, start=1):
                if self._stop:
                    self.done.emit("aborted")
                    return
                entry: PowerCalibEntry = item["entry"]
                filepath = item.get("filepath")
                self.status.emit(f"Move to {entry.position_deg:.3f} deg ({idx}/{len(self.plan)})")
                try:
                    self.stage.move_to(
                        entry.position_deg,
                        step_deg=self.ramp_step_deg,
                        controller=self._controller,
                    )
                except Exception as exc:
                    self.status.emit(f"Stage move failed: {exc}")
                    self.done.emit(f"error: stage move failed: {exc}")
                    return

                if self._stop:
                    self.done.emit("aborted")
                    return

                pos = None
                try:
                    pos = float(self.stage.get_position())
                    if abs(pos - entry.position_deg) > 0.05:
                        self.status.emit(f"Stage at {pos:.3f} deg (target {entry.position_deg:.3f})")
                except Exception:
                    pos = None
                pos_s = f"{pos:.3f}" if pos is not None else f"{entry.position_deg:.3f}"
                self.status.emit(f"Stage at {pos_s} deg | starting acquisition ({idx}/{len(self.plan)})")

                accum_n = max(1, int(self.accum_n))
                sum_img = None
                last_fr = None

                for j in range(1, accum_n + 1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    retries = 1
                    while True:
                        try:
                            fr = self.cam.get_frame()
                        except Exception as exc:
                            self.status.emit(f"Acquire failed: {exc}")
                            self.done.emit(f"error: acquire failed: {exc}")
                            return
                        if fr.get("ok"):
                            break
                        err = str(fr.get("err", "Acquire failed"))
                        if self._stop:
                            self.done.emit("aborted")
                            return
                        if ("timeout" in err.lower()) and (retries > 0):
                            retries -= 1
                            try:
                                timeout_scale = float(getattr(self.cam, "_timeout_scale", 2.0))
                                timeout_scale = min(timeout_scale * 2.0, 50.0)
                                self.cam._timeout_scale = timeout_scale
                                self.status.emit(f"Timeout; retrying with scale {timeout_scale:g}")
                                continue
                            except Exception:
                                pass
                        self.status.emit(err)
                        self.done.emit(f"error: acquire failed: {err}")
                        return
                    img = fr.get("image")
                    if img is None:
                        self.status.emit("Acquire failed: no image")
                        self.done.emit("error: acquire failed (no image)")
                        return
                    if sum_img is None:
                        sum_img = np.asarray(img, dtype=np.float64)
                    else:
                        sum_img += np.asarray(img, dtype=np.float64)
                    last_fr = fr
                    fr_out = dict(fr)
                    fr_out["image"] = sum_img
                    fr_out["accum_idx"] = j
                    fr_out["accum_n"] = accum_n
                    fr_out["entry"] = entry
                    self.frame_ready.emit(fr_out)

                if last_fr is None:
                    self.status.emit("Acquire failed: no frames")
                    self.done.emit("error: acquire failed (no frames)")
                    return

                if filepath:
                    meta = dict(self.metadata_base)
                    meta.update({
                        "series": entry.series,
                        "power_w": entry.power_w,
                        "position_deg": entry.position_deg,
                    })
                    try:
                        self.cam.save_ascii(filepath, sum_img, metadata=meta, wavelength_axis_nm=self.wavelength_axis_nm)
                    except Exception as exc:
                        self.status.emit(f"Save failed: {exc}")
                        self.done.emit(f"error: save failed: {exc}")
                        return

                out = {
                    "entry": entry,
                    "image": sum_img,
                    "accum_n": accum_n,
                    "filepath": filepath,
                }
                self.power_done.emit(out)

            try:
                if prev_timeout_scale is not None:
                    self.cam._timeout_scale = prev_timeout_scale
                else:
                    self.cam._timeout_scale = 1.0
            except Exception:
                pass
            self.done.emit("ok")
        except Exception as exc:
            try:
                if prev_timeout_scale is not None:
                    self.cam._timeout_scale = prev_timeout_scale
                else:
                    self.cam._timeout_scale = 1.0
            except Exception:
                pass
            self.status.emit(f"Unexpected sweep error: {exc}")
            self.done.emit(f"error: {exc}")


class DualWheelSweepThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    power_done = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(
        self,
        cam,
        stage_a,
        stage_b,
        plan: List[dict],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        metadata_base: dict,
        wavelength_axis_nm,
        parent=None,
    ):
        super().__init__(parent)
        self.cam = cam
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.plan = plan
        self.accum_n = int(accum_n)
        self.exposure_ms = float(exposure_ms)
        self.ramp_step_deg = float(ramp_step_deg)
        self.metadata_base = metadata_base or {}
        self.wavelength_axis_nm = wavelength_axis_nm
        self._stop = False
        self._controller = MotionController()

    def stop(self):
        self._stop = True
        try:
            self._controller.abort()
        except Exception:
            pass
        for stage in (self.stage_a, self.stage_b):
            try:
                if stage is not None:
                    stage.stop()
            except Exception:
                pass

    def run(self):
        prev_timeout_scale = None
        try:
            if self.cam is None or self.stage_a is None or self.stage_b is None:
                self.status.emit("Sweep error: device not connected")
                self.done.emit("error: device not connected")
                return

            try:
                if hasattr(self.cam, "stop_stream"):
                    self.cam.stop_stream()
                self.cam.set_frame_api("Snap")
                self.cam.set_acquisition_mode("Single")
                self.cam.set_trigger_mode("Internal")
                self.cam.set_shutter("auto")
                self.cam.set_exposure_ms(self.exposure_ms)
            except Exception:
                pass

            try:
                prev_timeout_scale = getattr(self.cam, "_timeout_scale", None)
            except Exception:
                prev_timeout_scale = None
            try:
                timeout_scale = 2.0
                if self.exposure_ms > 10000.0:
                    timeout_scale = max(timeout_scale, (self.exposure_ms / 10000.0) * 2.0)
                self.cam._timeout_scale = timeout_scale
            except Exception:
                pass

            for idx, item in enumerate(self.plan, start=1):
                if self._stop:
                    self.done.emit("aborted")
                    return
                entry: DualWheelPowerEntry = item["entry"]
                filepath = item.get("filepath")
                self.status.emit(
                    f"Move to A={entry.a_deg:.3f}, B={entry.b_deg:.3f} deg ({idx}/{len(self.plan)})"
                )
                try:
                    self.stage_a.move_to(
                        entry.a_deg,
                        step_deg=self.ramp_step_deg,
                        controller=self._controller,
                    )
                    self.stage_b.move_to(
                        entry.b_deg,
                        step_deg=self.ramp_step_deg,
                        controller=self._controller,
                    )
                except Exception as exc:
                    self.status.emit(f"Stage move failed: {exc}")
                    self.done.emit(f"error: stage move failed: {exc}")
                    return

                if self._stop:
                    self.done.emit("aborted")
                    return

                pos_a = None
                pos_b = None
                try:
                    pos_a = float(self.stage_a.get_position())
                except Exception:
                    pos_a = None
                try:
                    pos_b = float(self.stage_b.get_position())
                except Exception:
                    pos_b = None
                pos_a_s = f"{pos_a:.3f}" if pos_a is not None else f"{entry.a_deg:.3f}"
                pos_b_s = f"{pos_b:.3f}" if pos_b is not None else f"{entry.b_deg:.3f}"
                self.status.emit(
                    f"Stage at A={pos_a_s} deg, B={pos_b_s} deg | starting acquisition ({idx}/{len(self.plan)})"
                )

                accum_n = max(1, int(self.accum_n))
                sum_img = None
                last_fr = None

                for j in range(1, accum_n + 1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    retries = 1
                    while True:
                        try:
                            fr = self.cam.get_frame()
                        except Exception as exc:
                            self.status.emit(f"Acquire failed: {exc}")
                            self.done.emit(f"error: acquire failed: {exc}")
                            return
                        if fr.get("ok"):
                            break
                        err = str(fr.get("err", "Acquire failed"))
                        if self._stop:
                            self.done.emit("aborted")
                            return
                        if ("timeout" in err.lower()) and (retries > 0):
                            retries -= 1
                            try:
                                timeout_scale = float(getattr(self.cam, "_timeout_scale", 2.0))
                                timeout_scale = min(timeout_scale * 2.0, 50.0)
                                self.cam._timeout_scale = timeout_scale
                                self.status.emit(f"Timeout; retrying with scale {timeout_scale:g}")
                                continue
                            except Exception:
                                pass
                        self.status.emit(err)
                        self.done.emit(f"error: acquire failed: {err}")
                        return
                    img = fr.get("image")
                    if img is None:
                        self.status.emit("Acquire failed: no image")
                        self.done.emit("error: acquire failed (no image)")
                        return
                    if sum_img is None:
                        sum_img = np.asarray(img, dtype=np.float64)
                    else:
                        sum_img += np.asarray(img, dtype=np.float64)
                    last_fr = fr
                    fr_out = dict(fr)
                    fr_out["image"] = sum_img
                    fr_out["accum_idx"] = j
                    fr_out["accum_n"] = accum_n
                    fr_out["entry"] = entry
                    self.frame_ready.emit(fr_out)

                if last_fr is None:
                    self.status.emit("Acquire failed: no frames")
                    self.done.emit("error: acquire failed (no frames)")
                    return

                if filepath:
                    meta = dict(self.metadata_base)
                    meta.update({
                        "power_w": entry.power_w,
                        "a_deg": entry.a_deg,
                        "b_deg": entry.b_deg,
                    })
                    try:
                        self.cam.save_ascii(filepath, sum_img, metadata=meta, wavelength_axis_nm=self.wavelength_axis_nm)
                    except Exception as exc:
                        self.status.emit(f"Save failed: {exc}")
                        self.done.emit(f"error: save failed: {exc}")
                        return

                out = {
                    "entry": entry,
                    "image": sum_img,
                    "accum_n": accum_n,
                    "filepath": filepath,
                }
                self.power_done.emit(out)

            try:
                if prev_timeout_scale is not None:
                    self.cam._timeout_scale = prev_timeout_scale
                else:
                    self.cam._timeout_scale = 1.0
            except Exception:
                pass
            self.done.emit("ok")
        except Exception as exc:
            try:
                if prev_timeout_scale is not None:
                    self.cam._timeout_scale = prev_timeout_scale
                else:
                    self.cam._timeout_scale = 1.0
            except Exception:
                pass
            self.status.emit(f"Unexpected sweep error: {exc}")
            self.done.emit(f"error: {exc}")

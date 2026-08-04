from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from rot.rot_wrapper import MotionController


class IntensitySweepThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    point_ready = QtCore.pyqtSignal(float, float, object)  # position_deg, intensity, image
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(
        self,
        cam,
        stage,
        positions: Iterable[float],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        crop: Tuple[int, int, int, int],
        roi: Optional[Tuple[int, int, int, int]],
        parent=None,
    ):
        super().__init__(parent)
        self.cam = cam
        self.stage = stage
        self.positions = list(positions)
        self.accum_n = int(accum_n)
        self.exposure_ms = float(exposure_ms)
        self.ramp_step_deg = float(ramp_step_deg)
        self.crop = crop
        self.roi = roi
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

            self._warmup_acq()

            for idx, target in enumerate(self.positions, start=1):
                if self._stop:
                    self.done.emit("aborted")
                    return
                self.status.emit(f"Move to {target:.3f} deg ({idx}/{len(self.positions)})")
                try:
                    self.stage.move_to(
                        target,
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
                    self.frame_ready.emit(fr_out)

                if last_fr is None:
                    self.status.emit("Acquire failed: no frames")
                    self.done.emit("error: acquire failed (no frames)")
                    return

                disp = self._prepare_display_image(sum_img)
                intensity = self._roi_sum(disp, self.roi)
                self.point_ready.emit(float(target), float(intensity), sum_img)

            self.done.emit("ok")
        except Exception as exc:
            self.status.emit(f"Unexpected sweep error: {exc}")
            self.done.emit(f"error: {exc}")
        finally:
            try:
                if prev_timeout_scale is not None:
                    self.cam._timeout_scale = prev_timeout_scale
                else:
                    self.cam._timeout_scale = 1.0
            except Exception:
                pass

    def _prepare_display_image(self, raw):
        try:
            arr = np.asarray(raw)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        top, bottom, left, right = self.crop
        h, w = arr.shape
        top = max(0, min(int(top), h - 1))
        bottom = max(0, min(int(bottom), h - 1))
        left = max(0, min(int(left), w - 1))
        right = max(0, min(int(right), w - 1))
        y2 = max(top + 1, h - bottom)
        x2 = max(left + 1, w - right)
        crop = arr[top:y2, left:x2]
        return np.flipud(crop)

    @staticmethod
    def _roi_sum(img, roi: Optional[Tuple[int, int, int, int]]) -> float:
        if img is None:
            return 0.0
        if roi is None:
            return float(np.nansum(img))
        x1, x2, y1, y2 = roi
        try:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
        except Exception:
            return float(np.nansum(img))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        h, w = img.shape
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return float(np.nansum(img))
        return float(np.nansum(img[y1:y2, x1:x2]))

    def _warmup_acq(self) -> bool:
        retries = 1
        while True:
            try:
                fr = self.cam.get_frame()
            except Exception as exc:
                self.status.emit(f"Warmup acquire failed: {exc}")
                return False
            if fr.get("ok"):
                return True
            err = str(fr.get("err", "Acquire failed"))
            if ("timeout" in err.lower()) and (retries > 0):
                retries -= 1
                try:
                    timeout_scale = float(getattr(self.cam, "_timeout_scale", 2.0))
                    timeout_scale = min(timeout_scale * 2.0, 50.0)
                    self.cam._timeout_scale = timeout_scale
                except Exception:
                    pass
                continue
            self.status.emit(f"Warmup acquire failed: {err}")
            return False

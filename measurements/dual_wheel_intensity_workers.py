from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from rot.rot_wrapper import MotionController


class DualWheelBaseThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(
        self,
        cam,
        stage_a,
        stage_b,
        *,
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        ref_angles: Optional[Tuple[float, float]] = None,
        ref_every: int = 1,
        settle_ms: float = 0.0,
        crop: Tuple[int, int, int, int],
        roi: Optional[Tuple[int, int, int, int]],
        max_retries: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self.cam = cam
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.accum_n = int(accum_n)
        self.exposure_ms = float(exposure_ms)
        self.ramp_step_deg = float(ramp_step_deg)
        self.ref_angles = tuple(ref_angles) if ref_angles is not None else None
        self.ref_every = max(1, int(ref_every))
        self.settle_ms = float(settle_ms)
        self.crop = crop
        self.roi = roi
        self.max_retries = max(1, int(max_retries))
        self._stop = False
        self._controller = MotionController()
        self._ref_baseline = None
        self._ref_last = None

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

    def _configure_camera(self) -> Optional[float]:
        prev_timeout_scale = None
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
        return prev_timeout_scale

    def _restore_timeout_scale(self, prev_timeout_scale: Optional[float]) -> None:
        try:
            if prev_timeout_scale is not None:
                self.cam._timeout_scale = prev_timeout_scale
            else:
                self.cam._timeout_scale = 1.0
        except Exception:
            pass

    def _move_stage(self, stage, target: float, label: str) -> bool:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            if self._stop:
                return False
            try:
                stage.move_to(
                    target,
                    step_deg=self.ramp_step_deg,
                    controller=self._controller,
                )
                return True
            except Exception as exc:
                last_exc = exc
                self.status.emit(f"{label} move failed ({attempt}/{self.max_retries}): {exc}")
                time.sleep(0.1)
        if last_exc is not None:
            self.status.emit(f"{label} move failed: {last_exc}")
        return False

    def _settle(self) -> bool:
        if self._stop:
            return False
        delay_s = max(0.0, float(self.settle_ms) / 1000.0)
        if delay_s <= 0:
            return True
        time.sleep(delay_s)
        return not self._stop

    def _warmup_acq(self) -> bool:
        retries = 1
        while True:
            if self._stop:
                return False
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

    def _get_frame(self) -> Optional[dict]:
        retries = 1
        while True:
            if self._stop:
                return None
            try:
                fr = self.cam.get_frame()
            except Exception as exc:
                self.status.emit(f"Acquire failed: {exc}")
                return None
            if fr.get("ok"):
                return fr
            err = str(fr.get("err", "Acquire failed"))
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
            return None

    def _acquire_sum(self) -> Optional[np.ndarray]:
        accum_n = max(1, int(self.accum_n))
        for attempt in range(1, self.max_retries + 1):
            if self._stop:
                return None
            sum_img = None
            for j in range(1, accum_n + 1):
                fr = self._get_frame()
                if fr is None:
                    sum_img = None
                    break
                img = fr.get("image")
                if img is None:
                    sum_img = None
                    break
                if sum_img is None:
                    sum_img = np.asarray(img, dtype=np.float64)
                else:
                    sum_img += np.asarray(img, dtype=np.float64)
                fr_out = dict(fr)
                fr_out["image"] = sum_img
                fr_out["accum_idx"] = j
                fr_out["accum_n"] = accum_n
                self.frame_ready.emit(fr_out)
            if sum_img is not None:
                avg_img = sum_img / float(accum_n)
                return avg_img
            if attempt < self.max_retries:
                self.status.emit(f"Acquire failed ({attempt}/{self.max_retries}); retrying")
                time.sleep(0.1)
        return None

    def _acquire_reference(self) -> Optional[float]:
        if self.ref_angles is None:
            return None
        ref_a, ref_b = self.ref_angles
        self.status.emit(f"Move stages to reference A={ref_a:.3f}, B={ref_b:.3f}")
        if not self._move_stage(self.stage_a, float(ref_a), "Ref stage 1"):
            return None
        if not self._move_stage(self.stage_b, float(ref_b), "Ref stage 2"):
            return None
        if not self._settle():
            return None
        sum_img = self._acquire_sum()
        if sum_img is None:
            return None
        disp = self._prepare_display_image(sum_img)
        intensity = self._roi_sum(disp, self.roi)
        if np.isfinite(intensity) and intensity > 0:
            if self._ref_baseline is None:
                self._ref_baseline = float(intensity)
            self._ref_last = float(intensity)
        return float(intensity)

    def _normalize_intensity(self, raw: float, ref_intensity: Optional[float]) -> float:
        if ref_intensity is None:
            return float(raw)
        if self._ref_baseline is None:
            return float(raw)
        if not np.isfinite(ref_intensity) or ref_intensity <= 0:
            return float(raw)
        if not np.isfinite(raw):
            return float(raw)
        return float(raw) * (float(self._ref_baseline) / float(ref_intensity))

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


class DualWheelFineSweepThread(DualWheelBaseThread):
    point_ready = QtCore.pyqtSignal(float, float, float, object)  # angle_deg, intensity, ref_level, image

    def __init__(
        self,
        cam,
        stage_active,
        stage_fixed,
        *,
        fixed_angle: float,
        angles: Iterable[float],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        ref_angles: Optional[Tuple[float, float]] = None,
        ref_every: int = 1,
        settle_ms: float = 0.0,
        crop: Tuple[int, int, int, int],
        roi: Optional[Tuple[int, int, int, int]],
        max_retries: int = 3,
        parent=None,
    ):
        super().__init__(
            cam,
            stage_active,
            stage_fixed,
            accum_n=accum_n,
            exposure_ms=exposure_ms,
            ramp_step_deg=ramp_step_deg,
            ref_angles=ref_angles,
            ref_every=ref_every,
            settle_ms=settle_ms,
            crop=crop,
            roi=roi,
            max_retries=max_retries,
            parent=parent,
        )
        self.stage_active = stage_active
        self.stage_fixed = stage_fixed
        self.fixed_angle = float(fixed_angle)
        self.angles = list(angles)

    def run(self):
        prev_timeout_scale = None
        try:
            if self.cam is None or self.stage_active is None or self.stage_fixed is None:
                self.status.emit("Sweep error: device not connected")
                self.done.emit("error: device not connected")
                return

            prev_timeout_scale = self._configure_camera()
            if not self._warmup_acq():
                self.done.emit("error: warmup failed")
                return

            if self._stop:
                self.done.emit("aborted")
                return

            use_ref = self.ref_angles is not None
            ref_every = max(1, int(self.ref_every))

            if not use_ref:
                self.status.emit(f"Move fixed stage to {self.fixed_angle:.3f} deg")
                if not self._move_stage(self.stage_fixed, self.fixed_angle, "Fixed stage"):
                    self.done.emit("error: fixed stage move failed")
                    return
                if not self._settle():
                    self.done.emit("aborted")
                    return

            if use_ref and ref_every <= 1:
                for idx, target in enumerate(self.angles, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    ref_intensity = self._acquire_reference()
                    if ref_intensity is None:
                        self.done.emit("error: reference acquire failed")
                        return
                    self.status.emit(f"Move fixed stage to {self.fixed_angle:.3f} deg")
                    if not self._move_stage(self.stage_fixed, self.fixed_angle, "Fixed stage"):
                        self.done.emit("error: fixed stage move failed")
                        return
                    if not self._settle():
                        self.done.emit("aborted")
                        return
                    self.status.emit(f"Move active stage to {target:.3f} deg ({idx}/{len(self.angles)})")
                    if not self._move_stage(self.stage_active, target, "Active stage"):
                        self.done.emit("error: active stage move failed")
                        return
                    if not self._settle():
                        self.done.emit("aborted")
                        return

                    sum_img = self._acquire_sum()
                    if sum_img is None:
                        self.done.emit("error: acquire failed")
                        return

                    disp = self._prepare_display_image(sum_img)
                    raw_intensity = self._roi_sum(disp, self.roi)
                    intensity = self._normalize_intensity(raw_intensity, ref_intensity)
                    self.point_ready.emit(float(target), float(intensity), float(ref_intensity), sum_img)

                self.done.emit("ok")
                return

            if not use_ref:
                for idx, target in enumerate(self.angles, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    self.status.emit(f"Move active stage to {target:.3f} deg ({idx}/{len(self.angles)})")
                    if not self._move_stage(self.stage_active, target, "Active stage"):
                        self.done.emit("error: active stage move failed")
                        return
                    if not self._settle():
                        self.done.emit("aborted")
                        return
                    sum_img = self._acquire_sum()
                    if sum_img is None:
                        self.done.emit("error: acquire failed")
                        return
                    disp = self._prepare_display_image(sum_img)
                    raw_intensity = self._roi_sum(disp, self.roi)
                    self.point_ready.emit(float(target), float(raw_intensity), float("nan"), sum_img)

                self.done.emit("ok")
                return

            buffer = []
            ref_start_idx = None
            ref_start_val = None

            def _flush_block(next_idx: Optional[int], next_ref: Optional[float]) -> bool:
                nonlocal buffer
                if not buffer:
                    return True
                if ref_start_val is None:
                    for item in buffer:
                        self.point_ready.emit(
                            float(item["target"]),
                            float(item["raw_intensity"]),
                            float("nan"),
                            item["image"],
                        )
                else:
                    denom = None
                    if next_ref is not None and next_idx is not None:
                        denom = max(1, int(next_idx - ref_start_idx))
                    for item in buffer:
                        if denom is not None and denom > 0 and next_ref is not None:
                            frac = (item["idx"] - ref_start_idx) / float(denom)
                            ref_level = float(ref_start_val + frac * (next_ref - ref_start_val))
                        else:
                            ref_level = float(ref_start_val)
                        intensity = self._normalize_intensity(item["raw_intensity"], ref_level)
                        self.point_ready.emit(
                            float(item["target"]),
                            float(intensity),
                            float(ref_level),
                            item["image"],
                        )
                buffer = []
                return True

            for point_idx, target in enumerate(self.angles):
                if self._stop:
                    self.done.emit("aborted")
                    return
                if use_ref and (point_idx % ref_every == 0):
                    ref_intensity = self._acquire_reference()
                    if ref_intensity is None:
                        self.done.emit("error: reference acquire failed")
                        return
                    if ref_start_val is not None:
                        if not _flush_block(point_idx, ref_intensity):
                            self.done.emit("error: reference flush failed")
                            return
                    ref_start_idx = point_idx
                    ref_start_val = ref_intensity
                    self.status.emit(f"Move fixed stage to {self.fixed_angle:.3f} deg")
                    if not self._move_stage(self.stage_fixed, self.fixed_angle, "Fixed stage"):
                        self.done.emit("error: fixed stage move failed")
                        return
                    if not self._settle():
                        self.done.emit("aborted")
                        return

                self.status.emit(
                    f"Move active stage to {target:.3f} deg ({point_idx + 1}/{len(self.angles)})"
                )
                if not self._move_stage(self.stage_active, target, "Active stage"):
                    self.done.emit("error: active stage move failed")
                    return
                if not self._settle():
                    self.done.emit("aborted")
                    return

                sum_img = self._acquire_sum()
                if sum_img is None:
                    self.done.emit("error: acquire failed")
                    return

                disp = self._prepare_display_image(sum_img)
                raw_intensity = self._roi_sum(disp, self.roi)
                buffer.append(
                    {
                        "idx": point_idx,
                        "target": float(target),
                        "raw_intensity": float(raw_intensity),
                        "image": None,
                    }
                )

            _flush_block(None, None)

            self.done.emit("ok")
        except Exception as exc:
            self.status.emit(f"Unexpected sweep error: {exc}")
            self.done.emit(f"error: {exc}")
        finally:
            self._restore_timeout_scale(prev_timeout_scale)


class DualWheelGridThread(DualWheelBaseThread):
    point_ready = QtCore.pyqtSignal(int, int, float, float, float, float, object)

    def __init__(
        self,
        cam,
        stage_a,
        stage_b,
        *,
        angles_a: Iterable[float],
        angles_b: Iterable[float],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        ref_angles: Optional[Tuple[float, float]] = None,
        ref_every: int = 1,
        settle_ms: float = 0.0,
        crop: Tuple[int, int, int, int],
        roi: Optional[Tuple[int, int, int, int]],
        max_retries: int = 3,
        parent=None,
    ):
        super().__init__(
            cam,
            stage_a,
            stage_b,
            accum_n=accum_n,
            exposure_ms=exposure_ms,
            ramp_step_deg=ramp_step_deg,
            ref_angles=ref_angles,
            ref_every=ref_every,
            settle_ms=settle_ms,
            crop=crop,
            roi=roi,
            max_retries=max_retries,
            parent=parent,
        )
        self.angles_a = list(angles_a)
        self.angles_b = list(angles_b)

    def run(self):
        prev_timeout_scale = None
        try:
            if self.cam is None or self.stage_a is None or self.stage_b is None:
                self.status.emit("Grid error: device not connected")
                self.done.emit("error: device not connected")
                return

            prev_timeout_scale = self._configure_camera()
            if not self._warmup_acq():
                self.done.emit("error: warmup failed")
                return

            use_ref = self.ref_angles is not None
            ref_every = max(1, int(self.ref_every))

            if use_ref and ref_every <= 1:
                for b_idx, b_angle in enumerate(self.angles_b, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    for a_idx, a_angle in enumerate(self.angles_a, start=1):
                        if self._stop:
                            self.done.emit("aborted")
                            return
                        ref_intensity = self._acquire_reference()
                        if ref_intensity is None:
                            self.done.emit("error: reference acquire failed")
                            return
                        self.status.emit(
                            f"Move stage B to {b_angle:.3f} deg ({b_idx}/{len(self.angles_b)})"
                        )
                        if not self._move_stage(self.stage_b, b_angle, "Stage B"):
                            self.done.emit("error: stage B move failed")
                            return
                        if not self._settle():
                            self.done.emit("aborted")
                            return
                        self.status.emit(
                            f"Move stage A to {a_angle:.3f} deg (A {a_idx}/{len(self.angles_a)}, B {b_idx}/{len(self.angles_b)})"
                        )
                        if not self._move_stage(self.stage_a, a_angle, "Stage A"):
                            self.done.emit("error: stage A move failed")
                            return
                        if not self._settle():
                            self.done.emit("aborted")
                            return

                        sum_img = self._acquire_sum()
                        if sum_img is None:
                            self.done.emit("error: acquire failed")
                            return

                        disp = self._prepare_display_image(sum_img)
                        raw_intensity = self._roi_sum(disp, self.roi)
                        intensity = self._normalize_intensity(raw_intensity, ref_intensity)
                        self.point_ready.emit(
                            a_idx - 1,
                            b_idx - 1,
                            float(a_angle),
                            float(b_angle),
                            float(intensity),
                            float(ref_intensity),
                            sum_img,
                        )

                self.done.emit("ok")
                return

            if not use_ref:
                for b_idx, b_angle in enumerate(self.angles_b, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    self.status.emit(f"Move stage B to {b_angle:.3f} deg ({b_idx}/{len(self.angles_b)})")
                    if not self._move_stage(self.stage_b, b_angle, "Stage B"):
                        self.done.emit("error: stage B move failed")
                        return
                    if not self._settle():
                        self.done.emit("aborted")
                        return

                    for a_idx, a_angle in enumerate(self.angles_a, start=1):
                        if self._stop:
                            self.done.emit("aborted")
                            return
                        self.status.emit(
                            f"Move stage A to {a_angle:.3f} deg (A {a_idx}/{len(self.angles_a)}, B {b_idx}/{len(self.angles_b)})"
                        )
                        if not self._move_stage(self.stage_a, a_angle, "Stage A"):
                            self.done.emit("error: stage A move failed")
                            return
                        if not self._settle():
                            self.done.emit("aborted")
                            return

                        sum_img = self._acquire_sum()
                        if sum_img is None:
                            self.done.emit("error: acquire failed")
                            return

                        disp = self._prepare_display_image(sum_img)
                        raw_intensity = self._roi_sum(disp, self.roi)
                        self.point_ready.emit(
                            a_idx - 1,
                            b_idx - 1,
                            float(a_angle),
                            float(b_angle),
                            float(raw_intensity),
                            float("nan"),
                            sum_img,
                        )

                self.done.emit("ok")
                return

            buffer = []
            ref_start_idx = None
            ref_start_val = None
            point_idx = 0
            last_a = None
            last_b = None

            def _flush_block(next_idx: Optional[int], next_ref: Optional[float]) -> bool:
                nonlocal buffer
                if not buffer:
                    return True
                denom = None
                if next_ref is not None and next_idx is not None:
                    denom = max(1, int(next_idx - ref_start_idx))
                for item in buffer:
                    if denom is not None and denom > 0 and next_ref is not None:
                        frac = (item["idx"] - ref_start_idx) / float(denom)
                        ref_level = float(ref_start_val + frac * (next_ref - ref_start_val))
                    else:
                        ref_level = float(ref_start_val)
                    intensity = self._normalize_intensity(item["raw_intensity"], ref_level)
                    self.point_ready.emit(
                        item["a_idx"],
                        item["b_idx"],
                        float(item["a_angle"]),
                        float(item["b_angle"]),
                        float(intensity),
                        float(ref_level),
                        item["image"],
                    )
                buffer = []
                return True

            for b_idx, b_angle in enumerate(self.angles_b, start=1):
                for a_idx, a_angle in enumerate(self.angles_a, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    if point_idx % ref_every == 0:
                        ref_intensity = self._acquire_reference()
                        if ref_intensity is None:
                            self.done.emit("error: reference acquire failed")
                            return
                        if ref_start_val is not None:
                            if not _flush_block(point_idx, ref_intensity):
                                self.done.emit("error: reference flush failed")
                                return
                        ref_start_idx = point_idx
                        ref_start_val = ref_intensity
                        ref_a, ref_b = self.ref_angles
                        last_a = float(ref_a)
                        last_b = float(ref_b)

                    self.status.emit(
                        f"Move stage B to {b_angle:.3f} deg ({b_idx}/{len(self.angles_b)})"
                    )
                    if last_b is None or abs(b_angle - last_b) > 1e-9:
                        if not self._move_stage(self.stage_b, b_angle, "Stage B"):
                            self.done.emit("error: stage B move failed")
                            return
                    if not self._settle():
                        self.done.emit("aborted")
                        return
                    self.status.emit(
                        f"Move stage A to {a_angle:.3f} deg (A {a_idx}/{len(self.angles_a)}, B {b_idx}/{len(self.angles_b)})"
                    )
                    if last_a is None or abs(a_angle - last_a) > 1e-9:
                        if not self._move_stage(self.stage_a, a_angle, "Stage A"):
                            self.done.emit("error: stage A move failed")
                            return
                    if not self._settle():
                        self.done.emit("aborted")
                        return

                    sum_img = self._acquire_sum()
                    if sum_img is None:
                        self.done.emit("error: acquire failed")
                        return

                    disp = self._prepare_display_image(sum_img)
                    raw_intensity = self._roi_sum(disp, self.roi)
                    buffer.append(
                        {
                            "idx": point_idx,
                            "a_idx": a_idx - 1,
                            "b_idx": b_idx - 1,
                            "a_angle": float(a_angle),
                            "b_angle": float(b_angle),
                            "raw_intensity": float(raw_intensity),
                            "image": None,
                        }
                    )
                    last_a = float(a_angle)
                    last_b = float(b_angle)
                    point_idx += 1

            _flush_block(None, None)
            self.done.emit("ok")
        except Exception as exc:
            self.status.emit(f"Unexpected grid error: {exc}")
            self.done.emit(f"error: {exc}")
        finally:
            self._restore_timeout_scale(prev_timeout_scale)


class DualWheelListThread(DualWheelBaseThread):
    point_ready = QtCore.pyqtSignal(int, float, float, float, float, object)

    def __init__(
        self,
        cam,
        stage_a,
        stage_b,
        *,
        pairs: Iterable[Tuple[float, float]],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        ref_angles: Optional[Tuple[float, float]] = None,
        ref_every: int = 1,
        settle_ms: float = 0.0,
        crop: Tuple[int, int, int, int],
        roi: Optional[Tuple[int, int, int, int]],
        max_retries: int = 3,
        parent=None,
    ):
        super().__init__(
            cam,
            stage_a,
            stage_b,
            accum_n=accum_n,
            exposure_ms=exposure_ms,
            ramp_step_deg=ramp_step_deg,
            ref_angles=ref_angles,
            ref_every=ref_every,
            settle_ms=settle_ms,
            crop=crop,
            roi=roi,
            max_retries=max_retries,
            parent=parent,
        )
        self.pairs = list(pairs)

    def run(self):
        prev_timeout_scale = None
        try:
            if self.cam is None or self.stage_a is None or self.stage_b is None:
                self.status.emit("Sweep error: device not connected")
                self.done.emit("error: device not connected")
                return

            prev_timeout_scale = self._configure_camera()
            if not self._warmup_acq():
                self.done.emit("error: warmup failed")
                return

            use_ref = self.ref_angles is not None
            ref_every = max(1, int(self.ref_every))

            if use_ref and ref_every <= 1:
                last_a = None
                last_b = None
                for idx, (a_angle, b_angle) in enumerate(self.pairs, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    ref_intensity = self._acquire_reference()
                    if ref_intensity is None:
                        self.done.emit("error: reference acquire failed")
                        return
                    ref_a, ref_b = self.ref_angles
                    last_a = float(ref_a)
                    last_b = float(ref_b)
                    self.status.emit(
                        f"Move stages to A={a_angle:.3f}, B={b_angle:.3f} ({idx}/{len(self.pairs)})"
                    )
                    if last_a is None or abs(a_angle - last_a) > 1e-9:
                        if not self._move_stage(self.stage_a, float(a_angle), "Stage A"):
                            self.done.emit("error: stage A move failed")
                            return
                    if last_b is None or abs(b_angle - last_b) > 1e-9:
                        if not self._move_stage(self.stage_b, float(b_angle), "Stage B"):
                            self.done.emit("error: stage B move failed")
                            return
                    if not self._settle():
                        self.done.emit("aborted")
                        return

                    sum_img = self._acquire_sum()
                    if sum_img is None:
                        self.done.emit("error: acquire failed")
                        return

                    disp = self._prepare_display_image(sum_img)
                    raw_intensity = self._roi_sum(disp, self.roi)
                    intensity = self._normalize_intensity(raw_intensity, ref_intensity)
                    self.point_ready.emit(
                        idx - 1,
                        float(a_angle),
                        float(b_angle),
                        float(intensity),
                        float(ref_intensity),
                        sum_img,
                    )
                    last_a = float(a_angle)
                    last_b = float(b_angle)

                self.done.emit("ok")
                return

            if not use_ref:
                last_a = None
                last_b = None
                for idx, (a_angle, b_angle) in enumerate(self.pairs, start=1):
                    if self._stop:
                        self.done.emit("aborted")
                        return
                    self.status.emit(
                        f"Move stages to A={a_angle:.3f}, B={b_angle:.3f} ({idx}/{len(self.pairs)})"
                    )
                    if last_a is None or abs(a_angle - last_a) > 1e-9:
                        if not self._move_stage(self.stage_a, float(a_angle), "Stage A"):
                            self.done.emit("error: stage A move failed")
                            return
                    if last_b is None or abs(b_angle - last_b) > 1e-9:
                        if not self._move_stage(self.stage_b, float(b_angle), "Stage B"):
                            self.done.emit("error: stage B move failed")
                            return
                    if not self._settle():
                        self.done.emit("aborted")
                        return

                    sum_img = self._acquire_sum()
                    if sum_img is None:
                        self.done.emit("error: acquire failed")
                        return

                    disp = self._prepare_display_image(sum_img)
                    raw_intensity = self._roi_sum(disp, self.roi)
                    self.point_ready.emit(
                        idx - 1,
                        float(a_angle),
                        float(b_angle),
                        float(raw_intensity),
                        float("nan"),
                        sum_img,
                    )
                    last_a = float(a_angle)
                    last_b = float(b_angle)

                self.done.emit("ok")
                return

            buffer = []
            ref_start_idx = None
            ref_start_val = None
            last_a = None
            last_b = None

            def _flush_block(next_idx: Optional[int], next_ref: Optional[float]) -> bool:
                nonlocal buffer
                if not buffer:
                    return True
                denom = None
                if next_ref is not None and next_idx is not None:
                    denom = max(1, int(next_idx - ref_start_idx))
                for item in buffer:
                    if denom is not None and denom > 0 and next_ref is not None:
                        frac = (item["idx"] - ref_start_idx) / float(denom)
                        ref_level = float(ref_start_val + frac * (next_ref - ref_start_val))
                    else:
                        ref_level = float(ref_start_val)
                    intensity = self._normalize_intensity(item["raw_intensity"], ref_level)
                    self.point_ready.emit(
                        item["idx"],
                        float(item["a_angle"]),
                        float(item["b_angle"]),
                        float(intensity),
                        float(ref_level),
                        item["image"],
                    )
                buffer = []
                return True

            for point_idx, (a_angle, b_angle) in enumerate(self.pairs):
                if self._stop:
                    self.done.emit("aborted")
                    return
                if point_idx % ref_every == 0:
                    ref_intensity = self._acquire_reference()
                    if ref_intensity is None:
                        self.done.emit("error: reference acquire failed")
                        return
                    if ref_start_val is not None:
                        if not _flush_block(point_idx, ref_intensity):
                            self.done.emit("error: reference flush failed")
                            return
                    ref_start_idx = point_idx
                    ref_start_val = ref_intensity
                    ref_a, ref_b = self.ref_angles
                    last_a = float(ref_a)
                    last_b = float(ref_b)
                self.status.emit(
                    f"Move stages to A={a_angle:.3f}, B={b_angle:.3f} ({point_idx + 1}/{len(self.pairs)})"
                )
                if last_a is None or abs(a_angle - last_a) > 1e-9:
                    if not self._move_stage(self.stage_a, float(a_angle), "Stage A"):
                        self.done.emit("error: stage A move failed")
                        return
                if last_b is None or abs(b_angle - last_b) > 1e-9:
                    if not self._move_stage(self.stage_b, float(b_angle), "Stage B"):
                        self.done.emit("error: stage B move failed")
                        return
                if not self._settle():
                    self.done.emit("aborted")
                    return

                sum_img = self._acquire_sum()
                if sum_img is None:
                    self.done.emit("error: acquire failed")
                    return

                disp = self._prepare_display_image(sum_img)
                raw_intensity = self._roi_sum(disp, self.roi)
                buffer.append(
                    {
                        "idx": point_idx,
                        "a_angle": float(a_angle),
                        "b_angle": float(b_angle),
                        "raw_intensity": float(raw_intensity),
                        "image": None,
                    }
                )
                last_a = float(a_angle)
                last_b = float(b_angle)

            _flush_block(None, None)
            self.done.emit("ok")
        except Exception as exc:
            self.status.emit(f"Unexpected sweep error: {exc}")
            self.done.emit(f"error: {exc}")
        finally:
            self._restore_timeout_scale(prev_timeout_scale)

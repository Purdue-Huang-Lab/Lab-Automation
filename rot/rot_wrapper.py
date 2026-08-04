from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

try:
    from pylablib.devices import Thorlabs
except Exception:
    Thorlabs = None


ROT_RANGE_DEFAULT = (0.0, 360.0)
DEFAULT_RAMP_STEP_DEG = 5.0
DEFAULT_SWEEP_STEP_DEG = 5.0


class RotationStageError(RuntimeError):
    pass


class MotionController:
    def __init__(self):
        self._abort = threading.Event()

    def abort(self) -> None:
        self._abort.set()

    def is_aborted(self) -> bool:
        return self._abort.is_set()


def _extract_serials(devs: object) -> list[str]:
    serials: list[str] = []
    if devs is None:
        return serials

    def add_serial(val: object) -> None:
        if val is None:
            return
        if isinstance(val, bytes):
            try:
                val = val.decode("ascii", errors="ignore")
            except Exception:
                return
        if isinstance(val, (int, float)):
            serials.append(str(int(val)))
        elif isinstance(val, str):
            s = val.strip()
            if s:
                serials.append(s)

    if isinstance(devs, dict):
        for k in devs.keys():
            add_serial(k)
        for v in devs.values():
            if isinstance(v, dict):
                for key in ("serial", "sn", "serial_number", "id"):
                    if key in v:
                        add_serial(v[key])
            else:
                add_serial(v)
    elif isinstance(devs, (list, tuple)):
        for item in devs:
            if isinstance(item, (str, int, float, bytes)):
                add_serial(item)
            elif isinstance(item, dict):
                for key in ("serial", "sn", "serial_number", "id"):
                    if key in item:
                        add_serial(item[key])
            elif isinstance(item, (list, tuple)) and item:
                add_serial(item[0])
    else:
        add_serial(devs)

    out: list[str] = []
    seen = set()
    for s in serials:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def list_kinesis_serials() -> list[str]:
    if Thorlabs is None:
        return []

    candidates: list[tuple[str, dict]] = [
        ("list_kinesis_devices", {}),
        ("list_kinesis_motors", {}),
        ("list_devices", {}),
    ]
    for name, kwargs in candidates:
        fn = getattr(Thorlabs, name, None)
        if fn is None:
            continue
        try:
            devs = fn(**kwargs) if kwargs else fn()
        except Exception:
            continue
        serials = _extract_serials(devs)
        if serials:
            return serials
    return []


def _clamp_angle(deg: float, angle_range: tuple[float, float]) -> float:
    lo, hi = angle_range
    return max(lo, min(hi, deg))


def _gen_points(start: float, stop: float, step: float, angle_range: tuple[float, float]) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step == 0:
        return []
    if (stop - start) * step < 0:
        step = -abs(step)
    else:
        step = abs(step) if stop >= start else -abs(step)

    pts: list[float] = []
    x = start
    if step > 0:
        while x <= stop + 1e-9:
            pts.append(_clamp_angle(x, angle_range))
            x += step
    else:
        while x >= stop - 1e-9:
            pts.append(_clamp_angle(x, angle_range))
            x += step
    return pts


@dataclass
class SweepConfig:
    start_deg: float
    stop_deg: float
    step_deg: float
    ramp_step_deg: float = DEFAULT_RAMP_STEP_DEG
    dwell_s: float = 0.0
    settle_s: float = 0.0


class RotationStage:
    def __init__(
        self,
        serial: str,
        *,
        angle_range: tuple[float, float] = ROT_RANGE_DEFAULT,
        scale: str = "stage",
        is_rack_system: bool = False,
        wrap: Optional[bool] = None,
    ):
        if Thorlabs is None:
            raise RotationStageError("pylablib is not available")
        self.serial = str(serial)
        self.angle_range = (float(angle_range[0]), float(angle_range[1]))
        width = float(self.angle_range[1]) - float(self.angle_range[0])
        if wrap is None:
            self._wrap = width >= 360.0 - 1e-6
        else:
            self._wrap = bool(wrap)
        self.scale = scale
        self.is_rack_system = bool(is_rack_system)
        self._lock = threading.RLock()
        self._stage = None
        self._last_pos = float("nan")

    def open(self) -> str:
        self.close()
        try:
            self._stage = Thorlabs.KinesisMotor(self.serial, scale=self.scale, is_rack_system=self.is_rack_system)
            return self.serial
        except Exception as e:
            self._stage = None
            raise RotationStageError(f"Failed to open {self.serial}: {e}") from e

    def close(self) -> None:
        with self._lock:
            try:
                if self._stage is not None:
                    self._stage.close()
            except Exception:
                pass
            self._stage = None

    def _ensure_open(self) -> None:
        if self._stage is None:
            raise RotationStageError("Stage not connected")

    def _call(self, fn, *args, **kwargs):
        with self._lock:
            self._ensure_open()
            return fn(*args, **kwargs)

    def _wrap_angle(self, pos: float) -> float:
        lo, hi = self.angle_range
        width = float(hi) - float(lo)
        if width <= 0:
            return pos
        return ((pos - lo) % width) + lo

    def get_position(self) -> float:
        pos = float(self._call(self._stage.get_position))
        pos = self._normalize_angle(pos)
        self._last_pos = pos
        return pos

    def get_position_cached(self) -> float:
        return self._normalize_angle(float(self._last_pos))

    def _normalize_angle(self, pos: float) -> float:
        lo, hi = self.angle_range
        width = float(hi) - float(lo)
        if width <= 0:
            return pos
        if pos < lo or pos > hi:
            return ((pos - lo) % width) + lo
        return pos

    def is_moving(self) -> bool:
        try:
            return bool(self._call(self._stage.is_moving))
        except Exception:
            return False

    def is_homed(self) -> Optional[bool]:
        try:
            return bool(self._call(self._stage.is_homed))
        except Exception:
            return None

    def set_velocity(self, *, accel: Optional[float] = None, max_vel: Optional[float] = None) -> None:
        def _set():
            try:
                min_v, acc, max_v = self._stage.get_velocity_parameters()
            except Exception:
                return
            if accel is not None:
                acc = float(accel)
            if max_vel is not None:
                max_v = float(max_vel)
            try:
                self._stage.set_velocity_parameters(min_v, acc, max_v)
            except Exception:
                pass

        self._call(_set)

    def home(self, controller: Optional[MotionController] = None, on_progress: Optional[Callable[[float], None]] = None) -> None:
        def _home():
            self._stage.home()
            if hasattr(self._stage, "wait_move"):
                self._stage.wait_move()

        self._call(_home)
        if on_progress:
            try:
                on_progress(self.get_position())
            except Exception:
                pass

    def stop(self) -> None:
        try:
            self._call(self._stage.stop)
        except Exception:
            pass

    def move_to(
        self,
        target_deg: float,
        *,
        step_deg: float = DEFAULT_RAMP_STEP_DEG,
        accel: Optional[float] = None,
        controller: Optional[MotionController] = None,
        on_step: Optional[Callable[[float], None]] = None,
    ) -> float:
        target = _clamp_angle(float(target_deg), self.angle_range)
        if accel is not None:
            self.set_velocity(accel=accel)

        try:
            start = float(self.get_position())
        except Exception:
            start = float(self._last_pos) if not math.isnan(self._last_pos) else target

        step = float(abs(step_deg))
        if step <= 0:
            step = abs(target - start)

        lo, hi = self.angle_range
        width = float(hi) - float(lo)
        if self._wrap and width > 0:
            start = self._wrap_angle(start)
            target = self._wrap_angle(target)
            delta = target - start
            if delta > 0:
                alt_delta = delta - width
            elif delta < 0:
                alt_delta = delta + width
            else:
                alt_delta = 0.0
            use_delta = alt_delta if abs(alt_delta) < abs(delta) else delta

            if abs(use_delta) <= step + 1e-12:
                self._call(self._stage.move_to, target)
                if hasattr(self._stage, "wait_move"):
                    self._call(self._stage.wait_move)
                if on_step:
                    try:
                        on_step(self.get_position())
                    except Exception:
                        pass
                return target

            direction = 1.0 if use_delta > 0 else -1.0
            remaining = abs(use_delta)
            pos = start
            while remaining > step + 1e-12:
                if controller is not None and controller.is_aborted():
                    break
                pos = self._wrap_angle(pos + direction * step)
                if hasattr(self._stage, "move_by"):
                    self._call(self._stage.move_by, direction * step)
                else:
                    self._call(self._stage.move_to, pos)
                if hasattr(self._stage, "wait_move"):
                    self._call(self._stage.wait_move)
                if on_step:
                    try:
                        on_step(self.get_position())
                    except Exception:
                        pass
                remaining -= step

            if controller is None or not controller.is_aborted():
                final = self._wrap_angle(start + use_delta)
                if hasattr(self._stage, "move_by"):
                    self._call(self._stage.move_by, direction * remaining)
                else:
                    self._call(self._stage.move_to, final)
                if hasattr(self._stage, "wait_move"):
                    self._call(self._stage.wait_move)
                if on_step:
                    try:
                        on_step(self.get_position())
                    except Exception:
                        pass
            return self._wrap_angle(start + use_delta)

        if abs(target - start) <= step + 1e-12:
            self._call(self._stage.move_to, target)
            if hasattr(self._stage, "wait_move"):
                self._call(self._stage.wait_move)
            if on_step:
                try:
                    on_step(self.get_position())
                except Exception:
                    pass
            return target

        direction = 1.0 if target > start else -1.0
        pos = start

        while (direction > 0 and pos < target - 1e-12) or (direction < 0 and pos > target + 1e-12):
            if controller is not None and controller.is_aborted():
                break
            pos = pos + direction * step
            if (direction > 0 and pos > target) or (direction < 0 and pos < target):
                pos = target
            self._call(self._stage.move_to, pos)
            if hasattr(self._stage, "wait_move"):
                self._call(self._stage.wait_move)
            if on_step:
                try:
                    on_step(self.get_position())
                except Exception:
                    pass

        return target

    def move_by(
        self,
        delta_deg: float,
        *,
        step_deg: float = DEFAULT_RAMP_STEP_DEG,
        accel: Optional[float] = None,
        controller: Optional[MotionController] = None,
        on_step: Optional[Callable[[float], None]] = None,
    ) -> float:
        start = float(self.get_position())
        return self.move_to(
            start + float(delta_deg),
            step_deg=step_deg,
            accel=accel,
            controller=controller,
            on_step=on_step,
        )

    def jog(
        self,
        step_deg: float,
        direction: int,
        *,
        accel: Optional[float] = None,
        controller: Optional[MotionController] = None,
        on_step: Optional[Callable[[float], None]] = None,
    ) -> float:
        step = float(abs(step_deg))
        delta = step if direction >= 0 else -step
        return self.move_by(delta, step_deg=step, accel=accel, controller=controller, on_step=on_step)

    def sweep(
        self,
        cfg: SweepConfig,
        controller: Optional[MotionController] = None,
        on_step: Optional[Callable[[int, int, float], None]] = None,
        on_ramp_step: Optional[Callable[[float], None]] = None,
    ) -> None:
        pts = _gen_points(cfg.start_deg, cfg.stop_deg, cfg.step_deg, self.angle_range)
        if not pts:
            raise RotationStageError("No sweep points generated")

        max_step = float(abs(cfg.ramp_step_deg))
        for idx, target in enumerate(pts, start=1):
            if controller is not None and controller.is_aborted():
                break
            use_step = max_step if abs(cfg.step_deg) > max_step > 0 else abs(cfg.step_deg)
            self.move_to(
                target,
                step_deg=use_step,
                controller=controller,
                on_step=on_ramp_step if abs(cfg.step_deg) > max_step > 0 else None,
            )
            if cfg.settle_s > 0:
                time.sleep(cfg.settle_s)
            if cfg.dwell_s > 0:
                time.sleep(cfg.dwell_s)
            if on_step:
                try:
                    on_step(idx, len(pts), float(target))
                except Exception:
                    pass

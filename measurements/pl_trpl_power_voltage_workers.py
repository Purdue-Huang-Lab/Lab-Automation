from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from keithley.keithley_wrapper import ComplianceTrip, KeithleySMU, SweepController, VIReading
from measurements.dual_wheel_power_calibration import DualWheelPowerEntry
from measurements.power_calibration import PowerCalibEntry
from rot.rot_wrapper import MotionController
from spad23.spad23_tcspc_wrapper import Spad23TcspcClient, TrplSnapshot
from spad23.spad23_wrapper import CountSnapshot, Spad23CountClient


class GateRampThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, float)  # V, I(A) at every microstep
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)  # ok/aborted/error:...

    def __init__(
        self,
        dev: KeithleySMU,
        target_v: float,
        icomp_a: Optional[float],
        controller: SweepController,
        ramp_step_v: float,
        ramp_dwell_s: float,
        parent=None,
    ):
        super().__init__(parent)
        self.dev = dev
        self.target_v = float(target_v)
        self.icomp_a = icomp_a
        self.controller = controller
        self.ramp_step_v = float(ramp_step_v)
        self.ramp_dwell_s = float(ramp_dwell_s)

    def run(self):
        try:
            if self.icomp_a is not None:
                self.dev.set_compliance(float(self.icomp_a))
            self.dev.set_output(True)
            self.status.emit(f"Ramping to {self.target_v:.6g} V...")

            def on_microstep(vi: VIReading):
                self.progress.emit(vi.v, vi.i)

            vi_last = self.dev.ramp_to_voltage(
                self.target_v,
                ramp_step_v=self.ramp_step_v,
                ramp_dwell_s=self.ramp_dwell_s,
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


class SpadHistogramThread(QtCore.QThread):
    update = QtCore.pyqtSignal(object)
    done = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, host: str, port: int, measurement_ms: int, bin_width_ps: int, parent=None):
        super().__init__(parent)
        self.host = str(host)
        self.port = int(port)
        self.measurement_ms = int(measurement_ms)
        self.bin_width_ps = int(bin_width_ps)

    def run(self):
        try:
            with Spad23TcspcClient(host=self.host, port=self.port) as client:
                snapshot = client.acquire_trpl_stream(
                    measurement_ms=self.measurement_ms,
                    bin_width_ps=self.bin_width_ps,
                    on_update=lambda s: self.update.emit(s),
                    update_interval_s=0.4,
                )
            self.done.emit(snapshot)
        except Exception as exc:
            self.error.emit(str(exc))


class PowerVoltageSweepThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    vi_ready = QtCore.pyqtSignal(object)
    spad_count_ready = QtCore.pyqtSignal(object)
    trpl_update = QtCore.pyqtSignal(object)
    trpl_done = QtCore.pyqtSignal(object)
    point_done = QtCore.pyqtSignal(object)
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
        *,
        sweep_power: bool,
        sweep_voltage: bool,
        sweep_mode: str,
        front_dev: Optional[KeithleySMU],
        back_dev: Optional[KeithleySMU],
        front_v0: float,
        front_step: float,
        back_v0: float,
        back_step: float,
        nsteps: int,
        front_icomp_a: Optional[float],
        back_icomp_a: Optional[float],
        ramp_step_v: float,
        ramp_dwell_s: float,
        settle_ms: float,
        zero_v_eps: float,
        zero_v_extra_settle_ms: float,
        save_dir: Optional[str],
        tcspc_host: str,
        tcspc_port: int,
        tcspc_acq_s: float,
        tcspc_bin_ps: int,
        spad_count_integration_ms: int = 120,
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
        self.sweep_power = bool(sweep_power)
        self.sweep_voltage = bool(sweep_voltage)
        self.sweep_mode = sweep_mode
        self.front_dev = front_dev
        self.back_dev = back_dev
        self.front_v0 = float(front_v0)
        self.front_step = float(front_step)
        self.back_v0 = float(back_v0)
        self.back_step = float(back_step)
        self.nsteps = int(nsteps)
        self.front_icomp_a = front_icomp_a
        self.back_icomp_a = back_icomp_a
        self.ramp_step_v = float(ramp_step_v)
        self.ramp_dwell_s = float(ramp_dwell_s)
        self.settle_ms = float(settle_ms)
        self.zero_v_eps = float(zero_v_eps)
        self.zero_v_extra_settle_ms = float(zero_v_extra_settle_ms)

        self.save_dir = save_dir
        self.tcspc_host = str(tcspc_host)
        self.tcspc_port = int(tcspc_port)
        self.tcspc_acq_s = float(tcspc_acq_s)
        self.tcspc_bin_ps = int(tcspc_bin_ps)
        self.spad_count_integration_ms = max(20, int(spad_count_integration_ms))
        self._stop = False
        self._stage_controller = MotionController()
        self._gate_controller = SweepController()
        self._file_counters: Dict[str, int] = {}

    def _save_trpl_csv(self, base_step: str, snapshot: TrplSnapshot) -> Optional[str]:
        if not self.save_dir:
            return None
        base = f"TRPL_{base_step}"
        path = self._next_filename(base, ext=".csv")
        n_bins = int(snapshot.counts_matrix.shape[0])
        time_ns = (np.arange(n_bins, dtype=np.float64) * float(snapshot.bin_width_ps)) / 1000.0
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_ns"] + [f"pixel_{i}" for i in range(snapshot.counts_matrix.shape[1])])
                for i in range(n_bins):
                    writer.writerow([f"{time_ns[i]:.6f}"] + snapshot.counts_matrix[i, :].astype(np.int64).tolist())
        except Exception:
            return None
        return path

    def stop(self):
        self._stop = True
        try:
            self._gate_controller.abort()
        except Exception:
            pass
        try:
            self._stage_controller.abort()
        except Exception:
            pass
        for stage in (self.stage_a, self.stage_b):
            try:
                if stage is not None:
                    stage.stop()
            except Exception:
                pass

    def _next_filename(self, base: str, ext: str = ".asc") -> str:
        idx = self._file_counters.get(base, 0)
        while True:
            suffix = "" if idx == 0 else f"_{idx}"
            name = f"{base}{suffix}{ext}"
            path = os.path.join(self.save_dir or ".", name)
            if not os.path.exists(path):
                self._file_counters[base] = idx + 1
                return path
            idx += 1

    @staticmethod
    def _fmt_v_label(v: Optional[float]) -> str:
        if v is None:
            return "na"
        s = f"{v:.6g}"
        return s.replace("-", "m").replace(".", "p")

    def _gate_ramp(self, dev: KeithleySMU, target_v: float, icomp_a: Optional[float]) -> VIReading:
        tag = self._gate_tag(dev)
        for attempt in range(2):
            try:
                if icomp_a is not None:
                    dev.set_compliance(float(icomp_a))
                dev.set_output(True)
                return dev.ramp_to_voltage(
                    target_v,
                    ramp_step_v=self.ramp_step_v,
                    ramp_dwell_s=self.ramp_dwell_s,
                    controller=self._gate_controller,
                    micro_read=False,
                    on_microstep=None,
                    verify=True,
                )
            except Exception as exc:
                if attempt == 0 and self._recover_gate_device(dev, icomp_a=icomp_a, tag=tag):
                    self.status.emit(f"{tag} gate recovered; retrying ramp")
                    continue
                raise exc

    def _read_vi(
        self,
        dev: Optional[KeithleySMU],
        *,
        target_v: Optional[float] = None,
        icomp_a: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        if dev is None:
            return (None, None)

        tag = self._gate_tag(dev)
        for attempt in range(2):
            try:
                vi = dev.read_vi()
                return (float(vi.v), float(vi.i))
            except Exception as exc:
                if attempt == 0 and self._recover_gate_device(dev, icomp_a=icomp_a, tag=tag):
                    if target_v is not None:
                        self.status.emit(f"{tag} gate recovered; restoring {float(target_v):.6g} V")
                        self._gate_ramp(dev, float(target_v), icomp_a)
                    self.status.emit(f"{tag} gate recovered; retrying read")
                    continue
                raise exc

        return (None, None)

    def _gate_tag(self, dev: Optional[KeithleySMU]) -> str:
        if dev is self.front_dev:
            return "Front"
        if dev is self.back_dev:
            return "Back"
        return "Gate"

    def _recover_gate_device(self, dev: Optional[KeithleySMU], *, icomp_a: Optional[float], tag: str) -> bool:
        if dev is None:
            return False
        try:
            self.status.emit(f"{tag} gate comm lost; reconnecting...")
            try:
                dev.close()
            except Exception:
                pass
            dev.open()
            if icomp_a is not None:
                dev.set_compliance(float(icomp_a))
            dev.set_output(True)
            return True
        except Exception as exc:
            self.status.emit(f"{tag} gate reconnect failed: {exc}")
            return False

    def _check_trip(self, i_a: Optional[float], limit_a: Optional[float], tag: str) -> None:
        if i_a is None or limit_a is None:
            return
        if abs(float(i_a)) > abs(float(limit_a)) + 1e-15:
            msg = f"{tag} TRIP: |I|={abs(i_a)*1e9:.6g} nA > {abs(limit_a)*1e9:.6g} nA"
            raise ComplianceTrip(msg, float(i_a))

    def _acquire_image(self, entry, v_front: Optional[float], v_back: Optional[float], i_front: Optional[float], i_back: Optional[float], base: str):
        accum_n = max(1, int(self.accum_n))
        sum_img = None
        last_fr = None

        for j in range(1, accum_n + 1):
            if self._stop or self._gate_controller.is_aborted():
                return None, None
            fr, err = self._grab_frame()
            if err:
                if err == "aborted":
                    return None, "aborted"
                return None, err
            img = fr.get("image")
            if img is None:
                return None, "error: acquire failed (no image)"
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
            return None, "error: acquire failed (no frames)"

        if base:
            filepath = self._next_filename(base)
            meta = dict(self.metadata_base)
            if entry is not None:
                if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
                    meta.update({
                        "power_w": entry.power_w,
                        "a_deg": entry.a_deg,
                        "b_deg": entry.b_deg,
                    })
                else:
                    meta.update({
                        "series": entry.series,
                        "power_w": entry.power_w,
                        "position_deg": entry.position_deg,
                    })
            meta.update({
                "front_v_set": v_front,
                "back_v_set": v_back,
                "front_i_read": i_front,
                "back_i_read": i_back,
                "sweep_mode": self.sweep_mode,
            })
            try:
                self.cam.save_ascii(filepath, sum_img, metadata=meta, wavelength_axis_nm=self.wavelength_axis_nm)
            except Exception as exc:
                return None, f"error: save failed: {exc}"
        else:
            filepath = None

        return {
            "image": sum_img,
            "accum_n": accum_n,
            "filepath": filepath,
        }, None

    def _grab_frame(self):
        retries = 1
        while True:
            if self._stop or self._gate_controller.is_aborted():
                return None, "aborted"
            try:
                fr = self.cam.get_frame()
            except Exception as exc:
                self.status.emit(f"Acquire failed: {exc}")
                return None, f"error: acquire failed: {exc}"
            if fr.get("ok"):
                return fr, None
            err = str(fr.get("err", "Acquire failed"))
            if self._stop:
                return None, "aborted"
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
            return None, f"error: acquire failed: {err}"


    def _is_zero_step(self, v_front: Optional[float], v_back: Optional[float]) -> bool:
        eps = abs(float(self.zero_v_eps))
        if v_front is not None and abs(float(v_front)) <= eps:
            return True
        if v_back is not None and abs(float(v_back)) <= eps:
            return True
        return False

    def _iter_voltage_steps(self, reverse: bool = False):
        if not self.sweep_voltage:
            return [(0, None, None)]
        steps = []
        n = max(1, int(self.nsteps))
        indices = range(n - 1, -1, -1) if reverse else range(n)
        for k in indices:
            vf = None
            vb = None
            if self.sweep_mode in ("front", "dual"):
                vf = float(self.front_v0 + k * self.front_step)
            if self.sweep_mode in ("back", "dual"):
                vb = float(self.back_v0 + k * self.back_step)
            steps.append((k, vf, vb))
        return steps

    def run(self):
        prev_timeout_scale = None
        try:
            if self.cam is None:
                self.status.emit("Sweep error: camera not connected")
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

            power_plan = self.plan if self.sweep_power else [{"entry": None, "base": "PL_manual"}]
            snake_voltage = bool(self.sweep_power and self.sweep_voltage and max(1, int(self.nsteps)) > 1)
            sweep_forward = True

            for idx, item in enumerate(power_plan, start=1):
                if self._stop or self._gate_controller.is_aborted():
                    self.done.emit("aborted")
                    return

                entry = item.get("entry")
                base = item.get("base") or "PL"

                if self.sweep_power and entry is not None:
                    if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
                        if self.stage_a is None or self.stage_b is None:
                            self.done.emit("error: stage not connected")
                            return
                        self.status.emit(
                            f"Move to A={entry.a_deg:.3f}, B={entry.b_deg:.3f} deg ({idx}/{len(power_plan)})"
                        )
                        try:
                            self.stage_a.move_to(entry.a_deg, step_deg=self.ramp_step_deg, controller=self._stage_controller)
                            self.stage_b.move_to(entry.b_deg, step_deg=self.ramp_step_deg, controller=self._stage_controller)
                        except Exception as exc:
                            self.status.emit(f"Stage move failed: {exc}")
                            self.done.emit(f"error: stage move failed: {exc}")
                            return
                    else:
                        if self.stage_a is None:
                            self.done.emit("error: stage not connected")
                            return
                        self.status.emit(f"Move to {entry.position_deg:.3f} deg ({idx}/{len(power_plan)})")
                        try:
                            self.stage_a.move_to(entry.position_deg, step_deg=self.ramp_step_deg, controller=self._stage_controller)
                        except Exception as exc:
                            self.status.emit(f"Stage move failed: {exc}")
                            self.done.emit(f"error: stage move failed: {exc}")
                            return

                if snake_voltage:
                    direction = "forward" if sweep_forward else "reverse"
                    self.status.emit(f"Voltage sweep ({direction})")
                reverse_steps = bool(snake_voltage and not sweep_forward)
                for step_idx, v_front, v_back in self._iter_voltage_steps(reverse=reverse_steps):
                    if self._stop or self._gate_controller.is_aborted():
                        self.done.emit("aborted")
                        return

                    if self.sweep_voltage:
                        if v_front is not None and self.front_dev is not None:
                            self._gate_ramp(self.front_dev, v_front, self.front_icomp_a)
                        if v_back is not None and self.back_dev is not None:
                            self._gate_ramp(self.back_dev, v_back, self.back_icomp_a)

                        if self.settle_ms > 0:
                            time.sleep(max(0.0, self.settle_ms / 1000.0))

                    v_read_f, i_read_f = (
                        self._read_vi(
                            self.front_dev,
                            target_v=v_front,
                            icomp_a=self.front_icomp_a,
                        )
                        if self.front_dev is not None
                        else (None, None)
                    )
                    v_read_b, i_read_b = (
                        self._read_vi(
                            self.back_dev,
                            target_v=v_back,
                            icomp_a=self.back_icomp_a,
                        )
                        if self.back_dev is not None
                        else (None, None)
                    )

                    self.vi_ready.emit({
                        "v_front": v_read_f,
                        "i_front": i_read_f,
                        "v_back": v_read_b,
                        "i_back": i_read_b,
                        "step_idx": step_idx,
                        "entry": entry,
                    })

                    if self._is_zero_step(v_front, v_back):
                        if self.zero_v_extra_settle_ms > 0:
                            time.sleep(max(0.0, self.zero_v_extra_settle_ms / 1000.0))

                    try:
                        self._check_trip(i_read_f, self.front_icomp_a, "Front")
                        self._check_trip(i_read_b, self.back_icomp_a, "Back")
                    except ComplianceTrip:
                        self.done.emit("trip")
                        return

                    voltage_suffix = ""
                    if self.sweep_voltage:
                        if self.sweep_mode == "front":
                            voltage_suffix = f"_Vf{self._fmt_v_label(v_front)}"
                        elif self.sweep_mode == "back":
                            voltage_suffix = f"_Vb{self._fmt_v_label(v_back)}"
                        else:
                            voltage_suffix = f"_Vf{self._fmt_v_label(v_front)}_Vb{self._fmt_v_label(v_back)}"

                    base_step = f"{base}{voltage_suffix}" if voltage_suffix else base
                    result, err = self._acquire_image(entry, v_read_f, v_read_b, i_read_f, i_read_b, base_step)
                    if err:
                        if err == "aborted":
                            self.done.emit("aborted")
                        else:
                            self.status.emit(err)
                            self.done.emit(err)
                        return

                    voltage_key = None
                    if self.sweep_mode == "front":
                        voltage_key = v_front
                    elif self.sweep_mode == "back":
                        voltage_key = v_back
                    elif self.sweep_mode == "dual":
                        voltage_key = v_front

                    payload = {
                        "entry": entry,
                        "image": result["image"],
                        "accum_n": result["accum_n"],
                        "filepath": result["filepath"],
                        "step_idx": step_idx,
                        "voltage_key": voltage_key,
                        "v_front": v_read_f,
                        "i_front": i_read_f,
                        "v_back": v_read_b,
                        "i_back": i_read_b,
                    }
                    self.point_done.emit(payload)

                    # Quick SPAD count snapshot for map refresh.
                    try:
                        with Spad23CountClient(host=self.tcspc_host, port=self.tcspc_port) as count_client:
                            c_snap: CountSnapshot = count_client.get_counts_snapshot(
                                integration_ms=self.spad_count_integration_ms
                            )
                        self.spad_count_ready.emit(
                            {
                                "snapshot": c_snap,
                                "entry": entry,
                                "step_idx": step_idx,
                                "voltage_key": voltage_key,
                            }
                        )
                    except Exception as exc:
                        self.status.emit(f"SPAD quick count failed: {exc}")

                    # TCSPC histogram acquisition for this sweep point.
                    measurement_ms = max(1, int(round(self.tcspc_acq_s * 1000.0)))
                    bin_ps = max(1, int(self.tcspc_bin_ps))
                    self.status.emit(
                        f"TRPL acquire @ point (t={self.tcspc_acq_s:.3g}s, bin={bin_ps} ps)..."
                    )
                    try:
                        with Spad23TcspcClient(host=self.tcspc_host, port=self.tcspc_port) as tcspc_client:
                            trpl_snapshot = tcspc_client.acquire_trpl_stream(
                                measurement_ms=measurement_ms,
                                bin_width_ps=bin_ps,
                                on_update=lambda s, e=entry, si=step_idx, vk=voltage_key: self.trpl_update.emit(
                                    {
                                        "snapshot": s,
                                        "entry": e,
                                        "step_idx": si,
                                        "voltage_key": vk,
                                    }
                                ),
                                update_interval_s=0.4,
                            )
                    except Exception as exc:
                        self.done.emit(f"error: TRPL acquire failed: {exc}")
                        return

                    trpl_path = self._save_trpl_csv(base_step, trpl_snapshot)
                    self.trpl_done.emit(
                        {
                            "snapshot": trpl_snapshot,
                            "entry": entry,
                            "step_idx": step_idx,
                            "voltage_key": voltage_key,
                            "filepath": trpl_path,
                        }
                    )

                if self.sweep_power and entry is not None:
                    self.power_done.emit({"entry": entry})
                if snake_voltage:
                    sweep_forward = not sweep_forward

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

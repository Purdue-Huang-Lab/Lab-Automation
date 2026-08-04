# keithley_wrapper.py
# Clean wrapper for Keithley 2400/2450-family SMUs (voltage source use-case).
#
# Key behaviors:
# - Safe ramped voltage changes (default 10 mV step, 100 ms dwell)
# - During ramp/home: read V&I after every microstep via callback (for plotting)
# - During sweeps: still ramp safely, but only report macro-step readings
# - Separate read_voltage/read_current APIs exist, but GUI should prefer read_vi()
#
# Requires: pyvisa
#
# Notes:
# - PyVISA resource attributes like timeout and query_delay are documented and can be used
#   to reduce socket pacing issues on some instruments. (See PyVISA docs.)  :contentReference[oaicite:2]{index=2}
# - Keithley 2450 user manual examples include commands like SENS:FUNC "CURR", SOUR:FUNC VOLT, SOUR:VOLT ... :contentReference[oaicite:3]{index=3}

from __future__ import annotations

import math
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable

import pyvisa


# -------------------- Defaults & Safety --------------------

KEITHLEY_V_LIMIT = 210.0  # absolute max |V| enforced in wrapper

# verification after reaching a target (macro-level)
VERIFY_TOL_FRACTION = 0.005   # 0.5% of setpoint
VERIFY_TOL_ABS = 2e-3         # 2 mV floor
VERIFY_ATTEMPTS = 6
VERIFY_SLEEP_S = 0.20

# ramp defaults (your requested values)
DEFAULT_RAMP_STEP_V = 0.01    # 10 mV
DEFAULT_RAMP_DWELL_S = 0.10   # 100 ms

# sweep settle after each macro step before measuring (ms)
DEFAULT_SWEEP_SETTLE_MS = 100

DEFAULT_DRAIN_ERR_READS = 8


# -------------------- Exceptions --------------------

class KeithleyError(RuntimeError):
    pass


class ComplianceTrip(RuntimeError):
    """Raised when |I| exceeds the requested sweep compliance limit."""
    def __init__(self, message: str, current_a: float):
        super().__init__(message)
        self.current_a = current_a


# -------------------- Helper Types --------------------

@dataclass(frozen=True)
class VIReading:
    v: float
    i: float  # amps


@dataclass(frozen=True)
class SweepConfig:
    v0: float
    step: float
    nsteps: int
    icomp_limit_a: float  # absolute limit used for sweep trip check (macro only)
    manual: bool = True
    dwell_ms: int = 0     # used only if manual=False (time between steps)
    settle_ms: int = DEFAULT_SWEEP_SETTLE_MS  # wait after reaching Vk before measuring
    ramp_step_v: float = DEFAULT_RAMP_STEP_V
    ramp_dwell_s: float = DEFAULT_RAMP_DWELL_S


@dataclass(frozen=True)
class DualSweepConfig:
    a: SweepConfig
    b: SweepConfig
    nsteps: int
    manual: bool = True
    dwell_ms: int = 0
    ramp_step_v: float = DEFAULT_RAMP_STEP_V
    ramp_dwell_s: float = DEFAULT_RAMP_DWELL_S


class SweepController:
    """
    Thread-safe control for manual stepping and abort.
    - GUI calls next_step() or abort()
    - sweep/ramp/home functions can check is_aborted() and/or wait_next().
    """
    def __init__(self):
        self._abort = threading.Event()
        self._step = threading.Event()

    def abort(self) -> None:
        self._abort.set()
        self._step.set()  # release any wait

    def next_step(self) -> None:
        self._step.set()

    def is_aborted(self) -> bool:
        return self._abort.is_set()

    def wait_next(self, manual: bool, dwell_ms: int) -> None:
        if self._abort.is_set():
            return
        if manual:
            self._step.clear()
            while (not self._abort.is_set()) and (not self._step.is_set()):
                time.sleep(0.02)
        else:
            time.sleep(max(0.0, float(dwell_ms) / 1000.0))


# -------------------- ResourceManager helper --------------------

def make_resource_manager(visa_dll: str = "") -> pyvisa.ResourceManager:
    if visa_dll and os.path.exists(visa_dll):
        return pyvisa.ResourceManager(visa_dll)
    return pyvisa.ResourceManager()


# -------------------- Core Instrument Wrapper --------------------

class KeithleySMU:
    """
    Minimal Keithley SMU helper supporting 2400 and 2450-ish command sets for:
    - voltage sourcing
    - V/I measurement
    - output control
    - compliance setup
    - safe ramped voltage changes

    This class serializes VISA I/O; do not call concurrently without the lock.
    """
    def __init__(
        self,
        rm: pyvisa.ResourceManager,
        resource: str,
        timeout_ms: int = 20000,   # increased default to reduce socket timeouts during long ramps
        query_delay_s: float = 0.0,
        verbose: bool = False,
    ):
        self.rm = rm
        self.resource = resource
        self.timeout_ms = int(timeout_ms)
        self.query_delay_s = float(query_delay_s)
        self.verbose = bool(verbose)

        self.inst: Optional[pyvisa.resources.Resource] = None
        self.idn: str = ""
        self.is_2400: bool = False

        self._io_lock = threading.RLock()
        self._last_set_v: float = 0.0

    # ---- I/O primitives ----

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[KeithleySMU] {msg}")

    def open(self) -> str:
        self.close()
        try:
            self.inst = self.rm.open_resource(self.resource)

            # clear if supported
            try:
                self.inst.clear()
            except Exception:
                pass

            # terminations for socket-like resources
            try:
                self.inst.read_termination = "\n"
            except Exception:
                pass
            try:
                self.inst.write_termination = "\n"
            except Exception:
                pass

            # timeout (PyVISA documents timeout attr on resources) :contentReference[oaicite:4]{index=4}
            try:
                self.inst.timeout = self.timeout_ms
            except Exception:
                pass

            # optional pacing between write and read in query() (PyVISA query_delay) :contentReference[oaicite:5]{index=5}
            try:
                if self.query_delay_s > 0:
                    self.inst.query_delay = self.query_delay_s
            except Exception:
                pass

            self.write("*CLS")
            try:
                self.drain_errors()
            except Exception:
                pass

            self.idn = self.query("*IDN?").strip()
            self.is_2400 = ("2400" in self.idn.upper())
            self._log(f"Opened: {self.idn}")

            if self.is_2400:
                try:
                    self.write(":SYST:REM")
                except Exception:
                    pass

            self.init_voltage_source()
            try:
                self.drain_errors()
            except Exception:
                pass

            return self.idn
        except Exception as e:
            self.close()
            raise KeithleyError(f"Failed to open {self.resource}: {e}") from e

    def close(self) -> None:
        try:
            if self.inst is not None:
                try:
                    self.drain_errors()
                except Exception:
                    pass
                if self.is_2400:
                    try:
                        self.write(":SYST:LOC")
                    except Exception:
                        pass
                try:
                    self.inst.close()
                except Exception:
                    pass
        finally:
            self.inst = None

    def write(self, cmd: str) -> None:
        if self.inst is None:
            raise KeithleyError("Instrument not open")
        with self._io_lock:
            self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self.inst is None:
            raise KeithleyError("Instrument not open")
        with self._io_lock:
            return self.inst.query(cmd)

    def _try_write(self, *cmds: str) -> bool:
        for cmd in cmds:
            try:
                self.write(cmd)
                return True
            except Exception:
                pass
        return False

    def _query_float(self, cmd: str) -> float:
        return float(self.query(cmd).strip())

    # ---- Error queue ----

    def drain_errors(self, max_reads: int = DEFAULT_DRAIN_ERR_READS) -> list[str]:
        errs: list[str] = []
        for _ in range(max_reads):
            try:
                s = self.query(":SYST:ERR?").strip()
            except Exception:
                break
            errs.append(s)
            if s.startswith("0"):
                break
        return errs

    # ---- Basic configuration ----

    def init_voltage_source(self) -> None:
        # Voltage source function + auto-range so any voltage within spec is reachable
        self._try_write(":SOUR:FUNC VOLT", "SOUR:FUNC VOLT")
        self._try_write(":SOUR:VOLT:RANG:AUTO ON", "SOUR:VOLT:RANG:AUTO ON")

        if self.is_2400:
            # measure current
            self._try_write(':SENS:FUNC "CURR"', 'SENS:FUNC "CURR"')
            self._try_write(":SENS:CURR:RANG:AUTO ON", "SENS:CURR:RANG:AUTO ON")

            # make READ? return VOLT,CURR
            self._try_write(":FORM:ELEM VOLT,CURR", "FORM:ELEM VOLT,CURR")

            try:
                self.write(":TRIG:COUN 1")
            except Exception:
                pass

        else:
            self._try_write('SENS:FUNC "CURR"')
            self._try_write("SENS:CURR:RANG:AUTO ON")
            self._try_write("SOUR:FUNC VOLT")
            self._try_write(":SENS:CURR:RANG:AUTO ON")

        # Seed _last_set_v from the instrument's current source register
        # so ramps always start from the actual programmed voltage, not 0.
        try:
            self._last_set_v = float(self.query(":SOUR:VOLT?").strip())
        except Exception:
            self._last_set_v = 0.0

    # ---- Output & compliance ----

    def set_output(self, on: bool) -> None:
        if not self._try_write(f":OUTP {'ON' if on else 'OFF'}", f"OUTP {'ON' if on else 'OFF'}"):
            raise KeithleyError("Failed to set output state")

    def set_compliance(self, limit_amps: float) -> None:
        limit_amps = float(abs(limit_amps))
        if self.is_2400:
            if self._try_write(f":SENS:CURR:PROT {limit_amps:.6g}", f"SENS:CURR:PROT {limit_amps:.6g}"):
                return
        else:
            # 2450 typically supports ILIM; fall back to CURR:PROT if needed
            if self._try_write(f":SOUR:VOLT:ILIM {limit_amps:.6g}", f":SENS:CURR:PROT {limit_amps:.6g}"):
                return
        raise KeithleyError("Failed to set compliance/current limit")

    # ---- Reads ----

    def read_vi(self) -> VIReading:
        """
        Read both voltage and current.
        - 2400: single READ? returns "V,I,..." when configured with FORM:ELEM VOLT,CURR.
        - 2450: use separate measure queries.
        """
        if self.is_2400:
            ans = self.query("READ?").strip()
            parts = [p.strip() for p in ans.split(",")]
            v = float(parts[0]) if len(parts) >= 1 and parts[0] else float("nan")
            i = float(parts[1]) if len(parts) >= 2 and parts[1] else float("nan")
            return VIReading(v=v, i=i)
        else:
            v = self._query_float(":MEAS:VOLT?")
            i = self._query_float(":MEAS:CURR?")
            return VIReading(v=v, i=i)

    def read_vi_with_timeout(self, timeout_ms: Optional[int]) -> VIReading:
        if timeout_ms is None:
            return self.read_vi()
        if self.inst is None:
            raise KeithleyError("Instrument not open")
        with self._io_lock:
            inst = self.inst
            prev_timeout = None
            try:
                prev_timeout = inst.timeout
                inst.timeout = int(timeout_ms)
            except Exception:
                prev_timeout = None
            try:
                if self.is_2400:
                    ans = inst.query("READ?").strip()
                    parts = [p.strip() for p in ans.split(",")]
                    v = float(parts[0]) if len(parts) >= 1 and parts[0] else float("nan")
                    i = float(parts[1]) if len(parts) >= 2 and parts[1] else float("nan")
                    return VIReading(v=v, i=i)
                v = float(inst.query(":MEAS:VOLT?").strip())
                i = float(inst.query(":MEAS:CURR?").strip())
                return VIReading(v=v, i=i)
            finally:
                if prev_timeout is not None:
                    try:
                        inst.timeout = prev_timeout
                    except Exception:
                        pass

    def read_voltage(self) -> float:
        # Keep API, but do not double-query on 2400: we measure VI once and split.
        return float(self.read_vi().v)

    def read_current(self) -> float:
        # Keep API, but do not double-query on 2400: we measure VI once and split.
        return float(self.read_vi().i)

    # ---- Voltage setting (ramped) ----

    def _verify_ok(self, v_set: float, v_read: float) -> bool:
        tol = max(VERIFY_TOL_ABS, VERIFY_TOL_FRACTION * abs(v_set))
        return (not math.isnan(v_read)) and (abs(v_read - v_set) <= tol)

    def _set_voltage_instant(self, volts: float) -> None:
        volts = float(volts)
        if abs(volts) > KEITHLEY_V_LIMIT + 1e-12:
            raise KeithleyError(f"Voltage limit exceeded: {volts} V (limit +/-{KEITHLEY_V_LIMIT} V)")
        if not self._try_write(f":SOUR:VOLT {volts:.9g}", f"SOUR:VOLT {volts:.9g}"):
            raise KeithleyError("Failed to set voltage")
        self._last_set_v = volts

    def ramp_to_voltage(
        self,
        target_v: float,
        *,
        ramp_step_v: float = DEFAULT_RAMP_STEP_V,
        ramp_dwell_s: float = DEFAULT_RAMP_DWELL_S,
        controller: Optional[SweepController] = None,
        micro_read: bool = True,
        on_microstep: Optional[Callable[[VIReading], None]] = None,
        verify: bool = True,
    ) -> VIReading:
        """
        Safe ramp to target voltage.
        If micro_read=True: after each microstep, sleep dwell then read V&I and emit on_microstep(vi).
        """
        target = float(target_v)
        if abs(target) > KEITHLEY_V_LIMIT + 1e-12:
            raise KeithleyError(f"Voltage limit exceeded: {target} V (limit +/-{KEITHLEY_V_LIMIT} V)")

        step_v = max(1e-6, float(abs(ramp_step_v)))
        dwell_s = max(0.0, float(ramp_dwell_s))

        # start estimate
        try:
            start = float(self.read_voltage())
            if math.isnan(start):
                start = self._last_set_v
        except Exception:
            start = self._last_set_v

        # already close
        if abs(target - start) <= step_v:
            self._set_voltage_instant(target)
            if dwell_s > 0:
                time.sleep(dwell_s)
            vi = self.read_vi() if micro_read else VIReading(v=target, i=float("nan"))
            if micro_read and on_microstep:
                try:
                    on_microstep(vi)
                except Exception:
                    pass
            return self._verify_after_set(target, last_vi=vi) if verify else vi

        direction = 1.0 if (target > start) else -1.0
        v = start

        n_full = int(abs(target - start) // step_v)

        last_vi = VIReading(v=v, i=float("nan"))

        for _ in range(n_full):
            if controller is not None and controller.is_aborted():
                return last_vi

            v = v + direction * step_v
            if (direction > 0 and v > target) or (direction < 0 and v < target):
                v = target

            self._set_voltage_instant(v)

            if dwell_s > 0:
                time.sleep(dwell_s)

            if micro_read:
                last_vi = self.read_vi()
                if on_microstep is not None:
                    try:
                        on_microstep(last_vi)
                    except Exception:
                        pass

        # final snap if needed
        if abs(v - target) > 1e-12:
            if controller is not None and controller.is_aborted():
                return last_vi
            self._set_voltage_instant(target)
            if dwell_s > 0:
                time.sleep(dwell_s)
            if micro_read:
                last_vi = self.read_vi()
                if on_microstep is not None:
                    try:
                        on_microstep(last_vi)
                    except Exception:
                        pass
            else:
                last_vi = VIReading(v=target, i=float("nan"))

        return self._verify_after_set(target, last_vi=last_vi) if verify else last_vi

    def set_voltage(
        self,
        volts: float,
        ramp_step_v: float = DEFAULT_RAMP_STEP_V,
        ramp_dwell_s: float = DEFAULT_RAMP_DWELL_S,
        verify: bool = True,
    ) -> VIReading:
        # Backward-compatible: ramp but do not emit microsteps (GUI uses ramp_to_voltage with callback)
        return self.ramp_to_voltage(
            volts,
            ramp_step_v=ramp_step_v,
            ramp_dwell_s=ramp_dwell_s,
            micro_read=False,
            on_microstep=None,
            verify=verify,
        )

    def _verify_after_set(self, target: float, last_vi: Optional[VIReading] = None) -> VIReading:
        # Verify using VI reads so 2400 does not incur extra queries
        v_read = last_vi.v if last_vi else float("nan")
        i_read = last_vi.i if last_vi else float("nan")

        for _ in range(VERIFY_ATTEMPTS):
            vi = self.read_vi()
            v_read, i_read = vi.v, vi.i
            if self._verify_ok(target, v_read):
                return vi
            time.sleep(VERIFY_SLEEP_S)

        raise KeithleyError(f"Cannot reach setpoint: Vset={target:.6g} V, Vread={v_read:.6g} V")

    # ---- Home (ramp to 0) ----

    def home(
        self,
        *,
        controller: Optional[SweepController] = None,
        ramp_step_v: float = DEFAULT_RAMP_STEP_V,
        ramp_dwell_s: float = DEFAULT_RAMP_DWELL_S,
        tol_v: float = 2e-3,
        on_progress: Optional[Callable[[VIReading], None]] = None,
        verify: bool = False,
    ) -> str:
        """
        Ramp to 0 V; emits on_progress after every microstep (V&I).
        """
        tol_v = float(abs(tol_v))

        # If already close, still do a final set to 0 and emit one reading
        try:
            v_now = float(self.read_voltage())
            if math.isnan(v_now):
                v_now = self._last_set_v
        except Exception:
            v_now = self._last_set_v

        if abs(v_now) <= tol_v:
            vi0 = self.ramp_to_voltage(
                0.0,
                ramp_step_v=ramp_step_v,
                ramp_dwell_s=ramp_dwell_s,
                controller=controller,
                micro_read=True,
                on_microstep=on_progress,
                verify=verify,
            )
            return "aborted" if (controller and controller.is_aborted()) else "ok"

        self.ramp_to_voltage(
            0.0,
            ramp_step_v=ramp_step_v,
            ramp_dwell_s=ramp_dwell_s,
            controller=controller,
            micro_read=True,
            on_microstep=on_progress,
            verify=verify,
        )
        return "aborted" if (controller and controller.is_aborted()) else "ok"

    # ---- Sweep (single) ----

    def sweep_single(
        self,
        cfg: SweepConfig,
        controller: SweepController,
        on_progress: Optional[Callable[[int, VIReading], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_ramp_progress: Optional[Callable[[VIReading], None]] = None,
    ) -> str:
        """
        Manual (default) or auto sweep.

        Important: no microstep plotting during sweep steps.
        We ramp to V0 safely and can emit microstep readings there if on_ramp_progress is provided.
        We still ramp safely between macro voltages, but only emit macro readings.
        """
        v0 = float(cfg.v0)
        step = float(cfg.step)
        n = int(cfg.nsteps)
        if n < 1:
            raise KeithleyError("nsteps must be >= 1")

        vend = v0 + (n - 1) * step
        if abs(v0) > KEITHLEY_V_LIMIT + 1e-12 or abs(vend) > KEITHLEY_V_LIMIT + 1e-12:
            raise KeithleyError("Voltage limit exceeded by sweep endpoints")

        settle_s = max(0.0, float(cfg.settle_ms) / 1000.0)

        # ramp to V0 (safe, no micro reporting)
        if on_status:
            on_status("Ramping to V0...")
        self.ramp_to_voltage(
            v0,
            ramp_step_v=cfg.ramp_step_v,
            ramp_dwell_s=cfg.ramp_dwell_s,
            controller=controller,
            micro_read=on_ramp_progress is not None,
            on_microstep=on_ramp_progress,
            verify=True,
        )

        for k in range(n):
            if controller.is_aborted():
                return "aborted"

            # Wait for manual Next or auto dwell between steps
            if on_status:
                on_status(f"Waiting step {k+1}/{n}...")
            controller.wait_next(cfg.manual, cfg.dwell_ms)
            if controller.is_aborted():
                return "aborted"

            Vk = v0 + k * step

            # Ensure at Vk (safe ramp if needed, no micro reporting)
            self.ramp_to_voltage(
                Vk,
                ramp_step_v=cfg.ramp_step_v,
                ramp_dwell_s=cfg.ramp_dwell_s,
                controller=controller,
                micro_read=False,
                on_microstep=None,
                verify=True,
            )

            # settle then macro measure
            if settle_s > 0:
                time.sleep(settle_s)
            vi = self.read_vi()

            # macro-only trip check (your request)
            if (not math.isnan(vi.i)) and abs(vi.i) > abs(cfg.icomp_limit_a) + 1e-15:
                msg = f"Compliance TRIP: |I|={abs(vi.i)*1e9:.6g} nA > {abs(cfg.icomp_limit_a)*1e9:.6g} nA"
                if on_status:
                    on_status(msg)
                raise ComplianceTrip(msg, vi.i)

            if on_progress:
                on_progress(k + 1, vi)

        return "ok"


# -------------------- Dual sweep (module-level) --------------------

def sweep_dual(
    devA: KeithleySMU,
    devB: KeithleySMU,
    cfg: DualSweepConfig,
    controller: SweepController,
    on_progress: Optional[Callable[[int, VIReading, VIReading], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_ramp_progress_a: Optional[Callable[[VIReading], None]] = None,
    on_ramp_progress_b: Optional[Callable[[VIReading], None]] = None,
) -> str:
    """
    Dual sweep with common step count, per-device V0/step, and manual Next/Stop gating.
    No microstep plotting during sweep steps; macro readings only.
    If on_ramp_progress_a/b are provided, emit microsteps while ramping to V0.
    """
    n = int(cfg.nsteps)
    if n < 1:
        raise KeithleyError("nsteps must be >= 1")

    v0A, stepA = float(cfg.a.v0), float(cfg.a.step)
    v0B, stepB = float(cfg.b.v0), float(cfg.b.step)

    vendA = v0A + (n - 1) * stepA
    vendB = v0B + (n - 1) * stepB

    if abs(v0A) > KEITHLEY_V_LIMIT + 1e-12 or abs(vendA) > KEITHLEY_V_LIMIT + 1e-12:
        raise KeithleyError("A sweep endpoint exceeds voltage limit")
    if abs(v0B) > KEITHLEY_V_LIMIT + 1e-12 or abs(vendB) > KEITHLEY_V_LIMIT + 1e-12:
        raise KeithleyError("B sweep endpoint exceeds voltage limit")

    settle_s = max(0.0, float(cfg.a.settle_ms) / 1000.0)

    # ramp both to initial (safe)
    if on_status:
        on_status("Ramping both to V0...")
    devA.ramp_to_voltage(v0A, ramp_step_v=cfg.ramp_step_v, ramp_dwell_s=cfg.ramp_dwell_s,
                         controller=controller, micro_read=on_ramp_progress_a is not None,
                         on_microstep=on_ramp_progress_a, verify=True)
    devB.ramp_to_voltage(v0B, ramp_step_v=cfg.ramp_step_v, ramp_dwell_s=cfg.ramp_dwell_s,
                         controller=controller, micro_read=on_ramp_progress_b is not None,
                         on_microstep=on_ramp_progress_b, verify=True)

    for k in range(n):
        if controller.is_aborted():
            return "aborted"

        if on_status:
            on_status(f"Waiting step {k+1}/{n}...")
        controller.wait_next(cfg.manual, cfg.dwell_ms)
        if controller.is_aborted():
            return "aborted"

        VAk = v0A + k * stepA
        VBk = v0B + k * stepB

        devA.ramp_to_voltage(VAk, ramp_step_v=cfg.ramp_step_v, ramp_dwell_s=cfg.ramp_dwell_s,
                             controller=controller, micro_read=False, on_microstep=None, verify=True)
        devB.ramp_to_voltage(VBk, ramp_step_v=cfg.ramp_step_v, ramp_dwell_s=cfg.ramp_dwell_s,
                             controller=controller, micro_read=False, on_microstep=None, verify=True)

        if settle_s > 0:
            time.sleep(settle_s)

        viA = devA.read_vi()
        viB = devB.read_vi()

        if (not math.isnan(viA.i)) and abs(viA.i) > abs(cfg.a.icomp_limit_a) + 1e-15:
            msg = f"A TRIP: |I|={abs(viA.i)*1e9:.6g} nA > {abs(cfg.a.icomp_limit_a)*1e9:.6g} nA"
            if on_status:
                on_status(msg)
            raise ComplianceTrip(msg, viA.i)

        if (not math.isnan(viB.i)) and abs(viB.i) > abs(cfg.b.icomp_limit_a) + 1e-15:
            msg = f"B TRIP: |I|={abs(viB.i)*1e9:.6g} nA > {abs(cfg.b.icomp_limit_a)*1e9:.6g} nA"
            if on_status:
                on_status(msg)
            raise ComplianceTrip(msg, viB.i)

        if on_progress:
            on_progress(k + 1, viA, viB)

    return "ok"

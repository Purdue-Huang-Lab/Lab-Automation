from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    from pylablib.devices import Andor
except Exception as exc:
    Andor = None
    _IMPORT_ERR = exc


@dataclass
class CameraInfo:
    camera_name: str = "Andor"
    model_name: str = "unknown"
    serial_number: str = "unknown"
    firmware_version: str = "unknown"


class AndorSystem:
    """
    Andor SDK2 camera + Shamrock spectrograph wrapper.
    Designed for orchestration-friendly use with other devices.
    """

    def __init__(self, cam_index: int = 0, fan_mode: str = "low", verbose: bool = True):
        self.cam_index = int(cam_index)
        self._fan_mode = str(fan_mode)
        self.verbose = bool(verbose)

        self.cam = None
        self.spec = None

        self._lock = threading.RLock()
        self._frame_backend = "snap"
        self._streaming = False
        self._exposure_s = 0.05
        self._accum_n = 1
        self._acq_mode = "single"

        self._wavelength_axis_nm = None
        self._spec_index = 0
        self._spec_ready = False

        self._ts_window = []
        self._fps_smooth = None
        self._fps_alpha = 0.15
        self._last_wait_timeout_s = None
        self.last_img_wh = (None, None)
        self._timeout_scale = 1.0

    @contextmanager
    def _paused_acquisition(self, *, clear: bool = False):
        cam = getattr(self, "cam", None)
        if cam is None:
            yield
            return
        if hasattr(cam, "pausing_acquisition"):
            try:
                with cam.pausing_acquisition(clear=bool(clear), stop=True, start_after=True):
                    yield
                return
            except TypeError:
                with cam.pausing_acquisition():
                    yield
                return
        try:
            self.stop_stream()
        except Exception:
            pass
        if clear and hasattr(cam, "clear_acquisition"):
            try:
                cam.clear_acquisition()
            except Exception:
                pass
        yield

    # -----------------
    # Utilities
    # -----------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AndorSystem] {msg}")

    def _require_cam(self) -> None:
        if self.cam is None:
            raise RuntimeError("Camera not connected")

    @staticmethod
    def _preview8_from_u16(img16, p_lo: float = 1.0, p_hi: float = 99.0):
        if np is None:
            return None
        a = np.asarray(img16)
        if a.ndim != 2:
            a = np.squeeze(a)
            if a.ndim != 2:
                return None
        a = a.astype(np.float32, copy=False)
        lo = np.nanpercentile(a, p_lo)
        hi = np.nanpercentile(a, p_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
                return np.zeros(a.shape, dtype=np.uint8)
        x = (a - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    @staticmethod
    def _extract_image(r):
        if r is None:
            return None
        if isinstance(r, (tuple, list)):
            for item in r:
                try:
                    a = np.asarray(item) if np is not None else None
                except Exception:
                    continue
                if a is not None and getattr(a, "ndim", 0) == 2 and a.size > 0:
                    return a
            return None
        if np is None:
            return None
        try:
            a = np.asarray(r)
        except Exception:
            return None
        if getattr(a, "ndim", 0) == 2 and a.size > 0:
            return a
        return None

    def _dynamic_timeout_s(self) -> float:
        cam = self.cam
        timeout_s = None
        acq_mode = getattr(self, "_acq_mode", None)
        if acq_mode is None and cam is not None and hasattr(cam, "get_acquisition_mode"):
            try:
                acq_mode = str(cam.get_acquisition_mode())
            except Exception:
                acq_mode = None
        acq_mode_s = str(acq_mode or "").strip().lower()
        if "accum" in acq_mode_s:
            n_acc = int(getattr(self, "_accum_n", 1) or 1)
            cycle_time_acc = None
            try:
                if cam is not None and hasattr(cam, "get_accum_mode_parameters"):
                    par = cam.get_accum_mode_parameters()
                    if isinstance(par, (list, tuple)) and len(par) >= 2:
                        n_acc = int(par[0])
                        cycle_time_acc = float(par[1])
            except Exception:
                cycle_time_acc = None
            if cycle_time_acc and cycle_time_acc > 0:
                timeout_s = n_acc * cycle_time_acc
            else:
                exp_s = float(self._exposure_s or 0.05)
                timeout_s = n_acc * max(exp_s, 0.001)
            timeout_s = max(timeout_s * 1.2 + 0.2, 2.0)
            try:
                if cam is not None and hasattr(cam, "get_cycle_timings"):
                    t = cam.get_cycle_timings()
                    if t is not None:
                        cyc_acc = getattr(t, "accum_cycle_time", None)
                        if cyc_acc:
                            timeout_s = max(timeout_s, n_acc * float(cyc_acc) + 0.5)
            except Exception:
                pass
            try:
                if cam is not None and hasattr(cam, "get_readout_time"):
                    rt = cam.get_readout_time()
                    if rt:
                        exp_s = float(self._exposure_s or 0.05)
                        timeout_s = max(timeout_s, n_acc * (exp_s + float(rt)) + 0.5)
            except Exception:
                pass

        if timeout_s is None:
            try:
                if cam is not None and hasattr(cam, "get_cycle_timings"):
                    t = cam.get_cycle_timings()
                    if t is not None:
                        cyc = getattr(t, "kinetic_cycle_time", None) or getattr(t, "accum_cycle_time", None)
                        if cyc is not None:
                            timeout_s = float(cyc) * 2.0 + 0.2
                if timeout_s is None and cam is not None and hasattr(cam, "get_frame_timings"):
                    t = cam.get_frame_timings()
                    if isinstance(t, (list, tuple)) and len(t) >= 2 and t[1]:
                        timeout_s = float(t[1]) * 2.0 + 0.2
            except Exception:
                timeout_s = None
        if timeout_s is None:
            try:
                exp_s = float(self._exposure_s or 0.05)
            except Exception:
                exp_s = 0.05
            timeout_s = max(1.0, exp_s * 2.0 + 0.2)
        # Extend timeout to account for readout time, which the SDK-reported cycle
        # time may not include (seen on DU970P and similar CCDs).
        try:
            if cam is not None and hasattr(cam, "get_readout_time"):
                rt = cam.get_readout_time()
                if rt:
                    exp_s = float(self._exposure_s or 0.05)
                    timeout_s = max(timeout_s, float(rt) + exp_s + 0.5)
        except Exception:
            pass
        # Global minimum floor: always allow at least 2 s regardless of which
        # branch computed the timeout (prevents sub-200ms timeouts on short exposures).
        timeout_s = max(timeout_s, 2.0)
        try:
            scale = float(getattr(self, "_timeout_scale", 1.0) or 1.0)
        except Exception:
            scale = 1.0
        if scale < 1.0:
            scale = 1.0
        timeout_s *= scale
        self._last_wait_timeout_s = float(timeout_s)
        return float(timeout_s)

    def _recover_after_timeout(self) -> None:
        cam = self.cam
        if cam is None:
            return
        try:
            self.stop_stream()
        except Exception:
            pass
        for fn in ("abort_acquisition", "abort_acq", "abort", "stop_acquisition", "stop_acq", "stop"):
            if hasattr(cam, fn):
                try:
                    getattr(cam, fn)()
                    break
                except Exception:
                    pass
        for fn in ("clear_acquisition", "clear_acq", "clear"):
            if hasattr(cam, fn):
                try:
                    getattr(cam, fn)()
                    break
                except Exception:
                    pass
        try:
            if hasattr(cam, "set_trigger_mode"):
                cam.set_trigger_mode("int")
        except Exception:
            pass
        try:
            self.set_shutter("auto")
        except Exception:
            pass

    # -----------------
    # Connection
    # -----------------
    def connect(self) -> None:
        if self.cam is not None:
            return
        if Andor is None:
            raise ImportError(f"Cannot import pylablib.devices.Andor: {_IMPORT_ERR}")
        num = int(Andor.get_cameras_number_SDK2())
        if num <= 0:
            raise RuntimeError("No Andor SDK2 cameras detected")
        self.cam = Andor.AndorSDK2Camera(self.cam_index, fan_mode=self._fan_mode)
        self._streaming = False
        self._apply_defaults_on_connect()

    def disconnect(self) -> None:
        cam = self.cam
        if cam is None:
            return
        try:
            self.stop_stream()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        self.cam = None
        self._streaming = False

    # -----------------
    # Defaults
    # -----------------
    def _apply_defaults_on_connect(self) -> None:
        # Abort and clear any acquisition left running from a previous session.
        # Without this, the first snap on DU970P (and similar CCDs) times out
        # because the camera is still in a running state.
        self._recover_after_timeout()
        try:
            self.set_frame_api("snap")
            self.set_acquisition_mode("single")
            self.set_trigger_mode("internal")
        except Exception:
            pass
        try:
            self.set_preamp_gain_default()
            self.set_fastest_readout_default()
        except Exception:
            pass
        try:
            self.set_shutter("auto")
        except Exception:
            pass

    # -----------------
    # Camera info
    # -----------------
    def get_camera_info(self) -> Optional[CameraInfo]:
        if self.cam is None:
            return None
        info = CameraInfo()

        def _norm(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            try:
                s = str(v).strip()
            except Exception:
                return None
            if not s or s.lower() in ("none", "unknown", "n/a"):
                return None
            return s

        try:
            if hasattr(self.cam, "get_device_info"):
                di = self.cam.get_device_info()
                if isinstance(di, dict):
                    info.camera_name = _norm(di.get("vendor")) or info.camera_name
                    info.model_name = _norm(di.get("model")) or info.model_name
                    info.serial_number = _norm(di.get("serial")) or info.serial_number
                else:
                    try:
                        info.camera_name = _norm(di[0]) or info.camera_name
                        info.model_name = _norm(di[1]) or info.model_name
                        info.serial_number = _norm(di[2]) or info.serial_number
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if hasattr(self.cam, "get_controller_card_model"):
                v = self.cam.get_controller_card_model()
                info.camera_name = _norm(v) or info.camera_name
        except Exception:
            pass

        return info

    # -----------------
    # Camera settings
    # -----------------
    def set_frame_api(self, mode: str) -> str:
        s = (mode or "").strip().lower().replace(" ", "").replace("-", "")
        if s in ("snap", "single", "oneshot"):
            self._frame_backend = "snap"
        elif s in ("buffer", "stream", "streambuffer", "stream+buffer", "cont", "continuous"):
            self._frame_backend = "buffer"
        else:
            raise ValueError("frame_api must be 'Snap' or 'Stream+Buffer'")
        return self._frame_backend

    def get_frame_api(self) -> str:
        return str(getattr(self, "_frame_backend", "snap"))

    def set_exposure_ms(self, exposure_ms: float) -> bool:
        try:
            ms = float(exposure_ms)
        except Exception:
            return False
        if ms <= 0:
            ms = 0.1
        if ms > 1.0e7:
            ms = 1.0e7
        self._exposure_s = ms * 1e-3
        if self.cam is None:
            return True
        try:
            self.cam.set_exposure(float(self._exposure_s))
            return True
        except Exception:
            return False

    def get_exposure_ms(self) -> Optional[float]:
        if self.cam is not None:
            for fn in ("get_exposure_ms", "get_exposure_time_ms"):
                if hasattr(self.cam, fn):
                    try:
                        return float(getattr(self.cam, fn)())
                    except Exception:
                        pass
            for fn in ("get_exposure", "get_exposure_time", "get_exposure_s"):
                if hasattr(self.cam, fn):
                    try:
                        v = float(getattr(self.cam, fn)())
                        return v * 1e3
                    except Exception:
                        pass
        return float(self._exposure_s) * 1e3

    def set_em_gain(self, gain: int) -> bool:
        if self.cam is None:
            return False
        for fn in ("set_emccd_gain", "set_em_gain", "set_gain"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(int(gain))
                    return True
                except Exception:
                    pass
        return False

    def get_em_gain(self) -> Optional[int]:
        if self.cam is None:
            return None
        for fn in ("get_emccd_gain", "get_em_gain", "get_gain"):
            if hasattr(self.cam, fn):
                try:
                    return int(getattr(self.cam, fn)())
                except Exception:
                    pass
        return None

    def set_binning(self, hbin: int, vbin: int) -> bool:
        if self.cam is None:
            return False
        for fn in ("set_binning", "set_bin"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(int(hbin), int(vbin))
                    return True
                except Exception:
                    pass
        return False

    def get_binning(self) -> Optional[Tuple[int, int]]:
        if self.cam is None:
            return None
        for fn in ("get_binning", "get_bin"):
            if hasattr(self.cam, fn):
                try:
                    val = getattr(self.cam, fn)()
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        return int(val[0]), int(val[1])
                except Exception:
                    pass
        return None

    def set_trigger_mode(self, mode: str) -> bool:
        if self.cam is None or not hasattr(self.cam, "set_trigger_mode"):
            return False
        s = str(mode).strip().lower()
        if s in ("internal", "int"):
            s = "int"
        elif s in ("external", "ext"):
            s = "ext"
        elif s in ("software", "soft"):
            s = "software"
        try:
            self.cam.set_trigger_mode(s)
            return True
        except Exception:
            return False

    def get_trigger_mode(self) -> Optional[str]:
        if self.cam is None or not hasattr(self.cam, "get_trigger_mode"):
            return None
        try:
            return str(self.cam.get_trigger_mode())
        except Exception:
            return None

    def set_acquisition_mode(self, mode: str) -> bool:
        if self.cam is None or not hasattr(self.cam, "set_acquisition_mode"):
            return False
        s = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
        if s in ("run_till_abort", "run_until_abort", "cont", "continuous", "rta"):
            s = "cont"
        elif s in ("single", "single_scan", "singlescan"):
            s = "single"
        elif s in ("accum", "accumulate", "accumulation"):
            s = "single"
        elif s in ("kinetic", "kinetics"):
            s = "kinetic"
        self._acq_mode = s
        try:
            self.cam.set_acquisition_mode(s, setup_params=True)
            return True
        except TypeError:
            try:
                self.cam.set_acquisition_mode(s)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def get_acquisition_mode(self) -> Optional[str]:
        if self.cam is None or not hasattr(self.cam, "get_acquisition_mode"):
            return None
        try:
            return str(self.cam.get_acquisition_mode())
        except Exception:
            return None

    def set_read_mode(self, mode: str) -> bool:
        if self.cam is None or not hasattr(self.cam, "set_read_mode"):
            return False
        s = str(mode).strip().lower()
        s = "fvb" if s.startswith("fvb") else "image"
        try:
            self.cam.set_read_mode(s)
            return True
        except Exception:
            return False

    def get_read_mode(self) -> Optional[str]:
        if self.cam is None or not hasattr(self.cam, "get_read_mode"):
            return None
        try:
            return str(self.cam.get_read_mode())
        except Exception:
            return None

    def get_all_amp_modes(self, full: bool = True):
        if self.cam is None or not hasattr(self.cam, "get_all_amp_modes"):
            return None
        with self._paused_acquisition(clear=False):
            try:
                return self.cam.get_all_amp_modes(full=bool(full))
            except TypeError:
                return self.cam.get_all_amp_modes()

    def get_amp_mode(self, full: bool = True):
        if self.cam is None or not hasattr(self.cam, "get_amp_mode"):
            return None
        with self._paused_acquisition(clear=False):
            try:
                return self.cam.get_amp_mode(full=bool(full))
            except TypeError:
                return self.cam.get_amp_mode()

    def set_amp_mode(self, channel=None, oamp=None, hsspeed=None, preamp=None) -> bool:
        if self.cam is None or not hasattr(self.cam, "set_amp_mode"):
            return False
        try:
            self.cam.set_amp_mode(channel=channel, oamp=oamp, hsspeed=hsspeed, preamp=preamp)
        except Exception:
            return False
        # pylablib's set_amp_mode() writes SDK registers but does not reliably update
        # _cpar, so apply_settings() (called by start_acquisition / snap) would revert
        # the amp mode to whatever was in _cpar before. Patch it here to keep them in sync.
        try:
            cpar = getattr(self.cam, "_cpar", None)
            if isinstance(cpar, dict):
                if channel is not None:
                    cpar["channel"] = int(channel)
                if oamp is not None:
                    cpar["oamp"] = int(oamp)
                if hsspeed is not None:
                    cpar["hsspeed"] = int(hsspeed)
                if preamp is not None:
                    cpar["preamp"] = int(preamp)
        except Exception:
            pass
        return True

    def _parse_amp_mode(self, m):
        if isinstance(m, dict):
            ch = m.get("channel", m.get("ch", None))
            oa = m.get("oamp", m.get("output_amp", None))
            hs = m.get("hsspeed", m.get("readout_rate", None))
            pa = m.get("preamp", m.get("preamp_idx", None))
            oamp_kind = m.get("oamp_kind", m.get("output_amp_kind", ""))
            hs_mhz = m.get("hsspeed_MHz", m.get("hs_mhz", None))
            pre_gain = m.get("preamp_gain", m.get("preamp_x", None))
            return ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain
        if isinstance(m, (list, tuple)):
            if len(m) >= 8:
                return m[0], m[2], m[4], m[6], m[3], m[5], m[7]
            if len(m) == 7:
                return m[0], m[1], m[2], m[3], m[4], m[5], m[6]
            if len(m) >= 4:
                return m[0], m[1], m[2], m[3], str(m[1]), m[2], m[3]
        return None, None, None, None, "", None, None

    def get_amp_mode_choices(self):
        modes = self.get_all_amp_modes(full=True)
        if not modes:
            return [], [], []

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        amps = []
        rates = []
        gains = []
        for m in modes:
            ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain = self._parse_amp_mode(m)
            s = str(oamp_kind).strip().lower()
            amp_label = "EM" if (("conv" not in s) and (("em" in s) or ("multip" in s) or ("electron" in s))) else "Conventional"
            amps.append(amp_label)
            h = to_float(hs_mhz)
            if h is not None:
                rates.append(h)
            g = to_float(pre_gain)
            if g is not None:
                gains.append(g)

        amps = sorted(set(amps))
        rates = sorted(set(rates))
        gains = sorted(set(gains))

        rate_labels = []
        for mhz in rates:
            if mhz < 1.0:
                rate_labels.append(f"{mhz*1000.0:g}kHz at 16-bit")
            else:
                rate_labels.append(f"{mhz:g}MHz at 16-bit")
        gain_labels = [f"{g:g}x" for g in gains]

        return amps, rate_labels, gain_labels

    def set_amp_mode_by_labels(
        self,
        output_amp: str,
        readout_rate: str,
        preamp_gain: str,
        *,
        force_preamp: bool = False,
    ) -> bool:
        modes = self.get_all_amp_modes(full=True)
        if not modes:
            return False

        def norm(s: str) -> str:
            return (s or "").strip().lower()

        def parse_float(x):
            try:
                return float(x)
            except Exception:
                return None

        def parse_mhz(label: str):
            s = norm(label)
            import re
            m = re.search(r"(\d+(?:\.\d+)?)\s*(khz|mhz|hz)?", s)
            if not m:
                return None
            v = float(m.group(1))
            unit = (m.group(2) or "")
            if unit == "khz":
                return v / 1000.0
            if unit == "hz":
                return v / 1e6
            return v

        def parse_gain_x(label: str):
            s = norm(label).replace("a-", "x")
            import re
            m = re.search(r"(\d+(?:\.\d+)?)\s*x", s)
            if m:
                return float(m.group(1))
            return parse_float(s)

        def is_em_kind(kind: object) -> Optional[bool]:
            kind_s = norm(str(kind))
            if not kind_s:
                return None
            return ("conv" not in kind_s) and (("em" in kind_s) or ("multip" in kind_s) or ("electron" in kind_s))

        want_em = ("em" in norm(output_amp)) and ("conv" not in norm(output_amp))
        want_mhz = parse_mhz(readout_rate)
        want_gain = parse_gain_x(preamp_gain)
        out_amp_given = norm(output_amp) != ""
        tol_mhz = 0.05
        tol_gain = 0.05

        candidates = []
        for m in modes:
            ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain = self._parse_amp_mode(m)
            if ch is None or oa is None or hs is None or pa is None:
                continue
            candidates.append((ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain))

        if not candidates:
            return False

        def score(c):
            kind_s = norm(str(c[4]))
            is_em = (("conv" not in kind_s) and (("em" in kind_s) or ("multip" in kind_s) or ("electron" in kind_s)))
            total = 0.0
            if out_amp_given:
                total += 0.0 if (is_em == want_em) else 100.0
            if want_mhz is not None:
                hs_val = parse_float(c[5])
                total += abs(hs_val - want_mhz) * 10.0 if hs_val is not None else 50.0
            if want_gain is not None:
                g_val = parse_float(c[6])
                total += abs(g_val - want_gain) * 5.0 if g_val is not None else 50.0
            return total

        best = min(candidates, key=score)
        ch, oa, hs, pa = best[0], best[1], best[2], best[3]
        try:
            with self._paused_acquisition(clear=True):
                ok = self.set_amp_mode(channel=ch, oamp=oa, hsspeed=hs, preamp=pa)
                if force_preamp and want_gain is not None:
                    info = None
                    try:
                        info = self.get_amp_mode_info()
                    except Exception:
                        info = None
                    cur_gain = None
                    if info is not None:
                        cur_gain = info.get("preamp_gain")
                    if cur_gain is None or abs(float(cur_gain) - float(want_gain)) > tol_gain:
                        ok_force = self._set_amp_mode_force(ch, oa, hs, pa)
                        ok = ok or ok_force
                return ok
        except Exception:
            return False

    def _set_amp_mode_force(self, channel, oamp, hsspeed, preamp) -> bool:
        if self.cam is None:
            return False
        try:
            from pylablib.devices.Andor import AndorSDK2
        except Exception:
            return False
        try:
            ch = int(channel)
            oa = int(oamp)
            hs = int(hsspeed)
            pa = int(preamp)
        except Exception:
            return False
        try:
            with self._paused_acquisition(clear=True):
                AndorSDK2.lib.set_amp_mode((ch, oa, hs, pa))
                try:
                    self.cam._cpar["channel"] = ch
                    self.cam._cpar["oamp"] = oa
                    self.cam._cpar["hsspeed"] = hs
                    self.cam._cpar["preamp"] = pa
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def set_fastest_readout_default(self) -> bool:
        modes = self.get_all_amp_modes(full=True)
        if not modes:
            return False
        best = None
        best_mhz = -1.0
        best_gain = None
        for m in modes:
            ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain = self._parse_amp_mode(m)
            if ch is None or oa is None or hs is None or pa is None:
                continue
            try:
                mhz = float(hs_mhz)
            except Exception:
                continue
            if mhz > best_mhz:
                best_mhz = mhz
                best = (ch, oa, hs, pa)
                best_gain = pre_gain
            elif mhz == best_mhz:
                try:
                    g = float(pre_gain)
                    if best_gain is None or g < float(best_gain):
                        best = (ch, oa, hs, pa)
                        best_gain = pre_gain
                except Exception:
                    pass
        if best is None:
            return False
        ch, oa, hs, pa = best
        try:
            with self._paused_acquisition(clear=True):
                return self.set_amp_mode(channel=ch, oamp=oa, hsspeed=hs, preamp=pa)
        except Exception:
            return False

    def set_preamp_gain_default(self) -> bool:
        modes = self.get_all_amp_modes(full=True)
        if not modes:
            return False
        best = None
        best_gain = None
        for m in modes:
            ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain = self._parse_amp_mode(m)
            if ch is None or oa is None or hs is None or pa is None:
                continue
            try:
                g = float(pre_gain)
            except Exception:
                continue
            if best_gain is None or g < best_gain:
                best_gain = g
                best = (ch, oa, hs, pa)
        if best is None:
            return False
        ch, oa, hs, pa = best
        try:
            with self._paused_acquisition(clear=True):
                return self.set_amp_mode(channel=ch, oamp=oa, hsspeed=hs, preamp=pa)
        except Exception:
            return False

    def get_amp_mode_info(self) -> Optional[dict]:
        m = self.get_amp_mode(full=True)
        if m is None:
            return None

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        ch, oa, hs, pa, oamp_kind, hs_mhz, pre_gain = self._parse_amp_mode(m)
        s = str(oamp_kind).strip().lower()
        amp_label = "EM" if (("conv" not in s) and (("em" in s) or ("multip" in s) or ("electron" in s))) else "Conventional"
        mhz = to_float(hs_mhz)
        gain = to_float(pre_gain)
        rate_label = None
        gain_label = None
        if mhz is not None:
            rate_label = f"{mhz*1000.0:g}kHz at 16-bit" if mhz < 1.0 else f"{mhz:g}MHz at 16-bit"
        if gain is not None:
            gain_label = f"{gain:g}x"
        return {
            "amp_label": amp_label,
            "rate_label": rate_label,
            "gain_label": gain_label,
            "hsspeed_mhz": mhz,
            "preamp_gain": gain,
            "raw": m,
        }

    def set_baseline_clamp(self, on: bool) -> bool:
        if self.cam is None:
            return False
        for fn in ("set_baseline_clamp", "enable_baseline_clamp", "set_baseline_clamp_enabled"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(bool(on))
                    return True
                except Exception:
                    pass
        return False

    def set_shutter(self, mode: str = "auto") -> bool:
        if self.cam is None or not hasattr(self.cam, "setup_shutter"):
            return False
        try:
            self.cam.setup_shutter(str(mode))
            return True
        except Exception:
            return False

    def get_shutter_parameters(self):
        if self.cam is None or not hasattr(self.cam, "get_shutter_parameters"):
            return None
        try:
            return self.cam.get_shutter_parameters()
        except Exception:
            return None

    def set_vshift_us(self, us: float) -> bool:
        if self.cam is None:
            return False
        us = float(us)
        for fn in ("set_vshift_speed", "set_vshift_speed_us", "set_vertical_shift_speed"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(us)
                    return True
                except Exception:
                    try:
                        getattr(self.cam, fn)(us * 1e-6)
                        return True
                    except Exception:
                        pass
        return False

    def set_vclock_amp(self, amp: str) -> bool:
        if self.cam is None:
            return False
        s = str(amp).strip().lower()
        val = None
        if s in ("normal", "0", "+0"):
            val = 0
        else:
            try:
                val = int(s.replace("+", ""))
            except Exception:
                val = None
        for fn in ("set_vshift_amplitude", "set_vertical_clock_amp", "set_vclock_amp"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(val if val is not None else amp)
                    return True
                except Exception:
                    try:
                        getattr(self.cam, fn)(amp)
                        return True
                    except Exception:
                        pass
        return False

    def set_temperature_setpoint(self, celsius: float) -> bool:
        if self.cam is None:
            return False
        for fn in ("set_temperature_setpoint", "set_temperature", "set_setpoint_c"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(float(celsius))
                    return True
                except Exception:
                    pass
        return False

    def get_temperature_setpoint(self) -> Optional[float]:
        if self.cam is None:
            return None
        for fn in ("get_temperature_setpoint", "get_setpoint_c"):
            if hasattr(self.cam, fn):
                try:
                    return float(getattr(self.cam, fn)())
                except Exception:
                    pass
        return None

    def set_cooler(self, on: bool) -> bool:
        if self.cam is None:
            return False
        for fn in ("set_cooler", "enable_cooler", "set_cooling"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)(bool(on))
                    return True
                except Exception:
                    pass
        return False

    def is_cooler_on(self) -> Optional[bool]:
        if self.cam is None:
            return None
        for fn in ("is_cooler_on", "get_cooler", "get_cooling"):
            if hasattr(self.cam, fn):
                try:
                    v = getattr(self.cam, fn)()
                    if isinstance(v, bool):
                        return bool(v)
                    if isinstance(v, (int, float)):
                        return bool(int(v))
                except Exception:
                    pass
        return None

    def get_temperature_c(self) -> Optional[float]:
        if self.cam is None:
            return None
        for fn in ("get_temperature_c", "get_ccd_temperature_c", "get_temperature", "get_current_temperature"):
            if hasattr(self.cam, fn):
                try:
                    v = getattr(self.cam, fn)()
                    if isinstance(v, (list, tuple)) and v:
                        return float(v[0])
                    return float(v)
                except Exception:
                    pass
        return None

    def get_temperature_status(self) -> Optional[str]:
        if self.cam is None:
            return None
        for fn in ("get_temperature_status", "get_temp_status"):
            if hasattr(self.cam, fn):
                try:
                    return str(getattr(self.cam, fn)())
                except Exception:
                    pass
        return None

    # -----------------
    # Acquisition
    # -----------------
    def start_stream(self) -> bool:
        if self.cam is None:
            return False
        cam = self.cam

        # If already acquiring, treat as success (avoid DRV_ACQUIRING loops)
        try:
            if hasattr(cam, "get_status") and cam.get_status() == "acquiring":
                self._streaming = True
                return True
            if hasattr(cam, "acquisition_in_progress") and cam.acquisition_in_progress():
                self._streaming = True
                return True
        except Exception:
            pass

        # Stop/clear any previous acquisition state (best-effort)
        for fn in ("stop_acquisition", "stop", "abort_acquisition", "abort"):
            if hasattr(cam, fn):
                try:
                    getattr(cam, fn)()
                    break
                except Exception:
                    pass
        if hasattr(cam, "clear_acquisition"):
            try:
                cam.clear_acquisition()
            except Exception:
                pass

        try:
            if hasattr(cam, "set_trigger_mode"):
                cam.set_trigger_mode("int")
        except Exception:
            pass

        # Continuous acquisition mode for buffer backend
        try:
            if hasattr(cam, "set_acquisition_mode"):
                cam.set_acquisition_mode("cont", setup_params=True)
        except Exception:
            pass

        # Allocate ring buffer sized to ~2 seconds of data (clamped)
        nframes = 200
        try:
            frame_period = None
            if hasattr(cam, "get_frame_period"):
                frame_period = cam.get_frame_period()
            elif hasattr(cam, "get_frame_timings"):
                frame_period = cam.get_frame_timings()[1]
            if frame_period and frame_period > 0:
                fps = 1.0 / float(frame_period)
                nframes = int(max(50, min(2000, fps * 2.0)))
        except Exception:
            nframes = 200

        try:
            if hasattr(cam, "setup_acquisition"):
                cam.setup_acquisition(mode="sequence", nframes=int(nframes))
        except Exception:
            pass

        ok = False
        for fn in ("start_acquisition", "start_acq", "start", "run"):
            if hasattr(cam, fn):
                try:
                    getattr(cam, fn)()
                    ok = True
                    break
                except Exception:
                    pass
        self._streaming = bool(ok)
        return self._streaming

    def stop_stream(self) -> None:
        if self.cam is None:
            self._streaming = False
            return
        for fn in ("stop_acquisition", "stop_acq", "stop", "abort_acquisition", "abort"):
            if hasattr(self.cam, fn):
                try:
                    getattr(self.cam, fn)()
                    break
                except Exception:
                    pass
        self._streaming = False

    def set_accumulations(self, n: int, switch_mode: bool = False) -> bool:
        try:
            n = int(n)
        except Exception:
            return False
        if n < 1:
            n = 1
        self._accum_n = int(n)
        if switch_mode:
            self._acq_mode = "single"
        return True

    def get_frame(self) -> dict:
        if self.cam is None:
            try:
                self.connect()
            except Exception as exc:
                return {"ok": False, "err": f"connect failed: {type(exc).__name__}: {exc}"}

        img = None
        backend = self._frame_backend

        if backend == "buffer":
            if not self._streaming:
                try:
                    self.start_stream()
                except Exception as exc:
                    return {"ok": False, "err": f"start_stream failed: {type(exc).__name__}: {exc}"}
            cam = self.cam
            wait_err = None
            timeout_s = self._dynamic_timeout_s()
            if cam is not None and hasattr(cam, "wait_for_frame"):
                try:
                    cam.wait_for_frame(timeout=float(timeout_s), since="lastread")
                except TypeError:
                    try:
                        cam.wait_for_frame(timeout=float(timeout_s))
                    except Exception as exc:
                        wait_err = exc
                except Exception as exc:
                    wait_err = exc
            if wait_err is not None and "timeout" in type(wait_err).__name__.lower():
                self._recover_after_timeout()
            if cam is not None:
                for fn in ("read_newest_image", "read_oldest_image", "read_last_image", "read_newest_frame"):
                    if hasattr(cam, fn):
                        try:
                            r = getattr(cam, fn)()
                        except Exception:
                            r = None
                        img = self._extract_image(r)
                        if img is not None:
                            break
            if img is None:
                if wait_err is None:
                    try:
                        self.stop_stream()
                    except Exception:
                        pass
                if wait_err is not None:
                    return {"ok": False, "err": f"{type(wait_err).__name__}: {wait_err} (wait_timeout={timeout_s:.3f}s)"}
                return {"ok": False, "err": f"no unread frame (wait_timeout={timeout_s:.3f}s)"}
        else:
            try:
                self.stop_stream()
            except Exception:
                pass
            try:
                timeout_s = self._dynamic_timeout_s()
                try:
                    r = self.cam.snap(timeout=float(timeout_s))
                except TypeError:
                    r = self.cam.snap()
            except Exception as exc:
                if "timeout" in type(exc).__name__.lower():
                    self._recover_after_timeout()
                return {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
            img = self._extract_image(r)
            if img is None:
                return {"ok": False, "err": "snap returned non-image"}

        if np is None:
            return {"ok": False, "err": "numpy not available"}

        a = np.asarray(img)
        if a.ndim != 2 or a.size == 0:
            return {"ok": False, "err": "bad image shape"}
        a16 = a.astype(np.uint16, copy=False)
        g8 = self._preview8_from_u16(a16)
        h, w = int(a16.shape[0]), int(a16.shape[1])
        now = time.time()

        self._ts_window.append(now)
        if len(self._ts_window) > 40:
            self._ts_window = self._ts_window[-40:]
        fps_raw = 0.0
        if len(self._ts_window) >= 2:
            dt = self._ts_window[-1] - self._ts_window[0]
            if dt > 1e-3:
                fps_raw = (len(self._ts_window) - 1) / dt
        if self._fps_smooth is None:
            self._fps_smooth = fps_raw
        else:
            self._fps_smooth = (1 - self._fps_alpha) * self._fps_smooth + self._fps_alpha * fps_raw

        self.last_img_wh = (w, h)

        return {
            "ok": True,
            "image": a16,
            "image8": g8,
            "w": w,
            "h": h,
            "timestamp": now,
            "fps_raw": fps_raw,
            "fps_smooth": float(self._fps_smooth),
        }

    def acquire_accumulated(self, n_accum: int) -> dict:
        try:
            n = int(n_accum)
        except Exception:
            n = 1
        if n < 1:
            n = 1

        last_fr = None
        sum_img = None
        for i in range(1, n + 1):
            fr = self.get_frame()
            if not fr.get("ok"):
                return fr
            img = fr.get("image")
            if img is None:
                return {"ok": False, "err": "no image data for accumulation"}
            if sum_img is None:
                sum_img = np.asarray(img, dtype=np.float64)
            else:
                sum_img += np.asarray(img, dtype=np.float64)
            last_fr = fr

        if last_fr is None:
            return {"ok": False, "err": "no frames acquired"}

        out = dict(last_fr)
        out["ok"] = True
        out["image"] = sum_img
        out["image8"] = None
        out["accum_idx"] = n
        out["accum_n"] = n
        return out

    # -----------------
    # Spectrograph
    # -----------------
    def invalidate_wavelength_axis(self) -> None:
        self._wavelength_axis_nm = None
        if self.spec is None:
            self._spec_ready = False

    def connect_spectrograph(self, spec_index: int = 0) -> bool:
        self.connect()
        if Andor is None:
            return False
        try:
            self._spec_index = int(spec_index)
            self.spec = Andor.ShamrockSpectrograph(self._spec_index)
            if hasattr(self.spec, "setup_pixels_from_camera"):
                self.spec.setup_pixels_from_camera(self.cam)
            self._spec_ready = True
            self._wavelength_axis_nm = None
            try:
                wl_m = self.spec.get_calibration()
                wl_nm = np.asarray(wl_m, dtype=float).ravel() * 1e9
                self._wavelength_axis_nm = wl_nm
            except Exception:
                self._wavelength_axis_nm = None
            return True
        except Exception:
            self._spec_ready = False
            self._wavelength_axis_nm = None
            try:
                if self.spec is not None:
                    self.spec.close()
            except Exception:
                pass
            self.spec = None
            return False

    def disconnect_spectrograph(self) -> None:
        try:
            if self.spec is not None:
                self.spec.close()
        finally:
            self.spec = None
            self._spec_ready = False
            self._wavelength_axis_nm = None

    def get_wavelength_axis(self, force: bool = False):
        if self._wavelength_axis_nm is not None and not force:
            return self._wavelength_axis_nm
        ok = self.connect_spectrograph(getattr(self, "_spec_index", 0))
        if ok:
            return self._wavelength_axis_nm
        return None

    def set_center_wavelength_nm(self, wl_nm: float) -> bool:
        if self.spec is None:
            ok = self.connect_spectrograph(getattr(self, "_spec_index", 0))
            if not ok:
                return False
        try:
            self.spec.set_wavelength(float(wl_nm) * 1e-9)
            self.invalidate_wavelength_axis()
            return True
        except Exception:
            return False

    def get_center_wavelength_nm(self) -> Optional[float]:
        if self.spec is None:
            return None
        for fn in ("get_wavelength", "get_center_wavelength"):
            if hasattr(self.spec, fn):
                try:
                    wl_m = getattr(self.spec, fn)()
                    return float(wl_m) * 1e9
                except Exception:
                    pass
        return None

    def spec_get_gratings_number(self) -> Optional[int]:
        if self.spec is None:
            return None
        try:
            return int(self.spec.get_gratings_number())
        except Exception:
            return None

    def spec_get_grating(self) -> Optional[int]:
        if self.spec is None:
            return None
        try:
            return int(self.spec.get_grating())
        except Exception:
            return None

    def spec_set_grating(self, grating: int) -> bool:
        if self.spec is None:
            return False
        try:
            self.spec.set_grating(int(grating))
            self.invalidate_wavelength_axis()
            return True
        except Exception:
            return False

    def spec_get_grating_info(self, grating: int) -> Any:
        if self.spec is None:
            return None
        try:
            return self.spec.get_grating_info(int(grating))
        except Exception:
            return None

    def spec_is_slit_present(self, slit) -> Optional[bool]:
        if self.spec is None:
            return None
        try:
            return bool(self.spec.is_slit_present(slit))
        except Exception:
            return None

    def spec_get_slit_width_um(self, slit) -> Optional[float]:
        if self.spec is None:
            return None
        try:
            w_m = float(self.spec.get_slit_width(slit))
            return w_m * 1e6
        except Exception:
            return None

    def spec_set_slit_width_um(self, slit, width_um: float) -> bool:
        if self.spec is None:
            return False
        try:
            self.spec.set_slit_width(slit, float(width_um) * 1e-6)
            return True
        except Exception:
            return False

    # -----------------
    # Linecuts
    # -----------------
    def linecut_horizontal(self, image, row: int, width: int = 1, mode: str = "sum"):
        if np is None:
            return None
        a = np.asarray(image)
        if a.ndim != 2:
            return None
        h, _ = a.shape
        width = max(1, int(width))
        half = width // 2
        r1 = max(0, int(row) - half)
        r2 = min(h, int(row) + half + (1 if width % 2 else 0))
        sl = a[r1:r2, :]
        if mode == "mean":
            return sl.mean(axis=0)
        return sl.sum(axis=0)

    def linecut_vertical(self, image, col: int, width: int = 1, mode: str = "sum"):
        if np is None:
            return None
        a = np.asarray(image)
        if a.ndim != 2:
            return None
        _, w = a.shape
        width = max(1, int(width))
        half = width // 2
        c1 = max(0, int(col) - half)
        c2 = min(w, int(col) + half + (1 if width % 2 else 0))
        sl = a[:, c1:c2]
        if mode == "mean":
            return sl.mean(axis=1)
        return sl.sum(axis=1)

    # -----------------
    # Saving
    # -----------------
    def save_ascii(
        self,
        filepath: str,
        image,
        metadata: Optional[dict] = None,
        wavelength_axis_nm: Optional[Any] = None,
    ) -> None:
        if np is None:
            raise RuntimeError("numpy required for save_ascii")
        a = np.asarray(image)
        if a.ndim != 2:
            raise ValueError("image must be 2D")
        meta = metadata or {}
        header_lines = [
            "Andor ASCII Export",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        for k, v in meta.items():
            header_lines.append(f"{k}: {v}")
        wl = None
        if wavelength_axis_nm is not None:
            try:
                wl = np.asarray(wavelength_axis_nm, dtype=float).ravel()
            except Exception:
                wl = None
        else:
            try:
                wl = np.asarray(self._wavelength_axis_nm, dtype=float).ravel()
            except Exception:
                wl = None

        if wl is not None:
            if (wl.size != a.shape[1]) or np.allclose(wl, 0.0):
                wl = None

        if wl is not None:
            header_lines.append("Columns: wavelength_nm, intensity_row0..rowN")
            out = np.empty((a.shape[1], a.shape[0] + 1), dtype=np.float64)
            out[:, 0] = wl
            out[:, 1:] = a.T
            header = "\n".join(header_lines)
            np.savetxt(filepath, out, fmt="%g", header=header)
        else:
            header = "\n".join(header_lines)
            np.savetxt(filepath, a, fmt="%d", header=header)

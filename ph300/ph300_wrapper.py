# ph300_wrapper.py
# Thin ctypes wrapper for PicoQuant PicoHarp 300 (PHLib v3.x), histogramming mode only.
#
# Notes from PicoQuant:
# - PHLib is not generally re-entrant: do not call from multiple threads concurrently. :contentReference[oaicite:3]{index=3}
# - Only one program at a time can use the PicoHarp device. :contentReference[oaicite:4]{index=4}

from __future__ import annotations

import os
import time
import threading
import ctypes as ct
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

try:
    import numpy as np
except Exception:
    np = None  # allow non-numpy usage if needed


HISTCHAN = 65536  # PicoHarp histogram channels (PHLib calls require buffer of at least HISTCHAN) :contentReference[oaicite:5]{index=5}
WARN_TEXT_BUFSZ = 16384  # PH_GetWarningsText: buffer for at least 16384 chars :contentReference[oaicite:6]{index=6}
ERR_TEXT_BUFSZ = 40      # PH_GetErrorString: buffer for at least 40 chars :contentReference[oaicite:7]{index=7}
VERS_BUFSZ = 8           # PH_GetLibraryVersion: buffer for at least 8 chars :contentReference[oaicite:8]{index=8}


class PicoHarpError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"PHLib error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RateStatus:
    sync_rate_hz: int
    ch0_rate_hz: int
    ch1_rate_hz: int
    warnings_bitfield: int
    warnings_text: str


class PicoHarp300:
    """
    Minimal PicoHarp 300 histogramming wrapper around PHLib (Windows DLL).
    Designed to be GUI-agnostic and orchestration-friendly.

    Typical sequence:
      open() -> initialize_histogramming() -> calibrate()
      configure inputs (CFD, sync divider, binning, offset)
      clear_hist_mem()
      start_meas(t_ms)
      poll: read_histogram(), get_rates_and_warnings(), get_elapsed_ms(), ctc_done()
      stop_meas()
      close()
    """

    def __init__(self, dll_path: str, device_index: int = 0, verbose: bool = True):
        self.dll_path = os.path.abspath(dll_path)
        self.device_index = int(device_index)
        self.verbose = bool(verbose)

        self._lock = threading.RLock()
        self._ph = None  # type: Optional[ct.WinDLL]
        self._is_open = False
        self._is_initialized = False

        # cached resolution (ps) after set_binning / get_resolution
        self._resolution_ps = None  # type: Optional[float]

    # -----------------
    # Internal utilities
    # -----------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PH300] {msg}")

    def _load_dll(self) -> ct.WinDLL:
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"PHLib DLL not found: {self.dll_path}")

        dll_dir = os.path.dirname(self.dll_path)
        # Python 3.8+ on Windows: ensure the DLL directory is added to the DLL search path
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)

        ph = ct.WinDLL(self.dll_path)  # PHLib uses _stdcall :contentReference[oaicite:9]{index=9}
        return ph

    def _errstr(self, code: int) -> str:
        assert self._ph is not None
        buf = ct.create_string_buffer(ERR_TEXT_BUFSZ)
        # int PH_GetErrorString(char* errstring, int errcode); :contentReference[oaicite:10]{index=10}
        self._ph.PH_GetErrorString(buf, int(code))
        return buf.value.decode(errors="replace")

    def _check(self, ret: int, func: str) -> None:
        if ret < 0:
            raise PicoHarpError(int(ret), f"{func} failed: {self._errstr(int(ret))}")

    def _require_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("Device not open. Call open() first.")

    def _require_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("Device not initialized. Call initialize_histogramming() first.")

    def _bind_prototypes(self) -> None:
        """Bind only the functions we need (histogramming + rates/warnings)."""
        assert self._ph is not None
        ph = self._ph

        # General
        ph.PH_GetLibraryVersion.argtypes = [ct.c_char_p]
        ph.PH_GetLibraryVersion.restype = ct.c_int

        ph.PH_GetErrorString.argtypes = [ct.c_char_p, ct.c_int]
        ph.PH_GetErrorString.restype = ct.c_int

        # Open/Close/Initialize
        ph.PH_OpenDevice.argtypes = [ct.c_int, ct.c_char_p]
        ph.PH_OpenDevice.restype = ct.c_int

        ph.PH_CloseDevice.argtypes = [ct.c_int]
        ph.PH_CloseDevice.restype = ct.c_int

        ph.PH_Initialize.argtypes = [ct.c_int, ct.c_int]
        ph.PH_Initialize.restype = ct.c_int

        # Initialized device functions
        ph.PH_GetHardwareInfo.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p]
        ph.PH_GetHardwareInfo.restype = ct.c_int

        ph.PH_GetSerialNumber.argtypes = [ct.c_int, ct.c_char_p]
        ph.PH_GetSerialNumber.restype = ct.c_int

        ph.PH_GetBaseResolution.argtypes = [ct.c_int, ct.POINTER(ct.c_double)]
        ph.PH_GetBaseResolution.restype = ct.c_int

        ph.PH_Calibrate.argtypes = [ct.c_int]
        ph.PH_Calibrate.restype = ct.c_int

        ph.PH_SetInputCFD.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int]
        ph.PH_SetInputCFD.restype = ct.c_int

        ph.PH_SetSyncDiv.argtypes = [ct.c_int, ct.c_int]
        ph.PH_SetSyncDiv.restype = ct.c_int

        ph.PH_SetSyncOffset.argtypes = [ct.c_int, ct.c_int]
        ph.PH_SetSyncOffset.restype = ct.c_int

        ph.PH_SetStopOverflow.argtypes = [ct.c_int, ct.c_int, ct.c_int]
        ph.PH_SetStopOverflow.restype = ct.c_int

        ph.PH_SetBinning.argtypes = [ct.c_int, ct.c_int]
        ph.PH_SetBinning.restype = ct.c_int

        ph.PH_SetMultistopEnable.argtypes = [ct.c_int, ct.c_int]
        ph.PH_SetMultistopEnable.restype = ct.c_int

        ph.PH_SetOffset.argtypes = [ct.c_int, ct.c_int]
        ph.PH_SetOffset.restype = ct.c_int

        ph.PH_ClearHistMem.argtypes = [ct.c_int, ct.c_int]
        ph.PH_ClearHistMem.restype = ct.c_int

        ph.PH_StartMeas.argtypes = [ct.c_int, ct.c_int]
        ph.PH_StartMeas.restype = ct.c_int

        ph.PH_StopMeas.argtypes = [ct.c_int]
        ph.PH_StopMeas.restype = ct.c_int

        ph.PH_CTCStatus.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
        ph.PH_CTCStatus.restype = ct.c_int

        ph.PH_GetHistogram.argtypes = [ct.c_int, ct.POINTER(ct.c_uint), ct.c_int]
        ph.PH_GetHistogram.restype = ct.c_int

        ph.PH_GetResolution.argtypes = [ct.c_int, ct.POINTER(ct.c_double)]
        ph.PH_GetResolution.restype = ct.c_int

        ph.PH_GetCountRate.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_int)]
        ph.PH_GetCountRate.restype = ct.c_int

        ph.PH_GetFlags.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
        ph.PH_GetFlags.restype = ct.c_int

        ph.PH_GetElapsedMeasTime.argtypes = [ct.c_int, ct.POINTER(ct.c_double)]
        ph.PH_GetElapsedMeasTime.restype = ct.c_int

        ph.PH_GetWarnings.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
        ph.PH_GetWarnings.restype = ct.c_int

        ph.PH_GetWarningsText.argtypes = [ct.c_int, ct.c_char_p, ct.c_int]
        ph.PH_GetWarningsText.restype = ct.c_int

    # -----------------
    # Public API
    # -----------------
    def get_library_version(self) -> str:
        with self._lock:
            if self._ph is None:
                self._ph = self._load_dll()
                self._bind_prototypes()
            buf = ct.create_string_buffer(VERS_BUFSZ)
            ret = self._ph.PH_GetLibraryVersion(buf)
            self._check(ret, "PH_GetLibraryVersion")
            return buf.value.decode(errors="replace")

    def open(self) -> str:
        """Open device index and return serial string."""
        with self._lock:
            if self._ph is None:
                self._ph = self._load_dll()
                self._bind_prototypes()

            serial = ct.create_string_buffer(VERS_BUFSZ)
            ret = self._ph.PH_OpenDevice(self.device_index, serial)
            self._check(ret, "PH_OpenDevice")

            self._is_open = True
            s = serial.value.decode(errors="replace")
            self._log(f"Opened device index {self.device_index}, serial={s}")
            return s

    def close(self) -> None:
        with self._lock:
            if self._ph is None or not self._is_open:
                return
            # best-effort stop
            try:
                if self._is_initialized:
                    self.stop_meas()
            except Exception:
                pass

            ret = self._ph.PH_CloseDevice(self.device_index)
            self._check(ret, "PH_CloseDevice")
            self._is_open = False
            self._is_initialized = False
            self._log("Closed device.")

    def initialize_histogramming(self) -> None:
        """Initialize device in histogramming mode (mode=0)."""
        with self._lock:
            self._require_open()
            # int PH_Initialize(int devidx, int mode); mode: 0 histogramming :contentReference[oaicite:11]{index=11}
            ret = self._ph.PH_Initialize(self.device_index, 0)
            self._check(ret, "PH_Initialize(mode=0 histogramming)")
            self._is_initialized = True
            self._log("Initialized (histogramming mode).")

    def get_hardware_info(self) -> Dict[str, str]:
        with self._lock:
            self._require_initialized()
            model = ct.create_string_buffer(16)
            partnum = ct.create_string_buffer(8)
            vers = ct.create_string_buffer(8)
            ret = self._ph.PH_GetHardwareInfo(self.device_index, model, partnum, vers)
            self._check(ret, "PH_GetHardwareInfo")
            return {
                "model": model.value.decode(errors="replace"),
                "partnum": partnum.value.decode(errors="replace"),
                "version": vers.value.decode(errors="replace"),
            }

    def calibrate(self) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_Calibrate(self.device_index)
            self._check(ret, "PH_Calibrate")
            self._log("Calibration done.")

    def set_input_cfd(self, channel: int, level_mv: int, zerocross_mv: int) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetInputCFD(self.device_index, int(channel), int(level_mv), int(zerocross_mv))
            self._check(ret, f"PH_SetInputCFD(ch={channel})")

    def set_sync_div(self, div: int) -> None:
        """div must be 1,2,4,8 per manual."""
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetSyncDiv(self.device_index, int(div))
            self._check(ret, "PH_SetSyncDiv")
            # Allow rate meter to update (100 ms gate time) :contentReference[oaicite:12]{index=12}
            time.sleep(0.12)

    def set_sync_offset_ps(self, offset_ps: int) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetSyncOffset(self.device_index, int(offset_ps))
            self._check(ret, "PH_SetSyncOffset")

    def set_stop_overflow(self, enable: bool, stopcount: int = 65535) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetStopOverflow(self.device_index, 1 if enable else 0, int(stopcount))
            self._check(ret, "PH_SetStopOverflow")

    def set_multistop_enable(self, enable: bool) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetMultistopEnable(self.device_index, 1 if enable else 0)
            self._check(ret, "PH_SetMultistopEnable")

    def set_binning_code(self, binning_code: int) -> float:
        """
        Set hardware binning code:
          0 = 1x base resolution, 1 = 2x, 2 = 4x, ... :contentReference[oaicite:13]{index=13}
        Returns the resulting resolution in ps.
        """
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetBinning(self.device_index, int(binning_code))
            self._check(ret, "PH_SetBinning")
            res_ps = self.get_resolution_ps()
            self._log(f"Set binning={binning_code}, resolution={res_ps:.3f} ps")
            return res_ps

    def set_target_resolution_ps(self, target_ps: float = 8.0, max_steps: int = 8) -> Tuple[int, float]:
        """
        Choose a binning code (0..max_steps-1) whose resulting resolution is closest to target_ps.
        For PH300 base resolution is typically 4 ps, so binning=1 gives 8 ps. :contentReference[oaicite:14]{index=14}
        """
        with self._lock:
            self._require_initialized()
            best = (None, None, float("inf"))  # (binning, res_ps, abs_err)

            for b in range(max_steps):
                self._ph.PH_SetBinning(self.device_index, int(b))
                # no check here; we'll check via get_resolution
                res_ps = self.get_resolution_ps()
                err = abs(res_ps - float(target_ps))
                if err < best[2]:
                    best = (b, res_ps, err)

            assert best[0] is not None
            # set to best binning definitively
            self._check(self._ph.PH_SetBinning(self.device_index, int(best[0])), "PH_SetBinning(best)")
            self._resolution_ps = best[1]
            self._log(f"Target {target_ps} ps -> binning={best[0]}, resolution={best[1]:.3f} ps")
            return int(best[0]), float(best[1])

    def set_hist_offset_ps(self, offset_ps: int) -> None:
        """Histogramming/T3 offset in ps (difference between ch1 and ch0). :contentReference[oaicite:15]{index=15}"""
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_SetOffset(self.device_index, int(offset_ps))
            self._check(ret, "PH_SetOffset")

    def clear_hist_mem(self, block: int = 0) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_ClearHistMem(self.device_index, int(block))
            self._check(ret, "PH_ClearHistMem")

    def start_meas(self, tacq_ms: int) -> None:
        with self._lock:
            self._require_initialized()
            ret = self._ph.PH_StartMeas(self.device_index, int(tacq_ms))
            self._check(ret, "PH_StartMeas")
            self._log(f"Measurement started: tacq={tacq_ms} ms")

    def stop_meas(self) -> None:
        with self._lock:
            if self._ph is None or not self._is_initialized:
                return
            ret = self._ph.PH_StopMeas(self.device_index)
            self._check(ret, "PH_StopMeas")

    def ctc_done(self) -> bool:
        with self._lock:
            self._require_initialized()
            ctc = ct.c_int(0)
            ret = self._ph.PH_CTCStatus(self.device_index, ct.byref(ctc))
            self._check(ret, "PH_CTCStatus")
            return ctc.value > 0

    def get_elapsed_ms(self) -> float:
        with self._lock:
            self._require_initialized()
            elapsed = ct.c_double(0.0)
            ret = self._ph.PH_GetElapsedMeasTime(self.device_index, ct.byref(elapsed))
            self._check(ret, "PH_GetElapsedMeasTime")
            return float(elapsed.value)

    def get_flags(self) -> int:
        with self._lock:
            self._require_initialized()
            flags = ct.c_int(0)
            ret = self._ph.PH_GetFlags(self.device_index, ct.byref(flags))
            self._check(ret, "PH_GetFlags")
            return int(flags.value)

    def get_resolution_ps(self) -> float:
        with self._lock:
            self._require_initialized()
            res = ct.c_double(0.0)
            ret = self._ph.PH_GetResolution(self.device_index, ct.byref(res))
            self._check(ret, "PH_GetResolution")
            self._resolution_ps = float(res.value)
            return float(res.value)

    def get_base_resolution_ps(self) -> float:
        with self._lock:
            self._require_initialized()
            res = ct.c_double(0.0)
            ret = self._ph.PH_GetBaseResolution(self.device_index, ct.byref(res))
            self._check(ret, "PH_GetBaseResolution")
            self._resolution_ps = float(res.value)
            return float(res.value)

    def get_count_rate_hz(self, channel: int) -> int:
        with self._lock:
            self._require_initialized()
            rate = ct.c_int(0)
            ret = self._ph.PH_GetCountRate(self.device_index, int(channel), ct.byref(rate))
            self._check(ret, f"PH_GetCountRate(ch={channel})")
            return int(rate.value)

    def get_rates_and_warnings(self) -> RateStatus:
        """
        Returns sync/ch0/ch1 rates plus warnings bitfield + text.
        Manual note: must call PH_GetCountRate for all channels prior to PH_GetWarnings. :contentReference[oaicite:16]{index=16}
        """
        with self._lock:
            self._require_initialized()

            # Count rates (gate time ~100 ms) :contentReference[oaicite:17]{index=17}
            ch0 = self.get_count_rate_hz(0)
            ch1 = self.get_count_rate_hz(1)

            # For your wiring: channel 0 is sync, but PHLib provides rate per channel.
            # We'll report sync_rate_hz == ch0_rate_hz here.
            sync = ch0

            warnings = ct.c_int(0)
            ret = self._ph.PH_GetWarnings(self.device_index, ct.byref(warnings))
            self._check(ret, "PH_GetWarnings")

            w = int(warnings.value)
            wtxt = ""
            if w != 0:
                buf = ct.create_string_buffer(WARN_TEXT_BUFSZ)
                ret = self._ph.PH_GetWarningsText(self.device_index, buf, int(w))
                self._check(ret, "PH_GetWarningsText")
                wtxt = buf.value.decode(errors="replace").strip()

            return RateStatus(
                sync_rate_hz=sync,
                ch0_rate_hz=ch0,
                ch1_rate_hz=ch1,
                warnings_bitfield=w,
                warnings_text=wtxt,
            )

    def read_histogram(self, block: int = 0):
        """
        Read histogram block (block=0 for non-routing). Returns counts as:
          - numpy array (uint32) if numpy available
          - otherwise Python list[int]
        """
        with self._lock:
            self._require_initialized()

            buf = (ct.c_uint * HISTCHAN)()
            ret = self._ph.PH_GetHistogram(self.device_index, buf, int(block))
            self._check(ret, "PH_GetHistogram")

            if np is not None:
                arr = np.ctypeslib.as_array(buf).copy()
                return arr
            return list(buf)

    def make_time_axis_ps(self) -> "np.ndarray | list[float]":
        """
        Time axis in ps for HISTCHAN bins, based on current resolution.
        """
        res_ps = self.get_resolution_ps()
        if np is not None:
            return (np.arange(HISTCHAN, dtype=np.float64) * res_ps)
        return [i * res_ps for i in range(HISTCHAN)]

    def save_traces_csv(
        self,
        filepath: str,
        time_ps,
        traces: Dict[str, "np.ndarray | list[int]"],
    ) -> None:
        """
        Save multiple traces to CSV:
          col0 = time_ps
          col1.. = traces in dict insertion order (Python 3.7+ preserves)
        """
        # Minimal dependency: write with stdlib
        import csv

        keys = list(traces.keys())
        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ps"] + keys)
            n = len(time_ps)
            for i in range(n):
                row = [time_ps[i]]
                for k in keys:
                    row.append(int(traces[k][i]))
                w.writerow(row)

    # -----------------
    # Context manager
    # -----------------
    def __enter__(self) -> "PicoHarp300":
        self.open()
        self.initialize_histogramming()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except Exception:
            pass

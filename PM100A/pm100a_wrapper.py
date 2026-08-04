import ctypes
import os
import threading
import time
from typing import List

# ---- DLL loading ----

_DLL_CANDIDATES = [
    r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLPM_64.dll",
    r"C:\Program Files (x86)\IVI Foundation\VISA\WinNT\Bin\TLPM_32.dll",
]

# VISA type aliases used by TLPM
_ViSession = ctypes.c_uint32
_ViStatus  = ctypes.c_int32
_ViBoolean = ctypes.c_uint16
_ViUInt16  = ctypes.c_uint16
_ViReal64  = ctypes.c_double
_ViInt16   = ctypes.c_int16
_ViUInt32  = ctypes.c_uint32

_VI_SUCCESS   = 0
_ATTR_SET_VAL = 0          # query the currently set value
_RSRC_BUF_LEN = 256


def _load_dll() -> ctypes.WinDLL:
    for path in _DLL_CANDIDATES:
        if os.path.exists(path):
            return ctypes.WinDLL(path)
    raise PM100AError(
        "TLPM_64.dll not found. Searched:\n"
        + "\n".join(f"  {p}" for p in _DLL_CANDIDATES)
    )


# ---- Public API ----

class PM100AError(Exception):
    pass


class PM100A:
    DEFAULT_RESOURCE = "USB0::0x1313::0x8079::P1007408::INSTR"

    @staticmethod
    def list_resources() -> List[str]:
        """Return resource strings for all connected TLPM-compatible devices."""
        try:
            dll = _load_dll()
        except PM100AError:
            return []
        count = _ViUInt32(0)
        # Passing 0 (NULL session) enumerates without an open connection
        dll.TLPM_findRsrc(_ViSession(0), ctypes.byref(count))
        resources: List[str] = []
        buf = ctypes.create_string_buffer(_RSRC_BUF_LEN)
        for i in range(count.value):
            if dll.TLPM_getRsrcName(_ViSession(0), _ViUInt32(i), buf) == _VI_SUCCESS:
                resources.append(buf.value.decode())
        return resources

    def __init__(self, resource_str: str = DEFAULT_RESOURCE):
        self._lock = threading.RLock()
        self._dll = _load_dll()
        self._vi = _ViSession(0)

        status = self._dll.TLPM_init(
            resource_str.encode(),
            _ViBoolean(1),           # ID query
            _ViBoolean(0),           # no reset
            ctypes.byref(self._vi),
        )
        if status != _VI_SUCCESS:
            available = ", ".join(PM100A.list_resources()) or "(none)"
            raise PM100AError(
                f"TLPM_init failed for '{resource_str}' (0x{status & 0xFFFFFFFF:08X})\n"
                f"Available: {available}"
            )
        # Default averaging matches the Thorlabs software display
        self._dll.TLPM_setAvgCnt(self._vi, _ViInt16(10))

    def close(self) -> None:
        with self._lock:
            if self._vi.value != 0:
                self._dll.TLPM_close(self._vi)
                self._vi = _ViSession(0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---- Wavelength ----

    def set_wavelength(self, wavelength_nm: float) -> None:
        with self._lock:
            self._chk(
                self._dll.TLPM_setWavelength(self._vi, _ViReal64(wavelength_nm)),
                "setWavelength",
            )

    def get_wavelength(self) -> float:
        with self._lock:
            val = _ViReal64(0.0)
            self._chk(
                self._dll.TLPM_getWavelength(self._vi, _ViInt16(_ATTR_SET_VAL), ctypes.byref(val)),
                "getWavelength",
            )
            return val.value

    # ---- Zero ----

    def zero(self, timeout_s: float = 30.0) -> None:
        """Measure dark offset and apply it as zero. Block all light first."""
        with self._lock:
            self._chk(self._dll.TLPM_startDarkAdjust(self._vi), "startDarkAdjust")
        # TLPMX_getDarkAdjustState: 0 = idle/done, non-zero = busy
        state = _ViUInt16(1)
        deadline = time.monotonic() + timeout_s
        while state.value != 0:
            if time.monotonic() > deadline:
                raise PM100AError("Dark adjust timed out")
            time.sleep(0.1)
            with self._lock:
                self._dll.TLPMX_getDarkAdjustState(self._vi, ctypes.byref(state))

    # ---- Range ----

    def set_range(self, range_watts: float) -> None:
        with self._lock:
            self._chk(
                self._dll.TLPM_setPowerRange(self._vi, _ViReal64(range_watts)),
                "setPowerRange",
            )

    def get_range(self) -> float:
        with self._lock:
            val = _ViReal64(0.0)
            self._chk(
                self._dll.TLPM_getPowerRange(self._vi, _ViInt16(_ATTR_SET_VAL), ctypes.byref(val)),
                "getPowerRange",
            )
            return val.value

    def set_auto_range(self, enabled: bool) -> None:
        with self._lock:
            self._chk(
                self._dll.TLPM_setPowerAutoRange(self._vi, _ViBoolean(int(enabled))),
                "setPowerAutoRange",
            )

    def get_auto_range(self) -> bool:
        with self._lock:
            val = _ViBoolean(0)
            # Note: Thorlabs DLL inconsistency — getter has lowercase 'r'
            self._chk(
                self._dll.TLPM_getPowerAutorange(self._vi, ctypes.byref(val)),
                "getPowerAutorange",
            )
            return bool(val.value)

    # ---- Measurement ----

    def measure_power(self) -> float:
        """Return instantaneous power in Watts."""
        with self._lock:
            val = _ViReal64(0.0)
            self._chk(
                self._dll.TLPM_measPower(self._vi, ctypes.byref(val)),
                "measPower",
            )
            return val.value

    def set_averaging(self, count: int) -> None:
        with self._lock:
            self._chk(
                self._dll.TLPM_setAvgCnt(self._vi, _ViInt16(count)),
                "setAvgCnt",
            )

    # ---- Beam diameter ----

    def get_beam_diameter(self) -> float:
        """Return beam diameter in mm."""
        with self._lock:
            val = _ViReal64(0.0)
            self._chk(
                self._dll.TLPM_getBeamDia(self._vi, _ViInt16(_ATTR_SET_VAL), ctypes.byref(val)),
                "getBeamDia",
            )
            return val.value * 1e3  # m → mm

    def set_beam_diameter(self, diameter_mm: float) -> None:
        """Set beam diameter in mm (used for power density calculation)."""
        with self._lock:
            self._chk(
                self._dll.TLPM_setBeamDia(self._vi, _ViReal64(diameter_mm * 1e-3)),  # mm → m
                "setBeamDia",
            )

    # ---- Internal ----

    def _chk(self, status: int, func: str) -> None:
        if status != _VI_SUCCESS:
            raise PM100AError(f"TLPM_{func} failed (0x{status & 0xFFFFFFFF:08X})")

"""Count-mode wrapper for SPAD23 over the LabVIEW TCP server."""

from __future__ import annotations

from dataclasses import dataclass
import socket
from typing import Dict, List, Optional, Tuple

import numpy as np


NUM_PIXELS = 23
PIXEL_LIMIT_MCPS_23G = 6.7
PIXEL_LIMIT_MCPS_23R = 12.5


class Spad23Error(RuntimeError):
    """Base class for SPAD23 wrapper errors."""


class Spad23ConnectionError(Spad23Error):
    """Raised for connection-level failures."""


class Spad23ProtocolError(Spad23Error):
    """Raised when the server response is malformed or reports an error."""


class Spad23OverflowError(Spad23Error):
    """Raised when the device reports an overflow or the rate exceeds a limit."""

    def __init__(self, message: str, snapshot: Optional["CountSnapshot"] = None):
        super().__init__(message)
        self.snapshot = snapshot


@dataclass(frozen=True)
class CountSnapshot:
    integration_ms: int
    pixel_counts: np.ndarray
    pixel_limit_mcps: float
    warnings: Tuple[str, ...]
    raw_response: str

    @property
    def total_counts(self) -> int:
        return int(np.sum(self.pixel_counts))

    @property
    def pixel_rates_mcps(self) -> np.ndarray:
        dt_s = self.integration_ms / 1000.0
        if dt_s <= 0:
            return np.zeros(NUM_PIXELS, dtype=np.float64)
        return self.pixel_counts.astype(np.float64) / dt_s / 1e6

    @property
    def total_rate_mcps(self) -> float:
        return float(np.sum(self.pixel_rates_mcps))

    @property
    def max_pixel_index(self) -> int:
        return int(np.argmax(self.pixel_rates_mcps))

    @property
    def max_pixel_rate_mcps(self) -> float:
        return float(np.max(self.pixel_rates_mcps))

    @property
    def overload_ratio(self) -> float:
        if self.pixel_limit_mcps <= 0:
            return 0.0
        return self.max_pixel_rate_mcps / self.pixel_limit_mcps


class Spad23CountClient:
    """
    TCP wrapper around SPAD23 count mode.

    Commands used:
    - I,<integration_ms>: read count data
    - T,v,1: check TDC calibration validity
    - T,c,1: run TDC calibration if invalid
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        timeout_s: float = 2.0,
        quiet_timeout_s: float = 0.05,
        pixel_limit_mcps: float = PIXEL_LIMIT_MCPS_23G,
        raise_on_limit: bool = False,
        raise_on_server_overflow: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.quiet_timeout_s = float(quiet_timeout_s)
        self.pixel_limit_mcps = float(pixel_limit_mcps)
        self.raise_on_limit = bool(raise_on_limit)
        self.raise_on_server_overflow = bool(raise_on_server_overflow)

        self._sock: Optional[socket.socket] = None
        self.server_banner: str = ""

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> str:
        if self._sock is not None:
            return self.server_banner

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        try:
            sock.connect((self.host, self.port))
            self._sock = sock
            self.server_banner = self._recv_until_quiet()
            return self.server_banner
        except OSError as exc:
            try:
                sock.close()
            except OSError:
                pass
            raise Spad23ConnectionError(f"Could not connect to {self.host}:{self.port}: {exc}") from exc

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        finally:
            self._sock = None

    def __enter__(self) -> "Spad23CountClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize_device(self, run_calibration_if_invalid: bool = True) -> Dict[str, str]:
        """
        Basic initialization: check TDC calibration and optionally run it.

        Returns a dict with keys:
        - tdc_status
        - calibration (present only if calibration was attempted)
        """
        self._ensure_connected()
        out: Dict[str, str] = {}

        status = self.query("T,v,1")
        out["tdc_status"] = status.strip()

        if run_calibration_if_invalid and "invalid" in status.lower():
            out["calibration"] = self.query("T,c,1").strip()
        return out

    def query(self, command: str) -> str:
        self._ensure_connected()
        payload = self._query_raw(command)
        text = payload.decode("utf8", errors="ignore")
        if "ERROR" in text.upper():
            raise Spad23ProtocolError(f"Server returned ERROR for command '{command.strip()}': {text.strip()}")
        return text

    def get_counts_snapshot(self, integration_ms: int) -> CountSnapshot:
        integration_ms = int(integration_ms)
        if integration_ms <= 0:
            raise ValueError("integration_ms must be > 0")

        text = self.query(f"I,{integration_ms}")
        snapshot = self._parse_count_response(text=text, integration_ms=integration_ms)

        has_overflow_warning = any(self._looks_like_overflow_warning(w) for w in snapshot.warnings)
        if has_overflow_warning and self.raise_on_server_overflow:
            raise Spad23OverflowError("Server reported overflow/shutdown condition.", snapshot=snapshot)
        if snapshot.overload_ratio >= 1.0 and self.raise_on_limit:
            raise Spad23OverflowError(
                f"Max pixel rate {snapshot.max_pixel_rate_mcps:.3f} Mcps exceeds "
                f"limit {snapshot.pixel_limit_mcps:.3f} Mcps.",
                snapshot=snapshot,
            )
        return snapshot

    def read_pixel_count(self, pixel: int, integration_ms: int) -> int:
        pixel = self._validate_pixel(pixel)
        snap = self.get_counts_snapshot(integration_ms)
        return int(snap.pixel_counts[pixel])

    def read_total_count(self, integration_ms: int) -> int:
        snap = self.get_counts_snapshot(integration_ms)
        return snap.total_counts

    def read_pixel_rate_mcps(self, pixel: int, integration_ms: int = 100) -> float:
        pixel = self._validate_pixel(pixel)
        snap = self.get_counts_snapshot(integration_ms)
        return float(snap.pixel_rates_mcps[pixel])

    def read_total_rate_mcps(self, integration_ms: int = 100) -> float:
        snap = self.get_counts_snapshot(integration_ms)
        return snap.total_rate_mcps

    def read_all_pixel_rates_mcps(self, integration_ms: int = 100) -> np.ndarray:
        snap = self.get_counts_snapshot(integration_ms)
        return snap.pixel_rates_mcps.copy()

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def _query_raw(self, command: str) -> bytes:
        assert self._sock is not None
        cmd = command.rstrip("\n") + "\n"
        try:
            self._sock.send(cmd.encode("utf8"))
            return self._recv_until_quiet()
        except OSError as exc:
            self.close()
            raise Spad23ConnectionError(f"Connection failed while sending '{command}': {exc}") from exc

    def _recv_until_quiet(self) -> bytes:
        assert self._sock is not None
        chunks: List[bytes] = []
        try:
            first = self._sock.recv(8192)
            if first:
                chunks.append(first)
        except socket.timeout as exc:
            raise Spad23ConnectionError("Timed out waiting for server response.") from exc

        self._sock.settimeout(self.quiet_timeout_s)
        try:
            while True:
                try:
                    chunk = self._sock.recv(8192)
                except socket.timeout:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            self._sock.settimeout(self.timeout_s)

        return b"".join(chunks)

    @staticmethod
    def _validate_pixel(pixel: int) -> int:
        pixel = int(pixel)
        if pixel < 0 or pixel >= NUM_PIXELS:
            raise ValueError(f"pixel must be in [0, {NUM_PIXELS - 1}], got {pixel}")
        return pixel

    @staticmethod
    def _normalize_pixel_id(raw_pixel_id: int) -> Optional[int]:
        if 0 <= raw_pixel_id < NUM_PIXELS:
            return raw_pixel_id
        if 32 <= raw_pixel_id <= 54:
            return raw_pixel_id - 32
        return None

    @staticmethod
    def _looks_like_overflow_warning(line: str) -> bool:
        lower = line.lower()
        return ("overflow" in lower) or ("shutdown" in lower)

    def _parse_count_response(self, text: str, integration_ms: int) -> CountSnapshot:
        counts = np.zeros(NUM_PIXELS, dtype=np.int64)
        warnings: List[str] = []
        parsed_any = False

        for raw_line in text.replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper == "DONE":
                continue
            if "ERROR" in upper:
                raise Spad23ProtocolError(f"Server reported ERROR in count response: {line}")

            if self._looks_like_overflow_warning(line):
                warnings.append(line)
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                warnings.append(f"Unparsed line: {line}")
                continue

            try:
                raw_pixel_id = int(parts[0])
                count_val = int(parts[1])
            except ValueError:
                warnings.append(f"Unparsed line: {line}")
                continue

            pixel = self._normalize_pixel_id(raw_pixel_id)
            if pixel is None:
                warnings.append(f"Unknown pixel id {raw_pixel_id}: {line}")
                continue

            counts[pixel] += max(0, count_val)
            parsed_any = True

        if not parsed_any:
            raise Spad23ProtocolError(
                "Could not parse any pixel counts from server response. "
                f"Raw response was: {text.strip()}"
            )

        return CountSnapshot(
            integration_ms=integration_ms,
            pixel_counts=counts,
            pixel_limit_mcps=self.pixel_limit_mcps,
            warnings=tuple(warnings),
            raw_response=text,
        )

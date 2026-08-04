"""TCSPC timestamping wrapper for SPAD23 over the LabVIEW TCP server."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from spad23.spad23_wrapper import NUM_PIXELS, Spad23ConnectionError, Spad23Error, Spad23ProtocolError


UpdateCallback = Callable[["TrplSnapshot"], None]


@dataclass(frozen=True)
class TrplSnapshot:
    measurement_ms: int
    bin_width_ps: int
    counts_matrix: np.ndarray  # shape: [time_bins, pixels]
    warnings: Tuple[str, ...]
    elapsed_s: float
    done: bool

    @property
    def time_axis_ps(self) -> np.ndarray:
        return np.arange(self.counts_matrix.shape[0], dtype=np.int64) * self.bin_width_ps

    @property
    def pixel_totals(self) -> np.ndarray:
        return np.sum(self.counts_matrix, axis=0).astype(np.int64)

    @property
    def total_counts(self) -> int:
        return int(np.sum(self.counts_matrix))


class _TrplAccumulator:
    def __init__(self, bin_width_ps: int):
        self.bin_width_ps = int(bin_width_ps)
        self._matrix = np.zeros((256, NUM_PIXELS), dtype=np.int64)
        self._max_bin = 0
        self.warnings: List[str] = []

    @staticmethod
    def _normalize_pixel_id(raw_pixel_id: int) -> Optional[int]:
        if 0 <= raw_pixel_id < NUM_PIXELS:
            return raw_pixel_id
        if 32 <= raw_pixel_id <= 54:
            return raw_pixel_id - 32
        return None

    def _ensure_bin(self, bin_idx: int) -> None:
        if bin_idx < self._matrix.shape[0]:
            return
        new_n = self._matrix.shape[0]
        while new_n <= bin_idx:
            new_n *= 2
        grown = np.zeros((new_n, NUM_PIXELS), dtype=np.int64)
        grown[: self._matrix.shape[0], :] = self._matrix
        self._matrix = grown

    def _add_event(self, pixel: int, fine_ts_ps: int) -> None:
        if fine_ts_ps < 0:
            return
        bin_idx = int(fine_ts_ps // self.bin_width_ps)
        self._ensure_bin(bin_idx)
        self._matrix[bin_idx, pixel] += 1
        if bin_idx > self._max_bin:
            self._max_bin = bin_idx

    def consume_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        upper = text.upper()
        if "DONE" in upper:
            return
        if "ERROR" in upper:
            raise Spad23ProtocolError(f"Server returned ERROR: {text}")

        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 3:
            # Marker lines can be shorter; keep only meaningful warnings.
            if "OVERFLOW" in upper:
                self.warnings.append(text)
            return

        try:
            raw_pixel_id = int(parts[0])
        except ValueError:
            self.warnings.append(f"Unparsed line: {text}")
            return

        pixel = self._normalize_pixel_id(raw_pixel_id)
        if pixel is None:
            # Known markers, mostly for diagnostics.
            if raw_pixel_id in (9, 10, 12, 17, 55):
                if raw_pixel_id == 17:
                    self.warnings.append("FIFO overflow marker (17) detected")
                return
            self.warnings.append(f"Unknown pixel id {raw_pixel_id}: {text}")
            return

        try:
            fine_ts = int(parts[2])
        except ValueError:
            self.warnings.append(f"Invalid fine timestamp: {text}")
            return
        self._add_event(pixel, fine_ts)

    def snapshot(self, measurement_ms: int, elapsed_s: float, done: bool) -> TrplSnapshot:
        used = self._max_bin + 1
        matrix = self._matrix[:used, :].copy()
        return TrplSnapshot(
            measurement_ms=measurement_ms,
            bin_width_ps=self.bin_width_ps,
            counts_matrix=matrix,
            warnings=tuple(self.warnings),
            elapsed_s=float(elapsed_s),
            done=bool(done),
        )


class Spad23TcspcClient:
    """
    Wrapper for timestamping/TCSPC-relevant SPAD23 commands.

    Commands used:
    - T,v,1 : verify TDC calibration state
    - T,c,1 : run TDC calibration
    - T,a,<ms> : channel alignment using synchronous source
    - S,<ms> : text timestamp stream
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        timeout_s: float = 5.0,
        quiet_timeout_s: float = 0.05,
        recv_bytes: int = 2_097_152,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.quiet_timeout_s = float(quiet_timeout_s)
        self.recv_bytes = int(recv_bytes)
        self._sock: Optional[socket.socket] = None
        self.server_banner = ""

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
            banner = self._recv_until_quiet().decode("utf8", errors="ignore")
            self.server_banner = banner.strip()
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

    def __enter__(self) -> "Spad23TcspcClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def check_tdc_calibration(self) -> str:
        return self.query("T,v,1").strip()

    def run_tdc_calibration(self) -> str:
        return self.query("T,c,1").strip()

    def run_channel_alignment(self, align_ms: int = 1000) -> str:
        align_ms = int(align_ms)
        if align_ms <= 0:
            raise ValueError("align_ms must be > 0")
        return self.query(f"T,a,{align_ms}").strip()

    def calibrate_and_align(self, align_ms: int = 1000, calibrate_if_invalid: bool = True) -> List[str]:
        messages: List[str] = []
        status = self.check_tdc_calibration()
        messages.append(f"TDC status: {status}")
        if calibrate_if_invalid and "invalid" in status.lower():
            messages.append(f"TDC calibration: {self.run_tdc_calibration()}")
        messages.append(f"Alignment: {self.run_channel_alignment(align_ms=align_ms)}")
        return messages

    def query(self, command: str) -> str:
        self._ensure_connected()
        payload = self._query_raw(command)
        text = payload.decode("utf8", errors="ignore")
        if "ERROR" in text.upper():
            raise Spad23ProtocolError(f"Server returned ERROR for '{command.strip()}': {text.strip()}")
        return text

    def acquire_trpl_stream(
        self,
        measurement_ms: int,
        bin_width_ps: int = 10,
        on_update: Optional[UpdateCallback] = None,
        update_interval_s: float = 0.5,
    ) -> TrplSnapshot:
        """
        Start text stream acquisition and bin fine timestamps into a [time x pixel] matrix.
        """
        measurement_ms = int(measurement_ms)
        bin_width_ps = int(bin_width_ps)
        if measurement_ms <= 0:
            raise ValueError("measurement_ms must be > 0")
        if bin_width_ps <= 0:
            raise ValueError("bin_width_ps must be > 0")

        self._ensure_connected()
        accum = _TrplAccumulator(bin_width_ps=bin_width_ps)

        self._send_command(f"S,{measurement_ms}")
        buffer = ""
        start = time.time()
        last_emit = start
        done = False

        while True:
            assert self._sock is not None
            try:
                data = self._sock.recv(self.recv_bytes)
            except socket.timeout as exc:
                raise Spad23ConnectionError("Timed out while receiving stream data.") from exc

            if not data:
                break
            if b"ERROR" in data:
                tail = data[-160:].decode("utf8", errors="ignore")
                raise Spad23ProtocolError(f"Stream terminated with ERROR: {tail}")

            text = data.decode("utf8", errors="ignore")
            buffer += text
            lines = buffer.split("\n")
            buffer = lines.pop() if lines else ""
            for line in lines:
                if "DONE" in line.upper():
                    done = True
                    continue
                accum.consume_line(line)

            now = time.time()
            if on_update is not None and (now - last_emit) >= update_interval_s:
                on_update(accum.snapshot(measurement_ms=measurement_ms, elapsed_s=now - start, done=False))
                last_emit = now

            if b"DONE" in data or done:
                done = True
                break

        if buffer.strip() and "DONE" not in buffer.upper():
            accum.consume_line(buffer)

        final = accum.snapshot(measurement_ms=measurement_ms, elapsed_s=time.time() - start, done=done)
        if on_update is not None:
            on_update(final)
        return final

    def _send_command(self, command: str) -> None:
        assert self._sock is not None
        cmd = command.rstrip("\n") + "\n"
        try:
            self._sock.send(cmd.encode("utf8"))
        except OSError as exc:
            self.close()
            raise Spad23ConnectionError(f"Connection failed while sending '{command}': {exc}") from exc

    def _query_raw(self, command: str) -> bytes:
        self._send_command(command)
        return self._recv_until_quiet()

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

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


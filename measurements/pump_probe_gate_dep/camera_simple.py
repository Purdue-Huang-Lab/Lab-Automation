"""
camera_simple.py
Simple camera module for capturing frames.

Does NOT change any camera settings — preserves everything from PixeLink Capture
(pixel format, trigger, exposure, gain, ROI, etc.).
Auto-detects MONO8 vs MONO16 from the frame descriptor.
PixeLink cameras output MONO16 data in big-endian byte order.
"""

import numpy as np
import ctypes
import threading
import queue
import time
from pixelinkWrapper import PxLApi


class SimpleCamera:
    def __init__(self):
        self.h_camera = None
        self.is_initialized = False
        self.stream_running = False
        self.frame_desc = None
        self.raw_image_size = None
        self.height = None
        self.width = None
        self.frame_buffer = None
        self.capture_thread = None
        self.capture_thread_running = False
        self.capture_queue = None
        self.capture_timeout = 0.5
        self.dtype = np.uint16
        self.bytes_per_pixel = 2

    def initialize(self):
        """
        Initialize camera without changing any settings.

        All camera configuration (pixel format, trigger, exposure, gain, etc.)
        should be set beforehand in PixeLink Capture.
        """
        try:
            ret = PxLApi.initialize(0)
            if not PxLApi.apiSuccess(ret[0]):
                raise RuntimeError(f"Camera initialization failed: {ret[0]}")

            self.h_camera = ret[1]
            self.is_initialized = True
            print("Camera initialized (using PixeLink Capture settings)")

            self._report_trigger()
            self._ensure_streaming()
            return True
        except Exception as e:
            self.is_initialized = False
            raise e

    def _report_trigger(self):
        """Report current trigger configuration (read-only, does not change anything)."""
        try:
            ret = PxLApi.getFeature(self.h_camera, PxLApi.FeatureId.TRIGGER)
            if PxLApi.apiSuccess(ret[0]) and len(ret) >= 3:
                params = list(ret[2])
                mode = params[0] if len(params) > 0 else 0
                ttype = params[1] if len(params) > 1 else 0
                kind = "Hardware" if ttype >= 1 else "Free-run"
                print(f"  Trigger: {kind}, mode={mode:.0f}, type={ttype:.0f}")
        except Exception as e:
            print(f"  Could not read trigger: {e}")

    def _ensure_streaming(self):
        """Start streaming and cache frame descriptor / buffer."""
        if not self.is_initialized:
            raise RuntimeError("Camera not initialized")

        if not self.stream_running:
            ret = PxLApi.setStreamState(self.h_camera, PxLApi.StreamState.START)
            if not PxLApi.apiSuccess(ret[0]):
                raise RuntimeError("Failed to start streaming")
            self.stream_running = True

        if self.frame_desc is None or self.frame_buffer is None:
            ret_temp = PxLApi.getNextFrame(self.h_camera)
            if not PxLApi.apiSuccess(ret_temp[0]):
                raise RuntimeError("Could not get frame descriptor")

            self.frame_desc = ret_temp[1]
            self.raw_image_size = PxLApi.imageSize(self.frame_desc)
            self.height = int(self.frame_desc.Roi.fHeight /
                              self.frame_desc.PixelAddressingValue.fVertical)
            self.width = int(self.frame_desc.Roi.fWidth /
                             self.frame_desc.PixelAddressingValue.fHorizontal)
            self.frame_buffer = ctypes.create_string_buffer(self.raw_image_size)

            expected_pixels = self.height * self.width
            if self.raw_image_size >= expected_pixels * 2:
                self.dtype = np.uint16
                self.bytes_per_pixel = 2
            else:
                self.dtype = np.uint8
                self.bytes_per_pixel = 1
            fmt = "MONO16" if self.bytes_per_pixel == 2 else "MONO8"
            print(f"  Streaming: {self.width}x{self.height} {fmt} "
                  f"(buf={self.raw_image_size})")

    def _decode_frame(self, buf):
        """Convert raw frame buffer to numpy array with correct byte order."""
        if self.bytes_per_pixel == 2:
            # PixeLink outputs MONO16 in big-endian byte order
            return (np.frombuffer(buf, dtype=np.uint16)
                    .reshape((self.height, self.width))
                    .byteswap())
        return (np.frombuffer(buf, dtype=np.uint8)
                .reshape((self.height, self.width)))

    def capture_frames(self, n_frames, verbose=False, timeout=1.0):
        """
        Capture N frames as fast as possible.

        Returns:
            list of numpy arrays (one per frame)
        """
        self._ensure_streaming()

        if self.capture_thread_running and self.capture_queue is not None:
            frames = []
            for i in range(n_frames):
                try:
                    frame = self.capture_queue.get(timeout=timeout)
                    frames.append(frame)
                except queue.Empty:
                    raise TimeoutError("Timed out waiting for frame from async buffer")
                if verbose and (i % 50 == 0 or i == n_frames - 1):
                    print(f"Frame {i+1}/{n_frames} (async)")
            return frames

        frames = np.zeros((n_frames, self.height, self.width), dtype=self.dtype)

        for i in range(n_frames):
            ret = PxLApi.getNextFrame(self.h_camera, self.frame_buffer)

            if PxLApi.apiSuccess(ret[0]):
                frames[i] = self._decode_frame(self.frame_buffer)
                if verbose and (i % 50 == 0 or i == n_frames - 1):
                    print(f"Frame {i+1}/{n_frames}")
            else:
                raise RuntimeError(f"Failed to get frame: {ret[0]}")

        if verbose:
            print(f"Captured {n_frames} frames")

        return [frames[i] for i in range(n_frames)]

    def start_async_capture(self, buffer_size=200):
        """Start background thread that continuously captures into a queue."""
        self._ensure_streaming()
        if self.capture_thread_running:
            return
        self.capture_queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread_running = True
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def _capture_worker(self):
        while self.capture_thread_running:
            ret = PxLApi.getNextFrame(self.h_camera, self.frame_buffer)
            if PxLApi.apiSuccess(ret[0]):
                frame = self._decode_frame(self.frame_buffer).copy()
                try:
                    self.capture_queue.put(frame, timeout=self.capture_timeout)
                except queue.Full:
                    continue
            else:
                time.sleep(0.001)

    def stop_async_capture(self):
        if not self.capture_thread_running:
            return
        self.capture_thread_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        self.capture_thread = None
        self.capture_queue = None

    def close(self):
        self.stop_async_capture()
        if self.stream_running and self.h_camera:
            PxLApi.setStreamState(self.h_camera, PxLApi.StreamState.STOP)
            self.stream_running = False
        if self.h_camera and self.is_initialized:
            PxLApi.uninitialize(self.h_camera)
            self.h_camera = None
            self.is_initialized = False
            self.frame_desc = None
            self.frame_buffer = None

    def __del__(self):
        self.close()


def analyze_frames(frames, roi_x, roi_y, roi_w, roi_h, verbose=False):
    """
    Analyze captured frames and classify as pump on/off.

    Args:
        frames: List of numpy arrays (captured frames)
        roi_x, roi_y, roi_w, roi_h: Region of interest coordinates
        verbose: If True, print detailed analysis info

    Returns:
        dict with analysis results
    """
    if isinstance(frames, list):
        frames_array = np.array(frames)
    else:
        frames_array = frames

    n_frames = len(frames)
    frame_h, frame_w = frames[0].shape

    x = max(0, min(roi_x, frame_w - 1))
    y = max(0, min(roi_y, frame_h - 1))
    w = min(roi_w, frame_w - x)
    h = min(roi_h, frame_h - y)

    ref_intensities = np.mean(frames_array[:, y:y+h, x:x+w], axis=(1, 2))

    avg_ref = np.mean(ref_intensities)

    pump_states = ref_intensities > avg_ref

    pump_on_indices = np.where(pump_states)[0]
    pump_off_indices = np.where(~pump_states)[0]

    avg_pump_on = None
    avg_pump_off = None

    if len(pump_on_indices) > 0:
        avg_pump_on = np.mean(frames_array[pump_on_indices], axis=0)

    if len(pump_off_indices) > 0:
        avg_pump_off = np.mean(frames_array[pump_off_indices], axis=0)

    if verbose:
        print(f"Analyzed {n_frames} frames: {len(pump_on_indices)} ON, "
              f"{len(pump_off_indices)} OFF (thr={avg_ref:.1f})")

    return {
        'avg_pump_on': avg_pump_on,
        'avg_pump_off': avg_pump_off,
        'n_pump_on': len(pump_on_indices),
        'n_pump_off': len(pump_off_indices),
        'threshold': avg_ref,
        'ref_intensities': ref_intensities.tolist(),
        'pump_states': pump_states.tolist()
    }

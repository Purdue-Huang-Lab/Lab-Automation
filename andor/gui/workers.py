import time
from typing import Optional

from PyQt5 import QtCore
import numpy as np


class LiveAcqThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)

    def __init__(self, cam, parent=None):
        super().__init__(parent)
        self.cam = cam
        self._stop = False
        self._min_interval_s = 0.02

    def stop(self):
        self._stop = True
        try:
            if hasattr(self.cam, "_recover_after_timeout"):
                self.cam._recover_after_timeout()
        except Exception:
            pass
        try:
            if hasattr(self.cam, "stop_stream"):
                self.cam.stop_stream()
        except Exception:
            pass
        try:
            inner = getattr(self.cam, "cam", None)
            for fn in ("abort_acquisition", "abort_acq", "abort", "stop_acquisition", "stop_acq", "stop"):
                if inner is not None and hasattr(inner, fn):
                    try:
                        getattr(inner, fn)()
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if hasattr(self.cam, "_recover_after_timeout"):
                self.cam._recover_after_timeout()
        except Exception:
            pass
        try:
            if hasattr(self.cam, "stop_stream"):
                self.cam.stop_stream()
        except Exception:
            pass
        try:
            inner = getattr(self.cam, "cam", None)
            for fn in ("abort_acquisition", "abort_acq", "abort", "stop_acquisition", "stop_acq", "stop"):
                if inner is not None and hasattr(inner, fn):
                    try:
                        getattr(inner, fn)()
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    def run(self):
        self._stop = False
        if self.cam is None:
            self.status.emit("No camera object.")
            return
        while not self._stop:
            loop_start = time.monotonic()
            try:
                fr = self.cam.get_frame()
            except Exception as exc:
                if self._stop:
                    return
                self.status.emit(f"get_frame failed: {exc}")
                time.sleep(max(0.05, self._min_interval_s))
                continue
            if fr.get("ok"):
                self.frame_ready.emit(fr)
            else:
                err = fr.get("err", "get_frame failed")
                if self._stop:
                    return
                self.status.emit(str(err))
            elapsed = time.monotonic() - loop_start
            sleep_s = self._min_interval_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)


class SingleAcqThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)

    def __init__(self, cam, accum_n: int, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.accum_n = int(accum_n)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        if self.cam is None:
            self.status.emit("No camera object.")
            return
        prev_backend = None
        prev_acq = None
        prev_trig = None
        prev_timeout_scale = None
        try:
            if hasattr(self.cam, "get_frame_api"):
                prev_backend = self.cam.get_frame_api()
        except Exception:
            prev_backend = None
        try:
            if hasattr(self.cam, "get_acquisition_mode"):
                prev_acq = self.cam.get_acquisition_mode()
        except Exception:
            prev_acq = None
        try:
            if hasattr(self.cam, "get_trigger_mode"):
                prev_trig = self.cam.get_trigger_mode()
        except Exception:
            prev_trig = None
        try:
            prev_timeout_scale = getattr(self.cam, "_timeout_scale", None)
        except Exception:
            prev_timeout_scale = None

        try:
            if hasattr(self.cam, "stop_stream"):
                self.cam.stop_stream()
        except Exception:
            pass
        try:
            if hasattr(self.cam, "set_frame_api"):
                self.cam.set_frame_api("Snap")
        except Exception:
            pass
        try:
            if hasattr(self.cam, "set_acquisition_mode"):
                self.cam.set_acquisition_mode("Single")
        except Exception:
            pass
        try:
            if hasattr(self.cam, "set_trigger_mode"):
                self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            if hasattr(self.cam, "set_shutter"):
                self.cam.set_shutter("auto")
        except Exception:
            pass
        try:
            exp_ms = None
            if hasattr(self.cam, "get_exposure_ms"):
                try:
                    exp_ms = float(self.cam.get_exposure_ms())
                except Exception:
                    exp_ms = None
            timeout_scale = 2.0
            if exp_ms is not None and exp_ms > 10000.0:
                timeout_scale = max(timeout_scale, (exp_ms / 10000.0) * 2.0)
            self.cam._timeout_scale = timeout_scale
        except Exception:
            pass

        n = max(1, int(self.accum_n))
        sum_img = None
        last_fr = None

        for i in range(1, n + 1):
            retries = 1
            if self._stop:
                self.status.emit("Accumulation stopped")
                return
            while True:
                try:
                    fr = self.cam.get_frame()
                except Exception as exc:
                    if self._stop:
                        self.status.emit("Accumulation stopped")
                        return
                    self.status.emit(f"Accum failed: {exc}")
                    return
                if fr.get("ok"):
                    break
                err = str(fr.get("err", "Accum failed"))
                if self._stop:
                    self.status.emit("Accumulation stopped")
                    return
                if ("timeout" in err.lower()) and (retries > 0):
                    retries -= 1
                    try:
                        timeout_scale = float(getattr(self.cam, "_timeout_scale", 2.0))
                        timeout_scale = min(timeout_scale * 2.0, 50.0)
                        self.cam._timeout_scale = timeout_scale
                        self.status.emit(f"Timeout; retrying with scale {timeout_scale:g}")
                        continue
                    except Exception:
                        self.status.emit(err)
                        return
                self.status.emit(err)
                return
            img = fr.get("image")
            if img is None:
                self.status.emit("Accum failed: no image data")
                return
            if sum_img is None:
                sum_img = np.asarray(img, dtype=np.float64)
            else:
                sum_img += np.asarray(img, dtype=np.float64)

            last_fr = fr
            fr_out = dict(fr)
            fr_out["ok"] = True
            fr_out["image"] = sum_img
            fr_out["image8"] = None
            fr_out["accum_idx"] = i
            fr_out["accum_n"] = n
            self.frame_ready.emit(fr_out)
            self.status.emit(f"Accumulation {i}/{n}")

        if last_fr is None:
            self.status.emit("Accum failed: no frames acquired")
            return
        self.status.emit("Accumulation done")

        try:
            if prev_backend is not None and hasattr(self.cam, "set_frame_api"):
                self.cam.set_frame_api(prev_backend)
        except Exception:
            pass
        try:
            if prev_acq is not None and hasattr(self.cam, "set_acquisition_mode"):
                if "accum" in str(prev_acq).lower():
                    self.cam.set_acquisition_mode("single")
                else:
                    self.cam.set_acquisition_mode(prev_acq)
        except Exception:
            pass
        try:
            if prev_trig is not None and hasattr(self.cam, "set_trigger_mode"):
                self.cam.set_trigger_mode(prev_trig)
        except Exception:
            pass
        try:
            if prev_timeout_scale is not None:
                self.cam._timeout_scale = prev_timeout_scale
            else:
                self.cam._timeout_scale = 1.0
        except Exception:
            pass

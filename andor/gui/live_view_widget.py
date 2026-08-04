from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .config import DEFAULT_LINECUT_WIDTH


class AndorLiveViewWidget(QtWidgets.QGroupBox):
    linecut_changed = QtCore.pyqtSignal(int)
    cursor_changed = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        cam=None,
        *,
        title: str = "Live View",
        default_linecut_width: int = DEFAULT_LINECUT_WIDTH,
        parent=None,
    ):
        super().__init__(title, parent)
        self.cam = cam

        self._last_frame_raw = None
        self._last_frame = None
        self._last_frame8 = None
        self._last_frame_full = None
        self._last_frame8_full = None
        self._roi = None
        self._roi_patch = None
        self._cursor_rc = None
        self._wl_axis = None
        self._last_raw_width = None
        self._xaxis_mode = "pixel"
        self._crop = (0, 0, 0, 0)

        self._build_ui(default_linecut_width)

    # -----------------
    # UI
    # -----------------
    def _build_ui(self, default_linecut_width: int):
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.fig = Figure(figsize=(7, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[4, 1.2], wspace=0.25, hspace=0.25)
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self._ax_img_top = self.ax_img.twiny()
        self._ax_img_top.set_visible(False)
        self._ax_img_top.set_xlabel("Wavelength (nm)")
        self._ax_img_top.tick_params(axis="x", labelsize=8)
        self.ax_v = self.fig.add_subplot(gs[0, 1])
        self.ax_h = self.fig.add_subplot(gs[1, 0])
        self.ax_h.set_xlabel("X (px)")
        self.ax_v.set_ylabel("Y (px)")

        self.im = self.ax_img.imshow(np.zeros((10, 10)), cmap="gray", origin="upper", interpolation="nearest", aspect="auto")
        self._crosshair_h = self.ax_img.axhline(0, color="white", lw=0.6, alpha=0.9)
        self._crosshair_v = self.ax_img.axvline(0, color="white", lw=0.6, alpha=0.9)
        self._crosshair_h.set_visible(False)
        self._crosshair_v.set_visible(False)
        self.line_v, = self.ax_v.plot([], [])
        self.line_h, = self.ax_h.plot([], [])
        self.ax_img.set_title("Live")

        layout.addWidget(self.canvas, 0, 0, 2, 1)

        side = QtWidgets.QVBoxLayout()
        self.cursorLbl = QtWidgets.QLabel("Cursor: (row, col) = --, --")
        self.linecutWidthSpin = QtWidgets.QSpinBox()
        self.linecutWidthSpin.setRange(1, 999)
        self.linecutWidthSpin.setValue(int(default_linecut_width))
        side.addWidget(self.cursorLbl)
        side.addWidget(QtWidgets.QLabel("Linecut width (px):"))
        side.addWidget(self.linecutWidthSpin)
        side.addStretch()
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        layout.addWidget(side_widget, 0, 1)

        self.canvas.mpl_connect("button_press_event", self.on_image_click)

    # -----------------
    # Public API
    # -----------------
    def set_camera(self, cam) -> None:
        self.cam = cam

    def set_crop(self, top: int, bottom: int, left: int, right: int) -> None:
        self._crop = (int(top), int(bottom), int(left), int(right))
        if self._last_frame_full is not None:
            fr = {"image": self._last_frame_full, "image8": self._last_frame8_full}
            self.update_frame(fr)

    def set_linecut_row(self, row: int) -> None:
        try:
            row = int(row)
        except Exception:
            return
        if self._last_frame is None:
            self._cursor_rc = (row, 0)
            return
        h, w = self._last_frame.shape
        row = max(0, min(h - 1, row))
        col = self._cursor_rc[1] if self._cursor_rc else int(w // 2)
        self._cursor_rc = (row, col)
        self._update_cursor_overlays(h, w)
        self.canvas.draw_idle()
        self.linecut_changed.emit(row)

    def set_roi(self, x1: int, x2: int, y1: int, y2: int) -> None:
        try:
            x1i = int(round(x1))
            x2i = int(round(x2))
            y1i = int(round(y1))
            y2i = int(round(y2))
        except Exception:
            return
        self._roi = (x1i, x2i, y1i, y2i)
        self._update_roi_overlay()

    def clear_roi(self) -> None:
        self._roi = None
        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except Exception:
                pass
            self._roi_patch = None
        self.canvas.draw_idle()

    def roi_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi

    def linecut_row(self) -> Optional[int]:
        if self._cursor_rc is None:
            return None
        return int(self._cursor_rc[0])

    def linecut_width(self) -> int:
        try:
            return int(self.linecutWidthSpin.value())
        except Exception:
            return 1

    def get_wavelength_axis(self):
        return self._wl_axis

    def has_wavelength_axis(self) -> bool:
        return self._xaxis_mode == "wavelength"

    def get_linecut_axis(self):
        if self._last_frame_raw is None:
            return None
        return self._get_x_axis_data(self._last_frame_raw.shape[1])

    def prepare_display_image(self, raw):
        return self._prepare_display_image(raw)

    def refresh_wavelength_axis(self, force: bool = False):
        wl = self._update_wavelength_axis_cache(force=bool(force))
        if self._last_frame is not None:
            h, w = self._last_frame.shape
            self._maybe_update_xaxis_label(w)
            self._update_image_wavelength_axis(w)
            self._update_cursor_overlays(h, w)
            self.canvas.draw_idle()
        return wl

    def set_wavelength_axis_enabled(self, enabled: bool) -> None:
        if enabled:
            self.refresh_wavelength_axis(force=True)
            return
        self._wl_axis = None
        if self._last_frame is not None:
            h, w = self._last_frame.shape
            self._maybe_update_xaxis_label(w)
            self._update_image_wavelength_axis(w)
            self._update_cursor_overlays(h, w)
            self.canvas.draw_idle()

    def update_frame(self, fr: dict):
        img = fr.get("image")
        img8 = fr.get("image8")
        if img is None:
            return
        raw = np.asarray(img)
        if raw.ndim != 2 or raw.size == 0:
            return
        self._last_frame_full = raw
        self._last_raw_width = int(raw.shape[1])
        raw_crop = self._apply_crop(raw)
        disp = np.fliplr(np.flipud(raw_crop))
        disp8 = None
        if img8 is not None:
            try:
                raw8 = np.asarray(img8)
                self._last_frame8_full = raw8
                raw8_crop = self._apply_crop(raw8)
                disp8 = np.fliplr(np.flipud(raw8_crop))
            except Exception:
                disp8 = None
                self._last_frame8_full = None
        else:
            self._last_frame8_full = None

        h, w = disp.shape
        self._maybe_update_xaxis_label(w)
        self._update_image_wavelength_axis(w)

        self._last_frame_raw = raw_crop
        self._last_frame = disp
        self._last_frame8 = disp8

        if disp8 is not None:
            self.im.set_data(disp8)
            self.im.set_clim(0, 255)
        else:
            self.im.set_data(disp)
            try:
                vmin = float(np.nanmin(disp))
                vmax = float(np.nanmax(disp))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.im.set_clim(vmin, vmax)
            except Exception:
                pass

        self.im.set_extent((0, w, h, 0))
        self.ax_img.set_xlim(0, w)
        self.ax_img.set_ylim(h, 0)
        self.ax_h.set_xlim(0, w)
        self.ax_v.set_ylim(h, 0)
        self._update_cursor_overlays(h, w)
        self._update_roi_overlay()
        self.canvas.draw_idle()

    # -----------------
    # Events
    # -----------------
    def on_image_click(self, event):
        if event.inaxes not in (self.ax_img, self._ax_img_top):
            return
        if self._last_frame is None or self._last_frame_raw is None:
            return
        try:
            if event.inaxes == self.ax_img:
                xdata, ydata = event.xdata, event.ydata
            else:
                xdata, ydata = self.ax_img.transData.inverted().transform((event.x, event.y))
            x = int(round(xdata))
            y = int(round(ydata))
        except Exception:
            return
        h, w = self._last_frame.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        self._cursor_rc = (y, x)
        self._update_cursor_overlays(h, w)
        self.canvas.draw_idle()
        self.linecut_changed.emit(int(y))
        self.cursor_changed.emit(int(y), int(x))

    # -----------------
    # Helpers
    # -----------------
    def _apply_crop(self, arr) -> np.ndarray:
        if arr is None:
            return arr
        try:
            a = np.asarray(arr)
        except Exception:
            return arr
        if a.ndim != 2:
            return a
        h, w = a.shape
        top, bottom, left, right = self._crop
        top = max(0, min(int(top), h - 1))
        bottom = max(0, min(int(bottom), h - 1))
        left = max(0, min(int(left), w - 1))
        right = max(0, min(int(right), w - 1))
        y2 = max(top + 1, h - bottom)
        x2 = max(left + 1, w - right)
        return a[top:y2, left:x2]

    def _prepare_display_image(self, raw):
        return np.fliplr(np.flipud(self._apply_crop(raw)))

    def _update_cursor_overlays(self, h: int, w: int) -> None:
        if self._cursor_rc is None or self._last_frame_raw is None:
            self._crosshair_h.set_visible(False)
            self._crosshair_v.set_visible(False)
            return
        y, x = self._cursor_rc
        if not ((0 <= x < w) and (0 <= y < h)):
            self._crosshair_h.set_visible(False)
            self._crosshair_v.set_visible(False)
            return

        y_raw = h - 1 - y
        if not (0 <= y_raw < h):
            self._crosshair_h.set_visible(False)
            self._crosshair_v.set_visible(False)
            return

        x_raw = w - 1 - x  # display col x maps to physical col x_raw after fliplr
        try:
            val = float(self._last_frame_raw[y_raw, x_raw])
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}, I = {val:g}")
        except Exception:
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}")

        self._crosshair_h.set_ydata([y, y])
        self._crosshair_v.set_xdata([x, x])
        self._crosshair_h.set_visible(True)
        self._crosshair_v.set_visible(True)

        width = int(self.linecutWidthSpin.value())
        if self.cam is not None and hasattr(self.cam, "linecut_horizontal"):
            hcut = self.cam.linecut_horizontal(self._last_frame_raw, y_raw, width=width, mode="sum")
        else:
            hcut = self._linecut_horizontal(self._last_frame_raw, y_raw, width=width)
        if self.cam is not None and hasattr(self.cam, "linecut_vertical"):
            vcut = self.cam.linecut_vertical(self._last_frame_raw, x_raw, width=width, mode="sum")
        else:
            vcut = self._linecut_vertical(self._last_frame_raw, x_raw, width=width)

        if hcut is not None:
            x_axis = self._get_x_axis_data(w)
            self.line_h.set_data(x_axis, hcut[::-1])  # reverse to match fliplr display
            self.ax_h.relim()
            self.ax_h.autoscale_view()
            if len(x_axis):
                self.ax_h.set_xlim(float(x_axis[0]), float(x_axis[-1]))
        if vcut is not None:
            self.line_v.set_data(vcut, np.arange(vcut.size))
            self.ax_v.relim()
            self.ax_v.autoscale_view()

    def _update_roi_overlay(self) -> None:
        if self._last_frame is None:
            return
        h, w = self._last_frame.shape
        if self._roi is None:
            if self._roi_patch is not None:
                try:
                    self._roi_patch.remove()
                except Exception:
                    pass
                self._roi_patch = None
            return
        x1, x2, y1, y2 = self._roi
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if self._roi_patch is None:
            self._roi_patch = Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor="#00e5ff",
                linewidth=1.0,
            )
            self.ax_img.add_patch(self._roi_patch)
        else:
            self._roi_patch.set_xy((x1, y1))
            self._roi_patch.set_width(width)
            self._roi_patch.set_height(height)
    @staticmethod
    def _linecut_horizontal(image, row: int, width: int = 1):
        a = np.asarray(image)
        if a.ndim != 2:
            return None
        h, _ = a.shape
        width = max(1, int(width))
        half = width // 2
        r1 = max(0, int(row) - half)
        r2 = min(h, int(row) + half + (1 if width % 2 else 0))
        sl = a[r1:r2, :]
        return sl.sum(axis=0)

    @staticmethod
    def _linecut_vertical(image, col: int, width: int = 1):
        a = np.asarray(image)
        if a.ndim != 2:
            return None
        _, w = a.shape
        width = max(1, int(width))
        half = width // 2
        c1 = max(0, int(col) - half)
        c2 = min(w, int(col) + half + (1 if width % 2 else 0))
        sl = a[:, c1:c2]
        return sl.sum(axis=1)

    def _update_wavelength_axis_cache(self, force: bool = False):
        try:
            if self.cam is None:
                wl = None
            else:
                wl = self.cam.get_wavelength_axis(force=bool(force))
        except Exception:
            wl = None
        if wl is None:
            self._wl_axis = None
            return None
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            self._wl_axis = None
            return None
        if arr.size == 0 or np.allclose(arr, 0.0):
            self._wl_axis = None
            return None
        self._wl_axis = arr
        try:
            if self._last_frame_raw is not None:
                self._update_image_wavelength_axis(self._last_frame_raw.shape[1])
                self.canvas.draw_idle()
        except Exception:
            pass
        return arr

    def _get_x_axis_data(self, width: int):
        wl = getattr(self, "_wl_axis", None)
        if wl is None:
            return np.arange(width)
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            return np.arange(width)
        if self._last_raw_width is not None and arr.size == self._last_raw_width:
            _, _, left, right = self._crop
            if right > 0:
                arr = arr[: arr.size - right]
            if left > 0:
                arr = arr[left:]
        if arr.size != width or np.allclose(arr, 0.0):
            return np.arange(width)
        return arr

    def _maybe_update_xaxis_label(self, width: int) -> None:
        wl = getattr(self, "_wl_axis", None)
        has_wl = False
        if wl is not None:
            try:
                arr = np.asarray(wl, dtype=float).ravel()
                has_wl = arr.size == (self._last_raw_width or width) and (not np.allclose(arr, 0.0))
            except Exception:
                has_wl = False
        mode = "wavelength" if has_wl else "pixel"
        if mode == self._xaxis_mode:
            return
        self._xaxis_mode = mode
        if mode == "wavelength":
            self.ax_h.set_xlabel("Wavelength (nm)")
        else:
            self.ax_h.set_xlabel("X (px)")

    def _update_image_wavelength_axis(self, width: int) -> None:
        if self._ax_img_top is None:
            return
        wl = getattr(self, "_wl_axis", None)
        if wl is None:
            self._ax_img_top.set_visible(False)
            return
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            self._ax_img_top.set_visible(False)
            return
        if self._last_raw_width is not None and arr.size == self._last_raw_width:
            _, _, left, right = self._crop
            if right > 0:
                arr = arr[: arr.size - right]
            if left > 0:
                arr = arr[left:]
        if arr.size != width or np.allclose(arr, 0.0):
            self._ax_img_top.set_visible(False)
            return

        n_ticks = 6
        pix = np.linspace(0, max(1, width - 1), n_ticks)
        pix_int = np.unique(np.clip(np.round(pix).astype(int), 0, width - 1))
        labels = [f"{arr[i]:.1f}" for i in pix_int]
        self._ax_img_top.set_xlim(self.ax_img.get_xlim())
        self._ax_img_top.set_xticks(pix_int)
        self._ax_img_top.set_xticklabels(labels)
        self._ax_img_top.set_visible(True)

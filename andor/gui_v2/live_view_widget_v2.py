from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .config import DEFAULT_LINECUT_WIDTH


class AndorLiveViewWidgetV2(QtWidgets.QGroupBox):
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
        self._roi_curve = None
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

        self.glw = pg.GraphicsLayoutWidget()

        # --- Image plot ---
        self.img_plot = self.glw.addPlot(row=0, col=0)
        self.img_plot.setTitle("Live")
        self.img_plot.setAspectLocked(False)

        self.img_item = pg.ImageItem(np.zeros((10, 10), dtype=np.uint8))
        lut = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
        self.img_item.setLookupTable(lut)
        self.img_plot.addItem(self.img_item)
        self.img_plot.invertY(True)  # row 0 at top, matching matplotlib origin='upper'

        # Top axis for wavelength labels (mirrors matplotlib twiny())
        self.img_plot.showAxis("top")
        self._top_axis = self.img_plot.getAxis("top")
        self._top_axis.setLabel("Wavelength (nm)")
        self._top_axis.hide()

        # Crosshairs
        self._crosshair_h = pg.InfiniteLine(angle=0, pen=pg.mkPen("w", width=0.8))
        self._crosshair_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=0.8))
        self._crosshair_h.setVisible(False)
        self._crosshair_v.setVisible(False)
        self.img_plot.addItem(self._crosshair_h)
        self.img_plot.addItem(self._crosshair_v)

        # --- Vertical linecut plot (right of image) ---
        self.vcut_plot = self.glw.addPlot(row=0, col=1)
        self.vcut_plot.setLabel("left", "Y (px)")
        self.vcut_plot.invertY(True)
        self.vcut_curve = self.vcut_plot.plot([], [], pen=pg.mkPen("b", width=1))

        # --- Horizontal linecut plot (below image) ---
        self.hcut_plot = self.glw.addPlot(row=1, col=0)
        self.hcut_plot.setLabel("bottom", "X (px)")
        self.hcut_curve = self.hcut_plot.plot([], [], pen=pg.mkPen("b", width=1))

        # Proportions matching original width_ratios=[4,1.2], height_ratios=[4,1.2]
        self.glw.ci.layout.setColumnStretchFactor(0, 4)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)
        self.glw.ci.layout.setRowStretchFactor(0, 4)
        self.glw.ci.layout.setRowStretchFactor(1, 1)

        layout.addWidget(self.glw, 0, 0, 2, 1)

        # Side panel
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

        self.img_plot.scene().sigMouseClicked.connect(self._on_mouse_click)

    # -----------------
    # Public API
    # -----------------
    def set_camera(self, cam) -> None:
        self.cam = cam

    def set_crop(self, top: int, bottom: int, left: int, right: int) -> None:
        self._crop = (int(top), int(bottom), int(left), int(right))
        if self._last_frame_full is not None:
            self.update_frame({"image": self._last_frame_full, "image8": self._last_frame8_full})

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
        self.linecut_changed.emit(row)

    def set_roi(self, x1: int, x2: int, y1: int, y2: int) -> None:
        try:
            self._roi = (int(round(x1)), int(round(x2)), int(round(y1)), int(round(y2)))
        except Exception:
            return
        self._update_roi_overlay()

    def clear_roi(self) -> None:
        self._roi = None
        if self._roi_curve is not None:
            self.img_plot.removeItem(self._roi_curve)
            self._roi_curve = None

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
            display_arr = disp8.astype(np.uint8)
            self.img_item.setImage(display_arr, autoLevels=False)
            self.img_item.setLevels([0, 255])
        else:
            display_arr = disp
            self.img_item.setImage(display_arr, autoLevels=False)
            try:
                vmin = float(np.nanmin(display_arr))
                vmax = float(np.nanmax(display_arr))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.img_item.setLevels([vmin, vmax])
            except Exception:
                pass

        self.img_plot.setXRange(0, w, padding=0)
        self.img_plot.setYRange(0, h, padding=0)
        self.hcut_plot.setXRange(
            float(self._get_x_axis_data(w)[0]),
            float(self._get_x_axis_data(w)[-1]),
            padding=0,
        )
        self.vcut_plot.setYRange(0, h, padding=0)
        self._update_cursor_overlays(h, w)
        self._update_roi_overlay()

    # -----------------
    # Events
    # -----------------
    def _on_mouse_click(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._last_frame is None or self._last_frame_raw is None:
            return
        pos = event.scenePos()
        vb = self.img_plot.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        h, w = self._last_frame.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        self._cursor_rc = (y, x)
        self._update_cursor_overlays(h, w)
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
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)
            return
        y, x = self._cursor_rc
        if not ((0 <= x < w) and (0 <= y < h)):
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)
            return

        y_raw = h - 1 - y
        if not (0 <= y_raw < h):
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)
            return

        x_raw = w - 1 - x  # display col x maps to physical col x_raw after fliplr
        try:
            val = float(self._last_frame_raw[y_raw, x_raw])
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}, I = {val:g}")
        except Exception:
            self.cursorLbl.setText(f"Cursor: (row, col) = {y}, {x}")

        self._crosshair_h.setPos(y)
        self._crosshair_v.setPos(x)
        self._crosshair_h.setVisible(True)
        self._crosshair_v.setVisible(True)

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
            self.hcut_curve.setData(x_axis, hcut[::-1])  # reverse to match fliplr display
        if vcut is not None:
            self.vcut_curve.setData(vcut, np.arange(vcut.size))

    def _update_roi_overlay(self) -> None:
        if self._last_frame is None:
            return
        h, w = self._last_frame.shape
        if self._roi is None:
            if self._roi_curve is not None:
                self.img_plot.removeItem(self._roi_curve)
                self._roi_curve = None
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
        xs = [x1, x2, x2, x1, x1]
        ys = [y1, y1, y2, y2, y1]
        if self._roi_curve is None:
            self._roi_curve = self.img_plot.plot(
                xs, ys, pen=pg.mkPen("#00e5ff", width=1.0)
            )
        else:
            self._roi_curve.setData(xs, ys)

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
        return a[r1:r2, :].sum(axis=0)

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
        return a[:, c1:c2].sum(axis=1)

    def _update_wavelength_axis_cache(self, force: bool = False):
        try:
            wl = None if self.cam is None else self.cam.get_wavelength_axis(force=bool(force))
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
            self.hcut_plot.setLabel("bottom", "Wavelength (nm)")
        else:
            self.hcut_plot.setLabel("bottom", "X (px)")

    def _update_image_wavelength_axis(self, width: int) -> None:
        wl = getattr(self, "_wl_axis", None)
        if wl is None:
            self._top_axis.hide()
            return
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            self._top_axis.hide()
            return
        if self._last_raw_width is not None and arr.size == self._last_raw_width:
            _, _, left, right = self._crop
            if right > 0:
                arr = arr[: arr.size - right]
            if left > 0:
                arr = arr[left:]
        if arr.size != width or np.allclose(arr, 0.0):
            self._top_axis.hide()
            return

        n_ticks = 6
        pix = np.linspace(0, max(1, width - 1), n_ticks)
        pix_int = np.unique(np.clip(np.round(pix).astype(int), 0, width - 1))
        ticks = [(float(i), f"{arr[i]:.1f}") for i in pix_int]
        self._top_axis.setTicks([ticks, []])
        self._top_axis.show()

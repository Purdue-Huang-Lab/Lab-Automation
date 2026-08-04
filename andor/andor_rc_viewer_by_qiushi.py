"""
Andor spectrometer data viewer.
Displays .asc files as a 2D image (wavelength vs. pixel row)
and integrated 1D spectra.  Supports loading two files
(sample + substrate) and showing the difference spectrum.
"""

import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ── Dark theme ────────────────────────────────────────────────────────
pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLabel { color: #cdd6f4; }
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover { background-color: #585b70; }
QPushButton:pressed { background-color: #6c7086; }
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 6px;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 6px;
}
QStatusBar { color: #a6adc8; }
QSlider::groove:horizontal {
    height: 6px;
    background: #313244;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QCheckBox { color: #cdd6f4; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #585b70;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
"""


def load_asc(filepath):
    """Return (wavelengths, intensity_2d, metadata) from an Andor .asc file."""
    metadata = {}
    data_lines = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.lstrip("# ").split(":", 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
            else:
                data_lines.append(line)

    rows = []
    for dl in data_lines:
        vals = dl.split()
        rows.append([float(v) for v in vals])
    data = np.array(rows)
    wavelengths = data[:, 0]
    intensity = data[:, 1:]
    return wavelengths, intensity, metadata


class AndorViewer(QtWidgets.QMainWindow):
    def __init__(self, filepath=None):
        super().__init__()
        self.setWindowTitle("Andor Reflectance Viewer")
        self.resize(1500, 950)

        # data storage
        self.sample_wl = None
        self.sample_int = None
        self.sample_meta = {}
        self.substrate_wl = None
        self.substrate_int = None
        self.substrate_meta = {}
        self.selected_row = None  # pixel row clicked by user

        self._build_ui()
        if filepath and os.path.isfile(filepath):
            self._load_sample(filepath)

    # ─── UI ───────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        # ── top toolbar ──
        toolbar = QtWidgets.QHBoxLayout()

        btn_sample = QtWidgets.QPushButton("📂  Load Sample")
        btn_sample.setStyleSheet("QPushButton { background-color: #45475a; } QPushButton:hover { background-color: #585b70; }")
        btn_sample.clicked.connect(self._on_open_sample)
        toolbar.addWidget(btn_sample)

        self.lbl_sample = QtWidgets.QLabel("No sample")
        self.lbl_sample.setStyleSheet("color:#89b4fa; font-style:italic; padding: 0 8px;")
        toolbar.addWidget(self.lbl_sample)

        btn_substrate = QtWidgets.QPushButton("📂  Load Substrate")
        btn_substrate.setStyleSheet("QPushButton { background-color: #45475a; } QPushButton:hover { background-color: #585b70; }")
        btn_substrate.clicked.connect(self._on_open_substrate)
        toolbar.addWidget(btn_substrate)

        self.lbl_substrate = QtWidgets.QLabel("No substrate")
        self.lbl_substrate.setStyleSheet("color:#a6e3a1; font-style:italic; padding: 0 8px;")
        toolbar.addWidget(self.lbl_substrate)

        toolbar.addStretch(1)

        self.cb_log = QtWidgets.QCheckBox("Log scale")
        self.cb_log.toggled.connect(self._refresh_all)
        toolbar.addWidget(self.cb_log)

        self.cb_bg = QtWidgets.QCheckBox("Subtract BG")
        self.cb_bg.toggled.connect(self._refresh_all)
        toolbar.addWidget(self.cb_bg)

        root.addLayout(toolbar)

        # ── metadata bar ──
        self.lbl_meta = QtWidgets.QLabel("")
        self.lbl_meta.setStyleSheet("color:#94e2d5; font-size:12px;")
        root.addWidget(self.lbl_meta)

        # ── main content: 2D image on top, spectra on bottom ──
        vsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(vsplitter, 1)

        # --- top row: 2D image ---
        self.img_widget = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_widget.addPlot(title="2‑D Spectral Image (Sample)")
        self.img_plot.setLabel("bottom", "Wavelength", units="nm")
        self.img_plot.setLabel("left", "Pixel Row")
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.hist.gradient.loadPreset("inferno")
        self.img_widget.addItem(self.hist)

        # ROI for pixel‑row integration
        self.roi = pg.LinearRegionItem(orientation='horizontal',
                                       brush=pg.mkBrush(137, 180, 250, 40))
        self.roi.sigRegionChanged.connect(self._update_spectra)
        self.img_plot.addItem(self.roi)

        # crosshair (follows mouse)
        self.vLine = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#f9e2af", width=1, style=QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen("#f9e2af", width=1, style=QtCore.Qt.DashLine))
        self.img_plot.addItem(self.vLine, ignoreBounds=True)
        self.img_plot.addItem(self.hLine, ignoreBounds=True)

        # selected‑row marker (stays on click)
        self.selected_hLine = pg.InfiniteLine(angle=0, movable=False,
                                              pen=pg.mkPen("#fab387", width=2))
        self.selected_hLine.setVisible(False)
        self.img_plot.addItem(self.selected_hLine, ignoreBounds=True)

        self.proxy = pg.SignalProxy(self.img_plot.scene().sigMouseMoved,
                                    rateLimit=60, slot=self._on_mouse_moved)
        # click handler
        self.img_plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        vsplitter.addWidget(self.img_widget)

        # --- bottom: 2×2 grid of spectrum plots ---
        bottom = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(bottom)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        # (0,0) Sample + Substrate overlay (ROI integrated)
        self.spec_widget = pg.PlotWidget(title="Integrated Spectra (ROI rows)")
        self.spec_widget.setLabel("bottom", "Wavelength", units="nm")
        self.spec_widget.setLabel("left", "Intensity", units="counts")
        self.spec_widget.showGrid(x=True, y=True, alpha=0.25)
        self.spec_widget.addLegend(offset=(60, 10))
        self.curve_sample = self.spec_widget.plot(pen=pg.mkPen("#89b4fa", width=2), name="Sample")
        self.curve_substrate = self.spec_widget.plot(pen=pg.mkPen("#a6e3a1", width=2), name="Substrate")
        grid.addWidget(self.spec_widget, 0, 0)

        # (0,1) Difference (Sample − Substrate)
        self.diff_widget = pg.PlotWidget(title="δR(λ) = (R_sample − R_substrate) / R_substrate")
        self.diff_widget.setLabel("bottom", "Wavelength", units="nm")
        self.diff_widget.setLabel("left", "δR(λ)")
        self.diff_widget.showGrid(x=True, y=True, alpha=0.25)
        self.curve_diff = self.diff_widget.plot(pen=pg.mkPen("#f38ba8", width=2))
        grid.addWidget(self.diff_widget, 0, 1)

        # (1,0) Single‑row spectrum (click to select)
        self.row_widget = pg.PlotWidget(title="Single Row Spectrum (click on image)")
        self.row_widget.setLabel("bottom", "Wavelength", units="nm")
        self.row_widget.setLabel("left", "Intensity", units="counts")
        self.row_widget.showGrid(x=True, y=True, alpha=0.25)
        self.row_widget.addLegend(offset=(60, 10))
        self.curve_row_sample = self.row_widget.plot(pen=pg.mkPen("#fab387", width=2), name="Sample")
        self.curve_row_substrate = self.row_widget.plot(pen=pg.mkPen("#a6e3a1", width=2, style=QtCore.Qt.DashLine), name="Substrate")
        grid.addWidget(self.row_widget, 1, 0)

        # (1,1) Spatial cross‑section
        self.cross_widget = pg.PlotWidget(title="Spatial Cross‑section")
        self.cross_widget.setLabel("bottom", "Pixel Row")
        self.cross_widget.setLabel("left", "Intensity", units="counts")
        self.cross_widget.showGrid(x=True, y=True, alpha=0.25)
        self.cross_curve = self.cross_widget.plot(pen=pg.mkPen("#f9e2af", width=2))
        grid.addWidget(self.cross_widget, 1, 1)

        vsplitter.addWidget(bottom)
        vsplitter.setSizes([400, 550])

        self.statusBar().showMessage("Ready – load sample & substrate .asc files  |  Click image to select a row")

    # ─── file loading ─────────────────────────────────────────────────
    def _open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Andor .asc", "", "Andor ASCII (*.asc);;All files (*)"
        )
        return path

    def _on_open_sample(self):
        path = self._open_file_dialog()
        if path:
            self._load_sample(path)

    def _on_open_substrate(self):
        path = self._open_file_dialog()
        if path:
            self._load_substrate(path)

    def _load_sample(self, filepath):
        try:
            self.sample_wl, self.sample_int, self.sample_meta = load_asc(filepath)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        self.lbl_sample.setText(f"Sample: {os.path.basename(filepath)}")
        n_pix = self.sample_int.shape[1]
        self.roi.setRegion([0, n_pix])
        self._update_meta()
        self._refresh_all()
        self.statusBar().showMessage(
            f"Sample loaded: {len(self.sample_wl)} wl × {n_pix} px  "
            f"({self.sample_wl[0]:.1f}–{self.sample_wl[-1]:.1f} nm)"
        )

    def _load_substrate(self, filepath):
        try:
            self.substrate_wl, self.substrate_int, self.substrate_meta = load_asc(filepath)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        self.lbl_substrate.setText(f"Substrate: {os.path.basename(filepath)}")
        self._update_meta()
        self._refresh_all()
        self.statusBar().showMessage(
            f"Substrate loaded: {len(self.substrate_wl)} wl × {self.substrate_int.shape[1]} px"
        )

    def _update_meta(self):
        meta = self.sample_meta or self.substrate_meta
        parts = []
        for k in ("Exposure_ms", "Accumulations", "Slit_um", "Grating", "Center_nm"):
            if k in meta:
                parts.append(f"{k.replace('_', ' ')}: {meta[k]}")
        self.lbl_meta.setText("   |   ".join(parts))

    # ─── data helpers ─────────────────────────────────────────────────
    def _process(self, intensity):
        """Apply optional background subtraction."""
        data = intensity.copy()
        if self.cb_bg.isChecked():
            bg = np.median(
                np.concatenate([data[:, :20], data[:, -20:]], axis=1),
                axis=1, keepdims=True
            )
            data = data - bg
        return data

    def _integrate(self, data):
        """Integrate over ROI pixel rows."""
        lo, hi = self.roi.getRegion()
        lo = max(0, int(lo))
        hi = min(data.shape[1], int(hi))
        if hi <= lo:
            hi = lo + 1
        return data[:, lo:hi].sum(axis=1)

    def _apply_log(self, arr):
        if self.cb_log.isChecked():
            return np.log10(np.clip(arr, 1, None))
        return arr

    # ─── plotting ─────────────────────────────────────────────────────
    def _refresh_all(self):
        self._refresh_image()
        self._update_spectra()
        self._update_single_row()

    def _refresh_image(self):
        """Update the 2D image from sample data."""
        if self.sample_wl is None:
            return
        data = self._process(self.sample_int)
        disp = self._apply_log(data)

        wl = self.sample_wl
        n_pix = data.shape[1]

        self.img_item.setImage(disp, autoLevels=True)
        dx = (wl[-1] - wl[0]) / len(wl)
        tr = QtGui.QTransform()
        tr.translate(wl[0], 0)
        tr.scale(dx, 1)
        self.img_item.setTransform(tr)
        self.img_plot.setXRange(wl[0], wl[-1])
        self.img_plot.setYRange(0, n_pix)

    def _update_spectra(self):
        """Update 1D sample, substrate, and difference curves."""
        # Sample
        if self.sample_wl is not None:
            s_data = self._process(self.sample_int)
            s_spec = self._integrate(s_data)
            self.curve_sample.setData(self.sample_wl, self._apply_log(s_spec))
        else:
            self.curve_sample.clear()
            s_spec = None

        # Substrate
        if self.substrate_wl is not None:
            sub_data = self._process(self.substrate_int)
            sub_spec = self._integrate(sub_data)
            self.curve_substrate.setData(self.substrate_wl, self._apply_log(sub_spec))
        else:
            self.curve_substrate.clear()
            sub_spec = None

        # δR(λ) = (R_sample - R_substrate) / R_substrate
        if s_spec is not None and sub_spec is not None:
            if np.array_equal(self.sample_wl, self.substrate_wl):
                sub_ref = sub_spec
            else:
                sub_ref = np.interp(self.sample_wl, self.substrate_wl, sub_spec)
            # avoid division by zero
            sub_safe = np.where(np.abs(sub_ref) > 1e-6, sub_ref, 1e-6)
            delta_r = (s_spec - sub_ref) / sub_safe
            self.curve_diff.setData(self.sample_wl, delta_r)
        else:
            self.curve_diff.clear()

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self.sample_wl is None:
            return
        mouse_point = self.img_plot.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        self.vLine.setPos(x)
        self.hLine.setPos(y)

        data = self._process(self.sample_int)
        wl = self.sample_wl
        idx = np.clip(np.searchsorted(wl, x), 0, len(wl) - 1)

        cross = data[idx, :]
        self.cross_curve.setData(np.arange(len(cross)), self._apply_log(cross))
        self.cross_widget.setTitle(f"Spatial Cross‑section at {wl[idx]:.1f} nm")

    def _on_mouse_clicked(self, evt):
        """Click on the 2D image to pick a single pixel row."""
        if self.sample_wl is None:
            return
        pos = evt.scenePos()
        if not self.img_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.img_plot.vb.mapSceneToView(pos)
        row = int(round(mouse_point.y()))
        n_pix = self.sample_int.shape[1]
        row = max(0, min(row, n_pix - 1))
        self.selected_row = row

        # update marker
        self.selected_hLine.setPos(row)
        self.selected_hLine.setVisible(True)

        self._update_single_row()

    def _update_single_row(self):
        """Plot the spectrum at the selected pixel row."""
        if self.selected_row is None or self.sample_wl is None:
            return
        row = self.selected_row

        # sample
        s_data = self._process(self.sample_int)
        row_spec = s_data[:, row]
        self.curve_row_sample.setData(self.sample_wl, self._apply_log(row_spec))

        # substrate
        if self.substrate_wl is not None:
            sub_data = self._process(self.substrate_int)
            sub_row = min(row, sub_data.shape[1] - 1)
            sub_spec = sub_data[:, sub_row]
            self.curve_row_substrate.setData(self.substrate_wl, self._apply_log(sub_spec))
        else:
            self.curve_row_substrate.clear()

        self.row_widget.setTitle(f"Single Row Spectrum — Row {row}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)

    filepath = None
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "andor_exp5000ms_slit200um_gr1.asc")
        if os.path.isfile(candidate):
            filepath = candidate

    win = AndorViewer(filepath)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

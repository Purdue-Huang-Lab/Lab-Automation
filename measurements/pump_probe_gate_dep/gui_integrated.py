"""
gui_integrated.py
Integrated GUI for pump-probe measurements with delay stage control
PyQt5 + PyQtGraph version for fast real-time display
"""

import sys
import os
import numpy as np
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QLabel, QPushButton, QLineEdit,
    QCheckBox, QRadioButton, QButtonGroup, QTextEdit, QScrollArea,
    QSlider, QSplitter, QGroupBox, QFrame, QMessageBox, QFileDialog,
    QInputDialog, QSizePolicy, QFormLayout, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

import pyqtgraph as pg

from camera_simple import SimpleCamera, analyze_frames
from gaussian_fitting import GaussianFitter

try:
    from stage_control import DelayStage
    STAGE_AVAILABLE = True
except Exception as e:
    STAGE_AVAILABLE = False
    print(f"Stage control not available: {e}")
    print("GUI will run in camera-only mode.")

try:
    from rotation_stage_control import RotationStage
    ROTATION_AVAILABLE = True
except Exception as e:
    ROTATION_AVAILABLE = False
    print(f"Rotation stage control not available: {e}")

import pathlib as _pathlib
import sys as _sys
_REPO_ROOT = str(_pathlib.Path(__file__).parent.parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

try:
    from keithley.keithley_wrapper import make_resource_manager, KeithleySMU, SweepController
    from keithley.gui_v2.workers import RampThread as _RampThread
    from keithley.gui_v2.plots import PlotWidget as GatePlotWidget
    from keithley.gui_v2.config import POLL_TIMEOUT_MS as _GATE_POLL_TIMEOUT_MS
    KEITHLEY_AVAILABLE = True
except Exception as _ke:
    KEITHLEY_AVAILABLE = False
    GatePlotWidget = None
    print(f"Keithley modules not available: {_ke}")

try:
    from measurements.config import DATA_DIR
except Exception:
    DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

# ---- Gate control defaults ----
DEFAULT_FRONT_RESOURCE = "GPIB0::24::INSTR"
DEFAULT_BACK_RESOURCE  = "GPIB0::23::INSTR"
GATE_POLL_MS    = 500
GATE_OC_LIMIT_A = 0.01       # 10 mA over-current limit
GATE_OC_TRIP_N  = 2
GATE_RAMP_STEP_V  = 0.01
GATE_RAMP_DWELL_S = 0.10


def _make_rdbu_r_lut():
    pos = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    colors = np.array([
        [5, 48, 97, 255],
        [33, 102, 172, 255],
        [67, 147, 195, 255],
        [146, 197, 222, 255],
        [209, 229, 240, 255],
        [247, 247, 247, 255],
        [253, 219, 199, 255],
        [244, 165, 130, 255],
        [214, 96, 77, 255],
        [178, 24, 43, 255],
        [103, 0, 31, 255],
    ], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, colors)
    return cmap.getLookupTable(nPts=256)


RDBU_R_LUT = _make_rdbu_r_lut()


def _calculate_log_ratio(avg_on, avg_off):
    epsilon = 1e-6
    ratio = (avg_on + epsilon) / (avg_off + epsilon)
    return -1000 * np.log10(ratio)


# =====================================================================
#  Background worker threads (LabVIEW-style: hardware + math off the UI)
# =====================================================================

class _CaptureThread(QThread):
    """Continuous capture + analysis in background."""
    result_ready = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera, n_frames, roi):
        super().__init__()
        self.camera = camera
        self.n_frames = n_frames
        self.roi = roi
        self._running = True

    def run(self):
        while self._running:
            try:
                frames = self.camera.capture_frames(self.n_frames)
                results = analyze_frames(frames, *self.roi)
                if self._running:
                    self.result_ready.emit(results, frames)
            except Exception as e:
                if self._running:
                    self.error_occurred.emit(str(e))
                break

    def stop(self):
        self._running = False


class _DelayScanThread(QThread):
    """Delay scan in background."""
    progress = pyqtSignal(str)
    step_result = pyqtSignal(object, tuple, dict)
    finished = pyqtSignal(list, bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera, stage, time_delays_ps, positions_mm,
                 n_frames, roi, avg_map, default_avg):
        super().__init__()
        self.camera = camera
        self.stage = stage
        self.time_delays_ps = time_delays_ps
        self.positions_mm = positions_mm
        self.n_frames = n_frames
        self.roi = roi
        self.avg_map = avg_map
        self.default_avg = default_avg
        self._running = True
        self.scan_results = []

    def run(self):
        try:
            for i, (delay_ps, position_mm) in enumerate(
                    zip(self.time_delays_ps, self.positions_mm)):
                if not self._running:
                    self.progress.emit(f"Scan stopped at {i}/{len(self.positions_mm)}")
                    break
                delay_key = round(delay_ps, 2)
                avg_times = self.avg_map.get(delay_key, self.default_avg)
                self.stage.move_to(position_mm, wait=True)
                time.sleep(0.5)
                sum_on = None; sum_off = None
                total_on = 0; total_off = 0
                for avg_idx in range(avg_times):
                    if not self._running:
                        break
                    self.progress.emit(
                        f"Scanning {i+1}/{len(self.time_delays_ps)} | "
                        f"Avg {avg_idx+1}/{avg_times}")
                    frames = self.camera.capture_frames(self.n_frames)
                    results = analyze_frames(frames, *self.roi)
                    n_on = results['n_pump_on']
                    n_off = results['n_pump_off']
                    if results['avg_pump_on'] is not None and n_on > 0:
                        batch_sum = results['avg_pump_on'].astype(np.float64) * n_on
                        if sum_on is None:
                            sum_on = batch_sum
                        else:
                            sum_on += batch_sum
                        total_on += n_on
                    if results['avg_pump_off'] is not None and n_off > 0:
                        batch_sum = results['avg_pump_off'].astype(np.float64) * n_off
                        if sum_off is None:
                            sum_off = batch_sum
                        else:
                            sum_off += batch_sum
                        total_off += n_off
                final_on = sum_on / total_on if sum_on is not None else None
                final_off = sum_off / total_off if sum_off is not None else None
                final_results = {
                    'avg_pump_on': final_on, 'avg_pump_off': final_off,
                    'n_pump_on': total_on, 'n_pump_off': total_off,
                    'threshold': results['threshold'],
                    'ref_intensities': results['ref_intensities']
                }
                step_info = {
                    'time_delay_ps': delay_ps, 'position_mm': position_mm,
                    'results': final_results, 'avg_times_used': avg_times
                }
                self.scan_results.append(step_info)
                self.step_result.emit(final_results, self.roi, step_info)
            self.finished.emit(self.scan_results, self._running)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error_occurred.emit(str(e))

    def stop(self):
        self._running = False


class _PowerScanThread(QThread):
    """Power-dependent delay scan in background."""
    progress = pyqtSignal(str)
    step_result = pyqtSignal(object, tuple)
    angle_complete = pyqtSignal(float, list, int)
    finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera, stage, rotation_stage,
                 angles, time_delays_ps, positions_mm,
                 n_frames, roi, avg_map, default_avg, settle_time,
                 save_dir, save_base, timestamp, crop=None,
                 t0_ps=0.0, pixel_size=0.07, crop_h=128,
                 pol_enabled=False, pol_stage=None, lh_angle=0.0, pol_settle_s=0.5):
        super().__init__()
        self.camera = camera
        self.stage = stage
        self.rotation_stage = rotation_stage
        self.angles = angles
        self.time_delays_ps = time_delays_ps
        self.positions_mm = positions_mm
        self.n_frames = n_frames
        self.roi = roi
        self.avg_map = avg_map
        self.default_avg = default_avg
        self.settle_time = settle_time
        self.save_dir = save_dir
        self.save_base = save_base
        self.timestamp = timestamp
        self.crop = crop
        self.t0_ps = t0_ps
        self.pixel_size = pixel_size
        self.crop_h = crop_h
        self.pol_enabled = pol_enabled
        self.pol_stage = pol_stage
        self.lh_angle = lh_angle
        self.pol_settle_s = pol_settle_s
        self._running = True
        self.angles_completed = 0

    def _crop(self, img):
        if img is None or self.crop is None:
            return img
        x0, w = self.crop
        h, full_w = img.shape
        x0 = max(0, min(x0, full_w - 1))
        x1 = min(x0 + w, full_w)
        return img[:, x0:x1] if x1 > x0 else img

    def _pol_pairs(self):
        if self.pol_enabled and self.pol_stage is not None:
            return [('LH', self.lh_angle), ('RH', self.lh_angle + 45.0)]
        return [(None, None)]

    def _move_pol(self, pol_angle):
        if self.pol_enabled and self.pol_stage is not None and pol_angle is not None:
            self.pol_stage.move_to(pol_angle, wait=True)
            time.sleep(self.pol_settle_s)

    def _run_delay_scan_for_angle(self, ai, angle, pol_tag):
        """Run one delay scan at the given power angle (and current polarization)."""
        pol_suffix = f" | {pol_tag}" if pol_tag else ""
        delay_results = []
        for di, (delay_ps, pos_mm) in enumerate(
                zip(self.time_delays_ps, self.positions_mm)):
            if not self._running:
                break
            delay_key = round(delay_ps, 2)
            avg_times = self.avg_map.get(delay_key, self.default_avg)
            self.stage.move_to(pos_mm, wait=True)
            time.sleep(0.5)
            sum_on = None; sum_off = None
            total_on = 0; total_off = 0
            for avg_idx in range(avg_times):
                if not self._running:
                    break
                self.progress.emit(
                    f"Angle {ai+1}/{len(self.angles)} ({angle:.1f}){pol_suffix} | "
                    f"Delay {di+1}/{len(self.time_delays_ps)} "
                    f"({delay_ps:.2f} ps) | Avg {avg_idx+1}/{avg_times}")
                frames = self.camera.capture_frames(self.n_frames)
                results = analyze_frames(frames, *self.roi)
                n_on = results['n_pump_on']
                n_off = results['n_pump_off']
                if results['avg_pump_on'] is not None and n_on > 0:
                    batch_sum = results['avg_pump_on'].astype(np.float64) * n_on
                    sum_on = batch_sum if sum_on is None else sum_on + batch_sum
                    total_on += n_on
                if results['avg_pump_off'] is not None and n_off > 0:
                    batch_sum = results['avg_pump_off'].astype(np.float64) * n_off
                    sum_off = batch_sum if sum_off is None else sum_off + batch_sum
                    total_off += n_off
            final_on = sum_on / total_on if sum_on is not None else None
            final_off = sum_off / total_off if sum_off is not None else None
            final_results = {
                'avg_pump_on': final_on, 'avg_pump_off': final_off,
                'n_pump_on': total_on, 'n_pump_off': total_off,
                'threshold': results['threshold'],
                'ref_intensities': results['ref_intensities']
            }
            delay_results.append({
                'time_delay_ps': delay_ps, 'position_mm': pos_mm,
                'results': final_results, 'avg_times_used': avg_times
            })
            self.step_result.emit(final_results, self.roi)
        return delay_results

    def run(self):
        try:
            for ai, angle in enumerate(self.angles):
                if not self._running:
                    self.progress.emit(
                        f"Power scan stopped at angle {ai}/{len(self.angles)}")
                    break
                self.progress.emit(
                    f"Angle {ai+1}/{len(self.angles)}: rotating to {angle:.2f}")
                self.rotation_stage.move_to(angle, wait=True)
                time.sleep(self.settle_time)
                for pol_tag, pol_angle in self._pol_pairs():
                    if not self._running:
                        break
                    self._move_pol(pol_angle)
                    angle_delay_results = self._run_delay_scan_for_angle(ai, angle, pol_tag)
                    self._save_angle(angle, angle_delay_results, pol_tag=pol_tag)
                    self.angles_completed += 1
                    self.angle_complete.emit(angle, angle_delay_results,
                                             self.angles_completed)
                    del angle_delay_results
            if self._running:
                self._save_summary()
            self.finished.emit(self._running)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _save_angle(self, angle_deg, delay_results, pol_tag=None):
        """Save in TAM-code-new / LabVIEW compatible format (*sum.txt + *para.txt)."""
        try:
            astr = f"{angle_deg:.1f}".replace('.', 'p').replace('-', 'n')
            pol_tag_str = f"_{pol_tag}" if pol_tag else ""
            prefix = f"{self.save_base}_ND{astr}deg{pol_tag_str}_{self.timestamp}"
            delays = [r['time_delay_ps'] for r in delay_results]
            avg_used = [r.get('avg_times_used', 1) for r in delay_results]
            crop_w = self.crop[1] if self.crop else (self.camera.width if self.camera else 480)

            para_path = os.path.join(self.save_dir, f"{prefix}para.txt")
            with open(para_path, 'w') as f:
                f.write('\t'.join([f'{d:.6f}' for d in delays]) + '\n')
                f.write('\t'.join([f'{a:.6f}' for a in avg_used]) + '\n')
                f.write('0\n')
                f.write(f' {1.0:.6f} \n')
                f.write(f' {self.pixel_size:.6f} \n')
                f.write(f' {crop_w} \n')
                f.write(f' {self.pixel_size:.6f} \n')
                f.write(f' {self.crop_h} \n')
                f.write(f' {0.0:.6f} \n')
                f.write(f' {self.t0_ps:.6f}\n')
                f.write('0\n0\n0\n0\n0 \n')
                f.write(f' {0.0:.6f}\n')
                f.write(f' {angle_deg:.6f}\n')
                f.write(f'{1.000000}\n')
                f.write(f'{0}\n')
                f.write(f' {self.n_frames} \n')
                f.write(f' 1\n')

            sum_path = os.path.join(self.save_dir, f"{prefix}sum.txt")
            with open(sum_path, 'w') as f:
                for r in delay_results:
                    res = r['results']
                    on = self._crop(res['avg_pump_on'])
                    off = self._crop(res['avg_pump_off'])
                    if on is not None and off is not None:
                        lr = _calculate_log_ratio(on, off).T
                        for row in lr:
                            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
        except Exception as e:
            print(f"Warning: save angle {angle_deg} failed: {e}")

    def _save_summary(self):
        try:
            sf = os.path.join(self.save_dir,
                              f"{self.save_base}_power_summary_{self.timestamp}.txt")
            with open(sf, 'w') as f:
                f.write(f"Power Scan Summary\nTimestamp: {self.timestamp}\n"
                        f"Angles completed: {self.angles_completed}\n"
                        f"Delays per angle: {len(self.time_delays_ps)}\n\n")
                for a in self.angles[:self.angles_completed]:
                    f.write(f"{a:.4f}\n")
        except Exception as e:
            print(f"Warning: save summary failed: {e}")

    def stop(self):
        self._running = False


class _GateDevWidget(QGroupBox):
    """Lightweight gate SMU control: connect, set V, read V/I. No sweep controls."""
    def __init__(self, title, default_resource, parent=None):
        super().__init__(title, parent)
        self.dev = None
        self._rm = None
        self._ramp_ctrl = None
        self._ramp_thread = None
        self._ramping = False
        self._build_ui(default_resource)

    def _build_ui(self, default_resource):
        lay = QGridLayout(self)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(5)

        self.resource_edit = QLineEdit(default_resource)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedWidth(75)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setFixedWidth(75)
        self.disconnect_btn.setEnabled(False)

        lay.addWidget(QLabel("VISA:"), 0, 0)
        lay.addWidget(self.resource_edit, 0, 1, 1, 3)
        lay.addWidget(self.connect_btn, 0, 4)
        lay.addWidget(self.disconnect_btn, 0, 5)

        self.vset_spin = QDoubleSpinBox()
        self.vset_spin.setRange(-210, 210); self.vset_spin.setDecimals(3)
        self.vset_spin.setValue(0.0); self.vset_spin.setFixedWidth(75)

        self.icomp_spin = QDoubleSpinBox()
        self.icomp_spin.setRange(0.001, 1e6); self.icomp_spin.setDecimals(1)
        self.icomp_spin.setValue(10.0); self.icomp_spin.setFixedWidth(65)

        self.output_chk = QCheckBox("Out")
        self.output_chk.setEnabled(False)

        self.apply_btn = QPushButton("Apply V")
        self.apply_btn.setFixedWidth(65)
        self.apply_btn.setEnabled(False)

        lay.addWidget(QLabel("V (V):"), 1, 0)
        lay.addWidget(self.vset_spin, 1, 1)
        lay.addWidget(QLabel("Icomp (nA):"), 1, 2)
        lay.addWidget(self.icomp_spin, 1, 3)
        lay.addWidget(self.output_chk, 1, 4)
        lay.addWidget(self.apply_btn, 1, 5)

        mono = QFont("Courier New", 9)
        self.vmeas_lbl = QLabel("--"); self.vmeas_lbl.setFont(mono)
        self.imeas_lbl = QLabel("--"); self.imeas_lbl.setFont(mono)
        self.status_lbl = QLabel("Disconnected.")
        self.status_lbl.setStyleSheet("color: gray;")

        lay.addWidget(QLabel("V-meas:"), 2, 0)
        lay.addWidget(self.vmeas_lbl, 2, 1)
        lay.addWidget(QLabel("I (nA):"), 2, 2)
        lay.addWidget(self.imeas_lbl, 2, 3)
        lay.addWidget(self.status_lbl, 2, 4, 1, 2)
        lay.setColumnStretch(1, 1); lay.setColumnStretch(3, 1)

        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        self.apply_btn.clicked.connect(self._on_apply)
        self.output_chk.toggled.connect(self._on_output_toggled)

    def _on_connect(self):
        res = self.resource_edit.text().strip()
        if not res:
            QMessageBox.critical(self, "Error", "Enter a VISA resource string.")
            return
        try:
            if self._rm is None:
                self.status_lbl.setText("Opening VISA RM...")
                self.status_lbl.setStyleSheet("color: gray;")
                self._rm = make_resource_manager()
            self.dev = KeithleySMU(self._rm, res, timeout_ms=20000, verbose=False)
            idn = self.dev.open()
            self.dev.set_compliance(float(self.icomp_spin.value()) * 1e-9)
            self.dev.set_output(True)
            self.output_chk.blockSignals(True)
            self.output_chk.setChecked(True)
            self.output_chk.blockSignals(False)
            self.status_lbl.setText(("Connected: " + (idn or ""))[:50])
            self.status_lbl.setStyleSheet("color: green;")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
            self.output_chk.setEnabled(True)
        except Exception as e:
            self.dev = None
            self.status_lbl.setText(f"Failed: {e}"[:60])
            self.status_lbl.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Connection failed", str(e))

    def _on_disconnect(self):
        if self._ramp_ctrl:
            self._ramp_ctrl.abort()
        if self.dev:
            try:
                self.dev.set_output(False)
                self.dev.close()
            except Exception:
                pass
        self.dev = None
        self._ramping = False
        self.vmeas_lbl.setText("--"); self.imeas_lbl.setText("--")
        self.status_lbl.setText("Disconnected.")
        self.status_lbl.setStyleSheet("color: gray;")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.output_chk.setEnabled(False)
        self.output_chk.blockSignals(True)
        self.output_chk.setChecked(False)
        self.output_chk.blockSignals(False)

    def _on_apply(self):
        if not self.dev or self._ramping:
            return
        vset = float(self.vset_spin.value())
        try:
            self.dev.set_compliance(float(self.icomp_spin.value()) * 1e-9)
        except Exception:
            pass
        self._ramping = True
        self.apply_btn.setEnabled(False)
        self.status_lbl.setText(f"Ramping to {vset:.3f} V...")
        self.status_lbl.setStyleSheet("color: orange;")
        self._ramp_ctrl = SweepController()
        self._ramp_thread = _RampThread(self.dev, vset, self._ramp_ctrl, parent=self)
        self._ramp_thread.progress.connect(self._on_ramp_progress)
        self._ramp_thread.done.connect(self._on_ramp_done)
        self._ramp_thread.start()

    def _on_ramp_progress(self, v, iA):
        self.vmeas_lbl.setText(f"{v:.4g}")
        self.imeas_lbl.setText(f"{iA*1e9:.4g}")

    def _on_ramp_done(self, status):
        self._ramping = False
        self.apply_btn.setEnabled(bool(self.dev))
        if status == "ok":
            self.status_lbl.setText(f"Set to {self.vset_spin.value():.3f} V")
            self.status_lbl.setStyleSheet("color: green;")
        elif status == "aborted":
            self.status_lbl.setText("Ramp aborted.")
            self.status_lbl.setStyleSheet("color: orange;")
        else:
            self.status_lbl.setText(f"Ramp error: {status}"[:60])
            self.status_lbl.setStyleSheet("color: red;")

    def _on_output_toggled(self, checked):
        if not self.dev:
            return
        try:
            self.dev.set_output(checked)
            self.status_lbl.setText("Output " + ("ON" if checked else "OFF"))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Output toggle failed:\n{e}")

    def poll_once(self, allow=True):
        """Read V and I; updates labels. Returns (v_V, i_A) or (None, None)."""
        if not self.dev or not allow or self._ramping:
            return None, None
        try:
            vi = self.dev.read_vi_with_timeout(_GATE_POLL_TIMEOUT_MS)
            v, iA = float(vi.v), float(vi.i)
            self.vmeas_lbl.setText(f"{v:.4g}")
            self.imeas_lbl.setText(f"{iA*1e9:.4g}")
            return v, iA
        except Exception:
            return None, None


class _GateSweepThread(QThread):
    """Gate-voltage-dependent (+ optional power-dependent) delay scan."""
    progress        = pyqtSignal(str)
    step_result     = pyqtSignal(object, tuple, dict)
    gate_step_done  = pyqtSignal(float, float, int)   # vf, vb, gate_idx
    power_step_done = pyqtSignal(float, int)           # angle, angle_idx
    gate_vi_update  = pyqtSignal(str, float, float)    # "front"/"back", V, I_A
    finished        = pyqtSignal(bool)
    error_occurred  = pyqtSignal(str)

    def __init__(self, camera, stage, front_dev, back_dev,
                 gate_pairs, dual_mode, gate_settle_s,
                 time_delays_ps, positions_mm, n_frames, roi,
                 avg_map, default_avg,
                 power_enabled=False, rotation_stage=None, angles=None,
                 power_settle_s=0.5, gate_outer=True,
                 save_dir=".", save_base="gate_scan", timestamp="",
                 crop=None, t0_ps=0.0, pixel_size=0.07, crop_h=128,
                 pol_enabled=False, pol_stage=None, lh_angle=0.0, pol_settle_s=0.5,
                 parent=None):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self.front_dev = front_dev
        self.back_dev = back_dev
        self.gate_pairs = gate_pairs
        self.dual_mode = dual_mode
        self.gate_settle_s = gate_settle_s
        self.time_delays_ps = time_delays_ps
        self.positions_mm = positions_mm
        self.n_frames = n_frames
        self.roi = roi
        self.avg_map = avg_map
        self.default_avg = default_avg
        self.power_enabled = power_enabled
        self.rotation_stage = rotation_stage
        self.angles = angles or []
        self.power_settle_s = power_settle_s
        self.gate_outer = gate_outer
        self.save_dir = save_dir
        self.save_base = save_base
        self.timestamp = timestamp
        self.crop = crop
        self.t0_ps = t0_ps
        self.pixel_size = pixel_size
        self.crop_h = crop_h
        self.pol_enabled = pol_enabled
        self.pol_stage = pol_stage
        self.lh_angle = lh_angle
        self.pol_settle_s = pol_settle_s
        self._running = True
        self._gate_ctrl = SweepController() if KEITHLEY_AVAILABLE else None

    def _crop(self, img):
        if img is None or self.crop is None:
            return img
        x0, w = self.crop
        h, full_w = img.shape
        x0 = max(0, min(x0, full_w - 1))
        x1 = min(x0 + w, full_w)
        return img[:, x0:x1] if x1 > x0 else img

    def run(self):
        try:
            if not self.power_enabled:
                self._run_gate_only()
            elif self.gate_outer:
                self._run_gate_outer_power_inner()
            else:
                self._run_power_outer_gate_inner()
            self.finished.emit(self._running)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _pol_pairs(self):
        """Return [(pol_tag, pol_angle), ...] — two entries when pol enabled, else one."""
        if self.pol_enabled and self.pol_stage is not None:
            return [('LH', self.lh_angle), ('RH', self.lh_angle + 45.0)]
        return [(None, None)]

    def _move_pol(self, pol_angle):
        if self.pol_enabled and self.pol_stage is not None and pol_angle is not None:
            self.pol_stage.move_to(pol_angle, wait=True)
            time.sleep(self.pol_settle_s)

    def _run_gate_only(self):
        for i, (vf, vb) in enumerate(self.gate_pairs):
            if not self._running:
                break
            self._ramp_gates(vf, vb)
            for pol_tag, pol_angle in self._pol_pairs():
                if not self._running:
                    break
                self._move_pol(pol_angle)
                delay_results = self._run_delay_scan(vf, vb, angle=None, pol_tag=pol_tag)
                self._save_gate_step(vf, vb, angle=None, delay_results=delay_results,
                                     pol_tag=pol_tag)
            self.gate_step_done.emit(vf, vb, i)

    def _run_gate_outer_power_inner(self):
        for i, (vf, vb) in enumerate(self.gate_pairs):
            if not self._running:
                break
            self._ramp_gates(vf, vb)
            for j, angle in enumerate(self.angles):
                if not self._running:
                    break
                self.rotation_stage.move_to(angle, wait=True)
                time.sleep(self.power_settle_s)
                for pol_tag, pol_angle in self._pol_pairs():
                    if not self._running:
                        break
                    self._move_pol(pol_angle)
                    delay_results = self._run_delay_scan(vf, vb, angle=angle, pol_tag=pol_tag)
                    self._save_gate_step(vf, vb, angle=angle, delay_results=delay_results,
                                         pol_tag=pol_tag)
                self.power_step_done.emit(angle, j)
            self.gate_step_done.emit(vf, vb, i)

    def _run_power_outer_gate_inner(self):
        for j, angle in enumerate(self.angles):
            if not self._running:
                break
            self.rotation_stage.move_to(angle, wait=True)
            time.sleep(self.power_settle_s)
            for i, (vf, vb) in enumerate(self.gate_pairs):
                if not self._running:
                    break
                self._ramp_gates(vf, vb)
                for pol_tag, pol_angle in self._pol_pairs():
                    if not self._running:
                        break
                    self._move_pol(pol_angle)
                    delay_results = self._run_delay_scan(vf, vb, angle=angle, pol_tag=pol_tag)
                    self._save_gate_step(vf, vb, angle=angle, delay_results=delay_results,
                                         pol_tag=pol_tag)
                self.gate_step_done.emit(vf, vb, i)
            self.power_step_done.emit(angle, j)

    def _run_delay_scan(self, vf, vb, angle, pol_tag=None):
        """Verbatim copy of _DelayScanThread inner loop logic."""
        scan_results = []
        pol_suffix = f" | {pol_tag}" if pol_tag else ""
        for i, (delay_ps, position_mm) in enumerate(
                zip(self.time_delays_ps, self.positions_mm)):
            if not self._running:
                self.progress.emit(f"Gate scan stopped at delay {i}/{len(self.positions_mm)}")
                break
            delay_key = round(delay_ps, 2)
            avg_times = self.avg_map.get(delay_key, self.default_avg)
            self.stage.move_to(position_mm, wait=True)
            time.sleep(0.5)
            sum_on = None; sum_off = None
            total_on = 0; total_off = 0
            for avg_idx in range(avg_times):
                if not self._running:
                    break
                self.progress.emit(
                    f"Vf={vf:+.3f}V{pol_suffix} | "
                    f"Delay {i+1}/{len(self.time_delays_ps)} "
                    f"({delay_ps:.2f} ps) | Avg {avg_idx+1}/{avg_times}")
                frames = self.camera.capture_frames(self.n_frames)
                results = analyze_frames(frames, *self.roi)
                n_on = results['n_pump_on']
                n_off = results['n_pump_off']
                if results['avg_pump_on'] is not None and n_on > 0:
                    batch_sum = results['avg_pump_on'].astype(np.float64) * n_on
                    if sum_on is None:
                        sum_on = batch_sum
                    else:
                        sum_on += batch_sum
                    total_on += n_on
                if results['avg_pump_off'] is not None and n_off > 0:
                    batch_sum = results['avg_pump_off'].astype(np.float64) * n_off
                    if sum_off is None:
                        sum_off = batch_sum
                    else:
                        sum_off += batch_sum
                    total_off += n_off
            final_on = sum_on / total_on if sum_on is not None else None
            final_off = sum_off / total_off if sum_off is not None else None
            final_results = {
                'avg_pump_on': final_on, 'avg_pump_off': final_off,
                'n_pump_on': total_on, 'n_pump_off': total_off,
                'threshold': results['threshold'],
                'ref_intensities': results['ref_intensities']
            }
            step_info = {
                'time_delay_ps': delay_ps, 'position_mm': position_mm,
                'results': final_results, 'avg_times_used': avg_times
            }
            scan_results.append(step_info)
            self.step_result.emit(final_results, self.roi, step_info)
            # Poll gate V/I once per delay step for live display
            if self.front_dev:
                try:
                    vi = self.front_dev.read_vi()
                    self.gate_vi_update.emit("front", vi.v, vi.i)
                except Exception:
                    pass
            if self.dual_mode and self.back_dev:
                try:
                    vi = self.back_dev.read_vi()
                    self.gate_vi_update.emit("back", vi.v, vi.i)
                except Exception:
                    pass
        return scan_results

    def _ramp_gates(self, vf, vb):
        if not self.front_dev:
            return
        self.progress.emit(f"Ramping front gate to {vf:.3f} V ...")
        self._gate_ctrl = SweepController()

        def _on_front(vi):
            self.gate_vi_update.emit("front", vi.v, vi.i)

        self.front_dev.ramp_to_voltage(vf,
                                       ramp_step_v=GATE_RAMP_STEP_V,
                                       ramp_dwell_s=GATE_RAMP_DWELL_S,
                                       controller=self._gate_ctrl,
                                       on_microstep=_on_front)
        if self.dual_mode and self.back_dev:
            self.progress.emit(f"Ramping back gate to {vb:.3f} V ...")
            self._gate_ctrl = SweepController()

            def _on_back(vi):
                self.gate_vi_update.emit("back", vi.v, vi.i)

            self.back_dev.ramp_to_voltage(vb,
                                          ramp_step_v=GATE_RAMP_STEP_V,
                                          ramp_dwell_s=GATE_RAMP_DWELL_S,
                                          controller=self._gate_ctrl,
                                          on_microstep=_on_back)
        time.sleep(self.gate_settle_s)

    def _save_gate_step(self, vf, vb, angle, delay_results, pol_tag=None):
        """Save TAM-compatible *para.txt + *sum.txt for one gate voltage step."""
        try:
            def _fmt_v(v):
                s = f"{abs(v):.3f}".replace('.', 'd')
                return ('p' if v >= 0 else 'n') + s

            vf_tag = f"Vf{_fmt_v(vf)}V"
            vb_tag = f"Vb{_fmt_v(vb)}V" if self.dual_mode else ""
            angle_tag = ""
            if angle is not None:
                astr = f"{angle:.1f}".replace('.', 'p').replace('-', 'n')
                angle_tag = f"_ND{astr}deg"
            pol_tag_str = f"_{pol_tag}" if pol_tag else ""
            prefix = f"{self.save_base}_{vf_tag}{vb_tag}{angle_tag}{pol_tag_str}_{self.timestamp}"
            delays = [r['time_delay_ps'] for r in delay_results]
            avg_used = [r.get('avg_times_used', 1) for r in delay_results]
            crop_w = self.crop[1] if self.crop else (self.camera.width if self.camera else 480)

            para_path = os.path.join(self.save_dir, f"{prefix}para.txt")
            with open(para_path, 'w') as f:
                f.write('\t'.join([f'{d:.6f}' for d in delays]) + '\n')
                f.write('\t'.join([f'{a:.6f}' for a in avg_used]) + '\n')
                f.write('0\n')
                f.write(f' {1.0:.6f} \n')
                f.write(f' {self.pixel_size:.6f} \n')
                f.write(f' {crop_w} \n')
                f.write(f' {self.pixel_size:.6f} \n')
                f.write(f' {self.crop_h} \n')
                f.write(f' {0.0:.6f} \n')
                f.write(f' {self.t0_ps:.6f}\n')
                f.write('0\n0\n0\n0\n0 \n')
                f.write(f' {0.0:.6f}\n')
                f.write(f' {angle if angle is not None else 0.0:.6f}\n')
                f.write(f'{1.000000}\n')
                f.write(f'{0}\n')
                f.write(f' {self.n_frames} \n')
                f.write(f' 1\n')

            sum_path = os.path.join(self.save_dir, f"{prefix}sum.txt")
            with open(sum_path, 'w') as f:
                for r in delay_results:
                    res = r['results']
                    on = self._crop(res['avg_pump_on'])
                    off = self._crop(res['avg_pump_off'])
                    if on is not None and off is not None:
                        lr = _calculate_log_ratio(on, off).T
                        for row in lr:
                            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
        except Exception as e:
            print(f"Warning: save gate step failed: {e}")

    def stop(self):
        self._running = False
        if self._gate_ctrl:
            self._gate_ctrl.abort()


class PumpProbeScanGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pump-Probe Measurement with Delay Stage + Gate Control")
        self.resize(1800, 1000)

        self.camera = SimpleCamera()
        self.stage = DelayStage() if STAGE_AVAILABLE else None
        self.stage_available = STAGE_AVAILABLE
        self.rotation_stage = RotationStage() if ROTATION_AVAILABLE else None
        self.rotation_available = ROTATION_AVAILABLE
        self.pol_stage = RotationStage() if ROTATION_AVAILABLE else None

        self.captured_frames = []
        self.current_results = None
        self.log_ratio_data = None
        self.scan_results = []
        self.power_scan_results = []
        self.is_power_scanning = False

        self.crosshair_x = None
        self.crosshair_y = None

        self.z_min = -10
        self.z_max = 10
        self.use_auto_z = True

        self.wavelength_k = 1.0
        self.wavelength_b = 0.0
        self.space_scale = 0.07
        self.use_calibration = True

        self.use_custom_axis_range = False
        self.custom_x_min = 0
        self.custom_x_max = 256
        self.custom_y_min = 0
        self.custom_y_max = 256

        self.delay_t0_offset = 0.0

        self.use_custom_profile_x_xlim = False
        self.profile_x_xmin = 0.0
        self.profile_x_xmax = 256.0
        self.use_custom_profile_x_ylim = False
        self.profile_x_ymin = -100.0
        self.profile_x_ymax = 100.0
        self.use_custom_profile_y_xlim = False
        self.profile_y_xmin = -100.0
        self.profile_y_xmax = 100.0
        self.use_custom_profile_y_ylim = False
        self.profile_y_ymin = 0.0
        self.profile_y_ymax = 256.0

        self.is_scanning = False
        self.is_single_capturing = False
        self.updating_from_drag = False
        self.time_zero_position_mm = 0.0
        self.range_entries = []
        self.delay_to_avg_map = {}

        self.save_directory = os.path.join(os.path.expanduser("~"), "Desktop")
        self.save_filename = "pump_probe_scan"

        self._last_display_time = 0
        self.live_refresh_interval_ms = 100
        self.fast_live_display = True
        self.monitor_update_every_n = 5
        self._capture_count = 0
        self._display_cycle_count = 0

        self._cached_roi = None
        self._cached_image_shape = None
        self._last_vmin = None
        self._last_vmax = None

        self.width_history = []
        self.width_update_count = 0
        self.gaussian_fitter = GaussianFitter()
        self.gaussian_fit_enabled = True
        self.monitor_levels_auto = True
        self.monitor_levels_min = 0
        self.monitor_levels_max = 255

        self._capture_thread = None
        self._scan_thread = None
        self._power_thread = None
        self._capture_n_frames = 0
        self._capture_roi = (0, 0, 0, 0)

        # Gate sweep state
        self.is_gate_sweeping = False
        self._gate_thread = None
        self._gate_pairs = []
        self._gate_poll_timer = None
        self.front_widget = None   # _GateDevWidget
        self.back_widget = None    # _GateDevWidget

        self._setup_ui()

    # =========================================================
    # Helper methods for widget creation
    # =========================================================

    def _entry(self, default="", width=80):
        e = QLineEdit(str(default))
        e.setFixedWidth(width)
        return e

    def _btn(self, text, callback, color=None, bold=False, enabled=True):
        b = QPushButton(text)
        b.clicked.connect(callback)
        b.setEnabled(enabled)
        style = ""
        if color:
            style += f"background-color: {color};"
        if bold:
            b.setFont(QFont("Arial", 10, QFont.Bold))
        if style:
            b.setStyleSheet(style)
        return b

    def _label(self, text, bold=False, size=9, color=None):
        lbl = QLabel(text)
        weight = QFont.Bold if bold else QFont.Normal
        lbl.setFont(QFont("Arial", size, weight))
        if color:
            lbl.setStyleSheet(f"color: {color};")
        return lbl

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _scroll_widget(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        scroll.setWidget(content)
        return scroll, lay

    # =========================================================
    # UI Setup
    # =========================================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_tabs = QTabWidget()
        left_tabs.setFixedWidth(430)
        left_tabs.setFont(QFont("Arial", 9, QFont.Bold))
        left_tabs.setStyleSheet(
            "QTabBar::tab { min-width: 40px; padding: 4px 4px; }")
        splitter.addWidget(left_tabs)

        self._create_hardware_tab(left_tabs)
        self._create_scan_tab(left_tabs)
        self._create_analysis_tab(left_tabs)
        self._create_display_settings_tab(left_tabs)
        self._create_power_tab(left_tabs)
        self._create_gate_hw_tab(left_tabs)
        self._create_gate_sweep_tab(left_tabs)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        title = self._label("Pump-Probe Delay Scan Results", bold=True, size=14)
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)

        self.display_tabs = QTabWidget()
        self.display_tabs.setFont(QFont("Arial", 11, QFont.Bold))
        self.display_tabs.setStyleSheet(
            "QTabBar::tab { min-width: 120px; min-height: 30px; padding: 6px 14px; font-size: 13px; }"
            "QTabBar::tab:selected { background: #ddeeff; border-bottom: 3px solid #3388cc; }")
        right_layout.addWidget(self.display_tabs)

        self._create_main_analysis_tab()
        self._create_monitoring_tab()
        self._create_gaussian_tab()
        self._create_gate_vi_tab()

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status_label = self._label("Ready", size=10)
        right_layout.addWidget(self.status_label)

    # --- Tab 1: Hardware ---
    def _create_hardware_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "Hardware")

        lay.addWidget(self._label("Camera Control", bold=True, size=12))
        lay.addWidget(self._btn("Initialize Camera", self.init_camera, color="lightblue", bold=True))
        self.camera_status = self._label("○ Not Connected", color="red", bold=True, size=10)
        lay.addWidget(self.camera_status)

        lay.addWidget(self._label("Delay Stage Control", bold=True, size=12))
        lay.addWidget(self._label("Serial Number:"))
        self.stage_serial_entry = self._entry("104507475", 200)
        lay.addWidget(self.stage_serial_entry)
        lay.addWidget(self._btn("Initialize Stage", self.init_stage, color="lightblue", bold=True))
        self.stage_status = self._label("○ Not Connected", color="red", bold=True, size=10)
        lay.addWidget(self.stage_status)

        h = QHBoxLayout()
        h.addWidget(self._label("Time Delay (ps):"))
        self.pos_entry = self._entry("0.0", 80)
        h.addWidget(self.pos_entry)
        w = QWidget(); w.setLayout(h); lay.addWidget(w)
        lay.addWidget(self._btn("Move To Time Delay", self.move_stage, color="lightcyan"))
        lay.addWidget(self._btn("Home Stage", self.home_stage, color="lightcyan"))

        self.current_pos_label = self._label("Current: -- mm\n(-- ps from t=0)", color="blue", size=8)
        lay.addWidget(self.current_pos_label)

        t0_row = QHBoxLayout()
        t0_row.addWidget(self._label("Time Zero (mm):", bold=True, size=9))
        self.time_zero_entry = self._entry("0.0000", 120)
        t0_row.addWidget(self.time_zero_entry)
        t0w = QWidget(); t0w.setLayout(t0_row); lay.addWidget(t0w)
        t0_btns = QHBoxLayout()
        t0_btns.addWidget(self._btn("Set from Current Pos", self.set_time_zero_from_stage, color="yellow"))
        t0_btns.addWidget(self._btn("Apply Typed Value", self.apply_time_zero_entry, color="lightyellow"))
        t0bw = QWidget(); t0bw.setLayout(t0_btns); lay.addWidget(t0bw)
        self.time_zero_label = self._label("t=0 at: 0.0000 mm", color="green", size=8)
        lay.addWidget(self.time_zero_label)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Auto-Save Settings", bold=True, size=11))
        self.auto_save_chk = QCheckBox("Auto-save scan data")
        self.auto_save_chk.setChecked(True)
        lay.addWidget(self.auto_save_chk)
        lay.addWidget(self._label("Filename base:", size=8))
        self.save_filename_entry = self._entry(self.save_filename, 200)
        lay.addWidget(self.save_filename_entry)
        lay.addWidget(self._label("Save subfolder (under DATA_DIR):", size=8))
        self.save_name_edit = self._entry(self._default_save_name(), 200)
        lay.addWidget(self.save_name_edit)
        self.data_dir_label = self._label(f"DATA_DIR: {DATA_DIR}", color="blue", size=7)
        self.data_dir_label.setWordWrap(True)
        lay.addWidget(self.data_dir_label)

        lay.addWidget(self._separator())
        self.btn_single = self._btn("Start Continuous Capture", self.single_capture,
                                    color="lightgreen", bold=True, enabled=False)
        lay.addWidget(self.btn_single)
        self.btn_scan = self._btn("Start Delay Scan", self.start_scan,
                                  color="orange", bold=True, enabled=False)
        lay.addWidget(self.btn_scan)
        self.btn_stop = self._btn("Stop", self.stop_scan, color="red", bold=True, enabled=False)
        lay.addWidget(self.btn_stop)
        self.btn_save = self._btn("Save Results", self.save_results,
                                  color="lightyellow", bold=True, enabled=False)
        lay.addWidget(self.btn_save)
        lay.addWidget(self._btn("Close All", self.close_all, color="salmon", bold=True))
        lay.addStretch()

    # --- Tab 2: Scan Setup ---
    def _create_scan_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "Scan Setup")

        lay.addWidget(self._label("Scan Parameters", bold=True, size=12))
        lay.addWidget(self._label("Number of Frames (per capture):", size=10))
        self.n_frames_entry = self._entry("400", 100)
        lay.addWidget(self.n_frames_entry)
        lay.addWidget(self._label("Average Times (captures to average):", size=10))
        self.avg_times_entry = self._entry("1", 100)
        lay.addWidget(self.avg_times_entry)

        lay.addWidget(self._label("Delay Scan Setup:", bold=True, size=10))
        mode_w = QWidget()
        mode_h = QHBoxLayout(mode_w)
        self.radio_simple = QRadioButton("Simple")
        self.radio_multi = QRadioButton("Multi-Range")
        self.radio_simple.setChecked(True)
        self.scan_mode_group = QButtonGroup()
        self.scan_mode_group.addButton(self.radio_simple)
        self.scan_mode_group.addButton(self.radio_multi)
        mode_h.addWidget(self.radio_simple)
        mode_h.addWidget(self.radio_multi)
        lay.addWidget(mode_w)
        self.radio_simple.toggled.connect(self.update_scan_mode)

        self.simple_scan_frame = QWidget()
        sf_lay = QVBoxLayout(self.simple_scan_frame)
        sf_lay.setContentsMargins(0, 0, 0, 0)
        g = QGridLayout()
        g.addWidget(self._label("Start (ps):"), 0, 0)
        self.scan_start = self._entry("-10", 80); g.addWidget(self.scan_start, 0, 1)
        g.addWidget(self._label("Steps:"), 1, 0)
        self.scan_steps = self._entry("5", 80); g.addWidget(self.scan_steps, 1, 1)
        g.addWidget(self._label("Increment:"), 2, 0)
        self.scan_increment = self._entry("5", 80); g.addWidget(self.scan_increment, 2, 1)
        gw = QWidget(); gw.setLayout(g); sf_lay.addWidget(gw)
        self.scan_preview_label = self._label("End: 10.00 ps | Range: 20.00 ps", color="blue", size=8)
        sf_lay.addWidget(self.scan_preview_label)
        self.scan_start.textChanged.connect(self.update_scan_preview)
        self.scan_steps.textChanged.connect(self.update_scan_preview)
        self.scan_increment.textChanged.connect(self.update_scan_preview)
        bh = QHBoxLayout()
        bh.addWidget(self._btn("Around t=0", self.scan_around_zero, color="lightyellow"))
        bh.addWidget(self._btn("From t=0", self.scan_from_zero, color="lightyellow"))
        bw = QWidget(); bw.setLayout(bh); sf_lay.addWidget(bw)
        lay.addWidget(self.simple_scan_frame)

        self.multi_scan_frame = QWidget()
        mf_lay = QVBoxLayout(self.multi_scan_frame)
        mf_lay.setContentsMargins(0, 0, 0, 0)
        mf_lay.addWidget(self._label("Define Time Ranges:", bold=True, size=10))
        self.ranges_container = QVBoxLayout()
        rc_w = QWidget(); rc_w.setLayout(self.ranges_container); mf_lay.addWidget(rc_w)
        rb = QHBoxLayout()
        rb.addWidget(self._btn("+ Add Range", self._add_range_default, color="lightgreen"))
        rb.addWidget(self._btn("- Remove Last", self.remove_range_entry, color="lightcoral"))
        rb.addWidget(self._btn("+ Add Point", self.add_t0_point, color="yellow"))
        rbw = QWidget(); rbw.setLayout(rb); mf_lay.addWidget(rbw)
        mf_lay.addWidget(self._btn("Generate Multi-Range Scan", self.generate_multi_range_scan, color="orange"))
        self.multi_preview_label = self._label("Total: 0 points", color="blue", size=8)
        mf_lay.addWidget(self.multi_preview_label)
        lay.addWidget(self.multi_scan_frame)
        self.multi_scan_frame.hide()

        self.add_range_entry(-5, 5, 0.5, 3)
        self.add_range_entry(5, 20, 1, 2)
        self.add_range_entry(20, 100, 5, 1)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Generated Time Delays (ps):", bold=True, size=10))
        h_t0 = QHBoxLayout()
        h_t0.addWidget(self._label("t0 offset (ps):", bold=True))
        self.delay_t0_entry = self._entry("0.0", 80)
        h_t0.addWidget(self.delay_t0_entry)
        hw = QWidget(); hw.setLayout(h_t0); lay.addWidget(hw)
        self.delay_positions = QTextEdit()
        self.delay_positions.setMaximumHeight(100)
        self.delay_positions.setPlainText("-10, -5, 0, 5, 10")
        lay.addWidget(self.delay_positions)
        lay.addStretch()

    # --- Tab 3: Analysis ---
    def _create_analysis_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "Analysis")

        lay.addWidget(self._label("Reference ROI:", bold=True, size=11))
        g = QGridLayout()
        g.addWidget(self._label("X:"), 0, 0)
        self.roi_x = self._entry("625", 60); g.addWidget(self.roi_x, 0, 1)
        g.addWidget(self._label("Width:"), 0, 2)
        self.roi_w = self._entry("5", 60); g.addWidget(self.roi_w, 0, 3)
        g.addWidget(self._label("Y:"), 1, 0)
        self.roi_y = self._entry("10", 60); g.addWidget(self.roi_y, 1, 1)
        g.addWidget(self._label("Height:"), 1, 2)
        self.roi_h = self._entry("5", 60); g.addWidget(self.roi_h, 1, 3)
        gw = QWidget(); gw.setLayout(g); lay.addWidget(gw)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Save / Display Region:", bold=True, size=11))
        lay.addWidget(self._label(
            "Crop the image for display & save. Reference ROI still uses full frame.",
            size=8, color="gray"))
        sr = QGridLayout()
        sr.addWidget(self._label("X Start:"), 0, 0)
        self.save_region_x = self._entry("0", 60); sr.addWidget(self.save_region_x, 0, 1)
        sr.addWidget(self._label("Width:"), 0, 2)
        self.save_region_w = self._entry("480", 60); sr.addWidget(self.save_region_w, 0, 3)
        self.save_region_enable = QCheckBox("Enable crop")
        self.save_region_enable.setChecked(True)
        sr.addWidget(self.save_region_enable, 1, 0, 1, 2)
        srw = QWidget(); srw.setLayout(sr); lay.addWidget(srw)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Profile Position:", bold=True, size=11))
        lay.addWidget(self._label("X Position:"))
        self.profile_x_slider = QSlider(Qt.Horizontal)
        self.profile_x_slider.setRange(0, 255)
        self.profile_x_slider.setValue(128)
        self.profile_x_slider.valueChanged.connect(self.update_profiles_from_sliders)
        lay.addWidget(self.profile_x_slider)
        lay.addWidget(self._label("Y Position:"))
        self.profile_y_slider = QSlider(Qt.Horizontal)
        self.profile_y_slider.setRange(0, 255)
        self.profile_y_slider.setValue(128)
        self.profile_y_slider.valueChanged.connect(self.update_profiles_from_sliders)
        lay.addWidget(self.profile_y_slider)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Colormap Range:", bold=True, size=11))
        zh = QHBoxLayout()
        zh.addWidget(self._label("Min:"))
        self.z_min_entry = self._entry("-10", 80); zh.addWidget(self.z_min_entry)
        zh.addWidget(self._label("Max:"))
        self.z_max_entry = self._entry("10", 80); zh.addWidget(self.z_max_entry)
        zw = QWidget(); zw.setLayout(zh); lay.addWidget(zw)
        lay.addWidget(self._btn("Apply Z Range", self.apply_z_range, color="lightyellow"))
        lay.addStretch()

    # --- Tab 4: Display Settings ---
    def _create_display_settings_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "Display Settings")

        lay.addWidget(self._label("Intensity Range", bold=True, size=10))
        g = QGridLayout()
        g.addWidget(self._label("Min:"), 0, 0)
        self.intensity_min_entry = self._entry("-1.0", 80); g.addWidget(self.intensity_min_entry, 0, 1)
        g.addWidget(self._label("Max:"), 0, 2)
        self.intensity_max_entry = self._entry("1.0", 80); g.addWidget(self.intensity_max_entry, 0, 3)
        gw = QWidget(); gw.setLayout(g); lay.addWidget(gw)
        bh = QHBoxLayout()
        bh.addWidget(self._btn("Apply", self.apply_intensity_range, color="lightgreen"))
        bh.addWidget(self._btn("Auto", self.auto_intensity_range, color="lightblue"))
        bw = QWidget(); bw.setLayout(bh); lay.addWidget(bw)

        lay.addWidget(self._label("Main Plot Axis (X & Y)", bold=True, size=10))
        g2 = QGridLayout()
        g2.addWidget(self._label("X Min:"), 0, 0)
        self.x_min_entry = self._entry("0", 80); g2.addWidget(self.x_min_entry, 0, 1)
        g2.addWidget(self._label("X Max:"), 0, 2)
        self.x_max_entry = self._entry("256", 80); g2.addWidget(self.x_max_entry, 0, 3)
        g2.addWidget(self._label("Y Min:"), 1, 0)
        self.y_min_entry = self._entry("0", 80); g2.addWidget(self.y_min_entry, 1, 1)
        g2.addWidget(self._label("Y Max:"), 1, 2)
        self.y_max_entry = self._entry("256", 80); g2.addWidget(self.y_max_entry, 1, 3)
        gw2 = QWidget(); gw2.setLayout(g2); lay.addWidget(gw2)
        bh2 = QHBoxLayout()
        bh2.addWidget(self._btn("Apply", self.apply_axis_ranges, color="lightgreen"))
        bh2.addWidget(self._btn("Reset", self.reset_axis_ranges, color="lightblue"))
        bw2 = QWidget(); bw2.setLayout(bh2); lay.addWidget(bw2)

        lay.addWidget(self._label("Profile Plot Ranges", bold=True, size=10))
        lay.addWidget(self._label("X Profile (Top) - leave blank for auto", size=8, color="gray"))
        gp = QGridLayout()
        gp.addWidget(self._label("X Min:"), 0, 0)
        self.profile_x_xmin_entry = self._entry("", 60); gp.addWidget(self.profile_x_xmin_entry, 0, 1)
        gp.addWidget(self._label("X Max:"), 0, 2)
        self.profile_x_xmax_entry = self._entry("", 60); gp.addWidget(self.profile_x_xmax_entry, 0, 3)
        gp.addWidget(self._label("Y Min:"), 1, 0)
        self.profile_x_ymin_entry = self._entry("", 60); gp.addWidget(self.profile_x_ymin_entry, 1, 1)
        gp.addWidget(self._label("Y Max:"), 1, 2)
        self.profile_x_ymax_entry = self._entry("", 60); gp.addWidget(self.profile_x_ymax_entry, 1, 3)
        gpw = QWidget(); gpw.setLayout(gp); lay.addWidget(gpw)
        lay.addWidget(self._label("Y Profile (Right) - leave blank for auto", size=8, color="gray"))
        gp2 = QGridLayout()
        gp2.addWidget(self._label("X Min:"), 0, 0)
        self.profile_y_xmin_entry = self._entry("", 60); gp2.addWidget(self.profile_y_xmin_entry, 0, 1)
        gp2.addWidget(self._label("X Max:"), 0, 2)
        self.profile_y_xmax_entry = self._entry("", 60); gp2.addWidget(self.profile_y_xmax_entry, 0, 3)
        gp2.addWidget(self._label("Y Min:"), 1, 0)
        self.profile_y_ymin_entry = self._entry("", 60); gp2.addWidget(self.profile_y_ymin_entry, 1, 1)
        gp2.addWidget(self._label("Y Max:"), 1, 2)
        self.profile_y_ymax_entry = self._entry("", 60); gp2.addWidget(self.profile_y_ymax_entry, 1, 3)
        gpw2 = QWidget(); gpw2.setLayout(gp2); lay.addWidget(gpw2)
        bh3 = QHBoxLayout()
        bh3.addWidget(self._btn("Apply Profile Ranges", self.apply_profile_ranges, color="lightgreen"))
        bh3.addWidget(self._btn("Reset Profile Ranges", self.reset_profile_ranges, color="lightblue"))
        bw3 = QWidget(); bw3.setLayout(bh3); lay.addWidget(bw3)

        lay.addWidget(self._label("Axis Calibration", bold=True, size=10))
        gc = QGridLayout()
        gc.addWidget(self._label("k (nm/pix):"), 0, 0)
        self.wavelength_k_entry = self._entry("1.0", 80); gc.addWidget(self.wavelength_k_entry, 0, 1)
        gc.addWidget(self._label("b (nm):"), 1, 0)
        self.wavelength_b_entry = self._entry("0.0", 80); gc.addWidget(self.wavelength_b_entry, 1, 1)
        gc.addWidget(self._label("Space (um/pix):"), 2, 0)
        self.space_scale_entry = self._entry("0.07", 80); gc.addWidget(self.space_scale_entry, 2, 1)
        gcw = QWidget(); gcw.setLayout(gc); lay.addWidget(gcw)
        lay.addWidget(self._btn("Apply Calibration", self.apply_calibration, color="lightgreen"))
        lay.addStretch()

    # --- Tab 5: Power Control ---
    def _create_power_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "Power Control")

        self.power_enable_chk = QCheckBox("Enable Power Control")
        self.power_enable_chk.setFont(QFont("Arial", 11, QFont.Bold))
        self.power_enable_chk.toggled.connect(self._toggle_power_controls)
        lay.addWidget(self.power_enable_chk)

        self.pwr_controls_frame = QWidget()
        pwr_lay = QVBoxLayout(self.pwr_controls_frame)
        pwr_lay.setContentsMargins(0, 0, 0, 0)

        pwr_lay.addWidget(self._label("Rotation Stage", bold=True, size=12))
        pwr_lay.addWidget(self._label("Serial Number:"))
        self.rot_serial_entry = self._entry("27600911", 200)
        pwr_lay.addWidget(self.rot_serial_entry)
        pwr_lay.addWidget(self._btn("Initialize Rotation Stage", self.init_rotation_stage,
                                    color="lightblue", bold=True))
        self.rot_status = self._label("○ Not Connected", color="red", bold=True, size=10)
        pwr_lay.addWidget(self.rot_status)
        self.rot_angle_label = self._label("Angle: -- deg", color="blue")
        pwr_lay.addWidget(self.rot_angle_label)

        pwr_lay.addWidget(self._separator())
        pwr_lay.addWidget(self._label("Manual Angle Control", bold=True, size=11))
        ah = QHBoxLayout()
        ah.addWidget(self._label("Angle (deg):"))
        self.rot_angle_entry = self._entry("0.0", 80)
        ah.addWidget(self.rot_angle_entry)
        aw = QWidget(); aw.setLayout(ah); pwr_lay.addWidget(aw)
        pwr_lay.addWidget(self._btn("Move To Angle", self.move_rotation_stage, color="lightcyan"))
        sh = QHBoxLayout()
        sh.addWidget(self._label("Step (deg):"))
        self.rot_step_entry = self._entry("5.0", 60)
        sh.addWidget(self.rot_step_entry)
        sw = QWidget(); sw.setLayout(sh); pwr_lay.addWidget(sw)
        sbh = QHBoxLayout()
        sbh.addWidget(self._btn("+ Step", lambda: self.step_rotation_stage(+1), color="lightgreen"))
        sbh.addWidget(self._btn("- Step", lambda: self.step_rotation_stage(-1), color="lightyellow"))
        sbw = QWidget(); sbw.setLayout(sbh); pwr_lay.addWidget(sbw)
        hbh = QHBoxLayout()
        hbh.addWidget(self._btn("Home", self.home_rotation_stage, color="lightcyan"))
        hbh.addWidget(self._btn("Stop", self.stop_rotation_stage, color="salmon"))
        hbw = QWidget(); hbw.setLayout(hbh); pwr_lay.addWidget(hbw)

        pwr_lay.addWidget(self._separator())
        pwr_lay.addWidget(self._label("Power Dependence Scan", bold=True, size=11))
        pwr_lay.addWidget(self._label("Runs full delay scan (from Scan Setup tab)\nat each angle below.",
                                      size=8, color="gray"))
        pg2 = QGridLayout()
        pg2.addWidget(self._label("Start (deg):"), 0, 0)
        self.pwr_start_entry = self._entry("0", 60); pg2.addWidget(self.pwr_start_entry, 0, 1)
        pg2.addWidget(self._label("End (deg):"), 1, 0)
        self.pwr_end_entry = self._entry("90", 60); pg2.addWidget(self.pwr_end_entry, 1, 1)
        pg2.addWidget(self._label("Step (deg):"), 2, 0)
        self.pwr_step_entry = self._entry("10", 60); pg2.addWidget(self.pwr_step_entry, 2, 1)
        pg2.addWidget(self._label("Settle (s):"), 3, 0)
        self.pwr_settle_entry = self._entry("0.5", 60); pg2.addWidget(self.pwr_settle_entry, 3, 1)
        pg2w = QWidget(); pg2w.setLayout(pg2); pwr_lay.addWidget(pg2w)
        pwr_lay.addWidget(self._btn("Generate Angle List", self.generate_power_scan_positions,
                                    color="lightyellow"))
        pwr_lay.addWidget(self._label("Angles (deg) - edit freely:"))
        self.pwr_positions_text = QTextEdit()
        self.pwr_positions_text.setMaximumHeight(60)
        self.pwr_positions_text.setPlainText("0, 10, 20, 30, 45, 60, 90")
        pwr_lay.addWidget(self.pwr_positions_text)
        self.btn_power_scan = self._btn("Start Power Scan", self.start_power_scan,
                                        color="orange", bold=True, enabled=False)
        pwr_lay.addWidget(self.btn_power_scan)
        self.btn_power_stop = self._btn("Stop Power Scan", self.stop_power_scan,
                                        color="red", bold=True, enabled=False)
        pwr_lay.addWidget(self.btn_power_stop)

        lay.addWidget(self.pwr_controls_frame)
        self.pwr_controls_frame.hide()

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Polarization Control", bold=True, size=12))
        lay.addWidget(self._label(
            "Second rotation stage for polarization (LH / RH).",
            size=8, color="gray"))

        self.pol_enable_chk = QCheckBox("Enable Polarization Control")
        self.pol_enable_chk.setFont(QFont("Arial", 10, QFont.Bold))
        self.pol_enable_chk.toggled.connect(self._toggle_pol_controls)
        lay.addWidget(self.pol_enable_chk)

        self.pol_controls_frame = QWidget()
        pol_lay = QVBoxLayout(self.pol_controls_frame)
        pol_lay.setContentsMargins(0, 0, 0, 0)

        pol_lay.addWidget(self._label("Polarization Rotation Stage", bold=True, size=11))
        pol_lay.addWidget(self._label("Serial Number:"))
        self.pol_serial_entry = self._entry("27264008", 200)
        pol_lay.addWidget(self.pol_serial_entry)
        pol_lay.addWidget(self._btn("Initialize Pol. Stage", self.init_pol_stage,
                                    color="lightblue", bold=True))
        self.pol_status = self._label("○ Not Connected", color="red", bold=True, size=10)
        pol_lay.addWidget(self.pol_status)
        self.pol_angle_label = self._label("Angle: -- deg", color="blue")
        pol_lay.addWidget(self.pol_angle_label)

        pol_lay.addWidget(self._separator())
        pol_lay.addWidget(self._label("Manual Angle Control", bold=True, size=11))
        pah = QHBoxLayout()
        pah.addWidget(self._label("Angle (deg):"))
        self.pol_angle_entry = self._entry("0.0", 80)
        pah.addWidget(self.pol_angle_entry)
        paw = QWidget(); paw.setLayout(pah); pol_lay.addWidget(paw)
        pol_lay.addWidget(self._btn("Move To Angle", self.move_pol_stage, color="lightcyan"))
        psh = QHBoxLayout()
        psh.addWidget(self._label("Step (deg):"))
        self.pol_step_entry = self._entry("5.0", 60)
        psh.addWidget(self.pol_step_entry)
        psw = QWidget(); psw.setLayout(psh); pol_lay.addWidget(psw)
        psbh = QHBoxLayout()
        psbh.addWidget(self._btn("+ Step", lambda: self.step_pol_stage(+1), color="lightgreen"))
        psbh.addWidget(self._btn("- Step", lambda: self.step_pol_stage(-1), color="lightyellow"))
        psbw = QWidget(); psbw.setLayout(psbh); pol_lay.addWidget(psbw)
        phbh = QHBoxLayout()
        phbh.addWidget(self._btn("Home", self.home_pol_stage, color="lightcyan"))
        phbh.addWidget(self._btn("Stop", self.stop_pol_stage, color="salmon"))
        phbw = QWidget(); phbw.setLayout(phbh); pol_lay.addWidget(phbw)

        pol_lay.addWidget(self._separator())
        pol_lay.addWidget(self._label("Polarization Scan Settings", bold=True, size=11))
        pg3 = QGridLayout()
        pg3.addWidget(self._label("LH angle (deg):"), 0, 0)
        self.pol_lh_entry = self._entry("0.0", 80)
        self.pol_lh_entry.textChanged.connect(self._update_pol_rh_label)
        pg3.addWidget(self.pol_lh_entry, 0, 1)
        self.pol_rh_label = self._label("RH = 45.0 deg", color="blue", size=9)
        pg3.addWidget(self.pol_rh_label, 1, 0, 1, 2)
        pg3.addWidget(self._label("Settle (s):"), 2, 0)
        self.pol_settle_entry = self._entry("0.5", 60)
        pg3.addWidget(self.pol_settle_entry, 2, 1)
        pg3w = QWidget(); pg3w.setLayout(pg3); pol_lay.addWidget(pg3w)

        lay.addWidget(self.pol_controls_frame)
        self.pol_controls_frame.hide()
        lay.addStretch()

    # --- Tab 6: Gate Hardware (front + back in one tab) ---
    def _create_gate_hw_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "Gate HW")
        if not KEITHLEY_AVAILABLE:
            lay.addWidget(self._label("Keithley modules not available.", color="red"))
            lay.addStretch()
            return
        self.front_widget = _GateDevWidget("Front Gate", DEFAULT_FRONT_RESOURCE)
        self.back_widget  = _GateDevWidget("Back Gate",  DEFAULT_BACK_RESOURCE)
        lay.addWidget(self.front_widget)
        lay.addWidget(self.back_widget)
        lay.addStretch()

    # --- Tab 8: Gate Sweep ---
    def _create_gate_sweep_tab(self, tabs):
        scroll, lay = self._scroll_widget()
        tabs.addTab(scroll, "GateSw")

        lay.addWidget(self._label("Gate Sweep", bold=True, size=12))

        # Dual mode
        mode_w = QWidget(); mode_h = QHBoxLayout(mode_w)
        self.gate_single_radio = QRadioButton("Single (front only)")
        self.gate_dual_radio = QRadioButton("Dual (front + back)")
        self.gate_single_radio.setChecked(True)
        self.gate_mode_group = QButtonGroup()
        self.gate_mode_group.addButton(self.gate_single_radio)
        self.gate_mode_group.addButton(self.gate_dual_radio)
        mode_h.addWidget(self.gate_single_radio)
        mode_h.addWidget(self.gate_dual_radio)
        lay.addWidget(mode_w)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Front gate voltages:", bold=True, size=10))

        # Range vs manual
        rv_w = QWidget(); rv_h = QHBoxLayout(rv_w)
        self.gate_range_radio = QRadioButton("Range")
        self.gate_manual_radio = QRadioButton("Manual list")
        self.gate_range_radio.setChecked(True)
        self.gate_input_group = QButtonGroup()
        self.gate_input_group.addButton(self.gate_range_radio)
        self.gate_input_group.addButton(self.gate_manual_radio)
        rv_h.addWidget(self.gate_range_radio); rv_h.addWidget(self.gate_manual_radio)
        lay.addWidget(rv_w)

        # Range controls
        self.gate_range_frame = QWidget()
        rg = QGridLayout(self.gate_range_frame)
        rg.setContentsMargins(0, 0, 0, 0)
        rg.addWidget(self._label("Start (V):"), 0, 0)
        self.gate_v0_entry = self._entry("0.0", 70); rg.addWidget(self.gate_v0_entry, 0, 1)
        rg.addWidget(self._label("Step (V):"), 0, 2)
        self.gate_step_entry = self._entry("1.0", 70); rg.addWidget(self.gate_step_entry, 0, 3)
        rg.addWidget(self._label("N pts:"), 1, 0)
        self.gate_n_entry = self._entry("5", 70); rg.addWidget(self.gate_n_entry, 1, 1)
        self.gate_range_preview = self._label("End: 4.0 V", color="blue", size=8)
        rg.addWidget(self.gate_range_preview, 1, 2, 1, 2)
        lay.addWidget(self.gate_range_frame)

        for w in (self.gate_v0_entry, self.gate_step_entry, self.gate_n_entry):
            w.textChanged.connect(self._update_gate_range_preview)

        # Manual controls
        self.gate_manual_frame = QWidget()
        ml = QVBoxLayout(self.gate_manual_frame)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self._label("Comma-separated voltages (V):", size=8))
        self.gate_manual_edit = QLineEdit("0, 1, 2, 3, 4")
        self.gate_manual_edit.setMinimumWidth(200)
        ml.addWidget(self.gate_manual_edit)
        lay.addWidget(self.gate_manual_frame)
        self.gate_manual_frame.hide()

        self.gate_range_radio.toggled.connect(self._update_gate_input_mode)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Back gate (dual mode):", bold=True, size=10))
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self._label("Vback = Vfront \u00d7 ratio   Ratio:"))
        from PyQt5.QtWidgets import QDoubleSpinBox as _DSB
        self.gate_ratio_spin = _DSB()
        self.gate_ratio_spin.setDecimals(3); self.gate_ratio_spin.setRange(-1000.0, 1000.0)
        self.gate_ratio_spin.setValue(1.8); self.gate_ratio_spin.setFixedWidth(80)
        ratio_row.addWidget(self.gate_ratio_spin)
        rw = QWidget(); rw.setLayout(ratio_row); lay.addWidget(rw)
        self.gate_back_preview = self._label("Back: —", color="blue", size=8)
        self.gate_back_preview.setWordWrap(True)
        lay.addWidget(self.gate_back_preview)
        self.gate_ratio_spin.valueChanged.connect(self._update_gate_range_preview)
        self.gate_manual_edit.textChanged.connect(self._update_gate_range_preview)
        self.gate_dual_radio.toggled.connect(self._update_gate_range_preview)

        lay.addWidget(self._separator())
        timing_row = QHBoxLayout()
        timing_row.addWidget(self._label("Settle after ramp (ms):"))
        from PyQt5.QtWidgets import QSpinBox as _SB
        self.gate_settle_spin = _SB()
        self.gate_settle_spin.setRange(0, 60000); self.gate_settle_spin.setValue(500)
        self.gate_settle_spin.setFixedWidth(80)
        timing_row.addWidget(self.gate_settle_spin)
        tw = QWidget(); tw.setLayout(timing_row); lay.addWidget(tw)

        lay.addWidget(self._separator())
        lay.addWidget(self._label("Loop order (when power scan active):", bold=True, size=10))
        lo_w = QWidget(); lo_h = QHBoxLayout(lo_w)
        self.gate_outer_radio = QRadioButton("Gate outer → Power inner")
        self.power_outer_radio = QRadioButton("Power outer → Gate inner")
        self.gate_outer_radio.setChecked(True)
        self.loop_order_group = QButtonGroup()
        self.loop_order_group.addButton(self.gate_outer_radio)
        self.loop_order_group.addButton(self.power_outer_radio)
        lo_h.addWidget(self.gate_outer_radio); lo_h.addWidget(self.power_outer_radio)
        lay.addWidget(lo_w)

        lay.addWidget(self._separator())
        btn_row = QHBoxLayout()
        self.btn_gate_scan = self._btn("Start Gate Sweep", self.start_gate_sweep,
                                       color="darkorange", bold=True, enabled=False)
        self.btn_gate_stop = self._btn("Stop Gate", self.stop_gate_sweep,
                                       color="red", bold=True, enabled=False)
        btn_row.addWidget(self.btn_gate_scan); btn_row.addWidget(self.btn_gate_stop)
        bw = QWidget(); bw.setLayout(btn_row); lay.addWidget(bw)
        lay.addStretch()

        self._update_gate_range_preview()

    # =========================================================
    # Display tabs (PyQtGraph)
    # =========================================================

    def _create_gate_vi_tab(self):
        tab = QWidget()
        self.display_tabs.addTab(tab, "Gate V/I")
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)
        if not KEITHLEY_AVAILABLE:
            lay.addWidget(self._label("Keithley modules not available.", color="red"))
            return
        lay.addWidget(self._label("Front Gate V/I", bold=True, size=10))
        self.gate_vi_front = GatePlotWidget()
        lay.addWidget(self.gate_vi_front)
        lay.addWidget(self._label("Back Gate V/I", bold=True, size=10))
        self.gate_vi_back = GatePlotWidget()
        lay.addWidget(self.gate_vi_back)
        # Start the polling timer
        self._gate_poll_timer = QTimer(self)
        self._gate_poll_timer.setInterval(GATE_POLL_MS)
        self._gate_poll_timer.timeout.connect(self._poll_gates)
        self._gate_poll_timer.start()

    def _create_main_analysis_tab(self):
        tab = QWidget()
        self.display_tabs.addTab(tab, "Main Analysis")
        layout = QGridLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)

        self.profile_x_plot = pg.PlotWidget(title="X Profile")
        self.profile_x_plot.setMaximumHeight(200)
        self.profile_x_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.profile_x_plot, 0, 0, 1, 2)

        self.main_plot = pg.PlotWidget()
        self.main_plot.setLabel('bottom', 'X (pixels)')
        self.main_plot.setLabel('left', 'Y (pixels)')
        self.main_plot.setTitle('log(Pump ON / Pump OFF)', color='purple')
        self.main_img = pg.ImageItem()
        self.main_img.setLookupTable(RDBU_R_LUT)
        self.main_plot.addItem(self.main_img)
        layout.addWidget(self.main_plot, 1, 0)

        self.cbar = pg.ColorBarItem(values=(-1, 1), colorMap=pg.ColorMap(
            np.linspace(0, 1, RDBU_R_LUT.shape[0]), RDBU_R_LUT))
        self.cbar.setImageItem(self.main_img, insert_in=self.main_plot.plotItem)

        self.roi_rect_main = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.main_plot.addItem(self.roi_rect_main)

        self.h_crosshair = pg.InfiniteLine(angle=0, movable=True,
                                           pen=pg.mkPen('y', width=2))
        self.v_crosshair = pg.InfiniteLine(angle=90, movable=True,
                                           pen=pg.mkPen('y', width=2))
        self.main_plot.addItem(self.h_crosshair)
        self.main_plot.addItem(self.v_crosshair)
        self.h_crosshair.sigPositionChanged.connect(self._on_crosshair_moved)
        self.v_crosshair.sigPositionChanged.connect(self._on_crosshair_moved)
        self.main_plot.scene().sigMouseClicked.connect(self._on_main_click)

        self.profile_y_plot = pg.PlotWidget(title="Space Profile")
        self.profile_y_plot.setMaximumWidth(250)
        self.profile_y_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.profile_y_plot, 1, 1)

        self._px_line = self.profile_x_plot.plot(pen=pg.mkPen('b', width=2))
        self._px_marker = pg.ScatterPlotItem(pen=None, brush='r', size=10)
        self.profile_x_plot.addItem(self._px_marker)
        self.profile_x_plot.addItem(
            pg.InfiniteLine(angle=0, pen=pg.mkPen('gray', style=Qt.DashLine, width=0.5)))

        self._py_line = self.profile_y_plot.plot(pen=pg.mkPen('r', width=2))
        self._py_marker = pg.ScatterPlotItem(pen=None, brush='r', size=10)
        self.profile_y_plot.addItem(self._py_marker)
        self.profile_y_plot.addItem(
            pg.InfiniteLine(angle=90, pen=pg.mkPen('gray', style=Qt.DashLine, width=0.5)))

        btn_goto_fit = QPushButton("Show Gaussian Fit >>")
        btn_goto_fit.setFont(QFont("Arial", 9, QFont.Bold))
        btn_goto_fit.setStyleSheet("background-color: #eeddff; padding: 3px;")
        btn_goto_fit.setFixedHeight(26)
        btn_goto_fit.clicked.connect(lambda: self.display_tabs.setCurrentIndex(2))
        layout.addWidget(btn_goto_fit, 2, 0, 1, 2)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 3)
        layout.setRowStretch(2, 0)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 1)

    def _create_monitoring_tab(self):
        tab = QWidget()
        self.display_tabs.addTab(tab, "Monitoring")
        layout = QGridLayout(tab)

        self.img_on_plot = pg.PlotWidget(title="Pump ON")
        self.img_on_item = pg.ImageItem()
        self.img_on_plot.addItem(self.img_on_item)
        self.text_on = pg.TextItem('N=0', color='red', anchor=(0.5, 0))
        self.img_on_plot.addItem(self.text_on)
        self.roi_rect_on = pg.PlotDataItem(pen=pg.mkPen('y', width=1))
        self.img_on_plot.addItem(self.roi_rect_on)
        layout.addWidget(self.img_on_plot, 0, 0)

        self.img_off_plot = pg.PlotWidget(title="Pump OFF")
        self.img_off_item = pg.ImageItem()
        self.img_off_plot.addItem(self.img_off_item)
        self.text_off = pg.TextItem('N=0', color='blue', anchor=(0.5, 0))
        self.img_off_plot.addItem(self.text_off)
        self.roi_rect_off = pg.PlotDataItem(pen=pg.mkPen('y', width=1))
        self.img_off_plot.addItem(self.roi_rect_off)
        layout.addWidget(self.img_off_plot, 0, 1)

        self.intensity_plot = pg.PlotWidget(title="Ref Intensity")
        self.intensity_plot.setLabel('bottom', 'Frame')
        self.intensity_plot.showGrid(x=True, y=True, alpha=0.3)
        self._scatter_on = pg.ScatterPlotItem(pen=None, brush='r', size=5)
        self._scatter_off = pg.ScatterPlotItem(pen=None, brush='b', size=5)
        self._intensity_trace = self.intensity_plot.plot(pen=pg.mkPen('k', width=0.5))
        self._threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=Qt.DashLine))
        self.intensity_plot.addItem(self._scatter_on)
        self.intensity_plot.addItem(self._scatter_off)
        self.intensity_plot.addItem(self._threshold_line)
        layout.addWidget(self.intensity_plot, 1, 0, 1, 2)

        zbar = QWidget()
        zh = QHBoxLayout(zbar)
        zh.setContentsMargins(4, 2, 4, 2)
        zh.addWidget(self._label("Image Z Range:", bold=True, size=9))
        zh.addWidget(self._label("Min:", size=9))
        self.monitor_z_min_entry = self._entry("3000", 70)
        zh.addWidget(self.monitor_z_min_entry)
        zh.addWidget(self._label("Max:", size=9))
        self.monitor_z_max_entry = self._entry("8000", 70)
        zh.addWidget(self.monitor_z_max_entry)
        btn_apply_mz = QPushButton("Apply")
        btn_apply_mz.setFixedWidth(55)
        btn_apply_mz.setStyleSheet("background-color: lightgreen;")
        btn_apply_mz.clicked.connect(self._apply_monitor_z_range)
        zh.addWidget(btn_apply_mz)
        btn_auto_mz = QPushButton("Auto")
        btn_auto_mz.setFixedWidth(50)
        btn_auto_mz.setStyleSheet("background-color: lightblue;")
        btn_auto_mz.clicked.connect(self._auto_monitor_z_range)
        zh.addWidget(btn_auto_mz)
        zh.addStretch()
        layout.addWidget(zbar, 2, 0, 1, 2)

    def _create_gaussian_tab(self):
        tab = QWidget()
        self.display_tabs.addTab(tab, "Gaussian Fit")
        layout = QHBoxLayout(tab)

        ctrl = QWidget()
        ctrl.setFixedWidth(220)
        cl = QVBoxLayout(ctrl)
        cl.addWidget(self._label("Gaussian Fit Controls", bold=True, size=11))
        self.enable_gaussian_chk = QCheckBox("Enable Gaussian Fit")
        self.enable_gaussian_chk.setChecked(True)
        self.enable_gaussian_chk.setStyleSheet("font-weight: bold; font-size: 10pt;")
        self.enable_gaussian_chk.stateChanged.connect(self._toggle_gaussian_fit)
        cl.addWidget(self.enable_gaussian_chk)
        cl.addWidget(self._separator())

        cl.addWidget(self._label("Pixel Size (um/pix):", bold=True, size=10))
        ps_row = QHBoxLayout()
        self.pixel_size_entry = self._entry("0.07", 80)
        ps_row.addWidget(self.pixel_size_entry)
        btn_apply_ps = QPushButton("Apply")
        btn_apply_ps.setStyleSheet("background-color: lightgreen;")
        btn_apply_ps.clicked.connect(self._apply_pixel_size)
        ps_row.addWidget(btn_apply_ps)
        ps_w = QWidget(); ps_w.setLayout(ps_row); cl.addWidget(ps_w)
        cl.addWidget(self._separator())

        self.use_auto_fit_chk = QCheckBox("Auto Initial Guess")
        self.use_auto_fit_chk.setChecked(True)
        cl.addWidget(self.use_auto_fit_chk)
        cl.addWidget(self._label("Manual Initial Parameters:", bold=True, size=10))
        cl.addWidget(self._label("Amplitude:"))
        self.fit_amplitude_entry = self._entry("1.0", 100); cl.addWidget(self.fit_amplitude_entry)
        cl.addWidget(self._label("Center (um):"))
        self.fit_center_entry = self._entry("64", 100); cl.addWidget(self.fit_center_entry)
        cl.addWidget(self._label("Width sigma (um):"))
        self.fit_width_entry = self._entry("10", 100); cl.addWidget(self.fit_width_entry)
        cl.addWidget(self._label("Offset:"))
        self.fit_offset_entry = self._entry("0.0", 100); cl.addWidget(self.fit_offset_entry)
        cl.addWidget(self._btn("Update Fit", self.force_update_fit, color="lightgreen"))
        cl.addStretch()
        layout.addWidget(ctrl)

        plots = QWidget()
        pl = QVBoxLayout(plots)
        self.fit_plot = pg.PlotWidget(title="Real-Time Gaussian Fit")
        self.fit_plot.setLabel('bottom', 'Y Position (um)')
        self.fit_plot.setLabel('left', 'log(Pump ON / Pump OFF)')
        self.fit_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fit_plot.addLegend()
        self._fit_data_line = self.fit_plot.plot(pen=None, symbol='o', symbolBrush='b',
                                                  symbolSize=4, name='Data')
        self._fit_curve_line = self.fit_plot.plot(pen=pg.mkPen('r', width=3), name='Gaussian Fit')
        self._fit_center_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('g', style=Qt.DashLine, width=2))
        self._fit_center_line.setVisible(False)
        self.fit_plot.addItem(self._fit_center_line)
        self._fit_fwhm_region = pg.LinearRegionItem(values=(0, 1), brush=(255, 255, 0, 50),
                                                     movable=False)
        self._fit_fwhm_region.setVisible(False)
        self.fit_plot.addItem(self._fit_fwhm_region)
        pl.addWidget(self.fit_plot, stretch=3)

        self.width_history_plot = pg.PlotWidget(title="Width History")
        self.width_history_plot.setLabel('bottom', 'Update Number')
        self.width_history_plot.setLabel('left', 'Width sigma (um)')
        self.width_history_plot.showGrid(x=True, y=True, alpha=0.3)
        self._width_line = self.width_history_plot.plot(pen=pg.mkPen('b', width=2), symbol='o',
                                                         symbolSize=5)
        self.width_text = pg.TextItem('', color='k', anchor=(0, 0))
        self.width_history_plot.addItem(self.width_text)
        pl.addWidget(self.width_history_plot, stretch=1)
        layout.addWidget(plots)

    # =========================================================
    # Crosshair interaction
    # =========================================================

    def _on_crosshair_moved(self, line):
        if self.updating_from_drag or self.log_ratio_data is None:
            return
        x = self.v_crosshair.value()
        y = self.h_crosshair.value()
        self.update_profiles(x, y)

    def _on_main_click(self, event):
        if self.log_ratio_data is None:
            return
        pos = event.scenePos()
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        h, w = self.log_ratio_data.shape
        if 0 <= x < w and 0 <= y < h:
            self.v_crosshair.setValue(x)
            self.h_crosshair.setValue(y)
            self.update_profiles(x, y)

    # =========================================================
    # Hardware control
    # =========================================================

    def init_camera(self):
        try:
            self.camera.initialize()
            self.camera_status.setText("● Connected")
            self.camera_status.setStyleSheet("color: green; font-weight: bold;")
            self.update_button_states()
            QMessageBox.information(self, "Success", "Camera initialized!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize camera:\n{e}")

    def init_stage(self):
        if not self.stage_available:
            QMessageBox.critical(self, "Error", "Stage control not available!")
            return
        try:
            serial_number = int(self.stage_serial_entry.text())
            self.stage.initialize(serial_number=serial_number)
            self.stage_status.setText("● Connected")
            self.stage_status.setStyleSheet("color: green; font-weight: bold;")
            self.update_current_position()
            self.update_button_states()
            # Show axis range so user knows valid positions
            try:
                axis_info = self.stage.motor.get_stage_axis_info()
                range_str = f"\nAxis range: {axis_info}"
            except Exception:
                range_str = ""
            msg = (f"Stage initialized!\nSerial Number: {serial_number}{range_str}\n\n"
                   f"NOTE: The stage must be homed before any absolute moves will work.\n"
                   f"Click 'Home Stage' now if this is the first use since power-on.")
            reply = QMessageBox.question(self, "Stage Initialized",
                msg + "\n\nHome stage now?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.home_stage()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid serial number!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize stage:\n{e}")

    def update_current_position(self):
        if self.stage and self.stage.is_initialized:
            try:
                pos_mm = self.stage.get_position()
                relative_mm = pos_mm - self.time_zero_position_mm
                relative_ps = self.stage.mm_to_ps(relative_mm)
                self.current_pos_label.setText(
                    f"Current: {pos_mm:.4f} mm\n({relative_ps:+.2f} ps from t=0)")
            except Exception:
                self.current_pos_label.setText("Current: Error")

    def move_stage(self):
        try:
            time_delay_ps = float(self.pos_entry.text())
            relative_mm = self.stage.ps_to_mm(time_delay_ps)
            absolute_mm = self.time_zero_position_mm + relative_mm
            self.stage.move_to(absolute_mm, wait=True)
            self.update_current_position()
            QMessageBox.information(self, "Success",
                f"Moved to time delay: {time_delay_ps:+.2f} ps\n"
                f"Absolute position: {absolute_mm:.4f} mm")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid time delay value!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed:\n{e}")

    def set_time_zero_from_stage(self):
        if not self.stage or not self.stage.is_initialized:
            QMessageBox.critical(self, "Error", "Stage not initialized!")
            return
        try:
            current_pos = self.stage.get_position()
            self.time_zero_position_mm = current_pos
            self.time_zero_entry.setText(f"{current_pos:.4f}")
            self.time_zero_label.setText(f"t=0 at: {current_pos:.4f} mm")
            self.update_current_position()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read stage:\n{e}")

    def apply_time_zero_entry(self):
        try:
            val = float(self.time_zero_entry.text())
            self.time_zero_position_mm = val
            self.time_zero_label.setText(f"t=0 at: {val:.4f} mm")
            self.update_current_position()
        except ValueError:
            QMessageBox.critical(self, "Error", "Time zero must be a number (mm).")

    def home_stage(self):
        if not self.stage or not self.stage.is_initialized:
            QMessageBox.critical(self, "Error", "Stage not initialized!")
            return
        try:
            self.stage_status.setText("○ Homing...")
            self.stage_status.setStyleSheet("color: orange; font-weight: bold;")
            QApplication.processEvents()
            self.stage.home(wait=True)
            self.stage_status.setText("● Connected (homed)")
            self.stage_status.setStyleSheet("color: green; font-weight: bold;")
            self.update_current_position()
            QMessageBox.information(self, "Success",
                f"Stage homed!\nCurrent position: {self.stage.get_position():.4f} mm")
        except Exception as e:
            self.stage_status.setText("● Connected (home failed)")
            self.stage_status.setStyleSheet("color: orange; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Homing failed:\n{e}")

    def close_all(self):
        self.camera.close()
        if self.stage:
            self.stage.close()
        if self.rotation_stage:
            self.rotation_stage.close()
        if self.front_widget and self.front_widget.dev:
            self.front_widget._on_disconnect()
        if self.back_widget and self.back_widget.dev:
            self.back_widget._on_disconnect()
        self.camera_status.setText("○ Not Connected")
        self.camera_status.setStyleSheet("color: red; font-weight: bold;")
        self.stage_status.setText("○ Not Connected")
        self.stage_status.setStyleSheet("color: red; font-weight: bold;")
        self.rot_status.setText("○ Not Connected")
        self.rot_status.setStyleSheet("color: red; font-weight: bold;")
        self.rot_angle_label.setText("Angle: -- deg")
        self.update_button_states()
        QMessageBox.information(self, "Info", "All hardware closed")

    def _default_save_name(self):
        return time.strftime("ta_%Y%m%d_%H%M%S")

    def _resolve_save_dir(self) -> str:
        """Build and create DATA_DIR/<subfolder>, cache as self.save_directory."""
        name = self.save_name_edit.text().strip() if hasattr(self, 'save_name_edit') else ""
        if not name:
            name = self._default_save_name()
            if hasattr(self, 'save_name_edit'):
                self.save_name_edit.setText(name)
        path = os.path.join(DATA_DIR, name)
        os.makedirs(path, exist_ok=True)
        self.save_directory = path
        return path

    # =========================================================
    # Scan setup
    # =========================================================

    def _add_range_default(self):
        self.add_range_entry(0, 10, 1, 1)

    def add_range_entry(self, start=0, end=10, step=1, avg=1):
        frame = QWidget()
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(0, 0, 0, 0)
        num = len(self.range_entries) + 1
        hl.addWidget(self._label(f"R{num}:", bold=True, size=8))
        hl.addWidget(self._label("S:", size=7))
        e_start = self._entry(str(start), 45); hl.addWidget(e_start)
        hl.addWidget(self._label("E:", size=7))
        e_end = self._entry(str(end), 45); hl.addWidget(e_end)
        hl.addWidget(self._label("St:", size=7))
        e_step = self._entry(str(step), 35); hl.addWidget(e_step)
        hl.addWidget(self._label("Av:", size=7))
        e_avg = self._entry(str(avg), 30); hl.addWidget(e_avg)
        self.ranges_container.addWidget(frame)
        self.range_entries.append({
            'frame': frame, 'start': e_start, 'end': e_end, 'step': e_step, 'avg': e_avg
        })

    def remove_range_entry(self):
        if len(self.range_entries) > 1:
            entry = self.range_entries.pop()
            entry['frame'].setParent(None)
            entry['frame'].deleteLater()

    def add_t0_point(self):
        t0_value, ok = QInputDialog.getDouble(self, "Add Time Point", "Enter time delay (ps):",
                                               0.0, -10000, 10000, 2)
        if not ok:
            return
        avg_times, ok2 = QInputDialog.getInt(self, "Averaging",
                                              "How many times to average at this point?",
                                              5, 1, 100)
        if not ok2:
            avg_times = 5
        self.add_range_entry(start=t0_value, end=t0_value, step=1, avg=avg_times)

    def update_scan_mode(self):
        if self.radio_simple.isChecked():
            self.simple_scan_frame.show()
            self.multi_scan_frame.hide()
        else:
            self.simple_scan_frame.hide()
            self.multi_scan_frame.show()

    def generate_multi_range_scan(self):
        try:
            all_positions = []
            range_info = []
            self.delay_to_avg_map = {}
            t0_offset = self.get_delay_t0_offset()
            for i, re in enumerate(self.range_entries):
                s = float(re['start'].text())
                e = float(re['end'].text())
                st = float(re['step'].text())
                av = int(re['avg'].text())
                if st <= 0: raise ValueError(f"Range {i+1}: Step must be positive!")
                n_pts = int(abs(e - s) / st) + 1
                if s <= e:
                    positions = [s + j * st for j in range(n_pts) if s + j * st <= e]
                else:
                    positions = [s - j * st for j in range(n_pts) if s - j * st >= e]
                positions_with_offset = [p + t0_offset for p in positions]
                if all_positions and positions_with_offset:
                    if abs(positions_with_offset[0] - all_positions[-1]) < 0.001:
                        positions_with_offset = positions_with_offset[1:]
                        positions = positions[1:]
                for pos in positions_with_offset:
                    self.delay_to_avg_map[round(pos, 2)] = av
                all_positions.extend(positions_with_offset)
                range_info.append(f"  R{i+1}: {s:+.2f} to {e:+.2f}, step={st}, avg={av} -> {len(positions_with_offset)} pts")
            if not all_positions:
                raise ValueError("No valid ranges!")
            positions_str = ", ".join([f"{p:.2f}" for p in all_positions])
            self.delay_positions.setPlainText(positions_str)
            self.multi_preview_label.setText(
                f"Total: {len(all_positions)} points | {all_positions[0]:.2f} to {all_positions[-1]:.2f} ps")
            QMessageBox.information(self, "Generated", f"Generated {len(all_positions)} positions\n\n" +
                                    "\n".join(range_info))
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))

    def scan_around_zero(self):
        range_ps, ok = QInputDialog.getDouble(self, "Scan Around t=0",
                                               "Enter total range (ps):", 20.0, 0.1, 100000, 2)
        if not ok: return
        increment, ok2 = QInputDialog.getDouble(self, "Scan Around t=0",
                                                  "Enter step size (ps):", 2.0, 0.01, 10000, 2)
        if not ok2: return
        n_steps = int(range_ps / increment) + 1
        start = -(range_ps / 2)
        self.scan_start.setText(f"{start:.2f}")
        self.scan_steps.setText(str(n_steps))
        self.scan_increment.setText(f"{increment:.2f}")

    def scan_from_zero(self):
        end_ps, ok = QInputDialog.getDouble(self, "Scan From t=0",
                                             "Enter end time (ps):", 50.0, -100000, 100000, 2)
        if not ok: return
        increment, ok2 = QInputDialog.getDouble(self, "Scan From t=0",
                                                  "Enter step size (ps):", 5.0, 0.01, 10000, 2)
        if not ok2: return
        n_steps = int(abs(end_ps) / increment) + 1
        start = 0.0 if end_ps >= 0 else end_ps
        self.scan_start.setText(f"{start:.2f}")
        self.scan_steps.setText(str(n_steps))
        self.scan_increment.setText(f"{increment:.2f}")

    def get_delay_t0_offset(self, show_error=True):
        text = self.delay_t0_entry.text().strip()
        if text == "":
            text = "0.0"
            self.delay_t0_entry.setText(text)
        try:
            value = float(text)
            self.delay_t0_offset = value
            return value
        except ValueError:
            if show_error:
                QMessageBox.critical(self, "Error", "t0 offset must be a number (ps).")
            self.delay_t0_entry.setText(f"{self.delay_t0_offset:.2f}")
            return self.delay_t0_offset

    def update_scan_preview(self):
        try:
            start_ps = float(self.scan_start.text())
            n_steps = int(self.scan_steps.text())
            increment_ps = float(self.scan_increment.text())
            if n_steps > 0:
                end_ps = start_ps + (n_steps - 1) * increment_ps
                range_ps = abs(end_ps - start_ps)
                t0_offset = self.get_delay_t0_offset(show_error=False)
                abs_end = end_ps + t0_offset
                self.scan_preview_label.setText(
                    f"End: {end_ps:.2f} ps (abs {abs_end:.2f}) | Range: {range_ps:.2f} ps")
        except ValueError:
            self.scan_preview_label.setText("Enter valid numbers")

    def generate_scan_positions(self):
        try:
            start_ps = float(self.scan_start.text())
            n_steps = int(self.scan_steps.text())
            increment_ps = float(self.scan_increment.text())
            t0_offset = self.get_delay_t0_offset()
            if n_steps <= 0: raise ValueError("Number of steps must be positive!")
            positions = [start_ps + i * increment_ps for i in range(n_steps)]
            positions_with_offset = [p + t0_offset for p in positions]
            positions_str = ", ".join([f"{p:.2f}" for p in positions_with_offset])
            self.delay_positions.setPlainText(positions_str)
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))

    # =========================================================
    # Gate sweep helpers
    # =========================================================

    def _update_gate_input_mode(self):
        use_range = self.gate_range_radio.isChecked()
        self.gate_range_frame.setVisible(use_range)
        self.gate_manual_frame.setVisible(not use_range)
        self._update_gate_range_preview()

    def _update_gate_range_preview(self):
        try:
            front_vs = self._get_front_voltages()
            dual = self.gate_dual_radio.isChecked()
            if dual:
                ratio = self.gate_ratio_spin.value()
                back_vs = [vf * ratio for vf in front_vs]
                self.gate_back_preview.setText(
                    "Back: " + ", ".join(f"{v:.3f}" for v in back_vs))
            else:
                self.gate_back_preview.setText("Back: (single mode)")
            if self.gate_range_radio.isChecked() and front_vs:
                self.gate_range_preview.setText(f"End: {front_vs[-1]:.4f} V  ({len(front_vs)} pts)")
        except Exception:
            pass

    def _get_front_voltages(self):
        if self.gate_manual_radio.isChecked():
            text = self.gate_manual_edit.text()
            return [float(x.strip()) for x in text.split(',') if x.strip()]
        v0 = float(self.gate_v0_entry.text())
        step = float(self.gate_step_entry.text())
        n = int(self.gate_n_entry.text())
        return [v0 + k * step for k in range(n)]

    def _compute_gate_pairs(self):
        front_vs = self._get_front_voltages()
        if self.gate_dual_radio.isChecked():
            ratio = self.gate_ratio_spin.value()
            back_vs = [vf * ratio for vf in front_vs]
        else:
            back_vs = [0.0] * len(front_vs)
        return list(zip(front_vs, back_vs))

    def _poll_gates(self):
        if not KEITHLEY_AVAILABLE:
            return
        allow = not self.is_gate_sweeping
        if self.front_widget and self.front_widget.dev:
            v, iA = self.front_widget.poll_once(allow)
            if v is not None and hasattr(self, 'gate_vi_front'):
                self.gate_vi_front.add_point(v, iA)
        if self.back_widget and self.back_widget.dev:
            v, iA = self.back_widget.poll_once(allow)
            if v is not None and hasattr(self, 'gate_vi_back'):
                self.gate_vi_back.add_point(v, iA)

    def start_gate_sweep(self):
        try:
            self._gate_pairs = self._compute_gate_pairs()
            if not self._gate_pairs:
                QMessageBox.critical(self, "Error", "No gate voltages specified!")
                return
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid gate voltages: {e}")
            return

        front_dev = self.front_widget.dev if self.front_widget else None
        if not front_dev:
            QMessageBox.critical(self, "Error", "Front gate not connected!\nConnect on the Gate HW tab.")
            return

        dual = self.gate_dual_radio.isChecked()
        back_dev = self.back_widget.dev if (self.back_widget and dual) else None
        if dual and not back_dev:
            reply = QMessageBox.question(self, "Back gate not connected",
                "Dual mode selected but back gate not connected.\nContinue with front gate only?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            dual = False

        if not self.stage or not self.stage.is_initialized:
            QMessageBox.critical(self, "Error",
                "Delay stage not connected!\nConnect on the Hardware tab first.")
            return

        try:
            pos_text = self.delay_positions.toPlainText().strip()
            time_delays_ps = [float(p.strip()) for p in pos_text.split(',') if p.strip()]
            if not time_delays_ps:
                raise ValueError("No time delays in Scan Setup tab!")
            positions_mm = [self.time_zero_position_mm + self.stage.ps_to_mm(d)
                            for d in time_delays_ps]
            n_frames = int(self.n_frames_entry.text())
            default_avg = int(self.avg_times_entry.text())
            roi = (int(self.roi_x.text()), int(self.roi_y.text()),
                   int(self.roi_w.text()), int(self.roi_h.text()))
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Warn if time zero position looks unset (still 0.0)
        if self.time_zero_position_mm == 0.0:
            reply = QMessageBox.warning(self, "Time Zero Not Set",
                "Time zero position is 0.0 mm — it has probably not been set.\n\n"
                f"Computed stage positions will range from "
                f"{min(positions_mm):.4f} to {max(positions_mm):.4f} mm, "
                f"which may be outside the stage's travel range.\n\n"
                "To set time zero: Hardware tab → Stage section → move stage to t=0 "
                "and click 'Set from Current Pos'.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        power_enabled = (self.power_enable_chk.isChecked() and
                         self.rotation_stage is not None and
                         self.rotation_stage.is_initialized)
        angles = []
        if power_enabled:
            try:
                pos_text2 = self.pwr_positions_text.toPlainText().strip()
                angles = [float(a.strip()) for a in pos_text2.split(',') if a.strip()]
            except ValueError:
                pass

        pol_enabled = (self.pol_enable_chk.isChecked() and
                       self.pol_stage is not None and
                       self.pol_stage.is_initialized)
        lh_angle = 0.0
        pol_settle_s = 0.5
        if pol_enabled:
            try:
                lh_angle = float(self.pol_lh_entry.text())
                pol_settle_s = float(self.pol_settle_entry.text())
            except ValueError:
                pass

        gate_settle_s = self.gate_settle_spin.value() / 1000.0
        gate_outer = self.gate_outer_radio.isChecked()
        power_settle_s = float(self.pwr_settle_entry.text()) if power_enabled else 0.5
        save_base = self.save_filename_entry.text().strip() or "gate_scan"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._resolve_save_dir()
        crop = None
        if self.save_region_enable.isChecked():
            try:
                crop = (int(self.save_region_x.text()), int(self.save_region_w.text()))
            except ValueError:
                pass
        t0_ps = self.stage.mm_to_ps(self.time_zero_position_mm) if self.stage else 0.0
        pixel_size = float(self.pixel_size_entry.text()) if hasattr(self, 'pixel_size_entry') else 0.07
        crop_h = self.camera.height if self.camera and self.camera.is_initialized else 128

        vf_list = [p[0] for p in self._gate_pairs]
        vb_list = [p[1] for p in self._gate_pairs] if dual else None
        n_gate = len(self._gate_pairs)
        n_delays = len(time_delays_ps)
        n_angles = len(angles) if power_enabled else 1
        n_pol = 2 if pol_enabled else 1
        total_scans = n_gate * n_angles * n_pol
        total_captures = total_scans * n_delays * default_avg
        msg = (f"Start gate-dependent delay scan?\n\n"
               f"Gate (Vf):  {', '.join(f'{v:.3f}' for v in vf_list)} V\n")
        if dual and vb_list:
            msg += f"Gate (Vb):  {', '.join(f'{v:.3f}' for v in vb_list)} V\n"
        msg += (f"Gate steps: {n_gate}\n"
                f"Delays/gate: {n_delays}  ({time_delays_ps[0]:.1f} to {time_delays_ps[-1]:.1f} ps)\n"
                f"Stage positions: {min(positions_mm):.4f} – {max(positions_mm):.4f} mm  "
                f"(t=0 at {self.time_zero_position_mm:.4f} mm)\n"
                f"Frames/capture: {n_frames}  |  Avg: {default_avg}\n")
        if power_enabled:
            msg += (f"Power scan: {len(angles)} angles "
                    f"({angles[0]:.1f}° – {angles[-1]:.1f}°), "
                    f"{'gate outer' if gate_outer else 'power outer'}\n")
        if pol_enabled:
            msg += (f"Polarization: LH={lh_angle:.1f}° / RH={lh_angle+45:.1f}°  "
                    f"(2× per step)\n")
        msg += (f"\nTotal delay scans:  {total_scans}\n"
                f"Total captures:     {total_captures}")
        reply = QMessageBox.question(self, "Confirm", msg, QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.is_gate_sweeping = True
        self.btn_gate_scan.setEnabled(False)
        self.btn_gate_stop.setEnabled(True)
        self.btn_single.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_power_scan.setEnabled(False)

        self._gate_thread = _GateSweepThread(
            self.camera, self.stage, front_dev, back_dev,
            self._gate_pairs, dual, gate_settle_s,
            time_delays_ps, positions_mm, n_frames, roi,
            dict(self.delay_to_avg_map), default_avg,
            power_enabled=power_enabled,
            rotation_stage=self.rotation_stage,
            angles=angles,
            power_settle_s=power_settle_s,
            gate_outer=gate_outer,
            save_dir=self.save_directory,
            save_base=save_base,
            timestamp=timestamp,
            crop=crop, t0_ps=t0_ps,
            pixel_size=pixel_size, crop_h=crop_h,
            pol_enabled=pol_enabled,
            pol_stage=self.pol_stage if pol_enabled else None,
            lh_angle=lh_angle,
            pol_settle_s=pol_settle_s)
        self._gate_thread.progress.connect(self.update_status_text)
        self._gate_thread.step_result.connect(self._on_gate_step_result)
        self._gate_thread.gate_step_done.connect(self._on_gate_step_done)
        self._gate_thread.gate_vi_update.connect(self._on_gate_vi_update)
        self._gate_thread.finished.connect(self._on_gate_finished)
        self._gate_thread.error_occurred.connect(self._on_gate_error)
        self._gate_thread.start()

    def stop_gate_sweep(self):
        self.is_gate_sweeping = False
        if self._gate_thread is not None and self._gate_thread.isRunning():
            self._gate_thread.stop()

    def _on_gate_step_result(self, final_results, roi, step_info):
        self.current_results = final_results
        self.display_results(final_results, *roi, quick_mode=False)

    def _on_gate_step_done(self, vf, vb, idx):
        self.update_status_text(f"Gate step {idx+1} done: Vf={vf:+.3f} V")

    def _on_gate_finished(self, completed):
        self.is_gate_sweeping = False
        self.update_button_states()
        self.btn_gate_stop.setEnabled(False)
        if completed:
            self.update_status_text("Gate sweep complete!")
            QMessageBox.information(self, "Complete", "Gate sweep finished!\nData saved to:\n" +
                                    self.save_directory)
        else:
            self.update_status_text("Gate sweep stopped.")

    def _on_gate_error(self, msg):
        self.is_gate_sweeping = False
        self.update_button_states()
        self.btn_gate_stop.setEnabled(False)
        QMessageBox.critical(self, "Error", f"Gate sweep failed:\n{msg}")

    def _on_gate_vi_update(self, which, v, iA):
        """Feed live gate V/I from the sweep thread into the Gate V/I plot tab."""
        if which == "front":
            if hasattr(self, 'gate_vi_front'):
                self.gate_vi_front.add_point(v, iA)
            if self.front_widget:
                self.front_widget.vmeas_lbl.setText(f"{v:.4g}")
                self.front_widget.imeas_lbl.setText(f"{iA*1e9:.4g}")
        else:
            if hasattr(self, 'gate_vi_back'):
                self.gate_vi_back.add_point(v, iA)
            if self.back_widget:
                self.back_widget.vmeas_lbl.setText(f"{v:.4g}")
                self.back_widget.imeas_lbl.setText(f"{iA*1e9:.4g}")

    # =========================================================
    # Button states
    # =========================================================

    def update_button_states(self):
        camera_ready = self.camera.is_initialized
        stage_ready = self.stage and self.stage.is_initialized
        rotation_ready = self.rotation_stage and self.rotation_stage.is_initialized
        any_busy = self.is_scanning or self.is_power_scanning or self.is_gate_sweeping
        self.btn_single.setEnabled(camera_ready and not any_busy)
        self.btn_scan.setEnabled(camera_ready and stage_ready and not any_busy)
        self.btn_power_scan.setEnabled(camera_ready and stage_ready and rotation_ready and not any_busy)
        gate_hw_ok = (KEITHLEY_AVAILABLE and self.front_widget is not None
                      and self.front_widget.dev is not None)
        if hasattr(self, 'btn_gate_scan'):
            self.btn_gate_scan.setEnabled(camera_ready and stage_ready and gate_hw_ok and not any_busy)

    # =========================================================
    # Capture
    # =========================================================

    def single_capture(self):
        if self.is_single_capturing:
            return
        try:
            n_frames = int(self.n_frames_entry.text())
            roi_x = int(self.roi_x.text())
            roi_y = int(self.roi_y.text())
            roi_w = int(self.roi_w.text())
            roi_h = int(self.roi_h.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Check frame count and ROI values.")
            return
        self.is_single_capturing = True
        self._capture_count = 0
        self._display_cycle_count = 0
        self._last_display_time = 0
        self._capture_n_frames = n_frames
        self._capture_roi = (roi_x, roi_y, roi_w, roi_h)
        self.btn_single.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._capture_thread = _CaptureThread(self.camera, n_frames,
                                              (roi_x, roi_y, roi_w, roi_h))
        self._capture_thread.result_ready.connect(self._on_capture_result)
        self._capture_thread.error_occurred.connect(self._on_capture_error)
        self._capture_thread.finished.connect(self._on_capture_stopped)
        self._capture_thread.start()

    def _on_capture_result(self, results, frames):
        if not self.is_single_capturing:
            return
        self._capture_count += 1
        self.captured_frames = frames
        self.current_results = results
        roi_x, roi_y, roi_w, roi_h = self._capture_roi
        now = time.time() * 1000
        if (now - self._last_display_time) >= self.live_refresh_interval_ms:
            self._last_display_time = now
            self._display_cycle_count += 1
            self.update_status_text(
                f"Continuous capture #{self._capture_count} | Press Stop to end")
            self.display_results(results, roi_x, roi_y, roi_w, roi_h, quick_mode=False)
        self.btn_save.setEnabled(True)

    def _on_capture_error(self, msg):
        self.is_single_capturing = False
        self._on_capture_stopped()
        QMessageBox.critical(self, "Error", f"Capture failed:\n{msg}")

    def _on_capture_stopped(self):
        self.update_button_states()
        self.btn_stop.setEnabled(False)

    # =========================================================
    # Delay scan
    # =========================================================

    def start_scan(self):
        try:
            if not self.stage or not self.stage.is_initialized:
                QMessageBox.critical(self, "Error",
                    "Delay stage not connected!\nConnect on the Hardware tab first.")
                return
            pos_text = self.delay_positions.toPlainText().strip()
            time_delays_ps = [float(p.strip()) for p in pos_text.split(',') if p.strip()]
            if not time_delays_ps: raise ValueError("No time delays specified!")
            positions_mm = []
            for d in time_delays_ps:
                positions_mm.append(self.time_zero_position_mm + self.stage.ps_to_mm(d))
            n_frames = int(self.n_frames_entry.text())
            default_avg = int(self.avg_times_entry.text())
            roi = (int(self.roi_x.text()), int(self.roi_y.text()),
                   int(self.roi_w.text()), int(self.roi_h.text()))
            if self.time_zero_position_mm == 0.0:
                r = QMessageBox.warning(self, "Time Zero Not Set",
                    f"Time zero is 0.0 mm — may not be set.\n"
                    f"Stage will move to {min(positions_mm):.4f} – {max(positions_mm):.4f} mm.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No)
                if r != QMessageBox.Yes: return
            msg = (f"Start scan with {len(time_delays_ps)} time delays?\n\n"
                   f"Time zero at: {self.time_zero_position_mm:.4f} mm\n"
                   f"Stage positions: {min(positions_mm):.4f} – {max(positions_mm):.4f} mm\n"
                   f"Estimated time: ~{len(time_delays_ps) * 2} minutes")
            reply = QMessageBox.question(self, "Confirm Scan", msg,
                                         QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes: return
            self.is_scanning = True
            self.btn_scan.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_single.setEnabled(False)
            self.scan_results = []
            self._scan_thread = _DelayScanThread(
                self.camera, self.stage, time_delays_ps, positions_mm,
                n_frames, roi, dict(self.delay_to_avg_map), default_avg)
            self._scan_thread.progress.connect(self.update_status_text)
            self._scan_thread.step_result.connect(self._on_scan_step)
            self._scan_thread.finished.connect(self._on_scan_finished)
            self._scan_thread.error_occurred.connect(self._on_scan_error)
            self._scan_thread.start()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Scan failed:\n{e}")

    def _on_scan_step(self, final_results, roi, step_info):
        self.scan_results.append(step_info)
        self.current_results = final_results
        self.display_results(final_results, *roi, quick_mode=False)

    def _on_scan_finished(self, scan_results, completed):
        self.is_scanning = False
        self.scan_results = scan_results
        self.update_button_states()
        self.btn_stop.setEnabled(False)
        if completed:
            self.update_status_text(
                f"Scan complete! {len(scan_results)} positions acquired.")
            self.btn_save.setEnabled(True)
            if self.auto_save_chk.isChecked():
                self.auto_save_scan_data()
            QMessageBox.information(self, "Complete",
                                    f"Scan finished! {len(scan_results)} positions.")
        else:
            self.update_status_text(
                f"Scan stopped. {len(scan_results)} positions acquired.")

    def _on_scan_error(self, msg):
        self.is_scanning = False
        self.update_button_states()
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Error", f"Scan failed:\n{msg}")

    def _get_nd_angle_tag(self):
        """Return ND angle string for filenames, e.g. '_ND45p0deg', or '' if not active."""
        if self.rotation_stage and self.rotation_stage.is_initialized:
            try:
                angle = self.rotation_stage.get_angle()
                astr = f"{angle:.1f}".replace('.', 'p').replace('-', 'n')
                return f"_ND{astr}deg"
            except Exception:
                pass
        return ""

    def auto_save_scan_data(self):
        """Save in TAM-code-new / LabVIEW compatible format (*sum.txt + *para.txt)."""
        if not self.scan_results: return
        try:
            self._resolve_save_dir()
            base_filename = self.save_filename_entry.text().strip() or "pump_probe_scan"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nd_tag = self._get_nd_angle_tag()
            prefix = f"{base_filename}{nd_tag}_{timestamp}"
            time_delays_ps = np.array([r['time_delay_ps'] for r in self.scan_results])
            avg_times_list = [r.get('avg_times_used', 1) for r in self.scan_results]
            n_frames = int(self.n_frames_entry.text())
            pixel_size = float(self.pixel_size_entry.text()) if hasattr(self, 'pixel_size_entry') else 0.07
            crop_w = int(self.crop_w_entry.text()) if hasattr(self, 'crop_w_entry') else 480
            crop_h = self.camera.height if self.camera and self.camera.is_initialized else 128
            t0_ps = self.stage.mm_to_ps(self.time_zero_position_mm) if self.stage else 0.0
            nd_angle = 0.0
            if self.rotation_stage and self.rotation_stage.is_initialized:
                try: nd_angle = self.rotation_stage.get_angle()
                except Exception: pass

            log_ratios_list = []
            for r in self.scan_results:
                res = r['results']
                on = self._crop_image(res['avg_pump_on'])
                off = self._crop_image(res['avg_pump_off'])
                if on is not None and off is not None:
                    lr = self.calculate_log_ratio(on, off)
                    log_ratios_list.append(lr.T)

            para_file = os.path.join(self.save_directory, f"{prefix}para.txt")
            with open(para_file, 'w') as f:
                f.write('\t'.join([f'{d:.6f}' for d in time_delays_ps]) + '\n')
                f.write('\t'.join([f'{a:.6f}' for a in avg_times_list]) + '\n')
                f.write('0\n')
                f.write(f' {1.0:.6f} \n')
                f.write(f' {pixel_size:.6f} \n')
                f.write(f' {crop_w} \n')
                f.write(f' {pixel_size:.6f} \n')
                f.write(f' {crop_h} \n')
                f.write(f' {0.0:.6f} \n')
                f.write(f' {t0_ps:.6f}\n')
                f.write('0\n0\n0\n0\n0 \n')
                f.write(f' {0.0:.6f}\n')
                f.write(f' {nd_angle:.6f}\n')
                f.write(f'{1.000000}\n')
                f.write(f'{0}\n')
                f.write(f' {n_frames} \n')
                f.write(f' 1\n')

            sum_file = os.path.join(self.save_directory, f"{prefix}sum.txt")
            with open(sum_file, 'w') as f:
                for img in log_ratios_list:
                    for row in img:
                        f.write('\t'.join([f'{v:.6f}' for v in row]) + '\n')
            QMessageBox.information(self, "Auto-Save Complete",
                f"Saved:\n{os.path.basename(sum_file)}\n{os.path.basename(para_file)}\n\n"
                f"Location: {self.save_directory}")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to auto-save:\n{e}")

    def _stop_all_threads(self):
        """Signal all worker threads to stop and wait for them."""
        for thr in (self._capture_thread, self._scan_thread,
                    self._power_thread, self._gate_thread):
            if thr is not None and thr.isRunning():
                thr.stop()
                thr.wait(3000)

    def stop_scan(self):
        was_capturing = self.is_single_capturing
        self.is_scanning = False
        self.is_single_capturing = False
        self.is_power_scanning = False
        self.is_gate_sweeping = False
        self._stop_all_threads()
        if self.stage:
            self.stage.stop()
        self.update_button_states()
        self.btn_stop.setEnabled(False)
        if was_capturing:
            count = self._capture_count
            self.update_status_text(f"Stopped | {count} captures")
            if self.current_results is not None:
                try:
                    roi_x, roi_y, roi_w, roi_h = self._capture_roi
                    self.display_results(self.current_results, roi_x, roi_y, roi_w, roi_h,
                                         quick_mode=False)
                except Exception:
                    pass

    # =========================================================
    # Display (PyQtGraph)
    # =========================================================

    def update_status_text(self, text):
        self.status_label.setText(text)

    def calculate_log_ratio(self, avg_on, avg_off):
        epsilon = 1e-6
        ratio = (avg_on + epsilon) / (avg_off + epsilon)
        return -1000 * np.log10(ratio)

    def _crop_image(self, img):
        """Crop image columns to the save/display region if enabled."""
        if img is None:
            return None
        if not self.save_region_enable.isChecked():
            return img
        try:
            x0 = int(self.save_region_x.text())
            w = int(self.save_region_w.text())
        except ValueError:
            return img
        h, full_w = img.shape
        x0 = max(0, min(x0, full_w - 1))
        x1 = min(x0 + w, full_w)
        if x1 <= x0:
            return img
        return img[:, x0:x1]

    def display_results(self, results, roi_x, roi_y, roi_w, roi_h, quick_mode=False):
        avg_on = self._crop_image(results['avg_pump_on'])
        avg_off = self._crop_image(results['avg_pump_off'])
        n_on = results['n_pump_on']
        n_off = results['n_pump_off']
        threshold = results['threshold']
        ref_intensities = results['ref_intensities']

        new_roi = (roi_x, roi_y, roi_w, roi_h)
        roi_changed = (new_roi != self._cached_roi)
        if roi_changed:
            self._cached_roi = new_roi
            rx = [roi_x, roi_x + roi_w, roi_x + roi_w, roi_x, roi_x]
            ry = [roi_y, roi_y, roi_y + roi_h, roi_y + roi_h, roi_y]
            self.roi_rect_main.setData(rx, ry)

        if not quick_mode:
            mon_levels = None if self.monitor_levels_auto else (
                self.monitor_levels_min, self.monitor_levels_max)
            if avg_on is not None:
                self.img_on_item.setImage(avg_on.T, levels=mon_levels)
                self.text_on.setText(f'N={n_on}')
                self.text_on.setPos(avg_on.shape[1] / 2, 5)
            if avg_off is not None:
                self.img_off_item.setImage(avg_off.T, levels=mon_levels)
                self.text_off.setText(f'N={n_off}')
                self.text_off.setPos(avg_off.shape[1] / 2, 5)
            if roi_changed:
                rx = [roi_x, roi_x + roi_w, roi_x + roi_w, roi_x, roi_x]
                ry = [roi_y, roi_y, roi_y + roi_h, roi_y + roi_h, roi_y]
                self.roi_rect_on.setData(rx, ry)
                self.roi_rect_off.setData(rx, ry)
            frame_nums = np.arange(1, len(ref_intensities) + 1)
            ref_arr = np.array(ref_intensities)
            on_mask = ref_arr > threshold
            off_mask = ~on_mask
            if np.any(on_mask):
                self._scatter_on.setData(frame_nums[on_mask], ref_arr[on_mask])
            else:
                self._scatter_on.setData([], [])
            if np.any(off_mask):
                self._scatter_off.setData(frame_nums[off_mask], ref_arr[off_mask])
            else:
                self._scatter_off.setData([], [])
            self._threshold_line.setValue(threshold)
            self._intensity_trace.setData(frame_nums, ref_arr)

        if avg_on is not None and avg_off is not None:
            log_ratio = self.calculate_log_ratio(avg_on, avg_off)
            self.log_ratio_data = log_ratio
            if self.use_auto_z:
                vmax = max(abs(np.min(log_ratio)), abs(np.max(log_ratio)))
                vmin = -vmax
            else:
                vmin = self.z_min
                vmax = self.z_max
            self.main_img.setImage(log_ratio.T, levels=(vmin, vmax))
            if vmin != self._last_vmin or vmax != self._last_vmax:
                self.cbar.setLevels(values=(vmin, vmax))
                self._last_vmin = vmin
                self._last_vmax = vmax
            new_shape = log_ratio.shape
            if new_shape != self._cached_image_shape:
                self._cached_image_shape = new_shape
                h, w = new_shape
                self.profile_x_slider.setMaximum(w - 1)
                self.profile_y_slider.setMaximum(h - 1)
            if self.crosshair_x is None or self.crosshair_y is None:
                self.crosshair_x = log_ratio.shape[1] // 2
                self.crosshair_y = log_ratio.shape[0] // 2
                self.v_crosshair.setValue(self.crosshair_x)
                self.h_crosshair.setValue(self.crosshair_y)
            else:
                h, w = log_ratio.shape
                self.crosshair_x = int(np.clip(self.crosshair_x, 0, w - 1))
                self.crosshair_y = int(np.clip(self.crosshair_y, 0, h - 1))
            self.update_profiles(self.crosshair_x, self.crosshair_y, skip_gaussian=quick_mode)

    def update_profiles(self, x, y, skip_gaussian=False):
        if self.log_ratio_data is None:
            return
        lr = self.log_ratio_data
        h, w = lr.shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        self.crosshair_x = x
        self.crosshair_y = y
        self.updating_from_drag = True
        self.profile_x_slider.setValue(x)
        self.profile_y_slider.setValue(y)
        self.updating_from_drag = False

        x_profile = lr[y, :]
        if self.use_calibration:
            x_coords = self.wavelength_k * np.arange(w) + self.wavelength_b
            x_cur = self.wavelength_k * x + self.wavelength_b
        else:
            x_coords = np.arange(w)
            x_cur = x
        self._px_line.setData(x_coords, x_profile)
        self._px_marker.setData([x_cur], [x_profile[x]])

        y_profile = lr[:, x]
        y_coords = self.space_scale * np.arange(h)
        y_cur = self.space_scale * y
        self._py_line.setData(y_profile, y_coords)
        self._py_marker.setData([y_profile[y]], [y_cur])

        if not skip_gaussian and self.gaussian_fit_enabled:
            self.update_gaussian_fit_display(y_profile, y_coords)

    def update_profiles_from_sliders(self):
        if self.updating_from_drag or self.log_ratio_data is None:
            return
        x = self.profile_x_slider.value()
        y = self.profile_y_slider.value()
        h, w = self.log_ratio_data.shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        self.update_profiles(x, y)

    def update_gaussian_fit_display(self, y_profile, y_coords):
        use_auto = self.use_auto_fit_chk.isChecked()
        initial_params = None
        if not use_auto:
            try:
                amplitude = float(self.fit_amplitude_entry.text())
                center = float(self.fit_center_entry.text())
                width = float(self.fit_width_entry.text())
                offset = float(self.fit_offset_entry.text())
                initial_params = (amplitude, center, width, offset)
            except ValueError:
                use_auto = True
        result = self.gaussian_fitter.fit(y_profile, y_coords,
                                          initial_params=initial_params, use_auto=use_auto)
        if not result['success']:
            self._fit_data_line.setData([], [])
            self._fit_curve_line.setData([], [])
            self._fit_center_line.setVisible(False)
            self._fit_fwhm_region.setVisible(False)
            self.fit_plot.setTitle(f"Fit Failed: {result['error_message']}", color='r')
            return
        params = result['params']
        fit_curve = result['fit_curve']
        amplitude, center, width, offset = params
        fwhm = result['fwhm']
        self.width_history.append(abs(width))
        self.width_update_count += 1
        if len(self.width_history) > 100:
            self.width_history.pop(0)
        unit = 'um' if (self.use_calibration and self.space_scale != 1.0) else 'pix'
        self._fit_data_line.setData(y_coords, y_profile)
        self._fit_curve_line.setData(y_coords, fit_curve)
        self._fit_center_line.setValue(center)
        self._fit_center_line.setVisible(True)
        self._fit_fwhm_region.setRegion((center - fwhm / 2, center + fwhm / 2))
        self._fit_fwhm_region.setVisible(True)
        self.fit_plot.setTitle(
            f'Center: {center:.2f} {unit} | '
            f'Sigma: {abs(width):.3f} {unit} | '
            f'FWHM: {fwhm:.3f} {unit}',
            color='purple')
        updates = list(range(len(self.width_history)))
        self._width_line.setData(updates, self.width_history)
        if self.width_history:
            self.width_text.setText(f'Current: {self.width_history[-1]:.3f} {unit}')
            self.width_text.setPos(0, max(self.width_history))

    def force_update_fit(self):
        if self.log_ratio_data is not None and self.crosshair_x is not None:
            h, w = self.log_ratio_data.shape
            x = int(np.clip(self.crosshair_x, 0, w - 1))
            y_profile = self.log_ratio_data[:, x]
            y_coords = self.space_scale * np.arange(h)
            self.update_gaussian_fit_display(y_profile, y_coords)

    # =========================================================
    # Gaussian fit toggle & monitor Z range
    # =========================================================

    def _apply_pixel_size(self):
        try:
            ps = float(self.pixel_size_entry.text())
            if ps <= 0:
                raise ValueError("Pixel size must be positive")
            self.space_scale = ps
            self.use_calibration = True
            self.space_scale_entry.setText(str(ps))
            unit = 'um' if ps != 1.0 else 'pixels'
            self.fit_plot.setLabel('bottom', f'Y Position ({unit})')
            self.width_history_plot.setLabel('left', f'Width sigma ({unit})')
            if self.log_ratio_data is not None and self.crosshair_x is not None:
                self.update_profiles(self.crosshair_x, self.crosshair_y)
        except ValueError:
            QMessageBox.critical(self, "Error", "Pixel size must be a positive number!")

    def _toggle_gaussian_fit(self, state):
        self.gaussian_fit_enabled = bool(state)
        if not self.gaussian_fit_enabled:
            self._fit_data_line.setData([], [])
            self._fit_curve_line.setData([], [])
            self._fit_center_line.setVisible(False)
            self._fit_fwhm_region.setVisible(False)
            self.fit_plot.setTitle("Gaussian Fit (disabled)", color='gray')

    def _apply_monitor_z_range(self):
        try:
            zmin = float(self.monitor_z_min_entry.text())
            zmax = float(self.monitor_z_max_entry.text())
            if zmin >= zmax:
                QMessageBox.critical(self, "Error", "Min must be less than Max!")
                return
            self.monitor_levels_auto = False
            self.monitor_levels_min = zmin
            self.monitor_levels_max = zmax
            self.img_on_item.setLevels((zmin, zmax))
            self.img_off_item.setLevels((zmin, zmax))
        except ValueError:
            QMessageBox.critical(self, "Error", "Z values must be numbers!")

    def _auto_monitor_z_range(self):
        self.monitor_levels_auto = True
        self.img_on_item.setLevels(None)
        self.img_off_item.setLevels(None)

    # =========================================================
    # Display settings methods
    # =========================================================

    def apply_z_range(self):
        try:
            z_min = float(self.z_min_entry.text())
            z_max = float(self.z_max_entry.text())
            if z_min >= z_max:
                QMessageBox.critical(self, "Error", "Min must be less than Max!")
                return
            self.z_min = z_min
            self.z_max = z_max
            self.use_auto_z = False
            if self.log_ratio_data is not None:
                self.main_img.setImage(self.log_ratio_data.T, levels=(z_min, z_max))
                self.cbar.setLevels(values=(z_min, z_max))
                self._last_vmin = z_min
                self._last_vmax = z_max
        except ValueError:
            QMessageBox.critical(self, "Error", "Z values must be numbers!")

    def apply_intensity_range(self):
        try:
            imin = float(self.intensity_min_entry.text())
            imax = float(self.intensity_max_entry.text())
            if imin >= imax:
                QMessageBox.critical(self, "Error", "Min must be less than Max!")
                return
            self.z_min = imin; self.z_max = imax; self.use_auto_z = False
            if self.log_ratio_data is not None:
                self.main_img.setImage(self.log_ratio_data.T, levels=(imin, imax))
                self.cbar.setLevels(values=(imin, imax))
                self._last_vmin = imin; self._last_vmax = imax
        except ValueError:
            QMessageBox.critical(self, "Error", "Values must be numbers!")

    def auto_intensity_range(self):
        if self.log_ratio_data is not None:
            vmax = max(abs(np.min(self.log_ratio_data)), abs(np.max(self.log_ratio_data)))
            vmin = -vmax
            self.intensity_min_entry.setText(f"{vmin:.2f}")
            self.intensity_max_entry.setText(f"{vmax:.2f}")
            self.z_min = vmin; self.z_max = vmax; self.use_auto_z = True
            self.main_img.setImage(self.log_ratio_data.T, levels=(vmin, vmax))
            self.cbar.setLevels(values=(vmin, vmax))
            self._last_vmin = vmin; self._last_vmax = vmax

    def apply_axis_ranges(self):
        if self.log_ratio_data is None: return
        try:
            xmin = float(self.x_min_entry.text()); xmax = float(self.x_max_entry.text())
            ymin = float(self.y_min_entry.text()); ymax = float(self.y_max_entry.text())
            self.use_custom_axis_range = True
            self.custom_x_min = xmin; self.custom_x_max = xmax
            self.custom_y_min = ymin; self.custom_y_max = ymax
            self.main_plot.setXRange(xmin, xmax)
            self.main_plot.setYRange(ymin, ymax)
        except ValueError:
            QMessageBox.critical(self, "Error", "Values must be numbers!")

    def reset_axis_ranges(self):
        if self.log_ratio_data is not None:
            h, w = self.log_ratio_data.shape
            self.use_custom_axis_range = False
            self.x_min_entry.setText("0"); self.x_max_entry.setText(str(w))
            self.y_min_entry.setText("0"); self.y_max_entry.setText(str(h))
            self.main_plot.setXRange(0, w); self.main_plot.setYRange(0, h)

    def apply_profile_ranges(self):
        errors = []
        xx = self.profile_x_xmin_entry.text().strip()
        xX = self.profile_x_xmax_entry.text().strip()
        xy = self.profile_x_ymin_entry.text().strip()
        xY = self.profile_x_ymax_entry.text().strip()
        if xx and xX:
            try:
                mn, mx = float(xx), float(xX)
                if mn >= mx: raise ValueError
                self.profile_x_xmin = mn; self.profile_x_xmax = mx
                self.use_custom_profile_x_xlim = True
                self.profile_x_plot.setXRange(mn, mx)
            except ValueError: errors.append("Invalid X Profile X range")
        else:
            self.use_custom_profile_x_xlim = False; self.profile_x_plot.enableAutoRange(axis='x')
        if xy and xY:
            try:
                mn, mx = float(xy), float(xY)
                if mn >= mx: raise ValueError
                self.profile_x_ymin = mn; self.profile_x_ymax = mx
                self.use_custom_profile_x_ylim = True
                self.profile_x_plot.setYRange(mn, mx)
            except ValueError: errors.append("Invalid X Profile Y range")
        else:
            self.use_custom_profile_x_ylim = False; self.profile_x_plot.enableAutoRange(axis='y')
        yx = self.profile_y_xmin_entry.text().strip()
        yX = self.profile_y_xmax_entry.text().strip()
        yy = self.profile_y_ymin_entry.text().strip()
        yY = self.profile_y_ymax_entry.text().strip()
        if yx and yX:
            try:
                mn, mx = float(yx), float(yX)
                if mn >= mx: raise ValueError
                self.profile_y_xmin = mn; self.profile_y_xmax = mx
                self.use_custom_profile_y_xlim = True
                self.profile_y_plot.setXRange(mn, mx)
            except ValueError: errors.append("Invalid Y Profile X range")
        else:
            self.use_custom_profile_y_xlim = False; self.profile_y_plot.enableAutoRange(axis='x')
        if yy and yY:
            try:
                mn, mx = float(yy), float(yY)
                if mn >= mx: raise ValueError
                self.profile_y_ymin = mn; self.profile_y_ymax = mx
                self.use_custom_profile_y_ylim = True
                self.profile_y_plot.setYRange(mn, mx)
            except ValueError: errors.append("Invalid Y Profile Y range")
        else:
            self.use_custom_profile_y_ylim = False; self.profile_y_plot.enableAutoRange(axis='y')
        if errors:
            QMessageBox.critical(self, "Error", "\n".join(errors))

    def reset_profile_ranges(self):
        self.use_custom_profile_x_xlim = False; self.use_custom_profile_x_ylim = False
        self.use_custom_profile_y_xlim = False; self.use_custom_profile_y_ylim = False
        for e in [self.profile_x_xmin_entry, self.profile_x_xmax_entry,
                  self.profile_x_ymin_entry, self.profile_x_ymax_entry,
                  self.profile_y_xmin_entry, self.profile_y_xmax_entry,
                  self.profile_y_ymin_entry, self.profile_y_ymax_entry]:
            e.setText("")
        self.profile_x_plot.enableAutoRange()
        self.profile_y_plot.enableAutoRange()

    def apply_calibration(self):
        try:
            k = float(self.wavelength_k_entry.text())
            b = float(self.wavelength_b_entry.text())
            s = float(self.space_scale_entry.text())
            self.wavelength_k = k; self.wavelength_b = b; self.space_scale = s
            self.use_calibration = True
            if self.log_ratio_data is not None and self.crosshair_x is not None:
                self.update_profiles(self.crosshair_x, self.crosshair_y)
            QMessageBox.information(self, "Success",
                f"Calibration applied:\nX: lambda = {k:.4f} * pixel + {b:.2f} nm\n"
                f"Y: {s:.4f} um/pixel")
        except ValueError:
            QMessageBox.critical(self, "Error", "Calibration values must be numbers!")

    # =========================================================
    # Save results
    # =========================================================

    def save_results(self):
        if not self.scan_results and not self.current_results:
            QMessageBox.warning(self, "No Data", "No results to save!")
            return
        try:
            save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save results")
            if not save_dir: return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nd_tag = self._get_nd_angle_tag()
            nd_angle_val = None
            if nd_tag and self.rotation_stage and self.rotation_stage.is_initialized:
                try:
                    nd_angle_val = self.rotation_stage.get_angle()
                except Exception:
                    pass
            saved_files = []
            if self.scan_results:
                scan_file = os.path.join(save_dir, f"delay_scan{nd_tag}_{timestamp}.npz")
                time_delays_ps = np.array([r['time_delay_ps'] for r in self.scan_results])
                positions_mm = np.array([r['position_mm'] for r in self.scan_results])
                log_ratios = []
                for r in self.scan_results:
                    res = r['results']
                    on = self._crop_image(res['avg_pump_on'])
                    off = self._crop_image(res['avg_pump_off'])
                    if on is not None and off is not None:
                        log_ratios.append(self.calculate_log_ratio(on, off))
                if log_ratios:
                    save_kwargs = dict(
                        time_delays_ps=time_delays_ps, positions_mm=positions_mm,
                        time_zero_position_mm=self.time_zero_position_mm,
                        log_ratios=np.array(log_ratios), timestamp=timestamp)
                    if nd_angle_val is not None:
                        save_kwargs['nd_angle_deg'] = nd_angle_val
                    np.savez(scan_file, **save_kwargs)
                    saved_files.append(os.path.basename(scan_file))
            fig_file = os.path.join(save_dir, f"pump_probe_scan{nd_tag}_{timestamp}.png")
            pixmap = self.display_tabs.grab()
            pixmap.save(fig_file)
            saved_files.append(os.path.basename(fig_file))
            QMessageBox.information(self, "Success", f"Saved {len(saved_files)} files:\n" +
                                    "\n".join(saved_files))
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    # =========================================================
    # Rotation stage
    # =========================================================

    def _toggle_power_controls(self, checked):
        self.pwr_controls_frame.setVisible(checked)

    def _toggle_pol_controls(self, checked):
        self.pol_controls_frame.setVisible(checked)

    def _update_pol_rh_label(self):
        try:
            lh = float(self.pol_lh_entry.text())
            self.pol_rh_label.setText(f"RH = {lh + 45.0:.1f} deg")
        except ValueError:
            self.pol_rh_label.setText("RH = ??")

    def init_pol_stage(self):
        if not ROTATION_AVAILABLE:
            QMessageBox.critical(self, "Error", "Rotation stage library not available!")
            return
        try:
            serial_text = self.pol_serial_entry.text().strip()
            serial_number = int(serial_text) if serial_text else None
            self.pol_stage = RotationStage()
            self.pol_stage.initialize(serial_number=serial_number)
            self.pol_status.setText("● Connected")
            self.pol_status.setStyleSheet("color: green; font-weight: bold;")
            angle = self.pol_stage.get_angle()
            self.pol_angle_label.setText(f"Angle: {angle:.2f} deg")
            QMessageBox.information(self, "Success",
                f"Polarization stage initialized!\nSerial: {self.pol_stage.serial_number}")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid serial number!")
        except Exception as e:
            self.pol_status.setText("● Error")
            self.pol_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Failed to initialize polarization stage:\n{e}")

    def _update_pol_angle_label(self):
        if self.pol_stage and self.pol_stage.is_initialized:
            try:
                angle = self.pol_stage.get_angle()
                self.pol_angle_label.setText(f"Angle: {angle:.2f} deg")
            except Exception:
                pass

    def move_pol_stage(self):
        if not (self.pol_stage and self.pol_stage.is_initialized):
            QMessageBox.warning(self, "Not Initialized", "Polarization stage not initialized!")
            return
        try:
            angle = float(self.pol_angle_entry.text())
            self.pol_stage.move_to(angle, wait=True)
            self._update_pol_angle_label()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid angle value!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed:\n{e}")

    def step_pol_stage(self, direction):
        if not (self.pol_stage and self.pol_stage.is_initialized):
            QMessageBox.warning(self, "Not Initialized", "Polarization stage not initialized!")
            return
        try:
            step = float(self.pol_step_entry.text()) * direction
            self.pol_stage.move_by(step, wait=True)
            self._update_pol_angle_label()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid step value!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Step failed:\n{e}")

    def home_pol_stage(self):
        if not (self.pol_stage and self.pol_stage.is_initialized):
            QMessageBox.warning(self, "Not Initialized", "Polarization stage not initialized!")
            return
        try:
            self.pol_stage.home(wait=True)
            self._update_pol_angle_label()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Home failed:\n{e}")

    def stop_pol_stage(self):
        if self.pol_stage and self.pol_stage.is_initialized:
            try:
                self.pol_stage.stop()
                self._update_pol_angle_label()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Stop failed:\n{e}")

    def init_rotation_stage(self):
        if not self.rotation_available:
            QMessageBox.critical(self, "Error", "Rotation stage not available!")
            return
        try:
            serial_text = self.rot_serial_entry.text().strip()
            serial_number = int(serial_text) if serial_text else None
            self.rotation_stage.initialize(serial_number=serial_number)
            self.rot_status.setText("● Connected")
            self.rot_status.setStyleSheet("color: green; font-weight: bold;")
            self._update_rotation_angle()
            self.update_button_states()
            QMessageBox.information(self, "Success",
                f"Rotation stage initialized!\nSerial: {self.rotation_stage.serial_number}")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid serial number!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")

    def _update_rotation_angle(self):
        if self.rotation_stage and self.rotation_stage.is_initialized:
            try:
                angle = self.rotation_stage.get_angle()
                self.rot_angle_label.setText(f"Angle: {angle:.4f} deg")
            except Exception:
                self.rot_angle_label.setText("Angle: read error")

    def move_rotation_stage(self):
        if not (self.rotation_stage and self.rotation_stage.is_initialized):
            QMessageBox.warning(self, "Not Ready", "Rotation stage not initialized!")
            return
        try:
            angle = float(self.rot_angle_entry.text())
            self.rotation_stage.move_to(angle, wait=True)
            self._update_rotation_angle()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid angle!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed:\n{e}")

    def step_rotation_stage(self, direction):
        if not (self.rotation_stage and self.rotation_stage.is_initialized):
            QMessageBox.warning(self, "Not Ready", "Rotation stage not initialized!")
            return
        try:
            step = float(self.rot_step_entry.text()) * direction
            self.rotation_stage.move_by(step, wait=True)
            self._update_rotation_angle()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid step!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Step failed:\n{e}")

    def home_rotation_stage(self):
        if not (self.rotation_stage and self.rotation_stage.is_initialized):
            QMessageBox.warning(self, "Not Ready", "Rotation stage not initialized!")
            return
        try:
            self.rotation_stage.home(wait=True)
            self._update_rotation_angle()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Home failed:\n{e}")

    def stop_rotation_stage(self):
        if self.rotation_stage:
            self.rotation_stage.stop()

    def stop_power_scan(self):
        self.is_power_scanning = False
        if self._power_thread is not None and self._power_thread.isRunning():
            self._power_thread.stop()

    # =========================================================
    # Power scan
    # =========================================================

    def generate_power_scan_positions(self):
        try:
            s = float(self.pwr_start_entry.text())
            e = float(self.pwr_end_entry.text())
            st = float(self.pwr_step_entry.text())
            if st <= 0: raise ValueError("Step must be positive")
            angles = list(np.arange(s, e + st / 2, st))
            self.pwr_positions_text.setPlainText(", ".join([f"{a:.2f}" for a in angles]))
        except ValueError as ex:
            QMessageBox.critical(self, "Error", str(ex))

    def start_power_scan(self):
        try:
            if not self.stage or not self.stage.is_initialized:
                QMessageBox.critical(self, "Error",
                    "Delay stage not connected!\nConnect on the Hardware tab first.")
                return
            pos_text = self.pwr_positions_text.toPlainText().strip()
            angles = [float(a.strip()) for a in pos_text.split(',') if a.strip()]
            if not angles: raise ValueError("No angles specified!")
            delay_text = self.delay_positions.toPlainText().strip()
            time_delays_ps = [float(p.strip()) for p in delay_text.split(',') if p.strip()]
            if not time_delays_ps: raise ValueError("No time delays in Scan Setup tab!")
            positions_mm = [self.time_zero_position_mm + self.stage.ps_to_mm(d) for d in time_delays_ps]
            n_frames = int(self.n_frames_entry.text())
            default_avg = int(self.avg_times_entry.text())
            settle_time = float(self.pwr_settle_entry.text())
            roi = (int(self.roi_x.text()), int(self.roi_y.text()),
                   int(self.roi_w.text()), int(self.roi_h.text()))
            save_base = self.save_filename_entry.text().strip() or "power_scan"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._resolve_save_dir()
            if self.time_zero_position_mm == 0.0:
                r = QMessageBox.warning(self, "Time Zero Not Set",
                    f"Time zero is 0.0 mm — may not be set.\n"
                    f"Stage will move to {min(positions_mm):.4f} – {max(positions_mm):.4f} mm.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No)
                if r != QMessageBox.Yes: return
            pol_enabled = (self.pol_enable_chk.isChecked() and
                           self.pol_stage is not None and
                           self.pol_stage.is_initialized)
            lh_angle = 0.0
            pol_settle_s = 0.5
            if pol_enabled:
                try:
                    lh_angle = float(self.pol_lh_entry.text())
                    pol_settle_s = float(self.pol_settle_entry.text())
                except ValueError:
                    pass
            n_pol = 2 if pol_enabled else 1
            msg = (f"Start power-dependent delay scan?\n\n"
                   f"Angles: {len(angles)} ({angles[0]:.1f} to {angles[-1]:.1f})\n"
                   f"Delays: {len(time_delays_ps)} per angle\n"
                   f"Stage positions: {min(positions_mm):.4f} – {max(positions_mm):.4f} mm\n"
                   f"Frames: {n_frames} | Avg: {default_avg}\n")
            if pol_enabled:
                msg += f"Polarization: LH={lh_angle:.1f}° / RH={lh_angle+45:.1f}°  (2× per angle)\n"
            msg += f"Total delay scans: {len(angles) * n_pol}"
            reply = QMessageBox.question(self, "Confirm", msg, QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes: return
            self.is_power_scanning = True
            self.power_scan_results = []
            self._power_scan_timestamp = timestamp
            self._power_scan_angles_completed = 0
            self.btn_power_scan.setEnabled(False)
            self.btn_power_stop.setEnabled(True)
            self.btn_single.setEnabled(False)
            self.btn_scan.setEnabled(False)
            self._save_power_scan_delays(time_delays_ps, save_base, timestamp)
            crop = None
            if self.save_region_enable.isChecked():
                try:
                    crop = (int(self.save_region_x.text()),
                            int(self.save_region_w.text()))
                except ValueError:
                    pass
            t0_ps = self.stage.mm_to_ps(self.time_zero_position_mm) if self.stage else 0.0
            pixel_size = float(self.pixel_size_entry.text()) if hasattr(self, 'pixel_size_entry') else 0.07
            crop_h = self.camera.height if self.camera and self.camera.is_initialized else 128
            self._power_thread = _PowerScanThread(
                self.camera, self.stage, self.rotation_stage,
                angles, time_delays_ps, positions_mm,
                n_frames, roi, dict(self.delay_to_avg_map), default_avg,
                settle_time, self.save_directory, save_base, timestamp,
                crop=crop, t0_ps=t0_ps, pixel_size=pixel_size, crop_h=crop_h,
                pol_enabled=pol_enabled,
                pol_stage=self.pol_stage if pol_enabled else None,
                lh_angle=lh_angle, pol_settle_s=pol_settle_s)
            self._power_thread.progress.connect(self.update_status_text)
            self._power_thread.step_result.connect(self._on_power_step)
            self._power_thread.angle_complete.connect(self._on_power_angle_done)
            self._power_thread.finished.connect(self._on_power_finished)
            self._power_thread.error_occurred.connect(self._on_power_error)
            self._power_thread.start()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Power scan failed:\n{e}")

    def _on_power_step(self, final_results, roi):
        self.current_results = final_results
        self.display_results(final_results, *roi, quick_mode=False)

    def _on_power_angle_done(self, angle, delay_results, count):
        self._power_scan_angles_completed = count
        self.power_scan_results.append({
            'angle_deg': angle, 'n_delays': len(delay_results), 'saved': True})
        self.update_status_text(f"Angle {angle:.1f} saved ({count} done).")

    def _on_power_finished(self, completed):
        self.is_power_scanning = False
        self.update_button_states()
        self.btn_power_stop.setEnabled(False)
        if completed:
            self.update_status_text(
                f"Power scan complete! {self._power_scan_angles_completed} angles saved.")
            self.btn_save.setEnabled(True)
            QMessageBox.information(self, "Complete",
                f"Power scan finished!\n{self._power_scan_angles_completed} angles\n"
                f"All data saved to:\n{self.save_directory}")
        else:
            self.update_status_text(
                f"Power scan stopped. {self._power_scan_angles_completed} angles saved.")

    def _on_power_error(self, msg):
        self.is_power_scanning = False
        self.update_button_states()
        self.btn_power_stop.setEnabled(False)
        QMessageBox.critical(self, "Error", f"Power scan failed:\n{msg}")

    def _save_power_scan_delays(self, time_delays_ps, save_base, timestamp):
        try:
            path = os.path.join(self.save_directory,
                                f"{save_base}_power_delays_{timestamp}.txt")
            with open(path, 'w') as f:
                for d in time_delays_ps:
                    f.write(f"{d:.4f}\n")
        except Exception as e:
            print(f"Warning: failed to save delays file: {e}")

    # =========================================================
    # Window close
    # =========================================================

    def closeEvent(self, event):
        self.is_scanning = False
        self.is_single_capturing = False
        self.is_power_scanning = False
        self.is_gate_sweeping = False
        self._stop_all_threads()
        if self._gate_poll_timer is not None:
            self._gate_poll_timer.stop()
        if self.front_widget and self.front_widget.dev:
            try:
                self.front_widget._on_disconnect()
            except Exception:
                pass
        if self.back_widget and self.back_widget.dev:
            try:
                self.back_widget._on_disconnect()
            except Exception:
                pass
        self.camera.close()
        if self.stage:
            self.stage.close()
        if self.rotation_stage:
            self.rotation_stage.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = PumpProbeScanGUI()
    window.show()
    sys.exit(app.exec_())

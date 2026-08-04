"""
workers.py — Gate + Power sweep thread for pump-probe measurements.

The inner delay scan logic (_run_delay_scan) is copied verbatim from
_DelayScanThread.run() in V2_optimized/gui_integrated.py. Do not alter
that block — it is intentionally identical so the measurement behaviour
is unchanged.
"""

import os
import time
import numpy as np
import pathlib
import sys

# Allow importing camera/analysis helpers from V2_optimized
_V2_DIR = str(pathlib.Path(__file__).parent.parent / "V2_optimized")
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from PyQt5.QtCore import QThread, pyqtSignal

from camera_simple import analyze_frames

from .config import RAMP_STEP_V, RAMP_DWELL_S

try:
    from keithley.keithley_wrapper import SweepController
except ImportError:
    # Fallback: define a no-op if keithley is not importable
    class SweepController:
        def is_aborted(self): return False
        def abort(self): pass
        def next_step(self): pass


def _calculate_log_ratio(avg_on, avg_off):
    epsilon = 1e-6
    ratio = (avg_on + epsilon) / (avg_off + epsilon)
    return -1000 * np.log10(ratio)


class _GatePowerSweepThread(QThread):
    """
    Outer sweep thread for gate-voltage-dependent pump-probe measurements.

    Handles three loop configurations (selected by power_enabled + gate_outer):
      1. Gate-only  (power_enabled=False):  gate → delays
      2. Gate outer (power_enabled=True, gate_outer=True):  gate → angles → delays
      3. Power outer (power_enabled=True, gate_outer=False): angles → gate → delays

    The innermost delay scan (_run_delay_scan) is a verbatim copy of
    _DelayScanThread.run() from gui_integrated.py — zero logic changes.
    """

    progress       = pyqtSignal(str)
    step_result    = pyqtSignal(object, tuple, dict)   # results, roi, step_info
    ramp_vi        = pyqtSignal(str, float, float)      # "front"/"back", V, I
    gate_step_done = pyqtSignal(float, float, int)      # v_front, v_back, gate_idx
    power_step_done = pyqtSignal(float, int)            # angle, angle_idx
    finished       = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera, stage,
        front_dev, back_dev,
        gate_pairs,          # list of (v_front, v_back)
        dual_mode: bool,
        gate_settle_s: float,
        time_delays_ps, positions_mm,
        n_frames: int, roi: tuple,
        avg_map: dict, default_avg: int,
        power_enabled: bool = False,
        rotation_stage=None,
        angles=None,
        power_settle_s: float = 0.5,
        gate_outer: bool = True,
        save_dir: str = ".",
        save_base: str = "gate_scan",
        timestamp: str = "",
        crop=None,
        t0_ps: float = 0.0,
        pixel_size: float = 0.07,
        crop_h: int = 128,
        parent=None,
    ):
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
        self._running = True
        self._gate_ctrl = None

    # ------------------------------------------------------------------
    # Main run logic
    # ------------------------------------------------------------------

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

    def _run_gate_only(self):
        for i, (vf, vb) in enumerate(self.gate_pairs):
            if not self._running:
                break
            vb_str = f", Vb={vb:+.3f} V" if self.dual_mode else ""
            self.progress.emit(
                f"Gate step {i+1}/{len(self.gate_pairs)}: Vf={vf:+.3f} V{vb_str}")
            self._ramp_gates(vf, vb)
            delay_results = self._run_delay_scan()
            self._save_gate_step(vf, vb, None, delay_results)
            self.gate_step_done.emit(vf, vb, i)

    def _run_gate_outer_power_inner(self):
        for i, (vf, vb) in enumerate(self.gate_pairs):
            if not self._running:
                break
            vb_str = f", Vb={vb:+.3f} V" if self.dual_mode else ""
            self.progress.emit(
                f"Gate {i+1}/{len(self.gate_pairs)}: Vf={vf:+.3f} V{vb_str}")
            self._ramp_gates(vf, vb)
            for j, angle in enumerate(self.angles):
                if not self._running:
                    break
                self.progress.emit(
                    f"Gate {i+1}/{len(self.gate_pairs)} Vf={vf:+.3f} V | "
                    f"Angle {j+1}/{len(self.angles)} ({angle:.1f}°)")
                self.rotation_stage.move_to(angle, wait=True)
                time.sleep(self.power_settle_s)
                delay_results = self._run_delay_scan()
                self._save_gate_step(vf, vb, angle, delay_results)
                self.power_step_done.emit(angle, j)
            self.gate_step_done.emit(vf, vb, i)

    def _run_power_outer_gate_inner(self):
        for j, angle in enumerate(self.angles):
            if not self._running:
                break
            self.progress.emit(
                f"Angle {j+1}/{len(self.angles)} ({angle:.1f}°): rotating...")
            self.rotation_stage.move_to(angle, wait=True)
            time.sleep(self.power_settle_s)
            for i, (vf, vb) in enumerate(self.gate_pairs):
                if not self._running:
                    break
                vb_str = f", Vb={vb:+.3f} V" if self.dual_mode else ""
                self.progress.emit(
                    f"Angle {j+1}/{len(self.angles)} ({angle:.1f}°) | "
                    f"Gate {i+1}/{len(self.gate_pairs)}: Vf={vf:+.3f} V{vb_str}")
                self._ramp_gates(vf, vb)
                delay_results = self._run_delay_scan()
                self._save_gate_step(vf, vb, angle, delay_results)
                self.gate_step_done.emit(vf, vb, i)
            self.power_step_done.emit(angle, j)

    # ------------------------------------------------------------------
    # Gate ramping
    # ------------------------------------------------------------------

    def _ramp_gates(self, vf: float, vb: float):
        if self.front_dev is not None:
            self.progress.emit(f"Ramping front gate to {vf:+.3f} V ...")
            ctrl = SweepController()
            self._gate_ctrl = ctrl

            def on_front_ms(vi):
                self.ramp_vi.emit("front", vi.v, vi.i)

            self.front_dev.ramp_to_voltage(
                vf,
                ramp_step_v=RAMP_STEP_V,
                ramp_dwell_s=RAMP_DWELL_S,
                controller=ctrl,
                micro_read=True,
                on_microstep=on_front_ms,
                verify=True,
            )

        if self.dual_mode and self.back_dev is not None:
            self.progress.emit(f"Ramping back gate to {vb:+.3f} V ...")
            ctrl2 = SweepController()

            def on_back_ms(vi):
                self.ramp_vi.emit("back", vi.v, vi.i)

            self.back_dev.ramp_to_voltage(
                vb,
                ramp_step_v=RAMP_STEP_V,
                ramp_dwell_s=RAMP_DWELL_S,
                controller=ctrl2,
                micro_read=True,
                on_microstep=on_back_ms,
                verify=True,
            )

        time.sleep(self.gate_settle_s)

    # ------------------------------------------------------------------
    # Inner delay scan — verbatim copy of _DelayScanThread.run() body
    # from V2_optimized/gui_integrated.py lines 126-175.
    # DO NOT CHANGE THIS LOGIC.
    # ------------------------------------------------------------------

    def _run_delay_scan(self) -> list:
        scan_results = []
        for i, (delay_ps, position_mm) in enumerate(
                zip(self.time_delays_ps, self.positions_mm)):
            if not self._running:
                self.progress.emit(
                    f"Scan stopped at {i}/{len(self.positions_mm)}")
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
                'ref_intensities': results['ref_intensities'],
            }
            step_info = {
                'time_delay_ps': delay_ps, 'position_mm': position_mm,
                'results': final_results, 'avg_times_used': avg_times,
            }
            scan_results.append(step_info)
            self.step_result.emit(final_results, self.roi, step_info)
        return scan_results

    # ------------------------------------------------------------------
    # Save — TAM-compatible *para.txt + *sum.txt with gate voltage tags
    # ------------------------------------------------------------------

    def _save_gate_step(self, vf: float, vb: float, angle, delay_results: list):
        try:
            # Build filename tag for front gate voltage
            sign = 'p' if vf >= 0 else 'n'
            vf_tag = f"Vf{sign}{abs(vf):.3f}V".replace('.', 'd')
            tag = vf_tag
            if self.dual_mode and self.back_dev is not None:
                sign2 = 'p' if vb >= 0 else 'n'
                vb_tag = f"Vb{sign2}{abs(vb):.3f}V".replace('.', 'd')
                tag = f"{vf_tag}_{vb_tag}"

            if angle is not None:
                astr = f"{angle:.1f}".replace('.', 'p').replace('-', 'n')
                if self.gate_outer:
                    prefix = f"{self.save_base}_{tag}_ND{astr}deg_{self.timestamp}"
                else:
                    prefix = f"{self.save_base}_ND{astr}deg_{tag}_{self.timestamp}"
            else:
                prefix = f"{self.save_base}_{tag}_{self.timestamp}"

            if not delay_results:
                return

            delays   = [r['time_delay_ps'] for r in delay_results]
            avg_used = [r.get('avg_times_used', 1) for r in delay_results]
            crop_w   = (self.crop[1] if self.crop
                        else (self.camera.width if self.camera else 480))

            # --- para.txt (same format as _PowerScanThread._save_angle) ---
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
                # Extra gate metadata lines (prefixed # so TAM code ignores them)
                f.write(f'# FrontGateV={vf:.6f}\n')
                f.write(f'# BackGateV={vb:.6f}\n')

            # --- sum.txt ---
            sum_path = os.path.join(self.save_dir, f"{prefix}sum.txt")
            with open(sum_path, 'w') as f:
                for r in delay_results:
                    res = r['results']
                    on  = self._crop_img(res['avg_pump_on'])
                    off = self._crop_img(res['avg_pump_off'])
                    if on is not None and off is not None:
                        lr = _calculate_log_ratio(on, off).T
                        for row in lr:
                            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')

        except Exception as e:
            print(f"Warning: save gate step failed: {e}")

    def _crop_img(self, img):
        if img is None or self.crop is None:
            return img
        x0, w = self.crop
        h, full_w = img.shape
        x0 = max(0, min(x0, full_w - 1))
        x1 = min(x0 + w, full_w)
        return img[:, x0:x1] if x1 > x0 else img

    # ------------------------------------------------------------------

    def stop(self):
        self._running = False
        if self._gate_ctrl is not None:
            self._gate_ctrl.abort()

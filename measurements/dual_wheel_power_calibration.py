import csv
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DualWheelPowerEntry:
    a_deg: float
    b_deg: float
    power_w: float
    intensity: float
    index: int


@dataclass
class DualWheelPowerData:
    path: str
    unit: str
    scale: float
    entries: List[DualWheelPowerEntry]


def _normalize_unit(unit: str) -> str:
    val = unit.strip().lower()
    if val.endswith("w"):
        if val.startswith("n"):
            return "nW"
        if val.startswith("u"):
            return "uW"
        if val.startswith("m"):
            return "mW"
        return "W"
    if val in ("nw", "nanowatt"):
        return "nW"
    if val in ("uw", "micro", "microwatt"):
        return "uW"
    if val in ("mw", "milliwatt"):
        return "mW"
    if val == "w":
        return "W"
    return "nW"


def _unit_scale(unit: str) -> float:
    unit = _normalize_unit(unit)
    if unit == "W":
        return 1.0
    if unit == "mW":
        return 1e-3
    if unit == "uW":
        return 1e-6
    return 1e-9


def _row_has_alpha(row: List[str]) -> bool:
    for cell in row:
        if re.search(r"[a-zA-Z]", str(cell)):
            return True
    return False


def _parse_unit_from_header(header: List[str]) -> str:
    for cell in header:
        cell = cell.strip().lower()
        if cell.startswith("power_"):
            return _normalize_unit(cell.split("power_", 1)[-1])
        if "power" in cell and "(" in cell and ")" in cell:
            inner = cell.split("(", 1)[-1].split(")", 1)[0]
            return _normalize_unit(inner)
    return "nW"


def is_dual_wheel_calibration(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            header = [c.strip().lower() for c in row]
            if len(header) < 3:
                return False
            if "a" in header[0] and "b" in header[1] and "power" in header[2]:
                return True
            if _row_has_alpha(header):
                return False
            return False
    return False


def load_dual_wheel_calibration(path: str) -> DualWheelPowerData:
    if not path:
        raise ValueError("No calibration path provided")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            if header is None and _row_has_alpha(row):
                header = row
                continue
            rows.append(row)

    unit = _parse_unit_from_header([c.strip() for c in (header or [])])
    scale = _unit_scale(unit)

    entries: List[DualWheelPowerEntry] = []
    for idx, row in enumerate(rows):
        if len(row) < 3:
            continue
        try:
            a_deg = float(row[0])
            b_deg = float(row[1])
            power = float(row[2])
        except Exception:
            continue
        intensity = float("nan")
        if len(row) > 3:
            try:
                intensity = float(row[3])
            except Exception:
                intensity = float("nan")
        entries.append(
            DualWheelPowerEntry(
                a_deg=a_deg,
                b_deg=b_deg,
                power_w=power * scale,
                intensity=float(intensity),
                index=int(idx),
            )
        )

    if not entries:
        raise ValueError("No calibration data rows found")

    return DualWheelPowerData(
        path=path,
        unit=unit,
        scale=scale,
        entries=entries,
    )


# ------------------------------------------------------------------ #
# HWP + ND wheel calibration (wheel_hwp_power_calib format)           #
# ------------------------------------------------------------------ #

@dataclass
class HWPNDCalibData:
    path: str
    hwp_angles:   List[float] = field(default_factory=list)
    hwp_powers_w: List[float] = field(default_factory=list)
    hwp_stds_w:   List[float] = field(default_factory=list)
    nd_angles:    List[float] = field(default_factory=list)
    nd_powers_w:  List[float] = field(default_factory=list)
    nd_stds_w:    List[float] = field(default_factory=list)


def is_hwp_nd_calibration(path: str) -> bool:
    """Return True if the file is a wheel+HWP calibration (has [HWP_SWEEP] marker)."""
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", newline="") as f:
            for line in f:
                if line.strip() == "[HWP_SWEEP]":
                    return True
    except OSError:
        pass
    return False


def load_hwp_nd_calibration(path: str) -> HWPNDCalibData:
    """Load a wheel+HWP power calibration CSV (created by WheelHWPCalibWidget)."""
    if not path:
        raise ValueError("No path provided")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    hwp_rows: List[List[float]] = []
    nd_rows:  List[List[float]] = []
    section = None

    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            cell = row[0].strip()
            if cell.startswith("#"):
                continue
            if cell == "[HWP_SWEEP]":
                section = "hwp"; continue
            if cell == "[ND_SWEEP]":
                section = "nd"; continue
            if cell in ("hwp_angle_deg", "nd_angle_deg"):
                continue
            try:
                vals = [float(c) for c in row[:3]]
            except (ValueError, IndexError):
                continue
            if section == "hwp":
                hwp_rows.append(vals)
            elif section == "nd":
                nd_rows.append(vals)

    if not hwp_rows:
        raise ValueError("No [HWP_SWEEP] data found in file")

    return HWPNDCalibData(
        path=path,
        hwp_angles=[r[0] for r in hwp_rows],
        hwp_powers_w=[r[1] for r in hwp_rows],
        hwp_stds_w=[r[2] for r in hwp_rows],
        nd_angles=[r[0] for r in nd_rows],
        nd_powers_w=[r[1] for r in nd_rows],
        nd_stds_w=[r[2] for r in nd_rows],
    )


def compute_hwp_nd_entries(
    data: HWPNDCalibData, nd_angle: float
) -> Tuple[List[DualWheelPowerEntry], str, float]:
    """
    Given a chosen ND wheel angle, return (entries, unit, scale) where each
    entry has a_deg=HWP angle, b_deg=nd_angle (fixed), power_w=converted power.

    unit / scale are chosen automatically from the max converted power.
    """
    hwp_p = data.hwp_powers_w
    p_max = max(hwp_p) if hwp_p else 1.0

    # Find matching ND power
    p_nd = p_max  # fallback: same max power
    if data.nd_angles and nd_angle in data.nd_angles:
        idx = data.nd_angles.index(nd_angle)
        p_nd = data.nd_powers_w[idx]
    elif data.nd_angles:
        # nearest
        dists = [abs(a - nd_angle) for a in data.nd_angles]
        idx = dists.index(min(dists))
        p_nd = data.nd_powers_w[idx]

    entries: List[DualWheelPowerEntry] = []
    for i, (hwp_ang, pw) in enumerate(zip(data.hwp_angles, hwp_p)):
        norm  = pw / p_max if p_max > 0 else 0.0
        conv_w = norm * p_nd
        entries.append(DualWheelPowerEntry(
            a_deg=hwp_ang,
            b_deg=nd_angle,
            power_w=conv_w,
            intensity=norm,
            index=i,
        ))

    # Auto-select display unit from max converted power
    max_p = p_nd if p_nd > 0 else 1e-9
    if max_p < 1e-6:
        unit, scale = "nW", 1e-9
    elif max_p < 1e-3:
        unit, scale = "uW", 1e-6
    elif max_p < 1.0:
        unit, scale = "mW", 1e-3
    else:
        unit, scale = "W", 1.0

    return entries, unit, scale

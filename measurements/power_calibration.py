import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PowerCalibEntry:
    series: str
    position_deg: float
    power_w: float
    nd_filters: Tuple[int, ...]
    nd_ratio: Optional[float]

    @property
    def is_base(self) -> bool:
        return self.series.strip().lower() == "base"

    @property
    def nd_count(self) -> int:
        return len(self.nd_filters)


@dataclass
class PowerCalibData:
    path: str
    unit: str
    scale: float
    nd_ratios: Dict[int, float]
    entries: List[PowerCalibEntry]
    series_names: List[str]


def _unit_from_header(header: List[str]) -> str:
    if len(header) < 2:
        return "nW"
    h1 = header[1].strip().lower()
    if "uw" in h1:
        return "uW"
    if "nw" in h1:
        return "nW"
    return "nW"


def _unit_scale(unit: str) -> float:
    return 1e-6 if unit == "uW" else 1e-9


def _parse_series_filters(series: str) -> Tuple[int, ...]:
    if not series:
        return ()
    nums = re.findall(r"nd\\s*#\\s*(\\d+)", series.lower())
    if not nums:
        return ()
    return tuple(sorted({int(n) for n in nums}))


def load_power_calibration(path: str) -> PowerCalibData:
    if not path:
        raise ValueError("No calibration path provided")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    nd_ratios: Dict[int, float] = {}
    data_rows: List[List[str]] = []
    header_row: Optional[List[str]] = None
    nd_index = 1

    for row in rows:
        if not row:
            continue
        head = row[0].strip().lower()
        if head == "#nd_filter":
            if len(row) >= 3:
                try:
                    label = row[1].strip().lower()
                    m = re.search(r"(\\d+)", label)
                    if m:
                        idx = int(m.group(1))
                    else:
                        idx = nd_index
                        nd_index += 1
                    nd_ratios[idx] = float(row[2])
                except Exception:
                    continue
            continue
        if "position" in head and len(row) > 1 and "power" in row[1].strip().lower():
            header_row = row
            continue
        data_rows.append(row)

    unit = _unit_from_header(header_row or [])
    scale = _unit_scale(unit)

    entries: List[PowerCalibEntry] = []
    series_names: List[str] = []
    for row in data_rows:
        if len(row) < 2:
            continue
        try:
            pos = float(row[0])
            power = float(row[1])
        except Exception:
            continue
        series = row[2].strip() if len(row) > 2 else "base"
        if not series:
            series = "base"
        if series.strip().lower() == "base":
            series = "base"
        filters = _parse_series_filters(series)
        ratio = None
        if not filters:
            ratio = 1.0
        else:
            ratio_val = 1.0
            for idx in filters:
                if idx not in nd_ratios:
                    ratio_val = None
                    break
                ratio_val *= nd_ratios[idx]
            ratio = ratio_val
        entry = PowerCalibEntry(
            series=series,
            position_deg=pos,
            power_w=power * scale,
            nd_filters=filters,
            nd_ratio=ratio,
        )
        entries.append(entry)
        if entry.series not in series_names:
            series_names.append(entry.series)

    return PowerCalibData(
        path=path,
        unit=unit,
        scale=scale,
        nd_ratios=nd_ratios,
        entries=entries,
        series_names=series_names,
    )


def power_key(power_w: float, *, scale: float, digits: int = 9) -> float:
    return round(power_w / scale, digits)


def entry_priority(entry: PowerCalibEntry) -> Tuple[int, int, float]:
    base_rank = 0 if entry.is_base else 1
    nd_count = entry.nd_count if entry.nd_filters else 0
    ratio = entry.nd_ratio if entry.nd_ratio is not None else float("inf")
    return (base_rank, nd_count, ratio)

import csv
import itertools
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .config import BTN_W


@dataclass(frozen=True)
class _BasePoint:
    id: int
    position_deg: float
    power_w: float
    positions: dict[str, float]
    primary_label: str


@dataclass(frozen=True)
class _Filter:
    id: int
    ratio: float


@dataclass(frozen=True)
class _Entry:
    kind: str  # "base", "nd", "manual"
    base_id: int
    combo: tuple[int, ...]
    series: str
    position_deg: float
    power_w: float
    positions: dict[str, float]
    primary_label: str


class PowerCalibrationWidget(QtWidgets.QGroupBox):
    def __init__(self, position_provider: Optional[Callable[[], float]] = None, parent=None):
        super().__init__("Power Calibration", parent)
        self._position_provider = position_provider
        self._base_points: list[_BasePoint] = []
        self._manual_entries: list[_Entry] = []
        self._filters: list[_Filter] = []
        self._excluded_combos: set[tuple[int, tuple[int, ...]]] = set()
        self._next_point_id = 1
        self._next_filter_id = 1
        self._position_labels: list[str] = []

        self._build_ui()
        self._refresh()

    def set_position_provider(self, provider: Optional[Callable[[], float]]) -> None:
        self._position_provider = provider

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Power input
        self.powerSpin = QtWidgets.QDoubleSpinBox()
        self.powerSpin.setDecimals(6)
        self.powerSpin.setRange(0.0, 1e12)
        self.powerSpin.setValue(1.0)

        self.unitCombo = QtWidgets.QComboBox()
        self.unitCombo.addItems(["nW", "uW"])

        self.recordBtn = QtWidgets.QPushButton("Record this power")
        self.recordBtn.setFixedWidth(150)

        layout.addWidget(QtWidgets.QLabel("Power:"), 0, 0)
        layout.addWidget(self.powerSpin, 0, 1)
        layout.addWidget(self.unitCombo, 0, 2)
        layout.addWidget(self.recordBtn, 0, 3)

        # Table + controls
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Series", "Position (deg)", "Power"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.deleteBtn = QtWidgets.QPushButton("Delete Selected")
        self.deleteBtn.setFixedWidth(BTN_W)
        self.clearBtn = QtWidgets.QPushButton("Clear All Points")
        self.clearBtn.setFixedWidth(BTN_W)
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(BTN_W)
        self.loadBtn = QtWidgets.QPushButton("Load CSV")
        self.loadBtn.setFixedWidth(BTN_W)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.deleteBtn)
        btn_row.addWidget(self.clearBtn)
        btn_row.addWidget(self.exportBtn)
        btn_row.addWidget(self.loadBtn)
        btn_row.addStretch()

        # ND filters
        self.ndRatioSpin = QtWidgets.QDoubleSpinBox()
        self.ndRatioSpin.setDecimals(9)
        self.ndRatioSpin.setRange(1e-12, 1e9)
        self.ndRatioSpin.setValue(10.0)

        self.addNdBtn = QtWidgets.QPushButton("Add ND Filter")
        self.addNdBtn.setFixedWidth(BTN_W)
        self.removeNdBtn = QtWidgets.QPushButton("Remove ND")
        self.removeNdBtn.setFixedWidth(BTN_W)

        self.ndTable = QtWidgets.QTableWidget(0, 2)
        self.ndTable.setHorizontalHeaderLabels(["Filter", "Ratio"])
        self.ndTable.horizontalHeader().setStretchLastSection(True)
        self.ndTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ndTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        nd_box = QtWidgets.QVBoxLayout()
        nd_box.setContentsMargins(6, 0, 0, 0)
        nd_row = QtWidgets.QHBoxLayout()
        nd_row.addWidget(QtWidgets.QLabel("ND ratio (before/after):"))
        nd_row.addWidget(self.ndRatioSpin)
        nd_row.addWidget(self.addNdBtn)
        nd_row.addWidget(self.removeNdBtn)
        nd_row.addStretch()
        nd_box.addLayout(nd_row)
        nd_box.addWidget(self.ndTable)

        lists_row = QtWidgets.QHBoxLayout()
        lists_row.setSpacing(12)
        left_box = QtWidgets.QVBoxLayout()
        left_box.setContentsMargins(0, 0, 6, 0)
        left_box.addWidget(self.table)
        left_box.addLayout(btn_row)
        lists_row.addLayout(left_box, 2)
        lists_row.addLayout(nd_box, 1)

        # Plot
        self.fig = Figure(figsize=(4, 2.2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Index (sorted)")
        self.ax.set_ylabel("Power")
        layout.addLayout(lists_row, 2, 0, 1, 4)
        layout.addWidget(self.canvas, 3, 0, 1, 4)
        layout.setRowStretch(3, 1)

        # Signals
        self.recordBtn.clicked.connect(self.on_record)
        self.addNdBtn.clicked.connect(self.on_add_nd)
        self.removeNdBtn.clicked.connect(self.on_remove_nd)
        self.deleteBtn.clicked.connect(self.on_delete_selected)
        self.clearBtn.clicked.connect(self.on_clear_all)
        self.exportBtn.clicked.connect(self.on_export)
        self.loadBtn.clicked.connect(self.on_load)
        self.unitCombo.currentIndexChanged.connect(self._refresh)

    def _unit_scale(self) -> float:
        return 1e-9 if self.unitCombo.currentText() == "nW" else 1e-6

    def _unit_label(self) -> str:
        return self.unitCombo.currentText()

    def _normalize_positions(self, positions: dict, primary_label: str) -> tuple[float, dict[str, float], str]:
        cleaned = {}
        for key, val in positions.items():
            if key is None:
                continue
            label = str(key).strip()
            if not label:
                continue
            try:
                cleaned[label] = float(val)
            except Exception:
                continue
        if not cleaned:
            raise RuntimeError("No valid positions")
        if primary_label not in cleaned:
            primary_label = next(iter(cleaned.keys()))
        primary_pos = cleaned[primary_label]
        return float(primary_pos), cleaned, primary_label

    def _get_positions(self) -> tuple[float, dict[str, float], str]:
        if self._position_provider is None:
            raise RuntimeError("No position provider")
        raw = self._position_provider()
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
            primary = raw[0] if raw[0] is not None else ""
            return self._normalize_positions(raw[1], str(primary))
        if isinstance(raw, dict):
            return self._normalize_positions(raw, "position")
        return float(raw), {"position": float(raw)}, "position"

    def _update_position_labels(self, positions: dict[str, float], primary_label: str) -> None:
        if not positions:
            return
        if not self._position_labels:
            others = [p for p in positions.keys() if p != primary_label]
            self._position_labels = [primary_label] + sorted(others)
            return
        for label in positions.keys():
            if label not in self._position_labels:
                self._position_labels.append(label)

    def _filter_index_map(self) -> dict[int, int]:
        return {f.id: idx + 1 for idx, f in enumerate(self._filters)}

    def _series_label(self, combo: tuple[int, ...]) -> str:
        if not combo:
            return "base"
        idx_map = self._filter_index_map()
        parts = [f"nd #{idx_map[c]}" for c in combo if c in idx_map]
        return "+".join(parts) if parts else "nd"

    def _combo_keys(self) -> list[tuple[int, ...]]:
        ids = [f.id for f in self._filters]
        combos: list[tuple[int, ...]] = []
        for r in range(1, len(ids) + 1):
            combos.extend(tuple(c) for c in itertools.combinations(ids, r))
        return combos

    def _entries_sorted(self) -> list[_Entry]:
        entries: list[_Entry] = []
        entries.extend(self._manual_entries)
        combos = self._combo_keys()
        ratio_map = {f.id: f.ratio for f in self._filters}
        for base in self._base_points:
            entries.append(_Entry(
                kind="base",
                base_id=base.id,
                combo=(),
                series="base",
                position_deg=base.position_deg,
                power_w=base.power_w,
                positions=base.positions,
                primary_label=base.primary_label,
            ))
            for combo in combos:
                if (base.id, combo) in self._excluded_combos:
                    continue
                ratio = 1.0
                for fid in combo:
                    ratio *= ratio_map.get(fid, 1.0)
                if ratio <= 0:
                    continue
                power_w = base.power_w / ratio
                entries.append(_Entry(
                    kind="nd",
                    base_id=base.id,
                    combo=combo,
                    series=self._series_label(combo),
                    position_deg=base.position_deg,
                    power_w=power_w,
                    positions=base.positions,
                    primary_label=base.primary_label,
                ))
        entries.sort(key=lambda e: e.power_w)
        return entries

    def _refresh_table(self, entries: list[_Entry]) -> None:
        scale = self._unit_scale()
        unit = self._unit_label()
        position_labels = list(self._position_labels)
        if not position_labels:
            position_labels = ["position"]
        headers = ["Series"]
        if len(position_labels) == 1 and position_labels[0] == "position":
            headers.append("Position (deg)")
        else:
            headers.extend([f"{label} (deg)" for label in position_labels])
        headers.append(f"Power ({unit})")
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            series_item = QtWidgets.QTableWidgetItem(entry.series)
            series_item.setData(QtCore.Qt.UserRole, entry)
            self.table.setItem(row, 0, series_item)
            col = 1
            for label in position_labels:
                val = entry.positions.get(label)
                if val is None and label == entry.primary_label:
                    val = entry.position_deg
                if val is None:
                    pos_item = QtWidgets.QTableWidgetItem("")
                else:
                    pos_item = QtWidgets.QTableWidgetItem(f"{float(val):.3f}")
                self.table.setItem(row, col, pos_item)
                col += 1
            power_item = QtWidgets.QTableWidgetItem(f"{entry.power_w / scale:.6g} {unit}")
            self.table.setItem(row, col, power_item)
        self.table.resizeColumnsToContents()

    def _refresh_plot(self, entries: list[_Entry]) -> None:
        scale = self._unit_scale()
        unit = self._unit_label()
        indexed = [(idx + 1, e) for idx, e in enumerate(entries)]
        series_names = []
        for _, entry in indexed:
            if entry.series not in series_names:
                series_names.append(entry.series)

        color_map = {}
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        for idx, name in enumerate(series_names):
            color_map[name] = colors[idx % len(colors)]

        self.ax.clear()
        self.ax.set_xlabel("Index (sorted)")
        self.ax.set_ylabel(f"Power ({unit})")
        for name in series_names:
            xs = [idx for idx, e in indexed if e.series == name]
            ys = [e.power_w / scale for _, e in indexed if e.series == name]
            if xs:
                kwargs = {"s": 20, "label": name, "color": color_map[name]}
                self.ax.scatter(xs, ys, **kwargs)
        self.ax.grid(True, alpha=0.2)
        if len(series_names) > 1:
            self.ax.legend(fontsize=8, loc="best")
        self.canvas.draw_idle()

    def _refresh_filters(self) -> None:
        self.ndTable.setRowCount(len(self._filters))
        for row, f in enumerate(self._filters):
            name_item = QtWidgets.QTableWidgetItem(f"nd #{row + 1}")
            ratio_item = QtWidgets.QTableWidgetItem(f"{f.ratio:.6g}")
            name_item.setData(QtCore.Qt.UserRole, f.id)
            self.ndTable.setItem(row, 0, name_item)
            self.ndTable.setItem(row, 1, ratio_item)
        self.ndTable.resizeColumnsToContents()

    def _refresh(self) -> None:
        entries = self._entries_sorted()
        self._refresh_filters()
        self._refresh_table(entries)
        self._refresh_plot(entries)

    def on_record(self):
        try:
            pos, positions, primary_label = self._get_positions()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Position", f"Cannot read position: {e}")
            return

        scale = self._unit_scale()
        power_w = float(self.powerSpin.value()) * scale
        if power_w <= 0:
            QtWidgets.QMessageBox.warning(self, "Power", "Enter a positive power.")
            return

        self._update_position_labels(positions, primary_label)
        point = _BasePoint(self._next_point_id, pos, power_w, positions, primary_label)
        self._next_point_id += 1
        self._base_points.append(point)
        self._refresh()

    def on_add_nd(self):
        if len(self._filters) >= 4:
            QtWidgets.QMessageBox.information(self, "ND Filters", "Maximum 4 ND filters.")
            return
        ratio = float(self.ndRatioSpin.value())
        if ratio <= 0:
            QtWidgets.QMessageBox.warning(self, "ND Filters", "Ratio must be positive.")
            return
        self._filters.append(_Filter(self._next_filter_id, ratio))
        self._next_filter_id += 1
        self._refresh()

    def on_remove_nd(self):
        row = self.ndTable.currentRow()
        if row < 0 or row >= len(self._filters):
            return
        filt = self._filters.pop(row)
        self._excluded_combos = {
            (base_id, combo)
            for base_id, combo in self._excluded_combos
            if filt.id not in combo
        }
        self._refresh()

    def on_delete_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        entries = self._entries_sorted()
        for row in rows:
            if row < 0 or row >= len(entries):
                continue
            entry = entries[row]
            if entry.kind == "base":
                self._base_points = [p for p in self._base_points if p.id != entry.base_id]
                self._excluded_combos = {
                    (base_id, combo) for base_id, combo in self._excluded_combos
                    if base_id != entry.base_id
                }
            elif entry.kind == "manual":
                try:
                    self._manual_entries.remove(entry)
                except ValueError:
                    pass
            else:
                self._excluded_combos.add((entry.base_id, entry.combo))
        self._refresh()

    def on_clear_all(self):
        self._base_points.clear()
        self._manual_entries.clear()
        self._excluded_combos.clear()
        self._refresh()

    def on_export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", filter="CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        entries = self._entries_sorted()
        scale = self._unit_scale()
        position_labels = list(self._position_labels)
        if not position_labels:
            position_labels = ["position"]
        single_position = len(position_labels) == 1 and position_labels[0] == "position"
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                for idx, filt in enumerate(self._filters, start=1):
                    writer.writerow(["#nd_filter", f"nd #{idx}", f"{filt.ratio:.9g}"])
                if single_position:
                    header = ["position_deg"]
                else:
                    header = [f"position_{label.lower().replace(' ', '_')}_deg" for label in position_labels]
                header.append(f"power_{self._unit_label()}")
                header.append("series")
                writer.writerow(header)
                for entry in entries:
                    row = []
                    for label in position_labels:
                        val = entry.positions.get(label)
                        if val is None and label == entry.primary_label:
                            val = entry.position_deg
                        row.append("" if val is None else f"{float(val):.6f}")
                    row.append(f"{entry.power_w / scale:.9g}")
                    row.append(entry.series)
                    writer.writerow(row)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", f"Failed to write CSV:\n{e}")

    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load CSV", filter="CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if row]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to read CSV:\n{e}")
            return

        if not rows:
            return

        nd_meta: list[float] = []
        data_start = 0
        while data_start < len(rows):
            row = rows[data_start]
            if row and row[0].strip().lower() == "#nd_filter":
                if len(row) >= 3:
                    try:
                        nd_meta.append(float(row[2]))
                    except Exception:
                        pass
                data_start += 1
                continue
            break

        header = rows[data_start] if data_start < len(rows) else []
        has_header = any("power" in str(cell).lower() for cell in header)
        data_rows = rows[data_start + 1:] if has_header else rows[data_start:]

        unit = None
        pos_cols: list[int] = []
        power_col = 1
        series_col = None
        pos_labels: list[str] = []
        if has_header:
            for idx, cell in enumerate(header):
                low = str(cell).lower()
                if "power" in low:
                    power_col = idx
                    if "nw" in low:
                        unit = "nW"
                    elif "uw" in low:
                        unit = "uW"
                elif "series" in low:
                    series_col = idx
                elif "position" in low or low.startswith("pos"):
                    pos_cols.append(idx)
                    label = low.replace("position", "").replace("pos", "")
                    label = label.replace("_deg", "").replace("deg", "").replace("_", " ").strip()
                    if not label:
                        pos_labels.append("position")
                    else:
                        pos_labels.append(" ".join(label.split()).title())
            if not pos_cols:
                pos_cols = [0]
                pos_labels = ["position"]
            if series_col is None and len(header) > power_col + 1:
                series_col = power_col + 1
        else:
            pos_cols = [0]
            pos_labels = ["position"]
            power_col = 1
            series_col = 2 if len(rows[0]) > 2 else None

        if unit:
            self.unitCombo.setCurrentText(unit)
        scale = self._unit_scale()

        self._base_points.clear()
        self._manual_entries.clear()
        self._excluded_combos.clear()
        self._filters.clear()
        self._position_labels = pos_labels[:] if pos_labels else []
        self._next_point_id = 1
        self._next_filter_id = 1

        for ratio in nd_meta:
            if ratio > 0:
                self._filters.append(_Filter(self._next_filter_id, ratio))
                self._next_filter_id += 1

        base_rows = []
        other_rows = []
        for row in data_rows:
            if len(row) <= power_col:
                continue
            try:
                positions = {}
                for col, label in zip(pos_cols, pos_labels):
                    if col < len(row) and row[col] != "":
                        positions[label] = float(row[col])
                if not positions:
                    continue
                primary_label = pos_labels[0]
                pos = positions.get(primary_label, next(iter(positions.values())))
                power = float(row[power_col])
            except Exception:
                continue
            if series_col is not None and series_col < len(row):
                series = row[series_col].strip()
            else:
                series = "base"
            if not series:
                series = "base"
            if series == "base":
                base_rows.append((pos, power, positions, primary_label))
            else:
                other_rows.append((pos, power, series, positions, primary_label))

        for pos, power, positions, primary_label in base_rows:
            self._update_position_labels(positions, primary_label)
            self._base_points.append(_BasePoint(self._next_point_id, pos, power * scale, positions, primary_label))
            self._next_point_id += 1

        combo_map = {}
        if self._filters:
            for combo in self._combo_keys():
                combo_map[self._series_label(combo)] = combo

        ratio_map = {f.id: f.ratio for f in self._filters}
        present_combos: dict[int, set[tuple[int, ...]]] = {p.id: set() for p in self._base_points}
        found_nd_entries = False

        for pos, power, series, positions, primary_label in other_rows:
            self._update_position_labels(positions, primary_label)
            combo = combo_map.get(series)
            if combo and self._filters:
                found_nd_entries = True
                ratio = 1.0
                for fid in combo:
                    ratio *= ratio_map.get(fid, 1.0)
                expected_base = power * ratio
                best_id = None
                best_err = None
                for base in self._base_points:
                    if abs(base.position_deg - pos) > 1e-6:
                        continue
                    err = abs((base.power_w / scale) - expected_base)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_id = base.id
                if best_id is not None:
                    present_combos[best_id].add(combo)
                    continue

            self._manual_entries.append(_Entry(
                kind="manual",
                base_id=0,
                combo=(),
                series=series,
                position_deg=pos,
                power_w=power * scale,
                positions=positions,
                primary_label=primary_label,
            ))

        if found_nd_entries:
            all_combos = set(self._combo_keys())
            for base in self._base_points:
                for combo in all_combos:
                    if combo not in present_combos.get(base.id, set()):
                        self._excluded_combos.add((base.id, combo))

        self._refresh()

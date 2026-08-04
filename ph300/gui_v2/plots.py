import math
from typing import Dict, Optional

import pyqtgraph as pg
from PyQt5 import QtWidgets

_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#e377c2",  # pink
]


class HistogramPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.pw = pg.PlotWidget(background="w")
        self.pw.setLabel("bottom", "Time (ps)")
        self.pw.setLabel("left", "Counts")
        self.pw.showGrid(x=True, y=True, alpha=0.3)
        self.pw.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        for ax in ("bottom", "left"):
            self.pw.getAxis(ax).setPen(pg.mkPen("k"))
            self.pw.getAxis(ax).setTextPen(pg.mkPen("k"))
        self._legend = self.pw.addLegend(offset=(10, 10))

        vbox.addWidget(self.pw)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._live_curve: Optional[pg.PlotDataItem] = None
        self._color_idx = 0
        self._logy = False

    def _next_color(self) -> str:
        c = _COLORS[self._color_idx % len(_COLORS)]
        self._color_idx += 1
        return c

    def add_trace(self, label: str, time_ps, counts):
        if label in self._curves:
            self._curves[label].setData(x=time_ps, y=counts.astype(float))
            return
        color = self._next_color()
        curve = self.pw.plot(x=time_ps, y=counts.astype(float),
                             pen=pg.mkPen(color, width=1.2), name=label)
        self._curves[label] = curve

    def update_live(self, time_ps, counts):
        y = counts.astype(float)
        if self._live_curve is None:
            self._live_curve = self.pw.plot(x=time_ps, y=y,
                                            pen=pg.mkPen("#333333", width=1.8), name="LIVE")
        else:
            self._live_curve.setData(x=time_ps, y=y)

    def remove_live(self):
        if self._live_curve is not None:
            self.pw.removeItem(self._live_curve)
            self._live_curve = None

    def remove_trace(self, label: str):
        if label in self._curves:
            self.pw.removeItem(self._curves[label])
            del self._curves[label]

    def set_visible(self, label: str, visible: bool):
        if label in self._curves:
            self._curves[label].setVisible(visible)

    def clear_all(self):
        for c in list(self._curves.values()):
            self.pw.removeItem(c)
        self._curves.clear()
        self._color_idx = 0
        self.remove_live()

    def apply_range(self, xmin: float, xmax: float, ymin: float, ymax: float, logy: bool):
        if logy != self._logy:
            self.pw.setLogMode(x=False, y=logy)
            self._logy = logy

        if xmax > xmin:
            self.pw.setXRange(xmin, xmax, padding=0)

        if ymax > ymin:
            if logy:
                ymin_safe = max(ymin, 1e-3)
                self.pw.setYRange(math.log10(ymin_safe), math.log10(ymax), padding=0)
            else:
                self.pw.setYRange(ymin, ymax, padding=0)

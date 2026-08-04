from collections import deque

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .config import PLOT_MAX_POINTS


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 2.6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.canvas)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.axV = self.fig.add_subplot(111)
        self.axI = self.axV.twinx()
        self.axV.set_xlabel("Samples")
        self.axV.set_ylabel("V (V)")
        self.axI.set_ylabel("I (nA)")

        self.lineV, = self.axV.plot([], [], lw=1.5)
        self.lineI, = self.axI.plot([], [], lw=1.2, linestyle="--")

        self.vs = deque(maxlen=PLOT_MAX_POINTS)
        self.is_nA = deque(maxlen=PLOT_MAX_POINTS)

        self.fig.tight_layout()

    def add_point(self, v_volts: float, i_amps: float):
        self.vs.append(v_volts)
        self.is_nA.append(i_amps * 1e9)
        self._redraw()

    def clear(self):
        self.vs.clear()
        self.is_nA.clear()
        self._redraw()

    def _redraw(self):
        n = len(self.vs)
        self.lineV.set_data(range(n), list(self.vs))
        self.lineI.set_data(range(n), list(self.is_nA))

        self.axV.relim()
        self.axV.autoscale_view()
        self.axI.relim()
        self.axI.autoscale_view()

        self.axV.set_xlim(0, max(1, n - 1))
        self.canvas.draw_idle()

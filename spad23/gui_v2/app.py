import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from PyQt5 import QtGui, QtWidgets  # PyQt5 must come before pyqtgraph
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k', useOpenGL=False)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        from .widget import Spad23TrplWidget
        self.setWindowTitle("SPAD23 TRPL Viewer v2")
        self.resize(1800, 980)
        self._widget = Spad23TrplWidget()
        self.setCentralWidget(self._widget)

    def load_csv_path(self, path: str) -> None:
        self._widget.load_csv_path(path)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    win = MainWindow()
    win.show()
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        win.load_csv_path(args[0])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

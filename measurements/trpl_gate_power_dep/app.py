import sys
import os

# Allow running as a script inside the package directory
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from PyQt5 import QtWidgets
from .gui import TRPLGatePowerWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRPL — Gate & Power Dependent (ND wheel + 2× Keithley + PH300)")
        self.resize(1400, 860)
        self.widget = TRPLGatePowerWidget()
        self.setCentralWidget(self.widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

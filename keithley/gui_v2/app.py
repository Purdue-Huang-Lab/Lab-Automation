import sys

from PyQt5 import QtWidgets

from .dual_widget import DualKeithleyWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Keithley Controller v2 (Safe ramps, microstep plotting for ramp/home)")
        self.resize(1320, 760)

        self.widget = DualKeithleyWidget()
        self.setCentralWidget(self.widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

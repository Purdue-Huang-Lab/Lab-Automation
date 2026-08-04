import sys

from PyQt5 import QtWidgets

from measurements.pl_trpl_power_voltage_widget import PLTRPLPowerVoltageWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power/Voltage Dependent PL + TRPL")
        self.resize(1700, 950)
        self.widget = PLTRPLPowerVoltageWidget()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        try:
            self.widget.on_abort()
            self.widget.on_disconnect()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

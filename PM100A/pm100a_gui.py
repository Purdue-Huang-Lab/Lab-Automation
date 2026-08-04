import sys

from PyQt5 import QtWidgets

from PM100A.gui.pm100a_widget import PM100AWidget


class PM100AMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thorlabs PM100A Power Meter")
        self._widget = PM100AWidget()
        self.setCentralWidget(self._widget)
        self.resize(780, 430)

    def closeEvent(self, event):
        self._widget._on_disconnect()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PM100AMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

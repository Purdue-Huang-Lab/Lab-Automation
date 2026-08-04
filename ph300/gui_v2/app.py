import sys

from PyQt5 import QtWidgets

from .widget import PH300Widget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PicoHarp 300 TRPL v2 (Histogramming)")
        self.resize(1280, 760)

        self.widget = PH300Widget()
        self.setCentralWidget(self.widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

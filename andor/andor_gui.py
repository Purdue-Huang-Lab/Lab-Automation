import sys

from PyQt5 import QtWidgets

from andor.gui.andor_widget import AndorWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Andor Controller")
        self.resize(1400, 900)
        self.widget = AndorWidget()
        self.setCentralWidget(self.widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

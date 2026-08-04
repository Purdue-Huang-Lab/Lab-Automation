import sys

from PyQt5 import QtWidgets

from rot.gui.multi_widget import RotationStageWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rotation Stage Controller")
        self.resize(1000, 700)

        self.widget = RotationStageWidget()
        self.setCentralWidget(self.widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

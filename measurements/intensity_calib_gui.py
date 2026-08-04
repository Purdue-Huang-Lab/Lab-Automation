import sys

from PyQt5 import QtWidgets

from measurements.intensity_calib_widget import IntensityCalibWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Intensity Power Calibration")
        self.resize(1500, 900)
        self.widget = IntensityCalibWidget()
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

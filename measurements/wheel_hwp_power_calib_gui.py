import sys

from PyQt5 import QtWidgets

from measurements.wheel_hwp_power_calib_widget import WheelHWPCalibWidget


class WheelHWPCalibWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel + HWP Intensity Calibration")
        self._widget = WheelHWPCalibWidget()
        self.setCentralWidget(self._widget)
        self.resize(1100, 700)

    def closeEvent(self, event):
        # Stop any running sweep and disconnect devices
        w = self._widget
        if w._running and w._worker:
            w._worker.stop()
            w._worker.wait(3000)
        for disconnect in (w._disconnect_nd, w._disconnect_hwp, w._disconnect_pm):
            try:
                disconnect()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = WheelHWPCalibWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

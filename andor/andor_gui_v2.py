import sys

# PyQt5 must be imported before pyqtgraph so pyqtgraph detects it in sys.modules
# and uses it as its Qt binding — otherwise the widget types are incompatible.
from PyQt5 import QtWidgets

import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k')

from andor.gui_v2.andor_widget_v2 import AndorWidgetV2


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Andor Controller v2")
        self.resize(1400, 900)
        self.widget = AndorWidgetV2()
        self.setCentralWidget(self.widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

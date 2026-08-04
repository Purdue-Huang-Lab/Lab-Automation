from typing import Optional

from PyQt5 import QtCore, QtWidgets

from rot.rot_wrapper import list_kinesis_serials

from .config import MAX_MOTORS, POLL_MS_DEFAULT, BTN_W
from .device_panel import RotationStagePanel


class RotationStageWidget(QtWidgets.QWidget):
    def __init__(self, max_motors: int = MAX_MOTORS, parent=None):
        super().__init__(parent)
        self.max_motors = int(max_motors)
        self.panels: list[RotationStagePanel] = []

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        vbox.addLayout(top)

        self.detectBtn = QtWidgets.QPushButton("Detect Motors")
        self.detectBtn.setFixedWidth(150)
        self.addBtn = QtWidgets.QPushButton("Add Motor"); self.addBtn.setFixedWidth(BTN_W)
        self.removeBtn = QtWidgets.QPushButton("Remove Motor"); self.removeBtn.setFixedWidth(BTN_W)
        self.stopAllBtn = QtWidgets.QPushButton("Stop All"); self.stopAllBtn.setFixedWidth(BTN_W)

        self.pollSpin = QtWidgets.QSpinBox()
        self.pollSpin.setRange(100, 2000)
        self.pollSpin.setSingleStep(50)
        self.pollSpin.setValue(POLL_MS_DEFAULT)

        top.addWidget(self.detectBtn)
        top.addSpacing(8)
        top.addWidget(self.addBtn)
        top.addWidget(self.removeBtn)
        top.addSpacing(8)
        top.addWidget(self.stopAllBtn)
        top.addStretch()
        top.addWidget(QtWidgets.QLabel("Poll (ms):"))
        top.addWidget(self.pollSpin)

        self.panelBox = QtWidgets.QVBoxLayout()
        vbox.addLayout(self.panelBox)

        self.statusLbl = QtWidgets.QLabel("Idle")
        vbox.addWidget(self.statusLbl)

        # Timer for polling
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_poll)
        self.timer.start(self.pollSpin.value())

        # Signals
        self.detectBtn.clicked.connect(self.on_detect)
        self.addBtn.clicked.connect(self.on_add_panel)
        self.removeBtn.clicked.connect(self.on_remove_panel)
        self.stopAllBtn.clicked.connect(self.on_stop_all)
        self.pollSpin.valueChanged.connect(self.on_poll_interval_changed)

        self.on_add_panel()

    def on_add_panel(self):
        if len(self.panels) >= self.max_motors:
            self.statusLbl.setText(f"Max motors reached ({self.max_motors})")
            return
        panel = RotationStagePanel(f"Motor {len(self.panels) + 1}")
        self.panels.append(panel)
        self.panelBox.addWidget(panel)
        self.statusLbl.setText("Panel added")

    def on_remove_panel(self):
        if not self.panels:
            return
        panel = self.panels.pop()
        panel.on_disconnect()
        panel.setParent(None)
        panel.deleteLater()
        self.statusLbl.setText("Panel removed")

    def on_detect(self):
        try:
            serials = list_kinesis_serials()
            for panel in self.panels:
                panel.set_serials(serials)
            self.statusLbl.setText(f"Detected {len(serials)} motor(s)")
        except Exception as e:
            self.statusLbl.setText(f"Detect error: {e}")

    def on_stop_all(self):
        for panel in self.panels:
            panel.on_stop_any()
        self.statusLbl.setText("Stop sent to all motors")

    def on_poll_interval_changed(self, ms: int):
        if self.timer.isActive():
            self.timer.setInterval(ms)
        self.statusLbl.setText(f"Polling @ {ms} ms")

    def on_poll(self):
        for panel in self.panels:
            panel.poll_once()

import sys
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QGroupBox, QMessageBox, QFormLayout)
from PySide6.QtCore import QThread, Signal, Slot, Qt

import MultiPyVu as mpv

class OptiCoolWorker(QThread):
    """
    A worker thread to handle all MultiPyVu network communications 
    so the main GUI does not freeze during timeouts or slow polling.
    """
    # Signals to communicate back to the GUI
    connection_status = Signal(bool, str) # is_connected, message
    status_updated = Signal(float, str, float, str) # temp, temp_stat, field, field_stat
    
    def __init__(self, host_ip):
        super().__init__()
        self.host_ip = host_ip
        self.running = True
        self.client = None
        
        # Queues for sending commands to the OptiCool
        self.target_temp = None
        self.target_field = None
        
    def run(self):
        try:
            self.connection_status.emit(False, f"Connecting to {self.host_ip}...")
            # We don't use the 'with' context manager here so we can keep the socket open
            self.client = mpv.Client(host=self.host_ip)
            self.client.open()
            
            self.connection_status.emit(True, "Connected!")
            
            # Polling loop
            while self.running:
                # 1. Process Set Commands if any are queued
                if self.target_temp is not None:
                    sp, rate = self.target_temp
                    self.client.set_temperature(
                        set_point=sp,
                        rate_per_min=rate,
                        approach_mode=self.client.temperature.approach_mode.fast_settle
                    )
                    self.target_temp = None # clear queue
                    
                if self.target_field is not None:
                    sp, rate = self.target_field
                    self.client.set_field(
                        set_point=sp,
                        rate_per_sec=rate,
                        approach_mode=self.client.field.approach_mode.linear
                    )
                    self.target_field = None # clear queue

                # 2. Poll Current Status
                try:
                    t, t_stat = self.client.get_temperature()
                    f, f_stat = self.client.get_field()
                    self.status_updated.emit(t, t_stat, f, f_stat)
                except Exception as e:
                    print(f"Polling error: {e}")
                
                # Sleep briefly to avoid hammering the server
                time.sleep(1.0)
                
        except Exception as e:
            self.connection_status.emit(False, f"Error: {str(e)}\nMake sure server is running.")
        finally:
            if self.client:
                self.client.close_client()
            self.connection_status.emit(False, "Disconnected.")

    def stop(self):
        self.running = False
        
    def queue_set_temperature(self, temp, rate):
        self.target_temp = (temp, rate)
        
    def queue_set_field(self, field, rate):
        self.target_field = (field, rate)


class OptiCoolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptiCool Remote Control")
        self.resize(450, 450)
        
        self.worker = None

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ==========================================
        # Connection Group
        # ==========================================
        conn_group = QGroupBox("Network Connection")
        conn_layout = QHBoxLayout()
        self.ip_input = QLineEdit("10.164.14.243")
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setCheckable(True)
        self.btn_connect.clicked.connect(self.toggle_connection)
        
        conn_layout.addWidget(QLabel("Main PC IP:"))
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(self.btn_connect)
        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)
        
        self.lbl_conn_status = QLabel("Status: Disconnected")
        self.lbl_conn_status.setStyleSheet("color: gray; font-weight: bold;")
        main_layout.addWidget(self.lbl_conn_status)

        # ==========================================
        # Live Status Group
        # ==========================================
        status_group = QGroupBox("Live Status")
        status_layout = QFormLayout()
        
        self.lbl_temp = QLabel("--- K")
        self.lbl_temp.setStyleSheet("font-size: 18px; color: blue;")
        self.lbl_temp_stat = QLabel("Wait")
        
        self.lbl_field = QLabel("--- Oe")
        self.lbl_field.setStyleSheet("font-size: 18px; color: red;")
        self.lbl_field_stat = QLabel("Wait")
        
        status_layout.addRow(QLabel("Temperature:"), self.lbl_temp)
        status_layout.addRow(QLabel("Temp State:"), self.lbl_temp_stat)
        status_layout.addRow(QLabel("Magnetic Field:"), self.lbl_field)
        status_layout.addRow(QLabel("Field State:"), self.lbl_field_stat)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # ==========================================
        # Control Group
        # ==========================================
        control_group = QGroupBox("Instrument Control")
        control_layout = QFormLayout()
        
        # Temp
        self.inp_target_temp = QLineEdit("300.0")
        self.inp_temp_rate = QLineEdit("10.0")
        self.btn_set_temp = QPushButton("Set Temperature")
        self.btn_set_temp.setEnabled(False)
        self.btn_set_temp.clicked.connect(self.cmd_set_temperature)
        
        control_layout.addRow(QLabel("Target Temp (K):"), self.inp_target_temp)
        control_layout.addRow(QLabel("Ramp Rate (K/min):"), self.inp_temp_rate)
        control_layout.addRow("", self.btn_set_temp)
        
        # Field
        control_layout.addRow(QLabel(""), QLabel("")) # spacing
        self.inp_target_field = QLineEdit("0.0")
        self.inp_field_rate = QLineEdit("10.0")
        self.btn_set_field = QPushButton("Set Magnetic Field")
        self.btn_set_field.setEnabled(False)
        self.btn_set_field.setStyleSheet("background-color: #ffd6d6;")
        self.btn_set_field.clicked.connect(self.cmd_set_field)
        
        control_layout.addRow(QLabel("Target Field (Oe):"), self.inp_target_field)
        control_layout.addRow(QLabel("Ramp Rate (Oe/sec):"), self.inp_field_rate)
        control_layout.addRow("", self.btn_set_field)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        main_layout.addStretch()

    @Slot()
    def toggle_connection(self):
        if self.btn_connect.isChecked():
            # Connect
            ip = self.ip_input.text().strip()
            self.btn_connect.setText("Disconnect")
            self.ip_input.setEnabled(False)
            
            # Start Worker
            self.worker = OptiCoolWorker(ip)
            self.worker.connection_status.connect(self.on_connection_status)
            self.worker.status_updated.connect(self.on_status_updated)
            self.worker.start()
        else:
            # Disconnect
            self.btn_connect.setText("Connect")
            self.ip_input.setEnabled(True)
            self.btn_set_temp.setEnabled(False)
            self.btn_set_field.setEnabled(False)
            if self.worker:
                self.worker.stop()
                self.worker.wait(2000) # wait up to 2 seconds for clean shutdown
                self.worker = None

    @Slot(bool, str)
    def on_connection_status(self, is_connected, message):
        self.lbl_conn_status.setText(f"Status: {message}")
        if is_connected:
            self.lbl_conn_status.setStyleSheet("color: green; font-weight: bold;")
            self.btn_set_temp.setEnabled(True)
            self.btn_set_field.setEnabled(True)
        else:
            self.lbl_conn_status.setStyleSheet("color: red; font-weight: bold;")
            self.btn_connect.setChecked(False)
            self.btn_connect.setText("Connect")
            self.ip_input.setEnabled(True)
            self.btn_set_temp.setEnabled(False)
            self.btn_set_field.setEnabled(False)

    @Slot(float, str, float, str)
    def on_status_updated(self, temp, tstat, field, fstat):
        self.lbl_temp.setText(f"{temp:.3f} K")
        self.lbl_temp_stat.setText(tstat)
        self.lbl_field.setText(f"{field:.1f} Oe")
        self.lbl_field_stat.setText(fstat)
        
    def closeEvent(self, event):
        """Called automatically when the user clicks the 'X' to close the window."""
        if self.worker:
            print("Safely disconnecting from OptiCool Server before closing...")
            self.worker.stop()
            self.worker.wait(2000) # wait up to 2 seconds for a clean disconnect
        event.accept()
        
    @Slot()
    def cmd_set_temperature(self):
        try:
            target = float(self.inp_target_temp.text())
            rate = float(self.inp_temp_rate.text())
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Temperature target and rate must be numbers.")
            return

        reply = QMessageBox.question(self, 'Confirm Temperature Change', 
                                     f"Are you sure you want to ramp to {target} K at {rate} K/min?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes and self.worker:
            self.worker.queue_set_temperature(target, rate)

    @Slot()
    def cmd_set_field(self):
        try:
            target = float(self.inp_target_field.text())
            rate = float(self.inp_field_rate.text())
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Field target and rate must be numbers.")
            return

        reply = QMessageBox.warning(self, 'SAFETY WARNING: Confirm Field Change', 
                                    f"WARNING: Confirm magnet and sample limits.\n"
                                    f"Are you sure you want to ramp to {target} Oe at {rate} Oe/sec?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes and self.worker:
            self.worker.queue_set_field(target, rate)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptiCoolGUI()
    window.show()
    sys.exit(app.exec())

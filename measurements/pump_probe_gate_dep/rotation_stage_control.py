"""
rotation_stage_control.py
Rotation stage control module for Thorlabs rotation motor
Used for power control via half-wave plate rotation

Uses thorlabs_apt package, same pattern as stage_control.py.

Configuration:
    Edit DEFAULT_SERIAL_NUMBER below to match your rotation motor.
    Edit DEFAULT_CHANNEL if your motor uses a specific channel.
"""

import numpy as np
import time

# ============ CONFIGURATION ============
DEFAULT_SERIAL_NUMBER = 27600911  # Thorlabs rotation ND filter
DEFAULT_CHANNEL = 1            # Channel number if using a multi-channel controller
# =======================================

import os
apt_path = r"C:\Program Files\Thorlabs\APT\APT Server"
if apt_path not in os.environ['PATH']:
    os.environ['PATH'] = apt_path + os.pathsep + os.environ['PATH']
if hasattr(os, 'add_dll_directory') and os.path.exists(apt_path):
    os.add_dll_directory(apt_path)

try:
    import thorlabs_apt as apt
    ROTATION_APT_AVAILABLE = True
except (ImportError, OSError, FileNotFoundError) as e:
    ROTATION_APT_AVAILABLE = False
    print("=" * 60)
    print("Warning: thorlabs_apt not available for rotation stage")
    print(f"Reason: {e}")
    print("=" * 60)


class RotationStage:
    """
    Control class for a Thorlabs rotation stage / rotation motor.
    Provides absolute and relative angle moves, homing, and status.
    """

    def __init__(self, serial_number=DEFAULT_SERIAL_NUMBER, channel=DEFAULT_CHANNEL):
        self.serial_number = serial_number
        self.channel = channel
        self.motor = None
        self.is_initialized = False

    def initialize(self, serial_number=None):
        """
        Connect to the rotation stage.

        Args:
            serial_number: Override the serial number set in __init__.

        Returns:
            bool: True if successful.
        """
        if not ROTATION_APT_AVAILABLE:
            raise RuntimeError(
                "thorlabs_apt package not available!\n"
                "Install Thorlabs APT software and 'pip install thorlabs-apt'."
            )

        if serial_number is not None:
            self.serial_number = serial_number

        try:
            devices = apt.list_available_devices()
            print(f"Available APT devices: {devices}")

            if self.serial_number is None:
                if len(devices) == 0:
                    raise RuntimeError("No APT devices found!")
                self.serial_number = devices[0][1]
                print(f"Auto-selected device: {self.serial_number}")

            self.motor = apt.Motor(self.serial_number)
            if hasattr(self.motor, 'enable'):
                try:
                    self.motor.enable()
                except Exception as e:
                    print(f"Warning: Failed to enable motor ({e})")

            self.is_initialized = True

            print(f"Rotation stage initialized!")
            print(f"  Serial Number: {self.serial_number}")
            print(f"  Current Angle: {self.get_angle():.4f} deg")
            try:
                print(f"  Axis Info: {self.motor.get_stage_axis_info()}")
            except Exception:
                pass

            return True

        except Exception as e:
            self.is_initialized = False
            raise RuntimeError(f"Failed to initialize rotation stage: {e}")

    def get_angle(self):
        """Return current angle in degrees."""
        if not self.is_initialized:
            raise RuntimeError("Rotation stage not initialized!")
        return self.motor.position

    def move_to(self, angle_deg, wait=True):
        """Move to an absolute angle (degrees)."""
        if not self.is_initialized:
            raise RuntimeError("Rotation stage not initialized!")
        print(f"Rotating to {angle_deg:.4f} deg")
        self.motor.move_to(angle_deg, blocking=wait)
        if wait:
            print(f"Rotation complete. Angle: {self.get_angle():.4f} deg")

    def move_by(self, delta_deg, wait=True):
        """Move by a relative angle (degrees)."""
        if not self.is_initialized:
            raise RuntimeError("Rotation stage not initialized!")
        current = self.get_angle()
        target = current + delta_deg
        print(f"Rotating by {delta_deg:.4f} deg ({current:.4f} -> {target:.4f})")
        self.move_to(target, wait=wait)

    def home(self, wait=True):
        """Home the rotation stage."""
        if not self.is_initialized:
            raise RuntimeError("Rotation stage not initialized!")
        print("Homing rotation stage...")
        self.motor.move_home(blocking=wait)
        if wait:
            print(f"Home complete. Angle: {self.get_angle():.4f} deg")

    def is_moving(self):
        if not self.is_initialized:
            return False
        return self.motor.is_in_motion

    def stop(self):
        """Stop any ongoing rotation."""
        if self.is_initialized and self.motor:
            self.motor.stop_profiled()
            print("Rotation stage stopped")

    def close(self):
        """Close connection."""
        if self.motor and self.is_initialized:
            self.motor = None
            self.is_initialized = False
            print("Rotation stage connection closed")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    stage = RotationStage()
    try:
        stage.initialize()
        print(f"Angle: {stage.get_angle():.4f} deg")
        stage.move_to(45.0)
        stage.move_by(-10.0)
        stage.move_to(0.0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stage.close()

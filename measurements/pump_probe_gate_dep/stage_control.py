"""
stage_control.py
Delay stage control module for Thorlabs BBD301 controller
Using thorlabs_apt package for Python control

Configuration:
    BBD301 Serial Number: 103259148
    Channel: 1 (default, change if using channel 2 or 3)
"""

import numpy as np
import time

# ============ CONFIGURATION ============
DEFAULT_SERIAL_NUMBER = 104507475  # Your BBD301 serial number
DEFAULT_CHANNEL = 1  # Which channel your stage is connected to (1, 2, or 3)
# ======================================

import os
apt_path = r"C:\Program Files\Thorlabs\APT\APT Server"
if apt_path not in os.environ['PATH']:
    os.environ['PATH'] = apt_path + os.pathsep + os.environ['PATH']
if hasattr(os, 'add_dll_directory') and os.path.exists(apt_path):
    os.add_dll_directory(apt_path)

try:
    import thorlabs_apt as apt
    APT_AVAILABLE = True
except (ImportError, OSError, FileNotFoundError) as e:
    APT_AVAILABLE = False
    print("=" * 60)
    print("Warning: thorlabs_apt not available")
    print(f"Reason: {e}")
    print("")
    print("To use stage control, install Thorlabs APT software:")
    print("https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control")
    print("")
    print("After installation, the APT.dll will be available.")
    print("=" * 60)


class DelayStage:
    """
    Control class for Thorlabs BBD301 delay stage controller.
    
    Note: The BBD301 has 3 channels. Specify which channel your delay stage is on.
    
    For pump-probe measurements:
        Time delay (ps) = 4 × distance (mm) / speed_of_light
        Factor of 4 accounts for double-pass retroreflector configuration
    """
    
    # Speed of light in mm/ps
    SPEED_OF_LIGHT_MM_PS = 0.299792458  # mm/ps
    
    def __init__(self, serial_number=DEFAULT_SERIAL_NUMBER, channel=DEFAULT_CHANNEL):
        """
        Initialize delay stage controller.
        
        Args:
            serial_number: Serial number of BBD301 (default: 103259148)
            channel: Channel number (1, 2, or 3) for BBD301
        """
        self.serial_number = serial_number
        self.channel = channel
        self.motor = None
        self.is_initialized = False
    
    @staticmethod
    def mm_to_ps(distance_mm):
        """
        Convert stage position (mm) to time delay (ps).
        
        Args:
            distance_mm: Stage position in millimeters
            
        Returns:
            float: Time delay in picoseconds
        """
        return 4.0 * distance_mm / DelayStage.SPEED_OF_LIGHT_MM_PS
    
    @staticmethod
    def ps_to_mm(time_ps):
        """
        Convert time delay (ps) to stage position (mm).
        
        Args:
            time_ps: Time delay in picoseconds
            
        Returns:
            float: Stage position in millimeters
        """
        return time_ps * DelayStage.SPEED_OF_LIGHT_MM_PS / 4.0
        
    def initialize(self, serial_number=None):
        """
        Initialize connection to the delay stage.
        
        Args:
            serial_number: Optional serial number to use (overrides the one set in __init__)
        
        Returns:
            bool: True if successful
        """
        if not APT_AVAILABLE:
            raise RuntimeError("thorlabs_apt package not installed!")
        
        # Allow overriding serial number at initialization
        if serial_number is not None:
            self.serial_number = serial_number
        
        try:
            # List available devices
            devices = apt.list_available_devices()
            print(f"Available APT devices: {devices}")
            
            if self.serial_number is None:
                if len(devices) == 0:
                    raise RuntimeError("No APT devices found!")
                # Use first available device
                self.serial_number = devices[0][1]
                print(f"Using device: {self.serial_number}")
            
            # Connect to motor
            self.motor = apt.Motor(self.serial_number)
            # Enable the motor channel so that it can physically move
            if hasattr(self.motor, 'enable'):
                try:
                    self.motor.enable()
                except Exception as e:
                    print(f"Warning: Failed to enable motor ({e})")
                    
            self.is_initialized = True
            
            print(f"Stage initialized successfully!")
            print(f"  Serial Number: {self.serial_number}")
            print(f"  Current Position: {self.get_position():.4f} mm")
            print(f"  Position Range: {self.motor.get_stage_axis_info()}")
            
            return True
            
        except Exception as e:
            self.is_initialized = False
            raise RuntimeError(f"Failed to initialize stage: {e}")
    
    def get_position(self):
        """
        Get current stage position.
        
        Returns:
            float: Current position in mm
        """
        if not self.is_initialized:
            raise RuntimeError("Stage not initialized!")
        
        return self.motor.position
    
    def move_to(self, position_mm, wait=True):
        """
        Move stage to absolute position.
        
        Args:
            position_mm: Target position in mm
            wait: If True, wait for move to complete
        """
        if not self.is_initialized:
            raise RuntimeError("Stage not initialized!")
        
        print(f"Moving to position: {position_mm:.4f} mm")
        self.motor.move_to(position_mm, blocking=wait)
        
        if wait:
            print(f"Move complete. Current position: {self.get_position():.4f} mm")
    
    def move_by(self, distance_mm, wait=True):
        """
        Move stage by relative distance.
        
        Args:
            distance_mm: Distance to move in mm (positive or negative)
            wait: If True, wait for move to complete
        """
        if not self.is_initialized:
            raise RuntimeError("Stage not initialized!")
        
        current_pos = self.get_position()
        target_pos = current_pos + distance_mm
        print(f"Moving by {distance_mm:.4f} mm (from {current_pos:.4f} to {target_pos:.4f} mm)")
        self.move_to(target_pos, wait=wait)
    
    def home(self, wait=True):
        """
        Home the stage (move to home position).
        
        Args:
            wait: If True, wait for homing to complete
        """
        if not self.is_initialized:
            raise RuntimeError("Stage not initialized!")
        
        print("Homing stage...")
        self.motor.move_home(blocking=wait)
        
        if wait:
            print(f"Homing complete. Position: {self.get_position():.4f} mm")
    
    def set_velocity_params(self, max_velocity=None, acceleration=None):
        """
        Set velocity and acceleration parameters.
        
        Args:
            max_velocity: Maximum velocity in mm/s (None to keep current)
            acceleration: Acceleration in mm/s^2 (None to keep current)
        """
        if not self.is_initialized:
            raise RuntimeError("Stage not initialized!")
        
        current_vel = self.motor.get_velocity_parameters()
        print(f"Current velocity params: {current_vel}")
        
        if max_velocity is not None or acceleration is not None:
            if max_velocity is None:
                max_velocity = current_vel['maximum_velocity']
            if acceleration is None:
                acceleration = current_vel['acceleration']
            
            self.motor.set_velocity_parameters(
                max_velocity=max_velocity,
                acceleration=acceleration
            )
            print(f"Updated velocity params: max_vel={max_velocity}, accel={acceleration}")
    
    def is_moving(self):
        """
        Check if stage is currently moving.
        
        Returns:
            bool: True if moving
        """
        if not self.is_initialized:
            return False
        
        return self.motor.is_in_motion
    
    def stop(self):
        """Stop any ongoing motion."""
        if self.is_initialized and self.motor:
            self.motor.stop_profiled()
            print("Stage stopped")
    
    def close(self):
        """Close connection to the stage."""
        if self.motor and self.is_initialized:
            # Note: thorlabs_apt doesn't have explicit close method
            # Connection is closed when object is deleted
            self.motor = None
            self.is_initialized = False
            print("Stage connection closed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


# Example scan function for pump-probe measurements
def scan_delay_positions(stage, positions_mm, camera, capture_func):
    """
    Scan through delay positions and capture pump-probe data at each position.
    
    Args:
        stage: DelayStage instance
        positions_mm: List/array of delay positions in mm
        camera: Camera instance
        capture_func: Function to call at each position (receives position as arg)
                     Should return measurement results dict
    
    Returns:
        list: Results for each position
    """
    if not stage.is_initialized:
        raise RuntimeError("Stage not initialized!")
    
    results = []
    
    print(f"\n=== Starting delay scan: {len(positions_mm)} positions ===")
    
    for i, position in enumerate(positions_mm):
        print(f"\n--- Position {i+1}/{len(positions_mm)}: {position:.4f} mm ---")
        
        # Move to position
        stage.move_to(position, wait=True)
        
        # Wait for settling
        time.sleep(0.5)
        
        # Capture data at this position
        result = capture_func(position)
        results.append({
            'position_mm': position,
            'data': result
        })
        
        print(f"✓ Position {position:.4f} mm complete")
    
    print(f"\n=== Scan complete! ===")
    return results


if __name__ == "__main__":
    """Test the stage control"""
    
    # Initialize stage with your BBD301 serial number
    stage = DelayStage(serial_number=104259185, channel=1)
    
    try:
        stage.initialize()
        
        # Get current position
        print(f"\nCurrent position: {stage.get_position():.4f} mm")
        
        # Move to a position
        stage.move_to(5.0, wait=True)
        
        # Move by relative distance
        stage.move_by(1.0, wait=True)
        
        # Return to original position
        stage.move_to(0.0, wait=True)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stage.close()


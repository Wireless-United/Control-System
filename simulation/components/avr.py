"""
Automatic Voltage Regulator (AVR) component for grid simulation.
Implements simple AVR control logic.
"""

import time
from typing import Optional

class AVR:
    """Automatic Voltage Regulator for generator voltage control."""
    
    def __init__(self, avr_id: int, generator_id: int, 
                 voltage_setpoint: float = 1.0, 
                 kp: float = 5.0, ki: float = 0.5, kd: float = 0.1):
        """
        Initialize AVR controller.
        
        Args:
            avr_id: Unique identifier for AVR
            generator_id: ID of generator being controlled
            voltage_setpoint: Desired voltage setpoint (pu)
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
        """
        self.avr_id = avr_id
        self.generator_id = generator_id
        self.voltage_setpoint = voltage_setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Control variables
        self.voltage_measured = voltage_setpoint
        self.error = 0.0
        self.error_integral = 0.0
        self.error_prev = 0.0
        self.excitation_output = 1.0
        self.last_update_time = time.time()
        
        # Limits
        self.excitation_min = 0.5
        self.excitation_max = 1.5
        self.integral_limit = 0.5
        
    def update_measurement(self, voltage_measured: float):
        """Update voltage measurement."""
        self.voltage_measured = voltage_measured
        
    def set_voltage_setpoint(self, setpoint: float):
        """Set voltage setpoint."""
        self.voltage_setpoint = setpoint
        
    def calculate_control(self, dt: Optional[float] = None) -> float:
        """
        Calculate AVR control output using PID control.
        
        Args:
            dt: Time step (seconds). If None, calculated from system time.
            
        Returns:
            Excitation signal for generator
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Calculate error
        self.error = self.voltage_setpoint - self.voltage_measured
        
        # Proportional term
        p_term = self.kp * self.error
        
        # Integral term with windup protection
        self.error_integral += self.error * dt
        self.error_integral = max(-self.integral_limit, 
                                min(self.integral_limit, self.error_integral))
        i_term = self.ki * self.error_integral
        
        # Derivative term
        if dt > 0:
            error_derivative = (self.error - self.error_prev) / dt
        else:
            error_derivative = 0.0
        d_term = self.kd * error_derivative
        
        # Calculate total control output
        control_output = p_term + i_term + d_term
        
        # Convert to excitation signal (1.0 is nominal)
        self.excitation_output = 1.0 + control_output
        
        # Apply limits
        self.excitation_output = max(self.excitation_min, 
                                   min(self.excitation_max, self.excitation_output))
        
        # Store previous error for derivative calculation
        self.error_prev = self.error
        
        return self.excitation_output
    
    def reset_integral(self):
        """Reset integral term (useful when setpoint changes)."""
        self.error_integral = 0.0
        
    def get_status(self) -> dict:
        """Get AVR status information."""
        return {
            'avr_id': self.avr_id,
            'generator_id': self.generator_id,
            'voltage_setpoint': self.voltage_setpoint,
            'voltage_measured': self.voltage_measured,
            'error': self.error,
            'excitation_output': self.excitation_output,
            'error_integral': self.error_integral
        }
        
    def __str__(self):
        return (f"AVR {self.avr_id} for Gen {self.generator_id}: "
                f"Vref={self.voltage_setpoint:.3f}, Vmeas={self.voltage_measured:.3f}, "
                f"Error={self.error:.3f}, Exc={self.excitation_output:.3f}")

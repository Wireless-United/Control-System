"""
Bus component for grid simulation.
Represents electrical buses in the power system.
"""

class Bus:
    """Represents an electrical bus in the power system."""
    
    def __init__(self, bus_id: int, name: str, voltage_nominal: float = 1.0):
        """
        Initialize a bus.
        
        Args:
            bus_id: Unique identifier for the bus
            name: Name of the bus
            voltage_nominal: Nominal voltage in pu
        """
        self.bus_id = bus_id
        self.name = name
        self.voltage_nominal = voltage_nominal
        self.voltage_actual = voltage_nominal
        self.angle = 0.0
        
    def update_voltage(self, voltage: float, angle: float = 0.0):
        """Update bus voltage and angle."""
        self.voltage_actual = voltage
        self.angle = angle
        
    def get_voltage_deviation(self) -> float:
        """Get voltage deviation from nominal."""
        return self.voltage_actual - self.voltage_nominal
        
    def __str__(self):
        return f"Bus {self.bus_id} ({self.name}): {self.voltage_actual:.3f} pu"

"""
Generator component for grid simulation.
Represents generators with excitation control.
"""

class Generator:
    """Represents a generator in the power system."""
    
    def __init__(self, gen_id: int, bus_id: int, name: str, 
                 p_max: float = 100.0, q_max: float = 50.0):
        """
        Initialize a generator.
        
        Args:
            gen_id: Unique identifier for the generator
            bus_id: Bus where generator is connected
            name: Name of the generator
            p_max: Maximum active power (MW)
            q_max: Maximum reactive power (MVAR)
        """
        self.gen_id = gen_id
        self.bus_id = bus_id
        self.name = name
        self.p_max = p_max
        self.q_max = q_max
        self.p_output = 0.0
        self.q_output = 0.0
        self.excitation = 1.0  # Excitation level (pu)
        self.voltage_setpoint = 1.0  # Voltage setpoint (pu)
        
    def set_excitation(self, excitation: float):
        """Set generator excitation level."""
        # Limit excitation between 0.5 and 1.5 pu
        self.excitation = max(0.5, min(1.5, excitation))
        
    def set_power_output(self, p_mw: float, q_mvar: float):
        """Set generator power output."""
        self.p_output = min(p_mw, self.p_max)
        self.q_output = max(-self.q_max, min(q_mvar, self.q_max))
        
    def get_reactive_capability(self) -> tuple:
        """Get reactive power capability based on excitation."""
        # Higher excitation allows more reactive power
        q_factor = self.excitation
        return (-self.q_max * q_factor, self.q_max * q_factor)
        
    def __str__(self):
        return (f"Gen {self.gen_id} at Bus {self.bus_id}: "
                f"P={self.p_output:.1f}MW, Q={self.q_output:.1f}MVAR, "
                f"Exc={self.excitation:.3f}pu")

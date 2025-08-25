"""
Load component for grid simulation.
Represents electrical loads in the power system.
"""

class Load:
    """Represents an electrical load in the power system."""
    
    def __init__(self, load_id: int, bus_id: int, name: str,
                 p_mw: float = 10.0, q_mvar: float = 5.0):
        """
        Initialize a load.
        
        Args:
            load_id: Unique identifier for the load
            bus_id: Bus where load is connected
            name: Name of the load
            p_mw: Active power demand (MW)
            q_mvar: Reactive power demand (MVAR)
        """
        self.load_id = load_id
        self.bus_id = bus_id
        self.name = name
        self.p_base = p_mw
        self.q_base = q_mvar
        self.p_demand = p_mw
        self.q_demand = q_mvar
        self.load_factor = 1.0
        
    def set_demand(self, p_mw: float, q_mvar: float = None):
        """Set load demand."""
        self.p_demand = max(0, p_mw)
        if q_mvar is not None:
            self.q_demand = q_mvar
        else:
            # Maintain power factor if only P is specified
            if self.p_base > 0:
                pf_ratio = self.q_base / self.p_base
                self.q_demand = self.p_demand * pf_ratio
        
    def scale_load(self, factor: float):
        """Scale load by a factor."""
        self.load_factor = max(0, factor)
        self.p_demand = self.p_base * self.load_factor
        self.q_demand = self.q_base * self.load_factor
        
    def increase_demand(self, delta_p: float, delta_q: float = None):
        """Increase load demand by delta values."""
        new_p = self.p_demand + delta_p
        if delta_q is not None:
            new_q = self.q_demand + delta_q
        else:
            # Maintain power factor
            if self.p_demand > 0:
                pf_ratio = self.q_demand / self.p_demand
                new_q = new_p * pf_ratio
            else:
                new_q = self.q_demand
        
        self.set_demand(new_p, new_q)
        
    def __str__(self):
        return (f"Load {self.load_id} at Bus {self.bus_id}: "
                f"P={self.p_demand:.1f}MW, Q={self.q_demand:.1f}MVAR "
                f"(Factor: {self.load_factor:.2f})")

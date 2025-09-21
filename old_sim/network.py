import pandapower as pp
import pandapower.converter
import numpy as np
from typing import Callable, Optional

class IEEE39Network:
    def __init__(self):
        self.net = self._build_network()

    def _build_network(self):
        """Load IEEE-39 bus case and convert to pandapower network"""
        from pypower.case39 import case39
        ppc = case39()
        try:
            net = pp.converter.from_ppc(ppc, f_hz=60)  # always works

        except Exception:
            net = pp.converter.from_ppc(ppc, f_hz=60)
        return net

    def set_global_demand_factor(self, factor: float):
        """Set a global factor to scale all loads"""
        self.net.load.p_mw *= factor
        self.net.load.q_mvar *= factor

    def set_load_for_bus(self, bus_label: int, p_mw: Optional[float] = None, q_mvar: Optional[float] = None):
        """Override P/Q for a specific bus"""
        load_bus = self.net.load[self.net.load.bus == bus_label]
        if p_mw is not None: load_bus.p_mw = p_mw
        if q_mvar is not None: load_bus.q_mvar = q_mvar

    def add_battery_storage(self, bus_idx: int, p_mw: float, max_e_mwh: float = 100.0, min_e_mwh: float = 20.0, q_mvar: float = 0.0):
        """Add a battery storage system to the network"""
        return pp.create_storage(self.net, bus=bus_idx, p_mw=p_mw, max_e_mwh=max_e_mwh, min_e_mwh=min_e_mwh, q_mvar=q_mvar)
        
    def zip_load_model(self, bus_label: int, voltage: float, frequency: float):
        """ZIP load model that reacts to voltage and frequency variations"""
        Z_factor, I_factor, P_factor = 0.3, 0.3, 0.4  # Example coefficients
        base_load = self.net.load.loc[self.net.load.bus == bus_label, 'p_mw'].values[0]
        z_adjustment = Z_factor * (voltage ** 2)
        i_adjustment = I_factor * frequency
        p_adjustment = P_factor
        new_load = base_load * (1 + z_adjustment + i_adjustment + p_adjustment)
        self.set_load_for_bus(bus_label, new_load, self.net.load.loc[self.net.load.bus == bus_label, 'q_mvar'].values[0])

#!/usr/bin/env python3
"""
IoT Device Models for Load-Altering Attacks

This module defines various IoT device types that can be compromised
and controlled to execute coordinated load-altering attacks.

Author: Pranaav
Date: October 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

class IoTDeviceType(Enum):
    """Types of IoT devices"""
    SMART_THERMOSTAT = "smart_thermostat"
    SMART_WATER_HEATER = "smart_water_heater"
    SMART_EV_CHARGER = "smart_ev_charger"
    SMART_POOL_PUMP = "smart_pool_pump"
    SMART_APPLIANCE = "smart_appliance"#tv and fridge
    INDUSTRIAL_CONTROLLER = "industrial_controller"
    SMART_METER = "smart_meter"
    SMART_LIGHTING = "smart_lighting"

class IoTDeviceState(Enum):
    """Device operational states"""
    OFFLINE = 0
    IDLE = 1
    NORMAL_OPERATION = 2
    COMPROMISED = 3
    ATTACK_MODE = 4

@dataclass
class IoTDeviceParams:
    """Parameters for IoT device modeling"""
    device_id: str
    device_type: IoTDeviceType
    bus_location: int                    # IEEE 39-bus system bus number
    rated_power_kw: float               # Rated power consumption (kW)
    base_load_kw: float                 # Normal operating load (kW)
    controllable: bool = True           # Can load be controlled remotely?
    response_time: float = 1.0          # Response time to commands (seconds)
    duty_cycle: float = 0.5             # Typical duty cycle (0-1)
    vulnerability_score: float = 0.7    # Vulnerability to compromise (0-1)

class SmartThermostat:
    """
    Smart HVAC thermostat model for LAA attacks.
    
    Controls heating/cooling systems that represent significant
    residential and commercial loads.
    
    Base Power: 2 kW (scaled by N)
    """
    
    def __init__(self, device_id: str, bus_location: int, 
                 hvac_capacity_kw: float = 2.0, N: int = 1):
        """
        Initialize smart thermostat.
        
        Args:
            device_id: Unique device identifier
            bus_location: Bus number (1-39)
            hvac_capacity_kw: Base HVAC system capacity in kW (default 2kW)
            N: Scaling factor for number of devices (default 1)
        """
        self.device_id = device_id
        self.bus_location = bus_location
        self.N = N
        self.hvac_capacity_kw = hvac_capacity_kw * N  # Scale by N
        
        # State variables
        self.is_compromised = False
        self.current_setpoint = 21.0        # °C
        self.current_temperature = 21.0     # °C
        self.hvac_running = False       #iot device state by default set to false 
        self.power_consumption_kw = 0.0
        
        # Attack parameters
        self.attack_setpoint = None
        self.attack_mode = False
        
        # Net power tracking
        self.net_power_history = []
    
    def normal_operation(self, dt: float, ambient_temp: float = 20.0):
        """Normal thermostat operation"""
        # Simple thermal model
        temp_error = self.current_setpoint - self.current_temperature
        
        if abs(temp_error) > 0.5:
            self.hvac_running = True
            self.power_consumption_kw = self.hvac_capacity_kw
            
            # Update temperature
            if temp_error > 0:  # Heating
                self.current_temperature += 0.1 * dt
            else:  # Cooling
                self.current_temperature -= 0.1 * dt
        else:
            self.hvac_running = False
            self.power_consumption_kw = 0.0
        
        # Drift toward ambient
        self.current_temperature += (ambient_temp - self.current_temperature) * 0.01 * dt
        
        net_power_mw = self.power_consumption_kw / 1000.0  # Convert to MW
        self.net_power_history.append(net_power_mw)
        return net_power_mw
    
    def execute_attack(self, attack_setpoint: float):
        """
        Execute attack by changing setpoint.
        
        Args:
            attack_setpoint: Malicious temperature setpoint
        """
        if self.is_compromised:
            self.attack_mode = True
            self.current_setpoint = attack_setpoint
            self.hvac_running = True
            self.power_consumption_kw = self.hvac_capacity_kw
    
    def get_net_power_kw(self):
        """Get current net power consumption in kW"""
        return self.power_consumption_kw

class SmartWaterHeater:
    """
    Smart water heater model for LAA attacks.
    
    Represents high-power resistive loads that can be synchronized
    for coordinated attacks.
    
    Base Power: 3 kW (scaled by N)
    """
    
    def __init__(self, device_id: str, bus_location: int,
                 capacity_kw: float = 3.0, N: int = 1):
        """
        Initialize smart water heater.
        
        Args:
            device_id: Unique device identifier
            bus_location: Bus number (1-39)
            capacity_kw: Base water heater capacity in kW (default 3kW)
            N: Scaling factor for number of devices (default 1)
        """
        self.device_id = device_id
        self.bus_location = bus_location
        self.N = N
        self.capacity_kw = capacity_kw * N  # Scale by N
        
        # State variables
        self.is_compromised = False
        self.water_temp = 50.0          # °C
        self.setpoint = 60.0            # °C
        self.heating = False
        self.power_consumption_kw = 0.0
        
        # Attack parameters
        self.attack_mode = False
        
        # Net power tracking
        self.net_power_history = []
    
    def normal_operation(self, dt: float):
        """Normal water heater operation"""
        # Thermal losses
        self.water_temp -= 0.02 * dt
        
        # Heating control
        if self.water_temp < self.setpoint - 2.0:
            self.heating = True
            self.power_consumption_kw = self.capacity_kw
            self.water_temp += 0.15 * dt
        elif self.water_temp > self.setpoint:
            self.heating = False
            self.power_consumption_kw = 0.0
        
        net_power_mw = self.power_consumption_kw / 1000.0  # Convert to MW
        self.net_power_history.append(net_power_mw)
        return net_power_mw
    
    def execute_attack(self, forced_on: bool = True):
        """Execute attack by forcing heater on/off"""
        if self.is_compromised and forced_on:
            self.attack_mode = True
            self.heating = True
            self.power_consumption_kw = self.capacity_kw
    
    def get_net_power_kw(self):
        """Get current net power consumption in kW"""
        return self.power_consumption_kw

class SmartEVCharger:
    """
    Smart EV charger for LAA attacks.
    
    High-power device that can be synchronized for significant load changes.
    
    Base Power: 6 kW (scaled by N)
    """
    
    def __init__(self, device_id: str, bus_location: int,
                 max_power_kw: float = 6.0, N: int = 1):
        """
        Initialize smart EV charger.
        
        Args:
            device_id: Unique device identifier
            bus_location: Bus number (1-39)
            max_power_kw: Base maximum charging power in kW (default 6kW)
            N: Scaling factor for number of devices (default 1)
        """
        self.device_id = device_id
        self.bus_location = bus_location
        self.N = N
        self.max_power_kw = max_power_kw * N  # Scale by N
        
        # State variables
        self.is_compromised = False
        self.vehicle_connected = True
        self.battery_soc = 0.3
        self.charging_power_kw = 0.0
        
        # Attack parameters
        self.attack_mode = False
        
        # Net power tracking
        self.net_power_history = []
    
    def normal_operation(self, dt: float):
        """Normal EV charging operation"""
        if self.vehicle_connected and self.battery_soc < 0.9:
            # Smart charging with time-of-use optimization
            current_hour = (dt % 86400) / 3600
            
            if 22 <= current_hour or current_hour <= 6:  # Off-peak hours
                self.charging_power_kw = self.max_power_kw
            else:
                self.charging_power_kw = self.max_power_kw * 0.3
            
            # Update SOC
            energy_kwh = self.charging_power_kw * dt / 3600
            self.battery_soc += energy_kwh / 60.0  # 60 kWh battery
            self.battery_soc = min(self.battery_soc, 1.0)
        else:
            self.charging_power_kw = 0.0
        
        net_power_mw = self.charging_power_kw / 1000.0  # Convert to MW
        self.net_power_history.append(net_power_mw)
        return net_power_mw
    
    def execute_attack(self, forced_power_kw: float):
        """Execute attack by forcing charging power"""
        if self.is_compromised:
            self.attack_mode = True
            self.charging_power_kw = min(forced_power_kw, self.max_power_kw)
    
    def get_net_power_kw(self):
        """Get current net power consumption in kW"""
        return self.charging_power_kw

class SmartPoolPump:
    """
    Smart pool pump model for LAA attacks.
    
    Represents pool filtration systems with controllable pump motors.
    
    Base Power: 3 kW (scaled by N)
    """
    
    def __init__(self, device_id: str, bus_location: int,
                 pump_power_kw: float = 3.0, N: int = 1):
        """
        Initialize smart pool pump.
        
        Args:
            device_id: Unique device identifier
            bus_location: Bus number (1-39)
            pump_power_kw: Base pump motor power in kW (default 3kW)
            N: Scaling factor for number of devices (default 1)
        """
        self.device_id = device_id
        self.bus_location = bus_location
        self.N = N
        self.pump_power_kw = pump_power_kw * N  # Scale by N
        
        # State variables
        self.is_compromised = False
        self.pump_running = False
        self.power_consumption_kw = 0.0
        self.runtime_hours = 0.0
        self.daily_runtime_target = 8.0  # hours per day
        
        # Attack parameters
        self.attack_mode = False
        
        # Net power tracking
        self.net_power_history = []
    
    def normal_operation(self, dt: float):
        """Normal pool pump operation"""
        # Typical operation: 8 hours per day during off-peak
        current_hour = (dt % 86400) / 3600
        
        # Run during off-peak hours (6 AM - 2 PM)
        if 6 <= current_hour <= 14 and self.runtime_hours < self.daily_runtime_target:
            self.pump_running = True
            self.power_consumption_kw = self.pump_power_kw
            self.runtime_hours += dt / 3600
        else:
            self.pump_running = False
            self.power_consumption_kw = 0.0
        
        # Reset daily counter
        if current_hour < 1:
            self.runtime_hours = 0.0
        
        net_power_mw = self.power_consumption_kw / 1000.0  # Convert to MW
        self.net_power_history.append(net_power_mw)
        return net_power_mw
    
    def execute_attack(self, forced_on: bool = True):
        """Execute attack by forcing pump on/off"""
        if self.is_compromised:
            self.attack_mode = True
            if forced_on:
                self.pump_running = True
                self.power_consumption_kw = self.pump_power_kw
            else:
                self.pump_running = False
                self.power_consumption_kw = 0.0
    
    def get_net_power_kw(self):
        """Get current net power consumption in kW"""
        return self.power_consumption_kw

class IndustrialIoTController:
    """
    Industrial IoT controller for factory equipment.
    
    Controls high-power industrial loads that can create significant
    power swings when coordinated.
    
    Base Power: 50 kW (scaled by N)
    """
    
    def __init__(self, device_id: str, bus_location: int,
                 equipment_power_kw: float = 50.0, N: int = 1):
        """
        Initialize industrial IoT controller.
        
        Args:
            device_id: Unique device identifier
            bus_location: Bus number (1-39)
            equipment_power_kw: Base equipment power in kW (default 50kW)
            N: Scaling factor for number of devices (default 1)
        """
        self.device_id = device_id
        self.bus_location = bus_location
        self.N = N
        self.equipment_power_kw = equipment_power_kw * N  # Scale by N
        
        # State variables
        self.is_compromised = False
        self.running = True
        self.production_rate = 1.0
        self.power_consumption_kw = self.equipment_power_kw
        
        # Attack parameters
        self.attack_mode = False
        
        # Net power tracking
        self.net_power_history = []
    
    def normal_operation(self, dt: float):
        """Normal industrial operation"""
        if self.running:
            self.power_consumption_kw = self.equipment_power_kw * self.production_rate
        else:
            self.power_consumption_kw = self.equipment_power_kw * 0.1  # Standby
        
        net_power_mw = self.power_consumption_kw / 1000.0  # Convert to MW
        self.net_power_history.append(net_power_mw)
        return net_power_mw
    
    def execute_attack(self, shutdown: bool = False):
        """Execute attack by shutting down equipment"""
        if self.is_compromised:
            self.attack_mode = True
            if shutdown:
                self.running = False
                self.power_consumption_kw = 0.0
            else:
                self.running = True
                self.power_consumption_kw = self.equipment_power_kw
    
    def get_net_power_kw(self):
        """Get current net power consumption in kW"""
        return self.power_consumption_kw

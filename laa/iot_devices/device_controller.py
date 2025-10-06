#!/usr/bin/env python3
"""
IoT Botnet Controller for Coordinated LAA Attacks

This module implements a botnet controller that coordinates
compromised IoT devices to execute synchronized load-altering attacks.

Author: Pranaav
Date: October 2025
"""

from typing import Dict, List, Optional
import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .device_models import (
    SmartThermostat, SmartWaterHeater, SmartEVCharger,
    IndustrialIoTController, IoTDeviceType, IoTDeviceState
)

logger = logging.getLogger(__name__)

class BotnetStrategy(Enum):
    """Attack coordination strategies"""
    SIMULTANEOUS = "simultaneous"        # All devices at once
    CASCADING = "cascading"             # Sequential activation
    RANDOM = "random"                   # Random timing
    ADAPTIVE = "adaptive"               # Based on system response

@dataclass
class BotnetStats:
    """Botnet operation statistics"""
    total_devices: int = 0
    compromised_devices: int = 0
    active_attack_devices: int = 0
    total_attack_power_mw: float = 0.0
    attack_success_rate: float = 0.0

class IoTBotnetController:
    """
    Coordinated botnet controller for IoT-based LAA attacks.
    
    Manages a fleet of compromised IoT devices and coordinates
    their operation to execute load-altering attacks.
    """
    
    def __init__(self):
        """Initialize botnet controller"""
        # Device registry
        self.devices: Dict[str, any] = {}
        self.devices_by_type: Dict[str, List] = {}
        self.devices_by_bus: Dict[int, List] = {}
        
        # Attack state
        self.attack_active = False
        self.attack_strategy = BotnetStrategy.SIMULTANEOUS
        self.stats = BotnetStats()
        
        logger.info("IoT Botnet Controller initialized")
        
    def register_device(self, device):
        """Register a device with the botnet"""
        self.devices[device.device_id] = device
        
        # Index by type
        device_type = type(device).__name__
        if device_type not in self.devices_by_type:
            self.devices_by_type[device_type] = []
        self.devices_by_type[device_type].append(device)
        
        # Index by bus location
        bus = device.bus_location
        if bus not in self.devices_by_bus:
            self.devices_by_bus[bus] = []
        self.devices_by_bus[bus].append(device)
        
        self.stats.total_devices += 1
    
    def compromise_devices(self, compromise_rate: float = 0.8):
        """
        Compromise a percentage of registered devices.
        
        Args:
            compromise_rate: Percentage of devices to compromise (0-1)
        """
        compromised_count = 0
        for device in self.devices.values():
            if np.random.random() < compromise_rate:
                device.is_compromised = True
                compromised_count += 1
        
        self.stats.compromised_devices = compromised_count
        logger.info(f"Compromised {compromised_count}/{self.stats.total_devices} devices")
    
    def execute_coordinated_attack(self, 
                                   target_buses: List[int],
                                   attack_magnitude_mw: float,
                                   strategy: BotnetStrategy = BotnetStrategy.SIMULTANEOUS):
        """
        Execute coordinated attack across compromised devices.
        
        Args:
            target_buses: List of buses to attack
            attack_magnitude_mw: Total attack magnitude in MW
            strategy: Attack coordination strategy
        """
        self.attack_active = True
        self.attack_strategy = strategy
        
        # Collect compromised devices on target buses
        attack_devices = []
        for bus in target_buses:
            if bus in self.devices_by_bus:
                for device in self.devices_by_bus[bus]:
                    if device.is_compromised:
                        attack_devices.append(device)
        
        if not attack_devices:
            logger.warning("No compromised devices available for attack")
            return
        
        # Distribute attack power among devices
        power_per_device_kw = (attack_magnitude_mw * 1000) / len(attack_devices)
        
        # Execute attack based on strategy
        if strategy == BotnetStrategy.SIMULTANEOUS:
            self._execute_simultaneous_attack(attack_devices, power_per_device_kw)
        elif strategy == BotnetStrategy.CASCADING:
            self._execute_cascading_attack(attack_devices, power_per_device_kw)
        elif strategy == BotnetStrategy.ADAPTIVE:
            self._execute_adaptive_attack(attack_devices, power_per_device_kw)
        
        self.stats.active_attack_devices = len(attack_devices)
        self.stats.total_attack_power_mw = attack_magnitude_mw
        
        logger.info(f"Coordinated attack executed: {len(attack_devices)} devices, {attack_magnitude_mw} MW")
    
    def _execute_simultaneous_attack(self, devices: List, power_kw: float):
        """All devices attack simultaneously"""
        for device in devices:
            if isinstance(device, SmartThermostat):
                device.execute_attack(attack_setpoint=30.0)  # Max heating
            elif isinstance(device, SmartWaterHeater):
                device.execute_attack(forced_on=True)
            elif isinstance(device, SmartEVCharger):
                device.execute_attack(forced_power_kw=power_kw)
            elif isinstance(device, IndustrialIoTController):
                device.execute_attack(shutdown=False)  # Full power
    
    def _execute_cascading_attack(self, devices: List, power_kw: float):
        """
        Devices attack in sequence (cascading).
        ##Not needed to implemented now 
        TODO: Implement time-delayed activation
        1. Sort devices by priority or bus voltage sensitivity
        2. Calculate delay interval between device activations
        3. Use asyncio.sleep() for non-blocking delays
        4. Activate each device with specified delay
        5. Monitor cumulative attack power growth over time
        """
        logger.info("Cascading attack strategy - sequential device activation")
        # Implementation placeholder
        self._execute_simultaneous_attack(devices, power_kw)
    
    def _execute_adaptive_attack(self, devices: List, power_kw: float):
        """
        Adaptive attack based on system response.
        
        TODO: Implement feedback-based attack control
        1. Monitor system frequency deviation in real-time
        2. Calculate required attack power to reach target deviation
        3. Activate devices incrementally based on system response
        4. Use PID control for smooth attack power adjustment
        5. Stop when target frequency deviation is achieved
        """
        logger.info("Adaptive attack strategy - feedback-based control")
        # Implementation placeholder
        self._execute_simultaneous_attack(devices, power_kw)
    
    def get_total_attack_power(self) -> float:
        """Calculate total attack power from all active devices"""
        total_power_mw = 0.0
        for device in self.devices.values():
            if hasattr(device, 'attack_mode') and device.attack_mode:
                if hasattr(device, 'power_consumption_kw'):
                    total_power_mw += device.power_consumption_kw / 1000.0
                elif hasattr(device, 'charging_power_kw'):
                    total_power_mw += device.charging_power_kw / 1000.0
        
        return total_power_mw
    
    def stop_attack(self):
        """Stop all attack operations"""
        self.attack_active = False
        for device in self.devices.values():
            if hasattr(device, 'attack_mode'):
                device.attack_mode = False
        
        self.stats.active_attack_devices = 0
        logger.info("All attacks stopped")
    
    def get_statistics(self) -> BotnetStats:
        """Get current botnet statistics"""
        return self.stats
    
    def get_devices_by_bus(self, bus: int) -> List:
        """Get all devices on a specific bus"""
        return self.devices_by_bus.get(bus, [])
    
    def get_compromised_devices(self) -> List:
        """Get list of all compromised devices"""
        return [dev for dev in self.devices.values() if dev.is_compromised]

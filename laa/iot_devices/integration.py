#!/usr/bin/env python3
"""
Integration layer between IoT devices and LAA framework

This module connects IoT device models with the existing
LAA attack execution framework.

Author: Pranaav
Date: October 2025
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Import LAA framework components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from attacker.laa_config import AttackConfig, AttackType, AttackSeverity

# Import IoT device components
from .device_models import (
    SmartThermostat, SmartWaterHeater, SmartEVCharger, SmartPoolPump, IndustrialIoTController
)
from .device_controller import IoTBotnetController, BotnetStrategy

logger = logging.getLogger(__name__)

class IoTLAAIntegrator:
    """
    Integration layer between IoT devices and LAA framework.
    
    This class bridges the gap between IoT device models and
    the existing LAA attack execution system.
    """
    
    def __init__(self, ieee39_system):
        """
        Initialize integrator.
        
        Args:
            ieee39_system: Reference to IEEE 39-bus system
        """
        self.ieee39_system = ieee39_system
        self.botnet_controller = IoTBotnetController()
        
        # Device deployment configuration
        self.deployment_config = self._create_default_deployment()
        
        # Net power array for all devices
        self.net_power_array = []  # Stores net power for each device
        self.device_net_power_map = {}  # Maps device_id to net power
        
        logger.info("IoT-LAA Integrator initialized")
    
    def _create_default_deployment(self) -> Dict[int, List]:
        """
        Create default IoT device deployment on IEEE 39-bus system.
        
        Deploys devices on load buses with realistic distributions.
        """
        deployment = {
            # Residential areas (thermostats, water heaters, EVs, pool pumps)
            3: ["thermostat", "water_heater", "ev_charger", "pool_pump"],
            4: ["thermostat", "water_heater", "pool_pump"],
            7: ["thermostat", "ev_charger", "pool_pump"],
            8: ["thermostat", "water_heater", "ev_charger"],
            12: ["thermostat", "water_heater", "pool_pump"],
            15: ["thermostat", "ev_charger", "pool_pump"],
            16: ["thermostat", "water_heater", "ev_charger", "pool_pump"],
            18: ["thermostat", "water_heater"],
            20: ["thermostat", "ev_charger", "pool_pump"],
            21: ["thermostat", "water_heater", "ev_charger", "pool_pump"],
            
            # Commercial/Industrial areas
            23: ["industrial", "industrial"],
            24: ["industrial", "industrial", "industrial"],
            27: ["industrial"],
            28: ["industrial", "industrial"],
            29: ["industrial"],
        }
        
        return deployment
    
    def deploy_iot_devices(self, N: int = 1):
        """
        Deploy IoT devices according to configuration.
        
        Args:
            N: Scaling factor for number of devices (default 1)
        """
        device_count = 0
        
        for bus, device_types in self.deployment_config.items():
            for idx, device_type in enumerate(device_types):
                device_id = f"IoT_{bus}_{device_type}_{idx}"
                
                if device_type == "thermostat":
                    device = SmartThermostat(
                        device_id=device_id,
                        bus_location=bus,
                        hvac_capacity_kw=2.0,  # Base 2kW
                        N=N
                    )
                elif device_type == "water_heater":
                    device = SmartWaterHeater(
                        device_id=device_id,
                        bus_location=bus,
                        capacity_kw=3.0,  # Base 3kW
                        N=N
                    )
                elif device_type == "ev_charger":
                    device = SmartEVCharger(
                        device_id=device_id,
                        bus_location=bus,
                        max_power_kw=6.0,  # Base 6kW
                        N=N
                    )
                elif device_type == "pool_pump":
                    device = SmartPoolPump(
                        device_id=device_id,
                        bus_location=bus,
                        pump_power_kw=3.0,  # Base 3kW
                        N=N
                    )
                elif device_type == "industrial":
                    device = IndustrialIoTController(
                        device_id=device_id,
                        bus_location=bus,
                        equipment_power_kw=50.0,  # Base 50kW
                        N=N
                    )
                else:
                    continue
                
                self.botnet_controller.register_device(device)
                self.device_net_power_map[device_id] = 0.0
                device_count += 1
        
        logger.info(f"Deployed {device_count} IoT devices (N={N}) across IEEE 39-bus system")
        print(f"Deployed {device_count} IoT devices (N={N}) across IEEE 39-bus system")
        return device_count
    
    def configure_attack_from_laa_config(self, attack_config: AttackConfig):
        """
        Configure IoT attack from LAA attack configuration.
        
        Args:
            attack_config: LAA attack configuration
        """
        # Compromise devices based on severity
        compromise_rate = {
            AttackSeverity.LOW: 0.3,
            AttackSeverity.MEDIUM: 0.6,
            AttackSeverity.HIGH: 0.85,
            AttackSeverity.CRITICAL: 0.95
        }.get(attack_config.severity, 0.6)
        
        self.botnet_controller.compromise_devices(compromise_rate)
        
        # Map attack type to botnet strategy
        strategy_map = {
            AttackType.STEP: BotnetStrategy.SIMULTANEOUS,
            AttackType.RANDOM: BotnetStrategy.RANDOM,
            AttackType.PERIODIC: BotnetStrategy.CASCADING,
            AttackType.FEEDBACK: BotnetStrategy.ADAPTIVE
        }
        
        strategy = strategy_map.get(attack_config.attack_type, BotnetStrategy.SIMULTANEOUS)
        
        # Execute coordinated attack
        self.botnet_controller.execute_coordinated_attack(
            target_buses=attack_config.target_buses,
            attack_magnitude_mw=attack_config.magnitude_mw,
            strategy=strategy
        )
        
        logger.info(f"IoT attack configured: {attack_config.attack_type.value}, "
                   f"{attack_config.magnitude_mw} MW on buses {attack_config.target_buses}")
    
    def get_iot_load_contribution(self, bus: int, time: float) -> float:
        """
        Get total load contribution from IoT devices on a bus.
        
        Args:
            bus: Bus number
            time: Current simulation time
            
        Returns:
            Total load in MW
        """
        if bus not in self.botnet_controller.devices_by_bus:
            return 0.0
        
        total_load_mw = 0.0
        dt = 1.0  # 1 second time step
        
        for device in self.botnet_controller.devices_by_bus[bus]:
            if hasattr(device, 'normal_operation'):
                load_mw = device.normal_operation(dt)
                total_load_mw += load_mw
        
        return total_load_mw
    
    def update_ieee39_loads_with_iot(self, current_time: float):
        """
        Update IEEE 39-bus system loads with IoT contributions.
        
        Args:
            current_time: Current simulation time
        """
        for bus in range(1, 40):
            iot_load = self.get_iot_load_contribution(bus, current_time)
            
            # Update IEEE 39 system loads
            if hasattr(self.ieee39_system, 'ieee_loads'):
                load_key = f"IEEE_Load_{bus}"
                if load_key in self.ieee39_system.ieee_loads:
                    self.ieee39_system.ieee_loads[load_key]['p_mw'] += iot_load
    
    def get_deployment_summary(self) -> Dict:
        """Get summary of IoT device deployment"""
        summary = {
            'total_devices': self.botnet_controller.stats.total_devices,
            'devices_by_type': {},
            'devices_by_bus': {}
        }
        
        # Count by type
        for dev_type, devices in self.botnet_controller.devices_by_type.items():
            summary['devices_by_type'][dev_type] = len(devices)
        
        # Count by bus
        for bus, devices in self.botnet_controller.devices_by_bus.items():
            summary['devices_by_bus'][bus] = len(devices)
        
        return summary
    
    def update_net_power_array(self):
        """
        Update net power array for all IoT devices.
        
        Retrieves current power consumption from each device and stores in array.
        """
        self.net_power_array = []
        
        for device_id, device in self.botnet_controller.devices.items():
            if hasattr(device, 'get_net_power_kw'):
                net_power_kw = device.get_net_power_kw()
                self.device_net_power_map[device_id] = net_power_kw
                self.net_power_array.append({
                    'device_id': device_id,
                    'bus': device.bus_location,
                    'power_kw': net_power_kw,
                    'power_mw': net_power_kw / 1000.0,
                    'type': device.__class__.__name__,
                    'N': getattr(device, 'N', 1)
                })
        
        return self.net_power_array
    
    def get_net_power_summary(self) -> Dict:
        """
        Get summary statistics of net power array.
        
        Returns:
            Dictionary with power statistics
        """
        self.update_net_power_array()
        
        if not self.net_power_array:
            return {
                'total_power_kw': 0.0,
                'total_power_mw': 0.0,
                'device_count': 0,
                'avg_power_kw': 0.0
            }
        
        total_kw = sum(d['power_kw'] for d in self.net_power_array)
        
        return {
            'total_power_kw': total_kw,
            'total_power_mw': total_kw / 1000.0,
            'device_count': len(self.net_power_array),
            'avg_power_kw': total_kw / len(self.net_power_array) if self.net_power_array else 0.0,
            'net_power_array': self.net_power_array
        }
    
    def print_net_power_array(self):
        """Print net power array in formatted table"""
        self.update_net_power_array()
        
        print("\n" + "="*100)
        print("NET POWER ARRAY - ALL IOT DEVICES")
        print("="*100)
        
        print(f"\n{'Index':<6} {'Device ID':<30} {'Bus':<5} {'Type':<25} {'N':<4} {'Power (kW)':<12} {'Power (MW)'}")
        print("-"*100)
        
        for idx, device in enumerate(self.net_power_array):
            print(f"{idx:<6} {device['device_id']:<30} {device['bus']:<5} "
                  f"{device['type']:<25} {device['N']:<4} "
                  f"{device['power_kw']:<12.2f} {device['power_mw']:.6f}")
        
        summary = self.get_net_power_summary()
        print("-"*100)
        print(f"TOTAL: {summary['device_count']} devices | "
              f"Total Power: {summary['total_power_kw']:.2f} kW ({summary['total_power_mw']:.4f} MW) | "
              f"Average: {summary['avg_power_kw']:.2f} kW/device")
        print("="*100 + "\n")
    
    def get_bus_power_array(self) -> Dict[int, Dict]:
        """
        Get net power and device details for each bus as arrays.
        
        Returns:
            Dictionary with bus number as key and power details as value
        """
        bus_power_data = {}
        
        for bus in range(1, 40):
            if bus in self.botnet_controller.devices_by_bus:
                devices = self.botnet_controller.devices_by_bus[bus]
                
                # Arrays for device data
                device_ids = []
                device_types = []
                device_powers_kw = []
                device_states = []
                
                total_power_mw = 0.0
                dt = 1.0  # 1 second time step
                
                for device in devices:
                    device_ids.append(device.device_id)
                    
                    # Get device type from class name
                    device_type = device.__class__.__name__
                    device_types.append(device_type)
                    
                    # Get device state
                    if hasattr(device, 'attack_mode'):
                        state = "ATTACK" if device.attack_mode else "NORMAL"
                    elif hasattr(device, 'is_compromised'):
                        state = "COMPROMISED" if device.is_compromised else "NORMAL"
                    else:
                        state = "NORMAL"
                    device_states.append(state)
                    
                    # Get current power
                    if hasattr(device, 'normal_operation'):
                        power_mw = device.normal_operation(dt)
                        power_kw = power_mw * 1000  # Convert to kW
                        device_powers_kw.append(power_kw)
                        total_power_mw += power_mw
                    else:
                        device_powers_kw.append(0.0)
                
                bus_power_data[bus] = {
                    'bus_number': bus,
                    'device_count': len(devices),
                    'total_power_mw': round(total_power_mw, 4),
                    'total_power_kw': round(total_power_mw * 1000, 2),
                    'device_ids': device_ids,
                    'device_types': device_types,
                    'device_powers_kw': [round(p, 2) for p in device_powers_kw],
                    'device_states': device_states
                }
        
        return bus_power_data
    
    def print_bus_power_report(self, show_devices: bool = True):
        """
        Print comprehensive power report for all buses and devices.
        
        Args:
            show_devices: If True, show individual device details
        """
        bus_data = self.get_bus_power_array()
        
        print("\n" + "="*80)
        print("IoT DEVICES - BUS POWER REPORT")
        print("="*80)
        
        # Summary statistics
        total_buses = len(bus_data)
        total_devices = sum(data['device_count'] for data in bus_data.values())
        total_power_mw = sum(data['total_power_mw'] for data in bus_data.values())
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Buses with IoT Devices: {total_buses}")
        print(f"   Total IoT Devices: {total_devices}")
        print(f"   Total Net Power: {total_power_mw:.4f} MW ({total_power_mw*1000:.2f} kW)")
        print(f"   Average Power per Device: {(total_power_mw/total_devices)*1000:.2f} kW")
        
        # Bus-by-bus breakdown
        print("\n" + "-"*80)
        print("BUS-BY-BUS BREAKDOWN:")
        print("-"*80)
        
        for bus in sorted(bus_data.keys()):
            data = bus_data[bus]
            print(f"\n🔌 BUS {data['bus_number']:2d}:")
            print(f"   Devices: {data['device_count']}")
            print(f"   Net Power: {data['total_power_mw']:.4f} MW ({data['total_power_kw']:.2f} kW)")
            
            if show_devices and data['device_count'] > 0:
                print(f"   Device Array:")
                for i in range(len(data['device_ids'])):
                    print(f"      [{i}] {data['device_ids'][i]:30s} | "
                          f"Type: {data['device_types'][i]:15s} | "
                          f"Power: {data['device_powers_kw'][i]:7.2f} kW | "
                          f"State: {data['device_states'][i]}")
        
        print("\n" + "="*80)
        print("END OF REPORT")
        print("="*80 + "\n")
    
    def get_power_arrays_for_analysis(self) -> Dict[str, np.ndarray]:
        """
        Get numpy arrays of power values for analysis.
        
        Returns:
            Dictionary with array name as key and numpy array as value
        """
        bus_data = self.get_bus_power_array()
        
        # Create arrays
        bus_numbers = []
        bus_powers_mw = []
        device_counts = []
        
        for bus in sorted(bus_data.keys()):
            data = bus_data[bus]
            bus_numbers.append(data['bus_number'])
            bus_powers_mw.append(data['total_power_mw'])
            device_counts.append(data['device_count'])
        
        return {
            'bus_numbers': np.array(bus_numbers, dtype=int),
            'bus_powers_mw': np.array(bus_powers_mw, dtype=float),
            'device_counts': np.array(device_counts, dtype=int),
            'total_power_mw': np.sum(bus_powers_mw)
        }
    
    def print_power_arrays(self):
        """Print power data as numpy arrays"""
        arrays = self.get_power_arrays_for_analysis()
        
        print("\n" + "="*80)
        print("POWER DATA ARRAYS")
        print("="*80)
        
        print(f"\n📍 Bus Numbers Array (length={len(arrays['bus_numbers'])}):")
        print(f"   {arrays['bus_numbers']}")
        
        print(f"\n⚡ Bus Powers Array [MW] (length={len(arrays['bus_powers_mw'])}):")
        print(f"   {arrays['bus_powers_mw']}")
        
        print(f"\n📱 Device Count Array (length={len(arrays['device_counts'])}):")
        print(f"   {arrays['device_counts']}")
        
        print(f"\n🔋 Total System Power: {arrays['total_power_mw']:.4f} MW")
        print(f"   Min Bus Power: {np.min(arrays['bus_powers_mw']):.4f} MW")
        print(f"   Max Bus Power: {np.max(arrays['bus_powers_mw']):.4f} MW")
        print(f"   Mean Bus Power: {np.mean(arrays['bus_powers_mw']):.4f} MW")
        print(f"   Std Bus Power: {np.std(arrays['bus_powers_mw']):.4f} MW")
        
        print("\n" + "="*80 + "\n")

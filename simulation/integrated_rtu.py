#!/usr/bin/env python3
"""
Integrated RTU System for IEEE 39-Bus Power System

This module provides RTU outstation functionality that integrates
directly with the power system simulation and reports data through mock DNP3.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import numpy as np
from mock_dnp3 import dnp3_channel, DNP3Point, DNP3ObjectType, DNP3Quality

logger = logging.getLogger(__name__)

@dataclass 
class RTUConfig:
    """RTU Configuration"""
    rtu_id: int
    name: str
    bus_number: int
    measurement_types: List[str]  # voltage, frequency, active_power, reactive_power
    update_rate: float = 1.0  # seconds

class IntegratedRTU:
    """Integrated RTU Outstation"""
    
    def __init__(self, config: RTUConfig, power_system=None):
        self.config = config
        self.power_system = power_system
        self.is_running = False
        self.update_thread = None
        
        # Measurement point mapping
        self.measurement_points = {
            'voltage_magnitude': DNP3Point(0, 'voltage_magnitude', DNP3ObjectType.ANALOG_INPUT, 1.0),
            'voltage_angle': DNP3Point(1, 'voltage_angle', DNP3ObjectType.ANALOG_INPUT, 0.0),
            'frequency': DNP3Point(2, 'frequency', DNP3ObjectType.ANALOG_INPUT, 50.0),
            'active_power': DNP3Point(3, 'active_power', DNP3ObjectType.ANALOG_INPUT, 0.0),
            'reactive_power': DNP3Point(4, 'reactive_power', DNP3ObjectType.ANALOG_INPUT, 0.0),
            'breaker_status': DNP3Point(10, 'breaker_status', DNP3ObjectType.BINARY_INPUT, 1.0)
        }
        
        # Control points
        self.control_points = {
            'breaker_command': DNP3Point(20, 'breaker_command', DNP3ObjectType.BINARY_OUTPUT, 0.0)
        }
        
        # Register with DNP3 channel
        dnp3_channel.register_rtu(self.config.rtu_id)
        
        logger.info(f"RTU {config.rtu_id} ({config.name}) initialized for Bus {config.bus_number}")
    
    def start(self):
        """Start RTU measurement updates"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start measurement update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info(f"RTU {self.config.rtu_id} started")
    
    def stop(self):
        """Stop RTU"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info(f"RTU {self.config.rtu_id} stopped")
    
    def _update_loop(self):
        """Main measurement update loop"""
        while self.is_running:
            try:
                self._update_measurements()
                time.sleep(self.config.update_rate)
            except Exception as e:
                logger.error(f"Error in RTU {self.config.rtu_id} update loop: {e}")
                time.sleep(1)
    
    def _update_measurements(self):
        """Update measurements from power system"""
        try:
            if self.power_system:
                # Get real power system data
                system_state = self.power_system.get_system_state()
                bus_idx = self.config.bus_number - 1  # Convert to 0-based index
                
                # Update voltage measurements correctly
                if 'bus_voltages' in system_state and len(system_state['bus_voltages']) > bus_idx:
                    voltage_magnitude = system_state['bus_voltages'][bus_idx]
                    self.measurement_points['voltage_magnitude'].value = float(voltage_magnitude)
                    logger.debug(f"RTU {self.config.rtu_id} Bus {self.config.bus_number}: V = {voltage_magnitude:.4f} pu")
                
                if 'bus_angles' in system_state and len(system_state['bus_angles']) > bus_idx:
                    voltage_angle = system_state['bus_angles'][bus_idx] 
                    self.measurement_points['voltage_angle'].value = float(voltage_angle)
                
                # Update frequency from system
                self.measurement_points['frequency'].value = system_state.get('frequency_hz', 50.0)
                
                # Get power data from power system directly
                if hasattr(self.power_system, 'ieee39_case') and self.power_system.power_flow_solved:
                    # Get generator and load data from the power system case
                    bus_data = self.power_system.bus_results
                    gen_data = self.power_system.gen_results
                    
                    # Update power measurements based on bus type
                    if bus_idx < len(bus_data):
                        # For generator buses, get generation data
                        gen_mask = gen_data[:, 0] == self.config.bus_number  # Generator bus column
                        if np.any(gen_mask):
                            gen_row = gen_data[gen_mask][0]
                            self.measurement_points['active_power'].value = float(gen_row[1])  # PG column
                            self.measurement_points['reactive_power'].value = float(gen_row[2])  # QG column
                        else:
                            # For load buses, get load data
                            bus_row = bus_data[bus_idx]
                            self.measurement_points['active_power'].value = float(bus_row[2])  # PD column
                            self.measurement_points['reactive_power'].value = float(bus_row[3])  # QD column
                
            else:
                # Generate simulated measurements if no power system
                self._generate_simulated_measurements()
            
            # Update timestamps and quality
            current_time = time.time()
            for point in self.measurement_points.values():
                point.timestamp = current_time
                point.quality = DNP3Quality.GOOD
            
            # Send all measurement points to DNP3 channel
            for point in self.measurement_points.values():
                dnp3_channel.update_rtu_point(self.config.rtu_id, point)
            
            logger.debug(f"RTU {self.config.rtu_id} measurements updated: V={self.measurement_points['voltage_magnitude'].value:.4f}pu, P={self.measurement_points['active_power'].value:.1f}MW")
            
        except Exception as e:
            logger.error(f"Error updating RTU {self.config.rtu_id} measurements: {e}")
            # Set quality to communication error
            for point in self.measurement_points.values():
                point.quality = DNP3Quality.COMM_LOST
    
    def _generate_simulated_measurements(self):
        """Generate simulated measurements for testing"""
        # Generate realistic power system measurements
        bus_num = self.config.bus_number
        
        # Simulate voltage based on typical power system values (0.95 - 1.05 pu)
        base_voltage = 1.0 + 0.03 * np.sin(time.time() * 0.1 + bus_num * 0.1)
        if base_voltage < 0.95:
            base_voltage = 0.95 + 0.02 * np.random.random()
        elif base_voltage > 1.05:
            base_voltage = 1.05 - 0.02 * np.random.random()
        
        self.measurement_points['voltage_magnitude'].value = base_voltage
        self.measurement_points['voltage_angle'].value = -10.0 + 5.0 * np.sin(time.time() * 0.05 + bus_num * 0.2)
        
        # Simulate frequency close to 50 Hz
        self.measurement_points['frequency'].value = 50.0 + 0.05 * np.sin(time.time() * 0.3)
        
        # Simulate power measurements based on bus type
        if bus_num in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:  # Generator buses
            # Typical generator output
            base_power = 200.0 + 300.0 * np.sin(time.time() * 0.1 + bus_num * 0.1)
            self.measurement_points['active_power'].value = max(50.0, base_power)
            self.measurement_points['reactive_power'].value = base_power * 0.2
        else:  # Load buses
            # Typical load consumption
            base_power = 100.0 + 150.0 * np.sin(time.time() * 0.1 + bus_num * 0.15)
            self.measurement_points['active_power'].value = max(10.0, base_power)
            self.measurement_points['reactive_power'].value = base_power * 0.4
        
        # Breaker always closed
        self.measurement_points['breaker_status'].value = 1.0
    
    def execute_control_command(self, point_index: int, value: float) -> bool:
        """Execute control command received from SCADA"""
        try:
            if point_index in [p.index for p in self.control_points.values()]:
                # Find the control point
                for point in self.control_points.values():
                    if point.index == point_index:
                        point.value = value
                        point.timestamp = time.time()
                        logger.info(f"RTU {self.config.rtu_id} executed control: {point.name} = {value}")
                        return True
            else:
                logger.warning(f"RTU {self.config.rtu_id} received unknown control point {point_index}")
                return False
        except Exception as e:
            logger.error(f"Error executing control command on RTU {self.config.rtu_id}: {e}")
            return False
    
    def get_current_measurements(self) -> Dict[str, float]:
        """Get current measurement values"""
        return {name: point.value for name, point in self.measurement_points.items()}

# RTU Management
class RTUManager:
    """Manages multiple RTUs for IEEE 39-bus system"""
    
    def __init__(self):
        self.rtus: Dict[int, IntegratedRTU] = {}
        self.is_running = False
    
    def create_standard_rtus(self, power_system=None):
        """Create standard RTUs for IEEE 39-bus system"""
        
        # Generator RTUs
        generator_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        for i, bus in enumerate(generator_buses):
            config = RTUConfig(
                rtu_id=i + 1,
                name=f"Gen_{bus}_RTU",
                bus_number=bus,
                measurement_types=['voltage', 'frequency', 'active_power', 'reactive_power']
            )
            self.rtus[config.rtu_id] = IntegratedRTU(config, power_system)
        
        # Load RTUs for critical buses
        load_buses = [3, 4, 7, 8, 15, 16, 18, 20, 21, 23]
        for i, bus in enumerate(load_buses):
            config = RTUConfig(
                rtu_id=i + 11,  # Start after generator RTUs
                name=f"Load_{bus}_RTU", 
                bus_number=bus,
                measurement_types=['voltage', 'frequency', 'active_power', 'reactive_power']
            )
            self.rtus[config.rtu_id] = IntegratedRTU(config, power_system)
        
        logger.info(f"Created {len(self.rtus)} RTUs for IEEE 39-bus system")
    
    def start_all(self):
        """Start all RTUs"""
        if self.is_running:
            return
        
        self.is_running = True
        for rtu in self.rtus.values():
            rtu.start()
        
        logger.info(f"Started {len(self.rtus)} RTUs")
    
    def stop_all(self):
        """Stop all RTUs"""
        self.is_running = False
        for rtu in self.rtus.values():
            rtu.stop()
        
        logger.info("All RTUs stopped")
    
    def get_rtu_list(self) -> List[Dict]:
        """Get list of all RTUs with basic info"""
        return [
            {
                'rtu_id': rtu.config.rtu_id,
                'name': rtu.config.name,
                'bus_number': rtu.config.bus_number,
                'is_running': rtu.is_running
            }
            for rtu in self.rtus.values()
        ]

# Global RTU manager
rtu_manager = RTUManager()
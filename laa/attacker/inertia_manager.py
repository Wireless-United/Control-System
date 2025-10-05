#!/usr/bin/env python3
"""
Inertia Scenario Manager for LAA Simulation

This module manages different inertia conditions for the IEEE 39-bus system,
implementing low and high inertia scenarios by configuring generator parameters,
DER systems, and system response characteristics.

Author: Pranaav
Date: October 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import copy

from .laa_config import InertiaCondition, SystemInertiaConfig

# Import simulation modules with proper path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'simulation'))

try:
    from ieee39_system_strict import StrictIEEE39BusSystem
    from dynamic_models import GeneratorParams, ExciterParams, GovernorParams
    SIMULATION_AVAILABLE = True
except ImportError as e:
    print(f" Simulation modules not available: {e}")
    SIMULATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class InertiaScenarioManager:
    """
    Manages system inertia scenarios for LAA simulation
    
    This class configures the IEEE 39-bus system for different inertia conditions:
    - Low Inertia: High DER penetration, reduced synchronous generation
    - High Inertia: Low DER penetration, increased synchronous generation
    """
    
    def __init__(self, ieee39_system=None):
        """
        Initialize inertia scenario manager
        
        Args:
            ieee39_system: IEEE 39-bus system instance
        """
        self.system = ieee39_system
        self.original_generator_params = {}
        self.original_case_data = None
        self.current_inertia_config = None
        
        # Store original system parameters only if system is available
        if self.system is not None and SIMULATION_AVAILABLE:
            self._store_original_parameters()
        else:
            logger.warning("IEEE39 system not available - running in config-only mode")
    
    def _store_original_parameters(self):
        """Store original system parameters for restoration"""
        if not SIMULATION_AVAILABLE or self.system is None:
            logger.warning("Simulation not available - skipping parameter storage")
            return
            
        try:
            # Store original PyPower case data
            self.original_case_data = copy.deepcopy(self.system.ieee39_case)
            
            # Store original generator parameters if available
            if hasattr(self.system, 'generators'):
                for gen_id, generator in self.system.generators.items():
                    self.original_generator_params[gen_id] = copy.deepcopy(generator.params)
            
            logger.info("Original system parameters stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store original parameters: {e}")
            raise
    
    def configure_inertia_scenario(self, inertia_config: SystemInertiaConfig) -> bool:
        """
        Configure system for specified inertia scenario
        
        Args:
            inertia_config: Inertia configuration parameters
            
        Returns:
            bool: True if configuration successful
        """
        try:
            self.current_inertia_config = inertia_config
            
            logger.info(f"Configuring {inertia_config.condition.value} scenario...")
            
            # Configure generator parameters
            self._configure_generators(inertia_config)
            
            # Configure DER systems
            self._configure_der_systems(inertia_config)
            
            # Configure system response characteristics  
            self._configure_system_response(inertia_config)
            
            # Validate configuration
            validation_result = self._validate_inertia_configuration()
            
            if validation_result:
                logger.info(f" {inertia_config.condition.value} scenario configured successfully")
            else:
                logger.error(f" {inertia_config.condition.value} scenario configuration failed")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to configure inertia scenario: {e}")
            return False
    
    def _configure_generators(self, inertia_config: SystemInertiaConfig):
        """Configure generator parameters for inertia scenario"""
        
        # IEEE 39-bus generator locations (buses with generators)
        generator_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        
        logger.info(f"Configuring generators with inertia multiplier: {inertia_config.generator_inertia_multiplier}")
        
        # Modify generator parameters in PyPower case
        for i, bus_num in enumerate(generator_buses):
            gen_idx = i  # Generator index in PyPower case
            
            if gen_idx < len(self.system.ieee39_case['gen']):
                # Adjust generator reactive power limits based on inertia scenario
                if inertia_config.condition == InertiaCondition.LOW:
                    # Low inertia: Reduce generator capacity (simulating retirement)
                    capacity_factor = 0.8
                    self.system.ieee39_case['gen'][gen_idx][8] *= capacity_factor  # PMAX
                    self.system.ieee39_case['gen'][gen_idx][3] *= capacity_factor  # QMAX  
                    self.system.ieee39_case['gen'][gen_idx][4] *= capacity_factor  # QMIN
                    
                elif inertia_config.condition == InertiaCondition.HIGH:
                    # High inertia: Increase generator capacity
                    capacity_factor = 1.2
                    self.system.ieee39_case['gen'][gen_idx][8] *= capacity_factor  # PMAX
                    self.system.ieee39_case['gen'][gen_idx][3] *= capacity_factor  # QMAX
                    self.system.ieee39_case['gen'][gen_idx][4] *= capacity_factor  # QMIN
        
        # Configure dynamic generator models if available
        if hasattr(self.system, 'generators'):
            self._configure_dynamic_generators(inertia_config)
    
    def _configure_dynamic_generators(self, inertia_config: SystemInertiaConfig):
        """Configure dynamic generator models"""
        
        for gen_id, generator in self.system.generators.items():
            # Modify inertia constant (H)
            original_H = self.original_generator_params[gen_id].H
            generator.params.H = original_H * inertia_config.generator_inertia_multiplier
            
            # Modify damping coefficient (D)  
            original_D = self.original_generator_params[gen_id].D
            generator.params.D = original_D * inertia_config.damping_multiplier
            
            logger.debug(f"Generator {gen_id}: H={generator.params.H:.2f}, D={generator.params.D:.2f}")
    
    def _configure_der_systems(self, inertia_config: SystemInertiaConfig):
        """Configure DER system penetration levels"""
        
        try:
            # DER bus locations for IEEE 39-bus system
            der_bus_locations = {
                'solar_pv': [20, 21, 23, 24, 27],      # Load centers
                'wind': [6, 10, 13, 19, 22],           # Transmission buses
                'battery': [7, 15, 16, 18, 28],        # Strategic locations  
                'ev_aggregator': [3, 4, 8, 12, 29]     # Urban areas
            }
            
            penetration = inertia_config.der_penetration_level
            logger.info(f"Configuring DER systems with {penetration*100:.1f}% penetration")
            
            # Configure DER capacities based on penetration level
            if hasattr(self.system, 'der_systems'):
                for der_name, der_system in self.system.der_systems.items():
                    # Scale DER capacity based on penetration level
                    if 'Solar' in der_name:
                        base_capacity = 10.0  # MW
                        der_system.capacity_mw = base_capacity * penetration * 1.5
                        
                    elif 'Wind' in der_name:
                        base_capacity = 15.0  # MW  
                        der_system.capacity_mw = base_capacity * penetration * 1.2
                        
                    elif 'BESS' in der_name:
                        base_capacity = 5.0   # MW
                        der_system.capacity_mw = base_capacity * penetration * 2.0
                        
                    elif 'EV' in der_name:
                        base_capacity = 8.0   # MW
                        der_system.capacity_mw = base_capacity * penetration * 1.0
            
            # Add equivalent DER load/generation to PyPower case
            self._add_der_to_case(der_bus_locations, penetration)
            
        except Exception as e:
            logger.error(f"DER configuration failed: {e}")
    
    def _add_der_to_case(self, der_locations: Dict, penetration: float):
        """Add DER systems to PyPower case as equivalent load/generation"""
        
        for der_type, bus_list in der_locations.items():
            for bus_num in bus_list:
                bus_idx = bus_num - 1  # Convert to 0-based index
                
                if bus_idx < len(self.system.ieee39_case['bus']):
                    if der_type in ['solar_pv', 'wind']:
                        # Add generation (negative load)
                        generation_mw = 5.0 * penetration
                        self.system.ieee39_case['bus'][bus_idx][2] -= generation_mw  # PD
                        
                    elif der_type in ['battery']:
                        # Batteries can charge/discharge - neutral for now
                        pass
                        
                    elif der_type in ['ev_aggregator']:
                        # EVs add controllable load
                        additional_load = 3.0 * penetration
                        self.system.ieee39_case['bus'][bus_idx][2] += additional_load  # PD
    
    def _configure_system_response(self, inertia_config: SystemInertiaConfig):
        """Configure system response characteristics"""
        
        # Configure voltage regulation
        self._configure_voltage_regulation(inertia_config.voltage_regulation_strength)
        
        # Configure frequency response
        self._configure_frequency_response(inertia_config.frequency_response_rate)
        
        logger.info("System response characteristics configured")
    
    def _configure_voltage_regulation(self, regulation_strength: float):
        """Configure voltage regulation strength"""
        
        # Modify generator voltage set points based on regulation strength
        for i in range(len(self.system.ieee39_case['gen'])):
            # Adjust voltage set points (VG column)
            base_voltage = self.original_case_data['gen'][i][5]  # VG
            self.system.ieee39_case['gen'][i][5] = base_voltage * regulation_strength
    
    def _configure_frequency_response(self, response_rate: float):
        """Configure frequency response characteristics"""
        
        # Configure governor response if dynamic models are available
        if hasattr(self.system, 'governors'):
            for gov_id, governor in self.system.governors.items():
                # Modify governor response rate (affects droop and time constants)
                governor.params.R = governor.params.R / response_rate  # Adjust droop
                governor.params.TG = governor.params.TG / response_rate  # Adjust time constant
    
    def _validate_inertia_configuration(self) -> bool:
        """Validate inertia configuration"""
        
        try:
            # Run power flow to validate system stability
            result = self.system.run_strict_ieee39_analysis()
            
            if not result['pypower_analysis']:
                logger.error("Power flow did not converge after inertia configuration")
                return False
            
            # Check system state
            system_state = self.system.get_system_state()
            
            # Validate voltage levels
            min_voltage = min(result['pypower_analysis']['bus_voltages'])
            max_voltage = max(result['pypower_analysis']['bus_voltages'])
            
            if min_voltage < 0.9 or max_voltage > 1.1:
                logger.warning(f"Voltage levels outside acceptable range: {min_voltage:.3f} - {max_voltage:.3f}")
            
            # Validate power balance
            total_generation = system_state['total_generation_mw']
            total_load = system_state['total_load_mw']
            power_imbalance = abs(total_generation - total_load)
            
            if power_imbalance > 10.0:  # 10 MW tolerance
                logger.warning(f"Large power imbalance: {power_imbalance:.2f} MW")
            
            logger.info(" Inertia configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def restore_original_parameters(self):
        """Restore original system parameters"""
        
        try:
            # Restore PyPower case data
            if self.original_case_data:
                self.system.ieee39_case = copy.deepcopy(self.original_case_data)
            
            # Restore generator parameters
            if hasattr(self.system, 'generators'):
                for gen_id, generator in self.system.generators.items():
                    if gen_id in self.original_generator_params:
                        generator.params = copy.deepcopy(self.original_generator_params[gen_id])
            
            self.current_inertia_config = None
            logger.info("Original system parameters restored")
            
        except Exception as e:
            logger.error(f"Failed to restore parameters: {e}")
    
    def get_current_inertia_condition(self) -> Optional[InertiaCondition]:
        """Get current inertia condition"""
        if self.current_inertia_config:
            return self.current_inertia_config.condition
        return None
    
    def get_system_inertia_metrics(self) -> Dict[str, float]:
        """
        Calculate system-wide inertia metrics
        
        Returns:
            Dict containing inertia-related metrics
        """
        metrics = {
            'total_system_inertia': 0.0,
            'average_damping': 0.0,
            'der_penetration': 0.0,
            'synchronous_generation_ratio': 0.0
        }
        
        try:
            if self.current_inertia_config:
                metrics['der_penetration'] = self.current_inertia_config.der_penetration_level
                
                # Calculate total system inertia
                if hasattr(self.system, 'generators'):
                    total_inertia = 0.0
                    total_damping = 0.0
                    gen_count = 0
                    
                    for generator in self.system.generators.values():
                        total_inertia += generator.params.H * generator.params.MVA_base
                        total_damping += generator.params.D
                        gen_count += 1
                    
                    metrics['total_system_inertia'] = total_inertia
                    metrics['average_damping'] = total_damping / max(gen_count, 1)
                
                # Calculate synchronous generation ratio
                system_state = self.system.get_system_state()
                total_gen = system_state.get('total_generation_mw', 0)
                der_gen = total_gen * metrics['der_penetration']
                sync_gen = total_gen - der_gen
                metrics['synchronous_generation_ratio'] = sync_gen / max(total_gen, 1)
            
        except Exception as e:
            logger.error(f"Failed to calculate inertia metrics: {e}")
        
        return metrics

if __name__ == "__main__":
    # Test inertia scenario manager
    print("Testing Inertia Scenario Manager...")
    
    try:
        # Initialize IEEE 39-bus system
        system = StrictIEEE39BusSystem()
        inertia_manager = InertiaScenarioManager(system)
        
        # Test low inertia configuration
        from .laa_config import LAA_Configurations
        low_inertia_config = LAA_Configurations.get_low_inertia_config()
        
        success = inertia_manager.configure_inertia_scenario(low_inertia_config)
        print(f"Low inertia configuration: {' SUCCESS' if success else ' FAILED'}")
        
        # Get inertia metrics
        metrics = inertia_manager.get_system_inertia_metrics()
        print(f"System inertia metrics: {metrics}")
        
        # Test high inertia configuration
        high_inertia_config = LAA_Configurations.get_high_inertia_config()
        success = inertia_manager.configure_inertia_scenario(high_inertia_config)
        print(f"High inertia configuration: {' SUCCESS' if success else ' FAILED'}")
        
        # Restore original parameters
        inertia_manager.restore_original_parameters()
        print(" Original parameters restored")
        
    except Exception as e:
        print(f" Test failed: {e}")
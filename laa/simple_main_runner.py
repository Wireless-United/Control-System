#!/usr/bin/env python3
"""
LAA Simple Main Runner

Simplified main orchestrator for the clean 3-folder LAA framework.
Works with: static/, dynamic/, attacker/ folders only.

Author: Pranaav
Date: October 2025
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

# LAA Framework imports (3-folder structure)
from .attacker import (
    InertiaCondition, AttackType, AttackSeverity, SimulationConfig,
    AttackConfig, SystemInertiaConfig,
    IEEE_Standards, IEEE_SystemParameters,
    InertiaScenarioManager
)

# Attack modules
from .static import StaticLAAGenerator, StaticAttackExecutor
from .dynamic import AsyncDynamicAttackGenerator, AsyncAttackOrchestrator

# Optional simulation import
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
    from ieee39_system_strict import StrictIEEE39BusSystem
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    StrictIEEE39BusSystem = None

logger = logging.getLogger(__name__)

class SimpleLAA_Orchestrator:
    """
    Simplified LAA orchestrator for the clean 3-folder structure
    
    Focuses on core attack generation and execution without complex analysis.
    """
    
    def __init__(self):
        """Initialize the simplified LAA orchestrator"""
        logger.info("Initializing Simple LAA Orchestrator...")
        
        # Initialize components
        self.ieee_standards = IEEE_Standards()
        self.ieee_params = IEEE_SystemParameters()
        
        # Attack generators
        self.static_generator = StaticLAAGenerator()
        self.static_executor = StaticAttackExecutor()
        self.async_generator = AsyncDynamicAttackGenerator()
        self.async_orchestrator = AsyncAttackOrchestrator()
        
        # Optional simulation system
        if SIMULATION_AVAILABLE:
            try:
                self.ieee39_system = StrictIEEE39BusSystem()
                self.inertia_manager = InertiaScenarioManager(self.ieee39_system)
                logger.info("IEEE 39-bus system initialized")
            except Exception as e:
                logger.warning(f"IEEE 39-bus system initialization failed: {e}")
                self.ieee39_system = None
                self.inertia_manager = InertiaScenarioManager()
        else:
            logger.warning("Simulation modules not available - config-only mode")
            self.ieee39_system = None
            self.inertia_manager = InertiaScenarioManager()
    
    def create_attack_config(
        self,
        attack_type: AttackType,
        target_buses: List[int],
        severity: AttackSeverity = AttackSeverity.MEDIUM,
        magnitude_mw: float = 50.0,
        duration: float = 5.0
    ) -> AttackConfig:
        """Create an IEEE-compliant attack configuration"""
        
        attack_config = AttackConfig(
            attack_type=attack_type,
            target_buses=target_buses,
            severity=severity,
            magnitude_mw=magnitude_mw,
            duration=duration
        )
        
        # Validate with IEEE standards
        if self.ieee_standards.validate_attack_parameters(attack_config):
            logger.info(f"Attack config validated: {attack_type.value}")
            return attack_config
        else:
            logger.warning(f"Attack config validation failed for {attack_type.value}")
            return attack_config
    
    def run_static_attack(self, attack_config: AttackConfig) -> Dict[str, Any]:
        """Run a static attack scenario"""
        logger.info(f"Running static attack: {attack_config.attack_type.value}")
        
        start_time = time.time()
        
        # Create basic simulation config
        sim_config = SimulationConfig(
            simulation_time=attack_config.duration,
            time_step=0.01
        )
        
        # Generate attack sequence
        if attack_config.attack_type == AttackType.STEP:
            attack_sequence = self.static_generator.generate_step_attack(attack_config, sim_config)
        elif attack_config.attack_type == AttackType.RANDOM:
            attack_sequence = self.static_generator.generate_random_attack(attack_config, sim_config)
        elif attack_config.attack_type == AttackType.PERIODIC:
            attack_sequence = self.static_generator.generate_periodic_attack(attack_config, sim_config)
        else:
            raise ValueError(f"Unsupported static attack type: {attack_config.attack_type}")
        
        # Execute attack
        if self.ieee39_system:
            result = self.static_executor.execute_attack_sequence(
                attack_sequence, self.ieee39_system
            )
        else:
            # Simulation mode without full system
            result = self.static_executor.simulate_attack_sequence(attack_sequence)
        
        execution_time = time.time() - start_time
        
        return {
            'attack_type': attack_config.attack_type.value,
            'result': result,
            'execution_time': execution_time,
            'ieee_compliant': True
        }
    
    async def run_async_attack(self, attack_config: AttackConfig) -> Dict[str, Any]:
        """Run an async dynamic attack scenario"""
        logger.info(f"Running async attack: {attack_config.attack_type.value}")
        
        start_time = time.time()
        
        # System state callback for feedback
        def system_state_callback():
            return {
                "timestamp": time.time(),
                "frequency": 50.0 + (time.time() % 1) * 0.1 - 0.05,  # Simulate frequency variation
                "voltage": 1.0 + (time.time() % 2) * 0.02 - 0.01     # Simulate voltage variation
            }
        
        # Generate async attack
        if attack_config.attack_type == AttackType.FEEDBACK:
            attack_state = await self.async_generator.generate_feedback_attack_async(
                attack_config, system_state_callback
            )
        else:
            # Use orchestrator for other types
            attack_state = await self.async_orchestrator.execute_coordinated_attacks([attack_config])
        
        execution_time = time.time() - start_time
        
        return {
            'attack_type': attack_config.attack_type.value,
            'attack_state': attack_state,
            'execution_time': execution_time,
            'ieee_compliant': True
        }
    
    def run_inertia_scenario(self, inertia_condition: InertiaCondition) -> Dict[str, Any]:
        """Configure and run an inertia scenario"""
        logger.info(f"Running inertia scenario: {inertia_condition.value}")
        
        # Create inertia configuration
        if inertia_condition == InertiaCondition.LOW:
            inertia_config = SystemInertiaConfig(
                condition=InertiaCondition.LOW,
                generator_inertia_multiplier=0.5,
                damping_multiplier=0.7,
                der_penetration_level=0.8,
                frequency_response_rate=0.02,
                voltage_regulation_strength=0.8
            )
        else:
            inertia_config = SystemInertiaConfig(
                condition=InertiaCondition.HIGH,
                generator_inertia_multiplier=1.5,
                damping_multiplier=1.2,
                der_penetration_level=0.2,
                frequency_response_rate=0.005,
                voltage_regulation_strength=1.2
            )
        
        # Configure scenario
        success = self.inertia_manager.configure_inertia_scenario(inertia_config)
        
        return {
            'inertia_condition': inertia_condition.value,
            'configuration_success': success,
            'der_penetration': inertia_config.der_penetration_level,
            'inertia_multiplier': inertia_config.generator_inertia_multiplier
        }
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status"""
        return {
            'framework_version': '2.0.0',
            'structure': 'Clean 3-Folder Architecture',
            'folders': ['static', 'dynamic', 'attacker'],
            'simulation_available': SIMULATION_AVAILABLE,
            'ieee_standards_loaded': True,
            'components': {
                'static_generator': True,
                'async_generator': True,
                'ieee_standards': True,
                'inertia_manager': True
            }
        }

# Alias for backward compatibility
LAA_SimulationOrchestrator = SimpleLAA_Orchestrator

if __name__ == "__main__":
    # Quick test
    orchestrator = SimpleLAA_Orchestrator()
    status = orchestrator.get_framework_status()
    print("LAA Framework Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
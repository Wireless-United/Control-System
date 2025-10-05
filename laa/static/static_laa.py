#!/usr/bin/env python3
"""
Static Load-Altering Attacks (LAA) Module

This module implements static LAA attack scenarios including:
- Step attacks: Sudden load increase/decrease
- Random attacks: Stochastic load variations  
- Periodic attacks: Sinusoidal oscillations

Author: Pranaav
Date: October 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
import random

from ..attacker.laa_config import AttackType, AttackSeverity, SimulationConfig, AttackConfig

logger = logging.getLogger(__name__)

# ======================== ATTACK RESULT CLASSES ======================== #

@dataclass
class AttackResult:
    """Results from a single attack execution"""
    attack_type: AttackType
    target_buses: List[int]
    time_stamp: float
    load_changes: Dict[int, float]  # Bus -> Load change (MW)
    success: bool
    error_message: Optional[str] = None

@dataclass
class StaticAttackSequence:
    """Sequence of static attack results over time"""
    attack_config: AttackConfig
    time_series: np.ndarray
    load_series: Dict[int, np.ndarray]  # Bus -> Load time series
    attack_results: List[AttackResult]
    total_energy_injected: float  # Total MWh injected

# ======================== STATIC ATTACK GENERATORS ======================== #

class StaticLAAGenerator:
    """
    Generator for static Load-Altering Attacks
    
    This class implements various static attack patterns that can be applied
    to target buses in the power system.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize static LAA generator
        
        Args:
            random_seed: Seed for random number generation (for reproducibility)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.attack_history = []
        logger.info("Static LAA Generator initialized")
    
    def generate_step_attack(
        self, 
        attack_config: AttackConfig, 
        simulation_config: SimulationConfig
    ) -> StaticAttackSequence:
        """
        Generate step attack sequence
        
        Args:
            attack_config: Attack configuration
            simulation_config: Simulation configuration
            
        Returns:
            StaticAttackSequence: Complete attack sequence
        """
        if attack_config.attack_type != AttackType.STEP:
            raise ValueError("Attack config must be STEP type")
        
        logger.info(f"Generating step attack for buses {attack_config.target_buses}")
        
        time_steps = simulation_config.get_time_steps()
        attack_sequence = StaticAttackSequence(
            attack_config=attack_config,
            time_series=time_steps,
            load_series={},
            attack_results=[],
            total_energy_injected=0.0
        )
        
        # Initialize load series for each target bus
        for bus in attack_config.target_buses:
            attack_sequence.load_series[bus] = np.zeros_like(time_steps)
        
        # Calculate step timing
        start_idx = int(attack_config.start_time / simulation_config.time_step)
        end_idx = int((attack_config.start_time + attack_config.duration) / simulation_config.time_step)
        end_idx = min(end_idx, len(time_steps))
        
        # Generate step load changes
        step_magnitude = attack_config.step_magnitude or attack_config.magnitude_mw
        
        for bus in attack_config.target_buses:
            # Distribute attack magnitude among buses
            bus_load_change = step_magnitude / len(attack_config.target_buses)
            
            # Apply step change during attack period
            attack_sequence.load_series[bus][start_idx:end_idx] = bus_load_change
            
            # Record attack result
            attack_result = AttackResult(
                attack_type=AttackType.STEP,
                target_buses=[bus],
                time_stamp=attack_config.start_time,
                load_changes={bus: bus_load_change},
                success=True
            )
            attack_sequence.attack_results.append(attack_result)
        
        # Calculate total energy
        attack_sequence.total_energy_injected = self._calculate_total_energy(
            attack_sequence, simulation_config.time_step
        )
        
        logger.info(f"Step attack generated: {attack_sequence.total_energy_injected:.2f} MWh total energy")
        return attack_sequence
    
    def generate_random_attack(
        self, 
        attack_config: AttackConfig, 
        simulation_config: SimulationConfig
    ) -> StaticAttackSequence:
        """
        Generate random attack sequence with stochastic variations
        
        Args:
            attack_config: Attack configuration
            simulation_config: Simulation configuration
            
        Returns:
            StaticAttackSequence: Complete attack sequence
        """
        if attack_config.attack_type != AttackType.RANDOM:
            raise ValueError("Attack config must be RANDOM type")
        
        logger.info(f"Generating random attack for buses {attack_config.target_buses}")
        
        time_steps = simulation_config.get_time_steps()
        attack_sequence = StaticAttackSequence(
            attack_config=attack_config,
            time_series=time_steps,
            load_series={},
            attack_results=[],
            total_energy_injected=0.0
        )
        
        # Initialize load series for each target bus
        for bus in attack_config.target_buses:
            attack_sequence.load_series[bus] = np.zeros_like(time_steps)
        
        # Calculate attack timing
        start_idx = int(attack_config.start_time / simulation_config.time_step)
        end_idx = int((attack_config.start_time + attack_config.duration) / simulation_config.time_step)
        end_idx = min(end_idx, len(time_steps))
        
        # Generate random variations
        variance = attack_config.random_variance or (attack_config.magnitude_mw * 0.3)
        mean_load = attack_config.magnitude_mw / len(attack_config.target_buses)
        
        for bus in attack_config.target_buses:
            # Generate random walk for load changes
            random_changes = np.random.normal(
                loc=mean_load,
                scale=variance,
                size=end_idx - start_idx
            )
            
            # Apply smoothing to avoid unrealistic rapid changes
            random_changes = self._apply_smoothing_filter(random_changes, window_size=5)
            
            # Ensure non-negative loads (can't have negative load attacks)
            random_changes = np.maximum(random_changes, 0.0)
            
            # Apply random changes during attack period
            attack_sequence.load_series[bus][start_idx:end_idx] = random_changes
            
            # Record attack results (sample at key time points)
            sample_times = np.linspace(start_idx, end_idx-1, 10, dtype=int)
            for idx in sample_times:
                if idx < len(time_steps):
                    attack_result = AttackResult(
                        attack_type=AttackType.RANDOM,
                        target_buses=[bus],
                        time_stamp=time_steps[idx],
                        load_changes={bus: attack_sequence.load_series[bus][idx]},
                        success=True
                    )
                    attack_sequence.attack_results.append(attack_result)
        
        # Calculate total energy
        attack_sequence.total_energy_injected = self._calculate_total_energy(
            attack_sequence, simulation_config.time_step
        )
        
        logger.info(f"Random attack generated: {attack_sequence.total_energy_injected:.2f} MWh total energy")
        return attack_sequence
    
    def generate_periodic_attack(
        self, 
        attack_config: AttackConfig, 
        simulation_config: SimulationConfig
    ) -> StaticAttackSequence:
        """
        Generate periodic attack sequence with sinusoidal oscillations
        
        Args:
            attack_config: Attack configuration
            simulation_config: Simulation configuration
            
        Returns:
            StaticAttackSequence: Complete attack sequence
        """
        if attack_config.attack_type != AttackType.PERIODIC:
            raise ValueError("Attack config must be PERIODIC type")
        
        logger.info(f"Generating periodic attack for buses {attack_config.target_buses}")
        
        time_steps = simulation_config.get_time_steps()
        attack_sequence = StaticAttackSequence(
            attack_config=attack_config,
            time_series=time_steps,
            load_series={},
            attack_results=[],
            total_energy_injected=0.0
        )
        
        # Initialize load series for each target bus
        for bus in attack_config.target_buses:
            attack_sequence.load_series[bus] = np.zeros_like(time_steps)
        
        # Calculate attack timing
        start_idx = int(attack_config.start_time / simulation_config.time_step)
        end_idx = int((attack_config.start_time + attack_config.duration) / simulation_config.time_step)
        end_idx = min(end_idx, len(time_steps))
        
        # Periodic attack parameters
        frequency = attack_config.periodic_frequency or 0.1  # Hz
        amplitude = attack_config.periodic_amplitude or attack_config.magnitude_mw
        
        # Generate time vector for attack period
        attack_time = time_steps[start_idx:end_idx] - attack_config.start_time
        
        for bus in attack_config.target_buses:
            # Distribute amplitude among buses
            bus_amplitude = amplitude / len(attack_config.target_buses)
            
            # Generate sinusoidal load variation
            # Using rectified sine wave to ensure positive load changes
            periodic_load = bus_amplitude * (np.sin(2 * np.pi * frequency * attack_time) + 1) / 2
            
            # Add phase shift for different buses to create realistic diversity
            phase_shift = (attack_config.target_buses.index(bus) * np.pi / 4)
            periodic_load = bus_amplitude * (np.sin(2 * np.pi * frequency * attack_time + phase_shift) + 1) / 2
            
            # Apply periodic changes during attack period
            attack_sequence.load_series[bus][start_idx:end_idx] = periodic_load
            
            # Record attack results (sample at peak and trough points)
            period_samples = int(1.0 / (frequency * simulation_config.time_step))
            sample_indices = range(start_idx, end_idx, max(1, period_samples // 4))
            
            for idx in sample_indices:
                if idx < len(time_steps):
                    attack_result = AttackResult(
                        attack_type=AttackType.PERIODIC,
                        target_buses=[bus],
                        time_stamp=time_steps[idx],
                        load_changes={bus: attack_sequence.load_series[bus][idx]},
                        success=True
                    )
                    attack_sequence.attack_results.append(attack_result)
        
        # Calculate total energy
        attack_sequence.total_energy_injected = self._calculate_total_energy(
            attack_sequence, simulation_config.time_step
        )
        
        logger.info(f"Periodic attack generated: {attack_sequence.total_energy_injected:.2f} MWh total energy")
        return attack_sequence
    
    def generate_multi_stage_attack(
        self,
        attack_configs: List[AttackConfig],
        simulation_config: SimulationConfig
    ) -> List[StaticAttackSequence]:
        """
        Generate multi-stage attack sequence
        
        Args:
            attack_configs: List of attack configurations for different stages
            simulation_config: Simulation configuration
            
        Returns:
            List[StaticAttackSequence]: List of attack sequences for each stage
        """
        logger.info(f"Generating multi-stage attack with {len(attack_configs)} stages")
        
        attack_sequences = []
        
        for i, config in enumerate(attack_configs):
            logger.info(f"Generating stage {i+1}: {config.attack_type.value}")
            
            if config.attack_type == AttackType.STEP:
                sequence = self.generate_step_attack(config, simulation_config)
            elif config.attack_type == AttackType.RANDOM:
                sequence = self.generate_random_attack(config, simulation_config)
            elif config.attack_type == AttackType.PERIODIC:
                sequence = self.generate_periodic_attack(config, simulation_config)
            else:
                logger.warning(f"Unsupported attack type for stage {i+1}: {config.attack_type}")
                continue
            
            attack_sequences.append(sequence)
        
        logger.info(f"Multi-stage attack generated with {len(attack_sequences)} sequences")
        return attack_sequences
    
    def _apply_smoothing_filter(self, signal: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing filter"""
        if len(signal) < window_size:
            return signal
        
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return smoothed
    
    def _calculate_total_energy(self, attack_sequence: StaticAttackSequence, time_step: float) -> float:
        """Calculate total energy injected during attack (MWh)"""
        total_energy = 0.0
        
        for bus, load_series in attack_sequence.load_series.items():
            # Energy = Power * Time (MW * hours)
            energy_mwh = np.sum(load_series) * time_step / 3600.0  # Convert seconds to hours
            total_energy += energy_mwh
        
        return total_energy

# ======================== STATIC ATTACK EXECUTOR ======================== #

class StaticAttackExecutor:
    """
    Executes static LAA attacks on power system
    
    This class applies static attack sequences to a power system case,
    modifying load values at target buses according to attack patterns.
    """
    
    def __init__(self):
        """Initialize static attack executor"""
        self.execution_history = []
        logger.info("Static Attack Executor initialized")
    
    def apply_attack_to_system(
        self,
        ieee39_case: Dict,
        attack_sequence: StaticAttackSequence,
        time_index: int,
        original_loads: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """
        Apply attack at specific time index to power system
        
        Args:
            ieee39_case: IEEE 39-bus PyPower case data
            attack_sequence: Attack sequence to apply
            time_index: Time step index
            original_loads: Original load values (for restoration)
            
        Returns:
            Dict of actual load changes applied {bus: load_change_mw}
        """
        applied_changes = {}
        
        try:
            for bus, load_series in attack_sequence.load_series.items():
                if time_index < len(load_series):
                    # Get load change for this time step
                    load_change = load_series[time_index]
                    
                    # Apply to PyPower case (convert bus number to index)
                    bus_idx = bus - 1  # Convert to 0-based index
                    
                    if bus_idx < len(ieee39_case['bus']):
                        # Add load change to current load (PD column)
                        if original_loads and bus in original_loads:
                            # Apply relative to original load
                            ieee39_case['bus'][bus_idx][2] = original_loads[bus] + load_change
                        else:
                            # Add to current load
                            ieee39_case['bus'][bus_idx][2] += load_change
                        
                        applied_changes[bus] = load_change
            
            logger.debug(f"Applied attack at time {time_index}: {applied_changes}")
            
        except Exception as e:
            logger.error(f"Failed to apply attack: {e}")
        
        return applied_changes
    
    def reset_system_loads(
        self,
        ieee39_case: Dict,
        original_loads: Dict[int, float]
    ):
        """
        Reset system loads to original values
        
        Args:
            ieee39_case: IEEE 39-bus PyPower case data
            original_loads: Original load values
        """
        try:
            for bus, original_load in original_loads.items():
                bus_idx = bus - 1  # Convert to 0-based index
                if bus_idx < len(ieee39_case['bus']):
                    ieee39_case['bus'][bus_idx][2] = original_load
            
            logger.debug("System loads reset to original values")
            
        except Exception as e:
            logger.error(f"Failed to reset loads: {e}")
    
    def get_current_loads(self, ieee39_case: Dict) -> Dict[int, float]:
        """
        Get current load values from system
        
        Args:
            ieee39_case: IEEE 39-bus PyPower case data
            
        Returns:
            Dict of current loads {bus: load_mw}
        """
        current_loads = {}
        
        try:
            for i, bus_data in enumerate(ieee39_case['bus']):
                bus_num = int(bus_data[0])  # Bus number
                load_mw = bus_data[2]       # PD (active power demand)
                current_loads[bus_num] = load_mw
        
        except Exception as e:
            logger.error(f"Failed to get current loads: {e}")
        
        return current_loads

# ======================== UTILITY FUNCTIONS ======================== #

def create_coordinated_attack_sequence(
    primary_buses: List[int],
    secondary_buses: List[int],
    simulation_config: SimulationConfig,
    delay_between_attacks: float = 10.0
) -> List[AttackConfig]:
    """
    Create coordinated multi-stage attack sequence
    
    Args:
        primary_buses: Primary target buses for first attack
        secondary_buses: Secondary target buses for follow-up attack
        simulation_config: Simulation configuration
        delay_between_attacks: Delay between attack stages (seconds)
        
    Returns:
        List of AttackConfig objects for coordinated sequence
    """
    from ..attacker.laa_config import LAA_Configurations
    
    # Primary attack: Step attack on primary buses
    primary_attack = LAA_Configurations.get_step_attack_config(
        buses=primary_buses,
        severity=AttackSeverity.HIGH
    )
    primary_attack.start_time = 5.0
    primary_attack.duration = 20.0
    
    # Secondary attack: Periodic attack on secondary buses (delayed)
    secondary_attack = LAA_Configurations.get_periodic_attack_config(
        buses=secondary_buses,
        severity=AttackSeverity.MEDIUM
    )
    secondary_attack.start_time = primary_attack.start_time + delay_between_attacks
    secondary_attack.duration = 25.0
    
    return [primary_attack, secondary_attack]

if __name__ == "__main__":
    # Test static LAA generator
    print("Testing Static LAA Generator...")
    
    from ..attacker.laa_config import LAA_Configurations, TargetBusSelections, SimulationConfig
    
    # Create test configurations
    target_buses = TargetBusSelections.get_high_impact_selection(2)
    sim_config = SimulationConfig(simulation_time=60.0, time_step=0.1)
    
    # Initialize generator
    generator = StaticLAAGenerator(random_seed=42)
    
    # Test step attack
    step_config = LAA_Configurations.get_step_attack_config(target_buses, AttackSeverity.MEDIUM)
    step_sequence = generator.generate_step_attack(step_config, sim_config)
    print(f" Step attack: {step_sequence.total_energy_injected:.2f} MWh")
    
    # Test random attack
    random_config = LAA_Configurations.get_random_attack_config(target_buses, AttackSeverity.MEDIUM)
    random_sequence = generator.generate_random_attack(random_config, sim_config)
    print(f" Random attack: {random_sequence.total_energy_injected:.2f} MWh")
    
    # Test periodic attack
    periodic_config = LAA_Configurations.get_periodic_attack_config(target_buses, AttackSeverity.MEDIUM)
    periodic_sequence = generator.generate_periodic_attack(periodic_config, sim_config)
    print(f" Periodic attack: {periodic_sequence.total_energy_injected:.2f} MWh")
    
    print(" All static LAA tests passed")
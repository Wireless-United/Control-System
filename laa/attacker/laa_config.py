#!/usr/bin/env python3
"""
Load-Altering Attacks (LAA) Configuration Module

This module provides comprehensive configuration utilities for LAA simulation parameters,
attack scenarios, and system inertia conditions. It supports both static and dynamic
attack types with IEEE-compliant parameter ranges.

Key Components:
- InertiaCondition: Defines system inertia states (LOW/HIGH)
- AttackType: Defines various attack patterns (STEP/RANDOM/PERIODIC/FEEDBACK/MULTI_STAGE)
- AttackSeverity: Defines attack severity levels (LOW/MEDIUM/HIGH/CRITICAL)
- SystemInertiaConfig: Configuration for power system inertia conditions
- AttackConfig: Configuration for attack parameters and behavior
- SimulationConfig: Configuration for overall simulation settings

Author: Pranaav
Date: October 2025
Version: 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ======================== ENUMS AND CONSTANTS ======================== #

class InertiaCondition(Enum):
    """
    System inertia conditions representing different DER penetration scenarios.
    
    LOW: Low inertia system with high DER penetration (60%+)
         - Faster frequency response to disturbances
         - Reduced system damping
         - More susceptible to LAA attacks
    
    HIGH: High inertia system with low DER penetration (<30%)
          - Slower frequency response to disturbances
          - Higher system damping
          - More resilient to LAA attacks
    """
    LOW = "low_inertia"
    HIGH = "high_inertia"

class AttackType(Enum):
    """
    Types of Load-Altering Attacks (LAA) supported by the framework.
    
    STEP: Sudden step change in load demand
          - Instantaneous load increase/decrease
          - Tests system frequency response
          - Duration-based attack
    
    RANDOM: Random load variations with Gaussian noise
            - Stochastic attack pattern
            - Variance-based magnitude
            - Tests system stability margins
    
    PERIODIC: Sinusoidal load oscillations
              - Frequency-based attack
              - Can trigger resonance effects
              - Tests control system response
    
    FEEDBACK: Dynamic PID-controlled attack
              - Real-time system state feedback
              - Adaptive attack magnitude
              - Most sophisticated attack type
    
    MULTI_STAGE: Sequential attack with multiple phases
                 - Combination of attack types
                 - Time-staged execution
                 - Most complex attack scenario
    """
    STEP = "step_attack"
    RANDOM = "random_attack"
    PERIODIC = "periodic_attack"
    FEEDBACK = "feedback_attack"
    MULTI_STAGE = "multi_stage_attack"

class AttackSeverity(Enum):
    """
    Attack severity levels determining impact magnitude.
    
    LOW (1): Minor load variation (15-25 MW)
             - Minimal system impact
             - Testing/reconnaissance level
    
    MEDIUM (2): Moderate load variation (30-50 MW)
                - Noticeable frequency deviation
                - Standard attack scenario
    
    HIGH (3): Significant load variation (60-100 MW)
              - Major frequency/voltage impact
              - Critical attack level
    
    CRITICAL (4): Extreme load variation (120-200 MW)
                  - Potential system instability
                  - Blackout-level attack
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# ======================== CONFIGURATION CLASSES ======================== #

@dataclass
class SystemInertiaConfig:
    """
    Configuration for power system inertia conditions and DER penetration.
    
    This class defines the system's inherent dynamic response characteristics
    based on the mix of conventional generators vs. DER (Distributed Energy Resources).
    
    Attributes:
        condition: InertiaCondition enum (LOW or HIGH)
                   Determines the overall system inertia state
        
        generator_inertia_multiplier: float (0.3 - 2.0)
                                      Multiplier for generator inertia constant H
                                      - LOW inertia: 0.3-0.7
                                      - HIGH inertia: 1.2-2.0
        
        damping_multiplier: float (0.5 - 1.5)
                           Multiplier for system damping coefficient D
                           - Higher values = faster oscillation decay
                           - Lower values = sustained oscillations
        
        der_penetration_level: float (0.0 - 1.0)
                              Percentage of DER in total generation mix
                              - 0.0 = 0% (all conventional generators)
                              - 1.0 = 100% (all DER/inverter-based)
                              - Typical LOW inertia: 0.5-0.8
                              - Typical HIGH inertia: 0.1-0.3
        
        frequency_response_rate: float (0.05 - 0.5 Hz/s)
                                Rate of frequency change after disturbance
                                - LOW inertia: 0.05-0.15 Hz/s (fast)
                                - HIGH inertia: 0.2-0.5 Hz/s (slow)
        
        voltage_regulation_strength: float (0.5 - 1.5)
                                    Capability for voltage regulation
                                    - Higher = better voltage control
                                    - Affected by DER penetration
    
    Example:
        >>> config = SystemInertiaConfig(
        ...     condition=InertiaCondition.LOW,
        ...     generator_inertia_multiplier=0.5,
        ...     damping_multiplier=0.7,
        ...     der_penetration_level=0.6,
        ...     frequency_response_rate=0.1,
        ...     voltage_regulation_strength=0.8
        ... )
    """
    condition: InertiaCondition
    generator_inertia_multiplier: float  # Multiplier for H values (0.3-2.0)
    damping_multiplier: float           # Multiplier for D values (0.5-1.5)
    der_penetration_level: float        # DER penetration percentage (0.0-1.0)
    frequency_response_rate: float      # Hz/s response rate (0.05-0.5)
    voltage_regulation_strength: float  # Voltage regulation capability (0.5-1.5)
    
    def __post_init__(self):
        """
        Validate configuration parameters after initialization.
        
        Ensures all parameters are within acceptable physical and engineering limits.
        Raises ValueError if any parameter is out of range.
        """
        if self.generator_inertia_multiplier <= 0:
            raise ValueError("Generator inertia multiplier must be positive")
        if self.damping_multiplier <= 0:
            raise ValueError("Damping multiplier must be positive")
        if not 0 <= self.der_penetration_level <= 1:
            raise ValueError("DER penetration level must be between 0 and 1")

@dataclass
class AttackConfig:
    """
    Configuration for LAA attack parameters and behavior.
    
    This class defines all parameters needed to execute a Load-Altering Attack
    including target selection, timing, magnitude, and type-specific parameters.
    
    Attributes:
        attack_type: AttackType enum
                     Type of attack pattern to execute
        
        target_buses: List[int]
                     List of IEEE 39-bus system bus numbers to attack (1-39)
                     - Load buses: [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29]
                     - Critical buses: [20, 21, 23, 24, 27, 28, 29]
        
        severity: AttackSeverity enum
                  Overall attack severity level (LOW/MEDIUM/HIGH/CRITICAL)
        
        duration: float (seconds)
                 Total attack duration
                 - Typical range: 5-60 seconds
                 - Short attack: 5-10s
                 - Medium attack: 10-30s
                 - Long attack: 30-60s
        
        magnitude_mw: float (MW)
                     Base attack magnitude in megawatts
                     - LOW: 15-25 MW
                     - MEDIUM: 30-50 MW
                     - HIGH: 60-100 MW
                     - CRITICAL: 120-200 MW
        
        start_time: float (seconds, default=0.0)
                   Time when attack begins in simulation
        
        # STATIC ATTACK PARAMETERS (for STEP/RANDOM/PERIODIC)
        
        step_magnitude: Optional[float] (MW)
                       Step size for STEP attacks
                       - Should match magnitude_mw for simple step
                       - Can vary for multi-step scenarios
        
        random_variance: Optional[float] (MW)
                        Standard deviation for RANDOM attacks
                        - Typical range: 5-40 MW
                        - Defines load variation spread
        
        periodic_frequency: Optional[float] (Hz)
                           Oscillation frequency for PERIODIC attacks
                           - Typical range: 0.05-0.5 Hz
                           - Low freq (0.05-0.1 Hz): Inter-area oscillations
                           - Med freq (0.1-0.3 Hz): Local oscillations
                           - High freq (0.3-0.5 Hz): Control system testing
        
        periodic_amplitude: Optional[float] (MW)
                           Oscillation amplitude for PERIODIC attacks
                           - Peak-to-peak variation
                           - Should match severity level
        
        # DYNAMIC ATTACK PARAMETERS (for FEEDBACK)
        
        feedback_gain: Optional[float]
                      PID controller gain for FEEDBACK attacks
                      - Proportional gain: 10-100
                      - Higher gain = more aggressive response
                      - Must be tuned to avoid instability
        
        frequency_threshold: Optional[float] (Hz)
                            Frequency deviation threshold for FEEDBACK activation
                            - Typical: 0.05-0.2 Hz
                            - Attack activates when |f - f_nominal| > threshold
        
        adaptation_rate: Optional[float] (seconds)
                        Time constant for FEEDBACK adaptation
                        - How quickly attack responds to changes
                        - Typical range: 0.1-1.0 seconds
    
    Example:
        >>> # Step attack configuration
        >>> step_config = AttackConfig(
        ...     attack_type=AttackType.STEP,
        ...     target_buses=[20, 21, 23],
        ...     severity=AttackSeverity.MEDIUM,
        ...     duration=30.0,
        ...     magnitude_mw=50.0,
        ...     step_magnitude=50.0
        ... )
        
        >>> # Feedback attack configuration
        >>> feedback_config = AttackConfig(
        ...     attack_type=AttackType.FEEDBACK,
        ...     target_buses=[24, 27],
        ...     severity=AttackSeverity.HIGH,
        ...     duration=45.0,
        ...     magnitude_mw=100.0,
        ...     feedback_gain=50.0,
        ...     frequency_threshold=0.1,
        ...     adaptation_rate=0.5
        ... )
    """
    attack_type: AttackType
    target_buses: List[int]              # List of bus numbers to attack (1-39)
    severity: AttackSeverity
    duration: float                      # Attack duration in seconds
    magnitude_mw: float                  # Attack magnitude in MW
    start_time: float = 0.0             # Start time in seconds (default: 0.0)
    
    # Static attack parameters (optional, type-specific)
    step_magnitude: Optional[float] = None      # Step size for STEP attacks (MW)
    random_variance: Optional[float] = None     # Variance for RANDOM attacks (MW)
    periodic_frequency: Optional[float] = None  # Frequency for PERIODIC attacks (Hz)
    periodic_amplitude: Optional[float] = None  # Amplitude for PERIODIC attacks (MW)
    
    # Dynamic attack parameters (optional, type-specific)
    feedback_gain: Optional[float] = None           # PID gain for FEEDBACK attacks
    frequency_threshold: Optional[float] = None     # Frequency threshold for FEEDBACK (Hz)
    adaptation_rate: Optional[float] = None         # Adaptation time constant (seconds)
    
    def __post_init__(self):
        """
        Validate attack configuration parameters.
        
        Ensures attack is properly configured with valid buses, magnitude, and duration.
        Raises ValueError if any critical parameter is invalid.
        """
        if not self.target_buses:
            raise ValueError("At least one target bus must be specified")
        if self.magnitude_mw <= 0:
            raise ValueError("Attack magnitude must be positive")
        if self.duration <= 0:
            raise ValueError("Attack duration must be positive")

@dataclass
class SimulationConfig:
    """
    Overall simulation configuration and analysis parameters.
    
    This class defines the simulation time parameters, numerical solver settings,
    and analysis thresholds for LAA simulation and evaluation.
    
    Attributes:
        simulation_time: float (seconds, default=60.0)
                        Total duration of simulation
                        - Minimum: 10 seconds (basic testing)
                        - Typical: 30-60 seconds (standard analysis)
                        - Extended: 120-300 seconds (long-term dynamics)
        
        time_step: float (seconds, default=0.01)
                  Numerical integration time step
                  - 0.01s (10ms): Standard for power system transients
                  - 0.001s (1ms): High-resolution analysis
                  - Must be small enough to capture attack dynamics
        
        power_flow_tolerance: float (default=1e-6)
                             Convergence tolerance for power flow solver
                             - Newton-Raphson convergence criterion
                             - Smaller = more accurate but slower
        
        max_iterations: int (default=100)
                       Maximum iterations for power flow solver
                       - If exceeded, indicates non-convergence
                       - Typical convergence: 3-10 iterations
        
        frequency_nominal: float (Hz, default=50.0)
                          Nominal system frequency
                          - 50 Hz: European/Asian systems
                          - 60 Hz: North American systems
        
        voltage_nominal: float (pu, default=1.0)
                        Nominal system voltage in per-unit
                        - Base value for voltage calculations
                        - Per-unit system normalization
        
        # ANALYSIS PARAMETERS
        
        stability_window: float (seconds, default=5.0)
                         Time window for stability analysis
                         - Observes system behavior after disturbance
                         - Checks for oscillation decay
        
        oscillation_threshold: float (pu, default=0.02)
                              Voltage oscillation detection threshold
                              - 2% voltage variation considered oscillatory
                              - IEEE stability criterion
        
        frequency_deviation_limit: float (Hz, default=1.0)
                                  Maximum acceptable frequency deviation
                                  - 1.0 Hz: Emergency limit
                                  - 0.5 Hz: IEEE 1547 trip limit
                                  - 0.036 Hz: IEEE 1547 deadband
        
        voltage_deviation_limit: float (pu, default=0.1)
                                Maximum acceptable voltage deviation
                                - 0.1 pu (10%): Standard limit
                                - 0.12 pu (12%): Emergency limit
                                - IEEE 1547: 0.88-1.10 pu range
    
    Example:
        >>> config = SimulationConfig(
        ...     simulation_time=30.0,
        ...     time_step=0.01,
        ...     frequency_nominal=50.0,
        ...     frequency_deviation_limit=0.5
        ... )
    """
    simulation_time: float = 60.0        # Total simulation time (seconds)
    time_step: float = 0.01             # Simulation time step (seconds)
    power_flow_tolerance: float = 1e-6   # Power flow convergence tolerance
    max_iterations: int = 100           # Maximum power flow iterations
    frequency_nominal: float = 50.0     # Nominal frequency (Hz) [50 or 60]
    voltage_nominal: float = 1.0        # Nominal voltage (pu)
    
    # Analysis parameters for system stability assessment
    stability_window: float = 5.0       # Time window for stability analysis (seconds)
    oscillation_threshold: float = 0.02  # Voltage oscillation threshold (pu) [2%]
    frequency_deviation_limit: float = 1.0  # Frequency deviation limit (Hz)
    voltage_deviation_limit: float = 0.1    # Voltage deviation limit (pu) [10%]
    
    def get_time_steps(self) -> np.ndarray:
        """
        Generate array of simulation time steps.
        
        Returns:
            np.ndarray: Time steps from 0 to simulation_time with interval time_step
        
        Example:
            >>> config = SimulationConfig(simulation_time=10.0, time_step=0.1)
            >>> time_steps = config.get_time_steps()
            >>> len(time_steps)  # 101 time points (0.0, 0.1, 0.2, ..., 10.0)
            101
        """
        return np.arange(0, self.simulation_time + self.time_step, self.time_step)

# ======================== PREDEFINED CONFIGURATIONS ======================== #

class LAA_Configurations:
    """
    Predefined LAA simulation configurations for common scenarios.
    
    This class provides factory methods to create standard configurations
    for different attack types and system conditions. Eliminates the need
    for manual parameter tuning in most cases.
    
    Methods:
        get_low_inertia_config(): Get low inertia system configuration
        get_high_inertia_config(): Get high inertia system configuration
        get_step_attack_config(): Get step attack configuration
        get_random_attack_config(): Get random attack configuration
        get_periodic_attack_config(): Get periodic attack configuration
        get_feedback_attack_config(): Get feedback attack configuration
    """
    
    @staticmethod
    def get_low_inertia_config() -> SystemInertiaConfig:
        """
        Configuration for low inertia system (high DER penetration).
        
        Represents modern power grids with significant renewable integration:
        - 60% DER penetration (solar, wind, battery storage)
        - Reduced rotational inertia from conventional generators
        - Faster frequency response to disturbances
        - More vulnerable to LAA attacks
        
        Returns:
            SystemInertiaConfig: Low inertia system configuration
        
        Typical Use Cases:
            - Modern grids with high solar/wind penetration
            - Island systems with inverter-based generation
            - Future grid scenarios (2030-2050)
            - Worst-case vulnerability analysis
        """
        return SystemInertiaConfig(
            condition=InertiaCondition.LOW,
            generator_inertia_multiplier=0.5,    # 50% reduced generator inertia
            damping_multiplier=0.7,              # 30% reduced damping
            der_penetration_level=0.6,           # 60% DER penetration
            frequency_response_rate=0.1,         # Slower frequency response (0.1 Hz/s)
            voltage_regulation_strength=0.8      # Weaker voltage regulation
        )
    
    @staticmethod
    def get_high_inertia_config() -> SystemInertiaConfig:
        """
        Configuration for high inertia system (low DER penetration).
        
        Represents traditional power grids with conventional generation:
        - 20% DER penetration (mostly hydro, some solar/wind)
        - High rotational inertia from synchronous generators
        - Slower but more stable frequency response
        - More resilient to LAA attacks
        
        Returns:
            SystemInertiaConfig: High inertia system configuration
        
        Typical Use Cases:
            - Traditional grids with fossil/nuclear/hydro generation
            - Systems with strict stability requirements
            - Current grid scenarios (2020-2025)
            - Best-case resilience analysis
        """
        return SystemInertiaConfig(
            condition=InertiaCondition.HIGH,
            generator_inertia_multiplier=1.5,    # 50% increased generator inertia
            damping_multiplier=1.3,              # 30% increased damping
            der_penetration_level=0.2,           # 20% DER penetration
            frequency_response_rate=0.3,         # Faster frequency response (0.3 Hz/s)
            voltage_regulation_strength=1.2      # Stronger voltage regulation
        )
    
    @staticmethod
    def get_step_attack_config(buses: List[int], severity: AttackSeverity) -> AttackConfig:
        """
        Configuration for step attack (sudden load change).
        
        Step attacks represent instantaneous load changes that test the
        system's primary frequency response and generator governors.
        
        Args:
            buses: List of target bus numbers (1-39 for IEEE 39-bus system)
            severity: Attack severity level (LOW/MEDIUM/HIGH/CRITICAL)
        
        Returns:
            AttackConfig: Configured step attack
        
        Attack Characteristics:
            - Instantaneous load increase at start_time
            - Constant magnitude for duration
            - Tests frequency regulation reserves
            - Magnitude based on severity:
                * LOW: 20 MW (minor disturbance)
                * MEDIUM: 50 MW (moderate disturbance)
                * HIGH: 100 MW (major disturbance)
                * CRITICAL: 200 MW (extreme disturbance)
        
        Example:
            >>> config = LAA_Configurations.get_step_attack_config(
            ...     buses=[20, 21, 23],
            ...     severity=AttackSeverity.MEDIUM
            ... )
            >>> print(config.magnitude_mw)  # 50.0 MW
        """
        # Map severity to attack magnitude (MW)
        magnitude_map = {
            AttackSeverity.LOW: 20.0,        # 20 MW - Minor disturbance
            AttackSeverity.MEDIUM: 50.0,     # 50 MW - Moderate disturbance
            AttackSeverity.HIGH: 100.0,      # 100 MW - Major disturbance
            AttackSeverity.CRITICAL: 200.0   # 200 MW - Extreme disturbance
        }
        
        return AttackConfig(
            attack_type=AttackType.STEP,
            target_buses=buses,
            severity=severity,
            duration=30.0,                    # 30 second attack duration
            magnitude_mw=magnitude_map[severity],
            step_magnitude=magnitude_map[severity]
        )
    
    @staticmethod
    def get_random_attack_config(buses: List[int], severity: AttackSeverity) -> AttackConfig:
        """Configuration for random attack"""
        magnitude_map = {
            AttackSeverity.LOW: 15.0,
            AttackSeverity.MEDIUM: 30.0,
            AttackSeverity.HIGH: 60.0,
            AttackSeverity.CRITICAL: 120.0
        }
        
        variance_map = {
            AttackSeverity.LOW: 5.0,
            AttackSeverity.MEDIUM: 10.0,
            AttackSeverity.HIGH: 20.0,
            AttackSeverity.CRITICAL: 40.0
        }
        
        return AttackConfig(
            attack_type=AttackType.RANDOM,
            target_buses=buses,
            severity=severity,
            duration=45.0,
            magnitude_mw=magnitude_map[severity],
            random_variance=variance_map[severity]
        )
    
    @staticmethod
    def get_periodic_attack_config(buses: List[int], severity: AttackSeverity) -> AttackConfig:
        """Configuration for periodic attack"""
        amplitude_map = {
            AttackSeverity.LOW: 25.0,
            AttackSeverity.MEDIUM: 50.0,
            AttackSeverity.HIGH: 75.0,
            AttackSeverity.CRITICAL: 150.0
        }
        
        return AttackConfig(
            attack_type=AttackType.PERIODIC,
            target_buses=buses,
            severity=severity,
            duration=60.0,
            magnitude_mw=amplitude_map[severity],
            periodic_frequency=0.1,  # 0.1 Hz oscillation
            periodic_amplitude=amplitude_map[severity]
        )
    
    @staticmethod
    def get_feedback_attack_config(buses: List[int], severity: AttackSeverity) -> AttackConfig:
        """Configuration for feedback-based dynamic attack"""
        gain_map = {
            AttackSeverity.LOW: 10.0,
            AttackSeverity.MEDIUM: 25.0,
            AttackSeverity.HIGH: 50.0,
            AttackSeverity.CRITICAL: 100.0
        }
        
        return AttackConfig(
            attack_type=AttackType.FEEDBACK,
            target_buses=buses,
            severity=severity,
            duration=45.0,
            magnitude_mw=100.0,
            feedback_gain=gain_map[severity],
            frequency_threshold=0.1,  # 0.1 Hz threshold
            adaptation_rate=0.5       # 0.5 s adaptation time
        )

# ======================== TARGET BUS SELECTIONS ======================== #

class TargetBusSelections:
    """Predefined target bus selections for different attack scenarios"""
    
    # Critical load buses in IEEE 39-bus system
    CRITICAL_LOAD_BUSES = [20, 21, 23, 24, 27, 28, 29]
    
    # High load density buses
    HIGH_LOAD_BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21]
    
    # Strategic buses (near generators)
    STRATEGIC_BUSES = [4, 7, 9, 16, 18, 21, 23, 24]
    
    # Transmission critical buses
    TRANSMISSION_CRITICAL = [6, 9, 10, 13, 14, 19, 22, 25]
    
    @classmethod
    def get_random_selection(cls, count: int = 3) -> List[int]:
        """Get random selection of target buses"""
        available_buses = list(range(1, 40))  # IEEE 39-bus system
        return np.random.choice(available_buses, size=count, replace=False).tolist()
    
    @classmethod
    def get_high_impact_selection(cls, count: int = 3) -> List[int]:
        """Get high-impact target buses"""
        return cls.CRITICAL_LOAD_BUSES[:count]
    
    @classmethod
    def get_strategic_selection(cls, count: int = 3) -> List[int]:
        """Get strategic target buses"""
        return cls.STRATEGIC_BUSES[:count]

# ======================== UTILITY FUNCTIONS ======================== #

def validate_simulation_setup(
    inertia_config: SystemInertiaConfig,
    attack_config: AttackConfig,
    sim_config: SimulationConfig
) -> bool:
    """
    Validate complete simulation setup for consistency
    
    Args:
        inertia_config: System inertia configuration
        attack_config: Attack configuration
        sim_config: Simulation configuration
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check if attack duration fits within simulation time
        if attack_config.duration > sim_config.simulation_time:
            logger.warning("Attack duration exceeds simulation time")
            return False
        
        # Check if target buses are valid (1-39 for IEEE 39-bus)
        if not all(1 <= bus <= 39 for bus in attack_config.target_buses):
            logger.error("Invalid target bus numbers (must be 1-39)")
            return False
        
        # Check time step vs attack parameters
        if attack_config.attack_type == AttackType.PERIODIC:
            if attack_config.periodic_frequency:
                min_time_step = 1.0 / (20 * attack_config.periodic_frequency)
                if sim_config.time_step > min_time_step:
                    logger.warning(f"Time step may be too large for periodic attack")
        
        logger.info("Simulation configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def create_default_simulation_setup() -> Tuple[SystemInertiaConfig, AttackConfig, SimulationConfig]:
    """
    Create default simulation setup for quick testing
    
    Returns:
        Tuple of (inertia_config, attack_config, sim_config)
    """
    inertia_config = LAA_Configurations.get_low_inertia_config()
    attack_config = LAA_Configurations.get_step_attack_config(
        buses=TargetBusSelections.get_high_impact_selection(2),
        severity=AttackSeverity.MEDIUM
    )
    sim_config = SimulationConfig()
    
    return inertia_config, attack_config, sim_config

if __name__ == "__main__":
    # Test configuration creation
    print("Testing LAA Configuration Module...")
    
    # Test inertia configurations
    low_inertia = LAA_Configurations.get_low_inertia_config()
    high_inertia = LAA_Configurations.get_high_inertia_config()
    
    print(f"Low Inertia Config: {low_inertia}")
    print(f"High Inertia Config: {high_inertia}")
    
    # Test attack configurations
    target_buses = TargetBusSelections.get_high_impact_selection(2)
    step_attack = LAA_Configurations.get_step_attack_config(target_buses, AttackSeverity.MEDIUM)
    
    print(f"Step Attack Config: {step_attack}")
    
    # Test default setup
    inertia, attack, sim = create_default_simulation_setup()
    is_valid = validate_simulation_setup(inertia, attack, sim)
    
    print(f"Default setup validation: {'PASSED' if is_valid else 'FAILED'}")
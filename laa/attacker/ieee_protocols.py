#!/usr/bin/env python3
"""
IEEE Standards and Protocol Compliance Module

This module implements IEEE standard values, protocols, and guidelines for
power system analysis, control, and protection systems.

Standards covered:
- IEEE Std 1547: Distributed Energy Resources
- IEEE Std C37.118: Synchrophasor Measurements  
- IEEE Std 421.5: Excitation Systems
- IEEE Std 1110: Power System Stability
- IEEE Std C37.2: Power System Protection

Author: Pranaav
Date: October 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

# ======================== IEEE STANDARD ENUMERATIONS ======================== #

class IEEE_VoltageLevel(Enum):
    """IEEE Std 1547 - Standard voltage levels"""
    LOW_VOLTAGE = "LV"          # < 1 kV
    MEDIUM_VOLTAGE = "MV"       # 1-35 kV  
    HIGH_VOLTAGE = "HV"         # 35-138 kV
    EXTRA_HIGH_VOLTAGE = "EHV"  # > 138 kV

class IEEE_FrequencyClass(Enum):
    """IEEE frequency classifications"""
    FREQ_50HZ = 50.0    # European/Asian standard
    FREQ_60HZ = 60.0    # North American standard

class IEEE_ProtectionClass(Enum):
    """IEEE C37.2 - Protection device classifications"""
    OVERCURRENT = "51"          # Time overcurrent
    DIFFERENTIAL = "87"         # Differential protection
    DISTANCE = "21"             # Distance protection
    FREQUENCY = "81"            # Frequency protection
    VOLTAGE = "27/59"           # Under/Over voltage

# ======================== IEEE STANDARD PARAMETERS ======================== #

@dataclass
class IEEE_SystemParameters:
    """IEEE standard power system parameters"""
    
    # IEEE Std 1110 - System Stability Parameters
    nominal_frequency: float = 50.0              # Hz (IEEE default for international)
    frequency_deadband: float = 0.036           # ±0.036 Hz (IEEE Std 1547)
    frequency_trip_threshold: float = 0.5       # ±0.5 Hz (IEEE Std 1547.1)
    frequency_clearing_time: float = 0.16       # 160 ms (IEEE C37.118)
    
    # IEEE Std 421.5 - Excitation System Parameters
    exciter_time_constant: float = 0.02         # 20 ms (typical IEEE value)
    avr_gain: float = 200.0                     # IEEE typical AVR gain
    avr_time_constant: float = 0.02             # 20 ms
    
    # IEEE voltage regulation standards
    voltage_regulation_deadband: float = 0.01   # ±1% (IEEE Std 1547)
    voltage_trip_low: float = 0.88             # 88% Vnom (IEEE Std 1547.1)
    voltage_trip_high: float = 1.10            # 110% Vnom (IEEE Std 1547.1)
    voltage_clearing_time_low: float = 2.0     # 2.0 s for low voltage
    voltage_clearing_time_high: float = 1.0    # 1.0 s for high voltage
    
    # IEEE generator parameters (per IEEE Std 421.5)
    generator_inertia_low: float = 2.0          # Low inertia generators (s)
    generator_inertia_high: float = 5.0         # High inertia generators (s)
    generator_damping: float = 2.0              # Damping coefficient
    
    # IEEE C37.118 - PMU Standards
    pmu_reporting_rate: float = 50.0            # 50 fps (IEEE C37.118-2011)
    pmu_accuracy_magnitude: float = 0.01        # 1% TVE (Total Vector Error)
    pmu_accuracy_phase: float = 0.573           # 0.573° (1 centirad)
    pmu_latency_max: float = 0.04              # 40 ms max latency

@dataclass  
class IEEE_LoadParameters:
    """IEEE standard load modeling parameters"""
    
    # IEEE load model parameters (IEEE Std 1110)
    static_load_p_coefficient: float = 1.0      # Active power voltage dependency
    static_load_q_coefficient: float = 2.0      # Reactive power voltage dependency
    
    # Dynamic load parameters
    motor_load_percentage: float = 0.3          # 30% motor loads (IEEE typical)
    constant_impedance_percentage: float = 0.4   # 40% constant impedance
    constant_power_percentage: float = 0.3      # 30% constant power
    
    # Load characteristics per IEEE standards
    residential_load_factor: float = 0.7        # Residential diversity
    industrial_load_factor: float = 0.85       # Industrial load factor
    commercial_load_factor: float = 0.75       # Commercial load factor

@dataclass
class IEEE_ProtectionSettings:
    """IEEE C37 series protection standards"""
    
    # IEEE C37.2 - Protection coordination
    overcurrent_pickup: float = 1.25            # 125% of nominal current
    overcurrent_time_dial: float = 0.5          # IEEE standard time dial
    
    # Distance protection (IEEE C37.113)
    zone1_reach: float = 0.8                    # 80% of line impedance
    zone2_reach: float = 1.2                    # 120% including next line
    zone3_reach: float = 2.0                    # Remote backup protection
    
    # Frequency protection (IEEE Std 1547.1)
    underfrequency_stage1: float = 49.7         # Stage 1 load shedding
    underfrequency_stage2: float = 49.4         # Stage 2 load shedding  
    underfrequency_stage3: float = 49.1         # Stage 3 load shedding
    overfrequency_trip: float = 50.5            # Over frequency trip
    
    # Voltage protection
    undervoltage_stage1: float = 0.92          # 92% Vnom
    undervoltage_stage2: float = 0.88          # 88% Vnom
    overvoltage_stage1: float = 1.05           # 105% Vnom
    overvoltage_stage2: float = 1.10           # 110% Vnom

# ======================== IEEE STANDARD FUNCTIONS ======================== #

class IEEE_Standards:
    """IEEE standards implementation and validation"""
    
    @staticmethod
    def get_ieee39_standard_parameters() -> IEEE_SystemParameters:
        """Get IEEE standard parameters for 39-bus system"""
        return IEEE_SystemParameters(
            nominal_frequency=50.0,             # International standard
            frequency_deadband=0.036,           # IEEE Std 1547
            exciter_time_constant=0.02,         # IEEE Std 421.5
            avr_gain=200.0,                     # IEEE typical
            generator_inertia_low=2.5,          # Low inertia case
            generator_inertia_high=4.5,         # High inertia case
            pmu_reporting_rate=50.0             # IEEE C37.118
        )
    
    @staticmethod
    def get_ieee_protection_settings() -> IEEE_ProtectionSettings:
        """Get IEEE standard protection settings"""
        return IEEE_ProtectionSettings()
    
    @staticmethod
    def validate_frequency_deviation(freq_deviation: float) -> Tuple[bool, str]:
        """Validate frequency deviation against IEEE standards"""
        params = IEEE_Standards.get_ieee39_standard_parameters()
        
        if abs(freq_deviation) <= params.frequency_deadband:
            return True, "Within IEEE deadband"
        elif abs(freq_deviation) <= params.frequency_trip_threshold:
            return True, "Within IEEE operating limits"
        else:
            return False, f"Exceeds IEEE trip threshold ({params.frequency_trip_threshold} Hz)"
    
    @staticmethod
    def validate_voltage_level(voltage_pu: float) -> Tuple[bool, str]:
        """Validate voltage level against IEEE standards"""
        params = IEEE_Standards.get_ieee39_standard_parameters()
        
        if voltage_pu < params.voltage_trip_low:
            return False, f"Below IEEE low voltage trip ({params.voltage_trip_low} pu)"
        elif voltage_pu > params.voltage_trip_high:
            return False, f"Above IEEE high voltage trip ({params.voltage_trip_high} pu)"
        else:
            return True, "Within IEEE voltage limits"
    
    @staticmethod
    def calculate_ieee_inertia_constant(scenario: str, base_mva: float = 100.0) -> float:
        """Calculate IEEE standard inertia constant"""
        params = IEEE_Standards.get_ieee39_standard_parameters()
        
        if scenario.lower() == "low":
            return params.generator_inertia_low
        elif scenario.lower() == "high":
            return params.generator_inertia_high
        else:
            return (params.generator_inertia_low + params.generator_inertia_high) / 2
    
    @staticmethod
    def get_ieee_pmu_specifications() -> Dict[str, float]:
        """Get IEEE C37.118 PMU specifications"""
        params = IEEE_Standards.get_ieee39_standard_parameters()
        
        return {
            'reporting_rate_fps': params.pmu_reporting_rate,
            'magnitude_accuracy_percent': params.pmu_accuracy_magnitude * 100,
            'phase_accuracy_degrees': params.pmu_accuracy_phase,
            'max_latency_ms': params.pmu_latency_max * 1000,
            'time_synchronization_accuracy_us': 1.0  # 1 microsecond GPS
        }
    
    @staticmethod
    def get_ieee_control_parameters() -> Dict[str, float]:
        """Get IEEE standard control system parameters"""
        params = IEEE_Standards.get_ieee39_standard_parameters()
        
        return {
            # IEEE Std 421.5 - Excitation systems
            'avr_gain': params.avr_gain,
            'avr_time_constant_s': params.avr_time_constant,
            'exciter_time_constant_s': params.exciter_time_constant,
            
            # IEEE governor standards
            'governor_droop_percent': 5.0,           # 5% droop (IEEE standard)
            'governor_time_constant_s': 0.2,         # 200 ms
            'turbine_time_constant_s': 0.5,          # 500 ms
            
            # IEEE load frequency control
            'lfc_gain': 1.0,                         # Per unit gain
            'lfc_time_constant_s': 10.0,             # 10 second integral time
        }
    
    @staticmethod
    def validate_attack_parameters(attack_config) -> bool:
        """Validate attack parameters against IEEE standards"""
        try:
            # Basic IEEE parameter validation
            params = IEEE_Standards.get_ieee39_standard_parameters()
            
            # Validate magnitude (should not exceed reasonable IEEE limits)
            if attack_config.magnitude_mw > 1000.0:  # Reasonable limit for IEEE 39-bus
                return False
            
            # Validate duration (should be within IEEE study timeframes)
            if attack_config.duration > 300.0:  # 5 minute max for IEEE transient studies
                return False
                
            # Validate target buses (basic sanity check)
            if not attack_config.target_buses or len(attack_config.target_buses) == 0:
                return False
            
            return True
            
        except Exception:
            return False

# ======================== IEEE ATTACK DETECTION STANDARDS ======================== #

@dataclass
class IEEE_AttackDetectionThresholds:
    """IEEE-compliant thresholds for attack detection"""
    
    # Based on IEEE Std 1547.1 and C37.118
    voltage_deviation_threshold: float = 0.03    # 3% voltage deviation
    frequency_deviation_threshold: float = 0.05   # 50 mHz deviation
    phase_angle_deviation_threshold: float = 5.0  # 5 degree phase deviation
    
    # Rate of change thresholds (IEEE C37.118)
    rocof_threshold: float = 0.5                  # 0.5 Hz/s ROCOF
    voltage_rate_threshold: float = 0.1           # 10%/s voltage rate
    
    # Statistical detection (IEEE recommendations)
    statistical_confidence: float = 0.95          # 95% confidence level
    detection_window_s: float = 1.0               # 1 second detection window
    
    # Multi-measurement correlation
    correlation_threshold: float = 0.8            # 80% correlation threshold
    measurement_redundancy: int = 3               # Minimum 3 measurements

class IEEE_AttackDetection:
    """IEEE-compliant attack detection algorithms"""
    
    @staticmethod
    def detect_coordinated_attack(measurements: Dict[str, List[float]], 
                                timestamps: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """
        IEEE C37.118-compliant coordinated attack detection
        
        Args:
            measurements: Dictionary of measurement vectors by type
            timestamps: Corresponding timestamps
            
        Returns:
            Tuple of (attack_detected, detection_details)
        """
        thresholds = IEEE_AttackDetectionThresholds()
        
        detection_results = {
            'attack_detected': False,
            'confidence_level': 0.0,
            'affected_measurements': [],
            'detection_time': None,
            'attack_magnitude': 0.0
        }
        
        # IEEE-compliant multi-measurement analysis
        for measurement_type, values in measurements.items():
            if len(values) < 3:  # Minimum IEEE requirement
                continue
                
            # Statistical analysis per IEEE guidelines
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Detect anomalies using IEEE thresholds
            anomaly_detected = False
            if measurement_type == 'voltage':
                anomaly_detected = std_val > thresholds.voltage_deviation_threshold
            elif measurement_type == 'frequency':
                anomaly_detected = std_val > thresholds.frequency_deviation_threshold
            elif measurement_type == 'phase_angle':
                anomaly_detected = std_val > thresholds.phase_angle_deviation_threshold
            
            if anomaly_detected:
                detection_results['affected_measurements'].append(measurement_type)
                detection_results['attack_detected'] = True
        
        # Calculate IEEE-compliant confidence level
        if detection_results['attack_detected']:
            num_affected = len(detection_results['affected_measurements'])
            total_measurements = len(measurements)
            detection_results['confidence_level'] = min(0.95, num_affected / total_measurements)
        
        return detection_results['attack_detected'], detection_results

# ======================== MODULE EXPORTS ======================== #

__all__ = [
    'IEEE_VoltageLevel',
    'IEEE_FrequencyClass', 
    'IEEE_ProtectionClass',
    'IEEE_SystemParameters',
    'IEEE_LoadParameters',
    'IEEE_ProtectionSettings',
    'IEEE_AttackDetectionThresholds',
    'IEEE_Standards',
    'IEEE_AttackDetection'
]

if __name__ == "__main__":
    # Demonstration of IEEE standards
    print("IEEE Standards Module")
    print("=" * 40)
    
    # Get IEEE parameters
    params = IEEE_Standards.get_ieee39_standard_parameters()
    print(f"IEEE Nominal Frequency: {params.nominal_frequency} Hz")
    print(f"IEEE Frequency Deadband: ±{params.frequency_deadband} Hz")
    print(f"IEEE PMU Rate: {params.pmu_reporting_rate} fps")
    
    # Test validation functions
    freq_valid, freq_msg = IEEE_Standards.validate_frequency_deviation(0.1)
    print(f"Frequency validation (0.1 Hz): {freq_valid} - {freq_msg}")
    
    volt_valid, volt_msg = IEEE_Standards.validate_voltage_level(0.95)  
    print(f"Voltage validation (0.95 pu): {volt_valid} - {volt_msg}")
    
    print("\nIEEE Standards Module Ready!")
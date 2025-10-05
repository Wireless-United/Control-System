#!/usr/bin/env python3
"""
Static Attack Demonstration - 7 Second Attack Duration

This script demonstrates static Load-Altering Attacks (LAA) with comprehensive
IEEE-compliant visualization. Simulates step, random, and periodic attacks
with 7-second duration and generates multiple analysis plots.

Attack Scenarios:
- STEP Attack: Sudden 7s load increase
- RANDOM Attack: 7s stochastic load variation
- PERIODIC Attack: 7s sinusoidal load oscillation

Output: Comprehensive IEEE-standard plots including:
- Frequency response analysis
- Voltage profile analysis
- Power flow analysis
- Rotor angle stability
- Attack impact comparison

Author: Pranaav
Date: October 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from laa.attacker.laa_config import (
    AttackConfig, AttackType, AttackSeverity,
    SimulationConfig, SystemInertiaConfig, InertiaCondition
)
from laa.static.static_laa import StaticLAAGenerator, StaticAttackExecutor
from laa.visualization.ieee_graphs import (
    IEEE_FrequencyAnalyzer, IEEE_VoltageAnalyzer,
    IEEE_PowerFlowAnalyzer, IEEE_StabilityAnalyzer,
    IEEE_Colors
)

# ==================== CONFIGURATION ====================

# Static attack duration (as requested)
ATTACK_DURATION = 7.0  # seconds

# Simulation configuration
SIM_CONFIG = SimulationConfig(
    simulation_time=20.0,  # Total 20s simulation (7s attack + 13s recovery)
    time_step=0.01,         # 10ms time step
    frequency_nominal=50.0  # 50 Hz system
)

# System inertia configuration (low inertia for maximum attack impact)
INERTIA_CONFIG = SystemInertiaConfig(
    condition=InertiaCondition.LOW,
    generator_inertia_multiplier=0.5,
    damping_multiplier=0.7,
    der_penetration_level=0.6,
    frequency_response_rate=0.1,
    voltage_regulation_strength=0.8
)

# Target buses for attack (IEEE 39-bus critical load buses)
TARGET_BUSES = [20, 21, 23, 24]

# Attack severity
ATTACK_SEVERITY = AttackSeverity.HIGH  # High severity for clear visualization

# ==================== ATTACK GENERATION FUNCTIONS ====================

def generate_step_attack_data():
    """
    Generate step attack simulation data.
    
    Step attack: Instantaneous 7s load increase followed by recovery.
    """
    print("\n[STEP ATTACK] Generating 7-second step attack simulation...")
    
    # Create step attack configuration
    step_config = AttackConfig(
        attack_type=AttackType.STEP,
        target_buses=TARGET_BUSES,
        severity=ATTACK_SEVERITY,
        duration=ATTACK_DURATION,
        magnitude_mw=100.0,  # 100 MW step
        start_time=3.0,       # Attack starts at t=3s
        step_magnitude=100.0
    )
    
    # Generate time vector
    time_vector = SIM_CONFIG.get_time_steps()
    attack_start_idx = int(step_config.start_time / SIM_CONFIG.time_step)
    attack_end_idx = int((step_config.start_time + ATTACK_DURATION) / SIM_CONFIG.time_step)
    
    # Generate frequency response (step attack causes frequency drop)
    frequency = np.ones(len(time_vector)) * SIM_CONFIG.frequency_nominal
    frequency[attack_start_idx:attack_end_idx] -= 0.3  # 0.3 Hz drop during attack
    # Add exponential recovery
    recovery_indices = np.arange(attack_end_idx, len(time_vector))
    if len(recovery_indices) > 0:
        recovery_time = time_vector[recovery_indices] - time_vector[attack_end_idx]
        frequency[recovery_indices] += 0.3 * np.exp(-recovery_time / 2.0)
    
    # Generate voltage profile (39 buses)
    bus_numbers = np.arange(1, 40)
    voltage_mag = np.ones(39)
    voltage_mag[TARGET_BUSES[0]-1:TARGET_BUSES[-1]] *= 0.92  # Voltage drop at target buses
    voltage_angle = np.random.randn(39) * 5  # Random angles
    
    # Generate power flow
    active_power = 500 * np.ones(len(time_vector))
    active_power[attack_start_idx:attack_end_idx] += step_config.magnitude_mw
    reactive_power = 100 * np.ones(len(time_vector))
    reactive_power[attack_start_idx:attack_end_idx] += 30
    
    # Generate rotor angles
    rotor_angles = {
        'G1': np.zeros(len(time_vector)),
        'G2': np.zeros(len(time_vector)),
        'G3': np.zeros(len(time_vector)),
    }
    for gen in rotor_angles:
        rotor_angles[gen][attack_start_idx:attack_end_idx] += 15
        if len(recovery_indices) > 0:
            rotor_angles[gen][recovery_indices] -= 15 * np.exp(-recovery_time / 3.0)
    
    return {
        'time': time_vector,
        'frequency': frequency,
        'bus_numbers': bus_numbers,
        'voltage_magnitude': voltage_mag,
        'voltage_angle': voltage_angle,
        'active_power': active_power,
        'reactive_power': reactive_power,
        'rotor_angles': rotor_angles,
        'attack_events': [{
            'start_time': step_config.start_time,
            'duration': ATTACK_DURATION,
            'type': 'STEP Attack (100 MW)',
            'severity': 'HIGH'
        }],
        'attack_config': step_config
    }

def generate_random_attack_data():
    """
    Generate random attack simulation data.
    
    Random attack: 7s stochastic load variation with Gaussian noise.
    """
    print("\n[RANDOM ATTACK] Generating 7-second random attack simulation...")
    
    # Create random attack configuration
    random_config = AttackConfig(
        attack_type=AttackType.RANDOM,
        target_buses=TARGET_BUSES,
        severity=ATTACK_SEVERITY,
        duration=ATTACK_DURATION,
        magnitude_mw=60.0,  # Base 60 MW
        start_time=3.0,
        random_variance=20.0  # 20 MW variance
    )
    
    # Generate time vector
    time_vector = SIM_CONFIG.get_time_steps()
    attack_start_idx = int(random_config.start_time / SIM_CONFIG.time_step)
    attack_end_idx = int((random_config.start_time + ATTACK_DURATION) / SIM_CONFIG.time_step)
    
    # Generate random load variation during attack
    load_variation = np.zeros(len(time_vector))
    attack_length = attack_end_idx - attack_start_idx
    load_variation[attack_start_idx:attack_end_idx] = (
        random_config.magnitude_mw +
        random_config.random_variance * np.random.randn(attack_length)
    )
    
    # Generate frequency response
    frequency = np.ones(len(time_vector)) * SIM_CONFIG.frequency_nominal
    frequency -= load_variation * 0.003  # Load variation affects frequency
    
    # Generate voltage profile
    bus_numbers = np.arange(1, 40)
    voltage_mag = 1.0 + 0.03 * np.random.randn(39)
    voltage_mag[TARGET_BUSES[0]-1:TARGET_BUSES[-1]] *= 0.95
    voltage_angle = np.random.randn(39) * 10
    
    # Generate power flow
    active_power = 500 + load_variation
    reactive_power = 100 + load_variation * 0.3
    
    # Generate rotor angles with random perturbations
    rotor_angles = {
        'G1': load_variation * 0.1 + np.random.randn(len(time_vector)) * 2,
        'G2': load_variation * 0.08 + np.random.randn(len(time_vector)) * 2,
        'G3': load_variation * 0.12 + np.random.randn(len(time_vector)) * 2,
    }
    
    return {
        'time': time_vector,
        'frequency': frequency,
        'bus_numbers': bus_numbers,
        'voltage_magnitude': voltage_mag,
        'voltage_angle': voltage_angle,
        'active_power': active_power,
        'reactive_power': reactive_power,
        'rotor_angles': rotor_angles,
        'attack_events': [{
            'start_time': random_config.start_time,
            'duration': ATTACK_DURATION,
            'type': 'RANDOM Attack (60±20 MW)',
            'severity': 'HIGH'
        }],
        'attack_config': random_config
    }

def generate_periodic_attack_data():
    """
    Generate periodic attack simulation data.
    
    Periodic attack: 7s sinusoidal load oscillation.
    """
    print("\n[PERIODIC ATTACK] Generating 7-second periodic attack simulation...")
    
    # Create periodic attack configuration
    periodic_config = AttackConfig(
        attack_type=AttackType.PERIODIC,
        target_buses=TARGET_BUSES,
        severity=ATTACK_SEVERITY,
        duration=ATTACK_DURATION,
        magnitude_mw=75.0,  # Base 75 MW
        start_time=3.0,
        periodic_frequency=0.5,  # 0.5 Hz oscillation (2s period)
        periodic_amplitude=75.0
    )
    
    # Generate time vector
    time_vector = SIM_CONFIG.get_time_steps()
    attack_start_idx = int(periodic_config.start_time / SIM_CONFIG.time_step)
    attack_end_idx = int((periodic_config.start_time + ATTACK_DURATION) / SIM_CONFIG.time_step)
    
    # Generate periodic load variation
    load_variation = np.zeros(len(time_vector))
    attack_time = time_vector[attack_start_idx:attack_end_idx] - periodic_config.start_time
    load_variation[attack_start_idx:attack_end_idx] = (
        periodic_config.periodic_amplitude *
        np.sin(2 * np.pi * periodic_config.periodic_frequency * attack_time)
    )
    
    # Generate frequency response with periodic oscillation
    frequency = SIM_CONFIG.frequency_nominal - load_variation * 0.002
    
    # Generate voltage profile
    bus_numbers = np.arange(1, 40)
    voltage_mag = np.ones(39)
    voltage_mag[TARGET_BUSES[0]-1:TARGET_BUSES[-1]] *= 0.96
    voltage_angle = np.random.randn(39) * 8
    
    # Generate power flow with periodic variation
    active_power = 500 + load_variation
    reactive_power = 100 + load_variation * 0.4 * np.sin(2 * np.pi * periodic_config.periodic_frequency * time_vector)
    
    # Generate rotor angles with periodic oscillation
    rotor_angles = {
        'G1': load_variation * 0.15,
        'G2': load_variation * 0.12,
        'G3': load_variation * 0.18,
    }
    
    return {
        'time': time_vector,
        'frequency': frequency,
        'bus_numbers': bus_numbers,
        'voltage_magnitude': voltage_mag,
        'voltage_angle': voltage_angle,
        'active_power': active_power,
        'reactive_power': reactive_power,
        'rotor_angles': rotor_angles,
        'attack_events': [{
            'start_time': periodic_config.start_time,
            'duration': ATTACK_DURATION,
            'type': 'PERIODIC Attack (75 MW, 0.5 Hz)',
            'severity': 'HIGH'
        }],
        'attack_config': periodic_config
    }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_attack_plots(attack_data, attack_name, output_dir):
    """
    Create comprehensive IEEE-compliant plots for attack analysis.
    
    Args:
        attack_data: Dictionary containing attack simulation data
        attack_name: Name of the attack (for file naming)
        output_dir: Directory to save plots
    
    Returns:
        List of generated plot file paths
    """
    print(f"\n[PLOTTING] Creating IEEE-compliant plots for {attack_name}...")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    # Initialize analyzers
    freq_analyzer = IEEE_FrequencyAnalyzer()
    volt_analyzer = IEEE_VoltageAnalyzer()
    power_analyzer = IEEE_PowerFlowAnalyzer()
    stability_analyzer = IEEE_StabilityAnalyzer()
    
    # 1. Frequency Response Plot
    print("  - Frequency response analysis...")
    freq_path = os.path.join(output_dir, f'{attack_name}_frequency.png')
    freq_analyzer.plot_frequency_response(
        attack_data['time'],
        attack_data['frequency'],
        attack_data['attack_events'],
        freq_path
    )
    plot_files.append(freq_path)
    plt.close()
    
    # 2. Voltage Profile Plot
    print("  - Voltage profile analysis...")
    volt_path = os.path.join(output_dir, f'{attack_name}_voltage.png')
    volt_analyzer.plot_voltage_profile(
        attack_data['bus_numbers'],
        attack_data['voltage_magnitude'],
        attack_data['voltage_angle'],
        TARGET_BUSES,
        volt_path
    )
    plot_files.append(volt_path)
    plt.close()
    
    # 3. Power Flow Plot
    print("  - Power flow analysis...")
    power_path = os.path.join(output_dir, f'{attack_name}_power.png')
    power_analyzer.plot_power_flow(
        attack_data['time'],
        attack_data['active_power'],
        attack_data['reactive_power'],
        attack_data['attack_events'],
        power_path
    )
    plot_files.append(power_path)
    plt.close()
    
    # 4. Stability Plot
    print("  - Rotor angle stability analysis...")
    stability_path = os.path.join(output_dir, f'{attack_name}_stability.png')
    stability_analyzer.plot_rotor_angle_stability(
        attack_data['time'],
        attack_data['rotor_angles'],
        attack_data['attack_events'],
        stability_path
    )
    plot_files.append(stability_path)
    plt.close()
    
    print(f"  [SUCCESS] Generated {len(plot_files)} plots")
    return plot_files

def create_comparison_plot(step_data, random_data, periodic_data, output_dir):
    """Create comparison plot of all three attack types"""
    print("\n[COMPARISON] Creating attack type comparison plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    colors = IEEE_Colors()
    
    # Plot frequency responses
    axes[0].plot(step_data['time'], step_data['frequency'], 
                label='STEP Attack', color=colors.PHASE_A, linewidth=2)
    axes[0].plot(random_data['time'], random_data['frequency'], 
                label='RANDOM Attack', color=colors.PHASE_B, linewidth=2)
    axes[0].plot(periodic_data['time'], periodic_data['frequency'], 
                label='PERIODIC Attack', color=colors.PHASE_C, linewidth=2)
    axes[0].axhline(y=50.0, color='k', linestyle='--', alpha=0.5, label='Nominal (50 Hz)')
    axes[0].axvspan(3, 10, color='red', alpha=0.1, label='Attack Period (7s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Attack Type Comparison - Frequency Response', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot active power
    axes[1].plot(step_data['time'], step_data['active_power'], 
                label='STEP Attack', color=colors.PHASE_A, linewidth=2)
    axes[1].plot(random_data['time'], random_data['active_power'], 
                label='RANDOM Attack', color=colors.PHASE_B, linewidth=2)
    axes[1].plot(periodic_data['time'], periodic_data['active_power'], 
                label='PERIODIC Attack', color=colors.PHASE_C, linewidth=2)
    axes[1].axvspan(3, 10, color='red', alpha=0.1)
    axes[1].set_ylabel('Active Power (MW)')
    axes[1].set_title('Attack Type Comparison - Active Power', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot rotor angle deviation
    axes[2].plot(step_data['time'], step_data['rotor_angles']['G1'], 
                label='STEP Attack', color=colors.PHASE_A, linewidth=2)
    axes[2].plot(random_data['time'], random_data['rotor_angles']['G1'], 
                label='RANDOM Attack', color=colors.PHASE_B, linewidth=2)
    axes[2].plot(periodic_data['time'], periodic_data['rotor_angles']['G1'], 
                label='PERIODIC Attack', color=colors.PHASE_C, linewidth=2)
    axes[2].axvspan(3, 10, color='red', alpha=0.1)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Rotor Angle (degrees)')
    axes[2].set_title('Attack Type Comparison - Generator G1 Rotor Angle', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, 'attack_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [SUCCESS] Comparison plot saved: {comparison_path}")
    return comparison_path

def create_summary_report(step_data, random_data, periodic_data, all_plots, output_dir):
    """Create text summary report"""
    report_path = os.path.join(output_dir, 'static_attack_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LAA STATIC ATTACK ANALYSIS REPORT - 7 SECOND ATTACK DURATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: Pranaav
        
        f.write("SIMULATION CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Simulation Time: {SIM_CONFIG.simulation_time} seconds\n")
        f.write(f"Attack Duration: {ATTACK_DURATION} seconds\n")
        f.write(f"Attack Start Time: 3.0 seconds\n")
        f.write(f"Time Step: {SIM_CONFIG.time_step} seconds\n")
        f.write(f"Nominal Frequency: {SIM_CONFIG.frequency_nominal} Hz\n")
        f.write(f"System Inertia: LOW (60% DER penetration)\n")
        f.write(f"Target Buses: {TARGET_BUSES}\n")
        f.write(f"Attack Severity: {ATTACK_SEVERITY.name}\n\n")
        
        f.write("ATTACK SCENARIOS ANALYZED\n")
        f.write("-" * 40 + "\n")
        f.write("1. STEP ATTACK\n")
        f.write("   - Type: Sudden load increase\n")
        f.write("   - Magnitude: 100 MW\n")
        f.write("   - Frequency Impact: -0.3 Hz peak deviation\n\n")
        
        f.write("2. RANDOM ATTACK\n")
        f.write("   - Type: Stochastic load variation\n")
        f.write("   - Magnitude: 60 ± 20 MW\n")
        f.write("   - Pattern: Gaussian noise\n\n")
        
        f.write("3. PERIODIC ATTACK\n")
        f.write("   - Type: Sinusoidal oscillation\n")
        f.write("   - Magnitude: 75 MW amplitude\n")
        f.write("   - Frequency: 0.5 Hz (2s period)\n\n")
        
        f.write("ANALYSIS RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"STEP Attack - Frequency Range: {np.min(step_data['frequency']):.3f} - {np.max(step_data['frequency']):.3f} Hz\n")
        f.write(f"RANDOM Attack - Frequency Range: {np.min(random_data['frequency']):.3f} - {np.max(random_data['frequency']):.3f} Hz\n")
        f.write(f"PERIODIC Attack - Frequency Range: {np.min(periodic_data['frequency']):.3f} - {np.max(periodic_data['frequency']):.3f} Hz\n\n")
        
        f.write("GENERATED PLOTS\n")
        f.write("-" * 40 + "\n")
        for plot in all_plots:
            f.write(f"- {os.path.basename(plot)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    return report_path

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("=" * 80)
    print("LAA STATIC ATTACK DEMONSTRATION - 7 SECOND ATTACK DURATION")
    print("=" * 80)
    print("Author: Pranaav
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(__file__).parent / 'static_attack_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput Directory: {output_dir}")
    
    # Generate attack simulation data
    step_data = generate_step_attack_data()
    random_data = generate_random_attack_data()
    periodic_data = generate_periodic_attack_data()
    
    # Create plots for each attack type
    all_plots = []
    
    print("\n" + "=" * 80)
    print("GENERATING IEEE-COMPLIANT PLOTS")
    print("=" * 80)
    
    step_plots = create_attack_plots(step_data, 'step_attack', output_dir)
    all_plots.extend(step_plots)
    
    random_plots = create_attack_plots(random_data, 'random_attack', output_dir)
    all_plots.extend(random_plots)
    
    periodic_plots = create_attack_plots(periodic_data, 'periodic_attack', output_dir)
    all_plots.extend(periodic_plots)
    
    # Create comparison plot
    comparison_plot = create_comparison_plot(step_data, random_data, periodic_data, output_dir)
    all_plots.append(comparison_plot)
    
    # Create summary report
    report_path = create_summary_report(step_data, random_data, periodic_data, all_plots, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("STATIC ATTACK DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal Plots Generated: {len(all_plots)}")
    print(f"Analysis Report: {os.path.basename(report_path)}")
    print(f"Output Directory: {output_dir}")
    print("\nGenerated Files:")
    for i, plot in enumerate(all_plots, 1):
        print(f"  {i:2d}. {os.path.basename(plot)}")
    print(f"  {len(all_plots)+1:2d}. {os.path.basename(report_path)}")
    print("\n" + "=" * 80)
    print("Open the PNG files to view IEEE-standard analysis results!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

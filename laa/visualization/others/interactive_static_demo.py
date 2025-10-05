#!/usr/bin/env python3
"""
LAA Interactive Static Attack Demo

This interactive demo allows users to customize static attack parameters
and visualize the results. Users can specify attack duration, load variation
magnitude, attack type, target buses, and severity level.

Features:
- Interactive user input for all attack parameters
- Real-time parameter validation
- Customizable attack scenarios (STEP, RANDOM, PERIODIC)
- IEEE-compliant visualization
- Comprehensive analysis reports

IEEE Standards Applied:
- IEEE 1547.1: Grid stability requirements
- IEEE C37.118: Synchrophasor measurements
- IEEE 421.5: Excitation system response
- IEEE 1110: System stability analysis

Author: Pranaav
Date: October 2025
"""

import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
laa_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, laa_dir)

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import LAA framework components
try:
    from laa.attacker.laa_config import (
        InertiaCondition, AttackType, AttackSeverity,
        SystemInertiaConfig, AttackConfig, SimulationConfig
    )
    from laa.static.static_laa import StaticLAAGenerator
    from laa.visualization.ieee_graphs import IEEEGraphGenerator
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the LAA framework is properly installed")
    sys.exit(1)


class InteractiveStaticDemo:
    """
    Interactive Static Attack Demonstration
    
    Allows users to customize attack parameters and visualize results
    with real-time parameter validation and comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize the interactive demo"""
        self.graph_gen = IEEEGraphGenerator()
        self.output_dir = None
        
        print("\n" + "=" * 80)
        print("LAA INTERACTIVE STATIC ATTACK DEMONSTRATION")
        print("=" * 80)
        print("\nThis demo allows you to customize static attack parameters")
        print("and visualize the impact on the IEEE 39-Bus power system.")
        print("=" * 80 + "\n")
    
    def get_user_input(self) -> Dict:
        """
        Collect attack parameters from user with validation
        
        Returns:
            Dictionary containing all user-specified parameters
        """
        print("\n" + "-" * 80)
        print("ATTACK PARAMETER CONFIGURATION")
        print("-" * 80 + "\n")
        
        params = {}
        
        # 1. Attack Type
        print("1. SELECT ATTACK TYPE:")
        print("   [1] STEP Attack - Sudden constant load change")
        print("   [2] RANDOM Attack - Randomly varying load changes")
        print("   [3] PERIODIC Attack - Oscillating load pattern")
        
        while True:
            try:
                choice = input("\n   Enter choice (1-3): ").strip()
                if choice == '1':
                    params['attack_type'] = AttackType.STEP
                    params['attack_name'] = 'STEP'
                    break
                elif choice == '2':
                    params['attack_type'] = AttackType.RANDOM
                    params['attack_name'] = 'RANDOM'
                    break
                elif choice == '3':
                    params['attack_type'] = AttackType.PERIODIC
                    params['attack_name'] = 'PERIODIC'
                    break
                else:
                    print("   Invalid choice. Please enter 1, 2, or 3.")
            except Exception as e:
                print(f"   Error: {e}. Please try again.")
        
        print(f"   Selected: {params['attack_name']} Attack")
        
        # 2. Attack Duration
        print("\n2. ATTACK DURATION:")
        print("   Specify how long the attack should last (in seconds)")
        print("   Recommended range: 5-15 seconds")
        
        while True:
            try:
                duration = float(input("\n   Enter duration (s): ").strip())
                if 1.0 <= duration <= 30.0:
                    params['duration'] = duration
                    break
                else:
                    print("   Duration must be between 1 and 30 seconds. Please try again.")
            except ValueError:
                print("   Invalid input. Please enter a number.")
        
        print(f"   Attack duration set to: {params['duration']} seconds")
        
        # 3. Load Variation Magnitude
        print("\n3. LOAD VARIATION MAGNITUDE:")
        print("   Specify the magnitude of load manipulation (in MW)")
        print("   Recommended range: 20-150 MW")
        print("   Higher values = more aggressive attack")
        
        while True:
            try:
                magnitude = float(input("\n   Enter magnitude (MW): ").strip())
                if 10.0 <= magnitude <= 200.0:
                    params['magnitude'] = magnitude
                    break
                else:
                    print("   Magnitude must be between 10 and 200 MW. Please try again.")
            except ValueError:
                print("   Invalid input. Please enter a number.")
        
        print(f"   Load variation set to: {params['magnitude']} MW")
        
        # 4. Attack Severity
        print("\n4. ATTACK SEVERITY:")
        print("   [1] LOW - Minimal impact, easily recoverable")
        print("   [2] MEDIUM - Moderate impact, noticeable frequency deviation")
        print("   [3] HIGH - Significant impact, potential stability issues")
        print("   [4] CRITICAL - Severe impact, risk of cascading failures")
        
        while True:
            try:
                choice = input("\n   Enter choice (1-4): ").strip()
                if choice == '1':
                    params['severity'] = AttackSeverity.LOW
                    params['severity_name'] = 'LOW'
                    break
                elif choice == '2':
                    params['severity'] = AttackSeverity.MEDIUM
                    params['severity_name'] = 'MEDIUM'
                    break
                elif choice == '3':
                    params['severity'] = AttackSeverity.HIGH
                    params['severity_name'] = 'HIGH'
                    break
                elif choice == '4':
                    params['severity'] = AttackSeverity.CRITICAL
                    params['severity_name'] = 'CRITICAL'
                    break
                else:
                    print("   Invalid choice. Please enter 1, 2, 3, or 4.")
            except Exception as e:
                print(f"   Error: {e}. Please try again.")
        
        print(f"   Severity level: {params['severity_name']}")
        
        # 5. Target Buses
        print("\n5. TARGET BUSES:")
        print("   Select which buses to target (IEEE 39-Bus system)")
        print("   [1] High-Load Buses (16, 20, 23) - Recommended")
        print("   [2] Medium-Load Buses (4, 7, 12)")
        print("   [3] Critical Buses (3, 15, 18)")
        print("   [4] Custom - Specify your own")
        
        while True:
            try:
                choice = input("\n   Enter choice (1-4): ").strip()
                if choice == '1':
                    params['target_buses'] = [16, 20, 23]
                    params['bus_description'] = 'High-Load'
                    break
                elif choice == '2':
                    params['target_buses'] = [4, 7, 12]
                    params['bus_description'] = 'Medium-Load'
                    break
                elif choice == '3':
                    params['target_buses'] = [3, 15, 18]
                    params['bus_description'] = 'Critical'
                    break
                elif choice == '4':
                    bus_input = input("   Enter bus numbers (comma-separated, 1-39): ").strip()
                    buses = [int(b.strip()) for b in bus_input.split(',')]
                    if all(1 <= b <= 39 for b in buses):
                        params['target_buses'] = buses
                        params['bus_description'] = 'Custom'
                        break
                    else:
                        print("   Bus numbers must be between 1 and 39. Please try again.")
                else:
                    print("   Invalid choice. Please enter 1, 2, 3, or 4.")
            except (ValueError, AttributeError):
                print("   Invalid input. Please enter valid bus numbers.")
        
        print(f"   Target buses: {params['target_buses']}")
        
        # 6. Inertia Condition
        print("\n6. SYSTEM INERTIA CONDITION:")
        print("   [1] LOW - High renewable penetration, lower system stability")
        print("   [2] HIGH - Traditional generation, higher system stability")
        
        while True:
            try:
                choice = input("\n   Enter choice (1-2): ").strip()
                if choice == '1':
                    params['inertia'] = InertiaCondition.LOW
                    params['inertia_name'] = 'LOW'
                    break
                elif choice == '2':
                    params['inertia'] = InertiaCondition.HIGH
                    params['inertia_name'] = 'HIGH'
                    break
                else:
                    print("   Invalid choice. Please enter 1 or 2.")
            except Exception as e:
                print(f"   Error: {e}. Please try again.")
        
        print(f"   System inertia: {params['inertia_name']}")
        
        # 7. Additional parameters for PERIODIC attack
        if params['attack_type'] == AttackType.PERIODIC:
            print("\n7. PERIODIC ATTACK PARAMETERS:")
            print("   Specify oscillation frequency (Hz)")
            print("   Recommended range: 0.1-2.0 Hz")
            
            while True:
                try:
                    freq = float(input("\n   Enter frequency (Hz): ").strip())
                    if 0.05 <= freq <= 5.0:
                        params['variation_frequency'] = freq
                        break
                    else:
                        print("   Frequency must be between 0.05 and 5.0 Hz. Please try again.")
                except ValueError:
                    print("   Invalid input. Please enter a number.")
            
            print(f"   Oscillation frequency: {params['variation_frequency']} Hz")
        else:
            params['variation_frequency'] = 0.5  # Default
        
        # 8. Additional parameters for RANDOM attack
        if params['attack_type'] == AttackType.RANDOM:
            print("\n7. RANDOM ATTACK PARAMETERS:")
            print("   Specify variation amplitude (% of magnitude)")
            print("   Recommended range: 20-50%")
            
            while True:
                try:
                    amp = float(input("\n   Enter amplitude (%): ").strip())
                    if 10.0 <= amp <= 100.0:
                        params['variation_amplitude'] = params['magnitude'] * (amp / 100.0)
                        break
                    else:
                        print("   Amplitude must be between 10 and 100%. Please try again.")
                except ValueError:
                    print("   Invalid input. Please enter a number.")
            
            print(f"   Variation amplitude: {params['variation_amplitude']:.1f} MW")
        else:
            params['variation_amplitude'] = params['magnitude'] * 0.3  # Default
        
        # 9. Output Directory
        print("\n8. OUTPUT DIRECTORY:")
        default_dir = f"interactive_static_{params['attack_name'].lower()}_results"
        dir_input = input(f"\n   Enter directory name (default: {default_dir}): ").strip()
        params['output_dir'] = dir_input if dir_input else default_dir
        
        print(f"   Results will be saved to: {params['output_dir']}")
        
        return params
    
    def display_configuration_summary(self, params: Dict):
        """Display summary of user configuration"""
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"\nAttack Type:          {params['attack_name']}")
        print(f"Attack Duration:      {params['duration']} seconds")
        print(f"Load Variation:       {params['magnitude']} MW")
        print(f"Attack Severity:      {params['severity_name']}")
        print(f"Target Buses:         {params['target_buses']} ({params['bus_description']})")
        print(f"System Inertia:       {params['inertia_name']}")
        
        if params['attack_type'] == AttackType.PERIODIC:
            print(f"Oscillation Freq:     {params['variation_frequency']} Hz")
        elif params['attack_type'] == AttackType.RANDOM:
            print(f"Variation Amplitude:  {params['variation_amplitude']:.1f} MW")
        
        print(f"Output Directory:     {params['output_dir']}")
        print("=" * 80)
        
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        return confirm in ['y', 'yes']
    
    def generate_attack_data(self, params: Dict) -> Dict:
        """
        Generate attack simulation data based on user parameters
        
        Args:
            params: User-specified parameters
            
        Returns:
            Dictionary containing simulation results
        """
        print("\n" + "-" * 80)
        print("GENERATING ATTACK SIMULATION")
        print("-" * 80)
        
        # Create inertia configuration
        if params['inertia'] == InertiaCondition.LOW:
            inertia_config = SystemInertiaConfig(
                condition=InertiaCondition.LOW,
                generator_inertia_multiplier=0.5,
                renewable_penetration=0.35,
                load_damping_coefficient=1.0,
                governor_droop=0.05,
                frequency_deadband=0.036,
                renewable_inertia_constant=3.0
            )
        else:
            inertia_config = SystemInertiaConfig(
                condition=InertiaCondition.HIGH,
                generator_inertia_multiplier=1.5,
                renewable_penetration=0.15,
                load_damping_coefficient=1.5,
                governor_droop=0.04,
                frequency_deadband=0.036,
                renewable_inertia_constant=5.0
            )
        
        # Create attack configuration
        attack_config = AttackConfig(
            attack_type=params['attack_type'],
            severity=params['severity'],
            target_buses=params['target_buses'],
            attack_start_time=2.0,
            attack_duration=params['duration'],
            attack_magnitude=params['magnitude'],
            frequency_setpoint=59.5,
            pid_gains={'kp': 100.0, 'ki': 10.0, 'kd': 20.0},
            ramp_rate=20.0,
            variation_amplitude=params['variation_amplitude'],
            variation_frequency=params['variation_frequency'],
            multi_stage_config=None
        )
        
        # Create simulation configuration
        total_time = params['duration'] + 8.0  # Attack duration + pre/post time
        sim_config = SimulationConfig(
            total_time=total_time,
            time_step=0.01,
            measurement_noise=0.001,
            communication_delay=0.05,
            max_load_change_rate=50.0,
            frequency_measurement_rate=30,
            enable_governor_response=True,
            enable_agc=False,
            contingency_events=None,
            renewable_variability=0.05
        )
        
        # Generate time vector
        t = np.arange(0, total_time, 0.01)
        
        # Simulate system response
        print("\nSimulating system response...")
        frequency = self._simulate_frequency_response(t, attack_config, sim_config, inertia_config)
        voltage = self._simulate_voltage_response(t, frequency, attack_config)
        power = self._simulate_power_response(t, attack_config)
        attack_signal = self._calculate_attack_signal(t, attack_config)
        
        print("Simulation complete.")
        
        return {
            'time': t,
            'frequency': frequency,
            'voltage': voltage,
            'power': power,
            'attack_signal': attack_signal,
            'params': params,
            'attack_config': attack_config,
            'inertia_config': inertia_config,
            'sim_config': sim_config
        }
    
    def _simulate_frequency_response(
        self,
        t: np.ndarray,
        attack_config: AttackConfig,
        sim_config: SimulationConfig,
        inertia_config: SystemInertiaConfig
    ) -> np.ndarray:
        """Simulate frequency response to attack"""
        freq = np.zeros_like(t)
        freq[0] = 60.0
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        H = 3.0 * inertia_config.generator_inertia_multiplier
        D = 1.0 * inertia_config.load_damping_coefficient
        dt = sim_config.time_step
        
        for i in range(1, len(t)):
            if attack_start <= t[i] <= attack_end:
                # Attack active
                if attack_config.attack_type == AttackType.STEP:
                    attack_power = attack_config.attack_magnitude
                elif attack_config.attack_type == AttackType.RANDOM:
                    attack_power = attack_config.attack_magnitude + \
                                  np.random.uniform(-attack_config.variation_amplitude,
                                                   attack_config.variation_amplitude)
                elif attack_config.attack_type == AttackType.PERIODIC:
                    attack_power = attack_config.attack_magnitude * \
                                  np.sin(2 * np.pi * attack_config.variation_frequency * 
                                        (t[i] - attack_start))
                else:
                    attack_power = 0.0
            else:
                attack_power = 0.0
            
            # Governor response
            if sim_config.enable_governor_response:
                gov_response = -0.5 * (freq[i-1] - 60.0)
            else:
                gov_response = 0.0
            
            # Swing equation
            power_imbalance = -attack_power / 100.0 + gov_response - D * (freq[i-1] - 60.0)
            df_dt = power_imbalance / (2 * H)
            freq[i] = freq[i-1] + df_dt * dt
            freq[i] += np.random.normal(0, sim_config.measurement_noise)
        
        return freq
    
    def _simulate_voltage_response(
        self,
        t: np.ndarray,
        frequency: np.ndarray,
        attack_config: AttackConfig
    ) -> np.ndarray:
        """Simulate voltage response"""
        voltage = np.ones_like(t)
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        for i in range(len(t)):
            freq_dev = (frequency[i] - 60.0) / 60.0
            voltage[i] = 1.0 - 0.25 * freq_dev
            
            if attack_start <= t[i] <= attack_end:
                voltage[i] -= 0.03 * np.sin(2 * np.pi * 0.5 * (t[i] - attack_start))
            
            voltage[i] += np.random.normal(0, 0.002)
            voltage[i] = np.clip(voltage[i], 0.85, 1.15)
        
        return voltage
    
    def _simulate_power_response(
        self,
        t: np.ndarray,
        attack_config: AttackConfig
    ) -> np.ndarray:
        """Simulate power response"""
        power = np.ones_like(t) * 100.0
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        for i in range(len(t)):
            if attack_start <= t[i] <= attack_end:
                power[i] = 100.0 + attack_config.attack_magnitude * 0.6 * \
                          np.sin(2 * np.pi * 0.7 * (t[i] - attack_start))
            
            power[i] += np.random.normal(0, 1.0)
        
        return power
    
    def _calculate_attack_signal(
        self,
        t: np.ndarray,
        attack_config: AttackConfig
    ) -> np.ndarray:
        """Calculate attack signal over time"""
        attack_signal = np.zeros_like(t)
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        for i in range(len(t)):
            if attack_start <= t[i] <= attack_end:
                if attack_config.attack_type == AttackType.STEP:
                    attack_signal[i] = attack_config.attack_magnitude
                elif attack_config.attack_type == AttackType.RANDOM:
                    attack_signal[i] = attack_config.attack_magnitude + \
                                      np.random.uniform(-attack_config.variation_amplitude,
                                                       attack_config.variation_amplitude)
                elif attack_config.attack_type == AttackType.PERIODIC:
                    attack_signal[i] = attack_config.attack_magnitude * \
                                      np.sin(2 * np.pi * attack_config.variation_frequency * 
                                            (t[i] - attack_start))
        
        return attack_signal
    
    def create_visualization(self, data: Dict):
        """Create comprehensive visualization plots"""
        print("\n" + "-" * 80)
        print("GENERATING VISUALIZATION")
        print("-" * 80)
        
        params = data['params']
        self.output_dir = params['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        t = data['time']
        freq = data['frequency']
        volt = data['voltage']
        power = data['power']
        attack = data['attack_signal']
        attack_config = data['attack_config']
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'IEEE Static Attack Analysis - Interactive Demo\n'
                    f'{params["attack_name"]} Attack | {params["severity_name"]} Severity | '
                    f'{params["inertia_name"]} Inertia',
                    fontsize=16, fontweight='bold', fontname='Times New Roman')
        
        # Plot 1: Frequency
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, freq, 'b-', linewidth=2, label='System Frequency')
        ax1.axhline(y=60.0, color='g', linestyle='--', linewidth=1.5, label='Nominal')
        ax1.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack')
        ax1.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax1.set_ylabel('Frequency (Hz)', fontname='Times New Roman', fontsize=10)
        ax1.set_title('System Frequency Response', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Plot 2: Frequency Deviation
        ax2 = fig.add_subplot(gs[0, 1])
        freq_dev = freq - 60.0
        ax2.plot(t, freq_dev, 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, label='IEEE Limit')
        ax2.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax2.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax2.set_ylabel('Deviation (Hz)', fontname='Times New Roman', fontsize=10)
        ax2.set_title('Frequency Deviation', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Plot 3: Voltage
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t, volt, 'b-', linewidth=2, label='Bus Voltage')
        ax3.axhline(y=1.0, color='g', linestyle='--', linewidth=1.5, label='Nominal')
        ax3.axhline(y=0.95, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Limits')
        ax3.axhline(y=1.05, color='orange', linestyle='--', linewidth=1, alpha=0.6)
        ax3.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax3.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax3.set_ylabel('Voltage (pu)', fontname='Times New Roman', fontsize=10)
        ax3.set_title('Voltage Response', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # Plot 4: Power
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t, power, 'r-', linewidth=2, label='Active Power')
        ax4.axhline(y=100.0, color='g', linestyle='--', linewidth=1.5, label='Nominal')
        ax4.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax4.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax4.set_ylabel('Power (MW)', fontname='Times New Roman', fontsize=10)
        ax4.set_title('Active Power Flow', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        # Plot 5: Attack Signal
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(t, attack, 'purple', linewidth=2, label='Attack Signal')
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax5.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax5.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax5.set_ylabel('Load Manipulation (MW)', fontname='Times New Roman', fontsize=10)
        ax5.set_title('Attack Load Variation', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
        
        # Plot 6: Phase Portrait (Frequency vs Rate of Change)
        ax6 = fig.add_subplot(gs[2, 1])
        df_dt = np.gradient(freq, t)
        ax6.plot(freq, df_dt, 'b-', linewidth=1.5, alpha=0.7)
        ax6.scatter(freq[0], df_dt[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax6.scatter(freq[-1], df_dt[-1], color='red', s=100, marker='s', label='End', zorder=5)
        ax6.set_xlabel('Frequency (Hz)', fontname='Times New Roman', fontsize=10)
        ax6.set_ylabel('df/dt (Hz/s)', fontname='Times New Roman', fontsize=10)
        ax6.set_title('Phase Portrait', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
        
        plt.savefig(os.path.join(self.output_dir, 'interactive_static_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {self.output_dir}/interactive_static_analysis.png")
    
    def create_report(self, data: Dict):
        """Create detailed analysis report"""
        print("\nGenerating analysis report...")
        
        params = data['params']
        t = data['time']
        freq = data['frequency']
        volt = data['voltage']
        power = data['power']
        attack = data['attack_signal']
        attack_config = data['attack_config']
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        attack_mask = (t >= attack_start) & (t <= attack_end)
        
        report_file = os.path.join(self.output_dir, 'interactive_static_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LAA INTERACTIVE STATIC ATTACK - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Author: Pranaav\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("USER CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Attack Type:          {params['attack_name']}\n")
            f.write(f"Attack Duration:      {params['duration']} seconds\n")
            f.write(f"Load Variation:       {params['magnitude']} MW\n")
            f.write(f"Attack Severity:      {params['severity_name']}\n")
            f.write(f"Target Buses:         {params['target_buses']} ({params['bus_description']})\n")
            f.write(f"System Inertia:       {params['inertia_name']}\n")
            if params['attack_type'] == AttackType.PERIODIC:
                f.write(f"Oscillation Freq:     {params['variation_frequency']} Hz\n")
            elif params['attack_type'] == AttackType.RANDOM:
                f.write(f"Variation Amplitude:  {params['variation_amplitude']:.1f} MW\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Frequency Analysis:\n")
            f.write(f"  Minimum Frequency:              {np.min(freq):.4f} Hz\n")
            f.write(f"  Maximum Frequency:              {np.max(freq):.4f} Hz\n")
            f.write(f"  Average (during attack):        {np.mean(freq[attack_mask]):.4f} Hz\n")
            f.write(f"  Maximum Deviation:              {np.min(freq) - 60.0:.4f} Hz\n")
            f.write(f"  Standard Deviation:             {np.std(freq[attack_mask]):.4f} Hz\n\n")
            
            f.write("Voltage Analysis:\n")
            f.write(f"  Minimum Voltage:                {np.min(volt):.4f} pu\n")
            f.write(f"  Maximum Voltage:                {np.max(volt):.4f} pu\n")
            f.write(f"  Average (during attack):        {np.mean(volt[attack_mask]):.4f} pu\n")
            f.write(f"  Voltage Sag:                    {1.0 - np.min(volt):.4f} pu\n\n")
            
            f.write("Power Analysis:\n")
            f.write(f"  Peak Power:                     {np.max(power):.2f} MW\n")
            f.write(f"  Minimum Power:                  {np.min(power):.2f} MW\n")
            f.write(f"  Average (during attack):        {np.mean(power[attack_mask]):.2f} MW\n")
            f.write(f"  Power Swing:                    {np.max(power) - np.min(power):.2f} MW\n\n")
            
            f.write("Attack Characteristics:\n")
            f.write(f"  Peak Attack Magnitude:          {np.max(np.abs(attack)):.2f} MW\n")
            f.write(f"  Average Attack Magnitude:       {np.mean(np.abs(attack[attack_mask])):.2f} MW\n")
            f.write(f"  Attack Energy:                  {np.sum(np.abs(attack)) * 0.01:.2f} MWs\n\n")
            
            # Settling time
            post_attack_mask = t > attack_end
            if np.any(post_attack_mask):
                post_freq = freq[post_attack_mask]
                post_t = t[post_attack_mask]
                settled = np.abs(post_freq - 60.0) < 0.05
                if np.any(settled):
                    settling_idx = np.where(settled)[0][0]
                    settling_time = post_t[settling_idx] - attack_end
                    f.write(f"System Recovery:\n")
                    f.write(f"  Settling Time (±0.05 Hz):       {settling_time:.2f} seconds\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report saved to: {report_file}")
    
    def run_demo(self):
        """Run the interactive demo"""
        # Get user input
        params = self.get_user_input()
        
        # Display configuration and confirm
        if not self.display_configuration_summary(params):
            print("\nDemo cancelled by user.")
            return
        
        # Generate attack data
        data = self.generate_attack_data(params)
        
        # Create visualization
        self.create_visualization(data)
        
        # Create report
        self.create_report(data)
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {params['output_dir']}/")
        print("  - interactive_static_analysis.png (comprehensive plots)")
        print("  - interactive_static_report.txt (detailed analysis)")
        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    print("\n" + "*" * 80)
    print("LAA FRAMEWORK - INTERACTIVE STATIC ATTACK DEMO")
    print("Author: Pranaav")
    print("Date: October 2025")
    print("*" * 80)
    
    demo = InteractiveStaticDemo()
    demo.run_demo()
    
    print("\n" + "*" * 80)
    print("THANK YOU FOR USING THE LAA FRAMEWORK")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()

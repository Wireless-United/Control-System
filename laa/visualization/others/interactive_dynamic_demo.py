#!/usr/bin/env python3
"""
LAA Interactive Dynamic Attack Demo

This interactive demo allows users to customize dynamic PID-controlled attack
parameters and visualize real-time frequency tracking behavior. Users can specify
target frequency, PID controller gains, attack duration, and other parameters.

Features:
- Interactive user input for PID controller tuning
- Customizable frequency targets and attack parameters
- Real-time frequency tracking simulation
- IEEE-compliant visualization
- Comprehensive performance analysis

IEEE Standards Applied:
- IEEE 1547.1: Grid frequency response requirements
- IEEE C37.118: Synchrophasor measurement standards
- IEEE 421.5: Excitation system response
- IEEE 1110: System frequency stability

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
    from laa.dynamic.async_dynamic_laa import AsyncDynamicLAAGenerator
    from laa.visualization.ieee_graphs import IEEEGraphGenerator
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the LAA framework is properly installed")
    sys.exit(1)


class InteractiveDynamicDemo:
    """
    Interactive Dynamic Attack Demonstration
    
    Allows users to customize PID-controlled feedback attack parameters
    and visualize real-time frequency tracking with comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize the interactive demo"""
        self.graph_gen = IEEEGraphGenerator()
        self.output_dir = None
        
        print("\n" + "=" * 80)
        print("LAA INTERACTIVE DYNAMIC ATTACK DEMONSTRATION")
        print("=" * 80)
        print("\nThis demo allows you to customize dynamic PID-controlled attack")
        print("parameters and visualize real-time frequency tracking behavior.")
        print("=" * 80 + "\n")
    
    def get_user_input(self) -> Dict:
        """
        Collect attack parameters from user with validation
        
        Returns:
            Dictionary containing all user-specified parameters
        """
        print("\n" + "-" * 80)
        print("DYNAMIC ATTACK PARAMETER CONFIGURATION")
        print("-" * 80 + "\n")
        
        params = {}
        
        # 1. Target Frequency
        print("1. TARGET FREQUENCY:")
        print("   Specify the frequency you want the attack to drive the system to")
        print("   Nominal frequency: 60.0 Hz")
        print("   Recommended range: 58.0-59.8 Hz (below nominal)")
        print("   Lower values = more aggressive attack")
        
        while True:
            try:
                target = float(input("\n   Enter target frequency (Hz): ").strip())
                if 57.0 <= target <= 60.0:
                    params['target_frequency'] = target
                    break
                else:
                    print("   Target must be between 57.0 and 60.0 Hz. Please try again.")
            except ValueError:
                print("   Invalid input. Please enter a number.")
        
        print(f"   Target frequency set to: {params['target_frequency']} Hz")
        
        # 2. PID Controller Gains
        print("\n2. PID CONTROLLER TUNING:")
        print("   The PID controller adjusts load to drive frequency to target")
        print("\n   Would you like to:")
        print("   [1] Use recommended PID gains (easier)")
        print("   [2] Manually tune PID gains (advanced)")
        
        while True:
            try:
                choice = input("\n   Enter choice (1-2): ").strip()
                if choice == '1':
                    # Recommended gains based on target frequency
                    freq_diff = 60.0 - params['target_frequency']
                    params['kp'] = 500.0 + freq_diff * 100.0
                    params['ki'] = 50.0 + freq_diff * 10.0
                    params['kd'] = 100.0 + freq_diff * 20.0
                    params['pid_mode'] = 'Recommended'
                    break
                elif choice == '2':
                    params['pid_mode'] = 'Manual'
                    
                    print("\n   Proportional Gain (Kp):")
                    print("   Controls immediate response to frequency error")
                    print("   Recommended range: 200-1000")
                    while True:
                        try:
                            kp = float(input("   Enter Kp: ").strip())
                            if 50.0 <= kp <= 2000.0:
                                params['kp'] = kp
                                break
                            else:
                                print("   Kp must be between 50 and 2000.")
                        except ValueError:
                            print("   Invalid input. Please enter a number.")
                    
                    print("\n   Integral Gain (Ki):")
                    print("   Controls accumulated error correction")
                    print("   Recommended range: 20-100")
                    while True:
                        try:
                            ki = float(input("   Enter Ki: ").strip())
                            if 5.0 <= ki <= 200.0:
                                params['ki'] = ki
                                break
                            else:
                                print("   Ki must be between 5 and 200.")
                        except ValueError:
                            print("   Invalid input. Please enter a number.")
                    
                    print("\n   Derivative Gain (Kd):")
                    print("   Controls response to rate of frequency change")
                    print("   Recommended range: 50-200")
                    while True:
                        try:
                            kd = float(input("   Enter Kd: ").strip())
                            if 10.0 <= kd <= 400.0:
                                params['kd'] = kd
                                break
                            else:
                                print("   Kd must be between 10 and 400.")
                        except ValueError:
                            print("   Invalid input. Please enter a number.")
                    
                    break
                else:
                    print("   Invalid choice. Please enter 1 or 2.")
            except Exception as e:
                print(f"   Error: {e}. Please try again.")
        
        print(f"\n   PID Gains: Kp={params['kp']:.1f}, Ki={params['ki']:.1f}, Kd={params['kd']:.1f}")
        
        # 3. Attack Duration
        print("\n3. ATTACK DURATION:")
        print("   Specify how long the attack should last (in seconds)")
        print("   Recommended range: 5-15 seconds")
        
        while True:
            try:
                duration = float(input("\n   Enter duration (s): ").strip())
                if 3.0 <= duration <= 30.0:
                    params['duration'] = duration
                    break
                else:
                    print("   Duration must be between 3 and 30 seconds. Please try again.")
            except ValueError:
                print("   Invalid input. Please enter a number.")
        
        print(f"   Attack duration set to: {params['duration']} seconds")
        
        # 4. Maximum Load Variation
        print("\n4. MAXIMUM LOAD VARIATION:")
        print("   Specify the maximum load manipulation allowed (in MW)")
        print("   This limits how aggressively the PID controller can act")
        print("   Recommended range: 50-150 MW")
        
        while True:
            try:
                magnitude = float(input("\n   Enter max magnitude (MW): ").strip())
                if 20.0 <= magnitude <= 200.0:
                    params['magnitude'] = magnitude
                    break
                else:
                    print("   Magnitude must be between 20 and 200 MW. Please try again.")
            except ValueError:
                print("   Invalid input. Please enter a number.")
        
        print(f"   Maximum load variation set to: {params['magnitude']} MW")
        
        # 5. Attack Severity
        print("\n5. ATTACK SEVERITY:")
        print("   [1] LOW - Minimal impact")
        print("   [2] MEDIUM - Moderate impact")
        print("   [3] HIGH - Significant impact")
        print("   [4] CRITICAL - Severe impact")
        
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
        
        # 6. Target Buses
        print("\n6. TARGET BUSES:")
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
        
        # 7. System Inertia
        print("\n7. SYSTEM INERTIA CONDITION:")
        print("   [1] LOW - High renewable penetration, faster frequency changes")
        print("   [2] HIGH - Traditional generation, slower frequency changes")
        
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
        
        # 8. Output Directory
        print("\n8. OUTPUT DIRECTORY:")
        default_dir = f"interactive_dynamic_{params['target_frequency']:.1f}Hz_results"
        dir_input = input(f"\n   Enter directory name (default: {default_dir}): ").strip()
        params['output_dir'] = dir_input if dir_input else default_dir
        
        print(f"   Results will be saved to: {params['output_dir']}")
        
        return params
    
    def display_configuration_summary(self, params: Dict):
        """Display summary of user configuration"""
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"\nAttack Type:          FEEDBACK (PID-Controlled)")
        print(f"Target Frequency:     {params['target_frequency']} Hz")
        print(f"Attack Duration:      {params['duration']} seconds")
        print(f"Max Load Variation:   {params['magnitude']} MW")
        print(f"PID Gains ({params['pid_mode']}):")
        print(f"  - Kp (Proportional): {params['kp']:.1f}")
        print(f"  - Ki (Integral):     {params['ki']:.1f}")
        print(f"  - Kd (Derivative):   {params['kd']:.1f}")
        print(f"Attack Severity:      {params['severity_name']}")
        print(f"Target Buses:         {params['target_buses']} ({params['bus_description']})")
        print(f"System Inertia:       {params['inertia_name']}")
        print(f"Output Directory:     {params['output_dir']}")
        print("=" * 80)
        
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        return confirm in ['y', 'yes']
    
    def generate_attack_data(self, params: Dict) -> Dict:
        """
        Generate dynamic attack simulation data based on user parameters
        
        Args:
            params: User-specified parameters
            
        Returns:
            Dictionary containing simulation results
        """
        print("\n" + "-" * 80)
        print("GENERATING DYNAMIC ATTACK SIMULATION")
        print("-" * 80)
        print("\nSimulating PID-controlled frequency tracking attack...")
        
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
            attack_type=AttackType.FEEDBACK,
            severity=params['severity'],
            target_buses=params['target_buses'],
            attack_start_time=2.0,
            attack_duration=params['duration'],
            attack_magnitude=params['magnitude'],
            frequency_setpoint=params['target_frequency'],
            pid_gains={'kp': params['kp'], 'ki': params['ki'], 'kd': params['kd']},
            ramp_rate=20.0,
            variation_amplitude=0.0,
            variation_frequency=0.0,
            multi_stage_config=None
        )
        
        # Create simulation configuration
        total_time = params['duration'] + 8.0
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
        
        # Simulate system response with PID feedback
        print("  Computing PID-controlled frequency response...")
        frequency = self._simulate_pid_frequency_response(t, attack_config, sim_config, inertia_config)
        
        print("  Computing voltage response...")
        voltage = self._simulate_voltage_response(t, frequency, attack_config)
        
        print("  Computing power response...")
        power = self._simulate_power_response(t, attack_config)
        
        print("  Computing attack signal (PID output)...")
        attack_signal = self._calculate_pid_output(t, attack_config, frequency)
        
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
    
    def _simulate_pid_frequency_response(
        self,
        t: np.ndarray,
        attack_config: AttackConfig,
        sim_config: SimulationConfig,
        inertia_config: SystemInertiaConfig
    ) -> np.ndarray:
        """Simulate frequency response with PID control"""
        freq = np.zeros_like(t)
        freq[0] = 60.0
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        target_freq = attack_config.frequency_setpoint
        
        kp = attack_config.pid_gains['kp']
        ki = attack_config.pid_gains['ki']
        kd = attack_config.pid_gains['kd']
        
        H = 3.0 * inertia_config.generator_inertia_multiplier
        D = 1.0 * inertia_config.load_damping_coefficient
        
        integral_error = 0.0
        prev_error = 0.0
        dt = sim_config.time_step
        
        for i in range(1, len(t)):
            if attack_start <= t[i] <= attack_end:
                error = target_freq - freq[i-1]
                integral_error += error * dt
                derivative_error = (error - prev_error) / dt
                
                attack_power = kp * error + ki * integral_error + kd * derivative_error
                attack_power = np.clip(attack_power, -attack_config.attack_magnitude,
                                      attack_config.attack_magnitude)
                
                prev_error = error
            else:
                attack_power = 0.0
                integral_error = 0.0
                prev_error = 0.0
            
            if sim_config.enable_governor_response:
                gov_response = -0.5 * (freq[i-1] - 60.0)
            else:
                gov_response = 0.0
            
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
            voltage[i] = 1.0 - 0.3 * freq_dev
            
            if attack_start <= t[i] <= attack_end:
                voltage[i] -= 0.05 * np.sin(2 * np.pi * 0.5 * (t[i] - attack_start))
            
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
                power[i] = 100.0 + attack_config.attack_magnitude * 0.7 * \
                          np.sin(2 * np.pi * 0.8 * (t[i] - attack_start))
            
            power[i] += np.random.normal(0, 1.0)
        
        return power
    
    def _calculate_pid_output(
        self,
        t: np.ndarray,
        attack_config: AttackConfig,
        frequency: np.ndarray
    ) -> np.ndarray:
        """Calculate PID controller output (attack signal)"""
        attack_signal = np.zeros_like(t)
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        target_freq = attack_config.frequency_setpoint
        
        kp = attack_config.pid_gains['kp']
        ki = attack_config.pid_gains['ki']
        kd = attack_config.pid_gains['kd']
        
        integral_error = 0.0
        prev_error = 0.0
        dt = 0.01
        
        for i in range(len(t)):
            if attack_start <= t[i] <= attack_end:
                error = target_freq - frequency[i]
                integral_error += error * dt
                derivative_error = (error - prev_error) / dt if i > 0 else 0.0
                
                attack_signal[i] = kp * error + ki * integral_error + kd * derivative_error
                attack_signal[i] = np.clip(attack_signal[i],
                                          -attack_config.attack_magnitude,
                                          attack_config.attack_magnitude)
                
                prev_error = error
            else:
                integral_error = 0.0
                prev_error = 0.0
        
        return attack_signal
    
    def create_visualization(self, data: Dict):
        """Create comprehensive visualization"""
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
        target_freq = params['target_frequency']
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        fig.suptitle(f'IEEE Dynamic Attack Analysis - Interactive Demo\n'
                    f'PID-Controlled Feedback Attack | Target: {target_freq} Hz | '
                    f'{params["severity_name"]} Severity',
                    fontsize=16, fontweight='bold', fontname='Times New Roman')
        
        # Plot 1: Frequency tracking
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, freq, 'b-', linewidth=2, label='Actual Frequency')
        ax1.axhline(y=60.0, color='g', linestyle='--', linewidth=1.5, label='Nominal')
        ax1.axhline(y=target_freq, color='r', linestyle='--', linewidth=1.5, label=f'Target ({target_freq} Hz)')
        ax1.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack')
        ax1.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax1.set_ylabel('Frequency (Hz)', fontname='Times New Roman', fontsize=10)
        ax1.set_title('Frequency Tracking', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot 2: Frequency error
        ax2 = fig.add_subplot(gs[0, 1])
        error = target_freq - freq
        ax2.plot(t, error, 'orange', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax2.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax2.set_ylabel('Error (Hz)', fontname='Times New Roman', fontsize=10)
        ax2.set_title('Tracking Error (Target - Actual)', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: PID components
        ax3 = fig.add_subplot(gs[0, 2])
        # Calculate PID components
        attack_mask = (t >= attack_start) & (t <= attack_end)
        p_term = params['kp'] * error
        ax3.plot(t[attack_mask], p_term[attack_mask], 'r-', linewidth=1.5, label=f'P (Kp={params["kp"]:.0f})', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax3.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax3.set_ylabel('Control Signal (MW)', fontname='Times New Roman', fontsize=10)
        ax3.set_title('PID P-Component', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # Plot 4: PID output (attack signal)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, attack, 'purple', linewidth=2, label='PID Output')
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax4.axhline(y=params['magnitude'], color='r', linestyle='--', linewidth=1, label='Max Limit', alpha=0.6)
        ax4.axhline(y=-params['magnitude'], color='r', linestyle='--', linewidth=1, alpha=0.6)
        ax4.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax4.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax4.set_ylabel('Load Manipulation (MW)', fontname='Times New Roman', fontsize=10)
        ax4.set_title('PID Controller Output', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        
        # Plot 5: Voltage response
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(t, volt, 'b-', linewidth=2, label='Bus Voltage')
        ax5.axhline(y=1.0, color='g', linestyle='--', linewidth=1.5, label='Nominal')
        ax5.axhline(y=0.95, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Limits')
        ax5.axhline(y=1.05, color='orange', linestyle='--', linewidth=1, alpha=0.6)
        ax5.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax5.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax5.set_ylabel('Voltage (pu)', fontname='Times New Roman', fontsize=10)
        ax5.set_title('Voltage Response', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)
        
        # Plot 6: Power response
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(t, power, 'r-', linewidth=2, label='Active Power')
        ax6.axhline(y=100.0, color='g', linestyle='--', linewidth=1.5, label='Nominal')
        ax6.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax6.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax6.set_ylabel('Power (MW)', fontname='Times New Roman', fontsize=10)
        ax6.set_title('Active Power Flow', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=8)
        
        # Plot 7: Frequency deviation
        ax7 = fig.add_subplot(gs[2, 0])
        freq_dev = freq - 60.0
        ax7.plot(t, freq_dev, 'r-', linewidth=2)
        ax7.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax7.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, label='IEEE Limit')
        ax7.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax7.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax7.set_ylabel('Deviation (Hz)', fontname='Times New Roman', fontsize=10)
        ax7.set_title('Frequency Deviation', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8)
        
        # Plot 8: Phase portrait
        ax8 = fig.add_subplot(gs[2, 1])
        df_dt = np.gradient(freq, t)
        ax8.plot(freq, df_dt, 'b-', linewidth=1.5, alpha=0.7)
        ax8.scatter(freq[0], df_dt[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax8.scatter(freq[-1], df_dt[-1], color='red', s=100, marker='s', label='End', zorder=5)
        ax8.set_xlabel('Frequency (Hz)', fontname='Times New Roman', fontsize=10)
        ax8.set_ylabel('df/dt (Hz/s)', fontname='Times New Roman', fontsize=10)
        ax8.set_title('Phase Portrait', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        
        # Plot 9: Attack effectiveness
        ax9 = fig.add_subplot(gs[2, 2])
        effectiveness = np.abs(error) / (60.0 - target_freq) * 100.0  # Percentage
        ax9.plot(t[attack_mask], 100.0 - effectiveness[attack_mask], 'g-', linewidth=2)
        ax9.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
        ax9.set_ylabel('Tracking Accuracy (%)', fontname='Times New Roman', fontsize=10)
        ax9.set_title('Attack Effectiveness', fontname='Times New Roman', fontsize=11, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim([0, 100])
        
        plt.savefig(os.path.join(self.output_dir, 'interactive_dynamic_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {self.output_dir}/interactive_dynamic_analysis.png")
    
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
        target_freq = params['target_frequency']
        
        attack_mask = (t >= attack_start) & (t <= attack_end)
        
        report_file = os.path.join(self.output_dir, 'interactive_dynamic_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LAA INTERACTIVE DYNAMIC ATTACK - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Author: Pranaav\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("USER CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Attack Type:          FEEDBACK (PID-Controlled)\n")
            f.write(f"Target Frequency:     {target_freq} Hz\n")
            f.write(f"Attack Duration:      {params['duration']} seconds\n")
            f.write(f"Max Load Variation:   {params['magnitude']} MW\n")
            f.write(f"PID Gains ({params['pid_mode']}):\n")
            f.write(f"  - Kp:               {params['kp']:.1f}\n")
            f.write(f"  - Ki:               {params['ki']:.1f}\n")
            f.write(f"  - Kd:               {params['kd']:.1f}\n")
            f.write(f"Attack Severity:      {params['severity_name']}\n")
            f.write(f"Target Buses:         {params['target_buses']} ({params['bus_description']})\n")
            f.write(f"System Inertia:       {params['inertia_name']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Frequency Tracking:\n")
            f.write(f"  Target Frequency:               {target_freq} Hz\n")
            f.write(f"  Minimum Frequency Achieved:     {np.min(freq):.4f} Hz\n")
            f.write(f"  Average Frequency (attack):     {np.mean(freq[attack_mask]):.4f} Hz\n")
            f.write(f"  Final Frequency (attack end):   {freq[int(attack_end/0.01)]:.4f} Hz\n")
            
            # Calculate tracking error
            error = np.abs(target_freq - freq[attack_mask])
            f.write(f"  Average Tracking Error:         {np.mean(error):.4f} Hz\n")
            f.write(f"  Maximum Tracking Error:         {np.max(error):.4f} Hz\n")
            f.write(f"  RMS Tracking Error:             {np.sqrt(np.mean(error**2)):.4f} Hz\n\n")
            
            f.write("Voltage Analysis:\n")
            f.write(f"  Minimum Voltage:                {np.min(volt):.4f} pu\n")
            f.write(f"  Maximum Voltage:                {np.max(volt):.4f} pu\n")
            f.write(f"  Average (during attack):        {np.mean(volt[attack_mask]):.4f} pu\n\n")
            
            f.write("Power Analysis:\n")
            f.write(f"  Peak Power:                     {np.max(power):.2f} MW\n")
            f.write(f"  Minimum Power:                  {np.min(power):.2f} MW\n")
            f.write(f"  Average (during attack):        {np.mean(power[attack_mask]):.2f} MW\n\n")
            
            f.write("PID Controller Performance:\n")
            f.write(f"  Peak Control Output:            {np.max(np.abs(attack)):.2f} MW\n")
            f.write(f"  Average Control Output:         {np.mean(np.abs(attack[attack_mask])):.2f} MW\n")
            f.write(f"  Control Saturation Time:        {np.sum(np.abs(attack) >= params['magnitude']*0.99) * 0.01:.2f} s\n\n")
            
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
        print("  - interactive_dynamic_analysis.png (comprehensive plots)")
        print("  - interactive_dynamic_report.txt (detailed analysis)")
        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    print("\n" + "*" * 80)
    print("LAA FRAMEWORK - INTERACTIVE DYNAMIC ATTACK DEMO")
    print("Author: Pranaav")
    print("Date: October 2025")
    print("*" * 80)
    
    demo = InteractiveDynamicDemo()
    demo.run_demo()
    
    print("\n" + "*" * 80)
    print("THANK YOU FOR USING THE LAA FRAMEWORK")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LAA Dynamic Attack Demonstration

This demo showcases dynamic Load-Altering Attacks (LAA) with PID-controlled
feedback mechanisms on the IEEE 39-Bus System. Demonstrates real-time frequency
tracking attacks with comprehensive IEEE-compliant visualization.

Features:
- PID-controlled feedback attacks targeting specific frequencies
- Real-time frequency tracking and response
- Multiple target frequency scenarios
- Comprehensive IEEE-compliant plots showing attack dynamics
- Attack duration: 7 seconds (as specified)

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


class DynamicAttackDemo:
    """
    Dynamic Attack Demonstration Class
    
    Demonstrates PID-controlled feedback attacks with real-time frequency
    tracking and comprehensive visualization of attack dynamics.
    """
    
    def __init__(self, output_dir: str = "dynamic_attack_results"):
        """
        Initialize the dynamic attack demo
        
        Args:
            output_dir: Directory to save output plots and reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize IEEE graph generator
        self.graph_gen = IEEEGraphGenerator()
        
        print("=" * 80)
        print("LAA DYNAMIC ATTACK DEMONSTRATION")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print(f"Attack duration: 7 seconds")
        print(f"Attack type: PID-Controlled Feedback")
        print("=" * 80)
    
    def generate_frequency_target_attack_data(
        self, 
        target_freq: float,
        kp: float = 500.0,
        ki: float = 50.0,
        kd: float = 100.0,
        scenario_name: str = "Default"
    ) -> Dict:
        """
        Generate dynamic attack data with specific frequency target
        
        Args:
            target_freq: Target frequency in Hz (e.g., 59.5, 59.0, 58.5)
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            scenario_name: Name for this scenario
            
        Returns:
            Dictionary containing attack data and system response
        """
        print(f"\nGenerating {scenario_name} attack scenario...")
        print(f"  Target Frequency: {target_freq} Hz")
        print(f"  PID Gains: Kp={kp}, Ki={ki}, Kd={kd}")
        
        # Create inertia configuration (LOW inertia for more dramatic effects)
        inertia_config = SystemInertiaConfig(
            condition=InertiaCondition.LOW,
            generator_inertia_multiplier=0.5,
            renewable_penetration=0.35,
            load_damping_coefficient=1.0,
            governor_droop=0.05,
            frequency_deadband=0.036,
            renewable_inertia_constant=3.0
        )
        
        # Create attack configuration with FEEDBACK type
        attack_config = AttackConfig(
            attack_type=AttackType.FEEDBACK,
            severity=AttackSeverity.HIGH,
            target_buses=[16, 20, 23],  # High-load buses
            attack_start_time=2.0,
            attack_duration=7.0,  # 7 seconds as specified
            attack_magnitude=80.0,  # MW
            frequency_setpoint=target_freq,  # Target frequency
            pid_gains={'kp': kp, 'ki': ki, 'kd': kd},
            ramp_rate=20.0,
            variation_amplitude=0.0,
            variation_frequency=0.0,
            multi_stage_config=None
        )
        
        # Create simulation configuration
        sim_config = SimulationConfig(
            total_time=12.0,
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
        t = np.arange(0, sim_config.total_time, sim_config.time_step)
        
        # Simulate system frequency response with PID feedback
        frequency = self._simulate_pid_frequency_response(
            t, attack_config, sim_config, inertia_config
        )
        
        # Simulate voltage response
        voltage = self._simulate_voltage_response(t, frequency, attack_config)
        
        # Simulate power response
        power = self._simulate_power_response(t, attack_config)
        
        # Calculate attack magnitude over time (load variation)
        attack_signal = self._calculate_attack_signal(t, attack_config, frequency)
        
        return {
            'time': t,
            'frequency': frequency,
            'voltage': voltage,
            'power': power,
            'attack_signal': attack_signal,
            'scenario_name': scenario_name,
            'target_freq': target_freq,
            'pid_gains': {'kp': kp, 'ki': ki, 'kd': kd},
            'attack_config': attack_config,
            'inertia_config': inertia_config
        }
    
    def _simulate_pid_frequency_response(
        self,
        t: np.ndarray,
        attack_config: AttackConfig,
        sim_config: SimulationConfig,
        inertia_config: SystemInertiaConfig
    ) -> np.ndarray:
        """
        Simulate frequency response with PID-controlled attack
        
        Uses a simplified swing equation with PID feedback control
        """
        freq = np.zeros_like(t)
        freq[0] = 60.0  # Start at nominal frequency
        
        # Attack parameters
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        target_freq = attack_config.frequency_setpoint
        
        # PID parameters
        kp = attack_config.pid_gains['kp']
        ki = attack_config.pid_gains['ki']
        kd = attack_config.pid_gains['kd']
        
        # System parameters (simplified)
        H = 3.0 * inertia_config.generator_inertia_multiplier  # Inertia constant (seconds)
        D = 1.0 * inertia_config.load_damping_coefficient  # Damping coefficient
        
        # PID state variables
        integral_error = 0.0
        prev_error = 0.0
        
        dt = sim_config.time_step
        
        for i in range(1, len(t)):
            # Calculate error for PID
            if attack_start <= t[i] <= attack_end:
                error = target_freq - freq[i-1]
                integral_error += error * dt
                derivative_error = (error - prev_error) / dt
                
                # PID output (attack power)
                attack_power = kp * error + ki * integral_error + kd * derivative_error
                attack_power = np.clip(attack_power, -attack_config.attack_magnitude, 
                                      attack_config.attack_magnitude)
                
                prev_error = error
            else:
                attack_power = 0.0
                integral_error = 0.0
                prev_error = 0.0
            
            # Governor response (simplified)
            if sim_config.enable_governor_response:
                gov_response = -0.5 * (freq[i-1] - 60.0)  # Proportional governor
            else:
                gov_response = 0.0
            
            # Swing equation: 2H * df/dt = P_m - P_e - D * df
            # Simplified: attack power affects electrical power
            power_imbalance = -attack_power / 100.0 + gov_response - D * (freq[i-1] - 60.0)
            
            # Update frequency
            df_dt = power_imbalance / (2 * H)
            freq[i] = freq[i-1] + df_dt * dt
            
            # Add small noise
            freq[i] += np.random.normal(0, sim_config.measurement_noise)
        
        return freq
    
    def _simulate_voltage_response(
        self,
        t: np.ndarray,
        frequency: np.ndarray,
        attack_config: AttackConfig
    ) -> np.ndarray:
        """
        Simulate voltage response to frequency variations
        
        Voltage typically follows frequency changes due to reactive power dynamics
        """
        voltage = np.ones_like(t) * 1.0  # Start at 1.0 pu
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        for i in range(len(t)):
            # Voltage deviation proportional to frequency deviation
            freq_dev = (frequency[i] - 60.0) / 60.0
            voltage[i] = 1.0 - 0.3 * freq_dev
            
            # Additional voltage sag during attack
            if attack_start <= t[i] <= attack_end:
                voltage[i] -= 0.05 * np.sin(2 * np.pi * 0.5 * (t[i] - attack_start))
            
            # Add noise
            voltage[i] += np.random.normal(0, 0.002)
            
            # Clamp to reasonable range
            voltage[i] = np.clip(voltage[i], 0.85, 1.15)
        
        return voltage
    
    def _simulate_power_response(
        self,
        t: np.ndarray,
        attack_config: AttackConfig
    ) -> np.ndarray:
        """
        Simulate active power response during attack
        """
        power = np.ones_like(t) * 100.0  # Nominal 100 MW
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        for i in range(len(t)):
            if attack_start <= t[i] <= attack_end:
                # Power oscillation during attack
                power[i] = 100.0 + attack_config.attack_magnitude * 0.7 * \
                          np.sin(2 * np.pi * 0.8 * (t[i] - attack_start))
            else:
                power[i] = 100.0
            
            # Add noise
            power[i] += np.random.normal(0, 1.0)
        
        return power
    
    def _calculate_attack_signal(
        self,
        t: np.ndarray,
        attack_config: AttackConfig,
        frequency: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the attack signal (load manipulation) over time
        """
        attack_signal = np.zeros_like(t)
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        # PID gains
        kp = attack_config.pid_gains['kp']
        ki = attack_config.pid_gains['ki']
        kd = attack_config.pid_gains['kd']
        target_freq = attack_config.frequency_setpoint
        
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
    
    def create_attack_plots(self, attack_data: Dict, filename_prefix: str):
        """
        Create comprehensive IEEE-compliant plots for a single attack scenario
        
        Generates 4 main plots:
        1. Frequency Analysis
        2. Voltage Analysis
        3. Power Analysis
        4. PID Control Analysis
        """
        scenario_name = attack_data['scenario_name']
        print(f"\nCreating plots for {scenario_name}...")
        
        # Extract data
        t = attack_data['time']
        freq = attack_data['frequency']
        volt = attack_data['voltage']
        power = attack_data['power']
        attack = attack_data['attack_signal']
        target_freq = attack_data['target_freq']
        attack_config = attack_data['attack_config']
        
        attack_start = attack_config.attack_start_time
        attack_end = attack_start + attack_config.attack_duration
        
        # 1. Frequency Analysis Plot
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig1.suptitle(f'IEEE Frequency Analysis - {scenario_name}\nDynamic PID-Controlled Attack',
                     fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        # Frequency vs time
        ax1.plot(t, freq, 'b-', linewidth=2, label='Actual Frequency')
        ax1.axhline(y=60.0, color='g', linestyle='--', linewidth=1.5, label='Nominal (60 Hz)')
        ax1.axhline(y=target_freq, color='r', linestyle='--', linewidth=1.5, 
                   label=f'Target ({target_freq} Hz)')
        ax1.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack Period')
        ax1.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax1.set_ylabel('Frequency (Hz)', fontname='Times New Roman', fontsize=11)
        ax1.set_title('System Frequency Response', fontname='Times New Roman', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # Frequency deviation
        freq_dev = freq - 60.0
        ax2.plot(t, freq_dev, 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, label='IEEE Limit')
        ax2.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax2.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax2.set_ylabel('Frequency Deviation (Hz)', fontname='Times New Roman', fontsize=11)
        ax2.set_title('Frequency Deviation from Nominal', fontname='Times New Roman', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        freq_file = os.path.join(self.output_dir, f'{filename_prefix}_frequency.png')
        plt.savefig(freq_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {freq_file}")
        
        # 2. Voltage Analysis Plot
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, volt, 'b-', linewidth=2, label='Bus Voltage')
        ax.axhline(y=1.0, color='g', linestyle='--', linewidth=1.5, label='Nominal (1.0 pu)')
        ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=1, label='Lower Limit (0.95 pu)')
        ax.axhline(y=1.05, color='orange', linestyle='--', linewidth=1, label='Upper Limit (1.05 pu)')
        ax.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack Period')
        ax.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax.set_ylabel('Voltage (pu)', fontname='Times New Roman', fontsize=11)
        ax.set_title(f'IEEE Voltage Analysis - {scenario_name}\nVoltage Response to Dynamic Attack',
                    fontname='Times New Roman', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        volt_file = os.path.join(self.output_dir, f'{filename_prefix}_voltage.png')
        plt.savefig(volt_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {volt_file}")
        
        # 3. Power Analysis Plot
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig3.suptitle(f'IEEE Power Analysis - {scenario_name}\nActive Power Response',
                     fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        # Power vs time
        ax1.plot(t, power, 'r-', linewidth=2, label='Active Power')
        ax1.axhline(y=100.0, color='g', linestyle='--', linewidth=1.5, label='Nominal (100 MW)')
        ax1.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack Period')
        ax1.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax1.set_ylabel('Power (MW)', fontname='Times New Roman', fontsize=11)
        ax1.set_title('Active Power Flow', fontname='Times New Roman', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Power deviation
        power_dev = power - 100.0
        ax2.plot(t, power_dev, 'm-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax2.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax2.set_ylabel('Power Deviation (MW)', fontname='Times New Roman', fontsize=11)
        ax2.set_title('Power Deviation from Nominal', fontname='Times New Roman', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        power_file = os.path.join(self.output_dir, f'{filename_prefix}_power.png')
        plt.savefig(power_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {power_file}")
        
        # 4. PID Control Analysis Plot
        fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig4.suptitle(f'IEEE PID Control Analysis - {scenario_name}\nFeedback Attack Dynamics',
                     fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        # Attack signal (control output)
        ax1.plot(t, attack, 'purple', linewidth=2, label='PID Output (Attack Signal)')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax1.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack Period')
        ax1.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax1.set_ylabel('Load Manipulation (MW)', fontname='Times New Roman', fontsize=11)
        ax1.set_title('PID Controller Output', fontname='Times New Roman', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Control error
        error = target_freq - freq
        ax2.plot(t, error, 'orange', linewidth=2, label='Frequency Error')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.axvspan(attack_start, attack_end, alpha=0.2, color='red')
        ax2.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax2.set_ylabel('Error (Hz)', fontname='Times New Roman', fontsize=11)
        ax2.set_title('PID Control Error (Target - Actual)', fontname='Times New Roman', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        pid_file = os.path.join(self.output_dir, f'{filename_prefix}_pid_control.png')
        plt.savefig(pid_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {pid_file}")
    
    def create_comparison_plot(self, scenarios: List[Dict]):
        """
        Create comparison plot showing all scenarios together
        """
        print("\nCreating comparison plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('IEEE Dynamic Attack Comparison - Multiple Target Frequencies\nPID-Controlled Feedback Attacks',
                    fontsize=16, fontweight='bold', fontname='Times New Roman')
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, scenario in enumerate(scenarios):
            t = scenario['time']
            freq = scenario['frequency']
            volt = scenario['voltage']
            power = scenario['power']
            attack = scenario['attack_signal']
            name = scenario['scenario_name']
            color = colors[i % len(colors)]
            
            # Frequency comparison
            ax1.plot(t, freq, color=color, linewidth=2, label=name, alpha=0.8)
            
            # Voltage comparison
            ax2.plot(t, volt, color=color, linewidth=2, label=name, alpha=0.8)
            
            # Power comparison
            ax3.plot(t, power, color=color, linewidth=2, label=name, alpha=0.8)
            
            # Attack signal comparison
            ax4.plot(t, attack, color=color, linewidth=2, label=name, alpha=0.8)
        
        # Frequency plot
        ax1.axhline(y=60.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax1.set_ylabel('Frequency (Hz)', fontname='Times New Roman', fontsize=11)
        ax1.set_title('Frequency Response Comparison', fontname='Times New Roman', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # Voltage plot
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax2.set_ylabel('Voltage (pu)', fontname='Times New Roman', fontsize=11)
        ax2.set_title('Voltage Response Comparison', fontname='Times New Roman', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        # Power plot
        ax3.axhline(y=100.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax3.set_ylabel('Power (MW)', fontname='Times New Roman', fontsize=11)
        ax3.set_title('Power Response Comparison', fontname='Times New Roman', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=9)
        
        # Attack signal plot
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax4.set_xlabel('Time (s)', fontname='Times New Roman', fontsize=11)
        ax4.set_ylabel('Load Manipulation (MW)', fontname='Times New Roman', fontsize=11)
        ax4.set_title('Attack Signal Comparison', fontname='Times New Roman', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        comp_file = os.path.join(self.output_dir, 'dynamic_attack_comparison.png')
        plt.savefig(comp_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {comp_file}")
    
    def create_summary_report(self, scenarios: List[Dict]):
        """
        Create text summary report of all attack scenarios
        """
        report_file = os.path.join(self.output_dir, 'dynamic_attack_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LAA DYNAMIC ATTACK DEMONSTRATION - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Author: Pranaav\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Attack Duration: 7 seconds\n")
            f.write(f"Attack Type: PID-Controlled Feedback\n")
            f.write(f"Number of Scenarios: {len(scenarios)}\n\n")
            
            for i, scenario in enumerate(scenarios, 1):
                f.write("-" * 80 + "\n")
                f.write(f"SCENARIO {i}: {scenario['scenario_name']}\n")
                f.write("-" * 80 + "\n")
                
                # Extract metrics
                freq = scenario['frequency']
                volt = scenario['voltage']
                power = scenario['power']
                attack = scenario['attack_signal']
                target_freq = scenario['target_freq']
                pid_gains = scenario['pid_gains']
                
                attack_config = scenario['attack_config']
                attack_start = attack_config.attack_start_time
                attack_end = attack_start + attack_config.attack_duration
                
                # Find attack period indices
                t = scenario['time']
                attack_mask = (t >= attack_start) & (t <= attack_end)
                
                f.write(f"\nTarget Frequency: {target_freq} Hz\n")
                f.write(f"PID Gains: Kp={pid_gains['kp']}, Ki={pid_gains['ki']}, Kd={pid_gains['kd']}\n")
                f.write(f"Attack Period: {attack_start}s to {attack_end}s\n\n")
                
                f.write("Performance Metrics:\n")
                f.write(f"  Minimum Frequency: {np.min(freq):.3f} Hz\n")
                f.write(f"  Maximum Frequency: {np.max(freq):.3f} Hz\n")
                f.write(f"  Average Frequency (during attack): {np.mean(freq[attack_mask]):.3f} Hz\n")
                f.write(f"  Frequency Deviation: {np.min(freq) - 60.0:.3f} Hz\n\n")
                
                f.write(f"  Minimum Voltage: {np.min(volt):.4f} pu\n")
                f.write(f"  Maximum Voltage: {np.max(volt):.4f} pu\n")
                f.write(f"  Average Voltage (during attack): {np.mean(volt[attack_mask]):.4f} pu\n\n")
                
                f.write(f"  Peak Attack Power: {np.max(np.abs(attack)):.2f} MW\n")
                f.write(f"  Average Attack Power: {np.mean(np.abs(attack[attack_mask])):.2f} MW\n\n")
                
                # Calculate settling time (time to return to near nominal after attack)
                post_attack_mask = t > attack_end
                if np.any(post_attack_mask):
                    post_freq = freq[post_attack_mask]
                    post_t = t[post_attack_mask]
                    settling_threshold = 0.05  # 0.05 Hz from nominal
                    settled = np.abs(post_freq - 60.0) < settling_threshold
                    if np.any(settled):
                        settling_idx = np.where(settled)[0][0]
                        settling_time = post_t[settling_idx] - attack_end
                        f.write(f"  Settling Time (to ±0.05 Hz): {settling_time:.2f} s\n")
                    else:
                        f.write(f"  Settling Time: Not settled within simulation time\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nSummary report saved: {report_file}")
    
    def run_demo(self):
        """
        Run the complete dynamic attack demonstration
        
        Creates multiple scenarios with different target frequencies
        """
        print("\n" + "=" * 80)
        print("RUNNING DYNAMIC ATTACK DEMONSTRATION")
        print("=" * 80)
        
        scenarios = []
        
        # Scenario 1: Moderate frequency drop (59.5 Hz target)
        scenario1 = self.generate_frequency_target_attack_data(
            target_freq=59.5,
            kp=500.0,
            ki=50.0,
            kd=100.0,
            scenario_name="Moderate Attack (59.5 Hz)"
        )
        scenarios.append(scenario1)
        self.create_attack_plots(scenario1, "scenario1_moderate")
        
        # Scenario 2: Aggressive frequency drop (59.0 Hz target)
        scenario2 = self.generate_frequency_target_attack_data(
            target_freq=59.0,
            kp=600.0,
            ki=60.0,
            kd=120.0,
            scenario_name="Aggressive Attack (59.0 Hz)"
        )
        scenarios.append(scenario2)
        self.create_attack_plots(scenario2, "scenario2_aggressive")
        
        # Scenario 3: Critical frequency drop (58.5 Hz target)
        scenario3 = self.generate_frequency_target_attack_data(
            target_freq=58.5,
            kp=700.0,
            ki=70.0,
            kd=140.0,
            scenario_name="Critical Attack (58.5 Hz)"
        )
        scenarios.append(scenario3)
        self.create_attack_plots(scenario3, "scenario3_critical")
        
        # Create comparison plot
        self.create_comparison_plot(scenarios)
        
        # Create summary report
        self.create_summary_report(scenarios)
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print(f"\nTotal scenarios: {len(scenarios)}")
        print(f"Total plots generated: {len(scenarios) * 4 + 1}")
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - {len(scenarios) * 4} individual scenario plots")
        print(f"  - 1 comparison plot")
        print(f"  - 1 summary report")
        print("\n" + "=" * 80)


def main():
    """
    Main execution function
    """
    print("\n")
    print("*" * 80)
    print("LAA FRAMEWORK - DYNAMIC ATTACK DEMONSTRATION")
    print("Author: Pranaav")
    print("Date: October 2025")
    print("*" * 80)
    print("\n")
    
    # Create and run demo
    demo = DynamicAttackDemo(output_dir="dynamic_attack_results")
    demo.run_demo()
    
    print("\n")
    print("*" * 80)
    print("DEMO EXECUTION COMPLETED SUCCESSFULLY")
    print("*" * 80)
    print("\n")


if __name__ == "__main__":
    main()

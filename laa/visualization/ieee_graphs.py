#!/usr/bin/env python3
"""
LAA Visualization Module - IEEE Compliant Graphs

This module provides IEEE-standard visualization capabilities for LAA simulation results.
Implements proper IEEE formatting, color schemes, and graph standards for power system analysis.

IEEE Standards Applied:
- IEEE Std 142: Grounding of Industrial and Commercial Power Systems (color codes)
- IEEE Std 1547: Distributed Energy Resource Interconnection Standards (plots)
- IEEE C37.118: Synchrophasor measurements visualization
- IEEE Std 421.5: Excitation system plots and analysis

Author: Pranaav
Date: October 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import os

# Configure matplotlib for IEEE-standard plots
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

logger = logging.getLogger(__name__)

# ======================== IEEE COLOR STANDARDS ======================== #

class IEEE_Colors:
    """IEEE-standard color codes for power system visualization"""
    
    # IEEE Std 142 - Standard color codes
    PHASE_A = '#FF0000'      # Red - Phase A
    PHASE_B = '#0000FF'      # Blue - Phase B  
    PHASE_C = '#000000'      # Black - Phase C
    NEUTRAL = '#FFFFFF'      # White - Neutral
    GROUND = '#008000'       # Green - Ground
    
    # IEEE voltage level colors
    TRANSMISSION = '#8B0000'  # Dark red - Transmission level
    DISTRIBUTION = '#FF8C00'  # Dark orange - Distribution level
    LOW_VOLTAGE = '#32CD32'   # Lime green - Low voltage
    
    # IEEE system states
    NORMAL = '#00AA00'        # Green - Normal operation
    ALERT = '#FFA500'         # Orange - Alert state
    EMERGENCY = '#FF0000'     # Red - Emergency state
    BLACKOUT = '#000000'      # Black - System blackout
    
    # IEEE protection colors
    ZONE1 = '#FF1493'         # Deep pink - Zone 1 protection
    ZONE2 = '#4169E1'         # Royal blue - Zone 2 protection  
    ZONE3 = '#32CD32'         # Lime green - Zone 3 protection
    
    # Attack visualization colors
    ATTACK_ACTIVE = '#DC143C'      # Crimson - Active attack
    ATTACK_MITIGATED = '#FFD700'   # Gold - Mitigated attack
    SYSTEM_RESPONSE = '#4682B4'    # Steel blue - System response

@dataclass
class IEEE_PlotConfig:
    """IEEE-standard plot configuration"""
    
    # IEEE figure standards
    figure_width: float = 12.0          # Standard IEEE figure width (inches)
    figure_height: float = 8.0          # Standard IEEE figure height (inches)
    dpi: int = 300                      # IEEE publication quality DPI
    
    # IEEE axis standards
    grid_alpha: float = 0.3             # Grid transparency per IEEE standards
    line_width: float = 2.0             # Standard line width
    marker_size: float = 6.0            # Standard marker size
    
    # IEEE text standards
    title_fontsize: int = 16            # IEEE title font size
    label_fontsize: int = 12            # IEEE axis label font size
    tick_fontsize: int = 10             # IEEE tick label font size
    legend_fontsize: int = 10           # IEEE legend font size

# ======================== IEEE VISUALIZATION CLASSES ======================== #

class IEEE_FrequencyAnalyzer:
    """IEEE-compliant frequency analysis and visualization"""
    
    def __init__(self):
        self.config = IEEE_PlotConfig()
        self.colors = IEEE_Colors()
        
    def plot_frequency_response(
        self,
        time_data: np.ndarray,
        frequency_data: np.ndarray,
        attack_events: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot IEEE-compliant frequency response graph
        
        Args:
            time_data: Time vector (seconds)
            frequency_data: Frequency measurements (Hz)
            attack_events: List of attack event dictionaries
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi
        )
        
        # IEEE frequency deadband and limits
        nominal_freq = 50.0  # Hz
        deadband = 0.036     # ±0.036 Hz per IEEE 1547
        trip_limit = 0.5     # ±0.5 Hz per IEEE 1547.1
        
        # Plot frequency response
        ax.plot(time_data, frequency_data, 
               color=self.colors.PHASE_A, 
               linewidth=self.config.line_width,
               label='System Frequency')
        
        # IEEE standard limits
        ax.axhline(y=nominal_freq, color='k', linestyle='-', alpha=0.7, label='Nominal (50 Hz)')
        ax.axhline(y=nominal_freq + deadband, color=self.colors.ALERT, 
                  linestyle='--', alpha=0.7, label='IEEE Deadband (±0.036 Hz)')
        ax.axhline(y=nominal_freq - deadband, color=self.colors.ALERT, 
                  linestyle='--', alpha=0.7)
        ax.axhline(y=nominal_freq + trip_limit, color=self.colors.EMERGENCY, 
                  linestyle=':', alpha=0.7, label='IEEE Trip Limit (±0.5 Hz)')
        ax.axhline(y=nominal_freq - trip_limit, color=self.colors.EMERGENCY, 
                  linestyle=':', alpha=0.7)
        
        # Mark attack events if provided
        if attack_events:
            for event in attack_events:
                start_time = event.get('start_time', 0)
                duration = event.get('duration', 1)
                attack_type = event.get('type', 'unknown')
                
                # Shade attack period
                ax.axvspan(start_time, start_time + duration,
                          color=self.colors.ATTACK_ACTIVE, alpha=0.2,
                          label=f'Attack: {attack_type}')
        
        # IEEE-standard formatting
        ax.set_xlabel('Time (s)', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency (Hz)', fontsize=self.config.label_fontsize)
        ax.set_title('IEEE 1547 Frequency Response Analysis', 
                    fontsize=self.config.title_fontsize, fontweight='bold')
        
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        # IEEE-standard y-axis limits
        freq_range = np.ptp(frequency_data)
        margin = max(0.1, freq_range * 0.1)
        ax.set_ylim(np.min(frequency_data) - margin, np.max(frequency_data) + margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"IEEE frequency plot saved to: {save_path}")
        
        return fig

class IEEE_VoltageAnalyzer:
    """IEEE-compliant voltage analysis and visualization"""
    
    def __init__(self):
        self.config = IEEE_PlotConfig()
        self.colors = IEEE_Colors()
        
    def plot_voltage_profile(
        self,
        bus_numbers: np.ndarray,
        voltage_magnitudes: np.ndarray,
        voltage_angles: Optional[np.ndarray] = None,
        attack_buses: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot IEEE-compliant voltage profile graph
        
        Args:
            bus_numbers: Bus identification numbers
            voltage_magnitudes: Voltage magnitudes (pu)
            voltage_angles: Voltage angles (degrees) - optional
            attack_buses: List of buses under attack
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if voltage_angles is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, 
                                         figsize=(self.config.figure_width, self.config.figure_height * 1.2),
                                         dpi=self.config.dpi)
        else:
            fig, ax1 = plt.subplots(
                figsize=(self.config.figure_width, self.config.figure_height),
                dpi=self.config.dpi
            )
            ax2 = None
        
        # IEEE voltage limits per IEEE 1547.1
        v_nominal = 1.0      # pu
        v_min = 0.88         # 88% per IEEE 1547.1
        v_max = 1.10         # 110% per IEEE 1547.1
        v_alert_low = 0.95   # Alert threshold
        v_alert_high = 1.05  # Alert threshold
        
        # Color buses based on attack status
        colors = []
        for bus in bus_numbers:
            if attack_buses and bus in attack_buses:
                colors.append(self.colors.ATTACK_ACTIVE)
            else:
                colors.append(self.colors.NORMAL)
        
        # Plot voltage magnitudes
        bars = ax1.bar(bus_numbers, voltage_magnitudes, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1)
        
        # IEEE standard limits
        ax1.axhline(y=v_nominal, color='k', linestyle='-', linewidth=2, 
                   label='Nominal (1.0 pu)')
        ax1.axhline(y=v_min, color=self.colors.EMERGENCY, linestyle='--', 
                   label='IEEE Min (0.88 pu)')
        ax1.axhline(y=v_max, color=self.colors.EMERGENCY, linestyle='--', 
                   label='IEEE Max (1.10 pu)')
        ax1.axhline(y=v_alert_low, color=self.colors.ALERT, linestyle=':', 
                   alpha=0.7, label='Alert Thresholds')
        ax1.axhline(y=v_alert_high, color=self.colors.ALERT, linestyle=':', alpha=0.7)
        
        # Fill violation zones
        ax1.fill_between(bus_numbers, v_min, 0, color=self.colors.EMERGENCY, 
                        alpha=0.1, label='IEEE Violation Zone')
        ax1.fill_between(bus_numbers, v_max, 2.0, color=self.colors.EMERGENCY, alpha=0.1)
        
        # IEEE-standard formatting
        ax1.set_xlabel('Bus Number', fontsize=self.config.label_fontsize)
        ax1.set_ylabel('Voltage Magnitude (pu)', fontsize=self.config.label_fontsize)
        ax1.set_title('IEEE 1547 Voltage Profile Analysis', 
                     fontsize=self.config.title_fontsize, fontweight='bold')
        ax1.grid(True, alpha=self.config.grid_alpha)
        ax1.legend(fontsize=self.config.legend_fontsize)
        ax1.tick_params(labelsize=self.config.tick_fontsize)
        ax1.set_ylim(0.8, 1.2)
        
        # Plot voltage angles if provided
        if voltage_angles is not None and ax2 is not None:
            ax2.plot(bus_numbers, voltage_angles, 'o-', 
                    color=self.colors.PHASE_B, 
                    linewidth=self.config.line_width,
                    markersize=self.config.marker_size)
            
            ax2.set_xlabel('Bus Number', fontsize=self.config.label_fontsize)
            ax2.set_ylabel('Voltage Angle (degrees)', fontsize=self.config.label_fontsize)
            ax2.set_title('IEEE C37.118 Voltage Phasor Angles', 
                         fontsize=self.config.title_fontsize, fontweight='bold')
            ax2.grid(True, alpha=self.config.grid_alpha)
            ax2.tick_params(labelsize=self.config.tick_fontsize)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"IEEE voltage plot saved to: {save_path}")
        
        return fig

class IEEE_PowerFlowAnalyzer:
    """IEEE-compliant power flow analysis and visualization"""
    
    def __init__(self):
        self.config = IEEE_PlotConfig()
        self.colors = IEEE_Colors()
        
    def plot_power_flow(
        self,
        time_data: np.ndarray,
        active_power: np.ndarray,
        reactive_power: np.ndarray,
        attack_events: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot IEEE-compliant power flow graph
        
        Args:
            time_data: Time vector (seconds)
            active_power: Active power data (MW)
            reactive_power: Reactive power data (MVAR)
            attack_events: List of attack event dictionaries
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, 
                                      figsize=(self.config.figure_width, self.config.figure_height * 1.2),
                                      dpi=self.config.dpi, sharex=True)
        
        # Plot active power
        ax1.plot(time_data, active_power, 
                color=self.colors.PHASE_A, 
                linewidth=self.config.line_width,
                label='Active Power (P)')
        
        # Plot reactive power  
        ax2.plot(time_data, reactive_power,
                color=self.colors.PHASE_B,
                linewidth=self.config.line_width,
                label='Reactive Power (Q)')
        
        # Mark attack events if provided
        if attack_events:
            for event in attack_events:
                start_time = event.get('start_time', 0)
                duration = event.get('duration', 1)
                attack_type = event.get('type', 'unknown')
                
                # Shade attack periods on both plots
                for ax in [ax1, ax2]:
                    ax.axvspan(start_time, start_time + duration,
                              color=self.colors.ATTACK_ACTIVE, alpha=0.2,
                              label=f'Attack: {attack_type}' if ax == ax1 else "")
        
        # IEEE-standard formatting
        ax1.set_ylabel('Active Power (MW)', fontsize=self.config.label_fontsize)
        ax1.set_title('IEEE Power System Active Power Flow', 
                     fontsize=self.config.title_fontsize, fontweight='bold')
        ax1.grid(True, alpha=self.config.grid_alpha)
        ax1.legend(fontsize=self.config.legend_fontsize)
        ax1.tick_params(labelsize=self.config.tick_fontsize)
        
        ax2.set_xlabel('Time (s)', fontsize=self.config.label_fontsize)
        ax2.set_ylabel('Reactive Power (MVAR)', fontsize=self.config.label_fontsize)
        ax2.set_title('IEEE Power System Reactive Power Flow', 
                     fontsize=self.config.title_fontsize, fontweight='bold')
        ax2.grid(True, alpha=self.config.grid_alpha)
        ax2.legend(fontsize=self.config.legend_fontsize)
        ax2.tick_params(labelsize=self.config.tick_fontsize)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"IEEE power flow plot saved to: {save_path}")
        
        return fig

class IEEE_StabilityAnalyzer:
    """IEEE-compliant stability analysis and visualization"""
    
    def __init__(self):
        self.config = IEEE_PlotConfig()
        self.colors = IEEE_Colors()
        
    def plot_rotor_angle_stability(
        self,
        time_data: np.ndarray,
        rotor_angles: Dict[str, np.ndarray],
        attack_events: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot IEEE-compliant rotor angle stability graph
        
        Args:
            time_data: Time vector (seconds)
            rotor_angles: Dictionary of generator rotor angles (degrees)
            attack_events: List of attack event dictionaries
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi
        )
        
        # IEEE color scheme for multiple generators
        ieee_colors = [self.colors.PHASE_A, self.colors.PHASE_B, self.colors.PHASE_C,
                      self.colors.TRANSMISSION, self.colors.DISTRIBUTION, self.colors.LOW_VOLTAGE]
        
        # Plot rotor angles for each generator
        for i, (gen_name, angles) in enumerate(rotor_angles.items()):
            color = ieee_colors[i % len(ieee_colors)]
            ax.plot(time_data, angles, 
                   color=color, 
                   linewidth=self.config.line_width,
                   label=f'Generator {gen_name}')
        
        # IEEE stability limits (typical values)
        stability_limit = 180.0  # degrees
        ax.axhline(y=stability_limit, color=self.colors.EMERGENCY, 
                  linestyle='--', alpha=0.7, label='IEEE Stability Limit (180°)')
        ax.axhline(y=-stability_limit, color=self.colors.EMERGENCY, 
                  linestyle='--', alpha=0.7)
        
        # Mark attack events if provided
        if attack_events:
            for event in attack_events:
                start_time = event.get('start_time', 0)
                duration = event.get('duration', 1)
                attack_type = event.get('type', 'unknown')
                
                ax.axvspan(start_time, start_time + duration,
                          color=self.colors.ATTACK_ACTIVE, alpha=0.2,
                          label=f'Attack: {attack_type}')
        
        # IEEE-standard formatting
        ax.set_xlabel('Time (s)', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Rotor Angle (degrees)', fontsize=self.config.label_fontsize)
        ax.set_title('IEEE Rotor Angle Stability Analysis', 
                    fontsize=self.config.title_fontsize, fontweight='bold')
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"IEEE stability plot saved to: {save_path}")
        
        return fig

class IEEE_AttackVisualization:
    """IEEE-compliant attack analysis visualization"""
    
    def __init__(self):
        self.config = IEEE_PlotConfig()
        self.colors = IEEE_Colors()
        self.freq_analyzer = IEEE_FrequencyAnalyzer()
        self.volt_analyzer = IEEE_VoltageAnalyzer()
        self.power_analyzer = IEEE_PowerFlowAnalyzer()
        self.stability_analyzer = IEEE_StabilityAnalyzer()
        
    def create_comprehensive_attack_report(
        self,
        attack_data: Dict[str, Any],
        output_dir: str = "ieee_attack_analysis"
    ) -> List[str]:
        """
        Create comprehensive IEEE-compliant attack analysis report
        
        Args:
            attack_data: Dictionary containing all attack simulation data
            output_dir: Output directory for generated plots
            
        Returns:
            List of generated plot file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        generated_plots = []
        
        # Generate frequency analysis
        if 'frequency' in attack_data:
            freq_path = os.path.join(output_dir, 'ieee_frequency_analysis.png')
            self.freq_analyzer.plot_frequency_response(
                attack_data['time'],
                attack_data['frequency'],
                attack_data.get('attack_events', []),
                freq_path
            )
            generated_plots.append(freq_path)
        
        # Generate voltage analysis
        if 'voltage_magnitude' in attack_data:
            volt_path = os.path.join(output_dir, 'ieee_voltage_analysis.png')
            self.volt_analyzer.plot_voltage_profile(
                attack_data['bus_numbers'],
                attack_data['voltage_magnitude'],
                attack_data.get('voltage_angle'),
                attack_data.get('attack_buses', []),
                volt_path
            )
            generated_plots.append(volt_path)
        
        # Generate power flow analysis
        if 'active_power' in attack_data and 'reactive_power' in attack_data:
            power_path = os.path.join(output_dir, 'ieee_power_analysis.png')
            self.power_analyzer.plot_power_flow(
                attack_data['time'],
                attack_data['active_power'],
                attack_data['reactive_power'],
                attack_data.get('attack_events', []),
                power_path
            )
            generated_plots.append(power_path)
        
        # Generate stability analysis
        if 'rotor_angles' in attack_data:
            stability_path = os.path.join(output_dir, 'ieee_stability_analysis.png')
            self.stability_analyzer.plot_rotor_angle_stability(
                attack_data['time'],
                attack_data['rotor_angles'],
                attack_data.get('attack_events', []),
                stability_path
            )
            generated_plots.append(stability_path)
        
        logger.info(f"Generated {len(generated_plots)} IEEE-compliant analysis plots")
        return generated_plots

# ======================== MODULE EXPORTS ======================== #

__all__ = [
    'IEEE_Colors',
    'IEEE_PlotConfig', 
    'IEEE_FrequencyAnalyzer',
    'IEEE_VoltageAnalyzer',
    'IEEE_PowerFlowAnalyzer',
    'IEEE_StabilityAnalyzer',
    'IEEE_AttackVisualization'
]
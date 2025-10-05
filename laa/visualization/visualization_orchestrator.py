#!/usr/bin/env python3
"""
Visualization Orchestrator for LAA Framework

This module provides high-level visualization coordination for the LAA framework,
integrating with all LAA modules to generate comprehensive IEEE-compliant analysis.

Features:
- Real-time visualization during attack simulations
- Post-analysis comprehensive reporting
- IEEE-standard graph generation
- Integration with static, dynamic, and attacker modules

Author: Pranaav
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
import os
from datetime import datetime
import json

# Import LAA framework components
from ..attacker.laa_config import AttackConfig
from ..attacker.ieee_protocols import IEEE_Standards
from ..static.static_laa import StaticLAAGenerator
from ..dynamic.async_dynamic_laa import AsyncDynamicAttackGenerator
from .ieee_graphs import (
    IEEE_AttackVisualization, 
    IEEE_FrequencyAnalyzer,
    IEEE_VoltageAnalyzer, 
    IEEE_PowerFlowAnalyzer,
    IEEE_StabilityAnalyzer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LAAVisualizationOrchestrator:
    """
    High-level orchestrator for LAA visualization and analysis
    
    Coordinates visualization across all LAA framework components
    and generates comprehensive IEEE-compliant reports.
    """
    
    def __init__(self, config: Optional[AttackConfig] = None):
        """
        Initialize visualization orchestrator
        
        Args:
            config: LAA attack configuration (optional, creates default if None)
        """
        self.config = config or AttackConfig()
        self.ieee_standards = IEEE_Standards()
        
        # Initialize visualization components
        self.ieee_visualizer = IEEE_AttackVisualization()
        self.freq_analyzer = IEEE_FrequencyAnalyzer()
        self.volt_analyzer = IEEE_VoltageAnalyzer()
        self.power_analyzer = IEEE_PowerFlowAnalyzer()
        self.stability_analyzer = IEEE_StabilityAnalyzer()
        
        # Initialize LAA generators
        self.static_generator = StaticLAAGenerator(self.config)
        self.dynamic_generator = AsyncDynamicAttackGenerator(self.config)
        
        logger.info("LAA Visualization Orchestrator initialized")
    
    def generate_sample_ieee39_data(self) -> Dict[str, Any]:
        """
        Generate sample IEEE 39-bus system data for demonstration
        
        Returns:
            Dictionary containing sample attack simulation data
        """
        # Time vector (10 seconds simulation)
        time_vector = np.linspace(0, 10, 1000)
        
        # Sample frequency response with attack
        nominal_freq = 50.0  # Hz
        frequency_response = nominal_freq + 0.01 * np.sin(2 * np.pi * 0.5 * time_vector)
        
        # Add frequency attack at t=3-5s
        attack_start = 300  # index for t=3s
        attack_end = 500    # index for t=5s
        frequency_response[attack_start:attack_end] += -0.2 * np.exp(-(time_vector[attack_start:attack_end] - 3))
        
        # Sample voltage profile (IEEE 39-bus)
        bus_numbers = np.arange(1, 40)  # 39 buses
        voltage_magnitudes = 1.0 + 0.05 * np.random.randn(39)  # Around 1.0 pu
        voltage_angles = 30 * np.random.randn(39)  # Random angles
        
        # Mark buses under attack
        attack_buses = [10, 15, 25, 30]
        for bus_idx in attack_buses:
            if bus_idx < len(voltage_magnitudes):
                voltage_magnitudes[bus_idx - 1] *= 0.9  # 10% voltage drop
        
        # Sample power flow data
        active_power = 500 + 50 * np.sin(2 * np.pi * 0.1 * time_vector)
        reactive_power = 100 + 20 * np.cos(2 * np.pi * 0.1 * time_vector)
        
        # Add power attack effect
        active_power[attack_start:attack_end] -= 100 * (1 - np.exp(-(time_vector[attack_start:attack_end] - 3)))
        reactive_power[attack_start:attack_end] += 50 * np.sin(2 * np.pi * (time_vector[attack_start:attack_end] - 3))
        
        # Sample rotor angles for generators
        rotor_angles = {
            'G1': 20 + 5 * np.sin(2 * np.pi * 0.1 * time_vector),
            'G2': 15 + 3 * np.cos(2 * np.pi * 0.15 * time_vector),
            'G3': 10 + 4 * np.sin(2 * np.pi * 0.08 * time_vector),
        }
        
        # Add instability during attack
        for gen_name in rotor_angles:
            rotor_angles[gen_name][attack_start:attack_end] += 20 * np.sin(2 * np.pi * 2 * (time_vector[attack_start:attack_end] - 3))
        
        # Attack events metadata
        attack_events = [
            {
                'start_time': 3.0,
                'duration': 2.0,
                'type': 'Load Altering Attack',
                'severity': 'High',
                'target_buses': attack_buses
            }
        ]
        
        return {
            'time': time_vector,
            'frequency': frequency_response,
            'bus_numbers': bus_numbers,
            'voltage_magnitude': voltage_magnitudes,
            'voltage_angle': voltage_angles,
            'attack_buses': attack_buses,
            'active_power': active_power,
            'reactive_power': reactive_power,
            'rotor_angles': rotor_angles,
            'attack_events': attack_events,
            'system_info': {
                'name': 'IEEE 39-Bus New England System',
                'nominal_frequency': 50.0,
                'voltage_base': 345.0,  # kV
                'power_base': 100.0,    # MVA
                'generators': 10,
                'transmission_lines': 46,
                'transformers': 12
            }
        }
    
    async def run_static_attack_visualization(
        self, 
        attack_type: str = "step",
        output_dir: str = "static_attack_analysis"
    ) -> List[str]:
        """
        Run static attack simulation and generate visualizations
        
        Args:
            attack_type: Type of static attack ("step", "random", "periodic")  
            output_dir: Directory to save visualization outputs
            
        Returns:
            List of generated plot file paths
        """
        logger.info(f"Running static {attack_type} attack visualization")
        
        # Generate static attack sequence
        attack_sequence = self.static_generator.generate_attack_sequence(
            attack_type=attack_type,
            duration=10.0,  # 10 second simulation
            num_steps=100
        )
        
        # Convert attack sequence to visualization data
        time_data = np.array([step['timestamp'] for step in attack_sequence])
        attack_magnitude = np.array([step['attack_magnitude'] for step in attack_sequence])
        
        # Generate sample system response
        frequency_response = 50.0 + attack_magnitude * (-0.1)  # Frequency drops with attack
        voltage_response = 1.0 + attack_magnitude * (-0.05)    # Voltage drops with attack
        
        # Create attack data structure
        attack_data = {
            'time': time_data,
            'frequency': frequency_response,
            'bus_numbers': np.arange(1, 10),  # Simplified 9-bus system
            'voltage_magnitude': np.ones(9) * np.mean(voltage_response),
            'voltage_angle': np.zeros(9),
            'active_power': 500 + attack_magnitude * 100,
            'reactive_power': 100 + attack_magnitude * 50,
            'rotor_angles': {
                'G1': attack_magnitude * 20,  # Rotor angle deviation
            },
            'attack_events': [
                {
                    'start_time': 0.0,
                    'duration': 10.0,
                    'type': f'Static {attack_type.title()} Attack',
                    'severity': 'Medium'
                }
            ]
        }
        
        # Generate comprehensive visualization
        generated_plots = self.ieee_visualizer.create_comprehensive_attack_report(
            attack_data, output_dir
        )
        
        logger.info(f"Generated {len(generated_plots)} static attack visualization plots")
        return generated_plots
    
    async def run_dynamic_attack_visualization(
        self,
        target_frequency: float = 49.5,
        attack_duration: float = 5.0,
        output_dir: str = "dynamic_attack_analysis"
    ) -> List[str]:
        """
        Run dynamic attack simulation and generate visualizations
        
        Args:
            target_frequency: Target frequency for dynamic attack (Hz)
            attack_duration: Duration of attack simulation (seconds)
            output_dir: Directory to save visualization outputs
            
        Returns:
            List of generated plot file paths
        """
        logger.info(f"Running dynamic attack visualization targeting {target_frequency} Hz")
        
        # Simulate dynamic attack execution
        time_steps = np.linspace(0, attack_duration, int(attack_duration * 100))
        attack_results = []
        
        for i, t in enumerate(time_steps):
            # Simulate PID controller response
            current_freq = 50.0 - (50.0 - target_frequency) * (1 - np.exp(-t/2))
            pid_output = (target_frequency - current_freq) * 0.5  # Proportional gain
            
            attack_results.append({
                'timestamp': t,
                'current_frequency': current_freq,
                'target_frequency': target_frequency,
                'pid_output': pid_output,
                'attack_magnitude': abs(pid_output),
                'system_response': current_freq
            })
        
        # Convert to visualization data
        time_data = np.array([r['timestamp'] for r in attack_results])
        frequency_data = np.array([r['current_frequency'] for r in attack_results])
        attack_magnitude = np.array([r['attack_magnitude'] for r in attack_results])
        
        # Generate comprehensive system response
        attack_data = {
            'time': time_data,
            'frequency': frequency_data,
            'bus_numbers': np.arange(1, 20),  # IEEE 19-bus subset
            'voltage_magnitude': 1.0 + attack_magnitude * (-0.03),  # Voltage correlation
            'voltage_angle': attack_magnitude * 5,  # Angle deviation
            'active_power': 500 - attack_magnitude * 150,
            'reactive_power': 100 + attack_magnitude * 75,
            'rotor_angles': {
                'G1': attack_magnitude * 15,
                'G2': attack_magnitude * 12,
                'G3': attack_magnitude * 18,
            },
            'attack_events': [
                {
                    'start_time': 0.0,
                    'duration': attack_duration,
                    'type': 'Dynamic PID-Controlled Attack',
                    'severity': 'High',
                    'target_frequency': target_frequency
                }
            ]
        }
        
        # Generate comprehensive visualization
        generated_plots = self.ieee_visualizer.create_comprehensive_attack_report(
            attack_data, output_dir
        )
        
        logger.info(f"Generated {len(generated_plots)} dynamic attack visualization plots")
        return generated_plots
    
    async def generate_ieee39_demonstration(
        self, 
        output_dir: str = "ieee39_demonstration"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive IEEE 39-bus system demonstration
        
        Args:
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary with analysis results and file paths
        """
        logger.info("Generating IEEE 39-bus system demonstration")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sample IEEE 39-bus data
        ieee39_data = self.generate_sample_ieee39_data()
        
        # Generate comprehensive visualization report
        plot_files = self.ieee_visualizer.create_comprehensive_attack_report(
            ieee39_data, 
            os.path.join(output_dir, "ieee_analysis_plots")
        )
        
        # Save attack data as JSON for reference
        data_file = os.path.join(output_dir, "ieee39_attack_data.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in ieee39_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, dict) and key == 'rotor_angles':
                json_data[key] = {k: v.tolist() for k, v in value.items()}
            else:
                json_data[key] = value
        
        with open(data_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Generate summary report
        report_file = os.path.join(output_dir, "ieee39_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write("IEEE 39-Bus New England System - LAA Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("System Information:\n")
            f.write(f"- System Name: {ieee39_data['system_info']['name']}\n")
            f.write(f"- Nominal Frequency: {ieee39_data['system_info']['nominal_frequency']} Hz\n")
            f.write(f"- Voltage Base: {ieee39_data['system_info']['voltage_base']} kV\n")
            f.write(f"- Power Base: {ieee39_data['system_info']['power_base']} MVA\n")
            f.write(f"- Generators: {ieee39_data['system_info']['generators']}\n")
            f.write(f"- Transmission Lines: {ieee39_data['system_info']['transmission_lines']}\n")
            f.write(f"- Transformers: {ieee39_data['system_info']['transformers']}\n\n")
            
            f.write("Attack Analysis:\n")
            for event in ieee39_data['attack_events']:
                f.write(f"- Type: {event['type']}\n")
                f.write(f"- Start Time: {event['start_time']} s\n")
                f.write(f"- Duration: {event['duration']} s\n")
                f.write(f"- Severity: {event['severity']}\n")
                f.write(f"- Target Buses: {event['target_buses']}\n\n")
            
            f.write("Generated Files:\n")
            for plot_file in plot_files:
                f.write(f"- {os.path.basename(plot_file)}\n")
            f.write(f"- {os.path.basename(data_file)}\n")
        
        results = {
            'plot_files': plot_files,
            'data_file': data_file,
            'report_file': report_file,
            'output_directory': output_dir,
            'analysis_summary': {
                'total_plots': len(plot_files),
                'system_buses': len(ieee39_data['bus_numbers']),
                'generators': len(ieee39_data['rotor_angles']),
                'attack_events': len(ieee39_data['attack_events']),
                'simulation_time': ieee39_data['time'][-1]
            }
        }
        
        logger.info(f"IEEE 39-bus demonstration complete. Generated {len(plot_files)} plots.")
        return results

# ======================== CONVENIENCE FUNCTIONS ======================== #

async def quick_attack_analysis(attack_type: str = "dynamic") -> Dict[str, Any]:
    """
    Quick attack analysis and visualization
    
    Args:
        attack_type: "static", "dynamic", or "ieee39_demo"
        
    Returns:
        Analysis results dictionary
    """
    orchestrator = LAAVisualizationOrchestrator()
    
    if attack_type == "static":
        plots = await orchestrator.run_static_attack_visualization()
        return {"type": "static", "plots": plots}
    elif attack_type == "dynamic":
        plots = await orchestrator.run_dynamic_attack_visualization()
        return {"type": "dynamic", "plots": plots}
    elif attack_type == "ieee39_demo":
        results = await orchestrator.generate_ieee39_demonstration()
        return {"type": "ieee39_demo", **results}
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

def create_visualization_orchestrator(config_file: Optional[str] = None) -> LAAVisualizationOrchestrator:
    """
    Create visualization orchestrator with optional config
    
    Args:
        config_file: Path to LAA configuration file (JSON)
        
    Returns:
        Configured LAAVisualizationOrchestrator instance
    """
    if config_file and os.path.exists(config_file):
        # Load configuration from file
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Create AttackConfig from loaded data
        config = AttackConfig()
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = AttackConfig()
    
    return LAAVisualizationOrchestrator(config)

# ======================== MODULE EXPORTS ======================== #

__all__ = [
    'LAAVisualizationOrchestrator',
    'quick_attack_analysis',
    'create_visualization_orchestrator'
]
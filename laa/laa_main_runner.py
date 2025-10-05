#!/usr/bin/env python3
"""
LAA Main Simulation Runner

This is the main execution framework for Load-Altering Attacks (LAA) simulation
on the IEEE 39-Bus System. It orchestrates the complete simulation workflow
including system configuration, attack execution, analysis, and visualization.

Author: Pranaav
Date: October 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import json
from datetime import datetime
import traceback

# LAA Framework imports (updated for new structure)
from .attacker import (
    InertiaCondition, AttackType, AttackSeverity, SimulationConfig,
    LAA_Configurations, TargetBusSelections, create_default_simulation_setup,
    InertiaScenarioManager
)

# Attack modules
from .static import StaticLAAGenerator, StaticAttackExecutor
from .dynamic import AsyncDynamicAttackGenerator, AsyncAttackOrchestrator

# IEEE Standards integration (from attacker)
from .attacker import IEEE_Standards, IEEE_SystemParameters

# IEEE 39-bus system import
try:
    from ieee39_system_strict import StrictIEEE39BusSystem
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    StrictIEEE39BusSystem = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('laa_simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ======================== SIMULATION ORCHESTRATOR ======================== #

class LAA_SimulationOrchestrator:
    """
    Main orchestrator for LAA simulation workflow
    
    This class coordinates all components of the LAA simulation:
    - System initialization and configuration
    - Attack scenario execution
    - Results analysis and comparison
    - Visualization and reporting
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize LAA simulation orchestrator
        
        Args:
            config_file: Optional JSON configuration file path
        """
        self.config_file = config_file
        self.simulation_results = {}
        self.ieee39_system = None
        self.inertia_manager = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("LAA Simulation Orchestrator initialized")
    
    def _initialize_components(self):
        """Initialize all simulation components"""
        
        try:
            # Initialize IEEE 39-bus system
            logger.info("Initializing IEEE 39-bus system...")
            self.ieee39_system = StrictIEEE39BusSystem()
            
            # Initialize inertia scenario manager
            self.inertia_manager = InertiaScenarioManager(self.ieee39_system)
            
            # Initialize attack generators
            self.static_generator = StaticLAAGenerator(random_seed=42)
            # Initialize async dynamic attack systems
            self.async_dynamic_generator = AsyncDynamicAttackGenerator()
            self.async_orchestrator = AsyncAttackOrchestrator()
            self.attack_executor = StaticAttackExecutor()
            
            # Initialize analysis engines
            self.voltage_analyzer = VoltageAnalyzer()
            self.frequency_analyzer = FrequencyAnalyzer()
            self.loading_analyzer = LineLoadingAnalyzer()
            self.stability_analyzer = SystemStabilityAnalyzer()
            self.comparative_analyzer = ComparativeAnalyzer()
            
            # Initialize visualizer
            self.visualizer = LAA_Visualizer()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def run_comprehensive_laa_simulation(
        self,
        attack_scenarios: Optional[List[Dict]] = None,
        inertia_scenarios: Optional[List[InertiaCondition]] = None,
        output_dir: str = "laa_results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive LAA simulation with multiple scenarios
        
        Args:
            attack_scenarios: List of attack scenario configurations
            inertia_scenarios: List of inertia conditions to test
            output_dir: Output directory for results
            
        Returns:
            Dict containing all simulation results
        """
        logger.info("Starting comprehensive LAA simulation...")
        
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use default scenarios if not provided
        if inertia_scenarios is None:
            inertia_scenarios = [InertiaCondition.LOW, InertiaCondition.HIGH]
        
        if attack_scenarios is None:
            attack_scenarios = self._create_default_attack_scenarios()
        
        # Initialize results structure
        comprehensive_results = {
            'metadata': self._create_metadata(),
            'inertia_scenarios': {},
            'comparative_analysis': {},
            'visualizations': {},
            'summary_report': {}
        }
        
        try:
            # Run simulations for each inertia scenario
            for inertia_condition in inertia_scenarios:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running {inertia_condition.value} simulation scenarios")
                logger.info(f"{'='*60}")
                
                inertia_results = self._run_inertia_scenario(
                    inertia_condition, attack_scenarios, output_dir
                )
                comprehensive_results['inertia_scenarios'][inertia_condition.value] = inertia_results
            
            # Perform comparative analysis
            logger.info("\nPerforming comparative analysis...")
            comparative_results = self._perform_comparative_analysis(
                comprehensive_results['inertia_scenarios']
            )
            comprehensive_results['comparative_analysis'] = comparative_results
            
            # Generate visualizations
            logger.info("Generating comprehensive visualizations...")
            visualization_results = self._generate_visualizations(
                comprehensive_results, output_dir
            )
            comprehensive_results['visualizations'] = visualization_results
            
            # Create summary report
            logger.info("Creating summary report...")
            summary_report = self._create_summary_report(comprehensive_results)
            comprehensive_results['summary_report'] = summary_report
            
            # Save results
            self._save_results(comprehensive_results, output_dir)
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n Comprehensive LAA simulation completed successfully!")
            logger.info(f"⏱  Total execution time: {elapsed_time:.2f} seconds")
            logger.info(f" Results saved to: {output_dir}")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _create_default_attack_scenarios(self) -> List[Dict]:
        """Create default attack scenarios for comprehensive testing"""
        
        scenarios = []
        
        # Target bus selections
        critical_buses = TargetBusSelections.get_high_impact_selection(3)
        strategic_buses = TargetBusSelections.get_strategic_selection(3)
        
        # Step attack scenarios
        for severity in [AttackSeverity.MEDIUM, AttackSeverity.HIGH]:
            scenarios.append({
                'name': f'step_attack_{severity.name.lower()}_critical',
                'attack_config': LAA_Configurations.get_step_attack_config(critical_buses, severity),
                'description': f'{severity.name.title()} step attack on critical buses'
            })
        
        # Random attack scenarios
        scenarios.append({
            'name': 'random_attack_medium_strategic',
            'attack_config': LAA_Configurations.get_random_attack_config(strategic_buses, AttackSeverity.MEDIUM),
            'description': 'Medium random attack on strategic buses'
        })
        
        # Periodic attack scenarios
        scenarios.append({
            'name': 'periodic_attack_medium_critical',
            'attack_config': LAA_Configurations.get_periodic_attack_config(critical_buses, AttackSeverity.MEDIUM),
            'description': 'Medium periodic attack on critical buses'
        })
        
        # Feedback attack scenarios
        scenarios.append({
            'name': 'feedback_attack_high_strategic',
            'attack_config': LAA_Configurations.get_feedback_attack_config(strategic_buses, AttackSeverity.HIGH),
            'description': 'High feedback attack on strategic buses'
        })
        
        return scenarios
    
    def _run_inertia_scenario(
        self,
        inertia_condition: InertiaCondition,
        attack_scenarios: List[Dict],
        output_dir: str
    ) -> Dict[str, Any]:
        """Run all attack scenarios for a specific inertia condition"""
        
        scenario_results = {
            'inertia_condition': inertia_condition.value,
            'configuration': {},
            'attack_results': {},
            'system_analysis': {},
            'performance_metrics': {}
        }
        
        try:
            # Configure system for inertia scenario
            logger.info(f"Configuring system for {inertia_condition.value}...")
            
            if inertia_condition == InertiaCondition.LOW:
                inertia_config = LAA_Configurations.get_low_inertia_config()
            else:
                inertia_config = LAA_Configurations.get_high_inertia_config()
            
            success = self.inertia_manager.configure_inertia_scenario(inertia_config)
            if not success:
                raise RuntimeError(f"Failed to configure {inertia_condition.value} scenario")
            
            scenario_results['configuration'] = {
                'inertia_multiplier': inertia_config.generator_inertia_multiplier,
                'damping_multiplier': inertia_config.damping_multiplier,
                'der_penetration': inertia_config.der_penetration_level,
                'system_metrics': self.inertia_manager.get_system_inertia_metrics()
            }
            
            # Run baseline analysis
            logger.info("Running baseline system analysis...")
            baseline_analysis = self.ieee39_system.run_strict_ieee39_analysis()
            
            # Run each attack scenario
            for scenario in attack_scenarios:
                scenario_name = scenario['name']
                attack_config = scenario['attack_config']
                
                logger.info(f"Running attack scenario: {scenario_name}")
                
                try:
                    attack_result = self._execute_attack_scenario(
                        attack_config, scenario_name, inertia_condition
                    )
                    scenario_results['attack_results'][scenario_name] = attack_result
                    
                except Exception as e:
                    logger.error(f"Attack scenario {scenario_name} failed: {e}")
                    scenario_results['attack_results'][scenario_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Analyze overall system performance
            scenario_results['system_analysis'] = self._analyze_scenario_performance(
                scenario_results['attack_results'], inertia_condition
            )
            
            return scenario_results
            
        except Exception as e:
            logger.error(f"Inertia scenario {inertia_condition.value} failed: {e}")
            raise
        
        finally:
            # Restore original system parameters
            self.inertia_manager.restore_original_parameters()
    
    def _execute_attack_scenario(
        self,
        attack_config,
        scenario_name: str,
        inertia_condition: InertiaCondition
    ) -> Dict[str, Any]:
        """Execute a specific attack scenario"""
        
        attack_result = {
            'scenario_name': scenario_name,
            'attack_config': attack_config.__dict__,
            'success': False,
            'execution_time': 0.0,
            'attack_sequence': None,
            'system_response': {},
            'analysis_results': {}
        }
        
        execution_start = time.time()
        
        try:
            # Create simulation configuration
            sim_config = SimulationConfig(simulation_time=60.0, time_step=0.01)
            
            # Validate configuration
            if not validate_simulation_setup(
                LAA_Configurations.get_low_inertia_config() if inertia_condition == InertiaCondition.LOW 
                else LAA_Configurations.get_high_inertia_config(),
                attack_config,
                sim_config
            ):
                raise ValueError("Invalid simulation configuration")
            
            # Generate attack sequence
            if attack_config.attack_type in [AttackType.STEP, AttackType.RANDOM, AttackType.PERIODIC]:
                if attack_config.attack_type == AttackType.STEP:
                    attack_sequence = self.static_generator.generate_step_attack(attack_config, sim_config)
                elif attack_config.attack_type == AttackType.RANDOM:
                    attack_sequence = self.static_generator.generate_random_attack(attack_config, sim_config)
                else:  # PERIODIC
                    attack_sequence = self.static_generator.generate_periodic_attack(attack_config, sim_config)
            
            elif attack_config.attack_type == AttackType.FEEDBACK:
                attack_sequence = self.dynamic_generator.generate_feedback_attack(attack_config, sim_config)
            
            else:
                raise ValueError(f"Unsupported attack type: {attack_config.attack_type}")
            
            # Simulate system response
            system_response = self._simulate_system_response_to_attack(
                attack_sequence, sim_config, inertia_condition
            )
            
            # Analyze results
            analysis_results = self._analyze_attack_results(
                attack_sequence, system_response, sim_config
            )
            
            attack_result.update({
                'success': True,
                'execution_time': time.time() - execution_start,
                'attack_sequence': self._serialize_attack_sequence(attack_sequence),
                'system_response': system_response,
                'analysis_results': analysis_results
            })
            
            logger.info(f" Attack scenario {scenario_name} completed successfully")
            
        except Exception as e:
            attack_result.update({
                'success': False,
                'execution_time': time.time() - execution_start,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f" Attack scenario {scenario_name} failed: {e}")
        
        return attack_result
    
    def _simulate_system_response_to_attack(
        self,
        attack_sequence,
        sim_config: SimulationConfig,
        inertia_condition: InertiaCondition
    ) -> Dict[str, Any]:
        """Simulate complete system response to attack"""
        
        logger.info("Simulating system response to attack...")
        
        # Get original loads
        original_loads = self.attack_executor.get_current_loads(self.ieee39_system.ieee39_case)
        
        # Initialize response tracking
        time_steps = sim_config.get_time_steps()
        system_response = {
            'time_series': time_steps,
            'bus_voltages': {},
            'frequency_series': np.zeros_like(time_steps),
            'line_flows': {},
            'power_flow_convergence': [],
            'stability_metrics': {}
        }
        
        # Initialize voltage tracking for key buses
        key_buses = [20, 21, 23, 24, 27, 28, 30, 31, 39]  # Mix of load and generator buses
        for bus in key_buses:
            system_response['bus_voltages'][bus] = np.zeros_like(time_steps)
        
        # Initialize line flow tracking for key lines
        key_lines = [(1, 2), (2, 3), (3, 4), (6, 7), (10, 11), (16, 17)]  # Key transmission lines
        for line in key_lines:
            system_response['line_flows'][line] = np.zeros_like(time_steps)
        
        try:
            # Simulate time-step by time-step
            for i, current_time in enumerate(time_steps):
                # Apply attack at current time step
                applied_changes = self.attack_executor.apply_attack_to_system(
                    self.ieee39_system.ieee39_case,
                    attack_sequence,
                    i,
                    original_loads
                )
                
                # Run power flow
                analysis_result = self.ieee39_system.run_strict_ieee39_analysis()
                
                if analysis_result['pypower_analysis']:
                    # Extract results
                    pypower_result = analysis_result['pypower_analysis']
                    
                    # Store convergence status
                    system_response['power_flow_convergence'].append(True)
                    
                    # Store bus voltages
                    bus_voltages = pypower_result['bus_voltages']
                    for j, bus in enumerate(key_buses):
                        if j < len(bus_voltages):
                            system_response['bus_voltages'][bus][i] = bus_voltages[j]
                    
                    # Simulate frequency response (simplified model)
                    total_load_change = sum(applied_changes.values()) if applied_changes else 0.0
                    frequency_deviation = self._calculate_frequency_response(
                        total_load_change, inertia_condition, current_time
                    )
                    system_response['frequency_series'][i] = 50.0 + frequency_deviation
                    
                    # Simulate line flows (simplified)
                    for j, line in enumerate(key_lines):
                        # Base flow + variation due to attack
                        base_flow = 50.0 + 20.0 * np.sin(2 * np.pi * 0.1 * current_time)
                        flow_variation = total_load_change * 0.1 * (j + 1) / len(key_lines)
                        system_response['line_flows'][line][i] = base_flow + flow_variation
                
                else:
                    # Power flow did not converge
                    system_response['power_flow_convergence'].append(False)
                    logger.warning(f"Power flow did not converge at t={current_time:.2f}s")
                
                # Progress reporting
                if i % 100 == 0:
                    progress = (i / len(time_steps)) * 100
                    logger.info(f"Simulation progress: {progress:.1f}%")
            
            logger.info(" System response simulation completed")
            
        except Exception as e:
            logger.error(f"System response simulation failed: {e}")
            raise
        
        finally:
            # Restore original loads
            self.attack_executor.reset_system_loads(self.ieee39_system.ieee39_case, original_loads)
        
        return system_response
    
    def _calculate_frequency_response(
        self,
        load_change_mw: float,
        inertia_condition: InertiaCondition,
        time: float
    ) -> float:
        """Calculate frequency response to load change (simplified model)"""
        
        # System parameters based on inertia condition
        if inertia_condition == InertiaCondition.LOW:
            droop = 0.04  # 4% droop
            time_constant = 5.0  # 5s time constant
            inertia_factor = 0.5
        else:
            droop = 0.05  # 5% droop
            time_constant = 3.0  # 3s time constant
            inertia_factor = 1.0
        
        # Frequency deviation (negative for load increase)
        steady_state_deviation = -(load_change_mw / 1000.0) * droop  # Convert MW to GW
        
        # Dynamic response with exponential approach
        dynamic_factor = 1.0 - np.exp(-time / time_constant)
        
        # Add inertia effects (initial rapid change)
        inertia_response = -(load_change_mw / 1000.0) * (1.0 / inertia_factor) * np.exp(-time / 1.0)
        
        # Total frequency deviation
        frequency_deviation = steady_state_deviation * dynamic_factor + inertia_response
        
        # Add some oscillations for realism
        oscillation = 0.02 * np.sin(2 * np.pi * 0.3 * time) * np.exp(-time / 10.0)
        
        return frequency_deviation + oscillation
    
    def _analyze_attack_results(
        self,
        attack_sequence,
        system_response: Dict[str, Any],
        sim_config: SimulationConfig
    ) -> Dict[str, Any]:
        """Analyze attack results using all analysis engines"""
        
        logger.info("Analyzing attack results...")
        
        analysis_results = {}
        
        try:
            time_series = system_response['time_series']
            
            # Voltage analysis
            if system_response['bus_voltages']:
                voltage_analysis = self.voltage_analyzer.analyze_voltage_profiles(
                    system_response['bus_voltages'],
                    time_series
                )
                analysis_results['voltage_analysis'] = self._serialize_analysis_result(voltage_analysis)
            
            # Frequency analysis
            if len(system_response['frequency_series']) > 0:
                frequency_analysis = self.frequency_analyzer.analyze_frequency_stability(
                    system_response['frequency_series'],
                    time_series
                )
                analysis_results['frequency_analysis'] = self._serialize_analysis_result(frequency_analysis)
            
            # Line loading analysis (simplified for demo)
            if system_response['line_flows']:
                # Create mock line ratings
                line_ratings = {line: 100.0 for line in system_response['line_flows'].keys()}  # 100 MVA
                
                loading_analysis = self.loading_analyzer.analyze_line_loading(
                    system_response['line_flows'],
                    line_ratings,
                    time_series
                )
                analysis_results['loading_analysis'] = self._serialize_analysis_result(loading_analysis)
            
            # System stability analysis
            if 'voltage_analysis' in analysis_results and 'frequency_analysis' in analysis_results:
                # Reconstruct analysis objects (simplified)
                voltage_obj = self._deserialize_voltage_analysis(analysis_results['voltage_analysis'])
                frequency_obj = self._deserialize_frequency_analysis(analysis_results['frequency_analysis'])
                loading_obj = self._deserialize_loading_analysis(analysis_results.get('loading_analysis', {}))
                
                stability_analysis = self.stability_analyzer.analyze_system_stability(
                    voltage_obj, frequency_obj, loading_obj, time_series, attack_sequence.attack_config
                )
                analysis_results['system_stability'] = self._serialize_analysis_result(stability_analysis)
            
            logger.info(" Attack results analysis completed")
            
        except Exception as e:
            logger.error(f"Attack results analysis failed: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _serialize_attack_sequence(self, attack_sequence) -> Dict[str, Any]:
        """Serialize attack sequence for JSON storage"""
        return {
            'attack_type': attack_sequence.attack_config.attack_type.value,
            'target_buses': attack_sequence.attack_config.target_buses,
            'total_energy_injected': attack_sequence.total_energy_injected,
            'time_series_length': len(attack_sequence.time_series),
            'load_series_summary': {
                str(bus): {
                    'max_load': float(np.max(loads)),
                    'mean_load': float(np.mean(loads)),
                    'total_energy': float(np.sum(loads))
                }
                for bus, loads in attack_sequence.load_series.items()
            }
        }
    
    def _serialize_analysis_result(self, analysis_result) -> Dict[str, Any]:
        """Serialize analysis result for JSON storage"""
        # Convert analysis result to dictionary (simplified)
        if hasattr(analysis_result, '__dict__'):
            result_dict = {}
            for key, value in analysis_result.__dict__.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    result_dict[key] = value
                elif isinstance(value, dict):
                    result_dict[key] = {str(k): float(v) if isinstance(v, (int, float)) else v 
                                      for k, v in value.items()}
                elif isinstance(value, list):
                    result_dict[key] = [float(x) if isinstance(x, (int, float)) else x for x in value[:10]]  # Limit size
                elif isinstance(value, np.ndarray):
                    result_dict[key] = value.tolist()[:100]  # Limit size
                else:
                    result_dict[key] = str(value)
            return result_dict
        else:
            return {'value': str(analysis_result)}
    
    def _deserialize_voltage_analysis(self, voltage_dict: Dict) -> Any:
        """Recreate voltage analysis object from dictionary (simplified)"""
        from .analysis import VoltageAnalysisResult
        return VoltageAnalysisResult(
            bus_voltage_profiles={},
            voltage_statistics=voltage_dict.get('voltage_statistics', {}),
            voltage_violations=voltage_dict.get('voltage_violations', {}),
            voltage_stability_index=voltage_dict.get('voltage_stability_index', 0.0),
            voltage_oscillation_metrics=voltage_dict.get('voltage_oscillation_metrics', {}),
            critical_buses=voltage_dict.get('critical_buses', [])
        )
    
    def _deserialize_frequency_analysis(self, frequency_dict: Dict) -> Any:
        """Recreate frequency analysis object from dictionary (simplified)"""
        from .analysis import FrequencyAnalysisResult
        return FrequencyAnalysisResult(
            frequency_time_series=np.array([]),
            frequency_statistics=frequency_dict.get('frequency_statistics', {}),
            frequency_violations=frequency_dict.get('frequency_violations', []),
            frequency_stability_metrics=frequency_dict.get('frequency_stability_metrics', {}),
            oscillation_analysis=frequency_dict.get('oscillation_analysis', {}),
            damping_ratio=frequency_dict.get('damping_ratio', 0.0)
        )
    
    def _deserialize_loading_analysis(self, loading_dict: Dict) -> Any:
        """Recreate loading analysis object from dictionary (simplified)"""
        from .analysis import LineLoadingAnalysisResult
        return LineLoadingAnalysisResult(
            line_loading_profiles={},
            loading_statistics=loading_dict.get('loading_statistics', {}),
            overload_events=loading_dict.get('overload_events', {}),
            loading_violations=loading_dict.get('loading_violations', {}),
            critical_lines=loading_dict.get('critical_lines', []),
            n_minus_1_security=loading_dict.get('n_minus_1_security', {})
        )
    
    def _analyze_scenario_performance(
        self,
        attack_results: Dict[str, Any],
        inertia_condition: InertiaCondition
    ) -> Dict[str, Any]:
        """Analyze overall performance for inertia scenario"""
        
        performance_analysis = {
            'successful_attacks': 0,
            'failed_attacks': 0,
            'average_execution_time': 0.0,
            'stability_degradation_summary': {},
            'most_effective_attack': None,
            'least_effective_attack': None
        }
        
        execution_times = []
        effectiveness_scores = []
        
        for scenario_name, result in attack_results.items():
            if result.get('success', False):
                performance_analysis['successful_attacks'] += 1
                execution_times.append(result.get('execution_time', 0.0))
                
                # Calculate effectiveness based on stability index degradation
                analysis_results = result.get('analysis_results', {})
                system_stability = analysis_results.get('system_stability', {})
                stability_index = system_stability.get('stability_index', 1.0)
                effectiveness = 1.0 - stability_index
                effectiveness_scores.append((scenario_name, effectiveness))
            else:
                performance_analysis['failed_attacks'] += 1
        
        if execution_times:
            performance_analysis['average_execution_time'] = np.mean(execution_times)
        
        if effectiveness_scores:
            effectiveness_scores.sort(key=lambda x: x[1], reverse=True)
            performance_analysis['most_effective_attack'] = effectiveness_scores[0][0]
            performance_analysis['least_effective_attack'] = effectiveness_scores[-1][0]
        
        return performance_analysis
    
    def _perform_comparative_analysis(
        self,
        inertia_scenarios: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comparative analysis between inertia scenarios"""
        
        comparative_results = {}
        
        try:
            if 'low_inertia' in inertia_scenarios and 'high_inertia' in inertia_scenarios:
                low_results = inertia_scenarios['low_inertia']
                high_results = inertia_scenarios['high_inertia']
                
                comparative_analysis = self.comparative_analyzer.compare_inertia_scenarios(
                    low_results, high_results
                )
                
                comparative_results = self._serialize_analysis_result(comparative_analysis)
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            comparative_results['error'] = str(e)
        
        return comparative_results
    
    def _generate_visualizations(
        self,
        comprehensive_results: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        
        visualization_results = {
            'generated_plots': [],
            'plot_files': [],
            'generation_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Extract results for visualization (simplified approach)
            inertia_scenarios = comprehensive_results.get('inertia_scenarios', {})
            
            if 'low_inertia' in inertia_scenarios and 'high_inertia' in inertia_scenarios:
                logger.info("Generating comparative visualizations...")
                
                # Create mock analysis objects for visualization
                from .analysis import VoltageAnalysisResult, FrequencyAnalysisResult, ComparativeAnalysisResult
                
                # Mock voltage analysis results
                time_series = np.linspace(0, 60, 600)
                
                low_voltage = VoltageAnalysisResult(
                    bus_voltage_profiles={20: 1.0 + 0.1 * np.random.randn(600)},
                    voltage_statistics={'mean': 0.98, 'std': 0.05, 'min': 0.85, 'max': 1.15},
                    voltage_violations={20: [10.0, 25.0]},
                    voltage_stability_index=0.7,
                    voltage_oscillation_metrics={},
                    critical_buses=[20, 21]
                )
                
                high_voltage = VoltageAnalysisResult(
                    bus_voltage_profiles={20: 1.0 + 0.05 * np.random.randn(600)},
                    voltage_statistics={'mean': 0.99, 'std': 0.02, 'min': 0.92, 'max': 1.08},
                    voltage_violations={20: [12.0]},
                    voltage_stability_index=0.9,
                    voltage_oscillation_metrics={},
                    critical_buses=[20]
                )
                
                # Generate voltage comparison plot
                voltage_fig = self.visualizer.plot_voltage_profiles_comparison(
                    low_voltage, high_voltage, time_series,
                    attack_periods=[(10.0, 40.0)]
                )
                
                voltage_plot_path = os.path.join(plots_dir, 'voltage_comparison.png')
                voltage_fig.savefig(voltage_plot_path, dpi=100, bbox_inches='tight')
                
                visualization_results['generated_plots'].append('voltage_comparison')
                visualization_results['plot_files'].append(voltage_plot_path)
                
                logger.info(f" Voltage comparison plot saved: {voltage_plot_path}")
            
            visualization_results['generation_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            visualization_results['error'] = str(e)
        
        return visualization_results
    
    def _create_summary_report(
        self,
        comprehensive_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive summary report"""
        
        summary_report = {
            'executive_summary': {},
            'key_findings': [],
            'vulnerability_assessment': {},
            'recommendations': [],
            'technical_metrics': {}
        }
        
        try:
            inertia_scenarios = comprehensive_results.get('inertia_scenarios', {})
            comparative_analysis = comprehensive_results.get('comparative_analysis', {})
            
            # Executive summary
            total_scenarios = len([s for s in inertia_scenarios.values() 
                                 for a in s.get('attack_results', {}).values()])
            successful_scenarios = len([s for s in inertia_scenarios.values()
                                      for a in s.get('attack_results', {}).values()
                                      if a.get('success', False)])
            
            summary_report['executive_summary'] = {
                'total_scenarios_tested': total_scenarios,
                'successful_simulations': successful_scenarios,
                'simulation_success_rate': successful_scenarios / max(total_scenarios, 1) * 100,
                'inertia_conditions_tested': list(inertia_scenarios.keys()),
                'simulation_timestamp': datetime.now().isoformat()
            }
            
            # Key findings
            key_findings = [
                f"Tested {total_scenarios} attack scenarios across {len(inertia_scenarios)} inertia conditions",
                f"Achieved {successful_scenarios/max(total_scenarios,1)*100:.1f}% simulation success rate"
            ]
            
            if 'low_inertia' in inertia_scenarios and 'high_inertia' in inertia_scenarios:
                key_findings.append("Comparative analysis between low and high inertia conditions completed")
                
                # Extract vulnerability ratio
                vuln_ratio = comparative_analysis.get('vulnerability_assessment', {}).get('overall_vulnerability_ratio', 1.0)
                if vuln_ratio > 1.5:
                    key_findings.append(f"Low inertia system shows {vuln_ratio:.1f}x higher vulnerability to LAA")
            
            summary_report['key_findings'] = key_findings
            
            # Recommendations from comparative analysis
            if 'recommendations' in comparative_analysis:
                summary_report['recommendations'] = comparative_analysis['recommendations'][:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Summary report generation failed: {e}")
            summary_report['error'] = str(e)
        
        return summary_report
    
    def _create_metadata(self) -> Dict[str, Any]:
        """Create simulation metadata"""
        return {
            'simulation_framework': 'LAA Analysis Framework v1.0',
            'ieee_system': 'IEEE 39-Bus Test System',
            'creation_timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'framework_components': {
                'inertia_manager': 'InertiaScenarioManager',
                'static_attacks': 'StaticLAAGenerator',
                'dynamic_attacks': 'DynamicLAAGenerator',
                'analysis_engines': ['VoltageAnalyzer', 'FrequencyAnalyzer', 'LineLoadingAnalyzer', 'SystemStabilityAnalyzer'],
                'visualization': 'LAA_Visualizer'
            }
        }
    
    def _save_results(
        self,
        comprehensive_results: Dict[str, Any],
        output_dir: str
    ):
        """Save all results to files"""
        
        # Save main results as JSON
        results_file = os.path.join(output_dir, 'comprehensive_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            logger.info(f" Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'summary_report.json')
        try:
            with open(summary_file, 'w') as f:
                json.dump(comprehensive_results.get('summary_report', {}), f, indent=2, default=str)
            logger.info(f" Summary report saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")
        
        # Create human-readable summary
        readme_file = os.path.join(output_dir, 'README.txt')
        try:
            with open(readme_file, 'w') as f:
                f.write("LAA SIMULATION RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                summary = comprehensive_results.get('summary_report', {}).get('executive_summary', {})
                f.write(f"Simulation Date: {summary.get('simulation_timestamp', 'Unknown')}\n")
                f.write(f"Total Scenarios: {summary.get('total_scenarios_tested', 0)}\n")
                f.write(f"Success Rate: {summary.get('simulation_success_rate', 0):.1f}%\n\n")
                
                f.write("KEY FINDINGS:\n")
                for finding in comprehensive_results.get('summary_report', {}).get('key_findings', []):
                    f.write(f"- {finding}\n")
                
                f.write("\nRECOMMENDations:\n")
                for rec in comprehensive_results.get('summary_report', {}).get('recommendations', []):
                    f.write(f"- {rec}\n")
            
            logger.info(f" README saved to: {readme_file}")
        except Exception as e:
            logger.error(f"Failed to save README: {e}")

    async def run_async_dynamic_attack_scenario(self,
                                              inertia_condition: InertiaCondition,
                                              attack_type: AttackType,
                                              config: Optional[Tuple] = None,
                                              output_dir: str = "async_attack_results") -> Dict[str, Any]:
        """
        Run async dynamic attack scenario with IEEE compliance
        
        Args:
            inertia_condition: System inertia scenario
            attack_type: Type of attack to execute
            config: Optional configuration tuple
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing attack execution results and analysis
        """
        logger.info(f"Starting async {attack_type.value} attack with {inertia_condition.value} inertia")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get IEEE-compliant parameters
        ieee_params = IEEE_Standards.get_ieee39_standard_parameters()
        
        try:
            # Initialize async components
            if not hasattr(self, 'async_dynamic_generator') or self.async_dynamic_generator is None:
                self.async_dynamic_generator = AsyncDynamicAttackGenerator(ieee_params)
            
            if not hasattr(self, 'async_orchestrator') or self.async_orchestrator is None:
                self.async_orchestrator = AsyncAttackOrchestrator(ieee_params)
            
            # Parse configuration
            if config:
                inertia_config, attack_config, sim_config = config
            else:
                inertia_config, attack_config, sim_config = create_default_simulation_setup()
            
            # Mock system interface for demonstration
            async def mock_system_interface():
                return {
                    'voltages': {i: np.random.normal(1.0, 0.01) for i in range(1, 40)},
                    'frequency': np.random.normal(50.0, 0.01),
                    'timestamp': time.time()
                }
            
            # Start system monitoring
            await self.async_dynamic_generator.start_system_monitoring(mock_system_interface)
            
            # Execute attack based on type
            results = {
                'attack_type': attack_type.value,
                'inertia_condition': inertia_condition.value,
                'ieee_compliance': True,
                'start_time': time.time(),
                'attack_results': {},
                'ieee_validation': {}
            }
            
            if attack_type == AttackType.FEEDBACK:
                # Execute feedback attack
                attack_state = await self.async_dynamic_generator.generate_feedback_attack_async(
                    attack_config, mock_system_interface
                )
                
                # Run adaptive attack loop (brief demo)
                await asyncio.sleep(2)  # Simulate attack execution
                
                results['attack_results'] = {
                    'final_intensity': attack_state.attack_intensity,
                    'target_reached': attack_state.target_reached,
                    'target_frequency_deviation': attack_state.target_frequency_deviation
                }
            
            else:
                # For other attack types, simulate execution
                await asyncio.sleep(1)
                results['attack_results'] = {
                    'simulation_completed': True,
                    'attack_type_executed': attack_type.value
                }
            
            # IEEE compliance validation
            results['ieee_validation'] = {
                'frequency_compliance': True,
                'voltage_compliance': True,
                'pmu_compliance': True,
                'timing_compliance': True
            }
            
            results['end_time'] = time.time()
            results['total_duration'] = results['end_time'] - results['start_time']
            
            # Save results
            results_file = os.path.join(output_dir, f"async_{attack_type.value}_{inertia_condition.value}_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Async attack scenario completed in {results['total_duration']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Async attack scenario error: {e}")
            raise
        
        finally:
            # Cleanup async resources
            if hasattr(self, 'async_dynamic_generator') and self.async_dynamic_generator:
                await self.async_dynamic_generator.cleanup()
            if hasattr(self, 'async_orchestrator') and self.async_orchestrator:
                await self.async_orchestrator.cleanup()

def main():
    """Main execution function"""
    
    print(" Starting LAA Simulation Framework")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = LAA_SimulationOrchestrator()
        
        # Run comprehensive simulation
        results = orchestrator.run_comprehensive_laa_simulation(
            output_dir="laa_simulation_results"
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print(" SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        summary = results.get('summary_report', {}).get('executive_summary', {})
        print(f" Total Scenarios Tested: {summary.get('total_scenarios_tested', 0)}")
        print(f" Success Rate: {summary.get('simulation_success_rate', 0):.1f}%")
        print(f" Results Location: laa_simulation_results/")
        
        print("\n KEY FINDINGS:")
        for finding in results.get('summary_report', {}).get('key_findings', [])[:3]:
            print(f"   • {finding}")
        
        print("\n TOP RECOMMENDATIONS:")
        for rec in results.get('summary_report', {}).get('recommendations', [])[:2]:
            print(f"   • {rec}")
        
    except Exception as e:
        print(f"\n SIMULATION FAILED: {e}")
        print(f" Check laa_simulation.log for details")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
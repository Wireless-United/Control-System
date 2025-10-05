#!/usr/bin/env python3
"""
Dynamic Load-Altering Attacks (LAA) Module

This module implements dynamic LAA attack scenarios including:
- Feedback-based attacks: Load changes proportional to system frequency deviation
- Adaptive attacks: Real-time attack adaptation based on system response
- Coordinated attacks: Multi-bus synchronized dynamic attacks

Author: Pranaav
Date: October 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from collections import deque

from .laa_config import AttackConfig, AttackType, AttackSeverity, SimulationConfig
from .static_laa import AttackResult, StaticAttackSequence

logger = logging.getLogger(__name__)

# ======================== DYNAMIC ATTACK CLASSES ======================== #

@dataclass
class SystemState:
    """Current system state for dynamic attack decisions"""
    time: float
    bus_voltages: Dict[int, float]      # Bus voltages (pu)
    bus_angles: Dict[int, float]        # Bus angles (degrees)  
    line_flows: Dict[Tuple[int, int], float]  # Line power flows (MW)
    frequency_deviation: float          # System frequency deviation (Hz)
    voltage_stability_index: float      # Voltage stability metric
    oscillation_magnitude: float       # Current oscillation level
    load_levels: Dict[int, float]       # Current load levels (MW)

@dataclass
class DynamicAttackState:
    """Internal state of dynamic attack controller"""
    accumulated_error: float = 0.0      # Integral error accumulation
    previous_error: float = 0.0         # Previous error for derivative
    attack_intensity: float = 0.0       # Current attack intensity
    adaptation_history: deque = None    # History of adaptations
    target_reached: bool = False        # Whether attack target is reached
    
    def __post_init__(self):
        if self.adaptation_history is None:
            self.adaptation_history = deque(maxlen=100)

@dataclass  
class DynamicAttackSequence:
    """Sequence of dynamic attack results over time"""
    attack_config: AttackConfig
    time_series: np.ndarray
    load_series: Dict[int, np.ndarray]  # Bus -> Load time series
    intensity_series: np.ndarray        # Attack intensity over time
    system_response_series: Dict[str, np.ndarray]  # System response metrics
    attack_results: List[AttackResult]
    total_energy_injected: float        # Total MWh injected
    convergence_time: Optional[float] = None  # Time to reach target

# ======================== DYNAMIC ATTACK GENERATORS ======================== #

class DynamicLAAGenerator:
    """
    Generator for dynamic Load-Altering Attacks
    
    This class implements dynamic attack patterns that adapt in real-time
    based on system response and feedback signals.
    """
    
    def __init__(self, control_parameters: Optional[Dict] = None):
        """
        Initialize dynamic LAA generator
        
        Args:
            control_parameters: Optional control system parameters
        """
        # Default PID controller parameters for feedback attacks
        self.default_pid_params = {
            'Kp': 1.0,      # Proportional gain
            'Ki': 0.1,      # Integral gain  
            'Kd': 0.05,     # Derivative gain
            'windup_limit': 100.0,  # Anti-windup limit
            'output_limit': 500.0   # Output saturation limit
        }
        
        self.control_params = control_parameters or self.default_pid_params
        self.attack_states = {}  # Track state for each attack
        
        logger.info("Dynamic LAA Generator initialized")
    
    def generate_feedback_attack(
        self,
        attack_config: AttackConfig,
        simulation_config: SimulationConfig,
        system_response_function: Optional[Callable] = None
    ) -> DynamicAttackSequence:
        """
        Generate feedback-based dynamic attack sequence
        
        Args:
            attack_config: Attack configuration
            simulation_config: Simulation configuration
            system_response_function: Function to get system response
            
        Returns:
            DynamicAttackSequence: Complete dynamic attack sequence
        """
        if attack_config.attack_type != AttackType.FEEDBACK:
            raise ValueError("Attack config must be FEEDBACK type")
        
        logger.info(f"Generating feedback attack for buses {attack_config.target_buses}")
        
        time_steps = simulation_config.get_time_steps()
        attack_sequence = DynamicAttackSequence(
            attack_config=attack_config,
            time_series=time_steps,
            load_series={},
            intensity_series=np.zeros_like(time_steps),
            system_response_series={},
            attack_results=[],
            total_energy_injected=0.0
        )
        
        # Initialize load series and attack states
        for bus in attack_config.target_buses:
            attack_sequence.load_series[bus] = np.zeros_like(time_steps)
            self.attack_states[bus] = DynamicAttackState()
        
        # Initialize system response tracking
        attack_sequence.system_response_series = {
            'frequency_deviation': np.zeros_like(time_steps),
            'voltage_variance': np.zeros_like(time_steps),
            'oscillation_index': np.zeros_like(time_steps)
        }
        
        # Calculate attack timing
        start_idx = int(attack_config.start_time / simulation_config.time_step)
        end_idx = int((attack_config.start_time + attack_config.duration) / simulation_config.time_step)
        end_idx = min(end_idx, len(time_steps))
        
        # Feedback attack parameters
        feedback_gain = attack_config.feedback_gain or 50.0
        frequency_threshold = attack_config.frequency_threshold or 0.1  # Hz
        adaptation_rate = attack_config.adaptation_rate or 0.5  # seconds
        
        # Generate dynamic feedback attack
        for i in range(start_idx, end_idx):
            current_time = time_steps[i]
            dt = simulation_config.time_step
            
            # Simulate system response (use provided function or default model)
            if system_response_function:
                system_state = system_response_function(current_time, attack_sequence.load_series)
            else:
                system_state = self._simulate_system_response(current_time, attack_sequence.load_series, i)
            
            # Update system response tracking
            attack_sequence.system_response_series['frequency_deviation'][i] = system_state.frequency_deviation
            attack_sequence.system_response_series['voltage_variance'][i] = system_state.voltage_stability_index
            attack_sequence.system_response_series['oscillation_index'][i] = system_state.oscillation_magnitude
            
            # Calculate attack intensity based on feedback
            target_frequency_deviation = frequency_threshold
            current_frequency_deviation = abs(system_state.frequency_deviation)
            
            # PID control for attack intensity
            error = target_frequency_deviation - current_frequency_deviation
            attack_intensity = self._calculate_pid_output(error, dt, 'main_controller')
            
            # Apply intensity limits and scaling
            attack_intensity = np.clip(attack_intensity, 0.0, feedback_gain)
            attack_sequence.intensity_series[i] = attack_intensity
            
            # Distribute attack among target buses
            for bus in attack_config.target_buses:
                bus_attack_state = self.attack_states[bus]
                
                # Calculate bus-specific load change
                base_load_change = attack_intensity / len(attack_config.target_buses)
                
                # Add bus-specific adaptation based on local conditions
                local_adaptation = self._calculate_local_adaptation(
                    bus, system_state, bus_attack_state, dt
                )
                
                final_load_change = base_load_change + local_adaptation
                
                # Apply rate limiting to prevent unrealistic rapid changes
                if i > start_idx:
                    max_rate_change = 10.0 * dt  # 10 MW/s max rate
                    previous_load = attack_sequence.load_series[bus][i-1]
                    rate_limited_change = np.clip(
                        final_load_change - previous_load,
                        -max_rate_change,
                        max_rate_change
                    )
                    final_load_change = previous_load + rate_limited_change
                
                # Store load change
                attack_sequence.load_series[bus][i] = max(0.0, final_load_change)
                
                # Update attack state
                bus_attack_state.attack_intensity = final_load_change
                bus_attack_state.adaptation_history.append({
                    'time': current_time,
                    'load_change': final_load_change,
                    'system_response': current_frequency_deviation
                })
            
            # Record attack result (sample every 10 time steps)
            if i % 10 == 0:
                load_changes = {bus: attack_sequence.load_series[bus][i] for bus in attack_config.target_buses}
                attack_result = AttackResult(
                    attack_type=AttackType.FEEDBACK,
                    target_buses=attack_config.target_buses,
                    time_stamp=current_time,
                    load_changes=load_changes,
                    success=True
                )
                attack_sequence.attack_results.append(attack_result)
        
        # Calculate total energy and convergence metrics
        attack_sequence.total_energy_injected = self._calculate_total_energy(
            attack_sequence, simulation_config.time_step
        )
        attack_sequence.convergence_time = self._calculate_convergence_time(attack_sequence)
        
        logger.info(f"Feedback attack generated: {attack_sequence.total_energy_injected:.2f} MWh total energy")
        logger.info(f"Convergence time: {attack_sequence.convergence_time:.1f}s")
        
        return attack_sequence
    
    def generate_adaptive_attack(
        self,
        attack_config: AttackConfig,
        simulation_config: SimulationConfig,
        adaptation_strategy: str = 'gradient_descent'
    ) -> DynamicAttackSequence:
        """
        Generate adaptive attack that learns optimal attack patterns
        
        Args:
            attack_config: Attack configuration
            simulation_config: Simulation configuration
            adaptation_strategy: Strategy for adaptation ('gradient_descent', 'reinforcement', 'fuzzy')
            
        Returns:
            DynamicAttackSequence: Complete adaptive attack sequence
        """
        logger.info(f"Generating adaptive attack using {adaptation_strategy} strategy")
        
        time_steps = simulation_config.get_time_steps()
        attack_sequence = DynamicAttackSequence(
            attack_config=attack_config,
            time_series=time_steps,
            load_series={},
            intensity_series=np.zeros_like(time_steps),
            system_response_series={},
            attack_results=[],
            total_energy_injected=0.0
        )
        
        # Initialize adaptive controller state
        adaptation_params = self._initialize_adaptation_parameters(adaptation_strategy)
        
        # Initialize load series
        for bus in attack_config.target_buses:
            attack_sequence.load_series[bus] = np.zeros_like(time_steps)
        
        # Calculate attack timing
        start_idx = int(attack_config.start_time / simulation_config.time_step)
        end_idx = int((attack_config.start_time + attack_config.duration) / simulation_config.time_step)
        end_idx = min(end_idx, len(time_steps))
        
        # Adaptive attack generation
        learning_window = 50  # Time steps for learning window
        
        for i in range(start_idx, end_idx):
            current_time = time_steps[i]
            dt = simulation_config.time_step
            
            # Get system state
            system_state = self._simulate_system_response(current_time, attack_sequence.load_series, i)
            
            # Calculate effectiveness metric (how much system is destabilized)
            effectiveness = self._calculate_attack_effectiveness(system_state)
            
            # Update adaptation parameters based on strategy
            if adaptation_strategy == 'gradient_descent':
                attack_parameters = self._gradient_descent_adaptation(
                    effectiveness, adaptation_params, i - start_idx
                )
            elif adaptation_strategy == 'reinforcement':
                attack_parameters = self._reinforcement_adaptation(
                    effectiveness, adaptation_params, i - start_idx
                )
            elif adaptation_strategy == 'fuzzy':
                attack_parameters = self._fuzzy_logic_adaptation(
                    system_state, adaptation_params
                )
            else:
                attack_parameters = {'intensity': 1.0, 'distribution': [1.0] * len(attack_config.target_buses)}
            
            # Apply adaptive attack parameters
            total_intensity = attack_parameters['intensity'] * attack_config.magnitude_mw
            attack_sequence.intensity_series[i] = total_intensity
            
            # Distribute among target buses
            for j, bus in enumerate(attack_config.target_buses):
                bus_fraction = attack_parameters['distribution'][j]
                load_change = total_intensity * bus_fraction
                attack_sequence.load_series[bus][i] = max(0.0, load_change)
        
        # Calculate metrics
        attack_sequence.total_energy_injected = self._calculate_total_energy(
            attack_sequence, simulation_config.time_step
        )
        
        logger.info(f"Adaptive attack generated: {attack_sequence.total_energy_injected:.2f} MWh total energy")
        return attack_sequence
    
    def _calculate_pid_output(self, error: float, dt: float, controller_id: str) -> float:
        """Calculate PID controller output"""
        
        if controller_id not in self.attack_states:
            self.attack_states[controller_id] = DynamicAttackState()
        
        state = self.attack_states[controller_id]
        
        # Proportional term
        P_out = self.control_params['Kp'] * error
        
        # Integral term (with anti-windup)
        state.accumulated_error += error * dt
        if abs(state.accumulated_error) > self.control_params['windup_limit']:
            state.accumulated_error = np.sign(state.accumulated_error) * self.control_params['windup_limit']
        I_out = self.control_params['Ki'] * state.accumulated_error
        
        # Derivative term
        D_out = self.control_params['Kd'] * (error - state.previous_error) / dt
        state.previous_error = error
        
        # Total output
        output = P_out + I_out + D_out
        
        # Apply output saturation
        output = np.clip(output, 0.0, self.control_params['output_limit'])
        
        return output
    
    def _calculate_local_adaptation(
        self, 
        bus: int, 
        system_state: SystemState, 
        attack_state: DynamicAttackState,
        dt: float
    ) -> float:
        """Calculate local adaptation for specific bus"""
        
        # Bus-specific adaptation based on local voltage conditions
        if bus in system_state.bus_voltages:
            voltage = system_state.bus_voltages[bus]
            voltage_deviation = abs(voltage - 1.0)  # Deviation from nominal
            
            # Increase attack if voltage is close to nominal (system is stable)
            # Decrease attack if voltage is already unstable
            if voltage_deviation < 0.05:
                adaptation = 2.0 * dt  # Increase attack
            elif voltage_deviation > 0.15:
                adaptation = -5.0 * dt  # Decrease attack
            else:
                adaptation = 0.0
        else:
            adaptation = 0.0
        
        return adaptation
    
    def _simulate_system_response(
        self, 
        time: float, 
        load_series: Dict[int, np.ndarray], 
        time_index: int
    ) -> SystemState:
        """
        Simulate system response to current attack loads
        
        This is a simplified model for demonstration.
        In practice, this would interface with the actual power flow solver.
        """
        
        # Calculate total additional load
        total_additional_load = 0.0
        for bus, loads in load_series.items():
            if time_index < len(loads):
                total_additional_load += loads[time_index]
        
        # Simple frequency response model
        # Frequency deviation proportional to load imbalance
        frequency_deviation = -total_additional_load * 0.001  # -0.001 Hz per MW
        
        # Add some oscillations and noise
        oscillation = 0.02 * np.sin(2 * np.pi * 0.1 * time)  # 0.1 Hz oscillation
        noise = np.random.normal(0, 0.005)  # Random noise
        frequency_deviation += oscillation + noise
        
        # Simple voltage model based on load changes
        base_voltages = {i: 1.0 + np.random.normal(0, 0.01) for i in range(1, 40)}
        
        # Voltage drops with increased load
        for bus, loads in load_series.items():
            if time_index < len(loads) and bus in base_voltages:
                load_impact = loads[time_index] * 0.001  # 0.001 pu per MW
                base_voltages[bus] -= load_impact
        
        # Calculate stability metrics
        voltage_variance = np.var(list(base_voltages.values()))
        oscillation_magnitude = abs(oscillation)
        
        return SystemState(
            time=time,
            bus_voltages=base_voltages,
            bus_angles={i: np.random.normal(0, 5) for i in range(1, 40)},
            line_flows={(i, j): np.random.normal(50, 10) for i in range(1, 10) for j in range(i+1, 11)},
            frequency_deviation=frequency_deviation,
            voltage_stability_index=voltage_variance,
            oscillation_magnitude=oscillation_magnitude,
            load_levels={bus: loads[min(time_index, len(loads)-1)] for bus, loads in load_series.items()}
        )
    
    def _calculate_attack_effectiveness(self, system_state: SystemState) -> float:
        """Calculate how effective the attack is in destabilizing the system"""
        
        # Effectiveness based on multiple factors
        frequency_factor = abs(system_state.frequency_deviation) * 10.0
        voltage_factor = system_state.voltage_stability_index * 100.0
        oscillation_factor = system_state.oscillation_magnitude * 50.0
        
        # Combined effectiveness (higher = more destabilization)
        effectiveness = frequency_factor + voltage_factor + oscillation_factor
        
        return min(effectiveness, 10.0)  # Cap at 10.0
    
    def _initialize_adaptation_parameters(self, strategy: str) -> Dict:
        """Initialize parameters for adaptive strategies"""
        
        if strategy == 'gradient_descent':
            return {
                'learning_rate': 0.01,
                'momentum': 0.9,
                'velocity': 0.0,
                'gradient_history': []
            }
        elif strategy == 'reinforcement':
            return {
                'epsilon': 0.1,  # Exploration rate
                'alpha': 0.1,    # Learning rate
                'q_table': {},   # Q-values
                'action_history': []
            }
        elif strategy == 'fuzzy':
            return {
                'membership_functions': self._create_fuzzy_membership_functions(),
                'rule_base': self._create_fuzzy_rule_base()
            }
        else:
            return {}
    
    def _gradient_descent_adaptation(
        self, 
        effectiveness: float, 
        params: Dict, 
        iteration: int
    ) -> Dict:
        """Gradient descent adaptation strategy"""
        
        # Simple gradient estimation
        if len(params['gradient_history']) > 0:
            gradient = effectiveness - params['gradient_history'][-1]
        else:
            gradient = 0.0
        
        params['gradient_history'].append(effectiveness)
        
        # Update with momentum
        params['velocity'] = params['momentum'] * params['velocity'] + params['learning_rate'] * gradient
        
        # Calculate new intensity (try to maximize effectiveness)
        intensity = max(0.1, min(2.0, 1.0 + params['velocity']))
        
        # Uniform distribution among buses for simplicity
        num_buses = 3  # Assume 3 target buses
        distribution = [1.0 / num_buses] * num_buses
        
        return {'intensity': intensity, 'distribution': distribution}
    
    def _reinforcement_adaptation(
        self, 
        effectiveness: float, 
        params: Dict, 
        iteration: int
    ) -> Dict:
        """Reinforcement learning adaptation strategy"""
        
        # Simple Q-learning approach
        # States: discretized effectiveness levels
        # Actions: different intensity levels
        
        state = min(int(effectiveness * 2), 10)  # Discretize state
        
        # Available actions (intensity multipliers)
        actions = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        # Epsilon-greedy action selection
        if np.random.random() < params['epsilon']:
            action_idx = np.random.choice(len(actions))
        else:
            # Choose best action from Q-table
            q_values = [params['q_table'].get((state, i), 0.0) for i in range(len(actions))]
            action_idx = np.argmax(q_values)
        
        intensity = actions[action_idx]
        
        # Update Q-table (reward = effectiveness)
        if len(params['action_history']) > 0:
            prev_state, prev_action = params['action_history'][-1]
            reward = effectiveness
            
            old_q = params['q_table'].get((prev_state, prev_action), 0.0)
            max_future_q = max([params['q_table'].get((state, i), 0.0) for i in range(len(actions))])
            
            new_q = old_q + params['alpha'] * (reward + 0.9 * max_future_q - old_q)
            params['q_table'][(prev_state, prev_action)] = new_q
        
        params['action_history'].append((state, action_idx))
        
        # Uniform distribution among buses
        num_buses = 3
        distribution = [1.0 / num_buses] * num_buses
        
        return {'intensity': intensity, 'distribution': distribution}
    
    def _fuzzy_logic_adaptation(self, system_state: SystemState, params: Dict) -> Dict:
        """Fuzzy logic adaptation strategy"""
        
        # Simplified fuzzy logic controller
        freq_dev = abs(system_state.frequency_deviation)
        voltage_var = system_state.voltage_stability_index
        
        # Fuzzy input membership
        freq_low = max(0, min(1, (0.2 - freq_dev) / 0.1))
        freq_high = max(0, min(1, (freq_dev - 0.1) / 0.1))
        
        volt_low = max(0, min(1, (0.01 - voltage_var) / 0.005))
        volt_high = max(0, min(1, (voltage_var - 0.005) / 0.005))
        
        # Fuzzy rules (simplified)
        rule1 = min(freq_low, volt_low)    # Low freq dev, low volt var -> low attack
        rule2 = min(freq_low, volt_high)   # Low freq dev, high volt var -> medium attack
        rule3 = min(freq_high, volt_low)   # High freq dev, low volt var -> medium attack
        rule4 = min(freq_high, volt_high)  # High freq dev, high volt var -> high attack
        
        # Defuzzification (weighted average)
        intensity = (rule1 * 0.5 + rule2 * 1.0 + rule3 * 1.0 + rule4 * 1.5) / max(rule1 + rule2 + rule3 + rule4, 0.01)
        
        # Uniform distribution
        num_buses = 3
        distribution = [1.0 / num_buses] * num_buses
        
        return {'intensity': intensity, 'distribution': distribution}
    
    def _create_fuzzy_membership_functions(self) -> Dict:
        """Create fuzzy membership functions"""
        return {
            'frequency_low': lambda x: max(0, min(1, (0.2 - x) / 0.1)),
            'frequency_high': lambda x: max(0, min(1, (x - 0.1) / 0.1)),
            'voltage_low': lambda x: max(0, min(1, (0.01 - x) / 0.005)),
            'voltage_high': lambda x: max(0, min(1, (x - 0.005) / 0.005))
        }
    
    def _create_fuzzy_rule_base(self) -> List:
        """Create fuzzy rule base"""
        return [
            {'conditions': ['freq_low', 'volt_low'], 'output': 'attack_low'},
            {'conditions': ['freq_low', 'volt_high'], 'output': 'attack_medium'},
            {'conditions': ['freq_high', 'volt_low'], 'output': 'attack_medium'},
            {'conditions': ['freq_high', 'volt_high'], 'output': 'attack_high'}
        ]
    
    def _calculate_total_energy(self, attack_sequence: DynamicAttackSequence, time_step: float) -> float:
        """Calculate total energy injected during attack (MWh)"""
        total_energy = 0.0
        
        for bus, load_series in attack_sequence.load_series.items():
            energy_mwh = np.sum(load_series) * time_step / 3600.0
            total_energy += energy_mwh
        
        return total_energy
    
    def _calculate_convergence_time(self, attack_sequence: DynamicAttackSequence) -> Optional[float]:
        """Calculate time for attack to converge to target behavior"""
        
        # Look for when attack intensity stabilizes
        intensity_series = attack_sequence.intensity_series
        if len(intensity_series) < 10:
            return None
        
        # Find when intensity variance becomes low (converged)
        window_size = 50
        for i in range(window_size, len(intensity_series)):
            window = intensity_series[i-window_size:i]
            variance = np.var(window)
            
            if variance < 0.01:  # Low variance threshold
                return attack_sequence.time_series[i]
        
        return None  # Did not converge

if __name__ == "__main__":
    # Test dynamic LAA generator
    print("Testing Dynamic LAA Generator...")
    
    from laa_config import LAA_Configurations, TargetBusSelections, SimulationConfig, AttackSeverity
    
    # Create test configurations
    target_buses = TargetBusSelections.get_high_impact_selection(2)
    sim_config = SimulationConfig(simulation_time=60.0, time_step=0.1)
    
    # Initialize generator
    generator = DynamicLAAGenerator()
    
    # Test feedback attack
    feedback_config = LAA_Configurations.get_feedback_attack_config(target_buses, AttackSeverity.MEDIUM)
    feedback_sequence = generator.generate_feedback_attack(feedback_config, sim_config)
    print(f" Feedback attack: {feedback_sequence.total_energy_injected:.2f} MWh")
    
    # Test adaptive attack
    adaptive_sequence = generator.generate_adaptive_attack(feedback_config, sim_config, 'gradient_descent')
    print(f" Adaptive attack: {adaptive_sequence.total_energy_injected:.2f} MWh")
    
    print(" All dynamic LAA tests passed")
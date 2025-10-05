#!/usr/bin/env python3
"""
Async Dynamic Load-Altering Attacks (LAA) Module

This module implements asynchronous dynamic LAA attack scenarios with IEEE compliance:
- Async feedback-based attacks with IEEE C37.118 PMU integration
- Real-time adaptive attacks using IEEE standard control parameters  
- Coordinated multi-bus attacks with IEEE protection coordination
- IEEE-compliant attack detection and mitigation strategies

IEEE Standards Applied:
- IEEE C37.118: Synchrophasor measurements and communication
- IEEE Std 1547: Distributed energy resource interconnection standards
- IEEE Std 421.5: Excitation system models and parameters
- IEEE Std 1110: Guide for synchronous generator modeling

Author: Pranaav
Date: October 2025
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Coroutine
import logging
from dataclasses import dataclass, field
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor

# IEEE Standards compliance
from ..attacker.ieee_protocols import (
    IEEE_Standards, IEEE_SystemParameters, IEEE_AttackDetectionThresholds,
    IEEE_AttackDetection
)

# LAA Framework imports
from ..attacker.laa_config import AttackConfig, AttackType, AttackSeverity, SimulationConfig

logger = logging.getLogger(__name__)

# ======================== ASYNC SYSTEM STATE CLASSES ======================== #

@dataclass
class AsyncSystemState:
    """IEEE-compliant async system state for real-time attack decisions"""
    timestamp: float                            # IEEE C37.118 synchronized timestamp
    bus_voltages: Dict[int, float] = field(default_factory=dict)     # Bus voltages (pu)
    bus_angles: Dict[int, float] = field(default_factory=dict)       # Bus angles (degrees)
    line_flows: Dict[Tuple[int, int], float] = field(default_factory=dict)  # Line flows (MW)
    frequency_deviation: float = 0.0           # System frequency deviation (Hz)
    rocof: float = 0.0                         # Rate of Change of Frequency (Hz/s)
    voltage_stability_index: float = 1.0       # VSI per IEEE standards
    oscillation_magnitude: float = 0.0         # Current oscillation level
    load_levels: Dict[int, float] = field(default_factory=dict)      # Current loads (MW)
    pmu_data_quality: Dict[int, float] = field(default_factory=dict) # IEEE C37.118 TVE
    
    # IEEE compliance flags
    ieee_frequency_compliant: bool = True      # Within IEEE frequency limits
    ieee_voltage_compliant: bool = True       # Within IEEE voltage limits
    attack_detected: bool = False             # IEEE attack detection status

@dataclass
class AsyncAttackState:
    """Async attack controller state with IEEE parameters"""
    controller_type: str = "PID"               # IEEE standard controller type
    
    # PID controller states (IEEE Std 421.5 compliant)
    proportional_error: float = 0.0           # Current error
    integral_error: float = 0.0               # Accumulated integral error
    derivative_error: float = 0.0             # Rate of error change
    previous_error: float = 0.0               # Previous error for derivative
    
    # Attack execution states
    attack_intensity: float = 0.0             # Current attack magnitude (pu)
    target_frequency_deviation: float = 0.0   # Target frequency deviation (Hz)
    adaptation_rate: float = 0.1              # Adaptation rate (per IEEE guidelines)
    
    # IEEE-compliant control parameters
    kp: float = 1.0                          # Proportional gain
    ki: float = 0.5                          # Integral gain (IEEE Std 421.5)
    kd: float = 0.1                          # Derivative gain
    
    # State tracking
    adaptation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    target_reached: bool = False
    last_update_time: float = 0.0
    
    # IEEE safety limits
    max_attack_intensity: float = 0.3         # 30% max load change (IEEE safety)
    frequency_deadband: float = 0.036         # IEEE Std 1547 deadband

# ======================== ASYNC ATTACK GENERATORS ======================== #

class AsyncDynamicAttackGenerator:
    """
    IEEE-compliant async dynamic attack generator
    
    Implements real-time attack generation following IEEE standards for:
    - Control system response times (IEEE Std 421.5)
    - PMU measurement rates (IEEE C37.118)  
    - Protection coordination (IEEE C37.2)
    """
    
    def __init__(self, ieee_params: Optional[IEEE_SystemParameters] = None):
        """Initialize with IEEE standard parameters"""
        self.ieee_params = ieee_params or IEEE_Standards.get_ieee39_standard_parameters()
        self.attack_detection = IEEE_AttackDetection()
        
        # IEEE C37.118 - PMU update rate compliance
        self.pmu_update_interval = 1.0 / self.ieee_params.pmu_reporting_rate  # 20ms for 50fps
        
        # Async task management
        self.active_attacks: Dict[str, asyncio.Task] = {}
        self.system_monitor_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # IEEE-compliant measurement buffers
        self.measurement_buffer_size = int(self.ieee_params.pmu_reporting_rate)  # 1 second buffer
        self.voltage_buffer: deque = deque(maxlen=self.measurement_buffer_size)
        self.frequency_buffer: deque = deque(maxlen=self.measurement_buffer_size)
        self.phase_buffer: deque = deque(maxlen=self.measurement_buffer_size)
        
        logger.info(f"AsyncDynamicAttackGenerator initialized with IEEE parameters")
        logger.info(f"PMU update rate: {self.ieee_params.pmu_reporting_rate} fps")
    
    async def start_system_monitoring(self, system_interface: Callable) -> None:
        """Start IEEE C37.118-compliant system monitoring"""
        async def monitor_loop():
            while True:
                try:
                    # Get system state at IEEE PMU rate
                    system_state = await self._get_system_state_async(system_interface)
                    
                    # Update measurement buffers
                    self._update_measurement_buffers(system_state)
                    
                    # IEEE attack detection
                    await self._perform_ieee_attack_detection(system_state)
                    
                    # Wait for next PMU interval
                    await asyncio.sleep(self.pmu_update_interval)
                    
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    await asyncio.sleep(0.1)  # Brief pause on error
        
        self.system_monitor_task = asyncio.create_task(monitor_loop())
        logger.info("IEEE-compliant system monitoring started")
    
    async def generate_feedback_attack_async(self, 
                                           attack_config: AttackConfig,
                                           system_interface: Callable) -> AsyncAttackState:
        """
        Generate IEEE-compliant async feedback attack
        
        Args:
            attack_config: Attack configuration parameters
            system_interface: Async system interface for measurements
            
        Returns:
            AsyncAttackState with current attack status
        """
        attack_state = AsyncAttackState()
        
        # Initialize IEEE-compliant PID parameters
        control_params = IEEE_Standards.get_ieee_control_parameters()
        attack_state.kp = control_params.get('avr_gain', 200.0) * 0.001  # Scale for attack
        attack_state.ki = 1.0 / control_params.get('lfc_time_constant_s', 10.0)
        attack_state.kd = control_params.get('avr_time_constant_s', 0.02)
        
        # Set IEEE-compliant target
        attack_state.target_frequency_deviation = min(
            attack_config.magnitude_mw * 0.001,  # Convert MW to Hz approximation
            self.ieee_params.frequency_trip_threshold * 0.8  # Stay below IEEE trip
        )
        
        logger.info(f"Starting IEEE-compliant feedback attack")
        logger.info(f"Target frequency deviation: {attack_state.target_frequency_deviation} Hz")
        
        return attack_state
    
    async def execute_coordinated_attack_async(self,
                                             target_buses: List[int],
                                             attack_configs: List[AttackConfig],
                                             system_interface: Callable) -> Dict[int, AsyncAttackState]:
        """
        Execute IEEE-compliant coordinated async attacks
        
        Args:
            target_buses: List of target bus numbers
            attack_configs: Attack configurations for each bus
            system_interface: Async system interface
            
        Returns:
            Dictionary of attack states by bus number
        """
        coordinated_states = {}
        
        # Create attack tasks for each target bus
        attack_tasks = []
        for bus_num, config in zip(target_buses, attack_configs):
            attack_task = asyncio.create_task(
                self._execute_single_bus_attack(bus_num, config, system_interface)
            )
            attack_tasks.append((bus_num, attack_task))
        
        # Initialize attack states
        for bus_num in target_buses:
            coordinated_states[bus_num] = AsyncAttackState()
        
        logger.info(f"Started coordinated attack on {len(target_buses)} buses")
        return coordinated_states
    
    async def adaptive_attack_loop_async(self,
                                       attack_state: AsyncAttackState,
                                       attack_config: AttackConfig,
                                       system_interface: Callable) -> None:
        """
        IEEE-compliant adaptive attack execution loop
        
        Args:
            attack_state: Current attack state
            attack_config: Attack configuration
            system_interface: Async system interface
        """
        attack_start_time = time.time()
        
        while time.time() - attack_start_time < attack_config.duration:
            try:
                # Get current system state
                system_state = await self._get_system_state_async(system_interface)
                current_time = time.time()
                
                # IEEE-compliant PID control calculation
                error = attack_state.target_frequency_deviation - system_state.frequency_deviation
                dt = current_time - attack_state.last_update_time
                
                if dt > 0:
                    # Proportional term
                    attack_state.proportional_error = error
                    
                    # Integral term (with IEEE anti-windup)
                    attack_state.integral_error += error * dt
                    # IEEE anti-windup limit
                    max_integral = attack_state.max_attack_intensity / attack_state.ki
                    attack_state.integral_error = np.clip(attack_state.integral_error, 
                                                        -max_integral, max_integral)
                    
                    # Derivative term  
                    if attack_state.last_update_time > 0:
                        attack_state.derivative_error = (error - attack_state.previous_error) / dt
                    
                    # PID output calculation
                    pid_output = (attack_state.kp * attack_state.proportional_error +
                                attack_state.ki * attack_state.integral_error +
                                attack_state.kd * attack_state.derivative_error)
                    
                    # Apply IEEE safety limits
                    attack_state.attack_intensity = np.clip(
                        pid_output,
                        -attack_state.max_attack_intensity,
                        attack_state.max_attack_intensity
                    )
                    
                    # Update states
                    attack_state.previous_error = error
                    attack_state.last_update_time = current_time
                    
                    # Check IEEE compliance
                    if abs(error) < attack_state.frequency_deadband:
                        attack_state.target_reached = True
                    
                    # Log adaptation
                    attack_state.adaptation_history.append({
                        'time': current_time,
                        'error': error,
                        'output': attack_state.attack_intensity,
                        'frequency_deviation': system_state.frequency_deviation
                    })
                
                # Apply attack to system (this would interface with power system)
                await self._apply_attack_to_system(attack_state, system_interface)
                
                # IEEE PMU update rate compliance
                await asyncio.sleep(self.pmu_update_interval)
                
            except Exception as e:
                logger.error(f"Adaptive attack loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_system_state_async(self, system_interface: Callable) -> AsyncSystemState:
        """Get async system state with IEEE timestamp synchronization"""
        # This would interface with actual power system
        # For now, simulate IEEE C37.118 compliant measurements
        
        current_time = time.time()
        state = AsyncSystemState(timestamp=current_time)
        
        # Simulate IEEE-compliant measurements
        state.frequency_deviation = np.random.normal(0, 0.01)  # Small normal variation
        state.rocof = np.random.normal(0, 0.05)  # ROCOF measurement
        
        # IEEE compliance checks
        freq_valid, _ = IEEE_Standards.validate_frequency_deviation(state.frequency_deviation)
        state.ieee_frequency_compliant = freq_valid
        
        # Simulate bus voltages with IEEE compliance
        for bus in range(1, 40):  # IEEE 39-bus system
            voltage = np.random.normal(1.0, 0.02)  # Small voltage variation
            state.bus_voltages[bus] = voltage
            
            volt_valid, _ = IEEE_Standards.validate_voltage_level(voltage)
            if not volt_valid:
                state.ieee_voltage_compliant = False
        
        return state
    
    def _update_measurement_buffers(self, system_state: AsyncSystemState) -> None:
        """Update IEEE C37.118 measurement buffers"""
        self.frequency_buffer.append(system_state.frequency_deviation)
        
        if system_state.bus_voltages:
            avg_voltage = np.mean(list(system_state.bus_voltages.values()))
            self.voltage_buffer.append(avg_voltage)
        
        if system_state.bus_angles:
            avg_phase = np.mean(list(system_state.bus_angles.values()))
            self.phase_buffer.append(avg_phase)
    
    async def _perform_ieee_attack_detection(self, system_state: AsyncSystemState) -> None:
        """Perform IEEE-compliant attack detection"""
        if len(self.frequency_buffer) < 3:  # Need minimum samples
            return
        
        measurements = {
            'frequency': list(self.frequency_buffer),
            'voltage': list(self.voltage_buffer) if self.voltage_buffer else [],
            'phase_angle': list(self.phase_buffer) if self.phase_buffer else []
        }
        
        timestamps = [system_state.timestamp - i * self.pmu_update_interval 
                     for i in range(len(self.frequency_buffer))]
        
        # IEEE attack detection
        attack_detected, detection_details = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            IEEE_AttackDetection.detect_coordinated_attack,
            measurements,
            timestamps
        )
        
        system_state.attack_detected = attack_detected
        
        if attack_detected:
            logger.warning(f"IEEE attack detection: {detection_details}")
    
    async def _execute_single_bus_attack(self,
                                       bus_num: int,
                                       attack_config: AttackConfig,
                                       system_interface: Callable) -> AsyncAttackState:
        """Execute attack on single bus with IEEE compliance"""
        attack_state = AsyncAttackState()
        
        logger.info(f"Starting attack on bus {bus_num}")
        
        # This would implement the actual bus-specific attack
        # For now, simulate the attack execution
        await asyncio.sleep(attack_config.duration)
        
        logger.info(f"Completed attack on bus {bus_num}")
        return attack_state
    
    async def _apply_attack_to_system(self,
                                    attack_state: AsyncAttackState,
                                    system_interface: Callable) -> None:
        """Apply attack intensity to power system interface"""
        # This would interface with actual power system simulation
        # Implementation depends on the specific power system interface
        pass
    
    async def cleanup(self) -> None:
        """Cleanup async resources"""
        # Cancel all active tasks
        for attack_id, task in self.active_attacks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if self.system_monitor_task and not self.system_monitor_task.done():
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logger.info("Async dynamic attack generator cleaned up")

# ======================== ASYNC ATTACK ORCHESTRATOR ======================== #

class AsyncAttackOrchestrator:
    """
    IEEE-compliant orchestrator for multiple async attacks
    """
    
    def __init__(self, ieee_params: Optional[IEEE_SystemParameters] = None):
        """Initialize with IEEE parameters"""
        self.ieee_params = ieee_params or IEEE_Standards.get_ieee39_standard_parameters()
        self.attack_generator = AsyncDynamicAttackGenerator(self.ieee_params)
        
        # Attack coordination
        self.active_orchestrations: Dict[str, Dict[str, Any]] = {}
        self.coordination_lock = asyncio.Lock()
        
        logger.info("AsyncAttackOrchestrator initialized")
    
    async def orchestrate_multi_stage_attack(self,
                                           attack_stages: List[Dict[str, Any]],
                                           system_interface: Callable) -> Dict[str, Any]:
        """
        Orchestrate multi-stage IEEE-compliant attack
        
        Args:
            attack_stages: List of attack stage configurations
            system_interface: Async system interface
            
        Returns:
            Orchestration results and timing
        """
        orchestration_id = f"multi_stage_{int(time.time())}"
        
        async with self.coordination_lock:
            self.active_orchestrations[orchestration_id] = {
                'stages': attack_stages,
                'current_stage': 0,
                'start_time': time.time(),
                'stage_results': []
            }
        
        logger.info(f"Starting multi-stage attack {orchestration_id} with {len(attack_stages)} stages")
        
        results = {
            'orchestration_id': orchestration_id,
            'total_stages': len(attack_stages),
            'stage_results': [],
            'total_duration': 0.0
        }
        
        # Execute stages sequentially with IEEE timing compliance
        for stage_num, stage_config in enumerate(attack_stages):
            logger.info(f"Executing stage {stage_num + 1}/{len(attack_stages)}")
            
            stage_start_time = time.time()
            
            # Execute stage attack
            stage_result = await self._execute_attack_stage(
                stage_config, system_interface, f"{orchestration_id}_stage_{stage_num}"
            )
            
            stage_duration = time.time() - stage_start_time
            stage_result['duration'] = stage_duration
            stage_result['stage_number'] = stage_num + 1
            
            results['stage_results'].append(stage_result)
            
            # Update orchestration state
            async with self.coordination_lock:
                self.active_orchestrations[orchestration_id]['current_stage'] = stage_num + 1
                self.active_orchestrations[orchestration_id]['stage_results'].append(stage_result)
        
        results['total_duration'] = time.time() - self.active_orchestrations[orchestration_id]['start_time']
        
        logger.info(f"Multi-stage attack {orchestration_id} completed in {results['total_duration']:.2f}s")
        
        # Cleanup orchestration
        async with self.coordination_lock:
            del self.active_orchestrations[orchestration_id]
        
        return results
    
    async def _execute_attack_stage(self,
                                  stage_config: Dict[str, Any],
                                  system_interface: Callable,
                                  stage_id: str) -> Dict[str, Any]:
        """Execute individual attack stage"""
        
        attack_type = stage_config.get('attack_type', AttackType.FEEDBACK)
        target_buses = stage_config.get('target_buses', [20, 21])
        attack_config = stage_config.get('attack_config')
        
        stage_result = {
            'stage_id': stage_id,
            'attack_type': attack_type.value if hasattr(attack_type, 'value') else str(attack_type),
            'target_buses': target_buses,
            'success': False,
            'error_message': None
        }
        
        try:
            if attack_type == AttackType.FEEDBACK:
                # Execute feedback attack
                attack_state = await self.attack_generator.generate_feedback_attack_async(
                    attack_config, system_interface
                )
                
                # Run adaptive loop
                await self.attack_generator.adaptive_attack_loop_async(
                    attack_state, attack_config, system_interface
                )
                
                stage_result['success'] = True
                stage_result['final_intensity'] = attack_state.attack_intensity
                stage_result['target_reached'] = attack_state.target_reached
                
            else:
                # Handle other attack types
                logger.warning(f"Attack type {attack_type} not yet implemented in async mode")
                stage_result['error_message'] = f"Attack type {attack_type} not implemented"
        
        except Exception as e:
            logger.error(f"Stage execution error: {e}")
            stage_result['error_message'] = str(e)
        
        return stage_result
    
    async def cleanup(self) -> None:
        """Cleanup orchestrator resources"""
        await self.attack_generator.cleanup()
        self.active_orchestrations.clear()
        logger.info("AsyncAttackOrchestrator cleaned up")

# ======================== MODULE EXPORTS ======================== #

__all__ = [
    'AsyncSystemState',
    'AsyncAttackState', 
    'AsyncDynamicAttackGenerator',
    'AsyncAttackOrchestrator'
]

# ======================== ASYNC DEMO FUNCTION ======================== #

async def demo_async_dynamic_attacks():
    """Demonstration of async dynamic attack capabilities"""
    print("=" * 60)
    print("ASYNC DYNAMIC LAA DEMONSTRATION")
    print("=" * 60)
    
    # Initialize IEEE-compliant generator
    ieee_params = IEEE_Standards.get_ieee39_standard_parameters()
    generator = AsyncDynamicAttackGenerator(ieee_params)
    orchestrator = AsyncAttackOrchestrator(ieee_params)
    
    print(f" Initialized with IEEE parameters:")
    print(f"   PMU Rate: {ieee_params.pmu_reporting_rate} fps")
    print(f"   Frequency Deadband: ±{ieee_params.frequency_deadband} Hz")
    print(f"   AVR Gain: {ieee_params.avr_gain}")
    
    # Simulate system interface
    async def mock_system_interface():
        return {"status": "ok", "timestamp": time.time()}
    
    try:
        # Start monitoring
        await generator.start_system_monitoring(mock_system_interface)
        print(" Started IEEE-compliant system monitoring")
        
        # Test feedback attack generation
        from ..attacker.laa_config import AttackConfig, AttackType
        attack_config = AttackConfig(
            attack_type=AttackType.FEEDBACK,
            target_buses=[20, 21],
            magnitude_mw=50.0,
            duration=10.0
        )
        
        attack_state = await generator.generate_feedback_attack_async(
            attack_config, mock_system_interface
        )
        print(" Generated IEEE-compliant feedback attack")
        print(f"   Target frequency deviation: {attack_state.target_frequency_deviation} Hz")
        
        # Brief simulation
        print(" Running 5-second simulation...")
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f" Demo error: {e}")
    
    finally:
        await generator.cleanup()
        await orchestrator.cleanup()
        print(" Async resources cleaned up")
    
    print(" Async Dynamic LAA Demo Complete!")

if __name__ == "__main__":
    asyncio.run(demo_async_dynamic_attacks())
"""
Main grid simulation using pandapower for IEEE-39 bus system.
Integrates AVR endpoints with PROFINET communication.
"""

import asyncio
import time
import logging
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from typing import Dict, List, Optional, Tuple
import threading
import sys
import os

# Add current directory to path for our imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our components
from components.avr import AVR
from components.generator import Generator
from components.load import Load
from endpoints.sensor import Sensor
from endpoints.controller import Controller
from endpoints.actuator import Actuator
from protocols.profinet import ProfinetProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GridSimulation:
    """
    Main grid simulation class that integrates pandapower with AVR endpoints.
    """
    
    def __init__(self):
        """Initialize the grid simulation."""
        # Pandapower network
        self.net = None
        self.ieee39_net = None
        
        # Our component objects
        self.generators: Dict[int, Generator] = {}
        self.loads: Dict[int, Load] = {}
        
        # AVR system components
        self.sensors: Dict[int, Sensor] = {}
        self.controllers: Dict[int, Controller] = {}
        self.actuators: Dict[int, Actuator] = {}
        
        # Communication protocol
        self.profinet = ProfinetProtocol("grid_profinet", network_latency=0.002)
        
        # Simulation state
        self.is_running = False
        self.simulation_task: Optional[asyncio.Task] = None
        self.current_time = 0.0
        self.time_step = 1.0  # seconds
        
        # Generator-AVR mappings
        self.avr_systems: Dict[int, Dict] = {}  # gen_id -> {sensor, controller, actuator}
        
    def create_ieee39_system(self):
        """Create IEEE-39 bus system using pandapower."""
        logger.info("Creating IEEE-39 bus system...")
        
        # Load the IEEE 39-bus system
        self.net = nw.case39()
        
        # Log system information
        logger.info(f"Created IEEE-39 system:")
        logger.info(f"  Buses: {len(self.net.bus)}")
        logger.info(f"  Generators: {len(self.net.gen)}")
        logger.info(f"  Loads: {len(self.net.load)}")
        logger.info(f"  Lines: {len(self.net.line)}")
        logger.info(f"  Transformers: {len(self.net.trafo)}")
        
        # Create our component objects for tracking
        self._create_component_objects()
        
        # Run initial power flow
        self._run_power_flow()
        
    def _create_component_objects(self):
        """Create component objects for generators and loads."""
        # Create generator objects
        for idx, gen in self.net.gen.iterrows():
            bus_id = gen['bus']
            
            # Get generator limits
            p_max = gen.get('max_p_mw', 100.0)
            q_max = gen.get('max_q_mvar', 50.0)
            
            gen_obj = Generator(
                gen_id=idx,
                bus_id=bus_id,
                name=f"Gen_{idx}_Bus_{bus_id}",
                p_max=p_max,
                q_max=q_max
            )
            
            # Set initial power output (generators start with P, Q will be calculated)
            initial_p = gen.get('p_mw', 0.0)
            gen_obj.set_power_output(initial_p, 0.0)  # Q will be determined by power flow
            self.generators[idx] = gen_obj
            
        # Create load objects
        for idx, load in self.net.load.iterrows():
            bus_id = load['bus']
            load_obj = Load(
                load_id=idx,
                bus_id=bus_id,
                name=f"Load_{idx}_Bus_{bus_id}",
                p_mw=load['p_mw'],
                q_mvar=load['q_mvar']
            )
            self.loads[idx] = load_obj
            
    def deploy_avr_system(self, generator_id: int, voltage_setpoint: float = 1.0):
        """
        Deploy AVR control system for a specific generator.
        
        Args:
            generator_id: ID of generator to control
            voltage_setpoint: Desired voltage setpoint in pu
        """
        if generator_id not in self.generators:
            logger.error(f"Generator {generator_id} not found")
            return
            
        gen = self.generators[generator_id]
        bus_id = gen.bus_id
        
        logger.info(f"Deploying AVR system for Generator {generator_id} at Bus {bus_id}")
        
        # Create AVR endpoints
        sensor = Sensor(
            sensor_id=generator_id,  # Use integer instead of string
            bus_id=bus_id,
            sampling_rate=50.0,  # 50 Hz
            communication_protocol=self.profinet
        )
        
        controller = Controller(
            controller_id=generator_id,  # Use integer instead of string
            generator_id=generator_id,
            bus_id=bus_id,
            voltage_setpoint=voltage_setpoint,
            communication_protocol=self.profinet
        )
        
        actuator = Actuator(
            actuator_id=generator_id,  # Use integer instead of string
            generator_id=generator_id,
            response_time=0.01,  # 10ms
            communication_protocol=self.profinet
        )
        
        # Set grid interfaces
        sensor.set_grid_interface(self)
        actuator.set_grid_interface(self)
        
        # Register with PROFINET
        self.profinet.register_endpoint(f"sensor_{generator_id}", sensor)
        self.profinet.register_endpoint(f"controller_{generator_id}", controller)
        self.profinet.register_endpoint(f"actuator_{generator_id}", actuator)
        
        # Store references
        self.sensors[generator_id] = sensor
        self.controllers[generator_id] = controller
        self.actuators[generator_id] = actuator
        
        self.avr_systems[generator_id] = {
            'sensor': sensor,
            'controller': controller,
            'actuator': actuator,
            'bus_id': bus_id
        }
        
        logger.info(f"AVR system deployed for Generator {generator_id}")
        
    async def start_simulation(self):
        """Start the simulation."""
        if self.is_running:
            logger.warning("Simulation already running")
            return
            
        logger.info("Starting grid simulation...")
        self.is_running = True
        
        # Start PROFINET network
        await self.profinet.start_network()
        
        # Start all AVR endpoints
        for gen_id, avr_system in self.avr_systems.items():
            await avr_system['sensor'].start_measurements()
            await avr_system['controller'].start_control()
            await avr_system['actuator'].start_actuator()
            
        # Start main simulation loop
        self.simulation_task = asyncio.create_task(self._simulation_loop())
        
        logger.info("Grid simulation started")
        
    async def stop_simulation(self):
        """Stop the simulation."""
        logger.info("Stopping grid simulation...")
        self.is_running = False
        
        # Stop simulation loop
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass
                
        # Stop all AVR endpoints
        for gen_id, avr_system in self.avr_systems.items():
            await avr_system['sensor'].stop_measurements()
            await avr_system['controller'].stop_control()
            await avr_system['actuator'].stop_actuator()
            
        # Stop PROFINET network
        await self.profinet.stop_network()
        
        logger.info("Grid simulation stopped")
        
    async def _simulation_loop(self):
        """Main simulation loop."""
        try:
            while self.is_running:
                start_time = time.time()
                
                # Run power flow calculation
                self._run_power_flow()
                
                # Update simulation time
                self.current_time += self.time_step
                
                # Log system status periodically
                if int(self.current_time) % 10 == 0:  # Every 10 seconds
                    self._log_system_status()
                    
                # Calculate how long to sleep to maintain time step
                elapsed = time.time() - start_time
                sleep_time = max(0, self.time_step - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
            
    def _run_power_flow(self):
        """Run power flow calculation."""
        try:
            pp.runpp(self.net, verbose=False)
            
            # Update component objects with results
            self._update_components_from_powerflow()
            
        except Exception as e:
            logger.error(f"Power flow calculation failed: {e}")
            
    def _update_components_from_powerflow(self):
        """Update component objects with power flow results."""
        # Update generators
        for gen_id, gen_obj in self.generators.items():
            if gen_id < len(self.net.res_gen):
                result = self.net.res_gen.iloc[gen_id]
                gen_obj.set_power_output(result['p_mw'], result['q_mvar'])
                
        # Update loads (they remain as set, but we could add voltage dependency here)
        for load_id, load_obj in self.loads.items():
            if load_id < len(self.net.load):
                net_load = self.net.load.iloc[load_id]
                load_obj.set_demand(net_load['p_mw'], net_load['q_mvar'])
                
    def get_bus_voltage(self, bus_id: int) -> Tuple[float, float]:
        """
        Get bus voltage magnitude and angle.
        
        Args:
            bus_id: Bus ID
            
        Returns:
            Tuple of (voltage_magnitude_pu, voltage_angle_deg)
        """
        try:
            if bus_id < len(self.net.res_bus):
                result = self.net.res_bus.iloc[bus_id]
                voltage_pu = result['vm_pu']
                angle_deg = result['va_degree']
                return voltage_pu, angle_deg
            else:
                logger.warning(f"Bus {bus_id} not found in results")
                return 1.0, 0.0
        except Exception as e:
            logger.error(f"Error getting voltage for bus {bus_id}: {e}")
            return 1.0, 0.0
            
    def set_generator_excitation(self, generator_id: int, excitation: float) -> bool:
        """
        Set generator excitation level.
        
        Args:
            generator_id: Generator ID
            excitation: Excitation level in pu
            
        Returns:
            True if successful
        """
        try:
            if generator_id in self.generators:
                gen_obj = self.generators[generator_id]
                gen_obj.set_excitation(excitation)
                
                # Update pandapower model - adjust voltage setpoint
                if generator_id < len(self.net.gen):
                    # Scale voltage setpoint based on excitation
                    base_voltage = 1.0
                    new_voltage = base_voltage * excitation
                    self.net.gen.at[generator_id, 'vm_pu'] = new_voltage
                    
                logger.debug(f"Set Generator {generator_id} excitation to {excitation:.3f}")
                return True
            else:
                logger.warning(f"Generator {generator_id} not found")
                return False
        except Exception as e:
            logger.error(f"Error setting excitation for generator {generator_id}: {e}")
            return False
            
    def change_load_demand(self, load_id: int, delta_p_mw: float, delta_q_mvar: float = None):
        """
        Change load demand dynamically.
        
        Args:
            load_id: Load ID
            delta_p_mw: Change in active power (MW)
            delta_q_mvar: Change in reactive power (MVAR), if None maintains power factor
        """
        try:
            if load_id in self.loads and load_id < len(self.net.load):
                load_obj = self.loads[load_id]
                
                # Update our load object
                load_obj.increase_demand(delta_p_mw, delta_q_mvar)
                
                # Update pandapower network
                self.net.load.at[load_id, 'p_mw'] = load_obj.p_demand
                self.net.load.at[load_id, 'q_mvar'] = load_obj.q_demand
                
                logger.info(f"Changed Load {load_id} demand by ΔP={delta_p_mw:.1f}MW, "
                           f"new demand: P={load_obj.p_demand:.1f}MW, Q={load_obj.q_demand:.1f}MVAR")
                
                # Log bus voltage before change
                bus_id = self.net.load.iloc[load_id]['bus']
                voltage_before, _ = self.get_bus_voltage(bus_id)
                logger.info(f"Bus {bus_id} voltage before load change: {voltage_before:.3f} pu")
                
            else:
                logger.warning(f"Load {load_id} not found")
        except Exception as e:
            logger.error(f"Error changing load demand for load {load_id}: {e}")
            
    def _log_system_status(self):
        """Log current system status."""
        logger.info(f"=== System Status at t={self.current_time:.1f}s ===")
        
        # Log AVR systems status
        for gen_id, avr_system in self.avr_systems.items():
            controller = avr_system['controller']
            bus_id = avr_system['bus_id']
            
            voltage, angle = self.get_bus_voltage(bus_id)
            avr_status = controller.get_avr_status()
            
            logger.info(f"Gen {gen_id} at Bus {bus_id}: "
                       f"V={voltage:.3f}pu, "
                       f"Vref={avr_status['voltage_setpoint']:.3f}pu, "
                       f"Error={avr_status['error']:.3f}pu, "
                       f"Exc={avr_status['excitation_output']:.3f}pu")
                       
        # Log PROFINET statistics
        profinet_stats = self.profinet.get_network_statistics()
        logger.info(f"PROFINET: {profinet_stats['messages_delivered']}/{profinet_stats['messages_sent']} "
                   f"delivered, avg latency: {profinet_stats['average_latency_ms']:.1f}ms")

async def main():
    """Main function to run the simulation."""
    # Create grid simulation
    grid = GridSimulation()
    
    # Create IEEE-39 system
    grid.create_ieee39_system()
    
    # Deploy AVR systems for selected generators
    # Let's control generators at buses with significant load
    avr_generators = [0, 1, 2]  # First 3 generators
    
    for gen_id in avr_generators:
        grid.deploy_avr_system(gen_id, voltage_setpoint=1.02)  # 1.02 pu setpoint
        
    # Start simulation
    await grid.start_simulation()
    
    try:
        # Let system stabilize
        logger.info("System stabilizing...")
        await asyncio.sleep(5)
        
        # Simulate load changes
        logger.info("\\n=== SIMULATING LOAD INCREASE ===")
        
        # Increase load at bus 3 (significant load bus)
        load_to_change = 0  # First load
        grid.change_load_demand(load_to_change, delta_p_mw=20.0)  # Increase by 20 MW
        
        # Let AVR respond
        logger.info("Waiting for AVR response...")
        await asyncio.sleep(10)
        
        # Another load change
        logger.info("\\n=== SIMULATING ANOTHER LOAD CHANGE ===")
        grid.change_load_demand(1, delta_p_mw=15.0)  # Increase another load
        
        # Let system respond
        await asyncio.sleep(10)
        
        # Demonstrate setpoint change
        logger.info("\\n=== CHANGING AVR SETPOINT ===")
        if 0 in grid.controllers:
            grid.controllers[0].set_voltage_setpoint(1.05)  # Increase setpoint
            logger.info("Changed Generator 0 voltage setpoint to 1.05 pu")
            
        # Final observation period
        await asyncio.sleep(15)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Stop simulation
        await grid.stop_simulation()
        logger.info("Simulation completed")

if __name__ == "__main__":
    # Run the simulation
    asyncio.run(main())

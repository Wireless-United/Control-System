"""
SCADA Simulation Main Entry Point

Demonstrates SCADA-RTU communication using DNP3 protocol.
Shows the difference between fast PROFINET AVR control and slower SCADA supervisory control.
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, List, Any

# Add simulation directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import grid simulation components
from grid import GridSimulation
from protocols.dnp3 import DNP3Protocol, DNP3ObjectGroup
from components.scada_master import SCADAMaster, SCADAMasterConfig
from components.rtu import RemoteTerminalUnit, RTUConfig

logger = logging.getLogger(__name__)


class SCADASystemDemo:
    """SCADA System Demonstration"""
    
    def __init__(self):
        self.grid_sim = None
        self.dnp3_network = None
        self.scada_master = None
        self.rtus: Dict[int, RemoteTerminalUnit] = {}
        
        # Demonstration parameters
        self.simulation_time = 60.0  # seconds
        self.status_interval = 10.0  # seconds
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Reduce noise from some loggers
        logging.getLogger('protocols.profinet').setLevel(logging.WARNING)
        logging.getLogger('protocols.dnp3').setLevel(logging.INFO)
    
    async def initialize_system(self):
        """Initialize the complete SCADA system"""
        logger.info("=== Initializing SCADA System ===")
        
        # 1. Initialize grid simulation (with existing PROFINET AVR control)
        self.grid_sim = GridSimulation()
        
        # Create IEEE-39 system and deploy AVR systems
        self.grid_sim.create_ieee39_system()
        self.grid_sim.deploy_avr_system(0, voltage_setpoint=1.02)  # Generator 0 at bus 30
        self.grid_sim.deploy_avr_system(1, voltage_setpoint=1.02)  # Generator 1 at bus 31
        self.grid_sim.deploy_avr_system(2, voltage_setpoint=1.02)  # Generator 2 at bus 32
        
        logger.info("Grid simulation initialized with PROFINET AVR control")
        
        # 2. Initialize DNP3 network
        self.dnp3_network = DNP3Protocol(
            station_id=0,  # Network coordinator
            network_name="scada_dnp3",
            communication_delay=(100, 500)  # 100-500ms delay
        )
        await self.dnp3_network.start_network()
        logger.info("DNP3 network started with 100-500ms communication delay")
        
        # 3. Initialize SCADA Master
        scada_config = SCADAMasterConfig(
            master_id=1,
            master_name="Control Center SCADA",
            polling_interval=5.0,  # Poll every 5 seconds
            timeout=10.0,
            max_retries=3
        )
        
        self.scada_master = SCADAMaster(scada_config, self.dnp3_network)
        await self.scada_master.start()
        logger.info("SCADA Master started with 5-second polling interval")
        
        # 4. Initialize RTUs for key buses
        await self._initialize_rtus()
        
        logger.info("SCADA system initialization complete")
    
    async def _initialize_rtus(self):
        """Initialize RTUs for monitoring key grid buses"""
        # RTU configurations for key buses in IEEE-39 system
        rtu_configs = [
            {
                'station_id': 10,
                'name': 'Generation RTU 1',
                'location': 'Power Plant Alpha',
                'bus_id': 29,  # Generator bus (Gen 0)
                'generator_id': 0
            },
            {
                'station_id': 11,
                'name': 'Generation RTU 2', 
                'location': 'Power Plant Beta',
                'bus_id': 31,  # Generator bus (Gen 1)
                'generator_id': 1
            },
            {
                'station_id': 12,
                'name': 'Load Center RTU',
                'location': 'City Substation',
                'bus_id': 3,   # Major load bus
                'generator_id': None
            }
        ]
        
        for config in rtu_configs:
            # Create RTU configuration
            rtu_config = RTUConfig(
                station_id=config['station_id'],
                station_name=config['name'],
                location=config['location'],
                bus_id=config['bus_id'],
                unsolicited_enabled=True,
                unsolicited_threshold=0.02  # 2% change threshold
            )
            
            # Create RTU instance
            rtu = RemoteTerminalUnit(rtu_config, self.dnp3_network, self.grid_sim)
            
            # Register sensor callbacks for this RTU
            await self._setup_rtu_sensors(rtu, config)
            
            # Register control callbacks for this RTU
            await self._setup_rtu_controls(rtu, config)
            
            # Start RTU
            await rtu.start()
            
            # Add RTU to SCADA master monitoring
            self.scada_master.add_rtu(
                station_id=config['station_id'],
                name=config['name'],
                location=config['location'],
                bus_id=config['bus_id']
            )
            
            self.rtus[config['station_id']] = rtu
            
            logger.info(f"RTU {config['name']} initialized for Bus {config['bus_id']}")
    
    async def _setup_rtu_sensors(self, rtu: RemoteTerminalUnit, config: Dict):
        """Setup sensor callbacks for RTU data collection"""
        bus_id = config['bus_id']
        gen_id = config.get('generator_id')
        
        # Voltage measurement (pu)
        def get_bus_voltage():
            try:
                if hasattr(self.grid_sim, 'net') and self.grid_sim.net is not None:
                    voltage_pu = self.grid_sim.net.res_bus.vm_pu.iloc[bus_id]
                    return float(voltage_pu)
                return 1.0
            except:
                return 1.0
        
        # Active power (MW)
        def get_active_power():
            try:
                if hasattr(self.grid_sim, 'net') and self.grid_sim.net is not None:
                    if gen_id is not None:
                        # Generator power
                        power_mw = self.grid_sim.net.res_gen.p_mw.iloc[gen_id]
                    else:
                        # Load power
                        power_mw = -self.grid_sim.net.res_load.p_mw.iloc[0] if len(self.grid_sim.net.res_load) > 0 else 0.0
                    return float(power_mw)
                return 0.0
            except:
                return 0.0
        
        # Reactive power (MVAR)
        def get_reactive_power():
            try:
                if hasattr(self.grid_sim, 'net') and self.grid_sim.net is not None:
                    if gen_id is not None:
                        # Generator reactive power
                        power_mvar = self.grid_sim.net.res_gen.q_mvar.iloc[gen_id]
                    else:
                        # Load reactive power
                        power_mvar = -self.grid_sim.net.res_load.q_mvar.iloc[0] if len(self.grid_sim.net.res_load) > 0 else 0.0
                    return float(power_mvar)
                return 0.0
            except:
                return 0.0
        
        # Breaker status (always online for simulation)
        def get_breaker_status():
            return True
        
        # Generator online status
        def get_generator_status():
            return gen_id is not None
        
        # Register analog input callbacks
        rtu.register_sensor_callback("analog_input", 1, get_bus_voltage)     # Bus voltage (pu)
        rtu.register_sensor_callback("analog_input", 2, get_active_power)    # Active power
        rtu.register_sensor_callback("analog_input", 3, get_reactive_power)  # Reactive power
        
        # Register binary input callbacks
        rtu.register_sensor_callback("binary_input", 0, get_breaker_status)   # Breaker status
        rtu.register_sensor_callback("binary_input", 1, get_generator_status) # Generator status
    
    async def _setup_rtu_controls(self, rtu: RemoteTerminalUnit, config: Dict):
        """Setup control callbacks for RTU control actions"""
        bus_id = config['bus_id']
        gen_id = config.get('generator_id')
        
        # Breaker control
        async def control_breaker(state: bool):
            action = "CLOSE" if state else "OPEN"
            logger.info(f"🔌 RTU {rtu.station_id}: Breaker {action} command for Bus {bus_id}")
            # In a real system, this would control actual breaker
            # For simulation, we just log the action
        
        # Generator start/stop
        async def control_generator(state: bool):
            if gen_id is not None:
                action = "START" if state else "STOP"
                logger.info(f"⚡ RTU {rtu.station_id}: Generator {action} command for Gen {gen_id}")
                # In a real system, this would start/stop generator
                # For simulation, we just log the action
        
        # Voltage setpoint control
        async def control_voltage_setpoint(setpoint: float):
            if gen_id is not None:
                logger.info(f"📊 RTU {rtu.station_id}: Voltage setpoint changed to {setpoint:.3f} pu for Gen {gen_id}")
                # Could integrate with AVR system here
                if hasattr(self.grid_sim, 'controllers') and gen_id in self.grid_sim.controllers:
                    controller = self.grid_sim.controllers[gen_id]
                    if hasattr(controller, 'avr'):
                        controller.avr.setpoint = setpoint
                        logger.info(f"📊 Updated AVR setpoint for Gen {gen_id} to {setpoint:.3f} pu")
        
        # Register control callbacks
        rtu.register_control_callback("binary_output", 0, control_breaker)
        rtu.register_control_callback("binary_output", 1, control_generator)
        rtu.register_control_callback("analog_output", 0, control_voltage_setpoint)
    
    async def run_demonstration(self):
        """Run the SCADA system demonstration"""
        logger.info("=== Starting SCADA System Demonstration ===")
        
        try:
            # Start grid simulation
            await self.grid_sim.start_simulation()
            grid_task = asyncio.create_task(self._keep_grid_running())
            
            # Start status monitoring
            status_task = asyncio.create_task(self._status_monitor())
            
            # Start demonstration scenarios
            demo_task = asyncio.create_task(self._demonstration_scenarios())
            
            # Run for specified time
            await asyncio.sleep(self.simulation_time)
            
            logger.info("=== Stopping SCADA System Demonstration ===")
            
            # Cancel tasks
            grid_task.cancel()
            status_task.cancel()
            demo_task.cancel()
            
            # Wait for cleanup
            await asyncio.gather(grid_task, status_task, demo_task, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in demonstration: {e}")
        
        finally:
            await self._cleanup()
    
    async def _keep_grid_running(self):
        """Keep grid simulation running"""
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Grid simulation task cancelled")
            raise
    
    async def _status_monitor(self):
        """Monitor and display system status"""
        try:
            while True:
                await asyncio.sleep(self.status_interval)
                await self._display_status()
                
        except asyncio.CancelledError:
            logger.info("Status monitor cancelled")
            raise
    
    async def _display_status(self):
        """Display current system status"""
        current_time = time.time() - getattr(self, 'start_time', time.time())
        
        logger.info(f"\n=== System Status at t={current_time:.1f}s ===")
        
        # SCADA Master status
        scada_status = self.scada_master.get_status()
        logger.info(f"SCADA Master: {scada_status['measurement_count']} measurements, "
                   f"{scada_status['alarm_count']} alarms")
        
        # DNP3 Network statistics
        dnp3_stats = self.dnp3_network.get_statistics()
        logger.info(f"DNP3 Network: {dnp3_stats['messages_delivered']}/{dnp3_stats['messages_sent']} delivered, "
                   f"avg latency: {dnp3_stats['average_latency_ms']}ms")
        
        # RTU status
        rtu_status = self.scada_master.get_rtu_status()
        for rtu_id, status in rtu_status.items():
            comm_status = status['communication_status']
            last_comm = status['communication_age_seconds']
            logger.info(f"RTU {status['name']}: {comm_status}, "
                       f"last communication: {last_comm:.1f}s ago")
        
        # Grid measurements from SCADA perspective
        measurements = self.scada_master.get_measurements()
        voltage_measurements = {k: v for k, v in measurements.items() 
                              if 'ANALOG_INPUT_1' in k}  # Bus voltage measurements
        
        for point_id, measurement in voltage_measurements.items():
            rtu_name = measurement['rtu_name']
            voltage = measurement['value']
            age = time.time() - measurement['received_time']
            logger.info(f"SCADA: {rtu_name} voltage = {voltage:.3f} pu (age: {age:.1f}s)")
    
    async def _demonstration_scenarios(self):
        """Run demonstration scenarios"""
        try:
            # Wait for system to settle
            await asyncio.sleep(15.0)
            
            # Scenario 1: Operator adjusts voltage setpoint via SCADA
            logger.info("\n🎯 === SCENARIO 1: Operator Voltage Setpoint Change ===")
            logger.info("Operator adjusts voltage setpoint at Generation RTU 1 via SCADA...")
            
            await self.scada_master.queue_operator_command({
                'type': 'voltage_setpoint',
                'rtu_id': 10,
                'setpoint': 1.04  # Increase setpoint to 1.04 pu
            })
            
            await asyncio.sleep(10.0)
            
            # Scenario 2: Breaker control
            logger.info("\n🔌 === SCENARIO 2: Remote Breaker Operation ===")
            logger.info("Operator opens breaker at Load Center RTU via SCADA...")
            
            await self.scada_master.queue_operator_command({
                'type': 'breaker_control',
                'rtu_id': 12,
                'action': 'open'
            })
            
            await asyncio.sleep(5.0)
            
            # Close breaker again
            logger.info("Operator closes breaker at Load Center RTU...")
            await self.scada_master.queue_operator_command({
                'type': 'breaker_control',
                'rtu_id': 12,
                'action': 'close'
            })
            
            await asyncio.sleep(10.0)
            
            # Scenario 3: Generator control
            logger.info("\n⚡ === SCENARIO 3: Generator Remote Control ===")
            logger.info("Operator stops generator at Generation RTU 2...")
            
            await self.scada_master.queue_operator_command({
                'type': 'generator_control',
                'rtu_id': 11,
                'action': 'stop'
            })
            
            await asyncio.sleep(5.0)
            
            # Start generator again
            logger.info("Operator starts generator at Generation RTU 2...")
            await self.scada_master.queue_operator_command({
                'type': 'generator_control',
                'rtu_id': 11,
                'action': 'start'
            })
            
        except asyncio.CancelledError:
            logger.info("Demonstration scenarios cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in demonstration scenarios: {e}")
    
    async def _cleanup(self):
        """Cleanup system resources"""
        logger.info("Cleaning up SCADA system...")
        
        try:
            # Stop RTUs
            for rtu in self.rtus.values():
                await rtu.stop()
            
            # Stop SCADA master
            if self.scada_master:
                await self.scada_master.stop()
            
            # Stop DNP3 network
            if self.dnp3_network:
                await self.dnp3_network.stop_network()
            
            # Stop grid simulation
            if self.grid_sim:
                await self.grid_sim.stop_simulation()
            
            logger.info("SCADA system cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main function to run SCADA demonstration"""
    print("🏭 SCADA-RTU Communication Demonstration using DNP3")
    print("=" * 60)
    print("This demo shows:")
    print("• SCADA master polling RTUs every 5 seconds")
    print("• RTUs responding with grid measurements")
    print("• Manual control commands via SCADA")
    print("• Comparison with fast PROFINET AVR control")
    print("=" * 60)
    
    # Create and run demonstration
    demo = SCADASystemDemo()
    demo.start_time = time.time()
    
    try:
        await demo.initialize_system()
        await demo.run_demonstration()
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

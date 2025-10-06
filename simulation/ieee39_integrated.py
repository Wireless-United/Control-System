#!/usr/bin/env python3
"""
Integrated IEEE 39-Bus SCADA-RTU-Attack Simulation Runner

Main simulation runner that orchestrates the complete IEEE 39-bus system with:
- IEEE 39-bus power system simulation
- Multiple        for rtu_id, name, ip_address, port, bus_number in scada_config:
            if rtu_id <= self.config.rtu_count:  # Only add configured RTUs
                self.scada.add_rtu(rtu_id, name, ip_address, port, bus_number, poll_interval=3.0)U outstations
- SCADA master station
- Simultaneous MiTM attacks with DNP3 manipulation
- Real-time monitoring and logging

This is the main entry point for comprehensive cybersecurity testing
of power system communication infrastructure.
"""

import asyncio
import logging
import time
import json
import signal
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import argparse

# Add simulation modules to path
sys.path.append(os.path.dirname(__file__))

# Import simulation components
from ieee39_system_strict import StrictIEEE39BusSystem
from rtu import IEEE39RTU, RTUConfiguration, create_ieee39_rtu_configurations
from scada import SCADAMaster, create_ieee39_scada_configuration
from ieee39_mitm import IEEE39MiTMController, IEEE39AttackScenario, ATTACK_CONFIGURATIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ieee39_simulation.log')
    ]
)
logger = logging.getLogger(__name__)

class SimulationMode(Enum):
    """Simulation operation modes"""
    NORMAL_OPERATION = "normal"
    SCADA_ONLY = "scada_only"
    ATTACK_SIMULATION = "attack"
    FULL_CYBERSECURITY = "full_cyber"

@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""
    mode: SimulationMode
    duration: int = 300  # 5 minutes default
    enable_power_system: bool = True
    enable_rtus: bool = True
    enable_scada: bool = True
    enable_attacks: bool = False
    attack_scenarios: List[str] = None
    attack_delay: int = 60  # Start attacks after 60 seconds
    rtu_count: int = 10
    log_level: str = "INFO"
    save_results: bool = True

class IEEE39IntegratedSimulation:
    """
    Integrated IEEE 39-Bus System Simulation
    
    Orchestrates the complete power system cybersecurity simulation including:
    - Power system dynamics
    - SCADA-RTU communication
    - Cybersecurity attacks
    - Real-time monitoring
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize integrated simulation.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.is_running = False
        self.start_time = 0.0
        
        # Simulation components
        self.power_system: Optional[StrictIEEE39BusSystem] = None
        self.rtus: Dict[int, IEEE39RTU] = {}
        self.scada: Optional[SCADAMaster] = None
        self.mitm_controller: Optional[IEEE39MiTMController] = None
        
        # Simulation statistics
        self.stats = {
            'start_time': 0.0,
            'uptime': 0.0,
            'power_system_updates': 0,
            'rtu_polls': 0,
            'scada_commands': 0,
            'attack_attempts': 0,
            'successful_attacks': 0,
            'communication_errors': 0
        }
        
        # Results storage
        self.simulation_results = {
            'configuration': config.__dict__,
            'power_system_data': [],
            'communication_logs': [],
            'attack_results': [],
            'performance_metrics': {}
        }
        
        logger.info(f"IEEE 39-Bus Integrated Simulation initialized in {config.mode.value} mode")
    
    async def start_simulation(self):
        """Start the complete integrated simulation"""
        if self.is_running:
            logger.warning("Simulation already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.stats['start_time'] = self.start_time
        
        logger.info("🚀 STARTING IEEE 39-BUS INTEGRATED SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Duration: {self.config.duration} seconds")
        logger.info(f"Components: Power System: {self.config.enable_power_system}, "
                   f"RTUs: {self.config.enable_rtus}, SCADA: {self.config.enable_scada}, "
                   f"Attacks: {self.config.enable_attacks}")
        
        try:
            # Start system components based on configuration
            tasks = []
            
            # 1. Start IEEE 39-bus power system
            if self.config.enable_power_system:
                logger.info("🔋 Starting IEEE 39-Bus Power System...")
                self.power_system = StrictIEEE39BusSystem()
                # Power system runs continuously in background
                tasks.append(asyncio.create_task(self._power_system_loop()))
            
            # 2. Start RTU outstations
            if self.config.enable_rtus:
                logger.info("📡 Starting RTU Outstations...")
                await self._start_rtus()
                tasks.extend([asyncio.create_task(rtu.start()) for rtu in self.rtus.values()])
            
            # 3. Start SCADA master station
            if self.config.enable_scada:
                logger.info("🖥️ Starting SCADA Master Station...")
                await self._start_scada()
                tasks.append(asyncio.create_task(self.scada.start()))
            
            # 4. Wait for system stabilization
            logger.info("⏳ Waiting for system stabilization...")
            await asyncio.sleep(10)
            
            # 5. Start attacks if enabled
            if self.config.enable_attacks:
                logger.info(f"⏰ Scheduling attacks to start in {self.config.attack_delay} seconds...")
                tasks.append(asyncio.create_task(self._delayed_attack_start()))
            
            # 6. Start monitoring and data collection
            tasks.append(asyncio.create_task(self._monitoring_loop()))
            tasks.append(asyncio.create_task(self._data_collection_loop()))
            
            # 7. Main simulation timer
            tasks.append(asyncio.create_task(self._simulation_timer()))
            
            logger.info("✅ All simulation components started successfully")
            logger.info(f"🕐 Simulation will run for {self.config.duration} seconds")
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            await self._stop_simulation()
    
    async def _start_rtus(self):
        """Start all RTU outstations"""
        rtu_configs = create_ieee39_rtu_configurations()
        
        # Limit RTU count based on configuration
        rtu_configs = rtu_configs[:self.config.rtu_count]
        
        for config in rtu_configs:
            try:
                # Pass power system reference for real measurements
                power_system_ref = self.power_system if self.config.enable_power_system else None
                rtu = IEEE39RTU(config, power_system_ref)
                self.rtus[config.rtu_id] = rtu
                
                logger.info(f"✓ RTU {config.rtu_id} configured for Bus {config.bus_number} at {config.ip_address}:{config.port}")
                
            except Exception as e:
                logger.error(f"Failed to create RTU {config.rtu_id}: {e}")
        
        logger.info(f"Configured {len(self.rtus)} RTU outstations")
    
    async def _start_scada(self):
        """Start SCADA master station"""
        self.scada = SCADAMaster(master_id=1)
        
        # Add RTUs to SCADA polling list
        scada_config = create_ieee39_scada_configuration()
        
        for rtu_id, name, ip_address, port, bus_number in scada_config:
            if rtu_id <= self.config.rtu_count:  # Only add configured RTUs
                self.scada.add_rtu(rtu_id, name, ip_address, port, bus_number, poll_interval=2.0)
        
        logger.info(f"SCADA master configured with {len(self.scada.rtu_connections)} RTUs")
    
    async def _delayed_attack_start(self):
        """Start attacks after specified delay"""
        await asyncio.sleep(self.config.attack_delay)
        
        if not self.is_running:
            return
        
        logger.warning("🚨 INITIATING CYBERSECURITY ATTACK PHASE 🚨")
        
        # Determine attack scenarios
        if self.config.attack_scenarios:
            scenarios = [IEEE39AttackScenario(s) for s in self.config.attack_scenarios 
                        if s in [sc.value for sc in IEEE39AttackScenario]]
        else:
            # Default to critical infrastructure attack
            scenarios = ATTACK_CONFIGURATIONS['critical_infrastructure']['scenarios']
        
        # Create and start MiTM controller
        self.mitm_controller = IEEE39MiTMController()
        
        # Calculate attack duration (remainder of simulation)
        attack_duration = max(60, self.config.duration - self.config.attack_delay - 30)
        
        try:
            await self.mitm_controller.launch_coordinated_attack(scenarios, attack_duration)
        except Exception as e:
            logger.error(f"Attack execution failed: {e}")
    
    async def _power_system_loop(self):
        """Continuous power system simulation loop"""
        update_interval = 1.0  # 1 second updates
        
        while self.is_running:
            try:
                # Run power system analysis
                if self.power_system:
                    analysis_result = self.power_system.run_strict_ieee39_analysis()
                    
                    if analysis_result['pypower_analysis']:
                        self.stats['power_system_updates'] += 1
                        
                        # Store power system state for RTUs
                        system_state = self.power_system.get_system_state()
                        
                        # Log system status periodically
                        if self.stats['power_system_updates'] % 60 == 0:  # Every minute
                            logger.debug(f"Power System: {system_state['total_load_mw']:.1f} MW load, "
                                       f"{system_state['frequency_hz']:.3f} Hz, "
                                       f"V: {system_state['voltage_min']:.3f}-{system_state['voltage_max']:.3f} pu")
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Power system loop error: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    async def _monitoring_loop(self):
        """Monitor system performance and log status"""
        report_interval = 60  # Report every minute
        
        while self.is_running:
            try:
                await asyncio.sleep(report_interval)
                
                uptime = time.time() - self.start_time
                self.stats['uptime'] = uptime
                
                # Collect system status
                power_status = "OK" if self.power_system else "N/A"
                rtu_status = f"{sum(1 for rtu in self.rtus.values() if hasattr(rtu, 'is_running') and rtu.is_running)}/{len(self.rtus)}"
                scada_status = "ACTIVE" if self.scada and self.scada.is_running else "N/A"
                attack_status = "ACTIVE" if self.mitm_controller and self.mitm_controller.is_attacking else "INACTIVE"
                
                logger.info("📊 SYSTEM STATUS REPORT:")
                logger.info(f"  Uptime: {uptime/60:.1f} minutes")
                logger.info(f"  Power System: {power_status}")
                logger.info(f"  RTUs: {rtu_status} active")
                logger.info(f"  SCADA: {scada_status}")
                logger.info(f"  Attack Status: {attack_status}")
                
                # Get detailed statistics
                if self.scada:
                    scada_stats = self.scada.get_system_status()
                    logger.info(f"  SCADA Polls: {scada_stats['statistics']['polls_sent']}")
                    logger.info(f"  Active Alarms: {scada_stats['active_alarms']}")
                
                if self.mitm_controller:
                    attack_stats = self.mitm_controller.get_attack_status()
                    if attack_stats['is_attacking']:
                        logger.info(f"  Attacks Executed: {attack_stats['statistics']['attacks_executed']}")
                        logger.info(f"  Success Rate: {attack_stats['statistics']['successful_attacks']}/{attack_stats['statistics']['attacks_executed']}")
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _data_collection_loop(self):
        """Collect simulation data for analysis"""
        collection_interval = 10  # Collect every 10 seconds
        
        while self.is_running:
            try:
                await asyncio.sleep(collection_interval)
                
                timestamp = time.time()
                
                # Collect power system data
                if self.power_system:
                    system_state = self.power_system.get_system_state()
                    power_data = {
                        'timestamp': timestamp,
                        'total_load_mw': system_state.get('total_load_mw', 0),
                        'total_generation_mw': system_state.get('total_generation_mw', 0),
                        'frequency_hz': system_state.get('frequency_hz', 50.0),
                        'voltage_min': system_state.get('voltage_min', 1.0),
                        'voltage_max': system_state.get('voltage_max', 1.0),
                        'power_flow_converged': system_state.get('power_flow_converged', False)
                    }
                    self.simulation_results['power_system_data'].append(power_data)
                
                # Collect SCADA communication data
                if self.scada:
                    scada_status = self.scada.get_system_status()
                    comm_data = {
                        'timestamp': timestamp,
                        'connected_rtus': scada_status['statistics']['connected_rtus'],
                        'polls_sent': scada_status['statistics']['polls_sent'],
                        'responses_received': scada_status['statistics']['responses_received'],
                        'active_alarms': scada_status['active_alarms'],
                        'current_measurements': len(scada_status.get('current_measurements', 0))
                    }
                    self.simulation_results['communication_logs'].append(comm_data)
                
                # Collect attack data
                if self.mitm_controller:
                    attack_status = self.mitm_controller.get_attack_status()
                    attack_data = {
                        'timestamp': timestamp,
                        'is_attacking': attack_status['is_attacking'],
                        'attacks_executed': attack_status['statistics']['attacks_executed'],
                        'successful_attacks': attack_status['statistics']['successful_attacks'],
                        'commands_injected': attack_status['statistics']['commands_injected'],
                        'data_corrupted': attack_status['statistics']['data_corrupted']
                    }
                    self.simulation_results['attack_results'].append(attack_data)
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
    
    async def _simulation_timer(self):
        """Main simulation timer"""
        await asyncio.sleep(self.config.duration)
        logger.info("⏰ Simulation duration completed")
        self.is_running = False
    
    async def _stop_simulation(self):
        """Stop all simulation components"""
        logger.info("🛑 STOPPING IEEE 39-BUS INTEGRATED SIMULATION")
        
        self.is_running = False
        
        try:
            # Stop attack components
            if self.mitm_controller:
                await self.mitm_controller._stop_attack()
            
            # Stop SCADA master
            if self.scada:
                await self.scada.stop()
            
            # Stop all RTUs
            for rtu in self.rtus.values():
                await rtu.stop()
            
            # Generate final results
            await self._generate_simulation_report()
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
        
        logger.info("✅ IEEE 39-Bus Integrated Simulation stopped successfully")
    
    async def _generate_simulation_report(self):
        """Generate comprehensive simulation report"""
        total_time = time.time() - self.start_time
        
        # Calculate performance metrics
        self.simulation_results['performance_metrics'] = {
            'total_duration': total_time,
            'power_system_updates': self.stats['power_system_updates'],
            'update_rate': self.stats['power_system_updates'] / total_time if total_time > 0 else 0,
            'rtus_deployed': len(self.rtus),
            'communication_success_rate': 0.0,  # Calculate from SCADA stats
            'attack_success_rate': 0.0  # Calculate from attack stats
        }
        
        # Calculate communication success rate
        if self.scada:
            scada_stats = self.scada.get_system_status()
            polls_sent = scada_stats['statistics']['polls_sent']
            responses_received = scada_stats['statistics']['responses_received']
            self.simulation_results['performance_metrics']['communication_success_rate'] = (
                responses_received / max(1, polls_sent) * 100
            )
        
        # Calculate attack success rate
        if self.mitm_controller:
            attack_stats = self.mitm_controller.get_attack_status()
            attacks_executed = attack_stats['statistics']['attacks_executed']
            successful_attacks = attack_stats['statistics']['successful_attacks']
            self.simulation_results['performance_metrics']['attack_success_rate'] = (
                successful_attacks / max(1, attacks_executed) * 100
            )
        
        # Log summary
        logger.info("📋 SIMULATION SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {total_time/60:.1f} minutes")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"RTUs Deployed: {len(self.rtus)}")
        logger.info(f"Power System Updates: {self.stats['power_system_updates']}")
        logger.info(f"Communication Success Rate: {self.simulation_results['performance_metrics']['communication_success_rate']:.1f}%")
        
        if self.config.enable_attacks:
            logger.info(f"Attack Success Rate: {self.simulation_results['performance_metrics']['attack_success_rate']:.1f}%")
        
        # Save results to file if requested
        if self.config.save_results:
            await self._save_results_to_file()
    
    async def _save_results_to_file(self):
        """Save simulation results to JSON file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ieee39_simulation_results_{self.config.mode.value}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.simulation_results, f, indent=2, default=str)
            
            logger.info(f"Simulation results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save simulation results: {e}")

def create_simulation_config(args) -> SimulationConfig:
    """Create simulation configuration from command line arguments"""
    
    # Parse attack scenarios
    attack_scenarios = []
    if args.attack_scenarios:
        attack_scenarios = args.attack_scenarios.split(',')
    
    config = SimulationConfig(
        mode=SimulationMode(args.mode),
        duration=args.duration,
        enable_power_system=args.mode in ['normal', 'full_cyber'] or args.power_system,
        enable_rtus=args.mode in ['normal', 'scada_only', 'attack', 'full_cyber'] or args.rtus,
        enable_scada=args.mode in ['normal', 'scada_only', 'attack', 'full_cyber'] or args.scada,
        enable_attacks=args.mode in ['attack', 'full_cyber'] or args.attacks,
        attack_scenarios=attack_scenarios,
        attack_delay=args.attack_delay,
        rtu_count=args.rtu_count,
        log_level=args.log_level,
        save_results=args.save_results
    )
    
    return config

def setup_signal_handlers(simulation):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        simulation.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IEEE 39-Bus Integrated SCADA-RTU-Attack Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simulation Modes:
  normal          : Normal SCADA-RTU operation without attacks
  scada_only      : SCADA and RTU components only
  attack          : SCADA-RTU with cybersecurity attacks
  full_cyber      : Complete cybersecurity simulation with all components

Examples:
  python ieee39_integrated.py --mode normal --duration 300
  python ieee39_integrated.py --mode attack --attack-scenarios generator_trip,voltage_manipulation
  python ieee39_integrated.py --mode full_cyber --duration 600 --rtu-count 5
        """
    )
    
    parser.add_argument('--mode', choices=['normal', 'scada_only', 'attack', 'full_cyber'],
                       default='normal', help='Simulation mode (default: normal)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Simulation duration in seconds (default: 300)')
    parser.add_argument('--attack-scenarios', type=str,
                       help='Comma-separated attack scenarios (generator_trip,voltage_manipulation,etc.)')
    parser.add_argument('--attack-delay', type=int, default=60,
                       help='Delay before starting attacks in seconds (default: 60)')
    parser.add_argument('--rtu-count', type=int, default=10,
                       help='Number of RTUs to deploy (default: 10)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--power-system', action='store_true',
                       help='Force enable power system simulation')
    parser.add_argument('--rtus', action='store_true',
                       help='Force enable RTU outstations')
    parser.add_argument('--scada', action='store_true',
                       help='Force enable SCADA master')
    parser.add_argument('--attacks', action='store_true',
                       help='Force enable attack simulation')
    parser.add_argument('--no-save', dest='save_results', action='store_false',
                       help='Do not save simulation results to file')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create simulation configuration
    config = create_simulation_config(args)
    
    # Create and run simulation
    simulation = IEEE39IntegratedSimulation(config)
    
    # Setup signal handlers
    setup_signal_handlers(simulation)
    
    try:
        await simulation.start_simulation()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
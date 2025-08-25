"""
Comprehensive Control System Integration

This module brings together all components of the control system simulation:
1. Grid + AVR (Automatic Voltage Regulation)
2. SCADA-RTU Communication
3. PMU-PDC Synchrophasor System  
4. PDC-SCADA Integration

The integration provides a realistic power system control environment with:
- Traditional SCADA monitoring and control
- Real-time synchrophasor measurements
- Coordinated voltage regulation
- Comprehensive data aggregation and analysis
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Grid and AVR Components
from grid import GridSimulation

# SCADA-RTU Components  
from scada import SCADASystemDemo

# PMU-PDC Components
from components.pmu import PhasorMeasurementUnit
from components.pdc import PhasorDataConcentrator

# PDC-SCADA Integration Components
from protocols.pdc_scada_link import get_pdc_scada_link
from components.scada_gateway import SCADAGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('control_system_integration.log')
    ]
)
logger = logging.getLogger(__name__)

class ControlSystemIntegration:
    """Comprehensive control system integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.running = False
        
        # System components
        self.grid_avr: Optional[GridAVRSimulation] = None
        self.scada_master: Optional[SCADAMaster] = None
        self.rtu_outstation: Optional[RTUOutstation] = None
        self.pmus: Dict[str, PhasorMeasurementUnit] = {}
        self.pdc: Optional[PhasorDataConcentrator] = None
        self.scada_gateway: Optional[SCADAGateway] = None
        
        # Integration tasks
        self.integration_tasks: List[asyncio.Task] = []
        
        # Statistics and monitoring
        self.system_stats = {
            'start_time': 0.0,
            'grid_updates': 0,
            'scada_transactions': 0,
            'pmu_data_points': 0,
            'pdc_aggregations': 0,
            'scada_gateway_updates': 0,
            'total_data_points_processed': 0
        }
        
        logger.info("Control System Integration initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'grid_avr': {
                'simulation_step': 0.01,  # 10ms
                'enable_disturbances': True,
                'voltage_setpoint': 1.05,
                'frequency_nominal': 60.0
            },
            'scada': {
                'master_port': 20000,
                'rtu_port': 20001,
                'polling_interval': 2.0,
                'timeout': 5.0
            },
            'pmu_pdc': {
                'pmu_count': 3,  # Start with 3 PMUs for reliable operation
                'pmu_data_rate': 20,  # 20 fps
                'pdc_aggregation_window': 200,  # 200ms window
                'pdc_port_start': 4712
            },
            'integration': {
                'pdc_scada_update_interval': 5.0,  # 5 second updates to SCADA
                'enable_cross_validation': True,
                'enable_automatic_control': False,
                'log_level': 'INFO'
            }
        }
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing Control System Integration...")
        
        # 1. Initialize Grid + AVR
        await self._initialize_grid_avr()
        
        # 2. Initialize SCADA-RTU
        await self._initialize_scada_rtu()
        
        # 3. Initialize PMU-PDC
        await self._initialize_pmu_pdc()
        
        # 4. Initialize PDC-SCADA Integration
        await self._initialize_pdc_scada_integration()
        
        logger.info("All components initialized successfully")
    
    async def _initialize_grid_avr(self):
        """Initialize Grid and AVR simulation"""
        logger.info("Initializing Grid + AVR simulation...")
        
        grid_config = self.config['grid_avr']
        self.grid_avr = GridAVRSimulation(
            simulation_step=grid_config['simulation_step'],
            enable_disturbances=grid_config['enable_disturbances']
        )
        
        # Configure initial conditions
        self.grid_avr.set_voltage_setpoint(grid_config['voltage_setpoint'])
        
        logger.info("Grid + AVR simulation initialized")
    
    async def _initialize_scada_rtu(self):
        """Initialize SCADA Master and RTU Outstation"""
        logger.info("Initializing SCADA-RTU communication...")
        
        scada_config = self.config['scada']
        
        # Initialize RTU Outstation (connects to grid)
        self.rtu_outstation = RTUOutstation(
            station_id=1,
            host='localhost',
            port=scada_config['rtu_port']
        )
        
        # Initialize SCADA Master
        self.scada_master = SCADAMaster(
            master_id='MAIN_SCADA',
            host='localhost',
            port=scada_config['master_port']
        )
        
        # Configure RTU to read from grid simulation
        if self.grid_avr:
            self.rtu_outstation.connect_to_grid(self.grid_avr)
        
        # Add RTU to SCADA master's station list
        await self.scada_master.add_station({
            'station_id': 1,
            'host': 'localhost',
            'port': scada_config['rtu_port'],
            'polling_interval': scada_config['polling_interval']
        })
        
        logger.info("SCADA-RTU communication initialized")
    
    async def _initialize_pmu_pdc(self):
        """Initialize PMU and PDC components"""
        logger.info("Initializing PMU-PDC synchrophasor system...")
        
        pmu_config = self.config['pmu_pdc']
        base_port = pmu_config['pdc_port_start']
        
        # Initialize PDC first
        self.pdc = PhasorDataConcentrator(
            concentrator_id='MAIN_PDC',
            host='localhost',
            port=base_port,
            aggregation_window_ms=pmu_config['pdc_aggregation_window']
        )
        
        # Initialize PMUs
        pmu_count = pmu_config['pmu_count']
        for i in range(pmu_count):
            pmu_id = f'PMU_{i+1}'
            pmu_port = base_port + i + 1
            
            pmu = PhasorMeasurementUnit(
                pmu_id=pmu_id,
                station_id=i+1,
                host='localhost',
                port=pmu_port,
                data_rate=pmu_config['pmu_data_rate']
            )
            
            # Connect PMU to grid for measurements
            if self.grid_avr:
                pmu.connect_to_grid(self.grid_avr, bus_id=i+1)
            
            self.pmus[pmu_id] = pmu
            
            # Register PMU with PDC
            await self.pdc.register_pmu({
                'pmu_id': pmu_id,
                'station_id': i+1,
                'host': 'localhost',
                'port': pmu_port,
                'data_rate': pmu_config['pmu_data_rate']
            })
        
        logger.info(f"PMU-PDC system initialized with {pmu_count} PMUs")
    
    async def _initialize_pdc_scada_integration(self):
        """Initialize PDC-SCADA integration"""
        logger.info("Initializing PDC-SCADA integration...")
        
        integration_config = self.config['integration']
        
        # Get PDC-SCADA link instance
        pdc_scada_link = get_pdc_scada_link('MAIN_INTEGRATION_LINK')
        
        # Initialize SCADA Gateway
        self.scada_gateway = SCADAGateway(
            gateway_id='MAIN_GATEWAY',
            pdc_link=pdc_scada_link,
            update_interval=integration_config['pdc_scada_update_interval']
        )
        
        # Connect PDC to the SCADA link
        if self.pdc:
            self.pdc.connect_scada_link(pdc_scada_link)
        
        logger.info("PDC-SCADA integration initialized")
    
    async def start(self):
        """Start the integrated control system"""
        if self.running:
            logger.warning("Control system is already running")
            return
        
        logger.info("Starting Integrated Control System...")
        self.running = True
        self.system_stats['start_time'] = time.time()
        
        # Start all components
        startup_tasks = []
        
        # Start Grid + AVR
        if self.grid_avr:
            startup_tasks.append(self.grid_avr.start())
        
        # Start SCADA-RTU
        if self.rtu_outstation:
            startup_tasks.append(self.rtu_outstation.start())
        if self.scada_master:
            startup_tasks.append(self.scada_master.start())
        
        # Start PMUs
        for pmu in self.pmus.values():
            startup_tasks.append(pmu.start())
        
        # Start PDC
        if self.pdc:
            startup_tasks.append(self.pdc.start())
        
        # Start SCADA Gateway
        if self.scada_gateway:
            startup_tasks.append(self.scada_gateway.start())
        
        # Wait for all components to start
        await asyncio.gather(*startup_tasks)
        
        # Start integration monitoring tasks
        self.integration_tasks = [
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._cross_validation_loop()),
            asyncio.create_task(self._performance_monitoring_loop())
        ]
        
        logger.info("Integrated Control System started successfully")
        await self._print_system_status()
    
    async def stop(self):
        """Stop the integrated control system"""
        if not self.running:
            return
        
        logger.info("Stopping Integrated Control System...")
        self.running = False
        
        # Cancel integration tasks
        for task in self.integration_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.integration_tasks, return_exceptions=True)
        
        # Stop all components
        stop_tasks = []
        
        if self.scada_gateway:
            stop_tasks.append(self.scada_gateway.stop())
        if self.pdc:
            stop_tasks.append(self.pdc.stop())
        for pmu in self.pmus.values():
            stop_tasks.append(pmu.stop())
        if self.scada_master:
            stop_tasks.append(self.scada_master.stop())
        if self.rtu_outstation:
            stop_tasks.append(self.rtu_outstation.stop())
        if self.grid_avr:
            stop_tasks.append(self.grid_avr.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("Integrated Control System stopped")
    
    async def _system_monitoring_loop(self):
        """Main system monitoring loop"""
        try:
            while self.running:
                # Update statistics
                await self._update_system_statistics()
                
                # Check system health
                await self._check_system_health()
                
                # Log periodic status
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    await self._print_system_status()
                
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("System monitoring loop cancelled")
    
    async def _cross_validation_loop(self):
        """Cross-validation between SCADA and PMU data"""
        try:
            while self.running:
                if not self.config['integration']['enable_cross_validation']:
                    await asyncio.sleep(5.0)
                    continue
                
                await self._perform_cross_validation()
                await asyncio.sleep(10.0)  # Cross-validate every 10 seconds
                
        except asyncio.CancelledError:
            logger.info("Cross-validation loop cancelled")
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        try:
            while self.running:
                await self._monitor_performance()
                await asyncio.sleep(5.0)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
    
    async def _update_system_statistics(self):
        """Update system statistics"""
        # Grid updates
        if self.grid_avr:
            self.system_stats['grid_updates'] = getattr(self.grid_avr, 'simulation_steps', 0)
        
        # SCADA transactions
        if self.scada_master:
            master_stats = self.scada_master.get_statistics()
            self.system_stats['scada_transactions'] = master_stats.get('total_requests', 0)
        
        # PMU data points
        total_pmu_points = 0
        for pmu in self.pmus.values():
            pmu_stats = pmu.get_statistics()
            total_pmu_points += pmu_stats.get('measurements_sent', 0)
        self.system_stats['pmu_data_points'] = total_pmu_points
        
        # PDC aggregations
        if self.pdc:
            pdc_stats = self.pdc.get_statistics()
            self.system_stats['pdc_aggregations'] = pdc_stats.get('aggregated_datasets', 0)
        
        # SCADA Gateway updates
        if self.scada_gateway:
            gateway_stats = self.scada_gateway.get_statistics()
            self.system_stats['scada_gateway_updates'] = gateway_stats.get('pdc_updates_received', 0)
        
        # Total data points
        self.system_stats['total_data_points_processed'] = (
            self.system_stats['grid_updates'] +
            self.system_stats['scada_transactions'] +
            self.system_stats['pmu_data_points'] +
            self.system_stats['pdc_aggregations'] +
            self.system_stats['scada_gateway_updates']
        )
    
    async def _check_system_health(self):
        """Check overall system health"""
        issues = []
        
        # Check Grid + AVR
        if self.grid_avr and not self.grid_avr.running:
            issues.append("Grid+AVR simulation not running")
        
        # Check SCADA
        if self.scada_master and not self.scada_master.running:
            issues.append("SCADA Master not running")
        if self.rtu_outstation and not self.rtu_outstation.running:
            issues.append("RTU Outstation not running")
        
        # Check PMUs
        offline_pmus = []
        for pmu_id, pmu in self.pmus.items():
            if not pmu.running:
                offline_pmus.append(pmu_id)
        if offline_pmus:
            issues.append(f"PMUs offline: {', '.join(offline_pmus)}")
        
        # Check PDC
        if self.pdc and not self.pdc.running:
            issues.append("PDC not running")
        
        # Check SCADA Gateway
        if self.scada_gateway and not self.scada_gateway.running:
            issues.append("SCADA Gateway not running")
        
        if issues:
            logger.warning(f"System health issues detected: {'; '.join(issues)}")
        
        return len(issues) == 0
    
    async def _perform_cross_validation(self):
        """Perform cross-validation between SCADA and PMU data"""
        try:
            # Get SCADA data
            scada_data = {}
            if self.scada_master:
                # Poll key measurements from SCADA
                scada_points = await self.scada_master.poll_station(1, ['voltage', 'frequency', 'power'])
                for point in scada_points:
                    scada_data[point['tag']] = point['value']
            
            # Get PMU data via SCADA Gateway
            pmu_data = {}
            if self.scada_gateway:
                gateway_points = self.scada_gateway.get_all_data_points()
                for tag, point in gateway_points.items():
                    if point.source_type == 'PMU':
                        pmu_data[tag] = point.value
            
            # Compare data where possible
            comparisons = []
            
            # Frequency comparison
            scada_freq = scada_data.get('frequency')
            pmu_freq = pmu_data.get('SYSTEM_FREQUENCY')
            if scada_freq and pmu_freq:
                freq_diff = abs(scada_freq - pmu_freq)
                comparisons.append(f"Frequency: SCADA={scada_freq:.3f}Hz, PMU={pmu_freq:.3f}Hz, diff={freq_diff:.3f}Hz")
            
            # Voltage comparison (if available)
            scada_voltage = scada_data.get('voltage')
            pmu_voltage = pmu_data.get('VOLTAGE_AVG')
            if scada_voltage and pmu_voltage:
                volt_diff = abs(scada_voltage - pmu_voltage)
                comparisons.append(f"Voltage: SCADA={scada_voltage:.3f}pu, PMU={pmu_voltage:.3f}pu, diff={volt_diff:.3f}pu")
            
            if comparisons:
                logger.debug(f"Cross-validation: {'; '.join(comparisons)}")
            
        except Exception as e:
            logger.error(f"Cross-validation error: {e}")
    
    async def _monitor_performance(self):
        """Monitor system performance"""
        current_time = time.time()
        
        # Calculate rates
        runtime = current_time - self.system_stats['start_time']
        if runtime > 0:
            data_rate = self.system_stats['total_data_points_processed'] / runtime
            
            # Log performance metrics
            logger.debug(f"Performance: {data_rate:.1f} data points/sec, "
                        f"Runtime: {runtime:.1f}s, "
                        f"Total points: {self.system_stats['total_data_points_processed']}")
    
    async def _print_system_status(self):
        """Print comprehensive system status"""
        current_time = time.time()
        runtime = current_time - self.system_stats['start_time']
        
        status_lines = [
            "=" * 80,
            "INTEGRATED CONTROL SYSTEM STATUS",
            "=" * 80,
            f"Runtime: {runtime:.1f} seconds",
            f"System Health: {'✓ HEALTHY' if await self._check_system_health() else '⚠ ISSUES DETECTED'}",
            "",
            "COMPONENT STATUS:",
            f"  Grid+AVR:      {'✓ Running' if self.grid_avr and self.grid_avr.running else '✗ Stopped'}",
            f"  SCADA Master:  {'✓ Running' if self.scada_master and self.scada_master.running else '✗ Stopped'}",
            f"  RTU Station:   {'✓ Running' if self.rtu_outstation and self.rtu_outstation.running else '✗ Stopped'}",
            f"  PMUs Online:   {sum(1 for pmu in self.pmus.values() if pmu.running)}/{len(self.pmus)}",
            f"  PDC:           {'✓ Running' if self.pdc and self.pdc.running else '✗ Stopped'}",
            f"  SCADA Gateway: {'✓ Running' if self.scada_gateway and self.scada_gateway.running else '✗ Stopped'}",
            "",
            "DATA FLOW STATISTICS:",
            f"  Grid Updates:       {self.system_stats['grid_updates']:,}",
            f"  SCADA Transactions: {self.system_stats['scada_transactions']:,}",
            f"  PMU Measurements:   {self.system_stats['pmu_data_points']:,}",
            f"  PDC Aggregations:   {self.system_stats['pdc_aggregations']:,}",
            f"  Gateway Updates:    {self.system_stats['scada_gateway_updates']:,}",
            f"  Total Data Points:  {self.system_stats['total_data_points_processed']:,}",
        ]
        
        # Add current measurements if available
        if self.scada_gateway:
            gateway_data = self.scada_gateway.get_all_data_points()
            if gateway_data:
                status_lines.extend([
                    "",
                    "CURRENT MEASUREMENTS:"
                ])
                
                key_points = ['SYSTEM_FREQUENCY', 'VOLTAGE_AVG', 'PMU_DATA_QUALITY', 'PMU_COUNT_ONLINE']
                for tag in key_points:
                    if tag in gateway_data:
                        point = gateway_data[tag]
                        status_lines.append(f"  {tag}: {point.value:.3f} {point.unit}")
                
                # Show active alarms
                active_alarms = self.scada_gateway.get_active_alarms()
                if active_alarms:
                    status_lines.extend([
                        "",
                        f"ACTIVE ALARMS ({len(active_alarms)}):"
                    ])
                    for alarm_key, alarm_data in active_alarms.items():
                        status_lines.append(f"  {alarm_data['severity']}: {alarm_data['description']}")
        
        status_lines.append("=" * 80)
        
        logger.info("\n" + "\n".join(status_lines))
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system_stats': self.system_stats.copy(),
            'components': {
                'grid_avr': self.grid_avr.get_status() if self.grid_avr else None,
                'scada_master': self.scada_master.get_statistics() if self.scada_master else None,
                'rtu_outstation': self.rtu_outstation.get_statistics() if self.rtu_outstation else None,
                'pdc': self.pdc.get_statistics() if self.pdc else None,
                'scada_gateway': self.scada_gateway.get_statistics() if self.scada_gateway else None,
                'pmus': {pmu_id: pmu.get_statistics() for pmu_id, pmu in self.pmus.items()}
            },
            'runtime': time.time() - self.system_stats['start_time'],
            'config': self.config
        }
        return status

# Demo function
async def run_integration_demo(duration: float = 60.0):
    """Run a comprehensive integration demonstration"""
    logger.info(f"Starting Control System Integration Demo (Duration: {duration}s)")
    
    # Create integration instance
    integration = ControlSystemIntegration()
    
    try:
        # Initialize and start
        await integration.initialize()
        await integration.start()
        
        # Run for specified duration
        await asyncio.sleep(duration)
        
        # Print final status
        logger.info("Demo completed - Final system status:")
        await integration._print_system_status()
        
        return integration.get_comprehensive_status()
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise
    finally:
        await integration.stop()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_integration_demo(120.0))  # 2 minute demo

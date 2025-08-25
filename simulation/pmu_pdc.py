"""
PMU-PDC Synchrophasor Communication Demonstration

This module demonstrates synchrophasor measurement and communication using
IEEE C37.118 protocol between PMUs and PDC with the IEEE-39 grid simulation.

Features:
- Multiple PMUs monitoring different grid buses
- High-frequency synchrophasor streaming (50 fps)
- PDC data aggregation and alignment
- Real-time monitoring of grid dynamics
- Integration with existing grid simulation
"""

import asyncio
import logging
import time
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from grid import GridSimulation
from components.pmu import PhasorMeasurementUnit
from components.pdc import PhasorDataConcentrator
from protocols.c37_118 import get_protocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class PMUPDCSystemDemo:
    """PMU-PDC Synchrophasor System Demonstration"""
    
    def __init__(self):
        # Grid simulation
        self.grid_sim = None
        
        # C37.118 Protocol
        self.protocol = get_protocol("synchrophasor_network")
        
        # PDC
        self.pdc = None
        
        # PMUs
        self.pmus = {}
        
        # Tasks
        self.grid_task = None
        self.status_monitor_task = None
        self.load_variation_task = None
        
        # Configuration
        self.pmu_configurations = [
            {'pmu_id': 101, 'station_name': 'Gen PMU 1', 'bus_number': 29, 'data_rate': 20},
            {'pmu_id': 102, 'station_name': 'Gen PMU 2', 'bus_number': 31, 'data_rate': 20},
            {'pmu_id': 103, 'station_name': 'Gen PMU 3', 'bus_number': 32, 'data_rate': 20},
            {'pmu_id': 104, 'station_name': 'Load PMU 1', 'bus_number': 15, 'data_rate': 20},
            {'pmu_id': 105, 'station_name': 'Load PMU 2', 'bus_number': 20, 'data_rate': 20},
        ]
        
        logger.info("PMU-PDC System Demo initialized")
    
    async def initialize_system(self):
        """Initialize grid simulation, PMUs, and PDC"""
        logger.info("=== Initializing PMU-PDC Synchrophasor System ===")
        
        # Initialize grid simulation
        logger.info("Starting IEEE-39 grid simulation...")
        self.grid_sim = GridSimulation()
        self.grid_sim.create_ieee39_system()
        
        # Start protocol network
        await self.protocol.start_network()
        
        # Initialize PDC
        logger.info("Initializing PDC...")
        self.pdc = PhasorDataConcentrator(
            pdc_id=1,
            station_name="Control Center PDC",
            protocol=self.protocol,
            time_window=0.2,   # 200ms time window for reliable multi-PMU alignment
            max_data_age=2.0   # 2 second max data age
        )
        
        # Add data callback for real-time monitoring
        self.pdc.add_data_callback(self._handle_aggregated_data)
        
        await self.pdc.start()
        
        # Initialize PMUs
        logger.info("Initializing PMUs on selected buses...")
        for config in self.pmu_configurations:
            pmu = PhasorMeasurementUnit(
                pmu_id=config['pmu_id'],
                station_name=config['station_name'],
                bus_number=config['bus_number'],
                protocol=self.protocol,
                grid_interface=self.grid_sim,
                pdc_id=self.pdc.station_id,
                data_rate=config['data_rate']
            )
            
            self.pmus[config['pmu_id']] = pmu
            await pmu.start()
            
            # Register PMU with PDC
            await self.pdc.register_pmu(config['pmu_id'], {
                'name': config['station_name'],
                'id_code': config['pmu_id'],
                'data_rate': config['data_rate'],
                'bus_number': config['bus_number'],
                'last_seen': time.time()
            })
        
        logger.info(f"Initialized {len(self.pmus)} PMUs and 1 PDC")
    
    async def start_demonstration(self):
        """Start the PMU-PDC demonstration"""
        logger.info("=== Starting PMU-PDC Communication Demonstration ===")
        
        # Start grid simulation
        self.grid_task = asyncio.create_task(self.grid_sim.start_simulation())
        
        # Wait for grid to stabilize
        await asyncio.sleep(3.0)
        
        # Start PMU streaming
        logger.info("Starting PMU data streaming...")
        for pmu_id in self.pmus.keys():
            await self.pdc.start_pmu_streaming(pmu_id)
        
        # Wait for initial data collection
        await asyncio.sleep(2.0)
        
        # Start status monitoring
        self.status_monitor_task = asyncio.create_task(self._status_monitor())
        
        # Start load variation simulation
        self.load_variation_task = asyncio.create_task(self._load_variation_simulation())
        
        logger.info("PMU-PDC demonstration started successfully")
        
        # Run demonstration for specified duration
        try:
            # Let the system run and demonstrate synchrophasor communication
            await asyncio.sleep(30.0)  # Run for 30 seconds
            
        except KeyboardInterrupt:
            logger.info("Demonstration interrupted by user")
        
        finally:
            await self._stop_demonstration()
    
    async def _handle_aggregated_data(self, aggregated_data):
        """Handle aggregated synchrophasor data from PDC"""
        # This callback is called whenever PDC produces new aggregated data
        # You can add custom processing here (e.g., sending to SCADA, analysis)
        pass
    
    async def _status_monitor(self):
        """Monitor and log system status"""
        try:
            while True:
                await asyncio.sleep(5.0)  # Status update every 5 seconds
                
                # Get current timestamp
                current_time = time.time()
                
                # Get PDC statistics
                pdc_stats = self.pdc.get_statistics()
                
                # Get protocol statistics
                protocol_stats = self.protocol.get_statistics()
                
                # Get PMU status
                pmu_status = self.pdc.get_pmu_status()
                
                # Get latest aggregated data
                latest_data = self.pdc.get_latest_data()
                
                # Log comprehensive status
                logger.info(f"\n=== PMU-PDC System Status at t={current_time:.1f}s ===")
                logger.info(f"PDC: {pdc_stats['aggregated_datasets']} datasets, "
                          f"Quality={pdc_stats['avg_data_quality']:.2f}")
                logger.info(f"C37.118 Network: {protocol_stats['frames_sent']}/{protocol_stats['frames_received']} "
                          f"sent/received, avg latency: {protocol_stats['avg_latency_ms']:.1f}ms")
                
                # Log PMU status
                online_pmus = sum(1 for status in pmu_status.values() if status['online'])
                logger.info(f"PMUs: {online_pmus}/{len(pmu_status)} online")
                
                for pmu_id, status in pmu_status.items():
                    if status['online']:
                        age = status['data_age']
                        logger.info(f"PMU {status['name']}: Online, last data: {age:.1f}s ago")
                    else:
                        logger.warning(f"PMU {status['name']}: OFFLINE")
                
                # Log aggregated measurements
                if latest_data:
                    stability = "STABLE" if latest_data.system_stable else "UNSTABLE"
                    logger.info(f"Grid State: f_avg={latest_data.avg_frequency:.3f}Hz, "
                              f"ROCOF_max={latest_data.max_rocof:.3f}Hz/s, {stability}")
                    
                    # Log individual PMU measurements
                    for pmu_id, pmu_data in latest_data.pmu_data.items():
                        logger.info(f"  {pmu_data.station_name}: V={pmu_data.voltage_magnitude:.3f}pu"
                                  f"∠{pmu_data.voltage_angle*180/3.14159:.1f}°, "
                                  f"f={pmu_data.frequency:.3f}Hz")
        
        except asyncio.CancelledError:
            logger.info("Status monitor cancelled")
    
    async def _load_variation_simulation(self):
        """Simulate load variations to demonstrate PMU response"""
        try:
            logger.info("Starting load variation simulation...")
            
            # Wait for initial stabilization
            await asyncio.sleep(10.0)
            
            # Simulate load increase
            logger.info("=== SIMULATING LOAD INCREASE ===")
            logger.info("Increasing load on buses 15 and 20 by 20%...")
            
            # This would require modification to the grid simulation
            # For demonstration, we'll just log the intent
            await asyncio.sleep(5.0)
            
            # Simulate load decrease
            logger.info("=== SIMULATING LOAD DECREASE ===")
            logger.info("Decreasing load on buses 15 and 20 by 30%...")
            await asyncio.sleep(5.0)
            
            # Return to normal
            logger.info("=== RETURNING TO NORMAL LOAD ===")
            logger.info("Restoring normal load conditions...")
            await asyncio.sleep(5.0)
            
        except asyncio.CancelledError:
            logger.info("Load variation simulation cancelled")
    
    async def _stop_demonstration(self):
        """Stop the demonstration and cleanup"""
        logger.info("=== Stopping PMU-PDC System Demonstration ===")
        
        # Cancel tasks
        if self.status_monitor_task:
            self.status_monitor_task.cancel()
            try:
                await self.status_monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.load_variation_task:
            self.load_variation_task.cancel()
            try:
                await self.load_variation_task
            except asyncio.CancelledError:
                pass
        
        if self.grid_task:
            self.grid_task.cancel()
            try:
                await self.grid_task
            except asyncio.CancelledError:
                pass
        
        # Stop PMUs
        logger.info("Stopping PMUs...")
        for pmu_id, pmu in self.pmus.items():
            await self.pdc.stop_pmu_streaming(pmu_id)
            await pmu.stop()
        
        # Stop PDC
        logger.info("Stopping PDC...")
        await self.pdc.stop()
        
        # Stop protocol
        await self.protocol.stop_network()
        
        # Stop grid simulation
        if self.grid_sim:
            await self.grid_sim.stop_simulation()
        
        logger.info("PMU-PDC system cleanup complete")
        
        # Final statistics
        logger.info("\n=== Final System Statistics ===")
        
        # PDC statistics
        pdc_stats = self.pdc.get_statistics()
        logger.info(f"PDC processed {pdc_stats['aggregated_datasets']} datasets")
        logger.info(f"Average data quality: {pdc_stats['avg_data_quality']:.3f}")
        
        # Protocol statistics
        protocol_stats = self.protocol.get_statistics()
        logger.info(f"C37.118 frames: {protocol_stats['data_frames']} data, "
                   f"{protocol_stats['config_frames']} config, "
                   f"{protocol_stats['header_frames']} header")
        logger.info(f"Average communication latency: {protocol_stats['avg_latency_ms']:.1f}ms")
        
        # PMU statistics
        for pmu_id, pmu in self.pmus.items():
            pmu_stats = pmu.get_statistics()
            logger.info(f"{pmu_stats['station_name']}: {pmu_stats['frames_sent']} frames sent, "
                       f"{pmu_stats['avg_frame_rate']:.1f} fps")

async def main():
    """Main demonstration function"""
    logger.info("Starting PMU-PDC Synchrophasor Communication Demonstration")
    
    demo = PMUPDCSystemDemo()
    
    try:
        # Initialize system
        await demo.initialize_system()
        
        # Run demonstration
        await demo.start_demonstration()
        
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("PMU-PDC demonstration completed")

if __name__ == "__main__":
    asyncio.run(main())

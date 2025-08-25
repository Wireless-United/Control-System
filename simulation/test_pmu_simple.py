"""
Simple PMU-PDC test to verify the system is working
"""

import asyncio
import logging
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

async def test_pmu_pdc():
    """Simple test of PMU-PDC system"""
    logger.info("=== Starting PMU-PDC Test ===")
    
    # Initialize grid
    grid_sim = GridSimulation()
    grid_sim.create_ieee39_system()
    
    # Start protocol
    protocol = get_protocol("test_network")
    await protocol.start_network()
    
    # Create PDC
    pdc = PhasorDataConcentrator(
        pdc_id=1,
        station_name="Test PDC",
        protocol=protocol,
        time_window=0.1,
        max_data_age=2.0
    )
    await pdc.start()
    
    # Create one PMU
    pmu = PhasorMeasurementUnit(
        pmu_id=101,
        station_name="Test PMU",
        bus_number=29,
        protocol=protocol,
        grid_interface=grid_sim,
        pdc_id=1,
        data_rate=10  # 10 fps for testing
    )
    
    # Register PMU with PDC
    await pdc.register_pmu(101, {
        'name': 'Test PMU',
        'id_code': 101,
        'data_rate': 10,
        'bus_number': 29
    })
    
    await pmu.start()
    
    # Start grid simulation
    grid_task = asyncio.create_task(grid_sim.start_simulation())
    
    # Wait a bit for grid to start
    await asyncio.sleep(2.0)
    
    # Start PMU streaming
    await pdc.start_pmu_streaming(101)
    
    # Let it run for 10 seconds
    logger.info("Running test for 10 seconds...")
    await asyncio.sleep(10.0)
    
    # Get statistics
    pdc_stats = pdc.get_statistics()
    protocol_stats = protocol.get_statistics()
    pmu_stats = pmu.get_statistics()
    
    logger.info(f"=== Test Results ===")
    logger.info(f"PMU sent {pmu_stats['frames_sent']} frames")
    logger.info(f"PDC processed {pdc_stats['aggregated_datasets']} datasets")
    logger.info(f"Protocol: {protocol_stats['frames_sent']} sent, {protocol_stats['frames_received']} received")
    logger.info(f"Average latency: {protocol_stats['avg_latency_ms']:.1f}ms")
    
    # Get latest data
    latest_data = pdc.get_latest_data()
    if latest_data:
        logger.info(f"Latest aggregated data:")
        logger.info(f"  Frequency: {latest_data.avg_frequency:.3f} Hz")
        logger.info(f"  PMU count: {latest_data.pmu_count}")
        logger.info(f"  Data quality: {latest_data.data_quality:.2f}")
    else:
        logger.warning("No aggregated data available")
    
    # Cleanup
    await pmu.stop()
    await pdc.stop()
    grid_task.cancel()
    try:
        await grid_task
    except asyncio.CancelledError:
        pass
    await grid_sim.stop_simulation()
    await protocol.stop_network()
    
    logger.info("=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_pmu_pdc())

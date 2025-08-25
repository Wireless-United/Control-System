"""
Test PMU-PDC system with 2 PMUs to debug aggregation issues
"""

import asyncio
import logging
import time
from grid import GridSimulation
from protocols.c37_118 import get_protocol
from components.pmu import PhasorMeasurementUnit
from components.pdc import PhasorDataConcentrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_multi_pmu():
    """Test PMU-PDC system with 2 PMUs"""
    logger.info("=== Starting Multi-PMU Test ===")
    
    # Initialize grid
    grid_sim = GridSimulation()
    grid_sim.create_ieee39_system()
    
    # Start protocol
    protocol = get_protocol("multi_test_network")
    await protocol.start_network()
    
    # Create PDC
    pdc = PhasorDataConcentrator(
        pdc_id=1,
        station_name="Multi Test PDC",
        protocol=protocol,
        time_window=0.2,  # Larger time window for better alignment
        max_data_age=2.0
    )
    await pdc.start()
    
    # Create 2 PMUs
    pmu1 = PhasorMeasurementUnit(
        pmu_id=101,
        station_name="Test PMU 1",
        bus_number=29,
        protocol=protocol,
        grid_interface=grid_sim,
        pdc_id=1,
        data_rate=10  # Lower rate for easier debugging
    )
    await pmu1.start()
    await pdc.register_pmu(101, {'name': 'Test PMU 1', 'id_code': 101, 'data_rate': 10})
    
    pmu2 = PhasorMeasurementUnit(
        pmu_id=102,
        station_name="Test PMU 2",
        bus_number=31,
        protocol=protocol,
        grid_interface=grid_sim,
        pdc_id=1,
        data_rate=10  # Lower rate for easier debugging
    )
    await pmu2.start()
    await pdc.register_pmu(102, {'name': 'Test PMU 2', 'id_code': 102, 'data_rate': 10})
    
    # Start grid simulation
    grid_task = asyncio.create_task(grid_sim.start_simulation())
    await asyncio.sleep(2.0)  # Let grid stabilize
    
    # Start PMU streaming
    await pdc.start_pmu_streaming(101)
    await pdc.start_pmu_streaming(102)
    
    logger.info("Running test for 10 seconds...")
    await asyncio.sleep(10.0)
    
    # Get results
    pmu1_stats = pmu1.get_statistics()
    pmu2_stats = pmu2.get_statistics()
    pdc_stats = pdc.get_statistics()
    protocol_stats = protocol.get_statistics()
    
    logger.info("=== Test Results ===")
    logger.info(f"PMU 1 sent {pmu1_stats['frames_sent']} frames")
    logger.info(f"PMU 2 sent {pmu2_stats['frames_sent']} frames")
    logger.info(f"PDC processed {pdc_stats['aggregated_datasets']} datasets")
    logger.info(f"Protocol: {protocol_stats['frames_sent']} sent, {protocol_stats['frames_received']} received")
    logger.info(f"Average latency: {protocol_stats['avg_latency_ms']:.1f}ms")
    
    if pdc.aggregated_data:
        logger.info("Latest aggregated data:")
        logger.info(f"  Frequency: {pdc.aggregated_data.avg_frequency:.3f} Hz")
        logger.info(f"  PMU count: {pdc.aggregated_data.pmu_count}")
        logger.info(f"  Data quality: {pdc.aggregated_data.data_quality:.2f}")
    else:
        logger.warning("No aggregated data available")
    
    # Cleanup
    await pmu1.stop()
    await pmu2.stop()
    await pdc.stop()
    grid_task.cancel()
    try:
        await grid_task
    except asyncio.CancelledError:
        pass
    await grid_sim.stop()
    await protocol.stop_network()
    
    logger.info("=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_multi_pmu())

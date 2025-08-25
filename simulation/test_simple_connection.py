"""
Simple PMU-PDC Connection Test

This test verifies basic communication between a single PMU and PDC
to identify the connection issue.
"""

import asyncio
import logging
import time

from components.pmu import PhasorMeasurementUnit
from components.pdc import PhasorDataConcentrator
from protocols.c37_118 import C37118Protocol

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGrid:
    def get_bus_measurements(self, bus_id: int) -> dict:
        return {
            'voltage_magnitude': 1.0,
            'voltage_angle': 0.0,
            'frequency': 60.0
        }

async def test_simple_connection():
    """Test simple PMU-PDC connection"""
    logger.info("Starting Simple PMU-PDC Connection Test")
    
    # Create protocol
    protocol = C37118Protocol('TEST_NET')
    
    # Create grid
    grid = SimpleGrid()
    
    # Create PDC
    pdc = PhasorDataConcentrator(
        pdc_id=1,
        station_name='TEST_PDC',
        protocol=protocol,
        time_window=0.2
    )
    
    # Create single PMU
    pmu = PhasorMeasurementUnit(
        pmu_id=2,  # Different ID from PDC
        station_name='TEST_PMU',
        bus_number=1,
        protocol=protocol,
        grid_interface=grid,
        pdc_id=1,  # Send to PDC station 1
        data_rate=5  # Slow rate for debugging
    )
    
    try:
        # Start PDC first
        logger.info("Starting PDC...")
        await pdc.start()
        
        # Register PMU manually
        await pdc.register_pmu(2, {
            'name': 'TEST_PMU',
            'id_code': 2,
            'data_rate': 5,
            'phasor_channels': 1
        })
        
        # Start PMU
        logger.info("Starting PMU...")
        await pmu.start()
        
        # Send start streaming command to PMU
        logger.info("Sending start streaming command...")
        await pdc.start_pmu_streaming(2)
        
        # Wait and monitor for 10 seconds
        logger.info("Monitoring for 10 seconds...")
        
        for i in range(10):
            await asyncio.sleep(1.0)
            
            # Check PMU stats
            pmu_stats = pmu.get_statistics()
            logger.info(f"PMU Stats: measurements={pmu_stats['measurements_taken']}, frames_sent={pmu_stats['frames_sent']}")
            
            # Check PDC stats
            pdc_stats = pdc.get_statistics()
            logger.info(f"PDC Stats: data_frames={pdc_stats['data_frames']}, registered={pdc_stats['registered_pmus']}, online={pdc_stats['online_pmus']}")
            
            # Check specific PMU status
            pmu_status = pdc.get_pmu_status()
            if 2 in pmu_status:
                status = pmu_status[2]
                logger.info(f"PMU 2 Status: online={status['online']}, last_seen={status['time_since_seen']:.1f}s ago, buffer_size={status['buffer_size']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await pmu.stop()
        await pdc.stop()

if __name__ == "__main__":
    result = asyncio.run(test_simple_connection())
    print(f"Test result: {'PASSED' if result else 'FAILED'}")

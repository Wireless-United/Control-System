"""
Test PMU Online Status Tracking

This test verifies that the PDC correctly tracks PMU online status
and shows PMUs as online when they are sending data frames.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import PMU-PDC components
from components.pmu import PhasorMeasurementUnit
from components.pdc import PhasorDataConcentrator
from protocols.c37_118 import C37118Protocol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockGrid:
    """Mock grid for PMU measurements"""
    
    def __init__(self):
        self.voltage = 1.0
        self.frequency = 60.0
        self.angle = 0.0
    
    def get_bus_measurements(self, bus_id: int) -> dict:
        """Get measurements for a specific bus"""
        # Add some variation based on bus ID
        voltage_var = 0.02 * (bus_id - 1)
        freq_var = 0.1 * (bus_id - 1)
        angle_var = 5.0 * (bus_id - 1)
        
        return {
            'voltage_magnitude': self.voltage + voltage_var,
            'voltage_angle': self.angle + angle_var,
            'frequency': self.frequency + freq_var
        }

async def test_pmu_online_status():
    """Test PMU online status tracking"""
    logger.info("Starting PMU Online Status Test")
    
    # Create protocol
    protocol = C37118Protocol('TEST_NETWORK')
    
    # Create mock grid
    grid = MockGrid()
    
    # Create PDC
    pdc = PhasorDataConcentrator(
        pdc_id=1,
        station_name='TEST_PDC',
        protocol=protocol,
        time_window=0.2,
        max_data_age=1.0
    )
    
    # Create PMUs
    pmus = {}
    pmu_count = 3
    
    for i in range(1, pmu_count + 1):
        pmu = PhasorMeasurementUnit(
            pmu_id=i,
            station_name=f'PMU_{i}',
            bus_number=i,
            protocol=protocol,
            grid_interface=grid,
            pdc_id=1,  # PDC station ID
            data_rate=20
        )
        pmus[f'PMU_{i}'] = pmu
        
        # Register PMU with PDC
        await pdc.register_pmu(i, {
            'name': f'PMU_{i}',
            'id_code': i,
            'data_rate': 20,
            'phasor_channels': 1
        })
    
    try:
        # Start PDC
        logger.info("Starting PDC...")
        await pdc.start()
        
        # Start PMUs
        logger.info("Starting PMUs...")
        for pmu_id, pmu in pmus.items():
            await pmu.start()
        
        # Monitor PMU status for 30 seconds
        test_duration = 30.0
        start_time = time.time()
        
        logger.info(f"Monitoring PMU status for {test_duration} seconds...")
        
        status_checks = []
        
        while time.time() - start_time < test_duration:
            await asyncio.sleep(2.0)  # Check every 2 seconds
            
            # Get PDC statistics
            pdc_stats = pdc.get_statistics()
            pmu_status = pdc_stats.get('pmu_status', {})
            
            runtime = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"PMU STATUS CHECK at {runtime:.1f}s")
            logger.info(f"{'='*60}")
            
            logger.info(f"PDC Stats:")
            logger.info(f"  Registered PMUs: {pdc_stats['registered_pmus']}")
            logger.info(f"  Online PMUs: {pdc_stats['online_pmus']}")
            logger.info(f"  Data frames received: {pdc_stats['data_frames']}")
            logger.info(f"  Config frames received: {pdc_stats['config_frames']}")
            
            logger.info(f"Individual PMU Status:")
            for pmu_id, status in pmu_status.items():
                online_status = "🟢 ONLINE" if status['online'] else "🔴 OFFLINE"
                logger.info(f"  PMU {pmu_id} ({status['name']}): {online_status}")
                logger.info(f"    Last seen: {status['time_since_seen']:.1f}s ago")
                logger.info(f"    Data age: {status['data_age']:.1f}s")
                logger.info(f"    Buffer size: {status['buffer_size']}")
                logger.info(f"    Sequence: {status['sequence_number']}")
            
            # Record status for analysis
            status_checks.append({
                'time': runtime,
                'registered_pmus': pdc_stats['registered_pmus'],
                'online_pmus': pdc_stats['online_pmus'],
                'data_frames': pdc_stats['data_frames'],
                'individual_status': {pmu_id: status['online'] for pmu_id, status in pmu_status.items()}
            })
            
            # Show PMU individual stats
            logger.info(f"PMU Individual Stats:")
            for pmu_id, pmu in pmus.items():
                pmu_stats = pmu.get_statistics()
                logger.info(f"  {pmu_id}: {pmu_stats['frames_sent']} frames sent, "
                          f"{pmu_stats['measurements_taken']} measurements")
        
        # Analysis
        logger.info("\n" + "="*60)
        logger.info("FINAL ANALYSIS")
        logger.info("="*60)
        
        final_stats = pdc.get_statistics()
        final_pmu_status = final_stats.get('pmu_status', {})
        
        # Calculate success metrics
        total_registered = final_stats['registered_pmus']
        final_online = final_stats['online_pmus']
        total_data_frames = final_stats['data_frames']
        
        logger.info(f"Test Results:")
        logger.info(f"  Total PMUs Registered: {total_registered}")
        logger.info(f"  PMUs Online at End: {final_online}")
        logger.info(f"  Total Data Frames Received: {total_data_frames}")
        logger.info(f"  Success Rate: {(final_online/total_registered)*100:.1f}%" if total_registered > 0 else "0%")
        
        # Check if all PMUs are showing online
        all_online = all(status['online'] for status in final_pmu_status.values())
        
        if all_online:
            logger.info("✅ SUCCESS: All PMUs are showing as ONLINE!")
            result = "PASSED"
        elif final_online > 0:
            logger.info(f"⚠️  PARTIAL: {final_online}/{total_registered} PMUs online")
            result = "PARTIAL"
        else:
            logger.info("❌ FAILED: No PMUs showing as online")
            result = "FAILED"
        
        # Show status progression
        logger.info(f"\nStatus Progression:")
        for i, check in enumerate(status_checks):
            logger.info(f"  Check {i+1} ({check['time']:.1f}s): "
                       f"{check['online_pmus']}/{check['registered_pmus']} online, "
                       f"{check['data_frames']} data frames")
        
        return {
            'result': result,
            'final_stats': final_stats,
            'status_checks': status_checks,
            'test_duration': test_duration
        }
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return {'result': 'ERROR', 'error': str(e)}
    
    finally:
        # Stop all components
        logger.info("Stopping components...")
        for pmu in pmus.values():
            await pmu.stop()
        await pdc.stop()
        logger.info("PMU Online Status Test completed")

if __name__ == "__main__":
    # Run the test
    try:
        result = asyncio.run(test_pmu_online_status())
        print(f"\n{'='*60}")
        print(f"TEST RESULT: {result['result']}")
        print(f"{'='*60}")
        
        if result['result'] == 'PASSED':
            print("🎉 PMU online status tracking is working correctly!")
        elif result['result'] == 'PARTIAL':
            print("⚠️  PMU online status tracking has some issues")
        else:
            print("❌ PMU online status tracking failed")
            
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

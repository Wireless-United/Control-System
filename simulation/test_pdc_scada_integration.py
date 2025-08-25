"""
Test script for PDC-SCADA Integration

This test demonstrates the complete PDC-SCADA integration system with:
1. PDC-SCADA communication protocol
2. SCADA Gateway functionality
3. Real-time data flow and alarms
4. Subscription-based data exchange
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import integration components
from protocols.pdc_scada_link import (
    PDCSCADALink, get_pdc_scada_link, DataSnapshot, EventNotification,
    create_data_snapshot, create_event_notification
)
from components.scada_gateway import SCADAGateway

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockPDC:
    """Mock PDC for testing the PDC-SCADA integration"""
    
    def __init__(self, pdc_id: str, scada_link: PDCSCADALink):
        self.pdc_id = pdc_id
        self.scada_link = scada_link
        self.running = False
        self.simulation_task = None
        
        # Mock PMU data
        self.pmu_data = {
            'PMU_1': {'freq': 60.0, 'voltage_mag': 1.02, 'voltage_ang': 0.0, 'quality': 0.95},
            'PMU_2': {'freq': 59.98, 'voltage_mag': 1.01, 'voltage_ang': -0.5, 'quality': 0.92},
            'PMU_3': {'freq': 60.02, 'voltage_mag': 1.03, 'voltage_ang': 0.8, 'quality': 0.98},
        }
        
        # Statistics
        self.stats = {
            'data_snapshots_sent': 0,
            'events_sent': 0,
            'start_time': 0.0
        }
    
    async def start(self):
        """Start the mock PDC"""
        if self.running:
            return
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start simulation task
        self.simulation_task = asyncio.create_task(self._simulation_loop())
        
        logger.info(f"Mock PDC {self.pdc_id} started")
    
    async def stop(self):
        """Stop the mock PDC"""
        if not self.running:
            return
        
        self.running = False
        
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Mock PDC {self.pdc_id} stopped")
    
    async def _simulation_loop(self):
        """Main simulation loop"""
        try:
            while self.running:
                # Update PMU data with some variation
                await self._update_pmu_data()
                
                # Create and send data snapshot
                await self._send_data_snapshot()
                
                # Occasionally send events
                if time.time() % 15 < 0.5:  # Every ~15 seconds
                    await self._send_random_event()
                
                await asyncio.sleep(1.0)  # Send data every second
                
        except asyncio.CancelledError:
            logger.info(f"Mock PDC {self.pdc_id}: Simulation loop cancelled")
    
    async def _update_pmu_data(self):
        """Update PMU data with realistic variations"""
        import random
        
        for pmu_id, data in self.pmu_data.items():
            # Frequency variation (±0.1 Hz)
            data['freq'] += random.uniform(-0.05, 0.05)
            data['freq'] = max(59.5, min(60.5, data['freq']))  # Clamp to reasonable range
            
            # Voltage variation (±0.02 pu)
            data['voltage_mag'] += random.uniform(-0.01, 0.01)
            data['voltage_mag'] = max(0.95, min(1.10, data['voltage_mag']))
            
            # Angle variation (±1 degree)
            data['voltage_ang'] += random.uniform(-0.5, 0.5)
            data['voltage_ang'] = max(-10.0, min(10.0, data['voltage_ang']))
            
            # Quality variation
            data['quality'] += random.uniform(-0.02, 0.02)
            data['quality'] = max(0.8, min(1.0, data['quality']))
    
    async def _send_data_snapshot(self):
        """Send a data snapshot to SCADA"""
        current_time = time.time()
        
        # Calculate summary statistics
        frequencies = [data['freq'] for data in self.pmu_data.values()]
        voltages = [data['voltage_mag'] for data in self.pmu_data.values()]
        qualities = [data['quality'] for data in self.pmu_data.values()]
        
        freq_avg = sum(frequencies) / len(frequencies)
        voltage_avg = sum(voltages) / len(voltages)
        quality_avg = sum(qualities) / len(qualities)
        
        # Create voltage profile
        voltage_profile = {}
        for i, (pmu_id, data) in enumerate(self.pmu_data.items(), 1):
            voltage_profile[i] = {  # Use integer bus IDs
                'magnitude': data['voltage_mag'],
                'angle': data['voltage_ang']
            }
        
        # Determine system status
        freq_dev = abs(freq_avg - 60.0)
        voltage_ok = all(0.95 <= v <= 1.05 for v in voltages)
        quality_ok = quality_avg > 0.9
        
        if freq_dev < 0.1 and voltage_ok and quality_ok:
            system_status = 'STABLE'
        elif freq_dev < 0.3 and quality_avg > 0.8:
            system_status = 'MARGINAL'
        else:
            system_status = 'UNSTABLE'
        
        # Create data snapshot
        grid_summary = {
            'system_frequency': freq_avg,
            'system_voltage': voltage_avg,
            'system_status': system_status
        }
        
        snapshot = create_data_snapshot(
            grid_summary=grid_summary,
            pmu_count=len(self.pmu_data),
            data_quality=quality_avg,
            voltage_profile=voltage_profile,
            frequency_summary={
                'avg': freq_avg,
                'min': min(frequencies),
                'max': max(frequencies),
                'std': (sum((f - freq_avg)**2 for f in frequencies) / len(frequencies))**0.5
            },
            system_status=system_status
        )
        
        # Send to SCADA
        await self.scada_link.send_pdc_to_scada(snapshot, sender_id=self.pdc_id, receiver_id='ALL')
        self.stats['data_snapshots_sent'] += 1
    
    async def _send_random_event(self):
        """Send a random event notification"""
        import random
        
        event_types = [
            ('FREQUENCY_DEVIATION', 'WARNING'),
            ('VOLTAGE_VIOLATION', 'WARNING'),
            ('DATA_QUALITY_LOW', 'INFO'),
            ('PMU_COMMUNICATION_LOSS', 'CRITICAL')
        ]
        
        event_type, severity = random.choice(event_types)
        current_time = time.time()
        
        # Create appropriate event data
        if event_type == 'FREQUENCY_DEVIATION':
            freq_avg = sum(data['freq'] for data in self.pmu_data.values()) / len(self.pmu_data)
            event = create_event_notification(
                event_type=event_type,
                severity=severity,
                location='SYSTEM_WIDE',
                description=f'System frequency deviation detected: {freq_avg:.3f} Hz',
                value=abs(freq_avg - 60.0),
                threshold=0.2
            )
        elif event_type == 'VOLTAGE_VIOLATION':
            voltages = [data['voltage_mag'] for data in self.pmu_data.values()]
            max_voltage = max(voltages)
            event = create_event_notification(
                event_type=event_type,
                severity=severity,
                location='BUS_SYSTEM',
                description=f'Voltage violation detected: {max_voltage:.3f} pu',
                value=max_voltage,
                threshold=1.05
            )
        else:
            event = create_event_notification(
                event_type=event_type,
                severity=severity,
                location='PMU_NETWORK',
                description=f'{event_type} event occurred',
                value=0.0,
                threshold=0.0
            )
        
        await self.scada_link.send_pdc_to_scada(event, sender_id=self.pdc_id, receiver_id='ALL')
        self.stats['events_sent'] += 1
        
        logger.info(f"Mock PDC {self.pdc_id}: Sent {severity} event - {event_type}")

async def test_pdc_scada_integration():
    """Test the complete PDC-SCADA integration"""
    logger.info("Starting PDC-SCADA Integration Test")
    
    # Get PDC-SCADA link
    pdc_link = get_pdc_scada_link('TEST_LINK')
    
    # Create mock PDC
    mock_pdc = MockPDC('TEST_PDC', pdc_link)
    
    # Create SCADA Gateway
    scada_gateway = SCADAGateway(
        gateway_id='TEST_GATEWAY',
        pdc_link=pdc_link,
        update_interval=2.0
    )
    
    try:
        # Start components
        logger.info("Starting components...")
        
        # Start the PDC-SCADA link first
        await pdc_link.start_link()
        
        await mock_pdc.start()
        await scada_gateway.start()
        
        logger.info("Integration test running... (Press Ctrl+C to stop)")
        
        # Monitor for 60 seconds
        start_time = time.time()
        test_duration = 60.0
        
        while time.time() - start_time < test_duration:
            await asyncio.sleep(5.0)
            
            # Print status every 5 seconds
            runtime = time.time() - start_time
            
            # Get statistics
            pdc_stats = mock_pdc.stats
            gateway_stats = scada_gateway.get_statistics()
            
            logger.info(f"\n" + "="*60)
            logger.info(f"PDC-SCADA INTEGRATION TEST STATUS ({runtime:.1f}s)")
            logger.info(f"="*60)
            logger.info(f"Mock PDC Stats:")
            logger.info(f"  Data snapshots sent: {pdc_stats['data_snapshots_sent']}")
            logger.info(f"  Events sent: {pdc_stats['events_sent']}")
            
            logger.info(f"SCADA Gateway Stats:")
            logger.info(f"  PDC updates received: {gateway_stats['pdc_updates_received']}")
            logger.info(f"  SCADA data points: {gateway_stats['data_points_count']}")
            logger.info(f"  Active alarms: {gateway_stats['active_alarms_count']}")
            logger.info(f"  Polling requests: {gateway_stats['polling_requests']}")
            
            # Show some key data points
            key_points = scada_gateway.poll_multiple_points([
                'SYSTEM_FREQUENCY', 'VOLTAGE_AVG', 'PMU_DATA_QUALITY'
            ])
            
            if key_points:
                logger.info(f"Current Measurements:")
                for tag, point in key_points.items():
                    logger.info(f"  {tag}: {point.value:.3f} {point.unit}")
            
            # Show active alarms
            active_alarms = scada_gateway.get_active_alarms()
            if active_alarms:
                logger.info(f"Active Alarms:")
                for alarm_key, alarm_data in active_alarms.items():
                    logger.info(f"  {alarm_data['severity']}: {alarm_data['description']}")
            
            logger.info(f"="*60)
        
        # Final test results
        logger.info("\n" + "="*60)
        logger.info("PDC-SCADA INTEGRATION TEST COMPLETED")
        logger.info("="*60)
        
        final_pdc_stats = mock_pdc.stats
        final_gateway_stats = scada_gateway.get_statistics()
        
        logger.info(f"Final Results:")
        logger.info(f"  Test Duration: {test_duration:.1f} seconds")
        logger.info(f"  PDC Data Snapshots: {final_pdc_stats['data_snapshots_sent']}")
        logger.info(f"  PDC Events: {final_pdc_stats['events_sent']}")
        logger.info(f"  Gateway Updates: {final_gateway_stats['pdc_updates_received']}")
        logger.info(f"  SCADA Data Points: {final_gateway_stats['data_points_count']}")
        logger.info(f"  Total Alarms: {final_gateway_stats['alarms_generated']}")
        
        # Calculate success rate
        expected_snapshots = int(test_duration)  # Should send ~1 per second
        success_rate = (final_pdc_stats['data_snapshots_sent'] / expected_snapshots) * 100
        
        logger.info(f"  Data Flow Success Rate: {success_rate:.1f}%")
        
        if success_rate > 80:
            logger.info("✅ PDC-SCADA Integration Test: PASSED")
        else:
            logger.warning("⚠️  PDC-SCADA Integration Test: MARGINAL")
        
        return {
            'success': success_rate > 80,
            'success_rate': success_rate,
            'pdc_stats': final_pdc_stats,
            'gateway_stats': final_gateway_stats,
            'test_duration': test_duration
        }
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {e}")
        raise
    finally:
        # Stop components
        logger.info("Stopping components...")
        await scada_gateway.stop()
        await mock_pdc.stop()
        logger.info("PDC-SCADA Integration Test completed")

if __name__ == "__main__":
    # Run the test
    try:
        result = asyncio.run(test_pdc_scada_integration())
        if result:
            print(f"\nTest Result: {'PASSED' if result['success'] else 'FAILED'}")
            print(f"Success Rate: {result['success_rate']:.1f}%")
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

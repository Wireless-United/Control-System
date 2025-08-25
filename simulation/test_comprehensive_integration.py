"""
Comprehensive Integration Test

Tests all communication layers working together:
1. Grid Simulation with IEEE 39-bus system
2. PMU-PDC Synchrophasor System
3. PDC-SCADA Integration
4. Complete data flow verification

This demonstrates the complete power system control architecture.
"""

import asyncio
import logging
import time
from typing import Dict, List

# Import existing working components
from grid import GridSimulation
from protocols.c37_118 import get_protocol
from components.pmu import PhasorMeasurementUnit
from components.pdc import PhasorDataConcentrator
from protocols.pdc_scada_link import get_pdc_scada_link
from components.scada_gateway import SCADAGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveIntegrationTest:
    """Complete integration test for all system components"""
    
    def __init__(self):
        self.grid_sim = None
        self.protocol = None
        self.pdc = None
        self.pmus = {}
        self.scada_gateway = None
        self.pdc_scada_link = None
        
        # PMU bus mapping for SCADA integration
        self.pmu_bus_mapping = {101: 29, 102: 15, 103: 35}
        
        # Test configuration
        self.config = {
            'pmu_count': 3,
            'data_rate': 10,  # 10 fps for stable operation
            'test_duration': 15,  # 15 seconds
            'pdc_time_window': 0.2,  # 200ms
            'scada_update_interval': 3.0  # 3 second SCADA updates
        }
        
        # Statistics
        self.stats = {
            'start_time': 0,
            'grid_steps': 0,
            'pmu_frames_sent': 0,
            'pdc_aggregations': 0,
            'scada_updates': 0,
            'total_data_points': 0
        }
        
        logger.info("Comprehensive Integration Test initialized")
    
    async def setup_grid_simulation(self):
        """Setup grid simulation with IEEE 39-bus system"""
        logger.info("Setting up Grid Simulation...")
        
        self.grid_sim = GridSimulation()
        self.grid_sim.create_ieee39_system()
        
        # Run initial power flow
        self.grid_sim._run_power_flow()
        
        logger.info("Grid Simulation setup complete - IEEE 39-bus system loaded")
    
    async def setup_c37_118_network(self):
        """Setup IEEE C37.118 communication network"""
        logger.info("Setting up IEEE C37.118 Communication Network...")
        
        self.protocol = get_protocol("comprehensive_test_network")
        await self.protocol.start_network()
        
        logger.info("IEEE C37.118 Network started")
    
    async def setup_pdc(self):
        """Setup Phasor Data Concentrator"""
        logger.info("Setting up PDC...")
        
        self.pdc = PhasorDataConcentrator(
            pdc_id=1,
            station_name="Integration Test PDC",
            protocol=self.protocol,
            time_window=self.config['pdc_time_window'],
            max_data_age=2.0
        )
        await self.pdc.start()
        
        logger.info("PDC setup complete")
    
    async def setup_pmus(self):
        """Setup multiple PMUs for comprehensive testing"""
        logger.info(f"Setting up {self.config['pmu_count']} PMUs...")
        
        # Strategic bus selection for diverse system coverage
        bus_numbers = [29, 15, 35]  # Different areas of IEEE 39-bus system
        
        for i in range(self.config['pmu_count']):
            pmu_id = 100 + i + 1
            bus_number = bus_numbers[i % len(bus_numbers)]
            
            pmu = PhasorMeasurementUnit(
                pmu_id=pmu_id,
                station_name=f"Integration PMU {i+1}",
                bus_number=bus_number,
                protocol=self.protocol,
                grid_interface=self.grid_sim,
                pdc_id=1,
                data_rate=self.config['data_rate']
            )
            
            await pmu.start()
            self.pmus[pmu_id] = pmu
            
            logger.info(f"PMU {pmu_id} started monitoring Bus {bus_number}")
        
        logger.info(f"All {len(self.pmus)} PMUs setup complete")
        
        # Register PMUs with PDC and start data streaming
        await asyncio.sleep(1)  # Allow PMUs to send configuration frames
        for pmu_id in self.pmus:
            await self.pdc.start_pmu_streaming(pmu_id)  # Start data streaming command
        logger.info(f"Started data streaming for all {len(self.pmus)} PMUs")
    
    async def setup_pdc_scada_integration(self):
        """Setup PDC-SCADA integration layer"""
        logger.info("Setting up PDC-SCADA Integration...")
        
        # Create PDC-SCADA communication link
        self.pdc_scada_link = get_pdc_scada_link('comprehensive_test_link')
        
        # Create SCADA Gateway
        self.scada_gateway = SCADAGateway(
            gateway_id='INTEGRATION_GATEWAY',
            pdc_link=self.pdc_scada_link,
            update_interval=self.config['scada_update_interval']
        )
        
        # Connect PDC to SCADA link via callback
        self.pdc.add_data_callback(self._pdc_to_scada_callback)
        
        await self.scada_gateway.start()
        
        logger.info("PDC-SCADA Integration setup complete")
    
    async def _pdc_to_scada_callback(self, aggregated_data):
        """Callback to convert PDC aggregated data to SCADA format"""
        try:
            from protocols.pdc_scada_link import create_data_snapshot
            
            # Extract PMU data for voltage profile
            voltage_profile = {}
            for pmu_id, pmu_data in aggregated_data.pmu_data.items():
                bus_id = self.pmu_bus_mapping.get(pmu_id, pmu_id)
                voltage_profile[bus_id] = {
                    'magnitude': pmu_data.voltage_magnitude,
                    'angle': pmu_data.voltage_angle
                }
            
            # Determine system status
            status = "NORMAL"
            if aggregated_data.avg_frequency < 59.8 or aggregated_data.avg_frequency > 60.2:
                status = "UNSTABLE"
            elif aggregated_data.data_quality < 0.8:
                status = "DEGRADED"
            
            # Create grid summary
            grid_summary = {
                'total_load_mw': 6150.0,
                'total_generation_mw': 6300.0,
                'available_generation_mw': 7000.0,
                'total_load_mvar': 1254.0
            }
            
            # Create frequency summary
            frequency_summary = {
                'nominal': 60.0,
                'current': aggregated_data.avg_frequency,
                'rocof_max': aggregated_data.max_rocof
            }
            
            # Create data snapshot
            snapshot = create_data_snapshot(
                grid_summary=grid_summary,
                pmu_count=len(aggregated_data.pmu_data),
                data_quality=aggregated_data.data_quality,
                voltage_profile=voltage_profile,
                frequency_summary=frequency_summary,
                system_status=status
            )
            
            # Send to SCADA link
            from protocols.pdc_scada_link import PDCSCADAMessage
            message = PDCSCADAMessage(
                timestamp=snapshot.timestamp,
                message_type='data_snapshot',
                sequence_number=0,
                sender_id='PDC_1',
                receiver_id='INTEGRATION_GATEWAY',
                payload=snapshot
            )
            await self.pdc_scada_link.send_pdc_to_scada(message, 'PDC_1', 'INTEGRATION_GATEWAY')
            self.stats['scada_updates'] += 1
            
        except Exception as e:
            logger.error(f"Error in PDC-SCADA callback: {e}")
    
    async def start_all_systems(self):
        """Start all system components"""
        logger.info("Starting all systems...")
        self.stats['start_time'] = time.time()
        
        # Setup in proper order
        await self.setup_grid_simulation()
        await self.setup_c37_118_network()
        await self.setup_pdc()
        await self.setup_pmus()
        await self.setup_pdc_scada_integration()
        
        logger.info("All systems started successfully!")
        await self.print_system_status()
    
    async def run_integration_test(self):
        """Run the complete integration test"""
        logger.info(f"Running integration test for {self.config['test_duration']} seconds...")
        
        start_time = time.time()
        last_status_time = start_time
        status_interval = 15  # Print status every 15 seconds
        
        while (time.time() - start_time) < self.config['test_duration']:
            # Update statistics
            await self.update_statistics()
            
            # Print status periodically
            if (time.time() - last_status_time) >= status_interval:
                await self.print_system_status()
                last_status_time = time.time()
            
            # Run grid simulation step
            if self.grid_sim:
                self.grid_sim._run_power_flow()
                self.stats['grid_steps'] += 1
            
            await asyncio.sleep(1.0)  # 1 second intervals
        
        logger.info("Integration test completed!")
        await self.print_final_results()
    
    async def update_statistics(self):
        """Update test statistics"""
        # PMU statistics
        total_pmu_frames = 0
        for pmu in self.pmus.values():
            pmu_stats = pmu.get_statistics()
            total_pmu_frames += pmu_stats.get('frames_sent', 0)
        self.stats['pmu_frames_sent'] = total_pmu_frames
        
        # PDC statistics
        if self.pdc:
            pdc_stats = self.pdc.get_statistics()
            self.stats['pdc_aggregations'] = pdc_stats.get('aggregated_datasets', 0)
        
        # SCADA Gateway statistics
        if self.scada_gateway:
            gateway_stats = self.scada_gateway.get_statistics()
            self.stats['scada_updates'] = gateway_stats.get('pdc_updates_received', 0)
        
        # Total data points processed
        self.stats['total_data_points'] = (
            self.stats['grid_steps'] +
            self.stats['pmu_frames_sent'] +
            self.stats['pdc_aggregations'] +
            self.stats['scada_updates']
        )
    
    async def print_system_status(self):
        """Print comprehensive system status"""
        runtime = time.time() - self.stats['start_time']
        
        status_lines = [
            "=" * 80,
            "COMPREHENSIVE INTEGRATION TEST STATUS",
            "=" * 80,
            f"Runtime: {runtime:.1f} seconds",
            "",
            "SYSTEM COMPONENTS:",
            f"  Grid Simulation:   {'✓ Running' if self.grid_sim else '✗ Not started'}",
            f"  C37.118 Network:   {'✓ Running' if self.protocol and self.protocol.running else '✗ Not started'}",
            f"  PDC:               {'✓ Running' if self.pdc and self.pdc.running else '✗ Not started'}",
            f"  PMUs:              {sum(1 for pmu in self.pmus.values() if pmu.running)}/{len(self.pmus)} online",
            f"  SCADA Gateway:     {'✓ Running' if self.scada_gateway and self.scada_gateway.running else '✗ Not started'}",
            "",
            "DATA FLOW STATISTICS:",
            f"  Grid Simulation Steps:  {self.stats['grid_steps']:,}",
            f"  PMU Frames Sent:        {self.stats['pmu_frames_sent']:,}",
            f"  PDC Aggregations:       {self.stats['pdc_aggregations']:,}",
            f"  SCADA Updates:          {self.stats['scada_updates']:,}",
            f"  Total Data Points:      {self.stats['total_data_points']:,}",
        ]
        
        # Add PDC status if available
        if self.pdc:
            pdc_stats = self.pdc.get_statistics()
            status_lines.extend([
                "",
                "PDC DETAILED STATUS:",
                f"  Registered PMUs:        {pdc_stats.get('registered_pmus', 0)}",
                f"  Online PMUs:            {pdc_stats.get('online_pmus', 0)}",
                f"  Data Frames Received:   {pdc_stats.get('data_frames', 0)}",
                f"  Average Data Quality:   {pdc_stats.get('avg_data_quality', 0):.3f}",
            ])
        
        # Add SCADA Gateway status if available
        if self.scada_gateway:
            gateway_stats = self.scada_gateway.get_statistics()
            scada_data = self.scada_gateway.get_all_data_points()
            
            status_lines.extend([
                "",
                "SCADA GATEWAY STATUS:",
                f"  Data Points Available:  {gateway_stats.get('data_points_count', 0)}",
                f"  Active Alarms:          {gateway_stats.get('active_alarms_count', 0)}",
                f"  Polling Requests:       {gateway_stats.get('polling_requests', 0)}",
            ])
            
            # Show key measurements
            if scada_data:
                key_measurements = ['SYSTEM_FREQUENCY', 'VOLTAGE_AVG', 'PMU_DATA_QUALITY']
                status_lines.append("")
                status_lines.append("CURRENT MEASUREMENTS:")
                for tag in key_measurements:
                    if tag in scada_data:
                        point = scada_data[tag]
                        status_lines.append(f"  {tag}: {point.value:.3f} {point.unit}")
        
        status_lines.append("=" * 80)
        
        logger.info("\n" + "\n".join(status_lines))
    
    async def print_final_results(self):
        """Print final test results"""
        runtime = time.time() - self.stats['start_time']
        
        results = [
            "=" * 80,
            "COMPREHENSIVE INTEGRATION TEST RESULTS",
            "=" * 80,
            f"Test Duration: {runtime:.1f} seconds",
            "",
            "FINAL STATISTICS:",
            f"  Grid Simulation Steps:  {self.stats['grid_steps']:,}",
            f"  PMU Frames Sent:        {self.stats['pmu_frames_sent']:,}",
            f"  PDC Aggregations:       {self.stats['pdc_aggregations']:,}",
            f"  SCADA Updates:          {self.stats['scada_updates']:,}",
            f"  Total Data Points:      {self.stats['total_data_points']:,}",
            "",
            "DATA FLOW RATES:",
            f"  Grid Steps/sec:         {self.stats['grid_steps']/runtime:.1f}",
            f"  PMU Frames/sec:         {self.stats['pmu_frames_sent']/runtime:.1f}",
            f"  PDC Aggregations/min:   {self.stats['pdc_aggregations']*60/runtime:.1f}",
            f"  SCADA Updates/min:      {self.stats['scada_updates']*60/runtime:.1f}",
            "",
            "SYSTEM HEALTH:"
        ]
        
        # Check system health
        health_issues = []
        
        if self.stats['pmu_frames_sent'] == 0:
            health_issues.append("No PMU frames sent")
        if self.stats['pdc_aggregations'] == 0:
            health_issues.append("No PDC aggregations")
        if self.stats['scada_updates'] == 0:
            health_issues.append("No SCADA updates")
        
        if not health_issues:
            results.append("  ✅ All systems operating normally")
            results.append("  ✅ Complete data flow established")
            results.append("  ✅ Integration test PASSED")
        else:
            results.append("  ⚠️  Issues detected:")
            for issue in health_issues:
                results.append(f"     - {issue}")
            results.append("  ⚠️  Integration test PARTIAL")
        
        results.append("=" * 80)
        
        logger.info("\n" + "\n".join(results))
        
        return len(health_issues) == 0
    
    async def stop_all_systems(self):
        """Stop all system components"""
        logger.info("Stopping all systems...")
        
        # Stop in reverse order
        if self.scada_gateway:
            await self.scada_gateway.stop()
        
        for pmu in self.pmus.values():
            await pmu.stop()
        
        if self.pdc:
            await self.pdc.stop()
        
        if self.protocol:
            await self.protocol.stop_network()
        
        logger.info("All systems stopped")

async def run_comprehensive_test():
    """Run the comprehensive integration test"""
    test = ComprehensiveIntegrationTest()
    
    try:
        # Start all systems
        await test.start_all_systems()
        
        # Run integration test
        await test.run_integration_test()
        
        # Return test success status
        return await test.print_final_results()
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await test.stop_all_systems()

if __name__ == "__main__":
    # Run comprehensive integration test
    logger.info("Starting Comprehensive Control System Integration Test")
    
    try:
        success = asyncio.run(run_comprehensive_test())
        if success:
            print("\n🎉 COMPREHENSIVE INTEGRATION TEST: PASSED")
            print("✅ All communication layers and controllers working together!")
        else:
            print("\n⚠️  COMPREHENSIVE INTEGRATION TEST: PARTIAL SUCCESS")
            print("Some components may need attention")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

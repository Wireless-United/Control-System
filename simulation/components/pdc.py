"""
Phasor Data Concentrator (PDC) Implementation

This module implements a PDC that collects synchrophasor data from multiple PMUs,
aligns timestamps, and provides aggregated grid state information.

Features:
- Multi-PMU data collection
- Timestamp alignment and synchronization
- Data aggregation and analysis
- Interface to downstream applications (SCADA)
- Real-time monitoring and logging
"""

import asyncio
import time
import logging
import math
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

from protocols.c37_118 import (
    C37118Station, C37118Protocol, C37118DataFrame, C37118ConfigFrame,
    C37118HeaderFrame, C37118CommandFrame,
    CMD_TURN_ON_TRANSMISSION, CMD_TURN_OFF_TRANSMISSION, CMD_SEND_CFG2
)

logger = logging.getLogger(__name__)

@dataclass
class PMUData:
    """PMU measurement data with metadata"""
    pmu_id: int
    station_name: str
    timestamp: float
    voltage_magnitude: float
    voltage_angle: float
    frequency: float
    rocof: float
    analogs: List[float]
    digitals: List[int]
    latency: float
    sequence_number: int

@dataclass
class AggregatedData:
    """Aggregated synchrophasor data from multiple PMUs"""
    timestamp: float
    pmu_count: int
    pmu_data: Dict[int, PMUData]
    avg_frequency: float
    max_rocof: float
    voltage_profile: Dict[int, tuple]  # bus_number -> (magnitude, angle)
    system_stable: bool
    data_quality: float

class PhasorDataConcentrator(C37118Station):
    """Phasor Data Concentrator for synchrophasor data aggregation"""
    
    def __init__(self, pdc_id: int, station_name: str, protocol: C37118Protocol,
                 time_window: float = 0.1, max_data_age: float = 1.0):
        """
        Initialize PDC
        
        Args:
            pdc_id: Unique PDC identifier
            station_name: PDC station name
            protocol: C37.118 protocol instance
            time_window: Time window for data alignment (seconds)
            max_data_age: Maximum age for valid data (seconds)
        """
        super().__init__(pdc_id, protocol)
        
        self.station_name = station_name
        self.time_window = time_window
        self.max_data_age = max_data_age
        
        # PMU management
        self.registered_pmus: Dict[int, dict] = {}
        self.pmu_data_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.pmu_configurations: Dict[int, Any] = {}
        self.pmu_headers: Dict[int, str] = {}
        
        # Data aggregation
        self.aggregated_data: Optional[AggregatedData] = None
        self.data_callbacks: List[Callable] = []
        self.aggregation_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'data_frames': 0,
            'config_frames': 0,
            'header_frames': 0,
            'command_frames': 0,
            'aggregated_datasets': 0,
            'start_time': time.time(),
            'last_aggregation_time': 0,
            'data_quality_history': deque(maxlen=100)
        }
        
        # Sequence tracking
        self.sequence_counters: Dict[int, int] = defaultdict(int)
        
        logger.info(f"PDC {pdc_id} ({station_name}) initialized")
    
    async def start(self):
        """Start PDC operations"""
        await super().start()
        
        # Start aggregation task
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        logger.info(f"PDC {self.station_id} started with time window {self.time_window}s")
    
    async def stop(self):
        """Stop PDC operations"""
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        logger.info(f"PDC {self.station_name} stopped")
    
    async def _handle_frame(self, frame, sender_id: int, latency: float):
        """Handle incoming C37.118 frames from PMUs"""
        self.stats['frames_received'] += 1
        
        # Add debug logging
        logger.debug(f"PDC {self.station_id}: Received frame from sender {sender_id}, type: {type(frame).__name__}")
        
        if isinstance(frame, C37118DataFrame):
            await self._handle_data_frame(frame, sender_id, latency)
            self.stats['data_frames'] += 1
        
        elif isinstance(frame, C37118ConfigFrame):
            await self._handle_config_frame(frame, sender_id)
            self.stats['config_frames'] += 1
        
        elif isinstance(frame, C37118HeaderFrame):
            await self._handle_header_frame(frame, sender_id)
            self.stats['header_frames'] += 1
        
        elif isinstance(frame, C37118CommandFrame):
            await self._handle_command_frame(frame, sender_id)
            self.stats['command_frames'] += 1
    
    async def _handle_data_frame(self, frame: C37118DataFrame, sender_id: int, latency: float):
        """Handle data frame from PMU"""
        logger.debug(f"PDC {self.station_id}: Processing data frame from PMU {sender_id}")
        try:
            # Extract measurement data
            if hasattr(frame, 'phasors') and frame.phasors and len(frame.phasors) > 0:
                phasor = frame.phasors[0]
                voltage_magnitude = phasor.magnitude
                voltage_angle = phasor.angle
            else:
                voltage_magnitude = 0.0
                voltage_angle = 0.0
            
            if hasattr(frame, 'frequency') and frame.frequency:
                frequency = frame.frequency.frequency
                rocof = frame.frequency.rocof
            else:
                frequency = 60.0
                rocof = 0.0
            
            analogs = [analog.value for analog in frame.analogs] if hasattr(frame, 'analogs') and frame.analogs else []
            digitals = [digital.value for digital in frame.digitals] if hasattr(frame, 'digitals') and frame.digitals else []
            
            # Create PMU data record
            pmu_data = PMUData(
                pmu_id=sender_id,
                station_name=self.registered_pmus.get(sender_id, {}).get('name', f'PMU-{sender_id}'),
                timestamp=frame.timestamp,
                voltage_magnitude=voltage_magnitude,
                voltage_angle=voltage_angle,
                frequency=frequency,
                rocof=rocof,
                analogs=analogs,
                digitals=digitals,
                latency=latency,
                sequence_number=self.sequence_counters[sender_id]
            )
            
            # Add to buffer
            self.pmu_data_buffers[sender_id].append(pmu_data)
            self.sequence_counters[sender_id] += 1
            
            logger.debug(f"PDC {self.station_id}: Buffered data from PMU {sender_id} - "
                        f"V={voltage_magnitude:.3f}pu, f={frequency:.3f}Hz, buffer size={len(self.pmu_data_buffers[sender_id])}")
        
        except Exception as e:
            logger.error(f"PDC {self.station_id}: Error handling data frame from {sender_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_config_frame(self, frame: C37118ConfigFrame, sender_id: int):
        """Handle configuration frame from PMU"""
        self.pmu_configurations[sender_id] = frame.config
        
        # Register PMU if not already registered
        if sender_id not in self.registered_pmus:
            self.registered_pmus[sender_id] = {
                'name': frame.config.station_name,
                'id_code': frame.config.id_code,
                'data_rate': frame.config.data_rate,
                'phasor_channels': frame.config.phasor_channels,
                'last_seen': time.time()
            }
        
        logger.info(f"PDC {self.station_id}: Registered PMU {sender_id} ({frame.config.station_name})")
    
    async def _handle_header_frame(self, frame: C37118HeaderFrame, sender_id: int):
        """Handle header frame from PMU"""
        self.pmu_headers[sender_id] = frame.header_data
        logger.info(f"PDC {self.station_id}: Received header from PMU {sender_id}: {frame.header_data}")
    
    async def _handle_command_frame(self, frame: C37118CommandFrame, sender_id: int):
        """Handle command frame from PMU"""
        logger.info(f"PDC {self.station_id}: Received command {frame.command} from PMU {sender_id}")
    
    async def _aggregation_loop(self):
        """Main data aggregation loop"""
        try:
            while self.running:
                # Perform data aggregation
                aggregated = await self._aggregate_data()
                
                if aggregated:
                    self.aggregated_data = aggregated
                    self.stats['aggregated_datasets'] += 1
                    self.stats['last_aggregation_time'] = time.time()
                    
                    # Notify callbacks
                    for callback in self.data_callbacks:
                        try:
                            await callback(aggregated)
                        except Exception as e:
                            logger.error(f"PDC {self.station_id}: Callback error: {e}")
                    
                    # Log aggregated data
                    await self._log_aggregated_data(aggregated)
                
                # Wait for next aggregation cycle (run faster than time window)
                await asyncio.sleep(self.time_window / 2)  # Run at 2x the time window frequency
        
        except asyncio.CancelledError:
            logger.info(f"PDC {self.station_id}: Aggregation loop cancelled")
    
    async def _aggregate_data(self) -> Optional[AggregatedData]:
        """Aggregate data from all PMUs within time window"""
        current_time = time.time()
        target_time = current_time - self.time_window / 2  # Target time for alignment
        
        aligned_data: Dict[int, PMUData] = {}
        frequencies = []
        rocofs = []
        voltage_profile = {}
        
        logger.debug(f"PDC {self.station_id}: Aggregating data - registered PMUs: {list(self.registered_pmus.keys())}, buffer sizes: {[(pmu_id, len(buf)) for pmu_id, buf in self.pmu_data_buffers.items()]}")
        
        # Find best data for each PMU within time window
        for pmu_id, data_buffer in self.pmu_data_buffers.items():
            if not data_buffer:
                logger.debug(f"PDC {self.station_id}: No data in buffer for PMU {pmu_id}")
                continue
            
            # Find data closest to target time within window
            best_data = None
            min_time_diff = float('inf')
            
            for data in data_buffer:
                if current_time - data.timestamp > self.max_data_age:
                    continue  # Data too old
                
                time_diff = abs(data.timestamp - target_time)
                if time_diff < min_time_diff and time_diff <= self.time_window:
                    min_time_diff = time_diff
                    best_data = data
            
            if best_data:
                aligned_data[pmu_id] = best_data
                frequencies.append(best_data.frequency)
                rocofs.append(abs(best_data.rocof))
                
                # Extract bus number from PMU configuration (simplified)
                bus_number = pmu_id * 10  # Placeholder mapping
                voltage_profile[bus_number] = (best_data.voltage_magnitude, best_data.voltage_angle)
                
                logger.debug(f"PDC {self.station_id}: Found aligned data for PMU {pmu_id}, time_diff={min_time_diff:.3f}s")
            else:
                logger.debug(f"PDC {self.station_id}: No valid data found for PMU {pmu_id} within time window")
        
        if not aligned_data:
            logger.debug(f"PDC {self.station_id}: No aligned data available for aggregation")
            return None
        
        # Calculate aggregated metrics
        avg_frequency = statistics.mean(frequencies) if frequencies else 60.0
        max_rocof = max(rocofs) if rocofs else 0.0
        
        # Assess system stability
        frequency_deviation = abs(avg_frequency - 60.0)
        rocof_threshold = 0.5  # Hz/s
        system_stable = frequency_deviation < 0.5 and max_rocof < rocof_threshold
        
        # Calculate data quality based on PMU count and timing alignment
        expected_pmus = len(self.registered_pmus)
        data_quality = len(aligned_data) / max(1, expected_pmus) if expected_pmus > 0 else 1.0
        
        # Track data quality history
        self.stats['data_quality_history'].append(data_quality)
        
        return AggregatedData(
            timestamp=target_time,
            pmu_count=len(aligned_data),
            pmu_data=aligned_data,
            avg_frequency=avg_frequency,
            max_rocof=max_rocof,
            voltage_profile=voltage_profile,
            system_stable=system_stable,
            data_quality=data_quality
        )
    
    async def _log_aggregated_data(self, data: AggregatedData):
        """Log aggregated synchrophasor data"""
        stability_status = "STABLE" if data.system_stable else "UNSTABLE"
        
        logger.info(f"PDC {self.station_id}: Aggregated data from {data.pmu_count} PMUs - "
                   f"f_avg={data.avg_frequency:.3f}Hz, ROCOF_max={data.max_rocof:.3f}Hz/s, "
                   f"Quality={data.data_quality:.2f}, Status={stability_status}")
        
        # Log individual PMU data
        for pmu_id, pmu_data in data.pmu_data.items():
            logger.info(f"PDC {self.station_id}: {pmu_data.station_name} - "
                        f"V={pmu_data.voltage_magnitude:.3f}pu∠{math.degrees(pmu_data.voltage_angle):.1f}°, "
                        f"f={pmu_data.frequency:.3f}Hz, latency={pmu_data.latency:.1f}ms")
    
    async def register_pmu(self, pmu_id: int, pmu_info: dict):
        """Manually register a PMU"""
        self.registered_pmus[pmu_id] = pmu_info
        logger.info(f"PDC {self.station_id}: Manually registered PMU {pmu_id}")
    
    async def start_pmu_streaming(self, pmu_id: int):
        """Send start streaming command to PMU"""
        command_frame = C37118CommandFrame(
            id_code=self.station_id,
            timestamp=time.time(),
            command=CMD_TURN_ON_TRANSMISSION,
            frame_size=0
        )
        
        await self.send_frame(command_frame, pmu_id)
        logger.info(f"PDC {self.station_id}: Sent start streaming command to PMU {pmu_id}")
    
    async def stop_pmu_streaming(self, pmu_id: int):
        """Send stop streaming command to PMU"""
        command_frame = C37118CommandFrame(
            id_code=self.station_id,
            timestamp=time.time(),
            command=CMD_TURN_OFF_TRANSMISSION,
            frame_size=0
        )
        
        await self.send_frame(command_frame, pmu_id)
        logger.info(f"PDC {self.station_id}: Sent stop streaming command to PMU {pmu_id}")
    
    async def request_pmu_config(self, pmu_id: int):
        """Request configuration from PMU"""
        command_frame = C37118CommandFrame(
            id_code=self.station_id,
            timestamp=time.time(),
            command=CMD_SEND_CFG2,
            frame_size=0
        )
        
        await self.send_frame(command_frame, pmu_id)
        logger.info(f"PDC {self.station_id}: Requested configuration from PMU {pmu_id}")
    
    def add_data_callback(self, callback: Callable):
        """Add callback for aggregated data updates"""
        self.data_callbacks.append(callback)
    
    def get_latest_data(self) -> Optional[AggregatedData]:
        """Get latest aggregated data"""
        return self.aggregated_data
    
    def get_pmu_status(self) -> Dict[int, dict]:
        """Get status of all registered PMUs"""
        current_time = time.time()
        status = {}
        
        for pmu_id, pmu_info in self.registered_pmus.items():
            # Get latest data from buffer
            latest_data = None
            if pmu_id in self.pmu_data_buffers and self.pmu_data_buffers[pmu_id]:
                latest_data = self.pmu_data_buffers[pmu_id][-1]
            
            status[pmu_id] = {
                'name': pmu_info['name'],
                'online': latest_data is not None and (current_time - latest_data.timestamp) < 2.0,
                'last_data_time': latest_data.timestamp if latest_data else 0,
                'data_age': current_time - latest_data.timestamp if latest_data else float('inf'),
                'sequence_number': latest_data.sequence_number if latest_data else 0,
                'buffer_size': len(self.pmu_data_buffers[pmu_id]) if pmu_id in self.pmu_data_buffers else 0
            }
        
        return status
    
    def get_statistics(self) -> dict:
        """Get PDC statistics"""
        runtime = time.time() - self.stats['start_time']
        avg_data_quality = (statistics.mean(self.stats['data_quality_history']) 
                          if self.stats['data_quality_history'] else 0.0)
        
        return {
            'station_id': self.station_id,
            'station_name': self.station_name,
            'registered_pmus': len(self.registered_pmus),
            'frames_received': self.stats['frames_received'],
            'data_frames': self.stats['data_frames'],
            'config_frames': self.stats['config_frames'],
            'header_frames': self.stats['header_frames'],
            'command_frames': self.stats['command_frames'],
            'aggregated_datasets': self.stats['aggregated_datasets'],
            'runtime_seconds': round(runtime, 1),
            'avg_data_quality': round(avg_data_quality, 3),
            'time_window': self.time_window,
            'max_data_age': self.max_data_age,
            'last_aggregation_age': round(time.time() - self.stats['last_aggregation_time'], 2)
        }

"""
Phasor Measurement Unit (PMU) Implementation

This module implements a PMU that generates synchrophasor data from grid measurements
and streams it using IEEE C37.118 protocol to a PDC.

Features:
- Voltage magnitude and angle measurement
- Frequency and ROCOF calculation
- High-frequency data streaming (50 fps)
- IEEE C37.118 frame generation
- Real-time grid state monitoring
"""

import asyncio
import time
import math
import logging
from typing import Optional, Callable, Any
import numpy as np

from protocols.c37_118 import (
    C37118Station, C37118Protocol, C37118DataFrame, C37118ConfigFrame, 
    C37118HeaderFrame, C37118CommandFrame,
    PhasorData, FrequencyData, AnalogData, DigitalData, PMUConfiguration,
    CMD_TURN_ON_TRANSMISSION, CMD_TURN_OFF_TRANSMISSION, CMD_SEND_CFG2,
    PHASOR_COMPLEX_FLOAT
)

logger = logging.getLogger(__name__)

class PhasorMeasurementUnit(C37118Station):
    """Phasor Measurement Unit for synchrophasor measurements"""
    
    def __init__(self, pmu_id: int, station_name: str, bus_number: int, 
                 protocol: C37118Protocol, grid_interface: Any,
                 pdc_id: int, data_rate: int = 50):
        """
        Initialize PMU
        
        Args:
            pmu_id: Unique PMU identifier
            station_name: PMU station name
            bus_number: Grid bus number to monitor
            protocol: C37.118 protocol instance
            grid_interface: Grid simulation interface
            pdc_id: PDC station ID to send data to
            data_rate: Data frames per second (default 50 fps)
        """
        super().__init__(pmu_id, protocol)
        
        self.station_name = station_name
        self.bus_number = bus_number
        self.grid_interface = grid_interface
        self.pdc_id = pdc_id
        self.data_rate = data_rate
        self.streaming = False
        
        # Measurement history for frequency calculation
        self.voltage_history = []
        self.frequency_history = []
        self.time_history = []
        self.last_frequency = 60.0  # Nominal frequency
        self.last_rocof = 0.0
        
        # PMU Configuration
        self.configuration = PMUConfiguration(
            station_name=station_name,
            id_code=pmu_id,
            data_format=PHASOR_COMPLEX_FLOAT,
            phasor_channels=1,  # Voltage phasor
            analog_channels=2,  # Frequency, ROCOF
            digital_channels=1,  # Status
            nominal_frequency=60.0,
            configuration_count=1,
            data_rate=data_rate,
            phasor_names=[f"V{bus_number}"],
            analog_names=["FREQ", "ROCOF"],
            digital_names=["STATUS"]
        )
        
        # Tasks
        self.streaming_task: Optional[asyncio.Task] = None
        self.measurement_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'frames_sent': 0,
            'measurements_taken': 0,
            'start_time': time.time(),
            'last_measurement_time': 0
        }
        
        logger.info(f"PMU {pmu_id} ({station_name}) initialized for bus {bus_number}")
    
    async def start(self):
        """Start PMU operations"""
        await super().start()
        
        # Send configuration frame
        await self._send_configuration_frame()
        
        # Send header frame
        await self._send_header_frame()
        
        # Start measurement task
        self.measurement_task = asyncio.create_task(self._measurement_loop())
        
        logger.info(f"PMU {self.station_id} started streaming at {self.data_rate} fps")
    
    async def stop(self):
        """Stop PMU operations"""
        self.streaming = False
        
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        
        if self.measurement_task:
            self.measurement_task.cancel()
            try:
                await self.measurement_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        logger.info(f"PMU {self.station_name} stopped")
    
    async def _handle_frame(self, frame, sender_id: int, latency: float):
        """Handle incoming C37.118 frames"""
        if hasattr(frame, 'command'):
            # Handle command frames
            if frame.command == CMD_TURN_ON_TRANSMISSION:
                await self._start_streaming()
            elif frame.command == CMD_TURN_OFF_TRANSMISSION:
                await self._stop_streaming()
            elif frame.command == CMD_SEND_CFG2:
                await self._send_configuration_frame()
            
            logger.info(f"PMU {self.station_id}: Received command {frame.command} from {sender_id}")
    
    async def _measurement_loop(self):
        """Continuous measurement loop"""
        try:
            while self.running:
                # Take measurement
                measurement = await self._take_measurement()
                
                if measurement and self.streaming:
                    # Create and send data frame
                    data_frame = await self._create_data_frame(measurement)
                    await self.send_frame(data_frame, self.pdc_id)
                    
                    self.stats['frames_sent'] += 1
                    logger.debug(f"PMU {self.station_id}: Sent data frame - "
                               f"V={measurement['voltage_mag']:.3f}pu∠{math.degrees(measurement['voltage_angle']):.1f}°, "
                               f"f={measurement['frequency']:.3f}Hz")
                
                # Wait for next measurement interval
                await asyncio.sleep(1.0 / self.data_rate)
        
        except asyncio.CancelledError:
            logger.info(f"PMU {self.station_id}: Measurement loop cancelled")
    
    async def _take_measurement(self) -> Optional[dict]:
        """Take synchrophasor measurement from grid"""
        try:
            # Get grid state for this bus
            if hasattr(self.grid_interface, 'get_bus_voltage'):
                voltage_pu, voltage_angle_deg = self.grid_interface.get_bus_voltage(self.bus_number)
                voltage_complex = voltage_pu * np.exp(1j * np.radians(voltage_angle_deg))
            else:
                # Fallback for grid simulation interface
                voltage_complex = 1.0 + 0j  # Default
            
            # Extract magnitude and angle
            voltage_mag = abs(voltage_complex)
            voltage_angle = np.angle(voltage_complex)
            
            # Calculate frequency and ROCOF
            current_time = time.time()
            frequency, rocof = self._calculate_frequency(voltage_angle, current_time)
            
            measurement = {
                'timestamp': current_time,
                'voltage_mag': voltage_mag,
                'voltage_angle': voltage_angle,
                'frequency': frequency,
                'rocof': rocof,
                'bus_number': self.bus_number
            }
            
            self.stats['measurements_taken'] += 1
            self.stats['last_measurement_time'] = current_time
            
            return measurement
        
        except Exception as e:
            logger.error(f"PMU {self.station_id}: Measurement error: {e}")
            return None
    
    def _calculate_frequency(self, voltage_angle: float, timestamp: float) -> tuple:
        """Calculate frequency and ROCOF from voltage angle measurements"""
        # Add current measurement to history
        self.voltage_history.append(voltage_angle)
        self.time_history.append(timestamp)
        
        # Keep only recent measurements (1 second window)
        window_size = min(50, len(self.voltage_history))  # Up to 1 second at 50 fps
        if len(self.voltage_history) > window_size:
            self.voltage_history = self.voltage_history[-window_size:]
            self.time_history = self.time_history[-window_size:]
        
        if len(self.voltage_history) < 3:
            return self.last_frequency, self.last_rocof
        
        try:
            # Calculate frequency from phase angle derivative
            # f = (1/2π) * dθ/dt + f_nominal
            
            # Use linear regression for robust frequency estimation
            times = np.array(self.time_history)
            angles = np.array(self.voltage_history)
            
            # Unwrap phase angles to handle wraparound
            angles = np.unwrap(angles)
            
            # Calculate frequency from slope
            if len(times) > 1:
                # Linear fit to get dθ/dt
                coeffs = np.polyfit(times - times[0], angles, 1)
                angle_rate = coeffs[0]  # dθ/dt in rad/s
                
                # Convert to frequency: f = f_nominal + (1/2π) * dθ/dt
                frequency = 60.0 + angle_rate / (2 * np.pi)
                
                # Calculate ROCOF from frequency change
                if len(self.frequency_history) > 0:
                    rocof = (frequency - self.frequency_history[-1]) * self.data_rate
                else:
                    rocof = 0.0
                
                # Update history
                self.frequency_history.append(frequency)
                if len(self.frequency_history) > 10:  # Keep last 10 measurements
                    self.frequency_history = self.frequency_history[-10:]
                
                # Clamp to reasonable values
                frequency = max(55.0, min(65.0, frequency))
                rocof = max(-2.0, min(2.0, rocof))
                
                self.last_frequency = frequency
                self.last_rocof = rocof
                
                return frequency, rocof
        
        except Exception as e:
            logger.debug(f"PMU {self.station_id}: Frequency calculation error: {e}")
        
        return self.last_frequency, self.last_rocof
    
    async def _create_data_frame(self, measurement: dict) -> C37118DataFrame:
        """Create C37.118 data frame from measurement"""
        # Phasor data
        phasor = PhasorData(
            magnitude=measurement['voltage_mag'],
            angle=measurement['voltage_angle']
        )
        
        # Frequency data
        frequency = FrequencyData(
            frequency=measurement['frequency'],
            rocof=measurement['rocof']
        )
        
        # Analog data
        analogs = [
            AnalogData(value=measurement['frequency']),
            AnalogData(value=measurement['rocof'])
        ]
        
        # Digital data (status)
        digitals = [
            DigitalData(value=0x0001)  # PMU online status
        ]
        
        return C37118DataFrame(
            id_code=self.station_id,
            timestamp=measurement['timestamp'],
            phasors=[phasor],
            frequency=frequency,
            analogs=analogs,
            digitals=digitals,
            frame_size=0  # Will be calculated during packing
        )
    
    async def _send_configuration_frame(self):
        """Send PMU configuration frame to PDC"""
        config_frame = C37118ConfigFrame(
            id_code=self.station_id,
            timestamp=time.time(),
            config=self.configuration,
            frame_size=0  # Will be calculated during packing
        )
        
        await self.send_frame(config_frame, self.pdc_id)
        logger.info(f"PMU {self.station_id}: Sent configuration frame")
    
    async def _send_header_frame(self):
        """Send PMU header frame"""
        header_data = (f"PMU {self.station_name} monitoring Bus {self.bus_number} "
                      f"at {self.data_rate} fps")
        
        header_frame = C37118HeaderFrame(
            id_code=self.station_id,
            timestamp=time.time(),
            header_data=header_data,
            frame_size=0  # Will be calculated during packing
        )
        
        await self.send_frame(header_frame, self.pdc_id)
        logger.info(f"PMU {self.station_id}: Sent header frame")
    
    async def _start_streaming(self):
        """Start data streaming"""
        if not self.streaming:
            self.streaming = True
            logger.info(f"PMU {self.station_id}: Data streaming started")
    
    async def _stop_streaming(self):
        """Stop data streaming"""
        if self.streaming:
            self.streaming = False
            logger.info(f"PMU {self.station_id}: Data streaming stopped")
    
    def get_statistics(self) -> dict:
        """Get PMU statistics"""
        runtime = time.time() - self.stats['start_time']
        return {
            'station_id': self.station_id,
            'station_name': self.station_name,
            'bus_number': self.bus_number,
            'data_rate': self.data_rate,
            'streaming': self.streaming,
            'frames_sent': self.stats['frames_sent'],
            'measurements_taken': self.stats['measurements_taken'],
            'runtime_seconds': round(runtime, 1),
            'last_frequency': round(self.last_frequency, 3),
            'last_rocof': round(self.last_rocof, 4),
            'avg_frame_rate': round(self.stats['frames_sent'] / max(1, runtime), 2)
        }

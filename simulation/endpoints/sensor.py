"""
Sensor endpoint for cyber-physical grid simulation.
Reads local bus voltage/current and sends to controller.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorReading(BaseModel):
    """Data model for sensor readings."""
    sensor_id: int | str  # Accept both int and string
    bus_id: int
    timestamp: float
    voltage_magnitude: float
    voltage_angle: float
    current_magnitude: Optional[float] = None
    current_angle: Optional[float] = None

class Sensor:
    """Sensor endpoint that measures grid parameters."""
    
    def __init__(self, sensor_id: int, bus_id: int, 
                 sampling_rate: float = 10.0,  # Hz
                 communication_protocol=None):
        """
        Initialize sensor.
        
        Args:
            sensor_id: Unique identifier for sensor
            bus_id: Bus being monitored
            sampling_rate: Measurement frequency in Hz
            communication_protocol: Protocol for communication
        """
        self.sensor_id = sensor_id
        self.bus_id = bus_id
        self.sampling_rate = sampling_rate
        self.sampling_period = 1.0 / sampling_rate
        self.communication_protocol = communication_protocol
        
        # Measurement data
        self.latest_reading: Optional[SensorReading] = None
        self.is_running = False
        self.measurement_task: Optional[asyncio.Task] = None
        
        # Grid reference (to be set by simulation)
        self.grid_interface = None
        
    def set_grid_interface(self, grid_interface):
        """Set reference to grid for measurements."""
        self.grid_interface = grid_interface
        
    async def start_measurements(self):
        """Start continuous measurement task."""
        if self.is_running:
            logger.warning(f"Sensor {self.sensor_id} already running")
            return
            
        self.is_running = True
        self.measurement_task = asyncio.create_task(self._measurement_loop())
        logger.info(f"Sensor {self.sensor_id} started measurements at {self.sampling_rate} Hz")
        
    async def stop_measurements(self):
        """Stop measurement task."""
        self.is_running = False
        if self.measurement_task:
            self.measurement_task.cancel()
            try:
                await self.measurement_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Sensor {self.sensor_id} stopped measurements")
        
    async def _measurement_loop(self):
        """Main measurement loop."""
        try:
            while self.is_running:
                await self._take_measurement()
                await asyncio.sleep(self.sampling_period)
        except asyncio.CancelledError:
            logger.info(f"Sensor {self.sensor_id} measurement loop cancelled")
        except Exception as e:
            logger.error(f"Sensor {self.sensor_id} measurement error: {e}")
            
    async def _take_measurement(self):
        """Take a measurement and store it."""
        if not self.grid_interface:
            logger.warning(f"Sensor {self.sensor_id}: No grid interface available")
            return
            
        try:
            # Get current grid state
            voltage_mag, voltage_angle = self.grid_interface.get_bus_voltage(self.bus_id)
            
            # Create measurement
            reading = SensorReading(
                sensor_id=self.sensor_id,
                bus_id=self.bus_id,
                timestamp=time.time(),
                voltage_magnitude=voltage_mag,
                voltage_angle=voltage_angle
            )
            
            self.latest_reading = reading
            
            # Send to controller via communication protocol
            if self.communication_protocol:
                await self.communication_protocol.send_message(
                    sender_id=self.sensor_id,
                    receiver_id=f"controller_{self.bus_id}",
                    message_type="sensor_reading",
                    data=reading.model_dump()
                )
                
            logger.debug(f"Sensor {self.sensor_id}: V={voltage_mag:.3f}∠{voltage_angle:.1f}°")
            
        except Exception as e:
            logger.error(f"Sensor {self.sensor_id} measurement failed: {e}")
            
    def get_latest_reading(self) -> Optional[SensorReading]:
        """Get the latest sensor reading."""
        return self.latest_reading
        
    async def get_instantaneous_reading(self) -> Optional[SensorReading]:
        """Get an instantaneous reading (bypasses normal sampling)."""
        await self._take_measurement()
        return self.latest_reading
        
    def __str__(self):
        status = "Running" if self.is_running else "Stopped"
        latest = f"V={self.latest_reading.voltage_magnitude:.3f}" if self.latest_reading else "No data"
        return f"Sensor {self.sensor_id} at Bus {self.bus_id} [{status}]: {latest}"

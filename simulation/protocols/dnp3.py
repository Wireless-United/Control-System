"""
DNP3 Protocol Implementation

Mock DNP3 (Distributed Network Protocol) for SCADA-RTU communication.
Simulates DNP3 Application Layer function codes, polling intervals, and communication delays.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field
from enum import Enum
import struct
import binascii

logger = logging.getLogger(__name__)


class DNP3FunctionCode(Enum):
    """DNP3 Application Layer Function Codes"""
    READ = 0x01
    WRITE = 0x02
    SELECT = 0x03
    OPERATE = 0x04
    DIRECT_OPERATE = 0x05
    RESPONSE = 0x81
    UNSOLICITED_RESPONSE = 0x82
    CONFIRM = 0x00


class DNP3ObjectGroup(Enum):
    """DNP3 Object Groups"""
    BINARY_INPUT = 0x01
    BINARY_OUTPUT = 0x0C
    ANALOG_INPUT = 0x1E
    ANALOG_OUTPUT = 0x28
    COUNTER = 0x14


class DNP3DataPoint(BaseModel):
    """DNP3 Data Point"""
    group: DNP3ObjectGroup
    index: int
    value: float
    quality: int = 0x01  # ONLINE
    timestamp: float = Field(default_factory=time.time)
    
    
class DNP3Message(BaseModel):
    """DNP3 Application Message"""
    source: int
    destination: int
    function_code: DNP3FunctionCode
    sequence: int
    data_points: List[DNP3DataPoint] = []
    control_code: Optional[int] = None
    message_id: str = Field(default_factory=lambda: f"dnp3_{int(time.time()*1000000)}")
    timestamp: float = Field(default_factory=time.time)


class DNP3Frame(BaseModel):
    """DNP3 Link Layer Frame"""
    start: int = 0x0564  # DNP3 start bytes
    length: int
    control: int
    destination: int
    source: int
    crc: int
    data: bytes
    
    @classmethod
    def create_frame(cls, source: int, destination: int, data: bytes) -> 'DNP3Frame':
        """Create a DNP3 frame with calculated CRC"""
        length = len(data) + 5  # 5 bytes for header fields
        control = 0x44  # Primary, user data
        
        # Simple CRC calculation (mock)
        crc_data = struct.pack('<BHBB', length, control, destination, source) + data
        crc = binascii.crc32(crc_data) & 0xFFFF
        
        return cls(
            length=length,
            control=control,
            destination=destination,
            source=source,
            crc=crc,
            data=data
        )
    
    def validate_crc(self) -> bool:
        """Validate frame CRC"""
        crc_data = struct.pack('<BHBB', self.length, self.control, self.destination, self.source) + self.data
        expected_crc = binascii.crc32(crc_data) & 0xFFFF
        return self.crc == expected_crc


class DNP3Protocol:
    """DNP3 Protocol Handler for SCADA-RTU Communication"""
    
    def __init__(self, 
                 station_id: int, 
                 network_name: str = "dnp3_network",
                 communication_delay: tuple = (100, 500)):  # ms
        self.station_id = station_id
        self.network_name = network_name
        self.communication_delay = communication_delay
        
        self.stations: Dict[int, 'DNP3Station'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.network_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_delivered = 0
        self.total_latency = 0.0
        
        logger.info(f"DNP3 Protocol initialized for station {station_id} on network {network_name}")
    
    async def start_network(self):
        """Start the DNP3 network communication"""
        if self.running:
            return
            
        self.running = True
        self.network_task = asyncio.create_task(self._network_loop())
        logger.info(f"DNP3 network {self.network_name} started")
    
    async def stop_network(self):
        """Stop the DNP3 network communication"""
        if not self.running:
            return
            
        self.running = False
        if self.network_task:
            self.network_task.cancel()
            try:
                await self.network_task
            except asyncio.CancelledError:
                pass
            
        logger.info(f"DNP3 network {self.network_name} stopped")
    
    async def _network_loop(self):
        """Main network communication loop"""
        try:
            while self.running:
                try:
                    # Process message queue with timeout
                    message = await asyncio.wait_for(
                        self.message_queue.get(), timeout=1.0
                    )
                    await self._deliver_message(message)
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"DNP3 {self.network_name} network loop cancelled")
            raise
    
    async def _deliver_message(self, message: DNP3Message):
        """Deliver message with simulated communication delay"""
        start_time = time.time()
        
        # Simulate communication delay
        delay_ms = random.uniform(*self.communication_delay)
        await asyncio.sleep(delay_ms / 1000.0)
        
        # Find destination station
        destination_station = self.stations.get(message.destination)
        if destination_station:
            await destination_station.receive_message(message)
            self.messages_delivered += 1
            
            # Update latency statistics
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency
            
            logger.debug(f"DNP3 {self.network_name}: Delivered message {message.message_id} "
                        f"from {message.source} to {message.destination} (latency: {latency:.1f}ms)")
        else:
            logger.warning(f"DNP3 {self.network_name}: Station {message.destination} not found")
    
    def register_station(self, station: 'DNP3Station'):
        """Register a DNP3 station on the network"""
        self.stations[station.station_id] = station
        logger.info(f"DNP3 {self.network_name}: Registered station {station.station_id}")
    
    def unregister_station(self, station_id: int):
        """Unregister a DNP3 station from the network"""
        if station_id in self.stations:
            del self.stations[station_id]
            logger.info(f"DNP3 {self.network_name}: Unregistered station {station_id}")
    
    async def send_message(self, message: DNP3Message):
        """Send a DNP3 message"""
        self.messages_sent += 1
        await self.message_queue.put(message)
        logger.debug(f"DNP3 {self.network_name}: Queued message {message.message_id} "
                    f"from {message.source} to {message.destination}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics"""
        avg_latency = self.total_latency / max(1, self.messages_delivered)
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_delivered": self.messages_delivered,
            "average_latency_ms": round(avg_latency, 1),
            "stations_count": len(self.stations)
        }


class DNP3Station:
    """Base class for DNP3 stations (Master or Outstation)"""
    
    def __init__(self, station_id: int, protocol: DNP3Protocol):
        self.station_id = station_id
        self.protocol = protocol
        self.sequence_number = 0
        self.running = False
        
        # Message handlers
        self.message_handlers: Dict[DNP3FunctionCode, Callable] = {}
        
        # Register with protocol
        self.protocol.register_station(self)
    
    async def receive_message(self, message: DNP3Message):
        """Receive and process a DNP3 message"""
        self.protocol.messages_received += 1
        
        logger.debug(f"DNP3 Station {self.station_id}: Received {message.function_code.name} "
                    f"from {message.source}")
        
        # Handle message based on function code
        handler = self.message_handlers.get(message.function_code)
        if handler:
            await handler(message)
        else:
            logger.warning(f"DNP3 Station {self.station_id}: No handler for {message.function_code.name}")
    
    async def send_message(self, 
                          destination: int, 
                          function_code: DNP3FunctionCode,
                          data_points: List[DNP3DataPoint] = None,
                          control_code: Optional[int] = None):
        """Send a DNP3 message"""
        message = DNP3Message(
            source=self.station_id,
            destination=destination,
            function_code=function_code,
            sequence=self._next_sequence(),
            data_points=data_points or [],
            control_code=control_code
        )
        
        await self.protocol.send_message(message)
        return message
    
    def _next_sequence(self) -> int:
        """Get next sequence number"""
        self.sequence_number = (self.sequence_number + 1) % 256
        return self.sequence_number
    
    async def start(self):
        """Start the station"""
        self.running = True
        logger.info(f"DNP3 Station {self.station_id} started")
    
    async def stop(self):
        """Stop the station"""
        self.running = False
        self.protocol.unregister_station(self.station_id)
        logger.info(f"DNP3 Station {self.station_id} stopped")


class DNP3PollingScheduler:
    """Handles DNP3 polling schedules for SCADA masters"""
    
    def __init__(self):
        self.poll_groups: Dict[str, Dict] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
    
    def add_poll_group(self, 
                      name: str,
                      interval: float,
                      callback: Callable,
                      *args, **kwargs):
        """Add a polling group with specified interval"""
        self.poll_groups[name] = {
            'interval': interval,
            'callback': callback,
            'args': args,
            'kwargs': kwargs,
            'last_poll': 0,
            'poll_count': 0
        }
        logger.info(f"DNP3 Polling: Added group '{name}' with {interval}s interval")
    
    def remove_poll_group(self, name: str):
        """Remove a polling group"""
        if name in self.poll_groups:
            del self.poll_groups[name]
            logger.info(f"DNP3 Polling: Removed group '{name}'")
    
    async def start_polling(self):
        """Start the polling scheduler"""
        if self.running:
            return
            
        self.running = True
        self.scheduler_task = asyncio.create_task(self._polling_loop())
        logger.info("DNP3 Polling scheduler started")
    
    async def stop_polling(self):
        """Stop the polling scheduler"""
        if not self.running:
            return
            
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
                
        logger.info("DNP3 Polling scheduler stopped")
    
    async def _polling_loop(self):
        """Main polling loop"""
        try:
            while self.running:
                current_time = time.time()
                
                for name, group in self.poll_groups.items():
                    if current_time - group['last_poll'] >= group['interval']:
                        try:
                            await group['callback'](*group['args'], **group['kwargs'])
                            group['last_poll'] = current_time
                            group['poll_count'] += 1
                        except Exception as e:
                            logger.error(f"DNP3 Polling error in group '{name}': {e}")
                
                await asyncio.sleep(0.1)  # 100ms resolution
                
        except asyncio.CancelledError:
            logger.info("DNP3 Polling loop cancelled")
            raise

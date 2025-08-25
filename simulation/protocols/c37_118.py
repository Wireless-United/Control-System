"""
IEEE C37.118 Synchrophasor Protocol Implementation

This module implements a lightweight subset of IEEE C37.118 standard for 
synchrophasor measurement and communication between PMUs and PDCs.

IEEE C37.118 Frame Types:
- Configuration Frame: PMU metadata and configuration
- Data Frame: Phasor measurements with timestamp
- Header Frame: Text information about PMU
- Command Frame: Control commands (start/stop streaming)
"""

import asyncio
import struct
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel
import crcmod

# IEEE C37.118 Frame Types
FRAME_TYPE_DATA = 0x00
FRAME_TYPE_HEADER = 0x01
FRAME_TYPE_CONFIG1 = 0x02
FRAME_TYPE_CONFIG2 = 0x03
FRAME_TYPE_CONFIG3 = 0x05
FRAME_TYPE_COMMAND = 0x04

# Command Frame Commands
CMD_TURN_OFF_TRANSMISSION = 0x01
CMD_TURN_ON_TRANSMISSION = 0x02
CMD_SEND_HDR = 0x03
CMD_SEND_CFG1 = 0x04
CMD_SEND_CFG2 = 0x05
CMD_SEND_CFG3 = 0x06

# Data formats
PHASOR_REAL_INT16 = 0
PHASOR_REAL_FLOAT = 1
PHASOR_COMPLEX_INT16 = 2
PHASOR_COMPLEX_FLOAT = 3

logger = logging.getLogger(__name__)

class PhasorData(BaseModel):
    """Phasor measurement data"""
    magnitude: float
    angle: float  # in radians
    
class FrequencyData(BaseModel):
    """Frequency and ROCOF data"""
    frequency: float  # in Hz
    rocof: float     # Rate of Change of Frequency in Hz/s

class AnalogData(BaseModel):
    """Analog measurement data"""
    value: float
    
class DigitalData(BaseModel):
    """Digital status data"""
    value: int

@dataclass
class PMUConfiguration:
    """PMU Configuration data"""
    station_name: str
    id_code: int
    data_format: int
    phasor_channels: int
    analog_channels: int
    digital_channels: int
    nominal_frequency: float
    configuration_count: int
    data_rate: int
    phasor_names: List[str]
    analog_names: List[str]
    digital_names: List[str]

class C37118Frame(BaseModel):
    """Base class for all IEEE C37.118 frames"""
    sync: int
    frame_size: int
    id_code: int
    timestamp: float
    frame_type: int
    
    def pack_common_header(self) -> bytes:
        """Pack common frame header"""
        # Convert timestamp to IEEE C37.118 format (seconds since epoch + fraction)
        timestamp_int = int(self.timestamp)
        timestamp_frac = int((self.timestamp - timestamp_int) * 1000000)
        
        return struct.pack('>HHHIII', 
                          self.sync,
                          self.frame_size, 
                          self.id_code,
                          timestamp_int,
                          timestamp_frac,
                          0)  # Time quality flags

class C37118DataFrame(C37118Frame):
    """IEEE C37.118 Data Frame"""
    phasors: List[PhasorData]
    frequency: FrequencyData
    analogs: List[AnalogData]
    digitals: List[DigitalData]
    
    def __init__(self, **data):
        if 'frame_type' not in data:
            data['frame_type'] = FRAME_TYPE_DATA
        if 'sync' not in data:
            data['sync'] = 0xAA01  # Data frame sync word
        super().__init__(**data)
    
    def pack(self) -> bytes:
        """Pack data frame into binary format"""
        # Start with common header
        data = self.pack_common_header()
        
        # Pack phasors (using float format for simplicity)
        for phasor in self.phasors:
            data += struct.pack('>ff', phasor.magnitude, phasor.angle)
        
        # Pack frequency and ROCOF
        data += struct.pack('>ff', self.frequency.frequency, self.frequency.rocof)
        
        # Pack analogs
        for analog in self.analogs:
            data += struct.pack('>f', analog.value)
        
        # Pack digitals
        for digital in self.digitals:
            data += struct.pack('>H', digital.value)
        
        # Update frame size
        frame_size = len(data) + 2  # +2 for CRC
        data = data[:2] + struct.pack('>H', frame_size) + data[4:]
        
        # Calculate and append CRC
        crc_func = crcmod.mkCrcFun(0x11021, 0xFFFF, True, 0x0000)
        crc = crc_func(data)
        data += struct.pack('>H', crc)
        
        return data

class C37118ConfigFrame(C37118Frame):
    """IEEE C37.118 Configuration Frame"""
    config: PMUConfiguration
    
    def __init__(self, **data):
        if 'frame_type' not in data:
            data['frame_type'] = FRAME_TYPE_CONFIG2
        if 'sync' not in data:
            data['sync'] = 0xAA21  # Config frame sync word
        super().__init__(**data)
    
    def pack(self) -> bytes:
        """Pack configuration frame into binary format"""
        # Start with common header
        data = self.pack_common_header()
        
        # Pack configuration data
        data += struct.pack('>H', 1)  # Number of PMUs
        
        # PMU configuration
        station_name = self.config.station_name.encode('ascii')[:16].ljust(16, b'\x00')
        data += station_name
        data += struct.pack('>H', self.config.id_code)
        data += struct.pack('>H', self.config.data_format)
        data += struct.pack('>H', self.config.phasor_channels)
        data += struct.pack('>H', self.config.analog_channels)
        data += struct.pack('>H', self.config.digital_channels)
        
        # Channel names (simplified)
        for name in self.config.phasor_names:
            ch_name = name.encode('ascii')[:16].ljust(16, b'\x00')
            data += ch_name
        
        for name in self.config.analog_names:
            ch_name = name.encode('ascii')[:16].ljust(16, b'\x00')
            data += ch_name
        
        for name in self.config.digital_names:
            ch_name = name.encode('ascii')[:16].ljust(16, b'\x00')
            data += ch_name
        
        # Nominal frequency and data rate
        data += struct.pack('>H', int(self.config.nominal_frequency))
        data += struct.pack('>H', self.config.configuration_count)
        data += struct.pack('>H', self.config.data_rate)
        
        # Update frame size
        frame_size = len(data) + 2  # +2 for CRC
        data = data[:2] + struct.pack('>H', frame_size) + data[4:]
        
        # Calculate and append CRC
        crc_func = crcmod.mkCrcFun(0x11021, 0xFFFF, True, 0x0000)
        crc = crc_func(data)
        data += struct.pack('>H', crc)
        
        return data

class C37118HeaderFrame(C37118Frame):
    """IEEE C37.118 Header Frame"""
    header_data: str
    
    def __init__(self, **data):
        if 'frame_type' not in data:
            data['frame_type'] = FRAME_TYPE_HEADER
        if 'sync' not in data:
            data['sync'] = 0xAA11  # Header frame sync word
        super().__init__(**data)
    
    def pack(self) -> bytes:
        """Pack header frame into binary format"""
        # Start with common header
        data = self.pack_common_header()
        
        # Pack header data
        header_bytes = self.header_data.encode('ascii')
        data += header_bytes
        
        # Update frame size
        frame_size = len(data) + 2  # +2 for CRC
        data = data[:2] + struct.pack('>H', frame_size) + data[4:]
        
        # Calculate and append CRC
        crc_func = crcmod.mkCrcFun(0x11021, 0xFFFF, True, 0x0000)
        crc = crc_func(data)
        data += struct.pack('>H', crc)
        
        return data

class C37118CommandFrame(C37118Frame):
    """IEEE C37.118 Command Frame"""
    command: int
    
    def __init__(self, **data):
        if 'frame_type' not in data:
            data['frame_type'] = FRAME_TYPE_COMMAND
        if 'sync' not in data:
            data['sync'] = 0xAA41  # Command frame sync word
        super().__init__(**data)
    
    def pack(self) -> bytes:
        """Pack command frame into binary format"""
        # Start with common header
        data = self.pack_common_header()
        
        # Pack command
        data += struct.pack('>H', self.command)
        
        # Update frame size
        frame_size = len(data) + 2  # +2 for CRC
        data = data[:2] + struct.pack('>H', frame_size) + data[4:]
        
        # Calculate and append CRC
        crc_func = crcmod.mkCrcFun(0x11021, 0xFFFF, True, 0x0000)
        crc = crc_func(data)
        data += struct.pack('>H', crc)
        
        return data

class C37118Protocol:
    """IEEE C37.118 Protocol Handler"""
    
    def __init__(self, network_name: str):
        self.network_name = network_name
        self.stations: Dict[int, Any] = {}
        self.message_handlers: Dict[int, List] = {}
        self.network_task: Optional[asyncio.Task] = None
        self.running = False
        self.message_queue = asyncio.Queue()
        self.stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'data_frames': 0,
            'config_frames': 0,
            'header_frames': 0,
            'command_frames': 0,
            'total_latency': 0.0,
            'frame_count': 0
        }
        
        logger.info(f"C37.118 protocol {network_name} initialized")
    
    async def start_network(self):
        """Start the protocol network"""
        if self.running:
            return
        
        self.running = True
        self.network_task = asyncio.create_task(self._network_loop())
        logger.info(f"C37.118 network {self.network_name} started")
    
    async def stop_network(self):
        """Stop the protocol network"""
        if not self.running:
            return
        
        self.running = False
        if self.network_task:
            self.network_task.cancel()
            try:
                await self.network_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"C37.118 network {self.network_name} stopped")
    
    async def _network_loop(self):
        """Main network processing loop"""
        try:
            while self.running:
                try:
                    # Process messages from queue
                    frame, sender_id, receiver_id, send_time = await asyncio.wait_for(
                        self.message_queue.get(), timeout=0.1)
                    
                    # Simulate network delay (10-50ms for PMU communication)
                    import random
                    delay = random.uniform(0.01, 0.05)  # 10-50ms delay
                    await asyncio.sleep(delay)
                    
                    # Calculate latency
                    latency = (time.time() - send_time) * 1000  # Convert to ms
                    self.stats['total_latency'] += latency
                    self.stats['frame_count'] += 1
                    
                    # Deliver message to receivers
                    if receiver_id in self.message_handlers:
                        for handler in self.message_handlers[receiver_id]:
                            try:
                                await handler(frame, sender_id, latency)
                                self.stats['frames_received'] += 1
                            except Exception as e:
                                logger.error(f"C37.118 {self.network_name}: Handler error: {e}")
                    else:
                        logger.warning(f"C37.118 {self.network_name}: Receiver {receiver_id} not found")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"C37.118 {self.network_name}: Network loop error: {e}")
        
        except asyncio.CancelledError:
            logger.info(f"C37.118 network {self.network_name} loop cancelled")
    
    def register_station(self, station_id: int, station):
        """Register a station (PMU or PDC)"""
        self.stations[station_id] = station
        if station_id not in self.message_handlers:
            self.message_handlers[station_id] = []
        logger.info(f"C37.118 {self.network_name}: Registered station {station_id}")
    
    def unregister_station(self, station_id: int):
        """Unregister a station"""
        if station_id in self.stations:
            del self.stations[station_id]
        if station_id in self.message_handlers:
            del self.message_handlers[station_id]
        logger.info(f"C37.118 {self.network_name}: Unregistered station {station_id}")
    
    def add_message_handler(self, station_id: int, handler):
        """Add a message handler for a station"""
        if station_id not in self.message_handlers:
            self.message_handlers[station_id] = []
        self.message_handlers[station_id].append(handler)
    
    async def send_frame(self, frame: C37118Frame, sender_id: int, receiver_id: int):
        """Send a frame from sender to receiver"""
        if not self.running:
            return
        
        send_time = time.time()
        await self.message_queue.put((frame, sender_id, receiver_id, send_time))
        self.stats['frames_sent'] += 1
        
        # Update frame type statistics
        if frame.frame_type == FRAME_TYPE_DATA:
            self.stats['data_frames'] += 1
        elif frame.frame_type in [FRAME_TYPE_CONFIG1, FRAME_TYPE_CONFIG2, FRAME_TYPE_CONFIG3]:
            self.stats['config_frames'] += 1
        elif frame.frame_type == FRAME_TYPE_HEADER:
            self.stats['header_frames'] += 1
        elif frame.frame_type == FRAME_TYPE_COMMAND:
            self.stats['command_frames'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        avg_latency = (self.stats['total_latency'] / self.stats['frame_count'] 
                      if self.stats['frame_count'] > 0 else 0)
        
        return {
            'frames_sent': self.stats['frames_sent'],
            'frames_received': self.stats['frames_received'],
            'data_frames': self.stats['data_frames'],
            'config_frames': self.stats['config_frames'],
            'header_frames': self.stats['header_frames'],
            'command_frames': self.stats['command_frames'],
            'avg_latency_ms': round(avg_latency, 2),
            'delivery_rate': (self.stats['frames_received'] / max(1, self.stats['frames_sent']) * 100)
        }

class C37118Station:
    """Base class for C37.118 stations (PMU/PDC)"""
    
    def __init__(self, station_id: int, protocol: C37118Protocol):
        self.station_id = station_id
        self.protocol = protocol
        self.running = False
        
        # Register with protocol
        protocol.register_station(station_id, self)
        protocol.add_message_handler(station_id, self._handle_frame)
    
    async def start(self):
        """Start the station"""
        self.running = True
        logger.info(f"C37.118 Station {self.station_id} started")
    
    async def stop(self):
        """Stop the station"""
        self.running = False
        self.protocol.unregister_station(self.station_id)
        logger.info(f"C37.118 Station {self.station_id} stopped")
    
    async def _handle_frame(self, frame: C37118Frame, sender_id: int, latency: float):
        """Handle incoming frame - to be implemented by subclasses"""
        pass
    
    async def send_frame(self, frame: C37118Frame, receiver_id: int):
        """Send a frame to another station"""
        await self.protocol.send_frame(frame, self.station_id, receiver_id)

# Global protocol instances
protocols: Dict[str, C37118Protocol] = {}

def get_protocol(network_name: str) -> C37118Protocol:
    """Get or create a protocol instance"""
    if network_name not in protocols:
        protocols[network_name] = C37118Protocol(network_name)
    return protocols[network_name]

def parse_frame(data: bytes) -> Optional[C37118Frame]:
    """Parse binary data into C37.118 frame"""
    if len(data) < 14:  # Minimum frame size
        return None
    
    try:
        # Parse common header
        sync, frame_size, id_code, timestamp_int, timestamp_frac, _ = struct.unpack('>HHHIII', data[:14])
        timestamp = timestamp_int + timestamp_frac / 1000000.0
        
        # Determine frame type from sync word
        if sync == 0xAA01:
            frame_type = FRAME_TYPE_DATA
        elif sync == 0xAA11:
            frame_type = FRAME_TYPE_HEADER
        elif sync in [0xAA21, 0xAA31, 0xAA51]:
            frame_type = FRAME_TYPE_CONFIG2  # Simplified
        elif sync == 0xAA41:
            frame_type = FRAME_TYPE_COMMAND
        else:
            return None
        
        # Create appropriate frame object (simplified parsing)
        if frame_type == FRAME_TYPE_DATA:
            return C37118DataFrame(
                sync=sync,
                frame_size=frame_size,
                id_code=id_code,
                timestamp=timestamp,
                frame_type=frame_type,
                phasors=[],
                frequency=FrequencyData(frequency=60.0, rocof=0.0),
                analogs=[],
                digitals=[]
            )
        elif frame_type == FRAME_TYPE_HEADER:
            header_data = data[14:-2].decode('ascii', errors='ignore')
            return C37118HeaderFrame(
                sync=sync,
                frame_size=frame_size,
                id_code=id_code,
                timestamp=timestamp,
                frame_type=frame_type,
                header_data=header_data
            )
        elif frame_type in [FRAME_TYPE_CONFIG1, FRAME_TYPE_CONFIG2, FRAME_TYPE_CONFIG3]:
            # Simplified config frame
            config = PMUConfiguration(
                station_name="Unknown",
                id_code=id_code,
                data_format=PHASOR_COMPLEX_FLOAT,
                phasor_channels=1,
                analog_channels=0,
                digital_channels=0,
                nominal_frequency=60.0,
                configuration_count=1,
                data_rate=50,
                phasor_names=["V1"],
                analog_names=[],
                digital_names=[]
            )
            return C37118ConfigFrame(
                sync=sync,
                frame_size=frame_size,
                id_code=id_code,
                timestamp=timestamp,
                frame_type=frame_type,
                config=config
            )
        elif frame_type == FRAME_TYPE_COMMAND:
            command = struct.unpack('>H', data[14:16])[0] if len(data) >= 16 else 0
            return C37118CommandFrame(
                sync=sync,
                frame_size=frame_size,
                id_code=id_code,
                timestamp=timestamp,
                frame_type=frame_type,
                command=command
            )
    
    except Exception as e:
        logger.error(f"Error parsing C37.118 frame: {e}")
        return None
    
    return None

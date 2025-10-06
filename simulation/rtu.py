#!/usr/bin/env python3
"""
RTU (Remote Terminal Unit) Implementation for IEEE 39-Bus System

This module implements RTU outstations that respond to DNP3 polling requests
from SCADA master station. RTUs collect measurements from power system buses
and respond to control commands.

Features:
- DNP3 outstation protocol implementation
- Real-time power system measurements from IEEE 39-bus system
- Binary and analog input/output points
- Event reporting and unsolicited responses
- Integration with IEEE 39-bus power system model
"""

import asyncio
import logging
import socket
import struct
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DNP3 Protocol Constants
DNP3_START_BYTES = b'\x05\x64'
DNP3_PORT = 20000

class DNP3FunctionCode(Enum):
    """DNP3 Function Codes for RTU responses"""
    READ = 0x01
    WRITE = 0x02
    SELECT = 0x03
    OPERATE = 0x04
    DIRECT_OPERATE = 0x05
    RESPONSE = 0x81
    UNSOLICITED_RESPONSE = 0x82

class DNP3ObjectGroup(Enum):
    """DNP3 Object Groups"""
    BINARY_INPUT = 0x01          # Breaker status, protection trips
    BINARY_OUTPUT = 0x0C         # Breaker commands
    ANALOG_INPUT = 0x1E          # Voltage, current, power measurements
    ANALOG_OUTPUT = 0x29         # Setpoint commands

@dataclass
class MeasurementPoint:
    """Measurement point configuration"""
    index: int
    name: str
    object_group: DNP3ObjectGroup
    data_type: str  # 'binary', 'analog_float', 'analog_int'
    unit: str
    scale_factor: float = 1.0
    current_value: float = 0.0
    quality: int = 0x01  # ONLINE
    timestamp: float = field(default_factory=time.time)

@dataclass
class RTUConfiguration:
    """RTU configuration parameters"""
    rtu_id: int
    bus_number: int
    name: str
    ip_address: str
    port: int = DNP3_PORT
    update_rate: float = 1.0  # seconds
    measurement_points: List[MeasurementPoint] = field(default_factory=list)

class IEEE39RTU:
    """
    RTU Outstation for IEEE 39-Bus System
    
    Simulates a Remote Terminal Unit that:
    - Collects measurements from assigned bus
    - Responds to DNP3 polling requests
    - Executes control commands
    - Reports events and alarms
    """
    
    def __init__(self, config: RTUConfiguration, power_system=None):
        """
        Initialize RTU outstation.
        
        Args:
            config: RTU configuration
            power_system: Reference to IEEE 39-bus system for measurements
        """
        self.config = config
        self.power_system = power_system
        self.is_running = False
        self.server_socket: Optional[socket.socket] = None
        self.clients: List[socket.socket] = []
        
        # Measurement database
        self.measurements = {mp.index: mp for mp in config.measurement_points}
        
        # Event buffer for unsolicited reporting
        self.event_buffer: List[Dict] = []
        self.max_events = 100
        
        # Statistics
        self.stats = {
            'requests_received': 0,
            'responses_sent': 0,
            'events_generated': 0,
            'last_poll_time': 0.0,
            'uptime_start': time.time()
        }
        
        logger.info(f"RTU {config.rtu_id} initialized for Bus {config.bus_number}")
        logger.info(f"Listening on {config.ip_address}:{config.port}")
        logger.info(f"Configured {len(config.measurement_points)} measurement points")
    
    async def start(self):
        """Start the RTU outstation server"""
        if self.is_running:
            logger.warning(f"RTU {self.config.rtu_id} already running")
            return
        
        self.is_running = True
        
        try:
            # Create server socket for asyncio
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setblocking(False)  # Non-blocking for asyncio
            self.server_socket.bind((self.config.ip_address, self.config.port))
            self.server_socket.listen(5)
            
            logger.info(f"🟢 RTU {self.config.rtu_id} started successfully")
            
            # Start measurement update task
            measurement_task = asyncio.create_task(self._measurement_update_loop())
            
            # Start server accept loop
            server_task = asyncio.create_task(self._server_loop())
            
            # Wait for both tasks
            await asyncio.gather(measurement_task, server_task)
            
        except Exception as e:
            logger.error(f"Error starting RTU {self.config.rtu_id}: {e}")
            raise
    
    async def stop(self):
        """Stop the RTU outstation"""
        logger.info(f"🔴 Stopping RTU {self.config.rtu_id}")
        
        self.is_running = False
        
        # Close client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info(f"RTU {self.config.rtu_id} stopped")
    
    async def _server_loop(self):
        """Main server loop for handling client connections"""
        while self.is_running:
            try:
                if self.server_socket:
                    try:
                        # Use asyncio to properly handle socket operations
                        loop = asyncio.get_event_loop()
                        client_socket, address = await loop.sock_accept(self.server_socket)
                        logger.info(f"RTU {self.config.rtu_id}: New connection from {address}")
                        
                        # Handle client in separate task
                        client_task = asyncio.create_task(
                            self._handle_client(client_socket, address)
                        )
                        
                    except asyncio.CancelledError:
                        break
                    except OSError as e:
                        if self.is_running:  # Only log if not shutting down
                            logger.debug(f"Socket accept error: {e}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                if self.is_running:
                    await asyncio.sleep(1)
    
    async def _handle_client(self, client_socket: socket.socket, address):
        """Handle individual client connection"""
        client_socket.setblocking(False)  # Non-blocking for asyncio
        self.clients.append(client_socket)
        
        try:
            loop = asyncio.get_event_loop()
            while self.is_running:
                try:
                    # Receive DNP3 request using asyncio
                    data = await loop.sock_recv(client_socket, 1024)
                    if not data:
                        break
                    
                    logger.debug(f"RTU {self.config.rtu_id}: Received {len(data)} bytes from {address}")
                    
                    # Process DNP3 request and send response
                    response = await self._process_dnp3_request(data)
                    if response:
                        await loop.sock_sendall(client_socket, response)
                        self.stats['responses_sent'] += 1
                        logger.debug(f"RTU {self.config.rtu_id}: Sent {len(response)} bytes response to {address}")
                    
                except asyncio.CancelledError:
                    break
                except ConnectionResetError:
                    logger.info(f"RTU {self.config.rtu_id}: Client {address} disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error handling client {address}: {e}")
                    break
        
        finally:
            # Clean up client connection
            try:
                client_socket.close()
            except:
                pass
            
            if client_socket in self.clients:
                self.clients.remove(client_socket)
    
    async def _process_dnp3_request(self, request_data: bytes) -> Optional[bytes]:
        """Process incoming DNP3 request and generate response"""
        self.stats['requests_received'] += 1
        self.stats['last_poll_time'] = time.time()
        
        try:
            # Parse DNP3 request (simplified)
            if len(request_data) < 10:
                return None
            
            # Check DNP3 start bytes
            if request_data[:2] != DNP3_START_BYTES:
                return None
            
            # Extract function code
            function_code = request_data[10] if len(request_data) > 10 else 0
            
            logger.debug(f"RTU {self.config.rtu_id}: Received DNP3 request, FC={function_code:02x}")
            
            # Generate appropriate response based on function code
            if function_code == DNP3FunctionCode.READ.value:
                return self._generate_read_response(request_data)
            elif function_code in [DNP3FunctionCode.OPERATE.value, DNP3FunctionCode.DIRECT_OPERATE.value]:
                return self._generate_operate_response(request_data)
            else:
                # Return generic acknowledgment
                return self._generate_ack_response(request_data)
        
        except Exception as e:
            logger.error(f"Error processing DNP3 request: {e}")
            return None
    
    def _generate_read_response(self, request: bytes) -> bytes:
        """Generate DNP3 read response with current measurements"""
        # Build response header
        response = bytearray()
        response.extend(DNP3_START_BYTES)  # Start bytes
        
        # Response will be built dynamically
        response_data = bytearray()
        
        # Add analog input measurements (use proper index mapping)
        analog_measurements = [mp for mp in self.measurements.values() 
                             if mp.object_group == DNP3ObjectGroup.ANALOG_INPUT]
        analog_measurements.sort(key=lambda x: x.index)  # Sort by index
        
        for measurement in analog_measurements:
            # Group 30 Variation 1: 32-bit analog input
            response_data.extend(b'\x1E\x01')  # Group 30, Variation 1
            # Use measurement-specific index (1=voltage, 2=frequency, 3=active_power, 4=reactive_power)
            mapped_index = 0
            if measurement.name == 'voltage_magnitude':
                mapped_index = 0
            elif measurement.name == 'frequency':
                mapped_index = 1
            elif measurement.name == 'active_power':
                mapped_index = 2
            elif measurement.name == 'reactive_power':
                mapped_index = 3
            
            response_data.extend(struct.pack('<H', mapped_index))  # Mapped index
            response_data.extend(struct.pack('<f', measurement.current_value))  # Value
            response_data.extend(struct.pack('<B', measurement.quality))  # Quality
        
        # Add binary input measurements
        binary_measurements = [mp for mp in self.measurements.values() 
                             if mp.object_group == DNP3ObjectGroup.BINARY_INPUT]
        
        for measurement in binary_measurements:
            # Group 1 Variation 1: Single-bit binary input
            response_data.extend(b'\x01\x01')  # Group 1, Variation 1
            response_data.extend(struct.pack('<H', 0))  # Use index 0 for breaker status
            response_data.extend(struct.pack('<B', int(measurement.current_value)))  # Value
            response_data.extend(struct.pack('<B', measurement.quality))  # Quality
        
        # Complete response packet
        response.append(len(response_data) + 5)  # Length
        response.append(0x00)  # Control
        response.extend(struct.pack('<H', self.config.rtu_id))  # Source
        response.extend(struct.pack('<H', 1))  # Destination (master)
        response.extend(b'\x00\x00')  # Header CRC (simplified)
        response.append(DNP3FunctionCode.RESPONSE.value)  # Function code
        response.extend(response_data)
        response.extend(b'\x00\x00')  # Data CRC (simplified)
        
        logger.debug(f"RTU {self.config.rtu_id}: Sent read response with {len(self.measurements)} measurements")
        return bytes(response)
    
    def _generate_operate_response(self, request: bytes) -> bytes:
        """Generate DNP3 operate response (acknowledgment)"""
        # Extract command data and execute (simplified)
        logger.info(f"RTU {self.config.rtu_id}: Executing control command")
        
        # For demonstration, just return success acknowledgment
        response = bytearray()
        response.extend(DNP3_START_BYTES)
        response.append(8)  # Length
        response.append(0x00)  # Control
        response.extend(struct.pack('<H', self.config.rtu_id))  # Source
        response.extend(struct.pack('<H', 1))  # Destination
        response.extend(b'\x00\x00')  # Header CRC
        response.append(DNP3FunctionCode.RESPONSE.value)
        response.extend(b'\x00')  # Success status
        response.extend(b'\x00\x00')  # Data CRC
        
        return bytes(response)
    
    def _generate_ack_response(self, request: bytes) -> bytes:
        """Generate generic DNP3 acknowledgment response"""
        response = bytearray()
        response.extend(DNP3_START_BYTES)
        response.append(8)  # Length
        response.append(0x00)  # Control
        response.extend(struct.pack('<H', self.config.rtu_id))  # Source
        response.extend(struct.pack('<H', 1))  # Destination
        response.extend(b'\x00\x00')  # Header CRC
        response.append(DNP3FunctionCode.RESPONSE.value)
        response.extend(b'\x00\x00\x00')  # Data CRC
        
        return bytes(response)
    
    async def _measurement_update_loop(self):
        """Continuously update measurements from power system"""
        while self.is_running:
            try:
                # Update measurements from power system
                await self._update_measurements()
                
                # Check for events and generate alarms
                await self._check_events()
                
                await asyncio.sleep(self.config.update_rate)
                
            except Exception as e:
                logger.error(f"Error in measurement update loop: {e}")
                if self.is_running:
                    await asyncio.sleep(1)
    
    async def _update_measurements(self):
        """Update measurement values from IEEE 39-bus system"""
        if not self.power_system:
            # Generate simulated measurements for testing
            self._generate_simulated_measurements()
            return
        
        try:
            # Get current system state
            system_state = self.power_system.get_system_state()
            bus_idx = self.config.bus_number - 1  # Convert to 0-based index
            current_time = time.time()
            
            # Update voltage measurements - handle both dict and object access
            bus_voltages = None
            if isinstance(system_state, dict) and 'bus_voltages' in system_state:
                bus_voltages = system_state['bus_voltages']
            elif hasattr(system_state, 'bus_voltages'):
                bus_voltages = system_state.bus_voltages
            
            if bus_voltages and len(bus_voltages) > bus_idx:
                voltage_pu = bus_voltages[bus_idx]
                voltage_kv = voltage_pu * 345.0  # Assuming 345 kV base voltage
                
                # Update voltage magnitude measurement
                for mp in self.measurements.values():
                    if mp.name == 'voltage_magnitude':
                        mp.current_value = voltage_kv
                        mp.timestamp = current_time
                        mp.quality = 0x01  # ONLINE
                        break
            
            # Update frequency measurement - handle both dict and object access
            frequency_hz = None
            if isinstance(system_state, dict) and 'frequency_hz' in system_state:
                frequency_hz = system_state['frequency_hz']
            elif hasattr(system_state, 'frequency_hz'):
                frequency_hz = system_state.frequency_hz
            
            if frequency_hz:
                for mp in self.measurements.values():
                    if mp.name == 'frequency':
                        mp.current_value = frequency_hz
                        mp.timestamp = current_time
                        mp.quality = 0x01  # ONLINE
                        break
            
            # Update power measurements using actual power system data
            power_flow_converged = False
            if isinstance(system_state, dict):
                power_flow_converged = system_state.get('power_flow_converged', False)
            elif hasattr(system_state, 'power_flow_converged'):
                power_flow_converged = system_state.power_flow_converged
            elif hasattr(self.power_system, 'power_flow_solved'):
                power_flow_converged = self.power_system.power_flow_solved
            
            if power_flow_converged and hasattr(self.power_system, 'gen_results'):
                # Check if this bus has a generator
                for gen_idx, (gen_name, gen_data) in enumerate(self.power_system.ieee_generators.items()):
                    if gen_data['bus'] == self.config.bus_number:
                        # Get generator power output
                        if hasattr(self.power_system, 'gen_results') and len(self.power_system.gen_results) > gen_idx:
                            p_gen_mw = self.power_system.gen_results[gen_idx, 1]  # PG column
                            q_gen_mvar = self.power_system.gen_results[gen_idx, 2]  # QG column
                            
                            for mp in self.measurements.values():
                                if mp.name == 'active_power':
                                    mp.current_value = p_gen_mw
                                    mp.timestamp = current_time
                                    mp.quality = 0x01
                                elif mp.name == 'reactive_power':
                                    mp.current_value = q_gen_mvar
                                    mp.timestamp = current_time
                                    mp.quality = 0x01
                            break
            
            # For load buses, estimate power consumption
            if hasattr(self.power_system, 'bus_results') and len(self.power_system.bus_results) > bus_idx:
                load_p = abs(self.power_system.bus_results[bus_idx, 2])  # PD column
                load_q = abs(self.power_system.bus_results[bus_idx, 3])  # QD column
                
                if load_p > 0:  # This is a load bus
                    for mp in self.measurements.values():
                        if mp.name == 'active_power' and mp.current_value == 0.0:  # Not set by generator
                            mp.current_value = -load_p  # Negative for load consumption
                            mp.timestamp = current_time
                            mp.quality = 0x01
                        elif mp.name == 'reactive_power' and mp.current_value == 0.0:
                            mp.current_value = -load_q
                            mp.timestamp = current_time
                            mp.quality = 0x01
            
            # Update breaker status (assume all breakers are closed in normal operation)
            for mp in self.measurements.values():
                if mp.name.startswith('breaker') and mp.object_group == DNP3ObjectGroup.BINARY_INPUT:
                    mp.current_value = 1.0  # Closed
                    mp.timestamp = current_time
                    mp.quality = 0x01
            
            logger.debug(f"RTU {self.config.rtu_id}: Updated measurements from power system")
        
        except Exception as e:
            logger.warning(f"RTU {self.config.rtu_id}: Error updating real measurements, using simulated: {e}")
            self._generate_simulated_measurements()
    
    def _generate_simulated_measurements(self):
        """Generate simulated measurements for testing"""
        import random
        current_time = time.time()
        
        for mp in self.measurements.values():
            if mp.object_group == DNP3ObjectGroup.ANALOG_INPUT:
                if mp.name == 'voltage_magnitude':
                    # Voltage: 330-350 kV (±5% variation)
                    mp.current_value = 345.0 + random.uniform(-17.25, 17.25)
                elif mp.name == 'frequency':
                    # Frequency: 49.8-50.2 Hz
                    mp.current_value = 50.0 + random.uniform(-0.2, 0.2)
                elif mp.name == 'active_power':
                    # Active power: 0-200 MW
                    mp.current_value = random.uniform(0, 200)
                elif mp.name == 'reactive_power':
                    # Reactive power: -50 to +50 MVAR
                    mp.current_value = random.uniform(-50, 50)
                else:
                    # Generic analog measurement
                    mp.current_value = random.uniform(0, 100)
                
                mp.timestamp = current_time
            
            elif mp.object_group == DNP3ObjectGroup.BINARY_INPUT:
                if mp.name.startswith('breaker'):
                    # Breaker status: mostly closed (90% probability)
                    mp.current_value = 1.0 if random.random() > 0.1 else 0.0
                else:
                    # Generic binary input
                    mp.current_value = 1.0 if random.random() > 0.5 else 0.0
                
                mp.timestamp = current_time
    
    async def _check_events(self):
        """Check for events and generate alarms"""
        current_time = time.time()
        
        for mp in self.measurements.values():
            # Check for alarm conditions
            alarm_generated = False
            
            if mp.name == 'voltage_magnitude':
                # Voltage alarm: outside 320-370 kV range
                if mp.current_value < 320.0 or mp.current_value > 370.0:
                    self._generate_event('voltage_alarm', f"Voltage {mp.current_value:.1f} kV outside limits", mp)
                    alarm_generated = True
            
            elif mp.name == 'frequency':
                # Frequency alarm: outside 49.5-50.5 Hz range
                if mp.current_value < 49.5 or mp.current_value > 50.5:
                    self._generate_event('frequency_alarm', f"Frequency {mp.current_value:.3f} Hz outside limits", mp)
                    alarm_generated = True
            
            # Update quality based on alarm status
            mp.quality = 0x01 if not alarm_generated else 0x10  # ONLINE or ALARM
    
    def _generate_event(self, event_type: str, description: str, measurement: MeasurementPoint):
        """Generate event for unsolicited reporting"""
        event = {
            'timestamp': time.time(),
            'rtu_id': self.config.rtu_id,
            'bus_number': self.config.bus_number,
            'event_type': event_type,
            'description': description,
            'measurement_index': measurement.index,
            'measurement_name': measurement.name,
            'current_value': measurement.current_value,
            'quality': measurement.quality
        }
        
        self.event_buffer.append(event)
        self.stats['events_generated'] += 1
        
        # Limit event buffer size
        if len(self.event_buffer) > self.max_events:
            self.event_buffer.pop(0)
        
        logger.warning(f"RTU {self.config.rtu_id} EVENT: {description}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get RTU status information"""
        uptime = time.time() - self.stats['uptime_start']
        
        return {
            'rtu_id': self.config.rtu_id,
            'bus_number': self.config.bus_number,
            'name': self.config.name,
            'is_running': self.is_running,
            'ip_address': self.config.ip_address,
            'port': self.config.port,
            'connected_clients': len(self.clients),
            'measurement_points': len(self.measurements),
            'statistics': {
                **self.stats,
                'uptime_seconds': uptime,
                'events_pending': len(self.event_buffer)
            },
            'recent_measurements': {
                mp.name: {
                    'value': mp.current_value,
                    'unit': mp.unit,
                    'quality': mp.quality,
                    'timestamp': mp.timestamp
                }
                for mp in list(self.measurements.values())[:5]  # Show first 5
            }
        }

def create_ieee39_rtu_configurations() -> List[RTUConfiguration]:
    """
    Create RTU configurations for key buses in IEEE 39-bus system.
    
    NOTE: All RTUs run on localhost (127.0.0.1) with different ports.
    Each RTU represents a physical outstation at a specific IEEE 39-bus location.
    """
    rtu_configs = []
    
    # Strategic RTU placement at critical IEEE 39-bus system buses
    rtu_locations = [
        # Generation buses (where generators are connected in IEEE 39-bus)
        {'bus': 30, 'name': 'Gen_30_RTU'},   # Generator bus 30
        {'bus': 31, 'name': 'Gen_31_RTU'},   # Generator bus 31  
        {'bus': 32, 'name': 'Gen_32_RTU'},   # Generator bus 32
        {'bus': 33, 'name': 'Gen_33_RTU'},   # Generator bus 33
        {'bus': 39, 'name': 'Gen_39_RTU'},   # Generator bus 39
        
        # Critical transmission buses (high voltage interconnections)
        {'bus': 16, 'name': 'Trans_16_RTU'}, # Transmission hub
        {'bus': 21, 'name': 'Trans_21_RTU'}, # Transmission hub
        {'bus': 25, 'name': 'Trans_25_RTU'}, # Transmission hub
        
        # Major load centers (high load buses in IEEE 39-bus)
        {'bus': 4, 'name': 'Load_04_RTU'},   # Load center (500 MW)
        {'bus': 20, 'name': 'Load_20_RTU'}   # Load center (680 MW)
    ]
    
    for i, location in enumerate(rtu_locations):
        # Create measurement points for each RTU
        measurement_points = []
        
        # Voltage measurement
        measurement_points.append(MeasurementPoint(
            index=1,
            name='voltage_magnitude',
            object_group=DNP3ObjectGroup.ANALOG_INPUT,
            data_type='analog_float',
            unit='kV',
            scale_factor=1.0
        ))
        
        # Frequency measurement
        measurement_points.append(MeasurementPoint(
            index=2,
            name='frequency',
            object_group=DNP3ObjectGroup.ANALOG_INPUT,
            data_type='analog_float',
            unit='Hz',
            scale_factor=1.0
        ))
        
        # Active power measurement
        measurement_points.append(MeasurementPoint(
            index=3,
            name='active_power',
            object_group=DNP3ObjectGroup.ANALOG_INPUT,
            data_type='analog_float',
            unit='MW',
            scale_factor=1.0
        ))
        
        # Reactive power measurement
        measurement_points.append(MeasurementPoint(
            index=4,
            name='reactive_power',
            object_group=DNP3ObjectGroup.ANALOG_INPUT,
            data_type='analog_float',
            unit='MVAR',
            scale_factor=1.0
        ))
        
        # Breaker status (binary)
        measurement_points.append(MeasurementPoint(
            index=5,
            name='breaker_status',
            object_group=DNP3ObjectGroup.BINARY_INPUT,
            data_type='binary',
            unit='status',
            scale_factor=1.0
        ))
        
        # Create RTU configuration
        # NOTE: All RTUs use localhost with different ports for simulation
        config = RTUConfiguration(
            rtu_id=i + 1,
            bus_number=location['bus'],               # IEEE 39-bus system bus number
            name=location['name'],
            ip_address='127.0.0.1',                   # Localhost for simulation
            port=DNP3_PORT + i,                       # Different port per RTU
            update_rate=1.0,                          # 1 second update rate
            measurement_points=measurement_points
        )
        
        rtu_configs.append(config)
    
    return rtu_configs

# Test function
async def test_rtu():
    """Test RTU functionality"""
    print("🧪 Testing RTU Implementation")
    
    # Create test RTU configuration
    test_config = RTUConfiguration(
        rtu_id=1,
        bus_number=16,
        name='Test_RTU',
        ip_address='127.0.0.1',
        port=20001,
        update_rate=2.0,
        measurement_points=[
            MeasurementPoint(1, 'voltage', DNP3ObjectGroup.ANALOG_INPUT, 'analog_float', 'kV'),
            MeasurementPoint(2, 'frequency', DNP3ObjectGroup.ANALOG_INPUT, 'analog_float', 'Hz'),
            MeasurementPoint(3, 'breaker', DNP3ObjectGroup.BINARY_INPUT, 'binary', 'status'),
        ]
    )
    
    # Create and start RTU
    rtu = IEEE39RTU(test_config)
    
    try:
        print(f"Starting RTU {test_config.rtu_id}...")
        
        # Start RTU in background
        rtu_task = asyncio.create_task(rtu.start())
        
        # Let it run for a bit
        await asyncio.sleep(5)
        
        # Print status
        status = rtu.get_status()
        print(f"RTU Status: {json.dumps(status, indent=2)}")
        
        # Stop RTU
        await rtu.stop()
        
        print("✅ RTU test completed successfully")
        
    except Exception as e:
        print(f"❌ RTU test failed: {e}")
        await rtu.stop()

if __name__ == "__main__":
    asyncio.run(test_rtu())
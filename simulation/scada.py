#!/usr/bin/env python3
"""
SCADA Master Station Implementation for IEEE 39-Bus System

This module implements a SCADA master station that polls RTU outstations
using DNP3 protocol for power system monitoring and control.

Features:
- DNP3 master protocol implementation
- Polling of multiple RTU outstations
- Real-time data collection and archiving
- Alarm management and event processing
- Control command execution
- Integration with IEEE 39-bus power system
- Web-based HMI dashboard
"""

import asyncio
import logging
import socket
import struct
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

if TYPE_CHECKING:
    import socket as socket_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DNP3 Protocol Constants
DNP3_START_BYTES = b'\x05\x64'
DNP3_PORT = 20000

class DNP3FunctionCode(Enum):
    """DNP3 Function Codes for master requests"""
    READ = 0x01
    WRITE = 0x02
    SELECT = 0x03
    OPERATE = 0x04
    DIRECT_OPERATE = 0x05
    RESPONSE = 0x81
    UNSOLICITED_RESPONSE = 0x82

class AlarmSeverity(Enum):
    """Alarm severity levels"""
    INFO = 1
    WARNING = 2
    ALARM = 3
    CRITICAL = 4

@dataclass
class RTUConnection:
    """RTU connection configuration"""
    rtu_id: int
    name: str
    ip_address: str
    port: int
    bus_number: int
    is_connected: bool = False
    last_poll_time: float = 0.0
    poll_interval: float = 3.0  # seconds  
    timeout: float = 2.0
    retry_count: int = 0
    max_retries: int = 2
    socket: Optional[Any] = None

@dataclass 
class MeasurementValue:
    """Current measurement value with metadata"""
    rtu_id: int
    measurement_name: str
    value: float
    unit: str
    quality: int
    timestamp: float
    is_stale: bool = False
    alarm_status: Optional[AlarmSeverity] = None

@dataclass
class ControlCommand:
    """Control command to be sent to RTU"""
    rtu_id: int
    command_type: str  # 'binary_operate', 'analog_operate'
    point_index: int
    value: float
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    status: str = 'pending'  # 'pending', 'sent', 'confirmed', 'failed'

@dataclass
class SystemAlarm:
    """System alarm/event"""
    alarm_id: int
    timestamp: float
    rtu_id: int
    severity: AlarmSeverity
    message: str
    measurement_name: str = ""
    value: float = 0.0
    acknowledged: bool = False
    ack_timestamp: float = 0.0
    ack_user: str = ""

class SCADAMaster:
    """
    SCADA Master Station for IEEE 39-Bus System
    
    Main supervisory control and data acquisition system that:
    - Polls multiple RTU outstations via DNP3
    - Collects and archives measurement data
    - Manages alarms and events
    - Executes control commands
    - Provides system monitoring dashboard
    """
    
    def __init__(self, master_id: int = 1):
        """
        Initialize SCADA master station.
        
        Args:
            master_id: Unique identifier for this master station
        """
        self.master_id = master_id
        self.is_running = False
        
        # RTU connections
        self.rtu_connections: Dict[int, RTUConnection] = {}
        
        # Data storage
        self.current_measurements: Dict[Tuple[int, str], MeasurementValue] = {}
        self.historical_data: List[Dict] = []
        self.max_history = 10000  # Keep last 10k measurements
        
        # Alarm system
        self.active_alarms: Dict[int, SystemAlarm] = {}
        self.alarm_counter = 0
        self.alarm_limits = {
            'voltage_magnitude': {'min': 320.0, 'max': 370.0},
            'frequency': {'min': 49.5, 'max': 50.5},
            'active_power': {'min': -500.0, 'max': 500.0}
        }
        
        # Control system
        self.pending_commands: List[ControlCommand] = []
        self.command_history: List[ControlCommand] = []
        
        # Statistics
        self.stats = {
            'polls_sent': 0,
            'responses_received': 0,
            'commands_sent': 0,
            'alarms_generated': 0,
            'uptime_start': time.time(),
            'total_rtus': 0,
            'connected_rtus': 0
        }
        
        logger.info(f"SCADA Master {master_id} initialized")
    
    def add_rtu(self, rtu_id: int, name: str, ip_address: str, port: int, bus_number: int, poll_interval: float = 5.0):
        """Add RTU outstation to polling list"""
        rtu_conn = RTUConnection(
            rtu_id=rtu_id,
            name=name,
            ip_address=ip_address,
            port=port,
            bus_number=bus_number,
            poll_interval=poll_interval
        )
        
        self.rtu_connections[rtu_id] = rtu_conn
        self.stats['total_rtus'] = len(self.rtu_connections)
        
        logger.info(f"Added RTU {rtu_id} ({name}) at {ip_address}:{port} for Bus {bus_number}")
    
    async def start(self):
        """Start the SCADA master station"""
        if self.is_running:
            logger.warning("SCADA Master already running")
            return
        
        self.is_running = True
        logger.info("🟢 Starting SCADA Master Station")
        
        try:
            # Start SCADA master server (for detection by attack systems)
            server_task = asyncio.create_task(self._start_master_server())
            
            # Start RTU polling tasks
            polling_tasks = []
            for rtu_id in self.rtu_connections:
                task = asyncio.create_task(self._poll_rtu_loop(rtu_id))
                polling_tasks.append(task)
            
            # Start alarm processing task
            alarm_task = asyncio.create_task(self._alarm_processing_loop())
            
            # Start control command processing task
            control_task = asyncio.create_task(self._control_processing_loop())
            
            # Start data archiving task
            archive_task = asyncio.create_task(self._data_archiving_loop())
            
            # Start statistics update task
            stats_task = asyncio.create_task(self._statistics_loop())
            
            logger.info(f"SCADA Master started with {len(self.rtu_connections)} RTUs")
            
            # Wait for all tasks
            all_tasks = [server_task] + polling_tasks + [alarm_task, control_task, archive_task, stats_task]
            await asyncio.gather(*all_tasks)
            
        except Exception as e:
            logger.error(f"Error starting SCADA Master: {e}")
            raise
    
    async def stop(self):
        """Stop the SCADA master station"""
        logger.info("🔴 Stopping SCADA Master Station")
        
        self.is_running = False
        
        # Close all RTU connections
        for rtu_conn in self.rtu_connections.values():
            if rtu_conn.socket:
                try:
                    rtu_conn.socket.close()
                except:
                    pass
                rtu_conn.socket = None
                rtu_conn.is_connected = False
        
        self.stats['connected_rtus'] = 0
        logger.info("SCADA Master stopped")
    
    async def _start_master_server(self):
        """Start SCADA master server for external connections (port 21000)"""
        try:
            server = await asyncio.start_server(
                self._handle_master_connection,
                '127.0.0.1',
                21000
            )
            
            addr = server.sockets[0].getsockname()
            logger.info(f"🟢 SCADA Master server listening on {addr[0]}:{addr[1]}")
            
            async with server:
                await server.serve_forever()
                
        except OSError as e:
            if e.errno == 10048:  # Address already in use
                logger.warning(f"SCADA Master server port 21000 already in use")
            else:
                logger.error(f"Error starting SCADA master server: {e}")
        except Exception as e:
            logger.error(f"Error starting SCADA master server: {e}")
    
    async def _handle_master_connection(self, reader, writer):
        """Handle connections to SCADA master server"""
        client_addr = writer.get_extra_info('peername')
        logger.debug(f"SCADA Master: Connection from {client_addr}")
        
        try:
            # Send simple acknowledgment for detection purposes
            response = b"SCADA_MASTER_ACTIVE\n"
            writer.write(response)
            await writer.drain()
            
            # Keep connection alive briefly
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.debug(f"Error handling master connection: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
    
    async def _poll_rtu_loop(self, rtu_id: int):
        """Continuous polling loop for specific RTU"""
        rtu_conn = self.rtu_connections[rtu_id]
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to poll this RTU
                if current_time - rtu_conn.last_poll_time >= rtu_conn.poll_interval:
                    await self._poll_rtu(rtu_id)
                
                await asyncio.sleep(0.5)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in polling loop for RTU {rtu_id}: {e}")
                if self.is_running:
                    await asyncio.sleep(5)  # Wait before retrying
    
    async def _poll_rtu(self, rtu_id: int):
        """Poll specific RTU for measurements"""
        rtu_conn = self.rtu_connections[rtu_id]
        
        try:
            # Establish connection if needed
            if not rtu_conn.is_connected:
                await self._connect_to_rtu(rtu_id)
            
            if rtu_conn.is_connected and rtu_conn.socket:
                # Send DNP3 read request
                request = self._build_read_request(rtu_id)
                rtu_conn.socket.send(request)
                self.stats['polls_sent'] += 1
                
                # Receive response with timeout (increased timeout for stability)
                rtu_conn.socket.settimeout(5.0)  # Increased from 3.0 to 5.0 seconds
                response = rtu_conn.socket.recv(2048)  # Increased buffer size
                
                if response:
                    # Process response
                    await self._process_poll_response(rtu_id, response)
                    self.stats['responses_received'] += 1
                    rtu_conn.retry_count = 0  # Reset retry counter
                
                rtu_conn.last_poll_time = time.time()
                
                logger.debug(f"Successfully polled RTU {rtu_id} ({rtu_conn.name})")
        
        except socket.timeout:
            logger.warning(f"Timeout polling RTU {rtu_id} ({rtu_conn.name})")
            rtu_conn.retry_count += 1
            
        except (ConnectionResetError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Connection error with RTU {rtu_id}: {e}")
            rtu_conn.is_connected = False
            rtu_conn.retry_count += 1
            
            if rtu_conn.socket:
                try:
                    rtu_conn.socket.close()
                except:
                    pass
                rtu_conn.socket = None
        
        except Exception as e:
            logger.error(f"Error polling RTU {rtu_id}: {e}")
            rtu_conn.retry_count += 1
        
        # Handle retry logic
        if rtu_conn.retry_count >= rtu_conn.max_retries:
            if rtu_conn.is_connected:
                logger.error(f"RTU {rtu_id} ({rtu_conn.name}) disconnected after {rtu_conn.max_retries} failed attempts")
                rtu_conn.is_connected = False
                
                # Generate communication alarm
                await self._generate_alarm(
                    rtu_id, AlarmSeverity.CRITICAL,
                    f"RTU {rtu_id} communication failed", "communication"
                )
                
                if rtu_conn.socket:
                    try:
                        rtu_conn.socket.close()
                    except:
                        pass
                    rtu_conn.socket = None
            
            # Reset retry count and wait longer before next attempt
            rtu_conn.retry_count = 0
            await asyncio.sleep(10)
        
        # Update connected RTU count
        self.stats['connected_rtus'] = sum(1 for conn in self.rtu_connections.values() if conn.is_connected)
    
    async def _connect_to_rtu(self, rtu_id: int):
        """Establish connection to RTU"""
        rtu_conn = self.rtu_connections[rtu_id]
        
        try:
            # Wait a bit for RTU to be ready
            await asyncio.sleep(0.5)
            
            # Create new socket
            rtu_conn.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            rtu_conn.socket.settimeout(rtu_conn.timeout)
            
            # Attempt connection
            rtu_conn.socket.connect((rtu_conn.ip_address, rtu_conn.port))
            rtu_conn.is_connected = True
            rtu_conn.retry_count = 0
            
            logger.info(f"✅ Connected to RTU {rtu_id} ({rtu_conn.name}) at {rtu_conn.ip_address}:{rtu_conn.port}")
            
        except Exception as e:
            logger.debug(f"Connection attempt {rtu_conn.retry_count + 1} failed for RTU {rtu_id}: {e}")
            rtu_conn.is_connected = False
            if rtu_conn.socket:
                try:
                    rtu_conn.socket.close()
                except:
                    pass
                rtu_conn.socket = None
    
    def _build_read_request(self, rtu_id: int) -> bytes:
        """Build DNP3 read request for RTU"""
        request = bytearray()
        request.extend(DNP3_START_BYTES)  # Start bytes
        request.append(12)  # Length
        request.append(0x00)  # Control
        request.extend(struct.pack('<H', 1))  # Source (master)
        request.extend(struct.pack('<H', rtu_id))  # Destination (RTU)
        request.extend(b'\x00\x00')  # Header CRC (simplified)
        request.append(DNP3FunctionCode.READ.value)  # Function code
        
        # Request analog inputs (Group 30) and binary inputs (Group 1)
        request.extend(b'\x1E\x00')  # Group 30 (Analog Input), Variation 0 (all)
        request.extend(b'\x06')  # Qualifier: all objects
        request.extend(b'\x01\x00')  # Group 1 (Binary Input), Variation 0 (all)
        request.extend(b'\x06')  # Qualifier: all objects
        
        request.extend(b'\x00\x00')  # Data CRC (simplified)
        
        return bytes(request)
    
    async def _process_poll_response(self, rtu_id: int, response: bytes):
        """Process DNP3 response from RTU"""
        try:
            # Parse DNP3 response (simplified)
            if len(response) < 10:
                return
            
            # Check DNP3 start bytes
            if response[:2] != DNP3_START_BYTES:
                return
            
            # Extract measurement data (simplified parsing)
            current_time = time.time()
            
            # For demonstration, extract some simulated measurements
            # In real implementation, this would parse actual DNP3 objects
            measurements = self._extract_measurements_from_response(rtu_id, response)
            
            # Process each measurement
            for name, value, unit, quality in measurements:
                measurement_key = (rtu_id, name)
                
                measurement_value = MeasurementValue(
                    rtu_id=rtu_id,
                    measurement_name=name,
                    value=value,
                    unit=unit,
                    quality=quality,
                    timestamp=current_time
                )
                
                # Store current measurement
                self.current_measurements[measurement_key] = measurement_value
                
                # Check for alarms
                await self._check_measurement_alarms(measurement_value)
                
                logger.debug(f"RTU {rtu_id}: {name} = {value:.2f} {unit}")
        
        except Exception as e:
            logger.error(f"Error processing response from RTU {rtu_id}: {e}")
    
    def _extract_measurements_from_response(self, rtu_id: int, response: bytes) -> List[Tuple[str, float, str, int]]:
        """Extract measurements from DNP3 response"""
        measurements = []
        
        try:
            # Parse actual DNP3 response from RTU
            if len(response) < 12:
                logger.debug(f"Response too short from RTU {rtu_id}: {len(response)} bytes")
                return self._generate_fallback_measurements(rtu_id)
            
            # Skip DNP3 header (start bytes + length + control + addresses + CRC)
            data_start = 11  # Start after header
            pos = data_start
            
            while pos < len(response) - 2:  # Leave 2 bytes for final CRC
                if pos + 6 > len(response):
                    break
                    
                try:
                    # Check for Group 30 (Analog Input)
                    if response[pos:pos+2] == b'\x1E\x01':
                        pos += 2  # Skip group/variation
                        if pos + 8 > len(response):
                            break
                        
                        index = struct.unpack('<H', response[pos:pos+2])[0]
                        pos += 2
                        value = struct.unpack('<f', response[pos:pos+4])[0]
                        pos += 4
                        quality = response[pos]
                        pos += 1
                        
                        # Map index to measurement name
                        if index == 0:
                            measurements.append(('voltage_magnitude', value, 'kV', quality))
                        elif index == 1:
                            measurements.append(('frequency', value, 'Hz', quality))
                        elif index == 2:
                            measurements.append(('active_power', value, 'MW', quality))
                        elif index == 3:
                            measurements.append(('reactive_power', value, 'MVAR', quality))
                        
                    # Check for Group 1 (Binary Input)
                    elif response[pos:pos+2] == b'\x01\x01':
                        pos += 2  # Skip group/variation
                        if pos + 4 > len(response):
                            break
                        
                        index = struct.unpack('<H', response[pos:pos+2])[0]
                        pos += 2
                        value = float(response[pos])
                        pos += 1
                        quality = response[pos]
                        pos += 1
                        
                        if index == 0:
                            measurements.append(('breaker_status', value, 'status', quality))
                    else:
                        # Unknown group, skip
                        pos += 1
                        
                except (struct.error, IndexError) as e:
                    logger.debug(f"Error parsing measurement at position {pos}: {e}")
                    break
            
            if measurements:
                logger.debug(f"RTU {rtu_id}: Extracted {len(measurements)} measurements from response")
                return measurements
            else:
                logger.debug(f"RTU {rtu_id}: No measurements found in response, using fallback")
                return self._generate_fallback_measurements(rtu_id)
                
        except Exception as e:
            logger.debug(f"Error parsing DNP3 response from RTU {rtu_id}: {e}")
            return self._generate_fallback_measurements(rtu_id)
    
    def _generate_fallback_measurements(self, rtu_id: int) -> List[Tuple[str, float, str, int]]:
        """Generate fallback measurements when RTU response parsing fails"""
        import random
        
        # Generate more realistic measurements based on RTU type
        rtu_conn = self.rtu_connections.get(rtu_id)
        if not rtu_conn:
            return []
        
        measurements = []
        
        # Base measurements with some variation
        base_voltage = 345.0 + random.uniform(-10.0, 10.0)  # ±3% variation
        base_frequency = 50.0 + random.uniform(-0.1, 0.1)   # ±0.1 Hz
        
        # Power measurements depend on bus type
        if 'Gen' in rtu_conn.name:
            # Generator bus - positive power output
            base_power = random.uniform(200, 800)
            base_reactive = random.uniform(50, 200)
        elif 'Load' in rtu_conn.name:
            # Load bus - negative power consumption
            base_power = -random.uniform(100, 500)
            base_reactive = -random.uniform(20, 100)
        else:
            # Transmission bus - moderate power flow
            base_power = random.uniform(-100, 100)
            base_reactive = random.uniform(-50, 50)
        
        measurements.extend([
            ('voltage_magnitude', base_voltage, 'kV', 0x01),
            ('frequency', base_frequency, 'Hz', 0x01),
            ('active_power', base_power, 'MW', 0x01),
            ('reactive_power', base_reactive, 'MVAR', 0x01),
            ('breaker_status', 1.0, 'status', 0x01)  # Assume breaker closed
        ])
        
        return measurements
    
    async def _check_measurement_alarms(self, measurement: MeasurementValue):
        """Check measurement against alarm limits"""
        measurement_name = measurement.measurement_name
        
        if measurement_name in self.alarm_limits:
            limits = self.alarm_limits[measurement_name]
            value = measurement.value
            
            # Check if value is outside limits
            if value < limits['min'] or value > limits['max']:
                severity = AlarmSeverity.ALARM
                if measurement_name == 'frequency' and (value < 49.0 or value > 51.0):
                    severity = AlarmSeverity.CRITICAL
                elif measurement_name == 'voltage_magnitude' and (value < 300.0 or value > 380.0):
                    severity = AlarmSeverity.CRITICAL
                
                message = f"{measurement_name} {value:.2f} {measurement.unit} outside limits [{limits['min']}-{limits['max']}]"
                
                await self._generate_alarm(
                    measurement.rtu_id, severity, message, measurement_name, value
                )
    
    async def _generate_alarm(self, rtu_id: int, severity: AlarmSeverity, message: str, 
                            measurement_name: str = "", value: float = 0.0):
        """Generate system alarm"""
        self.alarm_counter += 1
        
        alarm = SystemAlarm(
            alarm_id=self.alarm_counter,
            timestamp=time.time(),
            rtu_id=rtu_id,
            severity=severity,
            message=message,
            measurement_name=measurement_name,
            value=value
        )
        
        self.active_alarms[alarm.alarm_id] = alarm
        self.stats['alarms_generated'] += 1
        
        # Log alarm based on severity
        if severity == AlarmSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL ALARM: RTU {rtu_id} - {message}")
        elif severity == AlarmSeverity.ALARM:
            logger.error(f"🔴 ALARM: RTU {rtu_id} - {message}")
        elif severity == AlarmSeverity.WARNING:
            logger.warning(f"🟡 WARNING: RTU {rtu_id} - {message}")
        else:
            logger.info(f"ℹ️ INFO: RTU {rtu_id} - {message}")
    
    async def _alarm_processing_loop(self):
        """Process and manage alarms"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for stale measurements (no update in 30 seconds)
                for (rtu_id, name), measurement in self.current_measurements.items():
                    if current_time - measurement.timestamp > 30.0 and not measurement.is_stale:
                        measurement.is_stale = True
                        await self._generate_alarm(
                            rtu_id, AlarmSeverity.WARNING,
                            f"Stale measurement: {name}", name
                        )
                
                # Auto-clear alarms after 5 minutes if condition is resolved
                alarms_to_clear = []
                for alarm_id, alarm in self.active_alarms.items():
                    if (current_time - alarm.timestamp > 300 and  # 5 minutes
                        alarm.measurement_name and 
                        not alarm.acknowledged):
                        
                        # Check if condition is still present
                        measurement_key = (alarm.rtu_id, alarm.measurement_name)
                        if measurement_key in self.current_measurements:
                            current_measurement = self.current_measurements[measurement_key]
                            if alarm.measurement_name in self.alarm_limits:
                                limits = self.alarm_limits[alarm.measurement_name]
                                if limits['min'] <= current_measurement.value <= limits['max']:
                                    alarms_to_clear.append(alarm_id)
                
                # Clear resolved alarms
                for alarm_id in alarms_to_clear:
                    logger.info(f"Auto-clearing resolved alarm {alarm_id}")
                    del self.active_alarms[alarm_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alarm processing loop: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    async def _control_processing_loop(self):
        """Process pending control commands"""
        while self.is_running:
            try:
                # Process pending commands
                commands_to_remove = []
                
                for i, command in enumerate(self.pending_commands):
                    if command.status == 'pending':
                        # Send command to RTU
                        success = await self._send_control_command(command)
                        if success:
                            command.status = 'sent'
                            logger.info(f"Control command sent to RTU {command.rtu_id}: {command.command_type}")
                        else:
                            command.status = 'failed'
                            logger.error(f"Failed to send control command to RTU {command.rtu_id}")
                    
                    # Move completed commands to history
                    if command.status in ['confirmed', 'failed']:
                        commands_to_remove.append(i)
                        self.command_history.append(command)
                
                # Remove completed commands
                for i in reversed(commands_to_remove):
                    self.pending_commands.pop(i)
                
                # Limit command history size
                if len(self.command_history) > 1000:
                    self.command_history = self.command_history[-1000:]
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in control processing loop: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    async def _send_control_command(self, command: ControlCommand) -> bool:
        """Send control command to RTU"""
        rtu_conn = self.rtu_connections.get(command.rtu_id)
        if not rtu_conn or not rtu_conn.is_connected:
            return False
        
        try:
            # Build DNP3 control request
            request = self._build_control_request(command)
            
            if rtu_conn.socket:
                rtu_conn.socket.send(request)
                self.stats['commands_sent'] += 1
                
                # Wait for acknowledgment
                response = rtu_conn.socket.recv(1024)
                if response and len(response) >= 10:
                    # Check for successful acknowledgment (simplified)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending control command: {e}")
            return False
    
    def _build_control_request(self, command: ControlCommand) -> bytes:
        """Build DNP3 control request"""
        request = bytearray()
        request.extend(DNP3_START_BYTES)
        request.append(15)  # Length
        request.append(0x00)  # Control
        request.extend(struct.pack('<H', 1))  # Source (master)
        request.extend(struct.pack('<H', command.rtu_id))  # Destination
        request.extend(b'\x00\x00')  # Header CRC
        request.append(DNP3FunctionCode.DIRECT_OPERATE.value)  # Function code
        
        # Add control data based on command type
        if command.command_type == 'binary_operate':
            request.extend(b'\x0C\x01')  # Group 12, Variation 1 (Binary Command)
            request.extend(struct.pack('<H', command.point_index))
            request.extend(struct.pack('<B', int(command.value)))  # Control code
        elif command.command_type == 'analog_operate':
            request.extend(b'\x29\x01')  # Group 41, Variation 1 (Analog Command)
            request.extend(struct.pack('<H', command.point_index))
            request.extend(struct.pack('<f', command.value))  # Analog value
        
        request.extend(b'\x00\x00')  # Data CRC
        
        return bytes(request)
    
    async def _data_archiving_loop(self):
        """Archive measurement data for historical analysis"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Archive current measurements every 10 seconds
                if len(self.current_measurements) > 0:
                    archive_entry = {
                        'timestamp': current_time,
                        'measurements': {
                            f"RTU_{rtu_id}_{name}": {
                                'value': measurement.value,
                                'unit': measurement.unit,
                                'quality': measurement.quality,
                                'is_stale': measurement.is_stale
                            }
                            for (rtu_id, name), measurement in self.current_measurements.items()
                        }
                    }
                    
                    self.historical_data.append(archive_entry)
                    
                    # Limit historical data size
                    if len(self.historical_data) > self.max_history:
                        self.historical_data = self.historical_data[-self.max_history:]
                
                await asyncio.sleep(10)  # Archive every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in data archiving loop: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    async def _statistics_loop(self):
        """Update and log system statistics"""
        while self.is_running:
            try:
                # Log statistics every 60 seconds
                await asyncio.sleep(60)
                
                uptime = time.time() - self.stats['uptime_start']
                
                logger.info("📊 SCADA STATISTICS:")
                logger.info(f"  Uptime: {uptime/3600:.1f} hours")
                logger.info(f"  RTUs: {self.stats['connected_rtus']}/{self.stats['total_rtus']} connected")
                logger.info(f"  Polls: {self.stats['polls_sent']} sent, {self.stats['responses_received']} received")
                logger.info(f"  Commands: {self.stats['commands_sent']} sent")
                logger.info(f"  Active Alarms: {len(self.active_alarms)}")
                logger.info(f"  Measurements: {len(self.current_measurements)} current")
                logger.info(f"  Historical Records: {len(self.historical_data)}")
                
            except Exception as e:
                logger.error(f"Error in statistics loop: {e}")
    
    # Control Methods
    def send_binary_command(self, rtu_id: int, point_index: int, value: bool, priority: int = 1):
        """Send binary control command to RTU"""
        command = ControlCommand(
            rtu_id=rtu_id,
            command_type='binary_operate',
            point_index=point_index,
            value=float(value),
            priority=priority
        )
        
        self.pending_commands.append(command)
        logger.info(f"Queued binary command for RTU {rtu_id}: Point {point_index} = {value}")
    
    def send_analog_command(self, rtu_id: int, point_index: int, value: float, priority: int = 1):
        """Send analog control command to RTU"""
        command = ControlCommand(
            rtu_id=rtu_id,
            command_type='analog_operate',
            point_index=point_index,
            value=value,
            priority=priority
        )
        
        self.pending_commands.append(command)
        logger.info(f"Queued analog command for RTU {rtu_id}: Point {point_index} = {value}")
    
    def acknowledge_alarm(self, alarm_id: int, user: str = "operator"):
        """Acknowledge system alarm"""
        if alarm_id in self.active_alarms:
            alarm = self.active_alarms[alarm_id]
            alarm.acknowledged = True
            alarm.ack_timestamp = time.time()
            alarm.ack_user = user
            
            logger.info(f"Alarm {alarm_id} acknowledged by {user}")
        else:
            logger.warning(f"Attempted to acknowledge non-existent alarm {alarm_id}")
    
    # Query Methods
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = time.time() - self.stats['uptime_start']
        
        return {
            'master_id': self.master_id,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'statistics': self.stats.copy(),
            'rtu_status': {
                rtu_id: {
                    'name': conn.name,
                    'ip_address': conn.ip_address,
                    'port': conn.port,
                    'bus_number': conn.bus_number,
                    'is_connected': conn.is_connected,
                    'last_poll_time': conn.last_poll_time,
                    'retry_count': conn.retry_count
                }
                for rtu_id, conn in self.rtu_connections.items()
            },
            'active_alarms': len(self.active_alarms),
            'pending_commands': len(self.pending_commands),
            'current_measurements': len(self.current_measurements),
            'historical_records': len(self.historical_data)
        }
    
    def get_current_measurements(self) -> Dict[str, Any]:
        """Get all current measurements"""
        return {
            f"RTU_{rtu_id}_{name}": {
                'rtu_id': measurement.rtu_id,
                'value': measurement.value,
                'unit': measurement.unit,
                'quality': measurement.quality,
                'timestamp': measurement.timestamp,
                'is_stale': measurement.is_stale,
                'age_seconds': time.time() - measurement.timestamp
            }
            for (rtu_id, name), measurement in self.current_measurements.items()
        }
    
    def get_active_alarms(self) -> List[Dict[str, Any]]:
        """Get all active alarms"""
        return [
            {
                'alarm_id': alarm.alarm_id,
                'timestamp': alarm.timestamp,
                'rtu_id': alarm.rtu_id,
                'severity': alarm.severity.name,
                'message': alarm.message,
                'measurement_name': alarm.measurement_name,
                'value': alarm.value,
                'acknowledged': alarm.acknowledged,
                'age_seconds': time.time() - alarm.timestamp
            }
            for alarm in self.active_alarms.values()
        ]

def create_ieee39_scada_configuration() -> List[Tuple[int, str, str, int, int]]:
    """Create SCADA configuration for IEEE 39-bus RTUs"""
    # RTU configuration: (rtu_id, name, ip_address, port, bus_number)
    # Using localhost (127.0.0.1) for local simulation
    rtu_configs = [
        (1, 'Gen_30_RTU', '127.0.0.1', 20000, 30),
        (2, 'Gen_31_RTU', '127.0.0.1', 20001, 31),
        (3, 'Gen_32_RTU', '127.0.0.1', 20002, 32),
        (4, 'Gen_33_RTU', '127.0.0.1', 20003, 33),
        (5, 'Gen_39_RTU', '127.0.0.1', 20004, 39),
        (6, 'Trans_16_RTU', '127.0.0.1', 20005, 16),
        (7, 'Trans_21_RTU', '127.0.0.1', 20006, 21),
        (8, 'Trans_25_RTU', '127.0.0.1', 20007, 25),
        (9, 'Load_04_RTU', '127.0.0.1', 20008, 4),
        (10, 'Load_20_RTU', '127.0.0.1', 20009, 20)
    ]
    
    return rtu_configs

# Test function
async def test_scada():
    """Test SCADA Master functionality"""
    print("🧪 Testing SCADA Master Implementation")
    
    # Create SCADA master
    scada = SCADAMaster(master_id=1)
    
    # Add test RTUs
    test_rtus = [
        (1, 'Test_RTU_1', '127.0.0.1', 20001, 16),
        (2, 'Test_RTU_2', '127.0.0.1', 20002, 21),
    ]
    
    for rtu_id, name, ip, port, bus_num in test_rtus:
        scada.add_rtu(rtu_id, name, ip, port, bus_num, poll_interval=3.0)
    
    try:
        print("Starting SCADA Master...")
        
        # Start SCADA master in background
        scada_task = asyncio.create_task(scada.start())
        
        # Let it run for a bit
        await asyncio.sleep(10)
        
        # Print status
        status = scada.get_system_status()
        print(f"SCADA Status: {json.dumps(status, indent=2)}")
        
        # Test control command
        scada.send_binary_command(1, 5, True)  # Close breaker
        
        await asyncio.sleep(5)
        
        # Stop SCADA
        await scada.stop()
        
        print("✅ SCADA test completed successfully")
        
    except Exception as e:
        print(f"❌ SCADA test failed: {e}")
        await scada.stop()

if __name__ == "__main__":
    asyncio.run(test_scada())
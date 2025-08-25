"""
SCADA Master Component

SCADA Master acts as a DNP3 master that:
- Periodically polls RTUs using DNP3 read requests
- Receives measurements and logs them
- Has manual control functions to send control commands to RTUs
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Set
from pydantic import BaseModel
from dataclasses import dataclass

from protocols.dnp3 import (
    DNP3Station, DNP3Protocol, DNP3Message, DNP3DataPoint,
    DNP3FunctionCode, DNP3ObjectGroup, DNP3PollingScheduler
)

logger = logging.getLogger(__name__)


@dataclass
class RTUInfo:
    """Information about an RTU"""
    station_id: int
    name: str
    location: str
    bus_id: int
    last_poll_time: float = 0
    last_response_time: float = 0
    communication_status: str = "Unknown"
    poll_count: int = 0
    error_count: int = 0


class SCADADatabase(BaseModel):
    """SCADA Master Database"""
    measurements: Dict[str, Dict[str, Any]] = {}  # point_id -> measurement data
    alarms: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    
    
class SCADAMasterConfig(BaseModel):
    """SCADA Master Configuration"""
    master_id: int
    master_name: str
    polling_interval: float = 5.0  # seconds
    timeout: float = 10.0  # seconds
    max_retries: int = 3
    
    
class SCADAMaster(DNP3Station):
    """DNP3 SCADA Master Station"""
    
    def __init__(self, 
                 config: SCADAMasterConfig,
                 protocol: DNP3Protocol):
        super().__init__(config.master_id, protocol)
        
        self.config = config
        self.database = SCADADatabase()
        self.polling_scheduler = DNP3PollingScheduler()
        
        # RTU management
        self.rtus: Dict[int, RTUInfo] = {}
        self.pending_requests: Dict[str, Dict] = {}
        
        # Control and monitoring
        self.operator_commands: asyncio.Queue = asyncio.Queue()
        self.control_task: Optional[asyncio.Task] = None
        
        # Set up message handlers
        self.message_handlers = {
            DNP3FunctionCode.RESPONSE: self._handle_response,
            DNP3FunctionCode.UNSOLICITED_RESPONSE: self._handle_unsolicited_response
        }
        
        logger.info(f"SCADA Master {self.config.master_name} (ID: {self.station_id}) initialized")
    
    def add_rtu(self, station_id: int, name: str, location: str, bus_id: int):
        """Add an RTU to be monitored"""
        rtu_info = RTUInfo(
            station_id=station_id,
            name=name,
            location=location,
            bus_id=bus_id
        )
        
        self.rtus[station_id] = rtu_info
        
        # Add RTU to polling schedule
        self.polling_scheduler.add_poll_group(
            f"rtu_{station_id}",  # name
            self.config.polling_interval,  # interval
            self._poll_rtu,  # callback
            station_id  # *args
        )
        
        logger.info(f"SCADA Master {self.station_id}: Added RTU {name} "
                   f"(ID: {station_id}) at {location}")
    
    def remove_rtu(self, station_id: int):
        """Remove an RTU from monitoring"""
        if station_id in self.rtus:
            rtu_name = self.rtus[station_id].name
            del self.rtus[station_id]
            self.polling_scheduler.remove_poll_group(f"rtu_{station_id}")
            logger.info(f"SCADA Master {self.station_id}: Removed RTU {rtu_name} (ID: {station_id})")
    
    async def _poll_rtu(self, station_id: int):
        """Poll a specific RTU for data"""
        if station_id not in self.rtus:
            return
        
        rtu_info = self.rtus[station_id]
        rtu_info.last_poll_time = time.time()
        rtu_info.poll_count += 1
        
        logger.debug(f"SCADA Master {self.station_id}: Polling RTU {rtu_info.name} "
                    f"(poll #{rtu_info.poll_count})")
        
        try:
            # Send read request for all points
            message = await self.send_message(
                destination=station_id,
                function_code=DNP3FunctionCode.READ,
                data_points=[]  # Empty means read all
            )
            
            # Track pending request
            self.pending_requests[message.message_id] = {
                'rtu_id': station_id,
                'request_time': time.time(),
                'request_type': 'poll'
            }
            
            # Set timeout for response
            asyncio.create_task(self._check_response_timeout(message.message_id))
            
        except Exception as e:
            logger.error(f"SCADA Master {self.station_id}: Error polling RTU {station_id}: {e}")
            rtu_info.error_count += 1
            rtu_info.communication_status = "Error"
    
    async def _check_response_timeout(self, message_id: str):
        """Check for response timeout"""
        await asyncio.sleep(self.config.timeout)
        
        if message_id in self.pending_requests:
            request_info = self.pending_requests[message_id]
            rtu_id = request_info['rtu_id']
            
            if rtu_id in self.rtus:
                self.rtus[rtu_id].error_count += 1
                self.rtus[rtu_id].communication_status = "Timeout"
                
                logger.warning(f"SCADA Master {self.station_id}: Timeout waiting for "
                             f"response from RTU {rtu_id}")
            
            # Clean up pending request
            del self.pending_requests[message_id]
    
    async def _handle_response(self, message: DNP3Message):
        """Handle response from RTU"""
        # Find corresponding request
        request_info = None
        for msg_id, info in list(self.pending_requests.items()):
            if info['rtu_id'] == message.source:
                request_info = info
                del self.pending_requests[msg_id]
                break
        
        if not request_info:
            logger.warning(f"SCADA Master {self.station_id}: "
                         f"Unexpected response from RTU {message.source}")
            return
        
        # Update RTU status
        if message.source in self.rtus:
            rtu_info = self.rtus[message.source]
            rtu_info.last_response_time = time.time()
            rtu_info.communication_status = "Online"
            
            response_time = rtu_info.last_response_time - request_info['request_time']
            logger.debug(f"SCADA Master {self.station_id}: Received response from "
                        f"RTU {rtu_info.name} in {response_time*1000:.1f}ms")
        
        # Process data points
        self._process_measurements(message.source, message.data_points)
    
    async def _handle_unsolicited_response(self, message: DNP3Message):
        """Handle unsolicited response from RTU"""
        logger.info(f"SCADA Master {self.station_id}: "
                   f"Received unsolicited response from RTU {message.source}")
        
        # Process data points
        self._process_measurements(message.source, message.data_points)
        
        # Log as event
        self.database.events.append({
            'timestamp': time.time(),
            'type': 'unsolicited',
            'source': message.source,
            'data_points': len(message.data_points),
            'message': f"Unsolicited update from RTU {message.source}"
        })
    
    def _process_measurements(self, rtu_id: int, data_points: List[DNP3DataPoint]):
        """Process measurement data from RTU"""
        rtu_name = self.rtus.get(rtu_id, {}).name if rtu_id in self.rtus else f"RTU_{rtu_id}"
        
        for point in data_points:
            # Create unique point ID
            point_id = f"RTU_{rtu_id}_{point.group.name}_{point.index}"
            
            # Store measurement
            self.database.measurements[point_id] = {
                'rtu_id': rtu_id,
                'rtu_name': rtu_name,
                'group': point.group.name,
                'index': point.index,
                'value': point.value,
                'quality': point.quality,
                'timestamp': point.timestamp,
                'received_time': time.time()
            }
            
            # Check for alarms (example: voltage out of range)
            if point.group == DNP3ObjectGroup.ANALOG_INPUT and point.index == 1:  # Bus voltage (pu)
                if point.value < 0.95 or point.value > 1.05:
                    self._create_alarm(rtu_id, f"Voltage out of range: {point.value:.3f} pu")
        
        logger.debug(f"SCADA Master {self.station_id}: Processed {len(data_points)} "
                    f"measurements from {rtu_name}")
    
    def _create_alarm(self, rtu_id: int, message: str):
        """Create an alarm"""
        rtu_name = self.rtus.get(rtu_id, {}).name if rtu_id in self.rtus else f"RTU_{rtu_id}"
        
        alarm = {
            'timestamp': time.time(),
            'rtu_id': rtu_id,
            'rtu_name': rtu_name,
            'severity': 'HIGH',
            'message': message,
            'acknowledged': False
        }
        
        self.database.alarms.append(alarm)
        logger.warning(f"SCADA ALARM: {rtu_name} - {message}")
    
    async def send_control_command(self, 
                                 rtu_id: int,
                                 object_group: DNP3ObjectGroup,
                                 index: int,
                                 value: float,
                                 operation_type: str = "direct_operate"):
        """Send control command to RTU"""
        if rtu_id not in self.rtus:
            logger.error(f"SCADA Master {self.station_id}: RTU {rtu_id} not found")
            return False
        
        rtu_info = self.rtus[rtu_id]
        
        # Create control data point
        control_point = DNP3DataPoint(
            group=object_group,
            index=index,
            value=value,
            quality=0x01,
            timestamp=time.time()
        )
        
        # Determine function code
        if operation_type == "select":
            function_code = DNP3FunctionCode.SELECT
        elif operation_type == "operate":
            function_code = DNP3FunctionCode.OPERATE
        else:  # direct_operate
            function_code = DNP3FunctionCode.DIRECT_OPERATE
        
        try:
            # Send control command
            message = await self.send_message(
                destination=rtu_id,
                function_code=function_code,
                data_points=[control_point]
            )
            
            # Track pending request
            self.pending_requests[message.message_id] = {
                'rtu_id': rtu_id,
                'request_time': time.time(),
                'request_type': 'control'
            }
            
            logger.info(f"SCADA Master {self.station_id}: Sent {operation_type} command to "
                       f"{rtu_info.name} - {object_group.name}[{index}] = {value}")
            
            # Log as event
            self.database.events.append({
                'timestamp': time.time(),
                'type': 'control_command',
                'rtu_id': rtu_id,
                'object_group': object_group.name,
                'index': index,
                'value': value,
                'operation': operation_type
            })
            
            return True
            
        except Exception as e:
            logger.error(f"SCADA Master {self.station_id}: Error sending control command: {e}")
            return False
    
    async def manual_control_interface(self):
        """Manual control interface for operator commands"""
        try:
            while self.running:
                try:
                    command = await asyncio.wait_for(
                        self.operator_commands.get(), timeout=1.0
                    )
                    await self._execute_operator_command(command)
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"SCADA Master {self.station_id}: Manual control interface cancelled")
            raise
    
    async def _execute_operator_command(self, command: Dict[str, Any]):
        """Execute operator command"""
        try:
            cmd_type = command.get('type')
            rtu_id = command.get('rtu_id')
            
            if cmd_type == 'breaker_control':
                # Control breaker (binary output 0)
                action = command.get('action', 'close')
                value = 1.0 if action == 'close' else 0.0
                
                await self.send_control_command(
                    rtu_id=rtu_id,
                    object_group=DNP3ObjectGroup.BINARY_OUTPUT,
                    index=0,
                    value=value
                )
                
                logger.info(f"SCADA Operator Command: {action.upper()} breaker at RTU {rtu_id}")
            
            elif cmd_type == 'voltage_setpoint':
                # Adjust voltage setpoint (analog output 0)
                setpoint = command.get('setpoint', 1.02)
                
                await self.send_control_command(
                    rtu_id=rtu_id,
                    object_group=DNP3ObjectGroup.ANALOG_OUTPUT,
                    index=0,
                    value=setpoint
                )
                
                logger.info(f"SCADA Operator Command: Set voltage setpoint to {setpoint} pu at RTU {rtu_id}")
            
            elif cmd_type == 'generator_control':
                # Control generator start/stop (binary output 1)
                action = command.get('action', 'start')
                value = 1.0 if action == 'start' else 0.0
                
                await self.send_control_command(
                    rtu_id=rtu_id,
                    object_group=DNP3ObjectGroup.BINARY_OUTPUT,
                    index=1,
                    value=value
                )
                
                logger.info(f"SCADA Operator Command: {action.upper()} generator at RTU {rtu_id}")
                
        except Exception as e:
            logger.error(f"SCADA Master {self.station_id}: Error executing operator command: {e}")
    
    async def queue_operator_command(self, command: Dict[str, Any]):
        """Queue an operator command for execution"""
        await self.operator_commands.put(command)
    
    async def start(self):
        """Start the SCADA master"""
        await super().start()
        
        # Start polling scheduler
        await self.polling_scheduler.start_polling()
        
        # Start manual control interface
        self.control_task = asyncio.create_task(self.manual_control_interface())
        
        logger.info(f"SCADA Master {self.config.master_name} started")
    
    async def stop(self):
        """Stop the SCADA master"""
        # Stop polling scheduler
        await self.polling_scheduler.stop_polling()
        
        # Stop manual control interface
        if self.control_task:
            self.control_task.cancel()
            try:
                await self.control_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        logger.info(f"SCADA Master {self.config.master_name} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get SCADA master status"""
        return {
            "master_id": self.station_id,
            "master_name": self.config.master_name,
            "running": self.running,
            "rtu_count": len(self.rtus),
            "measurement_count": len(self.database.measurements),
            "alarm_count": len(self.database.alarms),
            "event_count": len(self.database.events),
            "polling_interval": self.config.polling_interval
        }
    
    def get_rtu_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all RTUs"""
        status = {}
        
        for rtu_id, rtu_info in self.rtus.items():
            last_comm = rtu_info.last_response_time
            comm_age = time.time() - last_comm if last_comm > 0 else float('inf')
            
            status[rtu_id] = {
                "name": rtu_info.name,
                "location": rtu_info.location,
                "bus_id": rtu_info.bus_id,
                "communication_status": rtu_info.communication_status,
                "last_communication": last_comm,
                "communication_age_seconds": comm_age,
                "poll_count": rtu_info.poll_count,
                "error_count": rtu_info.error_count
            }
        
        return status
    
    def get_measurements(self, rtu_id: Optional[int] = None) -> Dict[str, Any]:
        """Get current measurements"""
        if rtu_id is None:
            return self.database.measurements
        
        # Filter by RTU ID
        filtered = {}
        for point_id, measurement in self.database.measurements.items():
            if measurement['rtu_id'] == rtu_id:
                filtered[point_id] = measurement
        
        return filtered
    
    def get_alarms(self, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alarms"""
        if acknowledged is None:
            return self.database.alarms
        
        return [alarm for alarm in self.database.alarms 
                if alarm['acknowledged'] == acknowledged]
    
    def acknowledge_alarm(self, alarm_index: int) -> bool:
        """Acknowledge an alarm"""
        if 0 <= alarm_index < len(self.database.alarms):
            self.database.alarms[alarm_index]['acknowledged'] = True
            logger.info(f"SCADA Master {self.station_id}: Acknowledged alarm {alarm_index}")
            return True
        return False

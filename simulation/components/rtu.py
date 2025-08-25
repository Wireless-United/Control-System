"""
Remote Terminal Unit (RTU) Component

RTU acts as a DNP3 outstation that:
- Collects data from local sensors
- Responds to SCADA master's polling requests
- Supports both analog values (voltages, currents) and binary statuses (breaker states)
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel

from protocols.dnp3 import (
    DNP3Station, DNP3Protocol, DNP3Message, DNP3DataPoint,
    DNP3FunctionCode, DNP3ObjectGroup
)

logger = logging.getLogger(__name__)


class RTUDataBase(BaseModel):
    """RTU Local Database for DNP3 objects"""
    binary_inputs: Dict[int, Dict[str, Any]] = {}
    binary_outputs: Dict[int, Dict[str, Any]] = {}
    analog_inputs: Dict[int, Dict[str, Any]] = {}
    analog_outputs: Dict[int, Dict[str, Any]] = {}
    
    
class RTUConfig(BaseModel):
    """RTU Configuration"""
    station_id: int
    station_name: str
    location: str
    bus_id: int  # Grid bus this RTU monitors
    unsolicited_enabled: bool = True
    unsolicited_threshold: float = 0.05  # 5% change threshold
    
    
class RemoteTerminalUnit(DNP3Station):
    """DNP3 Remote Terminal Unit (Outstation)"""
    
    def __init__(self, 
                 config: RTUConfig,
                 protocol: DNP3Protocol,
                 grid_interface=None):
        super().__init__(config.station_id, protocol)
        
        self.config = config
        self.grid_interface = grid_interface
        self.database = RTUDataBase()
        
        # Data collection callbacks
        self.sensor_callbacks: Dict[str, Callable] = {}
        
        # Control callbacks
        self.control_callbacks: Dict[str, Callable] = {}
        
        # Unsolicited reporting
        self.last_values: Dict[str, float] = {}
        self.unsolicited_task: Optional[asyncio.Task] = None
        
        # Set up message handlers
        self.message_handlers = {
            DNP3FunctionCode.READ: self._handle_read_request,
            DNP3FunctionCode.WRITE: self._handle_write_request,
            DNP3FunctionCode.SELECT: self._handle_select_request,
            DNP3FunctionCode.OPERATE: self._handle_operate_request,
            DNP3FunctionCode.DIRECT_OPERATE: self._handle_direct_operate_request,
            DNP3FunctionCode.CONFIRM: self._handle_confirm
        }
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"RTU {self.config.station_name} (ID: {self.station_id}) "
                   f"initialized for Bus {self.config.bus_id}")
    
    def _initialize_database(self):
        """Initialize RTU database with default points"""
        # Binary Inputs (Status points)
        self.database.binary_inputs = {
            0: {"description": "Breaker Status", "value": True, "quality": 0x01},
            1: {"description": "Generator Online", "value": True, "quality": 0x01},
            2: {"description": "Protection Relay", "value": False, "quality": 0x01},
            3: {"description": "Communication Status", "value": True, "quality": 0x01}
        }
        
        # Binary Outputs (Control points)
        self.database.binary_outputs = {
            0: {"description": "Breaker Control", "value": True, "quality": 0x01},
            1: {"description": "Generator Start/Stop", "value": True, "quality": 0x01}
        }
        
        # Analog Inputs (Measurements)
        self.database.analog_inputs = {
            0: {"description": "Bus Voltage (kV)", "value": 345.0, "quality": 0x01, "unit": "kV"},
            1: {"description": "Bus Voltage (pu)", "value": 1.000, "quality": 0x01, "unit": "pu"},
            2: {"description": "Active Power (MW)", "value": 0.0, "quality": 0x01, "unit": "MW"},
            3: {"description": "Reactive Power (MVAR)", "value": 0.0, "quality": 0x01, "unit": "MVAR"},
            4: {"description": "Current (A)", "value": 0.0, "quality": 0x01, "unit": "A"},
            5: {"description": "Frequency (Hz)", "value": 50.0, "quality": 0x01, "unit": "Hz"}
        }
        
        # Analog Outputs (Setpoints)
        self.database.analog_outputs = {
            0: {"description": "Voltage Setpoint (pu)", "value": 1.020, "quality": 0x01, "unit": "pu"},
            1: {"description": "Power Setpoint (MW)", "value": 100.0, "quality": 0x01, "unit": "MW"}
        }
    
    def register_sensor_callback(self, data_type: str, index: int, callback: Callable):
        """Register callback to collect sensor data"""
        key = f"{data_type}_{index}"
        self.sensor_callbacks[key] = callback
        logger.debug(f"RTU {self.station_id}: Registered sensor callback for {key}")
    
    def register_control_callback(self, data_type: str, index: int, callback: Callable):
        """Register callback for control actions"""
        key = f"{data_type}_{index}"
        self.control_callbacks[key] = callback
        logger.debug(f"RTU {self.station_id}: Registered control callback for {key}")
    
    async def update_measurements(self):
        """Update measurements from sensors"""
        # Update analog inputs from sensor callbacks
        for key, callback in self.sensor_callbacks.items():
            if key.startswith("analog_input"):
                try:
                    index = int(key.split("_")[-1])
                    value = await callback() if asyncio.iscoroutinefunction(callback) else callback()
                    
                    if index in self.database.analog_inputs:
                        old_value = self.database.analog_inputs[index]["value"]
                        self.database.analog_inputs[index]["value"] = value
                        self.database.analog_inputs[index]["timestamp"] = time.time()
                        
                        # Check for unsolicited reporting
                        if self.config.unsolicited_enabled:
                            await self._check_unsolicited_change(f"AI_{index}", old_value, value)
                            
                except Exception as e:
                    logger.error(f"RTU {self.station_id}: Error updating measurement {key}: {e}")
        
        # Update binary inputs from sensor callbacks
        for key, callback in self.sensor_callbacks.items():
            if key.startswith("binary_input"):
                try:
                    index = int(key.split("_")[-1])
                    value = await callback() if asyncio.iscoroutinefunction(callback) else callback()
                    
                    if index in self.database.binary_inputs:
                        old_value = self.database.binary_inputs[index]["value"]
                        self.database.binary_inputs[index]["value"] = bool(value)
                        self.database.binary_inputs[index]["timestamp"] = time.time()
                        
                        # Check for unsolicited reporting
                        if self.config.unsolicited_enabled and old_value != bool(value):
                            await self._send_unsolicited_report(f"BI_{index}", bool(value))
                            
                except Exception as e:
                    logger.error(f"RTU {self.station_id}: Error updating binary input {key}: {e}")
    
    async def _check_unsolicited_change(self, point_name: str, old_value: float, new_value: float):
        """Check if change warrants unsolicited report"""
        if abs(new_value - old_value) / max(abs(old_value), 0.01) > self.config.unsolicited_threshold:
            await self._send_unsolicited_report(point_name, new_value)
    
    async def _send_unsolicited_report(self, point_name: str, value: Any):
        """Send unsolicited report for significant change"""
        # In a real implementation, this would send to all masters
        # For now, we'll just log the event
        logger.info(f"RTU {self.station_id}: Unsolicited change in {point_name} = {value}")
    
    async def _handle_read_request(self, message: DNP3Message):
        """Handle read request from SCADA master"""
        logger.debug(f"RTU {self.station_id}: Processing read request from {message.source}")
        
        response_points = []
        
        # If no specific points requested, send all current values
        if not message.data_points:
            # Add all analog inputs
            for index, data in self.database.analog_inputs.items():
                point = DNP3DataPoint(
                    group=DNP3ObjectGroup.ANALOG_INPUT,
                    index=index,
                    value=data["value"],
                    quality=data["quality"],
                    timestamp=data.get("timestamp", time.time())
                )
                response_points.append(point)
            
            # Add all binary inputs
            for index, data in self.database.binary_inputs.items():
                point = DNP3DataPoint(
                    group=DNP3ObjectGroup.BINARY_INPUT,
                    index=index,
                    value=float(data["value"]),
                    quality=data["quality"],
                    timestamp=data.get("timestamp", time.time())
                )
                response_points.append(point)
        else:
            # Send requested points
            for req_point in message.data_points:
                if req_point.group == DNP3ObjectGroup.ANALOG_INPUT:
                    if req_point.index in self.database.analog_inputs:
                        data = self.database.analog_inputs[req_point.index]
                        point = DNP3DataPoint(
                            group=DNP3ObjectGroup.ANALOG_INPUT,
                            index=req_point.index,
                            value=data["value"],
                            quality=data["quality"],
                            timestamp=data.get("timestamp", time.time())
                        )
                        response_points.append(point)
                
                elif req_point.group == DNP3ObjectGroup.BINARY_INPUT:
                    if req_point.index in self.database.binary_inputs:
                        data = self.database.binary_inputs[req_point.index]
                        point = DNP3DataPoint(
                            group=DNP3ObjectGroup.BINARY_INPUT,
                            index=req_point.index,
                            value=float(data["value"]),
                            quality=data["quality"],
                            timestamp=data.get("timestamp", time.time())
                        )
                        response_points.append(point)
        
        # Send response
        await self.send_message(
            destination=message.source,
            function_code=DNP3FunctionCode.RESPONSE,
            data_points=response_points
        )
        
        logger.debug(f"RTU {self.station_id}: Sent {len(response_points)} data points to {message.source}")
    
    async def _handle_write_request(self, message: DNP3Message):
        """Handle write request (setpoint changes)"""
        logger.info(f"RTU {self.station_id}: Processing write request from {message.source}")
        
        success_points = []
        
        for point in message.data_points:
            try:
                if point.group == DNP3ObjectGroup.ANALOG_OUTPUT:
                    if point.index in self.database.analog_outputs:
                        old_value = self.database.analog_outputs[point.index]["value"]
                        self.database.analog_outputs[point.index]["value"] = point.value
                        self.database.analog_outputs[point.index]["timestamp"] = time.time()
                        
                        # Execute control callback if registered
                        callback_key = f"analog_output_{point.index}"
                        if callback_key in self.control_callbacks:
                            callback = self.control_callbacks[callback_key]
                            if asyncio.iscoroutinefunction(callback):
                                await callback(point.value)
                            else:
                                callback(point.value)
                        
                        success_points.append(point)
                        logger.info(f"RTU {self.station_id}: Analog output {point.index} "
                                  f"changed from {old_value} to {point.value}")
                
                elif point.group == DNP3ObjectGroup.BINARY_OUTPUT:
                    if point.index in self.database.binary_outputs:
                        old_value = self.database.binary_outputs[point.index]["value"]
                        self.database.binary_outputs[point.index]["value"] = bool(point.value)
                        self.database.binary_outputs[point.index]["timestamp"] = time.time()
                        
                        # Execute control callback if registered
                        callback_key = f"binary_output_{point.index}"
                        if callback_key in self.control_callbacks:
                            callback = self.control_callbacks[callback_key]
                            if asyncio.iscoroutinefunction(callback):
                                await callback(bool(point.value))
                            else:
                                callback(bool(point.value))
                        
                        success_points.append(point)
                        logger.info(f"RTU {self.station_id}: Binary output {point.index} "
                                  f"changed from {old_value} to {bool(point.value)}")
                        
            except Exception as e:
                logger.error(f"RTU {self.station_id}: Error writing point {point.index}: {e}")
        
        # Send confirmation response
        await self.send_message(
            destination=message.source,
            function_code=DNP3FunctionCode.RESPONSE,
            data_points=success_points
        )
    
    async def _handle_select_request(self, message: DNP3Message):
        """Handle select request (prepare for operate)"""
        logger.info(f"RTU {self.station_id}: Processing select request from {message.source}")
        
        # For this simulation, we'll always confirm selection
        await self.send_message(
            destination=message.source,
            function_code=DNP3FunctionCode.RESPONSE,
            data_points=message.data_points
        )
    
    async def _handle_operate_request(self, message: DNP3Message):
        """Handle operate request (execute after select)"""
        logger.info(f"RTU {self.station_id}: Processing operate request from {message.source}")
        
        # Execute the operation (same as write for simulation)
        await self._handle_write_request(message)
    
    async def _handle_direct_operate_request(self, message: DNP3Message):
        """Handle direct operate request (select + operate in one)"""
        logger.info(f"RTU {self.station_id}: Processing direct operate request from {message.source}")
        
        # Execute the operation directly
        await self._handle_write_request(message)
    
    async def _handle_confirm(self, message: DNP3Message):
        """Handle confirmation message"""
        logger.debug(f"RTU {self.station_id}: Received confirmation from {message.source}")
    
    async def start(self):
        """Start the RTU"""
        await super().start()
        
        # Start unsolicited monitoring if enabled
        if self.config.unsolicited_enabled:
            self.unsolicited_task = asyncio.create_task(self._unsolicited_monitor())
        
        logger.info(f"RTU {self.config.station_name} started on Bus {self.config.bus_id}")
    
    async def stop(self):
        """Stop the RTU"""
        if self.unsolicited_task:
            self.unsolicited_task.cancel()
            try:
                await self.unsolicited_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        logger.info(f"RTU {self.config.station_name} stopped")
    
    async def _unsolicited_monitor(self):
        """Monitor for unsolicited events"""
        try:
            while self.running:
                await self.update_measurements()
                await asyncio.sleep(1.0)  # Update every second
                
        except asyncio.CancelledError:
            logger.info(f"RTU {self.station_id}: Unsolicited monitor cancelled")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get RTU status information"""
        return {
            "station_id": self.station_id,
            "station_name": self.config.station_name,
            "location": self.config.location,
            "bus_id": self.config.bus_id,
            "running": self.running,
            "unsolicited_enabled": self.config.unsolicited_enabled,
            "analog_inputs_count": len(self.database.analog_inputs),
            "binary_inputs_count": len(self.database.binary_inputs),
            "analog_outputs_count": len(self.database.analog_outputs),
            "binary_outputs_count": len(self.database.binary_outputs)
        }
    
    def get_measurements(self) -> Dict[str, Any]:
        """Get current measurements"""
        measurements = {}
        
        for index, data in self.database.analog_inputs.items():
            measurements[f"AI_{index}"] = {
                "value": data["value"],
                "description": data["description"],
                "unit": data.get("unit", ""),
                "quality": data["quality"],
                "timestamp": data.get("timestamp", time.time())
            }
        
        for index, data in self.database.binary_inputs.items():
            measurements[f"BI_{index}"] = {
                "value": data["value"],
                "description": data["description"],
                "quality": data["quality"],
                "timestamp": data.get("timestamp", time.time())
            }
        
        return measurements

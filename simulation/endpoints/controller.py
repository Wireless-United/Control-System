"""
Controller endpoint for cyber-physical grid simulation.
Receives measurements, applies AVR logic, sends control signals.
"""

import asyncio
import time
import logging
import sys
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from components.avr import AVR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ControlCommand(BaseModel):
    """Data model for control commands."""
    controller_id: int | str  # Accept both int and string
    generator_id: int
    timestamp: float
    command_type: str  # "excitation", "setpoint", etc.
    value: float
    reason: Optional[str] = None

class Controller:
    """Controller endpoint that implements AVR logic."""
    
    def __init__(self, controller_id: int, generator_id: int, bus_id: int,
                 voltage_setpoint: float = 1.0,
                 communication_protocol=None):
        """
        Initialize controller.
        
        Args:
            controller_id: Unique identifier for controller
            generator_id: Generator being controlled
            bus_id: Bus being monitored/controlled
            voltage_setpoint: Desired voltage setpoint
            communication_protocol: Protocol for communication
        """
        self.controller_id = controller_id
        self.generator_id = generator_id
        self.bus_id = bus_id
        self.communication_protocol = communication_protocol
        
        # AVR controller
        self.avr = AVR(
            avr_id=controller_id,
            generator_id=generator_id,
            voltage_setpoint=voltage_setpoint
        )
        
        # Control state
        self.is_running = False
        self.control_task: Optional[asyncio.Task] = None
        self.last_sensor_reading = None
        self.last_control_time = time.time()
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.message_handler_task: Optional[asyncio.Task] = None
        
    async def start_control(self):
        """Start control tasks."""
        if self.is_running:
            logger.warning(f"Controller {self.controller_id} already running")
            return
            
        self.is_running = True
        
        # Start message handler
        self.message_handler_task = asyncio.create_task(self._message_handler())
        
        # Start control loop
        self.control_task = asyncio.create_task(self._control_loop())
        
        logger.info(f"Controller {self.controller_id} started for Generator {self.generator_id}")
        
    async def stop_control(self):
        """Stop control tasks."""
        self.is_running = False
        
        # Cancel tasks
        if self.control_task:
            self.control_task.cancel()
            try:
                await self.control_task
            except asyncio.CancelledError:
                pass
                
        if self.message_handler_task:
            self.message_handler_task.cancel()
            try:
                await self.message_handler_task
            except asyncio.CancelledError:
                pass
                
        logger.info(f"Controller {self.controller_id} stopped")
        
    async def _control_loop(self):
        """Main control loop."""
        try:
            while self.is_running:
                await self._execute_control_cycle()
                await asyncio.sleep(0.1)  # 10 Hz control rate
        except asyncio.CancelledError:
            logger.info(f"Controller {self.controller_id} control loop cancelled")
        except Exception as e:
            logger.error(f"Controller {self.controller_id} control error: {e}")
            
    async def _execute_control_cycle(self):
        """Execute one control cycle."""
        current_time = time.time()
        dt = current_time - self.last_control_time
        self.last_control_time = current_time
        
        # Check if we have recent sensor data
        if not self.last_sensor_reading:
            return
            
        # Update AVR with latest measurement
        voltage = self.last_sensor_reading.get('voltage_magnitude', 1.0)
        self.avr.update_measurement(voltage)
        
        # Calculate control action
        excitation = self.avr.calculate_control(dt)
        
        # Create control command
        command = ControlCommand(
            controller_id=self.controller_id,
            generator_id=self.generator_id,
            timestamp=current_time,
            command_type="excitation",
            value=excitation,
            reason=f"AVR control: V={voltage:.3f}, Vref={self.avr.voltage_setpoint:.3f}, Error={self.avr.error:.3f}"
        )
        
        # Send command to actuator
        if self.communication_protocol:
            await self.communication_protocol.send_message(
                sender_id=self.controller_id,
                receiver_id=f"actuator_{self.generator_id}",
                message_type="control_command",
                data=command.model_dump()
            )
            
        logger.debug(f"Controller {self.controller_id}: Excitation={excitation:.3f}, Error={self.avr.error:.3f}")
        
    async def _message_handler(self):
        """Handle incoming messages."""
        try:
            while self.is_running:
                try:
                    # Check for messages with timeout
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"Controller {self.controller_id} message handler cancelled")
        except Exception as e:
            logger.error(f"Controller {self.controller_id} message handler error: {e}")
            
    async def _process_message(self, message: Dict[str, Any]):
        """Process incoming message."""
        try:
            message_type = message.get('message_type')
            data = message.get('data', {})
            
            if message_type == 'sensor_reading':
                # Update sensor reading
                self.last_sensor_reading = data
                logger.debug(f"Controller {self.controller_id} received sensor reading: V={data.get('voltage_magnitude', 0):.3f}")
                
            elif message_type == 'setpoint_change':
                # Update voltage setpoint
                new_setpoint = data.get('voltage_setpoint')
                if new_setpoint:
                    self.avr.set_voltage_setpoint(new_setpoint)
                    self.avr.reset_integral()  # Reset integral windup
                    logger.info(f"Controller {self.controller_id} setpoint changed to {new_setpoint:.3f}")
                    
        except Exception as e:
            logger.error(f"Controller {self.controller_id} message processing error: {e}")
            
    async def receive_message(self, message: Dict[str, Any]):
        """Receive message from communication protocol."""
        await self.message_queue.put(message)
        
    def set_voltage_setpoint(self, setpoint: float):
        """Set new voltage setpoint."""
        self.avr.set_voltage_setpoint(setpoint)
        self.avr.reset_integral()
        
    def get_avr_status(self) -> Dict[str, Any]:
        """Get AVR controller status."""
        return self.avr.get_status()
        
    def __str__(self):
        status = "Running" if self.is_running else "Stopped"
        avr_info = f"Vref={self.avr.voltage_setpoint:.3f}, Exc={self.avr.excitation_output:.3f}"
        return f"Controller {self.controller_id} for Gen {self.generator_id} [{status}]: {avr_info}"

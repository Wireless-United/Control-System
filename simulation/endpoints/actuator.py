"""
Actuator endpoint for cyber-physical grid simulation.
Applies excitation/voltage changes to generator in the grid.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActuatorAction(BaseModel):
    """Data model for actuator actions."""
    actuator_id: int | str  # Accept both int and string
    generator_id: int
    timestamp: float
    action_type: str  # "excitation", "power_output", etc.
    value: float
    applied: bool = False

class Actuator:
    """Actuator endpoint that applies control actions to generators."""
    
    def __init__(self, actuator_id: int, generator_id: int,
                 response_time: float = 0.01,  # 10ms response time
                 communication_protocol=None):
        """
        Initialize actuator.
        
        Args:
            actuator_id: Unique identifier for actuator
            generator_id: Generator being controlled
            response_time: Actuator response time in seconds
            communication_protocol: Protocol for communication
        """
        self.actuator_id = actuator_id
        self.generator_id = generator_id
        self.response_time = response_time
        self.communication_protocol = communication_protocol
        
        # Control state
        self.is_running = False
        self.actuator_task: Optional[asyncio.Task] = None
        self.action_queue = asyncio.Queue()
        
        # Grid reference (to be set by simulation)
        self.grid_interface = None
        
        # Statistics
        self.actions_applied = 0
        self.last_action_time = 0.0
        
    def set_grid_interface(self, grid_interface):
        """Set reference to grid for control actions."""
        self.grid_interface = grid_interface
        
    async def start_actuator(self):
        """Start actuator task."""
        if self.is_running:
            logger.warning(f"Actuator {self.actuator_id} already running")
            return
            
        self.is_running = True
        self.actuator_task = asyncio.create_task(self._actuator_loop())
        logger.info(f"Actuator {self.actuator_id} started for Generator {self.generator_id}")
        
    async def stop_actuator(self):
        """Stop actuator task."""
        self.is_running = False
        if self.actuator_task:
            self.actuator_task.cancel()
            try:
                await self.actuator_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Actuator {self.actuator_id} stopped")
        
    async def _actuator_loop(self):
        """Main actuator loop."""
        try:
            while self.is_running:
                try:
                    # Wait for control action with timeout
                    action_data = await asyncio.wait_for(
                        self.action_queue.get(), 
                        timeout=0.1
                    )
                    await self._apply_action(action_data)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"Actuator {self.actuator_id} loop cancelled")
        except Exception as e:
            logger.error(f"Actuator {self.actuator_id} loop error: {e}")
            
    async def _apply_action(self, action_data: Dict[str, Any]):
        """Apply control action to generator."""
        try:
            # Simulate actuator response time
            await asyncio.sleep(self.response_time)
            
            if not self.grid_interface:
                logger.warning(f"Actuator {self.actuator_id}: No grid interface available")
                return
                
            command_type = action_data.get('command_type')
            value = action_data.get('value')
            generator_id = action_data.get('generator_id')
            
            if command_type == 'excitation' and generator_id == self.generator_id:
                # Apply excitation to generator
                success = self.grid_interface.set_generator_excitation(generator_id, value)
                
                if success:
                    self.actions_applied += 1
                    self.last_action_time = time.time()
                    
                    # Create action record
                    action = ActuatorAction(
                        actuator_id=self.actuator_id,
                        generator_id=generator_id,
                        timestamp=self.last_action_time,
                        action_type=command_type,
                        value=value,
                        applied=True
                    )
                    
                    logger.debug(f"Actuator {self.actuator_id}: Applied excitation {value:.3f} to Gen {generator_id}")
                    
                    # Send confirmation back via protocol if available
                    if self.communication_protocol:
                        await self.communication_protocol.send_message(
                            sender_id=self.actuator_id,
                            receiver_id=f"controller_{generator_id}",
                            message_type="action_confirmation",
                            data=action.model_dump()
                        )
                else:
                    logger.error(f"Actuator {self.actuator_id}: Failed to apply excitation to Gen {generator_id}")
                    
            else:
                logger.warning(f"Actuator {self.actuator_id}: Unknown command type '{command_type}' or generator mismatch")
                
        except Exception as e:
            logger.error(f"Actuator {self.actuator_id} action application error: {e}")
            
    async def receive_message(self, message: Dict[str, Any]):
        """Receive message from communication protocol."""
        message_type = message.get('message_type')
        
        if message_type == 'control_command':
            # Queue control action
            data = message.get('data', {})
            await self.action_queue.put(data)
            
    async def apply_immediate_action(self, command_type: str, value: float):
        """Apply immediate action (bypass queue)."""
        action_data = {
            'command_type': command_type,
            'value': value,
            'generator_id': self.generator_id,
            'timestamp': time.time()
        }
        await self._apply_action(action_data)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get actuator statistics."""
        return {
            'actuator_id': self.actuator_id,
            'generator_id': self.generator_id,
            'actions_applied': self.actions_applied,
            'last_action_time': self.last_action_time,
            'is_running': self.is_running,
            'queue_size': self.action_queue.qsize()
        }
        
    def __str__(self):
        status = "Running" if self.is_running else "Stopped"
        return (f"Actuator {self.actuator_id} for Gen {self.generator_id} [{status}]: "
                f"{self.actions_applied} actions applied")

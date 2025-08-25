"""
PROFINET-like communication protocol for cyber-physical grid simulation.
Implements asynchronous communication between endpoints with low latency.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Set, Optional, Callable
from pydantic import BaseModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfinetMessage(BaseModel):
    """Data model for PROFINET messages."""
    message_id: str
    sender_id: int | str  # Accept both int and string
    receiver_id: int | str  # Accept both int and string
    message_type: str
    timestamp: float
    data: Dict[str, Any]
    priority: int = 1  # 1=high, 2=normal, 3=low

class ProfinetProtocol:
    """
    PROFINET-like protocol for fieldbus communication.
    Simulates low-latency communication between endpoints.
    """
    
    def __init__(self, protocol_id: str = "profinet_1", 
                 network_latency: float = 0.002,  # 2ms typical PROFINET latency
                 max_message_size: int = 1500):   # bytes
        """
        Initialize PROFINET protocol.
        
        Args:
            protocol_id: Unique identifier for protocol instance
            network_latency: Simulated network latency in seconds
            max_message_size: Maximum message size in bytes
        """
        self.protocol_id = protocol_id
        self.network_latency = network_latency
        self.max_message_size = max_message_size
        
        # Network topology
        self.endpoints: Dict[str, Any] = {}  # endpoint_id -> endpoint_object
        self.message_handlers: Dict[str, Callable] = {}  # endpoint_id -> message_handler
        
        # Communication state
        self.is_running = False
        self.message_queue = asyncio.PriorityQueue()
        self.network_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.messages_sent = 0
        self.messages_delivered = 0
        self.message_errors = 0
        self.total_latency = 0.0
        
        # Message ID counter
        self.message_counter = 0
        
    async def start_network(self):
        """Start the network communication."""
        if self.is_running:
            logger.warning(f"PROFINET {self.protocol_id} already running")
            return
            
        self.is_running = True
        self.network_task = asyncio.create_task(self._network_loop())
        logger.info(f"PROFINET {self.protocol_id} network started")
        
    async def stop_network(self):
        """Stop the network communication."""
        self.is_running = False
        if self.network_task:
            self.network_task.cancel()
            try:
                await self.network_task
            except asyncio.CancelledError:
                pass
        logger.info(f"PROFINET {self.protocol_id} network stopped")
        
    def register_endpoint(self, endpoint_id: str, endpoint_object: Any):
        """
        Register an endpoint with the protocol.
        
        Args:
            endpoint_id: Unique identifier for endpoint
            endpoint_object: Endpoint object (sensor, controller, actuator)
        """
        self.endpoints[endpoint_id] = endpoint_object
        
        # Check if endpoint has receive_message method
        if hasattr(endpoint_object, 'receive_message'):
            self.message_handlers[endpoint_id] = endpoint_object.receive_message
            
        logger.info(f"PROFINET {self.protocol_id}: Registered endpoint {endpoint_id}")
        
    def unregister_endpoint(self, endpoint_id: str):
        """Unregister an endpoint."""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
        if endpoint_id in self.message_handlers:
            del self.message_handlers[endpoint_id]
        logger.info(f"PROFINET {self.protocol_id}: Unregistered endpoint {endpoint_id}")
        
    async def send_message(self, sender_id: str, receiver_id: str, 
                          message_type: str, data: Dict[str, Any],
                          priority: int = 2):
        """
        Send a message through the protocol.
        
        Args:
            sender_id: ID of sending endpoint
            receiver_id: ID of receiving endpoint  
            message_type: Type of message
            data: Message payload
            priority: Message priority (1=high, 2=normal, 3=low)
        """
        if not self.is_running:
            logger.warning(f"PROFINET {self.protocol_id}: Cannot send message, network not running")
            return False
            
        # Check message size
        message_json = json.dumps(data)
        if len(message_json.encode('utf-8')) > self.max_message_size:
            logger.error(f"PROFINET {self.protocol_id}: Message too large ({len(message_json)} bytes)")
            self.message_errors += 1
            return False
            
        # Create message
        self.message_counter += 1
        message = ProfinetMessage(
            message_id=f"{self.protocol_id}_{self.message_counter}",
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            timestamp=time.time(),
            data=data,
            priority=priority
        )
        
        # Queue message with priority (lower number = higher priority)
        await self.message_queue.put((priority, time.time(), message))
        self.messages_sent += 1
        
        logger.debug(f"PROFINET {self.protocol_id}: Queued message {message.message_id} "
                    f"from {sender_id} to {receiver_id}")
        return True
        
    async def _network_loop(self):
        """Main network processing loop."""
        try:
            while self.is_running:
                try:
                    # Get next message with timeout
                    priority, queue_time, message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=0.01
                    )
                    
                    # Simulate network latency
                    await asyncio.sleep(self.network_latency)
                    
                    # Deliver message
                    await self._deliver_message(message, queue_time)
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"PROFINET {self.protocol_id} network loop cancelled")
        except Exception as e:
            logger.error(f"PROFINET {self.protocol_id} network loop error: {e}")
            
    async def _deliver_message(self, message: ProfinetMessage, queue_time: float):
        """Deliver message to receiver."""
        try:
            receiver_id = message.receiver_id
            
            # Check if receiver exists
            if receiver_id not in self.message_handlers:
                logger.warning(f"PROFINET {self.protocol_id}: Receiver {receiver_id} not found")
                self.message_errors += 1
                return
                
            # Calculate total latency
            current_time = time.time()
            total_latency = current_time - queue_time
            self.total_latency += total_latency
            
            # Prepare message for delivery
            delivery_message = {
                'message_id': message.message_id,
                'sender_id': message.sender_id,
                'message_type': message.message_type,
                'timestamp': message.timestamp,
                'data': message.data,
                'delivery_latency': total_latency
            }
            
            # Deliver to endpoint
            handler = self.message_handlers[receiver_id]
            await handler(delivery_message)
            
            self.messages_delivered += 1
            
            logger.debug(f"PROFINET {self.protocol_id}: Delivered message {message.message_id} "
                        f"to {receiver_id} (latency: {total_latency*1000:.1f}ms)")
            
        except Exception as e:
            logger.error(f"PROFINET {self.protocol_id}: Message delivery error: {e}")
            self.message_errors += 1
            
    async def broadcast_message(self, sender_id: str, message_type: str, 
                               data: Dict[str, Any], priority: int = 2):
        """Broadcast message to all endpoints except sender."""
        tasks = []
        for endpoint_id in self.endpoints:
            if endpoint_id != sender_id:
                task = self.send_message(sender_id, endpoint_id, message_type, data, priority)
                tasks.append(task)
                
        if tasks:
            await asyncio.gather(*tasks)
            
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        avg_latency = (self.total_latency / self.messages_delivered 
                      if self.messages_delivered > 0 else 0.0)
        
        return {
            'protocol_id': self.protocol_id,
            'is_running': self.is_running,
            'endpoints_registered': len(self.endpoints),
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'message_errors': self.message_errors,
            'average_latency_ms': avg_latency * 1000,
            'network_latency_ms': self.network_latency * 1000,
            'queue_size': self.message_queue.qsize()
        }
        
    def get_endpoint_list(self) -> Set[str]:
        """Get list of registered endpoints."""
        return set(self.endpoints.keys())
        
    def __str__(self):
        status = "Running" if self.is_running else "Stopped"
        return (f"PROFINET {self.protocol_id} [{status}]: "
                f"{len(self.endpoints)} endpoints, "
                f"{self.messages_delivered}/{self.messages_sent} delivered")

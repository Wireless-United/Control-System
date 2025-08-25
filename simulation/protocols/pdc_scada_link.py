"""
PDC-SCADA Link Protocol

This module implements the logical link between PDC and SCADA systems,
providing subscription-based data exchange and event notification.

Features:
- Subscription mechanism for SCADA to request aggregated PDC data
- Periodic transfer of selected phasor data to SCADA
- Event-driven updates when anomalies are detected
- Realistic communication latency (100-200ms)
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pydantic import BaseModel
from collections import defaultdict

logger = logging.getLogger(__name__)

# Message structures for PDC-SCADA communication
class PDCSCADAMessage(BaseModel):
    """Base class for PDC-SCADA messages"""
    timestamp: float
    message_type: str
    sequence_number: int

class SubscriptionRequest(PDCSCADAMessage):
    """SCADA request to subscribe to PDC data"""
    data_types: List[str]  # ['voltage', 'frequency', 'power_flow', 'events']
    update_interval: float  # seconds
    threshold_config: Dict[str, Any]

class DataSnapshot(PDCSCADAMessage):
    """PDC data snapshot for SCADA"""
    grid_summary: Dict[str, Any]
    pmu_count: int
    data_quality: float
    voltage_profile: Dict[int, Dict[str, float]]  # bus_id -> {magnitude, angle}
    frequency_summary: Dict[str, float]  # avg, min, max, deviation
    system_status: str  # STABLE, WARNING, ALARM
    alarms: List[Dict[str, Any]]

class EventNotification(PDCSCADAMessage):
    """PDC event notification to SCADA"""
    event_type: str  # FREQUENCY_DEVIATION, VOLTAGE_INSTABILITY, PMU_OFFLINE
    severity: str    # INFO, WARNING, CRITICAL
    location: str    # Bus number or PMU identifier
    description: str
    value: float
    threshold: float

class ControlCommand(PDCSCADAMessage):
    """SCADA control command"""
    command_type: str  # THRESHOLD_UPDATE, PMU_CONTROL, REPORTING_CONFIG
    target: str
    parameters: Dict[str, Any]

@dataclass
class PDCSCADALinkStats:
    """Statistics for PDC-SCADA link"""
    messages_sent: int = 0
    messages_received: int = 0
    data_snapshots: int = 0
    event_notifications: int = 0
    subscription_requests: int = 0
    average_latency: float = 0.0
    total_latency: float = 0.0
    message_count: int = 0

class PDCSCADALink:
    """PDC-SCADA communication link"""
    
    def __init__(self, link_name: str, base_latency: float = 0.15):
        self.link_name = link_name
        self.base_latency = base_latency  # Base latency in seconds (150ms)
        self.running = False
        
        # Message queues
        self.pdc_to_scada_queue = asyncio.Queue()
        self.scada_to_pdc_queue = asyncio.Queue()
        
        # Subscriptions and handlers
        self.subscriptions: Dict[str, Dict] = {}  # subscriber_id -> subscription_config
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = PDCSCADALinkStats()
        self.sequence_counter = 0
        
        # Background tasks
        self.message_processing_task: Optional[asyncio.Task] = None
        
        logger.info(f"PDC-SCADA Link {link_name} initialized with {base_latency*1000:.0f}ms base latency")
    
    async def start_link(self):
        """Start the PDC-SCADA communication link"""
        if self.running:
            return
        
        self.running = True
        self.message_processing_task = asyncio.create_task(self._message_processing_loop())
        logger.info(f"PDC-SCADA Link {self.link_name} started")
    
    async def stop_link(self):
        """Stop the PDC-SCADA communication link"""
        if not self.running:
            return
        
        self.running = False
        if self.message_processing_task:
            self.message_processing_task.cancel()
            try:
                await self.message_processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"PDC-SCADA Link {self.link_name} stopped")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        try:
            while self.running:
                # Process PDC to SCADA messages
                try:
                    message, sender_id, receiver_id = await asyncio.wait_for(
                        self.pdc_to_scada_queue.get(), timeout=0.1)
                    await self._deliver_message(message, sender_id, receiver_id, 'pdc_to_scada')
                except asyncio.TimeoutError:
                    pass
                
                # Process SCADA to PDC messages
                try:
                    message, sender_id, receiver_id = await asyncio.wait_for(
                        self.scada_to_pdc_queue.get(), timeout=0.1)
                    await self._deliver_message(message, sender_id, receiver_id, 'scada_to_pdc')
                except asyncio.TimeoutError:
                    pass
        
        except asyncio.CancelledError:
            logger.info(f"PDC-SCADA Link {self.link_name} message processing loop cancelled")
    
    async def _deliver_message(self, message: PDCSCADAMessage, sender_id: str, 
                              receiver_id: str, direction: str):
        """Deliver message with realistic latency"""
        send_time = time.time()
        
        # Simulate variable network latency (100-200ms)
        import random
        latency = self.base_latency + random.uniform(-0.05, 0.05)
        await asyncio.sleep(latency)
        
        # Calculate actual latency
        actual_latency = (time.time() - send_time) * 1000  # Convert to ms
        self.stats.total_latency += actual_latency
        self.stats.message_count += 1
        self.stats.average_latency = self.stats.total_latency / self.stats.message_count
        
        # Deliver to handlers based on message type
        message_type = message.message_type
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    await handler(message, sender_id, actual_latency)
                except Exception as e:
                    logger.error(f"PDC-SCADA Link {self.link_name}: Handler error: {e}")
        
        # For PDC to SCADA direction, also deliver to all subscribers interested in this data type
        if direction == 'pdc_to_scada':
            await self._deliver_to_subscribers(message, sender_id, actual_latency)
        
        self.stats.messages_received += 1
        logger.debug(f"PDC-SCADA Link {self.link_name}: Delivered {message_type} "
                    f"from {sender_id} to {receiver_id}, latency: {actual_latency:.1f}ms")
    
    async def _deliver_to_subscribers(self, message: PDCSCADAMessage, sender_id: str, latency: float):
        """Deliver message to all interested subscribers"""
        message_type = message.message_type
        current_time = time.time()
        
        for subscriber_id, subscription in self.subscriptions.items():
            # Check if subscriber is interested in this message type
            interested = False
            
            if message_type == 'data_snapshot':
                # Check if any of the subscriber's data types match
                subscriber_data_types = subscription.get('data_types', [])
                interested = any(dt in ['voltage', 'frequency', 'power_flow', 'all'] 
                               for dt in subscriber_data_types)
                
                # Check update interval
                last_update = subscription.get('last_update', 0.0)
                update_interval = subscription.get('update_interval', 1.0)
                if current_time - last_update < update_interval:
                    interested = False  # Too soon since last update
                
            elif message_type == 'event_notification':
                # All subscribers should receive events
                subscriber_data_types = subscription.get('data_types', [])
                interested = 'events' in subscriber_data_types or 'all' in subscriber_data_types
            
            if interested:
                # Update last delivery time
                subscription['last_update'] = current_time
                
                # Find handler for this subscriber
                if message_type in self.message_handlers:
                    for handler in self.message_handlers[message_type]:
                        try:
                            await handler(message, sender_id, latency)
                            logger.debug(f"PDC-SCADA Link {self.link_name}: Delivered {message_type} "
                                       f"to subscriber {subscriber_id}")
                        except Exception as e:
                            logger.error(f"PDC-SCADA Link {self.link_name}: "
                                       f"Subscriber {subscriber_id} handler error: {e}")
    
    async def send_pdc_to_scada(self, message: PDCSCADAMessage, sender_id: str, receiver_id: str):
        """Send message from PDC to SCADA"""
        message.sequence_number = self.sequence_counter
        self.sequence_counter += 1
        
        await self.pdc_to_scada_queue.put((message, sender_id, receiver_id))
        self.stats.messages_sent += 1
        
        # Update specific message type stats
        if isinstance(message, DataSnapshot):
            self.stats.data_snapshots += 1
        elif isinstance(message, EventNotification):
            self.stats.event_notifications += 1
    
    async def send_scada_to_pdc(self, message: PDCSCADAMessage, sender_id: str, receiver_id: str):
        """Send message from SCADA to PDC"""
        message.sequence_number = self.sequence_counter
        self.sequence_counter += 1
        
        await self.scada_to_pdc_queue.put((message, sender_id, receiver_id))
        self.stats.messages_sent += 1
        
        # Update specific message type stats
        if isinstance(message, SubscriptionRequest):
            self.stats.subscription_requests += 1
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add a message handler for specific message type"""
        self.message_handlers[message_type].append(handler)
        logger.debug(f"PDC-SCADA Link {self.link_name}: Added handler for {message_type}")
    
    def remove_message_handler(self, message_type: str, handler: Callable):
        """Remove a message handler"""
        if message_type in self.message_handlers:
            if handler in self.message_handlers[message_type]:
                self.message_handlers[message_type].remove(handler)
    
    async def subscribe(self, subscriber_id: str, data_types: List[str], 
                       update_interval: float, threshold_config: Dict[str, Any] = None):
        """Subscribe to PDC data"""
        subscription = {
            'data_types': data_types,
            'update_interval': update_interval,
            'threshold_config': threshold_config or {},
            'last_update': 0.0
        }
        
        self.subscriptions[subscriber_id] = subscription
        logger.info(f"PDC-SCADA Link {self.link_name}: Subscription created for {subscriber_id}")
        
        # Create subscription request message
        request = SubscriptionRequest(
            timestamp=time.time(),
            message_type='subscription_request',
            sequence_number=0,
            data_types=data_types,
            update_interval=update_interval,
            threshold_config=threshold_config or {}
        )
        
        await self.send_scada_to_pdc(request, subscriber_id, 'pdc')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get link statistics"""
        return {
            'messages_sent': self.stats.messages_sent,
            'messages_received': self.stats.messages_received,
            'data_snapshots': self.stats.data_snapshots,
            'event_notifications': self.stats.event_notifications,
            'subscription_requests': self.stats.subscription_requests,
            'average_latency_ms': round(self.stats.average_latency, 2),
            'active_subscriptions': len(self.subscriptions)
        }

# Global link instances
pdc_scada_links: Dict[str, PDCSCADALink] = {}

def get_pdc_scada_link(link_name: str, base_latency: float = 0.15) -> PDCSCADALink:
    """Get or create a PDC-SCADA link instance"""
    if link_name not in pdc_scada_links:
        pdc_scada_links[link_name] = PDCSCADALink(link_name, base_latency)
    return pdc_scada_links[link_name]

def create_data_snapshot(grid_summary: Dict, pmu_count: int, data_quality: float,
                        voltage_profile: Dict, frequency_summary: Dict, 
                        system_status: str, alarms: List = None) -> DataSnapshot:
    """Helper function to create data snapshot message"""
    return DataSnapshot(
        timestamp=time.time(),
        message_type='data_snapshot',
        sequence_number=0,
        grid_summary=grid_summary,
        pmu_count=pmu_count,
        data_quality=data_quality,
        voltage_profile=voltage_profile,
        frequency_summary=frequency_summary,
        system_status=system_status,
        alarms=alarms or []
    )

def create_event_notification(event_type: str, severity: str, location: str,
                            description: str, value: float, threshold: float) -> EventNotification:
    """Helper function to create event notification message"""
    return EventNotification(
        timestamp=time.time(),
        message_type='event_notification',
        sequence_number=0,
        event_type=event_type,
        severity=severity,
        location=location,
        description=description,
        value=value,
        threshold=threshold
    )

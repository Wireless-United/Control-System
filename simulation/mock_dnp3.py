#!/usr/bin/env python3
"""
Mock DNP3 Communication Protocol for SCADA-RTU System

This module implements an in-memory mock DNP3 protocol that simulates
communication between SCADA master and RTU outstations without network sockets.
All communication happens through shared data structures.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DNP3ObjectType(Enum):
    """DNP3 Object Types"""
    BINARY_INPUT = "binary_input"
    ANALOG_INPUT = "analog_input"
    BINARY_OUTPUT = "binary_output" 
    ANALOG_OUTPUT = "analog_output"

class DNP3Quality(Enum):
    """DNP3 Quality Flags"""
    GOOD = 0x00
    RESTART = 0x01
    COMM_LOST = 0x02
    REMOTE_FORCED = 0x04
    LOCAL_FORCED = 0x08
    OVER_RANGE = 0x10
    REFERENCE_ERR = 0x20
    RESERVED = 0x40
    ONLINE = 0x80
    QUESTIONABLE = 0x90

@dataclass
class DNP3Point:
    """DNP3 Data Point"""
    index: int
    name: str
    object_type: DNP3ObjectType
    value: float
    quality: DNP3Quality = DNP3Quality.GOOD
    timestamp: float = field(default_factory=time.time)
    
class MockDNP3Channel:
    """Mock DNP3 Communication Channel"""
    
    def __init__(self):
        self.rtu_data: Dict[int, Dict[int, DNP3Point]] = {}  # rtu_id -> {point_index -> DNP3Point}
        self.scada_requests: List[Dict] = []
        self.attack_interceptor: Optional[Callable] = None
        self._lock = threading.Lock()
        
    def register_rtu(self, rtu_id: int):
        """Register an RTU on the channel"""
        with self._lock:
            if rtu_id not in self.rtu_data:
                self.rtu_data[rtu_id] = {}
                logger.debug(f"RTU {rtu_id} registered on DNP3 channel")
    
    def update_rtu_point(self, rtu_id: int, point: DNP3Point):
        """Update a data point at the RTU"""
        with self._lock:
            if rtu_id not in self.rtu_data:
                self.register_rtu(rtu_id)
            
            self.rtu_data[rtu_id][point.index] = point
            logger.debug(f"RTU {rtu_id} updated point {point.index}: {point.value}")
    
    def poll_rtu(self, rtu_id: int, scada_id: int = 1) -> Dict[int, DNP3Point]:
        """SCADA polls RTU for all data points"""
        with self._lock:
            if rtu_id not in self.rtu_data:
                logger.warning(f"RTU {rtu_id} not found on channel")
                return {}
            
            # Copy data for response
            response_data = self.rtu_data[rtu_id].copy()
            
            # Apply attack interception if active
            if self.attack_interceptor:
                response_data = self.attack_interceptor(rtu_id, scada_id, response_data)
                logger.debug(f"Attack interceptor modified RTU {rtu_id} response")
            
            # Log the poll request
            self.scada_requests.append({
                'timestamp': time.time(),
                'scada_id': scada_id,
                'rtu_id': rtu_id,
                'points_count': len(response_data)
            })
            
            return response_data
    
    def send_control(self, rtu_id: int, point_index: int, value: float, scada_id: int = 1) -> bool:
        """SCADA sends control command to RTU"""
        with self._lock:
            if rtu_id not in self.rtu_data:
                logger.warning(f"RTU {rtu_id} not found for control command")
                return False
            
            # Create control point
            control_point = DNP3Point(
                index=point_index,
                name=f"control_{point_index}",
                object_type=DNP3ObjectType.BINARY_OUTPUT,
                value=value,
                timestamp=time.time()
            )
            
            # Apply attack interception if active
            if self.attack_interceptor:
                modified_data = self.attack_interceptor(rtu_id, scada_id, {point_index: control_point})
                if point_index in modified_data:
                    control_point = modified_data[point_index]
                    logger.debug(f"Attack interceptor modified control command to RTU {rtu_id}")
            
            self.rtu_data[rtu_id][point_index] = control_point
            logger.info(f"Control command sent to RTU {rtu_id}, point {point_index}: {value}")
            return True
    
    def set_attack_interceptor(self, interceptor: Optional[Callable]):
        """Set attack interceptor function"""
        with self._lock:
            self.attack_interceptor = interceptor
            if interceptor:
                logger.warning("DNP3 Attack Interceptor ACTIVATED")
            else:
                logger.info("DNP3 Attack Interceptor DEACTIVATED")
    
    def activate_attack_interceptor(self):
        """Activate the attack interceptor with manipulation capabilities"""
        self.attack_interceptor = self._manipulative_attack_interceptor
        self.attack_manipulations = {}  # Store attack parameters per RTU
        logger.warning("DNP3 Attack Interceptor ACTIVATED")
    
    def deactivate_attack_interceptor(self):
        """Deactivate the attack interceptor"""
        self.attack_interceptor = None
        if hasattr(self, 'attack_manipulations'):
            self.attack_manipulations = {}
        logger.info("DNP3 Attack Interceptor DEACTIVATED")
    
    def set_attack_manipulation(self, rtu_id: int, voltage_offset: float = 0.0, frequency_offset: float = 0.0):
        """Set attack manipulation parameters for specific RTU"""
        if not hasattr(self, 'attack_manipulations'):
            self.attack_manipulations = {}
        
        self.attack_manipulations[rtu_id] = {
            'voltage_offset': voltage_offset,
            'frequency_offset': frequency_offset,
            'timestamp': time.time()
        }
        logger.warning(f"Attack manipulation set for RTU {rtu_id}: V={voltage_offset:+.3f}pu, F={frequency_offset:+.2f}Hz")
    
    def _manipulative_attack_interceptor(self, rtu_id: int, scada_id: int, data: Dict[int, DNP3Point]) -> Dict[int, DNP3Point]:
        """Advanced attack interceptor that manipulates measurement data"""
        logger.warning(f"🔴 Attack intercepted RTU {rtu_id} data: {len(data)} points")
        
        # Check if we have manipulation parameters for this RTU
        if hasattr(self, 'attack_manipulations') and rtu_id in self.attack_manipulations:
            manipulations = self.attack_manipulations[rtu_id]
            voltage_offset = manipulations.get('voltage_offset', 0.0)
            frequency_offset = manipulations.get('frequency_offset', 0.0)
            
            # Apply manipulations to the data
            manipulated_data = data.copy()
            
            for point_index, point in manipulated_data.items():
                # Create a modified copy of the point
                modified_point = DNP3Point(
                    index=point.index,
                    object_type=point.object_type,
                    name=point.name,
                    value=point.value,
                    quality=point.quality,
                    timestamp=point.timestamp
                )
                
                # Apply voltage manipulation
                if 'voltage_magnitude' in point.name.lower() and voltage_offset != 0:
                    original_value = modified_point.value
                    modified_point.value = max(0.1, original_value + voltage_offset)  # Prevent negative voltages
                    modified_point.quality = DNP3Quality.QUESTIONABLE  # Mark as questionable
                    logger.debug(f"RTU {rtu_id} voltage manipulated: {original_value:.4f} -> {modified_point.value:.4f}")
                
                # Apply frequency manipulation
                elif 'frequency' in point.name.lower() and frequency_offset != 0:
                    original_value = modified_point.value
                    modified_point.value = original_value + frequency_offset
                    modified_point.quality = DNP3Quality.QUESTIONABLE  # Mark as questionable
                    logger.debug(f"RTU {rtu_id} frequency manipulated: {original_value:.2f} -> {modified_point.value:.2f}")
                
                manipulated_data[point_index] = modified_point
            
            return manipulated_data
        
        # No manipulation - just log the interception
        return data
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        with self._lock:
            return {
                'registered_rtus': len(self.rtu_data),
                'total_requests': len(self.scada_requests),
                'active_attack': self.attack_interceptor is not None,
                'rtu_ids': list(self.rtu_data.keys())
            }

# Global DNP3 channel instance
dnp3_channel = MockDNP3Channel()
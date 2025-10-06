#!/usr/bin/env python3
"""
Integrated SCADA System for IEEE 39-Bus Power System

This module provides SCADA master station functionality that integrates
directly with the power system simulation through mock DNP3 protocol.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from mock_dnp3 import dnp3_channel, DNP3Point, DNP3ObjectType, DNP3Quality

logger = logging.getLogger(__name__)

@dataclass
class SCADAMeasurement:
    """SCADA measurement with metadata"""
    rtu_id: int
    point_name: str
    value: float
    unit: str
    quality: str
    timestamp: float
    is_stale: bool = False

@dataclass
class SCADAAlarm:
    """SCADA alarm/event"""
    alarm_id: int
    rtu_id: int
    message: str
    severity: str  # INFO, WARNING, ALARM, CRITICAL
    timestamp: float
    acknowledged: bool = False

class SCADAMaster:
    """SCADA Master Station"""
    
    def __init__(self, station_id: int = 1):
        self.station_id = station_id
        self.is_running = False
        self.rtus: Dict[int, Dict] = {}  # rtu_id -> RTU info
        self.measurements: Dict[str, SCADAMeasurement] = {}  # measurement_key -> measurement
        self.alarms: List[SCADAAlarm] = []
        self.poll_interval = 2.0  # seconds
        self.poll_thread = None
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'polls_sent': 0,
            'responses_received': 0,
            'commands_sent': 0,
            'active_alarms': 0,
            'start_time': time.time()
        }
        
        logger.info(f"SCADA Master {station_id} initialized")
    
    def add_rtu(self, rtu_id: int, name: str, bus_number: int):
        """Add RTU to SCADA system"""
        with self._lock:
            self.rtus[rtu_id] = {
                'name': name,
                'bus_number': bus_number,
                'last_poll': 0,
                'is_online': False,
                'measurement_count': 0
            }
            logger.info(f"Added RTU {rtu_id} ({name}) for Bus {bus_number}")
    
    def start(self):
        """Start SCADA polling"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start polling thread
        self.poll_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.poll_thread.start()
        
        logger.info(f"SCADA Master {self.station_id} started with {len(self.rtus)} RTUs")
    
    def stop(self):
        """Stop SCADA polling"""
        self.is_running = False
        if self.poll_thread:
            self.poll_thread.join(timeout=5)
        logger.info(f"SCADA Master {self.station_id} stopped")
    
    def _polling_loop(self):
        """Main polling loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for rtu_id in self.rtus.keys():
                    rtu_info = self.rtus[rtu_id]
                    
                    # Check if it's time to poll this RTU
                    if current_time - rtu_info['last_poll'] >= self.poll_interval:
                        self._poll_rtu(rtu_id)
                        rtu_info['last_poll'] = current_time
                
                time.sleep(0.5)  # Small delay
                
            except Exception as e:
                logger.error(f"Error in SCADA polling loop: {e}")
                time.sleep(1)
    
    def _poll_rtu(self, rtu_id: int):
        """Poll specific RTU"""
        try:
            # Poll RTU through mock DNP3
            response_data = dnp3_channel.poll_rtu(rtu_id, self.station_id)
            
            self.stats['polls_sent'] += 1
            
            if response_data:
                self.stats['responses_received'] += 1
                self.rtus[rtu_id]['is_online'] = True
                self.rtus[rtu_id]['measurement_count'] = len(response_data)
                
                # Process measurements
                for point_index, point in response_data.items():
                    self._process_measurement(rtu_id, point)
                
                logger.debug(f"Successfully polled RTU {rtu_id}: {len(response_data)} points")
            else:
                self.rtus[rtu_id]['is_online'] = False
                logger.warning(f"No response from RTU {rtu_id}")
                
        except Exception as e:
            self.rtus[rtu_id]['is_online'] = False
            logger.error(f"Error polling RTU {rtu_id}: {e}")
    
    def _process_measurement(self, rtu_id: int, point: DNP3Point):
        """Process received measurement"""
        measurement_key = f"RTU_{rtu_id}_{point.name}"
        
        # Determine unit based on measurement type
        unit = "pu"
        if "voltage" in point.name.lower():
            unit = "pu"
        elif "power" in point.name.lower():
            unit = "MW"
        elif "frequency" in point.name.lower():
            unit = "Hz"
        
        # Create SCADA measurement
        measurement = SCADAMeasurement(
            rtu_id=rtu_id,
            point_name=point.name,
            value=point.value,
            unit=unit,
            quality=point.quality.name,
            timestamp=point.timestamp
        )
        
        with self._lock:
            self.measurements[measurement_key] = measurement
        
        # Check for alarms
        self._check_measurement_alarms(rtu_id, point)
    
    def _check_measurement_alarms(self, rtu_id: int, point: DNP3Point):
        """Check measurement for alarm conditions"""
        # Voltage magnitude alarms (not angles!)
        if point.name.lower() == "voltage_magnitude":
            if point.value < 0.95:
                self._generate_alarm(rtu_id, "CRITICAL", f"Low voltage: {point.value:.3f} pu")
            elif point.value > 1.05:
                self._generate_alarm(rtu_id, "CRITICAL", f"High voltage: {point.value:.3f} pu")
        
        # Frequency alarms
        elif "frequency" in point.name.lower():
            if abs(point.value - 50.0) > 0.5:
                self._generate_alarm(rtu_id, "ALARM", f"Frequency deviation: {point.value:.2f} Hz")
        
        # Communication quality alarms
        if point.quality != DNP3Quality.GOOD:
            self._generate_alarm(rtu_id, "WARNING", f"Poor data quality: {point.quality.name}")
    
    def _generate_alarm(self, rtu_id: int, severity: str, message: str):
        """Generate system alarm"""
        alarm_id = len(self.alarms) + 1
        alarm = SCADAAlarm(
            alarm_id=alarm_id,
            rtu_id=rtu_id,
            message=message,
            severity=severity,
            timestamp=time.time()
        )
        
        with self._lock:
            self.alarms.append(alarm)
            self.stats['active_alarms'] = len([a for a in self.alarms if not a.acknowledged])
        
        logger.warning(f"ALARM {alarm_id}: RTU {rtu_id} - {message}")
    
    def send_control_command(self, rtu_id: int, point_index: int, value: float) -> bool:
        """Send control command to RTU"""
        try:
            success = dnp3_channel.send_control(rtu_id, point_index, value, self.station_id)
            if success:
                self.stats['commands_sent'] += 1
                logger.info(f"Control command sent to RTU {rtu_id}: point {point_index} = {value}")
            return success
        except Exception as e:
            logger.error(f"Error sending control command to RTU {rtu_id}: {e}")
            return False
    
    def get_measurements(self) -> Dict[str, SCADAMeasurement]:
        """Get all current measurements"""
        with self._lock:
            return self.measurements.copy()
    
    def get_alarms(self, unacknowledged_only: bool = False) -> List[SCADAAlarm]:
        """Get system alarms"""
        with self._lock:
            if unacknowledged_only:
                return [a for a in self.alarms if not a.acknowledged]
            return self.alarms.copy()
    
    def acknowledge_alarm(self, alarm_id: int):
        """Acknowledge alarm"""
        with self._lock:
            for alarm in self.alarms:
                if alarm.alarm_id == alarm_id:
                    alarm.acknowledged = True
                    logger.info(f"Alarm {alarm_id} acknowledged")
                    break
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        online_rtus = sum(1 for rtu in self.rtus.values() if rtu['is_online'])
        uptime = time.time() - self.stats['start_time']
        
        return {
            'station_id': self.station_id,
            'uptime_hours': uptime / 3600,
            'rtus_online': f"{online_rtus}/{len(self.rtus)}",
            'polls_sent': self.stats['polls_sent'],
            'responses_received': self.stats['responses_received'],
            'commands_sent': self.stats['commands_sent'],
            'active_alarms': self.stats['active_alarms'],
            'total_measurements': len(self.measurements),
            'is_running': self.is_running
        }

# Global SCADA instance
scada_master = SCADAMaster()
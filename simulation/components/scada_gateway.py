"""
SCADA Gateway

Acts as a gateway between PDC and SCADA master, translating high-frequency
synchrophasor data into SCADA-friendly updates and providing integration
between real-time PMU data and traditional SCADA operations.

Features:
- Subscribes to PDC aggregated data streams
- Translates synchrophasor data into SCADA-friendly updates
- Provides polling interface for SCADA access to PMU-derived values
- Supports manual SCADA control actions
- Generates summaries, averages, and threshold alarms
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque, defaultdict

from protocols.pdc_scada_link import (
    PDCSCADALink, DataSnapshot, EventNotification, SubscriptionRequest,
    get_pdc_scada_link, create_event_notification
)

logger = logging.getLogger(__name__)

@dataclass
class SCADADataPoint:
    """SCADA data point derived from PMU data"""
    tag_name: str
    value: float
    quality: str
    timestamp: float
    source_type: str  # 'PMU', 'RTU', 'CALCULATED'
    unit: str
    description: str

@dataclass
class AlarmCondition:
    """Alarm condition configuration"""
    tag_name: str
    alarm_type: str  # 'HIGH', 'LOW', 'DEVIATION', 'RATE_OF_CHANGE'
    threshold_value: float
    severity: str    # 'INFO', 'WARNING', 'CRITICAL'
    enabled: bool = True
    hysteresis: float = 0.0
    description: str = ""

class SCADAGateway:
    """Gateway between PDC and SCADA systems"""
    
    def __init__(self, gateway_id: str, pdc_link: PDCSCADALink, 
                 update_interval: float = 5.0):
        self.gateway_id = gateway_id
        self.pdc_link = pdc_link
        self.update_interval = update_interval
        self.running = False
        
        # Data storage
        self.scada_data_points: Dict[str, SCADADataPoint] = {}
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alarm_conditions: Dict[str, AlarmCondition] = {}
        self.active_alarms: Dict[str, Dict] = {}
        
        # Subscription management
        self.pdc_subscription_active = False
        self.last_pdc_data: Optional[DataSnapshot] = None
        
        # Background tasks
        self.data_processing_task: Optional[asyncio.Task] = None
        self.alarm_monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'pdc_updates_received': 0,
            'scada_data_points_updated': 0,
            'alarms_generated': 0,
            'polling_requests': 0,
            'last_update_time': 0.0
        }
        
        # Configure default alarm conditions
        self._setup_default_alarms()
        
        # Register message handlers
        self.pdc_link.add_message_handler('data_snapshot', self._handle_pdc_data)
        self.pdc_link.add_message_handler('event_notification', self._handle_pdc_event)
        
        logger.info(f"SCADA Gateway {gateway_id} initialized")
    
    def _setup_default_alarms(self):
        """Setup default alarm conditions"""
        default_alarms = [
            AlarmCondition('SYSTEM_FREQUENCY', 'DEVIATION', 0.5, 'WARNING', 
                         description='System frequency deviation > ±0.5 Hz'),
            AlarmCondition('SYSTEM_FREQUENCY', 'HIGH', 60.5, 'CRITICAL',
                         description='System frequency > 60.5 Hz'),
            AlarmCondition('SYSTEM_FREQUENCY', 'LOW', 59.5, 'CRITICAL',
                         description='System frequency < 59.5 Hz'),
            AlarmCondition('VOLTAGE_DEVIATION', 'HIGH', 1.1, 'WARNING',
                         description='Bus voltage > 1.1 pu'),
            AlarmCondition('VOLTAGE_DEVIATION', 'LOW', 0.9, 'WARNING',
                         description='Bus voltage < 0.9 pu'),
            AlarmCondition('PMU_DATA_QUALITY', 'LOW', 0.8, 'WARNING',
                         description='PMU data quality < 80%'),
        ]
        
        for alarm in default_alarms:
            alarm_key = f"{alarm.tag_name}_{alarm.alarm_type}"
            self.alarm_conditions[alarm_key] = alarm
    
    async def start(self):
        """Start the SCADA gateway"""
        if self.running:
            return
        
        self.running = True
        
        # Start the PDC-SCADA link first
        await self.pdc_link.start_link()
        
        # Start background tasks
        self.data_processing_task = asyncio.create_task(self._data_processing_loop())
        self.alarm_monitoring_task = asyncio.create_task(self._alarm_monitoring_loop())
        
        # Subscribe to PDC data
        await self._subscribe_to_pdc()
        
        logger.info(f"SCADA Gateway {self.gateway_id} started")
    
    async def stop(self):
        """Stop the SCADA gateway"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.data_processing_task:
            self.data_processing_task.cancel()
            try:
                await self.data_processing_task
            except asyncio.CancelledError:
                pass
        
        if self.alarm_monitoring_task:
            self.alarm_monitoring_task.cancel()
            try:
                await self.alarm_monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop the PDC-SCADA link
        await self.pdc_link.stop_link()
        
        logger.info(f"SCADA Gateway {self.gateway_id} stopped")
    
    async def _subscribe_to_pdc(self):
        """Subscribe to PDC data streams"""
        data_types = ['voltage', 'frequency', 'power_flow', 'events']
        threshold_config = {
            'frequency_deviation': 0.5,
            'voltage_deviation': 0.1,
            'data_quality_minimum': 0.8
        }
        
        await self.pdc_link.subscribe(
            subscriber_id=self.gateway_id,
            data_types=data_types,
            update_interval=self.update_interval,
            threshold_config=threshold_config
        )
        
        self.pdc_subscription_active = True
        logger.info(f"SCADA Gateway {self.gateway_id}: Subscribed to PDC data")
    
    async def _handle_pdc_data(self, message: DataSnapshot, sender_id: str, latency: float):
        """Handle PDC data snapshot"""
        self.last_pdc_data = message
        self.stats['pdc_updates_received'] += 1
        self.stats['last_update_time'] = time.time()
        
        # Convert PDC data to SCADA data points
        await self._convert_pdc_to_scada_data(message)
        
        logger.debug(f"SCADA Gateway {self.gateway_id}: Processed PDC data snapshot, "
                    f"latency: {latency:.1f}ms, quality: {message.data_quality:.2f}")
    
    async def _handle_pdc_event(self, message: EventNotification, sender_id: str, latency: float):
        """Handle PDC event notification"""
        # Convert PDC event to SCADA alarm
        alarm_key = f"{message.event_type}_{message.location}"
        
        alarm_data = {
            'timestamp': message.timestamp,
            'event_type': message.event_type,
            'severity': message.severity,
            'location': message.location,
            'description': message.description,
            'value': message.value,
            'threshold': message.threshold,
            'acknowledged': False
        }
        
        self.active_alarms[alarm_key] = alarm_data
        self.stats['alarms_generated'] += 1
        
        logger.warning(f"SCADA Gateway {self.gateway_id}: PDC Event - {message.severity} "
                      f"{message.event_type} at {message.location}: {message.description}")
    
    async def _convert_pdc_to_scada_data(self, pdc_data: DataSnapshot):
        """Convert PDC data to SCADA data points"""
        timestamp = pdc_data.timestamp
        
        # System-wide frequency data
        freq_summary = pdc_data.frequency_summary
        self._update_scada_point('SYSTEM_FREQUENCY', freq_summary.get('avg', 60.0), 
                               'GOOD', timestamp, 'PMU', 'Hz', 'Average system frequency from PMUs')
        self._update_scada_point('FREQUENCY_DEVIATION', abs(freq_summary.get('avg', 60.0) - 60.0),
                               'GOOD', timestamp, 'CALCULATED', 'Hz', 'Frequency deviation from nominal')
        
        # PMU data quality
        self._update_scada_point('PMU_DATA_QUALITY', pdc_data.data_quality * 100,
                               'GOOD', timestamp, 'PMU', '%', 'Overall PMU data quality')
        self._update_scada_point('PMU_COUNT_ONLINE', pdc_data.pmu_count,
                               'GOOD', timestamp, 'PMU', 'count', 'Number of online PMUs')
        
        # Voltage profile summary
        if pdc_data.voltage_profile:
            voltages = []
            for bus_id, voltage_data in pdc_data.voltage_profile.items():
                voltage_mag = voltage_data.get('magnitude', 1.0)
                voltage_angle = voltage_data.get('angle', 0.0)
                voltages.append(voltage_mag)
                
                # Individual bus data points
                self._update_scada_point(f'BUS_{bus_id}_VOLTAGE_MAG', voltage_mag,
                                       'GOOD', timestamp, 'PMU', 'pu', f'Bus {bus_id} voltage magnitude')
                self._update_scada_point(f'BUS_{bus_id}_VOLTAGE_ANG', voltage_angle,
                                       'GOOD', timestamp, 'PMU', 'deg', f'Bus {bus_id} voltage angle')
            
            if voltages:
                # System voltage statistics
                self._update_scada_point('VOLTAGE_AVG', statistics.mean(voltages),
                                       'GOOD', timestamp, 'CALCULATED', 'pu', 'Average system voltage')
                self._update_scada_point('VOLTAGE_MIN', min(voltages),
                                       'GOOD', timestamp, 'CALCULATED', 'pu', 'Minimum system voltage')
                self._update_scada_point('VOLTAGE_MAX', max(voltages),
                                       'GOOD', timestamp, 'CALCULATED', 'pu', 'Maximum system voltage')
        
        # System status
        self._update_scada_point('SYSTEM_STATUS', 1.0 if pdc_data.system_status == 'STABLE' else 0.0,
                               'GOOD', timestamp, 'PMU', 'bool', 'Overall system stability status')
        
        self.stats['scada_data_points_updated'] += len(self.scada_data_points)
    
    def _update_scada_point(self, tag_name: str, value: float, quality: str, 
                          timestamp: float, source_type: str, unit: str, description: str):
        """Update a SCADA data point"""
        data_point = SCADADataPoint(
            tag_name=tag_name,
            value=value,
            quality=quality,
            timestamp=timestamp,
            source_type=source_type,
            unit=unit,
            description=description
        )
        
        self.scada_data_points[tag_name] = data_point
        self.historical_data[tag_name].append((timestamp, value, quality))
    
    async def _data_processing_loop(self):
        """Background data processing loop"""
        try:
            while self.running:
                # Perform periodic data processing tasks
                await self._calculate_derived_values()
                await asyncio.sleep(1.0)  # Process every second
        except asyncio.CancelledError:
            logger.info(f"SCADA Gateway {self.gateway_id}: Data processing loop cancelled")
    
    async def _calculate_derived_values(self):
        """Calculate derived values from historical data"""
        current_time = time.time()
        
        # Calculate rate of change for frequency
        if 'SYSTEM_FREQUENCY' in self.historical_data:
            freq_history = list(self.historical_data['SYSTEM_FREQUENCY'])
            if len(freq_history) >= 2:
                recent_data = freq_history[-10:]  # Last 10 samples
                if len(recent_data) >= 2:
                    time_diff = recent_data[-1][0] - recent_data[0][0]
                    freq_diff = recent_data[-1][1] - recent_data[0][1]
                    if time_diff > 0:
                        rocof = freq_diff / time_diff
                        self._update_scada_point('FREQUENCY_ROCOF', rocof, 'GOOD', current_time,
                                               'CALCULATED', 'Hz/s', 'Rate of change of frequency')
    
    async def _alarm_monitoring_loop(self):
        """Background alarm monitoring loop"""
        try:
            while self.running:
                await self._check_alarm_conditions()
                await asyncio.sleep(2.0)  # Check alarms every 2 seconds
        except asyncio.CancelledError:
            logger.info(f"SCADA Gateway {self.gateway_id}: Alarm monitoring loop cancelled")
    
    async def _check_alarm_conditions(self):
        """Check alarm conditions against current data"""
        current_time = time.time()
        
        for alarm_key, alarm_condition in self.alarm_conditions.items():
            if not alarm_condition.enabled:
                continue
            
            # Get current value for the alarm tag
            if alarm_condition.tag_name in self.scada_data_points:
                current_point = self.scada_data_points[alarm_condition.tag_name]
                current_value = current_point.value
                
                # Check alarm condition
                alarm_triggered = False
                
                if alarm_condition.alarm_type == 'HIGH':
                    alarm_triggered = current_value > alarm_condition.threshold_value
                elif alarm_condition.alarm_type == 'LOW':
                    alarm_triggered = current_value < alarm_condition.threshold_value
                elif alarm_condition.alarm_type == 'DEVIATION':
                    if alarm_condition.tag_name == 'SYSTEM_FREQUENCY':
                        deviation = abs(current_value - 60.0)
                        alarm_triggered = deviation > alarm_condition.threshold_value
                
                # Handle alarm state
                if alarm_triggered and alarm_key not in self.active_alarms:
                    # New alarm
                    await self._generate_alarm(alarm_condition, current_value, current_time)
                elif not alarm_triggered and alarm_key in self.active_alarms:
                    # Clear alarm
                    await self._clear_alarm(alarm_key, current_time)
    
    async def _generate_alarm(self, alarm_condition: AlarmCondition, value: float, timestamp: float):
        """Generate a new alarm"""
        alarm_key = f"{alarm_condition.tag_name}_{alarm_condition.alarm_type}"
        
        alarm_data = {
            'timestamp': timestamp,
            'tag_name': alarm_condition.tag_name,
            'alarm_type': alarm_condition.alarm_type,
            'severity': alarm_condition.severity,
            'description': alarm_condition.description,
            'value': value,
            'threshold': alarm_condition.threshold_value,
            'acknowledged': False
        }
        
        self.active_alarms[alarm_key] = alarm_data
        self.stats['alarms_generated'] += 1
        
        logger.warning(f"SCADA Gateway {self.gateway_id}: ALARM - {alarm_condition.severity} "
                      f"{alarm_condition.tag_name} {alarm_condition.alarm_type}: "
                      f"Value {value:.3f} vs threshold {alarm_condition.threshold_value:.3f}")
    
    async def _clear_alarm(self, alarm_key: str, timestamp: float):
        """Clear an active alarm"""
        if alarm_key in self.active_alarms:
            alarm_data = self.active_alarms[alarm_key]
            logger.info(f"SCADA Gateway {self.gateway_id}: ALARM CLEARED - "
                       f"{alarm_data['tag_name']} {alarm_data['alarm_type']}")
            del self.active_alarms[alarm_key]
    
    # SCADA Interface Methods
    def poll_data_point(self, tag_name: str) -> Optional[SCADADataPoint]:
        """Poll a specific SCADA data point (SCADA interface)"""
        self.stats['polling_requests'] += 1
        return self.scada_data_points.get(tag_name)
    
    def poll_multiple_points(self, tag_names: List[str]) -> Dict[str, SCADADataPoint]:
        """Poll multiple SCADA data points"""
        self.stats['polling_requests'] += len(tag_names)
        result = {}
        for tag_name in tag_names:
            if tag_name in self.scada_data_points:
                result[tag_name] = self.scada_data_points[tag_name]
        return result
    
    def get_all_data_points(self) -> Dict[str, SCADADataPoint]:
        """Get all available SCADA data points"""
        self.stats['polling_requests'] += 1
        return self.scada_data_points.copy()
    
    def get_active_alarms(self) -> Dict[str, Dict]:
        """Get all active alarms"""
        return self.active_alarms.copy()
    
    def acknowledge_alarm(self, alarm_key: str) -> bool:
        """Acknowledge an alarm"""
        if alarm_key in self.active_alarms:
            self.active_alarms[alarm_key]['acknowledged'] = True
            logger.info(f"SCADA Gateway {self.gateway_id}: Alarm {alarm_key} acknowledged")
            return True
        return False
    
    def get_historical_data(self, tag_name: str, max_samples: int = 50) -> List[tuple]:
        """Get historical data for a tag"""
        if tag_name in self.historical_data:
            history = list(self.historical_data[tag_name])
            return history[-max_samples:] if len(history) > max_samples else history
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        stats = self.stats.copy()
        stats['active_alarms_count'] = len(self.active_alarms)
        stats['data_points_count'] = len(self.scada_data_points)
        stats['pdc_subscription_active'] = self.pdc_subscription_active
        return stats
    
    # Control Interface Methods
    async def send_control_command(self, command_type: str, target: str, parameters: Dict[str, Any]):
        """Send control command through PDC link (if needed)"""
        # This could be extended to send commands back to PDC/PMUs
        logger.info(f"SCADA Gateway {self.gateway_id}: Control command {command_type} "
                   f"for {target} with parameters {parameters}")
    
    def update_alarm_condition(self, tag_name: str, alarm_type: str, 
                             threshold_value: float, enabled: bool = True):
        """Update alarm condition configuration"""
        alarm_key = f"{tag_name}_{alarm_type}"
        if alarm_key in self.alarm_conditions:
            self.alarm_conditions[alarm_key].threshold_value = threshold_value
            self.alarm_conditions[alarm_key].enabled = enabled
            logger.info(f"SCADA Gateway {self.gateway_id}: Updated alarm condition {alarm_key}")
        else:
            logger.warning(f"SCADA Gateway {self.gateway_id}: Alarm condition {alarm_key} not found")

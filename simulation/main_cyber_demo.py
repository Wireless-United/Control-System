#!/usr/bin/env python3
"""
Real-time SCADA Cybersecurity Demonstration
Entry point for starting the integrated power system with SCADA-RTU network
"""

import logging
import time
from datetime import datetime
from ieee39_system_strict import StrictIEEE39BusSystem
from integrated_scada import scada_master
from integrated_rtu import rtu_manager
from mock_dnp3 import dnp3_channel
from system_status import write_system_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberSecurityDemo:
    """Real-time cybersecurity demonstration system"""
    
    def __init__(self):
        self.power_system = None
        self.rtu_list = []
        self.running = False
        
    def initialize_system(self):
        """Initialize the complete power system and SCADA network"""
        try:
            print("\n🔧 INITIALIZING CYBERSECURITY DEMONSTRATION SYSTEM")
            print("=" * 60)
            
            # 1. Initialize power system
            print("1️⃣ Initializing IEEE 39-bus power system...")
            self.power_system = StrictIEEE39BusSystem()
            analysis = self.power_system.run_strict_ieee39_analysis()
            
            if not analysis['pypower_analysis']:
                raise RuntimeError("Power flow did not converge")
                
            state = self.power_system.get_system_state()
            print(f"  ✓ Power system operational")
            print(f"  ✓ Total Load: {state['total_load_mw']:.1f} MW")
            print(f"  ✓ Total Generation: {state['total_generation_mw']:.1f} MW")
            
            # 2. Create RTU outstations
            print("\n2️⃣ Creating RTU outstations...")
            rtu_manager.create_standard_rtus(self.power_system)
            self.rtu_list = rtu_manager.get_rtu_list()
            print(f"  ✓ Created {len(self.rtu_list)} RTU outstations")
            
            # 3. Configure SCADA master
            print("\n3️⃣ Configuring SCADA master station...")
            for rtu_info in self.rtu_list:
                scada_master.add_rtu(
                    rtu_info['rtu_id'],
                    rtu_info['name'],
                    rtu_info['bus_number']
                )
            print(f"  ✓ SCADA master configured with {len(self.rtu_list)} RTUs")
            
            print("\n✅ SYSTEM INITIALIZATION COMPLETE")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def start_real_time_operation(self):
        """Start real-time SCADA operations"""
        try:
            print("\n🚀 STARTING REAL-TIME OPERATIONS")
            print("=" * 60)
            
            # Start all RTUs
            print("Starting RTU outstations...")
            rtu_manager.start_all()
            print(f"  ✓ Started {len(self.rtu_list)} RTUs")
            
            # Start SCADA master
            print("\nStarting SCADA master...")
            scada_master.start()
            print("  ✓ SCADA master started")
            
            self.running = True
            print(f"\n🌟 SYSTEM IS NOW OPERATIONAL")
            print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   RTUs Online: {len(self.rtu_list)}")
            print(f"   SCADA Status: ACTIVE")
            print(f"   DNP3 Channel: READY")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start real-time operations: {e}")
            return False
    
    def get_system_status(self):
        """Get current system status"""
        if not self.running:
            return {"status": "OFFLINE"}
            
        try:
            # Get SCADA status
            scada_status = scada_master.get_system_status()
            
            return {
                "status": "ONLINE",
                "timestamp": datetime.now().isoformat(),
                "active_rtus": scada_status.get('rtus_online', 0),
                "total_rtus": len(self.rtu_list),
                "recent_measurements": scada_status.get('total_measurements', 0),
                "scada_running": scada_master.is_running,
                "dnp3_attacks_active": dnp3_channel.attack_interceptor is not None
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def get_real_time_data(self):
        """Get real-time SCADA data for monitoring"""
        if not self.running:
            return []
            
        try:
            measurements_dict = scada_master.get_measurements()
            
            # Group measurements by RTU
            rtu_data = {}
            for key, measurement in measurements_dict.items():
                rtu_id = measurement.rtu_id
                if rtu_id not in rtu_data:
                    # Get RTU info
                    rtu_info = next((r for r in self.rtu_list if r['rtu_id'] == rtu_id), None)
                    rtu_data[rtu_id] = {
                        'rtu_id': rtu_id,
                        'bus_number': rtu_info['bus_number'] if rtu_info else 0,
                        'timestamp': measurement.timestamp,
                        'voltage_magnitude': 0.0,
                        'voltage_angle': 0.0,
                        'active_power': 0.0,
                        'reactive_power': 0.0,
                        'frequency': 50.0,
                        'status': measurement.quality
                    }
                
                # Populate measurement values based on point name
                if 'voltage_magnitude' in measurement.point_name:
                    rtu_data[rtu_id]['voltage_magnitude'] = measurement.value
                elif 'voltage_angle' in measurement.point_name:
                    rtu_data[rtu_id]['voltage_angle'] = measurement.value
                elif 'active_power' in measurement.point_name:
                    rtu_data[rtu_id]['active_power'] = measurement.value
                elif 'reactive_power' in measurement.point_name:
                    rtu_data[rtu_id]['reactive_power'] = measurement.value
                elif 'frequency' in measurement.point_name:
                    rtu_data[rtu_id]['frequency'] = measurement.value
            
            return list(rtu_data.values())
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return []
    
    def activate_attack_mode(self, voltage_offset=0.0, frequency_offset=0.0):
        """Activate DNP3 attack with specified parameters"""
        try:
            # Activate attack interceptor first
            dnp3_channel.activate_attack_interceptor()
            
            # Set attack parameters for all RTUs
            for rtu_info in self.rtu_list:
                rtu_id = rtu_info['rtu_id']
                dnp3_channel.set_attack_manipulation(
                    rtu_id=rtu_id,
                    voltage_offset=voltage_offset,
                    frequency_offset=frequency_offset
                )
            
            logger.info(f"Attack activated: voltage_offset={voltage_offset}, frequency_offset={frequency_offset}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate attack: {e}")
            return False
    
    def deactivate_attack_mode(self):
        """Deactivate DNP3 attack"""
        try:
            dnp3_channel.deactivate_attack_interceptor()
            logger.info("Attack deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate attack: {e}")
            return False
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        try:
            print("\n🛑 SHUTTING DOWN SYSTEM")
            print("=" * 60)
            
            self.running = False
            
            # Stop SCADA master
            scada_master.stop()
            print("  ✓ SCADA master stopped")
            
            # Stop all RTUs
            rtu_manager.stop_all()
            print("  ✓ All RTUs stopped")
            
            # Deactivate any attacks
            if dnp3_channel.attack_interceptor is not None:
                dnp3_channel.deactivate_attack_interceptor()
                print("  ✓ Attack interceptor deactivated")
            
            print("\n✅ SYSTEM SHUTDOWN COMPLETE")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global demo instance for UI access
demo_instance = None

def get_demo_instance():
    """Get the global demo instance"""
    return demo_instance

def main():
    """Main demonstration function"""
    global demo_instance
    
    print("🎯 REAL-TIME SCADA CYBERSECURITY DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Create demo instance
        demo_instance = CyberSecurityDemo()
        
        # Initialize system
        if not demo_instance.initialize_system():
            print("❌ System initialization failed")
            return
        
        # Start real-time operations
        if not demo_instance.start_real_time_operation():
            print("❌ Failed to start real-time operations")
            return
        
        print("\n📱 UI APPLICATIONS AVAILABLE:")
        print("  1. SCADA Monitor: streamlit run scada_monitor_ui.py")
        print("  2. Attack Interface: streamlit run attack_ui.py")
        print("\nPress Ctrl+C to shutdown...")
        
                # Keep running until interrupted
        try:
            while True:
                status = demo_instance.get_system_status()
                print(f"\r🔄 System Status: {status['status']} | RTUs: {status.get('active_rtus', 0)}/{status.get('total_rtus', 0)} | Attacks: {'ACTIVE' if status.get('dnp3_attacks_active', False) else 'INACTIVE'}", end="")
                
                # Get real-time measurement data and add it to the status
                real_time_data = demo_instance.get_real_time_data()
                status["real_time_data"] = real_time_data
                
                # Write status to file for UI access
                write_system_status(status)
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown requested by user")
    
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
    
    finally:
        if demo_instance:
            demo_instance.shutdown()

if __name__ == "__main__":
    main()
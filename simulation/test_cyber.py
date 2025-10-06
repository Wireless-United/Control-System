#!/usr/bin/env python3
"""
Quick test of the integrated cybersecurity simulation
"""

import sys
import os
import time
import logging

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_cybersecurity_simulation():
    """Test the integrated cybersecurity simulation"""
    print("🔒 TESTING INTEGRATED CYBERSECURITY SIMULATION")
    print("=" * 60)
    
    try:
        # Import components
        from ieee39_system_strict import StrictIEEE39BusSystem
        from integrated_scada import scada_master
        from integrated_rtu import rtu_manager
        from mock_dnp3 import dnp3_channel
        
        print("✓ All components imported successfully")
        
        # Initialize power system
        print("\n📊 Initializing IEEE 39-bus power system...")
        system = StrictIEEE39BusSystem()
        analysis = system.run_strict_ieee39_analysis()
        
        if analysis['pypower_analysis']:
            print("✓ Power system operational")
            state = system.get_system_state()
            print(f"  • Total Load: {state['total_load_mw']:.1f} MW")
            print(f"  • Total Generation: {state['total_generation_mw']:.1f} MW")
        else:
            print("❌ Power system failed to initialize")
            return False
        
        # Initialize SCADA-RTU system
        print("\n🖥️ Setting up SCADA-RTU system...")
        
        # Create RTUs
        rtu_manager.create_standard_rtus(system)
        rtu_list = rtu_manager.get_rtu_list()
        print(f"✓ Created {len(rtu_list)} RTU outstations")
        
        # Configure SCADA
        for rtu_info in rtu_list:
            scada_master.add_rtu(
                rtu_info['rtu_id'],
                rtu_info['name'], 
                rtu_info['bus_number']
            )
        print(f"✓ SCADA configured with {len(rtu_list)} RTUs")
        
        # Start systems
        print("\n🚀 Starting SCADA-RTU system...")
        rtu_manager.start_all()
        scada_master.start()
        print("✓ All systems started")
        
        # Test communication
        print("\n📡 Testing SCADA-RTU communication...")
        for i in range(5):
            time.sleep(2)
            status = scada_master.get_system_status()
            measurements = scada_master.get_measurements()
            print(f"  Poll {i+1}: {status['responses_received']} responses, {len(measurements)} measurements")
        
        # Show final status
        final_status = scada_master.get_system_status()
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY")
        print(f"  • RTUs Online: {final_status['rtus_online']}")
        print(f"  • Total Polls: {final_status['polls_sent']}")
        print(f"  • Successful Responses: {final_status['responses_received']}")
        print(f"  • Total Measurements: {final_status['total_measurements']}")
        
        # Test attack functionality
        print(f"\n🕷️ Testing attack capability...")
        
        def test_attack_interceptor(rtu_id, scada_id, data):
            """Test attack interceptor"""
            print(f"  🔴 Attack intercepted RTU {rtu_id} data: {len(data)} points")
            return data
        
        dnp3_channel.set_attack_interceptor(test_attack_interceptor)
        print("✓ Attack interceptor activated")
        
        # Test one more poll with attack active
        time.sleep(2)
        status_after_attack = scada_master.get_system_status()
        print(f"  • Poll with attack: {status_after_attack['responses_received'] - final_status['responses_received']} additional responses")
        
        # Clean up
        dnp3_channel.set_attack_interceptor(None)
        scada_master.stop()
        rtu_manager.stop_all()
        print("✓ System shutdown complete")
        
        print(f"\n🎯 CYBERSECURITY SIMULATION READY FOR USE")
        print(f"  Run 'python main_demo.py' and select option 2")
        print(f"  Then run 'streamlit run mitm/streamlit_attacker.py' for the attack UI")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cybersecurity_simulation()
    sys.exit(0 if success else 1)
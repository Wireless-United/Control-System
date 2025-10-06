#!/usr/bin/env python3
"""
Quick validation of the fixed system
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from main_cyber_demo import get_demo_instance
import time

def validate_system():
    """Validate that the system is working correctly"""
    print("🔍 VALIDATING SYSTEM...")
    
    demo = get_demo_instance()
    if not demo:
        print("❌ Demo instance not found - start main_cyber_demo.py first")
        return False
    
    # Test system status
    status = demo.get_system_status()
    print(f"✓ System Status: {status['status']}")
    print(f"✓ RTUs Online: {status['active_rtus']}/{status['total_rtus']}")
    print(f"✓ SCADA Running: {status['scada_running']}")
    print(f"✓ Attacks Active: {status['dnp3_attacks_active']}")
    
    # Test data retrieval
    data = demo.get_real_time_data()
    print(f"✓ Real-time Data Points: {len(data)}")
    
    if len(data) > 0:
        sample = data[0]
        print(f"✓ Sample RTU {sample['rtu_id']} - Bus {sample['bus_number']}: {sample['voltage_magnitude']:.3f} pu")
    
    # Test attack activation
    print("\n🎯 Testing attack capability...")
    success = demo.activate_attack_mode(voltage_offset=0.02, frequency_offset=0.5)
    print(f"✓ Attack activation: {'SUCCESS' if success else 'FAILED'}")
    
    time.sleep(3)
    
    status_attack = demo.get_system_status()
    print(f"✓ Attack Status: {'ACTIVE' if status_attack['dnp3_attacks_active'] else 'INACTIVE'}")
    
    # Deactivate attack
    success = demo.deactivate_attack_mode()
    print(f"✓ Attack deactivation: {'SUCCESS' if success else 'FAILED'}")
    
    print("\n✅ SYSTEM VALIDATION COMPLETE")
    return True

if __name__ == "__main__":
    validate_system()
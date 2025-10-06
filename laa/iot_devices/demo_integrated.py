#!/usr/bin/env python3
"""
IoT Devices Power Array Demo - Integrated Version

Demonstrates:
- Device deployment with N scaling factor
- Net power array tracking
- Pool pump integration  
- Updated power specifications (Thermostat=2kW, Water Heater=3kW, EV=6kW, Pool=3kW)

Author: Pranaav
Date: October 2025
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from laa.iot_devices.integration import IoTLAAIntegrator
from laa.iot_devices.device_controller import BotnetStrategy

# Mock IEEE 39 system
class MockIEEE39:
    def __init__(self):
        self.ieee_loads = {}

def main():
    print("\n" + "="*100)
    print("IOT DEVICES - INTEGRATED DEMO WITH NET POWER ARRAY")
    print("="*100)
    print("Module: laa/iot_devices/")
    print("Author: Pranaav")
    print("Date: October 2025")
    print("\nFeatures:")
    print("  - Device power specs: Thermostat=2kW, Water Heater=3kW, EV Charger=6kW, Pool Pump=3kW")
    print("  - N scaling factor (currently N=1)")
    print("  - Net power array tracking for all devices")
    print("  - Pool pump integration\n")
    
    # Initialize system
    print("Initializing IEEE 39-Bus System with IoT Devices...")
    ieee39 = MockIEEE39()
    integrator = IoTLAAIntegrator(ieee39)
    
    # Deploy devices with N=1 (can be changed to N=2, N=3, etc.)
    N = 1
    print(f"\nDeploying IoT devices with scaling factor N={N}...")
    device_count = integrator.deploy_iot_devices(N=N)
    
    print("\n" + "="*100)
    print("FEATURE 1: NET POWER ARRAY")
    print("="*100)
    integrator.print_net_power_array()
    
    print("\n" + "="*100)
    print("FEATURE 2: NET POWER SUMMARY")
    print("="*100)
    summary = integrator.get_net_power_summary()
    print(f"\nTotal Devices: {summary['device_count']}")
    print(f"Total Power: {summary['total_power_kw']:.2f} kW ({summary['total_power_mw']:.4f} MW)")
    print(f"Average Power per Device: {summary['avg_power_kw']:.2f} kW")
    
    print("\n" + "="*100)
    print("FEATURE 3: BUS POWER REPORT")
    print("="*100)
    integrator.print_bus_power_report(show_devices=True)
    
    print("\n" + "="*100)
    print("FEATURE 4: POWER ARRAYS FOR ANALYSIS")
    print("="*100)
    integrator.print_power_arrays()
    
    # Simulate attack scenario
    print("\n" + "="*100)
    print("FEATURE 5: ATTACK SCENARIO - NET POWER CHANGES")
    print("="*100)
    
    print("\nCompromising 70% of devices...")
    integrator.botnet_controller.compromise_devices(compromise_rate=0.7)
    
    print("Executing coordinated attack on buses [20, 21, 23, 24]...\n")
    integrator.botnet_controller.execute_coordinated_attack(
        target_buses=[20, 21, 23, 24],
        attack_magnitude_mw=50.0,
        strategy=BotnetStrategy.SIMULTANEOUS
    )
    
    # Show attack statistics
    stats = integrator.botnet_controller.get_statistics()
    print(f"Attack Statistics:")
    print(f"  Total Devices: {stats.total_devices}")
    print(f"  Compromised Devices: {stats.compromised_devices} "
          f"({stats.compromised_devices/stats.total_devices*100:.1f}%)")
    print(f"  Active Attack Devices: {stats.active_attack_devices}")
    print(f"  Total Attack Power: {stats.total_attack_power_mw:.2f} MW")
    
    # Show power changes after attack
    print("\n" + "="*100)
    print("NET POWER ARRAY AFTER ATTACK")
    print("="*100)
    integrator.print_net_power_array()
    
    print("\n" + "="*100)
    print("DEMONSTRATION COMPLETE")
    print("="*100)
    
    print("\nUsage Examples:")
    print("""
    # Get net power array
    net_power = integrator.get_net_power_summary()
    print(net_power['net_power_array'])
    
    # Deploy with different N scaling
    integrator.deploy_iot_devices(N=2)  # Double the devices
    
    # Print net power array
    integrator.print_net_power_array()
    
    # Get bus power data
    bus_data = integrator.get_bus_power_array()
    
    # Get numpy arrays for analysis
    arrays = integrator.get_power_arrays_for_analysis()
    """)
    
    print("\nAll features demonstrated!")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()

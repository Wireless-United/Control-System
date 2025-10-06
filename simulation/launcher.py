#!/usr/bin/env python3
"""
SCADA Cybersecurity Demo Launcher
Quick launcher for the complete cybersecurity demonstration
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 80)
    print("🎯 SCADA CYBERSECURITY DEMONSTRATION LAUNCHER")
    print("=" * 80)
    print()

def print_menu():
    """Print main menu"""
    print("📋 Available Options:")
    print("  1. 🚀 Start Main System (Required First)")
    print("  2. 📊 Launch SCADA Monitor UI")
    print("  3. 🎯 Launch Attack Interface")
    print("  4. 🔧 Run System Test")
    print("  5. 📚 Show Documentation")
    print("  0. ❌ Exit")
    print()

def start_main_system():
    """Start the main cybersecurity system"""
    print("🚀 Starting main cybersecurity system...")
    print("💡 This will run in the foreground. Use Ctrl+C to stop.")
    print("💡 Open new terminals for the UIs.")
    print()
    
    script_path = Path(__file__).parent / "main_cyber_demo.py"
    python_path = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
    
    try:
        subprocess.run([str(python_path), str(script_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except FileNotFoundError:
        print("❌ Python executable not found. Please check your virtual environment.")
    except Exception as e:
        print(f"❌ Failed to start system: {e}")

def launch_scada_monitor():
    """Launch SCADA monitoring UI"""
    print("📊 Launching SCADA Monitor UI...")
    print("💡 This will open in your default web browser")
    print()
    
    script_path = Path(__file__).parent / "scada_monitor_ui.py"
    
    try:
        subprocess.run([
            "streamlit", "run", str(script_path),
            "--server.port", "8501",
            "--server.headless", "true"
        ], check=True)
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it: pip install streamlit")
    except Exception as e:
        print(f"❌ Failed to launch SCADA monitor: {e}")

def launch_attack_interface():
    """Launch attack interface UI"""
    print("🎯 Launching Attack Interface...")
    print("💡 This will open in your default web browser")
    print()
    
    script_path = Path(__file__).parent / "attack_ui.py"
    
    try:
        subprocess.run([
            "streamlit", "run", str(script_path),
            "--server.port", "8502",
            "--server.headless", "true"
        ], check=True)
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it: pip install streamlit")
    except Exception as e:
        print(f"❌ Failed to launch attack interface: {e}")

def run_system_test():
    """Run the system test"""
    print("🔧 Running system test...")
    
    script_path = Path(__file__).parent / "test_cyber.py"
    python_path = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
    
    try:
        subprocess.run([str(python_path), str(script_path)], check=True)
        print("✅ System test completed successfully")
    except FileNotFoundError:
        print("❌ Python executable not found. Please check your virtual environment.")
    except Exception as e:
        print(f"❌ System test failed: {e}")

def show_documentation():
    """Show documentation and usage instructions"""
    print("📚 DOCUMENTATION")
    print("=" * 50)
    print()
    print("🎯 SCADA Cybersecurity Demonstration System")
    print()
    print("📋 Quick Start Guide:")
    print("  1. First, start the main system (Option 1)")
    print("  2. In separate terminals, launch the UIs:")
    print("     - SCADA Monitor (Option 2): http://localhost:8501")
    print("     - Attack Interface (Option 3): http://localhost:8502")
    print()
    print("🔧 System Components:")
    print("  • IEEE 39-bus power system simulation")
    print("  • 20 RTU outstations (10 generators, 10 loads)")
    print("  • SCADA master station")
    print("  • Mock DNP3 communication protocol")
    print("  • Attack interceptor for DNP3 manipulation")
    print()
    print("🎯 Attack Capabilities:")
    print("  • Voltage measurement manipulation")
    print("  • Frequency measurement manipulation")
    print("  • Real-time attack impact visualization")
    print()
    print("📊 Monitoring Features:")
    print("  • Real-time SCADA measurements")
    print("  • Voltage and power flow charts")
    print("  • System status indicators")
    print("  • Attack detection and alerts")
    print()
    print("💡 Tips:")
    print("  • Run system test first to verify setup")
    print("  • Keep main system running while using UIs")
    print("  • Use Ctrl+C to stop any running component")
    print("  • Check console output for detailed logs")
    print()

def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("🎯 Select option (0-5): ").strip()
            print()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                start_main_system()
            elif choice == "2":
                launch_scada_monitor()
            elif choice == "3":
                launch_attack_interface()
            elif choice == "4":
                run_system_test()
            elif choice == "5":
                show_documentation()
            else:
                print("❌ Invalid option. Please select 0-5.")
            
            print()
            input("Press Enter to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

if __name__ == "__main__":
    main()
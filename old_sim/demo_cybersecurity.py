#!/usr/bin/env python3
"""
Cybersecurity Demonstration Script for Control Systems
=====================================================

This script demonstrates various cybersecurity scenarios in industrial control systems
including SCADA simulation, MiTM attacks, and security analysis.

Author: Control Systems Security Lab
Purpose: Educational and research purposes only
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path


class SecurityDemo:
    """Main demonstration controller for cybersecurity scenarios."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        
    def print_banner(self):
        """Print the demonstration banner."""
        print("\n" + "="*80)
        print("🔒 INDUSTRIAL CONTROL SYSTEM CYBERSECURITY DEMONSTRATION")
        print("="*80)
        print("📋 Available Scenarios:")
        print("   1. Normal SCADA Operation")
        print("   2. SCADA with MiTM Attack")
        print("   3. Standalone MiTM Attack")
        print("   4. Dual Terminal Demo")
        print("   5. Security Analysis")
        print("="*80)
        
    def run_normal_scada(self, duration=30):
        """Run normal SCADA simulation."""
        print(f"\n🟢 Starting Normal SCADA Operation ({duration}s)")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "simulation.scada",
            f"--duration={duration}"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.base_dir, timeout=duration + 10)
        except subprocess.TimeoutExpired:
            print("⏰ Demo completed (timeout)")
        except KeyboardInterrupt:
            print("\n⛔ Demo stopped by user")
            
    def run_scada_with_attack(self, duration=60):
        """Run SCADA simulation with integrated MiTM attack."""
        print(f"\n🔴 Starting SCADA with MiTM Attack ({duration}s)")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "simulation.scada",
            "--enable-attack",
            f"--duration={duration}"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.base_dir, timeout=duration + 10)
        except subprocess.TimeoutExpired:
            print("⏰ Demo completed (timeout)")
        except KeyboardInterrupt:
            print("\n⛔ Demo stopped by user")
            
    def run_standalone_attack(self, attack_type="fci", duration=30):
        """Run standalone MiTM attack."""
        print(f"\n⚔️  Starting Standalone MiTM Attack ({attack_type}, {duration}s)")
        print("="*60)
        
        cmd = [
            sys.executable, "mitm/attacker.py",
            "--target", "scada",
            "--victim", "rtu", 
            "--attack", attack_type,
            "--duration", str(duration)
        ]
        
        try:
            subprocess.run(cmd, cwd=self.base_dir, timeout=duration + 10)
        except subprocess.TimeoutExpired:
            print("⏰ Attack completed (timeout)")
        except KeyboardInterrupt:
            print("\n⛔ Attack stopped by user")
            
    def print_dual_terminal_instructions(self):
        """Print instructions for dual terminal demonstration."""
        print("\n🖥️  DUAL TERMINAL DEMONSTRATION SETUP")
        print("="*60)
        print("📋 Instructions:")
        print("   1. Open TWO separate terminal windows")
        print("   2. Navigate both to the Control-System directory")
        print(f"      cd \"{self.base_dir}\"")
        print("\n🖥️  Terminal 1 (SCADA System):")
        print("      python -m simulation.scada --enable-attack")
        print("\n🖥️  Terminal 2 (MiTM Attacker):")
        print("      python mitm/attacker.py --target scada --victim rtu --attack fci")
        print("\n📋 Alternative attack commands for Terminal 2:")
        print("   🔸 False Data Injection:")
        print("      python mitm/attacker.py --attack fdi --duration 30")
        print("   🔸 Combined attacks:")
        print("      python mitm/attacker.py --attack fci fdi --scenario all")
        print("   🔸 Targeted scenario:")
        print("      python mitm/attacker.py --scenario breaker_trip_close")
        print("\n⚠️  Note: Start Terminal 1 first, then Terminal 2")
        print("="*60)
        
    def run_security_analysis(self):
        """Run security analysis on the available datasets."""
        print("\n🔍 SECURITY ANALYSIS")
        print("="*60)
        print("📊 Available Datasets:")
        
        datasets_dir = self.base_dir / "detection" / "Datasets"
        if datasets_dir.exists():
            # List adversary datasets
            adv_dir = datasets_dir / "Adversary"
            if adv_dir.exists():
                print(f"\n📁 Adversary Data ({len(list(adv_dir.glob('*.json')))} files):")
                for file in sorted(adv_dir.glob("*.json"))[:5]:  # Show first 5
                    print(f"   • {file.name}")
                if len(list(adv_dir.glob('*.json'))) > 5:
                    print(f"   ... and {len(list(adv_dir.glob('*.json'))) - 5} more files")
                    
            # List CSV datasets
            csv_dir = datasets_dir / "csvs"
            if csv_dir.exists():
                print(f"\n📁 CSV Data:")
                for uc_dir in sorted(csv_dir.iterdir()):
                    if uc_dir.is_dir():
                        csv_count = len(list(uc_dir.glob("*.csv")))
                        print(f"   • {uc_dir.name}: {csv_count} CSV files")
        else:
            print("❌ No datasets found in detection/Datasets/")
            
        print("\n💡 Analysis suggestions:")
        print("   • Use datasets for machine learning model training")
        print("   • Analyze attack patterns in the JSON files")
        print("   • Compare normal vs. adversarial behavior")
        print("   • Develop anomaly detection algorithms")
        
    def interactive_menu(self):
        """Run interactive menu for demonstration."""
        while True:
            self.print_banner()
            
            try:
                choice = input("\n🎯 Select scenario (1-5) or 'q' to quit: ").strip().lower()
                
                if choice == 'q' or choice == 'quit':
                    print("\n👋 Goodbye!")
                    break
                elif choice == '1':
                    duration = input("Duration in seconds (default 30): ").strip()
                    duration = int(duration) if duration.isdigit() else 30
                    self.run_normal_scada(duration)
                elif choice == '2':
                    duration = input("Duration in seconds (default 60): ").strip()
                    duration = int(duration) if duration.isdigit() else 60
                    self.run_scada_with_attack(duration)
                elif choice == '3':
                    attack_type = input("Attack type (fci/fdi/both) [default: fci]: ").strip()
                    if not attack_type:
                        attack_type = "fci"
                    elif attack_type == "both":
                        attack_type = "fci fdi"
                    duration = input("Duration in seconds (default 30): ").strip()
                    duration = int(duration) if duration.isdigit() else 30
                    self.run_standalone_attack(attack_type, duration)
                elif choice == '4':
                    self.print_dual_terminal_instructions()
                    input("\nPress Enter to continue...")
                elif choice == '5':
                    self.run_security_analysis()
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Invalid choice. Please select 1-5 or 'q'.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except ValueError:
                print("❌ Invalid input. Please try again.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Industrial Control System Cybersecurity Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_cybersecurity.py                    # Interactive menu
  python demo_cybersecurity.py --normal           # Normal SCADA demo
  python demo_cybersecurity.py --attack           # SCADA with MiTM
  python demo_cybersecurity.py --mitm fci         # Standalone MiTM
  python demo_cybersecurity.py --dual             # Dual terminal setup
  python demo_cybersecurity.py --analysis         # Security analysis
        """
    )
    
    parser.add_argument("--normal", action="store_true",
                        help="Run normal SCADA simulation")
    parser.add_argument("--attack", action="store_true", 
                        help="Run SCADA with MiTM attack")
    parser.add_argument("--mitm", nargs="*", default=None,
                        help="Run standalone MiTM attack (fci, fdi, or both)")
    parser.add_argument("--dual", action="store_true",
                        help="Show dual terminal setup instructions")
    parser.add_argument("--analysis", action="store_true",
                        help="Run security analysis")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration for timed demonstrations (default: 30s)")
    
    args = parser.parse_args()
    
    demo = SecurityDemo()
    
    # Handle command line arguments
    if args.normal:
        demo.run_normal_scada(args.duration)
    elif args.attack:
        demo.run_scada_with_attack(args.duration)
    elif args.mitm is not None:
        attack_type = " ".join(args.mitm) if args.mitm else "fci"
        demo.run_standalone_attack(attack_type, args.duration)
    elif args.dual:
        demo.print_dual_terminal_instructions()
    elif args.analysis:
        demo.run_security_analysis()
    else:
        # No arguments provided, run interactive menu
        demo.interactive_menu()


if __name__ == "__main__":
    main()

"""
Test script to verify the cyber-physical grid simulation setup.
Run this before running the main simulation.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'pandapower',
        'asyncio', 
        'pydantic',
        'pandas',
        'numpy'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\\nFailed to import: {', '.join(failed_imports)}")
        print("Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\\nAll packages imported successfully!")
        return True

def test_pandapower():
    """Test pandapower basic functionality."""
    try:
        import pandapower as pp
        import pandapower.networks as nw
        
        print("\\nTesting pandapower...")
        
        # Load IEEE 39-bus system
        net = nw.case39()
        print(f"✓ IEEE-39 system loaded: {len(net.bus)} buses, {len(net.gen)} generators")
        
        # Run power flow
        pp.runpp(net, verbose=False)
        print("✓ Power flow calculation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Pandapower test failed: {e}")
        return False

def test_components():
    """Test our custom components."""
    try:
        print("\\nTesting custom components...")
        
        # Add current directory to path for imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test component imports
        from components.avr import AVR
        from components.generator import Generator
        from components.load import Load
        from components.bus import Bus
        print("✓ Components imported")
        
        # Test endpoint imports
        from endpoints.sensor import Sensor
        from endpoints.controller import Controller
        from endpoints.actuator import Actuator
        print("✓ Endpoints imported")
        
        # Test protocol import
        from protocols.profinet import ProfinetProtocol
        print("✓ PROFINET protocol imported")
        
        # Quick component test
        avr = AVR(1, 1, 1.0)
        gen = Generator(1, 1, "Test Gen")
        load = Load(1, 1, "Test Load")
        bus = Bus(1, "Test Bus")
        print("✓ Component objects created")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    try:
        import asyncio
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from protocols.profinet import ProfinetProtocol
        
        print("\\nTesting async functionality...")
        
        # Test PROFINET protocol
        profinet = ProfinetProtocol("test_profinet")
        await profinet.start_network()
        await asyncio.sleep(0.1)
        await profinet.stop_network()
        print("✓ PROFINET protocol async operations")
        
        return True
        
    except Exception as e:
        print(f"✗ Async test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Cyber-Physical Grid Simulation - Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test pandapower
    if not test_pandapower():
        sys.exit(1)
    
    # Test components
    if not test_components():
        sys.exit(1)
    
    # Test async functionality
    if not await test_async_functionality():
        sys.exit(1)
    
    print("\\n" + "=" * 50)
    print("All tests passed! ✓")
    print("You can now run: python grid.py")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

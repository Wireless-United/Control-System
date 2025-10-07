#!/usr/bin/env python3
"""
Simplified launcher for the attack detection UI
Bypasses compatibility issues with Python 3.13
"""

import os
import sys
import subprocess
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """Launch the attack detection UI"""
    print("🛡️ SCADA Attack Detection System")
    print("=" * 50)
    
    # Get the project root directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Project directory: {project_dir}")
    
    # Run streamlit directly
    ui_script = os.path.join(project_dir, 'inference', 'detection_ui.py')
    
    if not os.path.exists(ui_script):
        print(f"❌ Error: UI script not found at {ui_script}")
        return 1
    
    try:
        print("\n🔄 Launching detection UI...")
        print(f"Use Ctrl+C to stop the UI when done.")
        
        # Launch streamlit
        result = subprocess.run(["streamlit", "run", ui_script])
        return result.returncode
    except FileNotFoundError:
        print("❌ Error: Streamlit not found.")
        print("Please install it with: pip install streamlit")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 UI stopped by user.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
SCADA Attack Detection System Launcher

This script provides a convenient way to launch different components of the
SCADA Attack Detection System, including data generation, model training,
and the detection UI.
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tensorflow",
        "streamlit",
        "joblib"
    ]
    
    print("📦 Checking required packages...")
    
    try:
        import pip
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package} is installed")
            except ImportError:
                print(f"  ✗ {package} is not installed. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✓ {package} has been installed")
    except Exception as e:
        logger.error(f"Error checking/installing dependencies: {e}")
        print("⚠️ Please manually install the required packages:")
        for package in required_packages:
            print(f"  - {package}")
        return False
    
    return True

def ensure_directories():
    """Ensure all required directories exist"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    directories = [
        os.path.join(base_dir, "training", "data"),
        os.path.join(base_dir, "training", "data", "processed"),
        os.path.join(base_dir, "training", "models"),
        os.path.join(base_dir, "inference")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
    
    return True

def run_data_generation():
    """Run data generation script"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "training", "data_generation.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Data generation script not found at {script_path}")
        return False
    
    print("\n🔄 Generating training data...")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running data generation script: {e}")
        return False

def run_data_preprocessing():
    """Run data preprocessing script"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "training", "data_preprocessing.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Data preprocessing script not found at {script_path}")
        return False
    
    print("\n🔄 Preprocessing training data...")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running data preprocessing script: {e}")
        return False

def run_model_training():
    """Run model training script"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "training", "model_training.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Model training script not found at {script_path}")
        return False
    
    print("\n🔄 Training detection models...")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running model training script: {e}")
        return False

def run_detection_ui():
    """Launch detection UI"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "inference", "detection_ui.py")
    
    # Check if compatibility fix exists
    fix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "inference", "compatibility_fix.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Detection UI script not found at {script_path}")
        return False
    
    # Pre-run the compatibility fix
    if os.path.exists(fix_path):
        try:
            # Run the compatibility fix
            print("\n� Applying Python compatibility fixes...")
            subprocess.run([sys.executable, fix_path], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running compatibility fix: {e}")
            # Continue anyway
    
    print("\n�🔄 Launching detection UI...")
    
    try:
        # Setup environment variables to fix readline issues
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        
        # Use streamlit to run the UI
        subprocess.run(["streamlit", "run", script_path], check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching detection UI: {e}")
        return False
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install using 'pip install streamlit'")
        return False

def run_full_pipeline():
    """Run the complete pipeline: data generation, preprocessing, and training"""
    success = True
    
    success = success and run_data_generation()
    success = success and run_data_preprocessing()
    success = success and run_model_training()
    
    return success

def main():
    """Main function"""
    print("🛡️ SCADA ATTACK DETECTION SYSTEM LAUNCHER")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="SCADA Attack Detection System Launcher")
    
    parser.add_argument(
        "--action", 
        choices=["generate", "preprocess", "train", "detect", "full"],
        default="detect",
        help="Action to perform: generate data, preprocess data, train models, " \
             "run detection UI, or full pipeline (default: detect)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not ensure_dependencies():
        print("❌ Dependencies check failed. Please install required packages manually.")
        return
    
    # Ensure directories exist
    ensure_directories()
    
    # Perform requested action
    if args.action == "generate":
        if run_data_generation():
            print("\n✅ Data generation completed successfully.")
        else:
            print("\n❌ Data generation failed.")
    
    elif args.action == "preprocess":
        if run_data_preprocessing():
            print("\n✅ Data preprocessing completed successfully.")
        else:
            print("\n❌ Data preprocessing failed.")
    
    elif args.action == "train":
        if run_model_training():
            print("\n✅ Model training completed successfully.")
        else:
            print("\n❌ Model training failed.")
    
    elif args.action == "detect":
        if run_detection_ui():
            print("\n✅ Detection UI closed.")
        else:
            print("\n❌ Detection UI failed to launch.")
    
    elif args.action == "full":
        print("\n🔄 Running full pipeline...")
        
        if run_full_pipeline():
            print("\n✅ Full pipeline completed successfully.")
            
            launch_ui = input("\nWould you like to launch the detection UI now? (y/n): ")
            if launch_ui.lower() == 'y':
                run_detection_ui()
        else:
            print("\n❌ Full pipeline failed.")
    
    print("\n🏁 Launcher finished.")

if __name__ == "__main__":
    main()
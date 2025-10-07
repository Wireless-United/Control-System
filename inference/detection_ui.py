#!/usr/bin/env python3
"""
SCADA Attack Detection UI

This module provides a Streamlit-based user interface for visualizing real-time 
attack detection results on the SCADA system.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import threading
import logging

# Add parent directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Try to import compatibility fix
try:
    import compatibility_fix
except ImportError:
    pass

# Import attack detector with error handling
try:
    from inference.attack_detector import AttackDetector
except ImportError as e:
    st.error(f"❌ Error importing AttackDetector: {e}")
    st.info("This might be due to a compatibility issue with Python 3.13 and TensorFlow.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SYSTEM_STATUS_FILE = os.path.join(parent_dir, "system_status.json")
REFRESH_INTERVAL = 1  # seconds
MAX_HISTORY = 100  # number of data points to display in graphs
ALERT_THRESHOLD = 0.7  # confidence threshold for alerts

# Global state
class AppState:
    def __init__(self):
        self.detector = None
        self.detection_history = []
        self.voltage_history = []
        self.frequency_history = []
        self.last_detection = None
        self.auto_refresh = True
        self.alert_sound = False
        self.detection_count = {
            "normal": 0,
            "attack": 0
        }
        self.attack_types = {}
        
state = AppState()

def initialize_detector():
    """Initialize the attack detector"""
    detector = AttackDetector()
    success = detector.load_model_and_transformers()
    
    if not success:
        st.error("❌ Failed to load attack detection model and transformers")
        st.info("Please ensure that you have trained a model and stored it in the correct location")
        return None
    
    return detector

def get_system_data():
    """Read the current system status data"""
    try:
        if not os.path.exists(SYSTEM_STATUS_FILE):
            return None
            
        with open(SYSTEM_STATUS_FILE, 'r') as f:
            status_data = json.load(f)
        
        return status_data
    except Exception as e:
        logger.error(f"Error reading system status: {e}")
        return None

def perform_detection():
    """Perform attack detection on current system data"""
    if state.detector is None:
        state.detector = initialize_detector()
        if state.detector is None:
            return None
    
    # Read system status and detect attacks
    result = state.detector.read_system_status(SYSTEM_STATUS_FILE)
    
    if result is not None:
        # Update detection count
        if result["is_attack"]:
            state.detection_count["attack"] += 1
            
            # Update attack types
            attack_type = result["attack_type"]
            if attack_type in state.attack_types:
                state.attack_types[attack_type] += 1
            else:
                state.attack_types[attack_type] = 1
        else:
            state.detection_count["normal"] += 1
        
        # Get the system data to extract measurements
        system_data = get_system_data()
        if system_data:
            # Extract voltage and frequency for trending
            voltage = None
            frequency = None
            
            if "measurements" in system_data:
                for rtu_id, rtu_data in system_data["measurements"].items():
                    for bus_id, bus_data in rtu_data.get("buses", {}).items():
                        voltage = bus_data.get("voltage_magnitude")
                        break
                    
                    frequency = rtu_data.get("frequency")
                    break
            
            # Add to history with timestamp
            timestamp = datetime.now()
            state.detection_history.append({
                "timestamp": timestamp,
                "is_attack": result["is_attack"],
                "attack_type": result["attack_type"],
                "confidence": result["confidence"],
                "probability": result["probability"]
            })
            
            if voltage is not None:
                state.voltage_history.append({
                    "timestamp": timestamp,
                    "value": voltage,
                    "is_attack": result["is_attack"]
                })
            
            if frequency is not None:
                state.frequency_history.append({
                    "timestamp": timestamp,
                    "value": frequency,
                    "is_attack": result["is_attack"]
                })
            
            # Limit history size
            if len(state.detection_history) > MAX_HISTORY:
                state.detection_history.pop(0)
                
            if len(state.voltage_history) > MAX_HISTORY:
                state.voltage_history.pop(0)
                
            if len(state.frequency_history) > MAX_HISTORY:
                state.frequency_history.pop(0)
        
        state.last_detection = result
        return result
    
    return None

def auto_refresh_data():
    """Background thread for auto-refreshing data"""
    while state.auto_refresh:
        perform_detection()
        time.sleep(REFRESH_INTERVAL)

def create_ui():
    """Create the Streamlit UI"""
    st.set_page_config(
        page_title="SCADA Attack Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.title("🛡️ SCADA Attack Detection")
    st.sidebar.markdown("---")
    
    # Controls
    st.sidebar.subheader("Controls")
    state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    state.alert_sound = st.sidebar.checkbox("Alert Sound", value=False)
    
    if st.sidebar.button("Manual Refresh"):
        perform_detection()
    
    # Model settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")
    
    if state.detector is not None:
        current_threshold = state.detector.threshold
        new_threshold = st.sidebar.slider(
            "Detection Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=current_threshold, 
            step=0.05
        )
        
        if new_threshold != current_threshold:
            state.detector.update_threshold(new_threshold)
    
    # System information
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Information")
    
    if state.detector is not None:
        stats = state.detector.get_performance_stats()
        st.sidebar.info(f"""
        **Inference Performance:**
        - Avg: {stats['avg_inference_time_ms']:.2f} ms
        - Min: {stats['min_inference_time_ms']:.2f} ms
        - Max: {stats['max_inference_time_ms']:.2f} ms
        """)
    
    # Main area
    st.title("SCADA Attack Detection Dashboard")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Current Status")
        if state.last_detection is None:
            st.info("Waiting for data...")
        elif state.last_detection["is_attack"]:
            st.error("⚠️ **ATTACK DETECTED**")
            
            # Play alert sound if enabled
            if state.alert_sound and state.last_detection["confidence"] > ALERT_THRESHOLD:
                st.audio("https://www.soundjay.com/buttons/sounds/beep-09.mp3", autoplay=True)
        else:
            st.success("✅ System Normal")
    
    with col2:
        st.subheader("Detection Details")
        if state.last_detection is not None:
            confidence = state.last_detection["confidence"]
            confidence_color = "red" if confidence > 0.7 else "orange" if confidence > 0.5 else "green"
            
            st.markdown(f"""
            - **Attack Type:** {state.last_detection["attack_type"]}
            - **Confidence:** <span style='color:{confidence_color}'>{confidence:.2f}</span>
            - **Time:** {state.last_detection["timestamp"]}
            """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("Detection Summary")
        total = state.detection_count["normal"] + state.detection_count["attack"]
        
        if total > 0:
            attack_percentage = (state.detection_count["attack"] / total) * 100
            
            st.markdown(f"""
            - **Total Detections:** {total}
            - **Attacks Detected:** {state.detection_count["attack"]} ({attack_percentage:.1f}%)
            - **Normal Operations:** {state.detection_count["normal"]} ({100-attack_percentage:.1f}%)
            """)
    
    # Measurement trends
    st.markdown("---")
    st.subheader("Measurement Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if state.voltage_history:
            df_voltage = pd.DataFrame(state.voltage_history)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot normal points in blue
            normal_points = df_voltage[~df_voltage["is_attack"]]
            ax.scatter(
                range(len(normal_points)), 
                normal_points["value"], 
                color="blue", 
                label="Normal"
            )
            
            # Plot attack points in red
            attack_points = df_voltage[df_voltage["is_attack"]]
            ax.scatter(
                range(len(attack_points)), 
                attack_points["value"], 
                color="red", 
                label="Attack"
            )
            
            # Plot the line
            ax.plot(df_voltage["value"], color="gray", alpha=0.5)
            
            # Add reference lines
            ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
            ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5)
            ax.axhline(y=1.05, color="orange", linestyle="--", alpha=0.5)
            
            ax.set_title("Voltage Magnitude Trend")
            ax.set_ylabel("Voltage (p.u.)")
            ax.set_xlabel("Time")
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.info("No voltage data available yet")
    
    with col2:
        if state.frequency_history:
            df_frequency = pd.DataFrame(state.frequency_history)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot normal points in blue
            normal_points = df_frequency[~df_frequency["is_attack"]]
            ax.scatter(
                range(len(normal_points)), 
                normal_points["value"], 
                color="blue", 
                label="Normal"
            )
            
            # Plot attack points in red
            attack_points = df_frequency[df_frequency["is_attack"]]
            ax.scatter(
                range(len(attack_points)), 
                attack_points["value"], 
                color="red", 
                label="Attack"
            )
            
            # Plot the line
            ax.plot(df_frequency["value"], color="gray", alpha=0.5)
            
            # Add reference lines
            ax.axhline(y=50.0, color="green", linestyle="--", alpha=0.5)
            ax.axhline(y=49.5, color="orange", linestyle="--", alpha=0.5)
            ax.axhline(y=50.5, color="orange", linestyle="--", alpha=0.5)
            
            ax.set_title("Frequency Trend")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time")
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.info("No frequency data available yet")
    
    # Attack type distribution
    st.markdown("---")
    st.subheader("Attack Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if state.attack_types:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            attack_types = list(state.attack_types.keys())
            attack_counts = list(state.attack_types.values())
            
            ax.bar(attack_types, attack_counts, color="crimson")
            ax.set_title("Attack Type Distribution")
            ax.set_ylabel("Count")
            ax.set_xlabel("Attack Type")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y')
            
            st.pyplot(fig)
        else:
            st.info("No attacks detected yet")
    
    with col2:
        if state.detection_history:
            df_detection = pd.DataFrame(state.detection_history)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(df_detection["confidence"], color="purple")
            ax.set_title("Detection Confidence Trend")
            ax.set_ylabel("Confidence")
            ax.set_xlabel("Time")
            ax.grid(True)
            
            # Add reference line for the alert threshold
            ax.axhline(y=ALERT_THRESHOLD, color="red", linestyle="--", alpha=0.5, 
                      label=f"Alert Threshold ({ALERT_THRESHOLD})")
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.info("No detection history available yet")
    
    # Recent detections table
    st.markdown("---")
    st.subheader("Recent Detections")
    
    if state.detection_history:
        df = pd.DataFrame(state.detection_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", ascending=False).head(10)
        
        # Format the table
        df = df[["timestamp", "is_attack", "attack_type", "confidence"]]
        df.columns = ["Timestamp", "Attack", "Type", "Confidence"]
        
        # Add styling
        st.dataframe(df, width='stretch')
    else:
        st.info("No detection history available yet")

def main():
    """Main function"""
    # Start the detector
    state.detector = initialize_detector()
    
    # Create UI
    create_ui()
    
    # Start auto-refresh thread if enabled
    if state.auto_refresh:
        refresh_thread = threading.Thread(target=auto_refresh_data)
        refresh_thread.daemon = True
        refresh_thread.start()

if __name__ == "__main__":
    main()
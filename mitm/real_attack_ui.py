#!/usr/bin/env python3
"""
Real-Time SCADA-RTU Attack Interface
Connects to actual running cybersecurity simulation system
"""

import streamlit as st
import sys
import os
import time
import threading
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Add simulation directory to path
simulation_dir = Path(__file__).parent.parent / "simulation"
sys.path.append(str(simulation_dir))

try:
    from ieee39_system_strict import StrictIEEE39BusSystem
    from integrated_scada import SCADAMaster
    from integrated_rtu import IntegratedRTU, RTUConfig
    from mock_dnp3 import MockDNP3Channel, DNP3Point, DNP3Quality
except ImportError as e:
    st.error(f"❌ Failed to import simulation components: {e}")
    st.info("Please ensure you're running this from the correct directory with the simulation components available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Real-Time SCADA Attack Console",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #ff4444;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .attack-panel {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ff4444;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(255, 68, 68, 0.3);
    }
    
    .status-active {
        color: #ff4444;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    
    .status-online {
        color: #00ff00;
        font-weight: bold;
    }
    
    .status-offline {
        color: #999999;
        font-weight: bold;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.4; }
    }
    
    .console-output {
        background: #000000;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #333;
        height: 300px;
        overflow-y: scroll;
        font-size: 12px;
    }
    
    .measurement-row {
        padding: 0.3rem;
        border-bottom: 1px solid #333;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for the simulation system
if 'sim_system' not in st.session_state:
    st.session_state.sim_system = None
    st.session_state.sim_scada = None
    st.session_state.sim_rtus = []
    st.session_state.sim_dnp3 = None
    st.session_state.system_running = False
    st.session_state.attack_active = False
    st.session_state.attack_log = []
    st.session_state.last_measurements = {}
    st.session_state.attack_interceptor_active = False

def log_message(message):
    """Add message to attack log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.attack_log.append(f"[{timestamp}] {message}")
    if len(st.session_state.attack_log) > 100:
        st.session_state.attack_log.pop(0)

def initialize_real_system():
    """Initialize the actual SCADA-RTU system"""
    try:
        log_message("🔌 Initializing IEEE 39-Bus Power System...")
        st.session_state.sim_system = StrictIEEE39BusSystem()
        st.session_state.sim_system.solve_power_flow()
        
        log_message("🔗 Creating mock DNP3 channel...")
        st.session_state.sim_dnp3 = MockDNP3Channel()
        
        log_message("🖥️ Initializing SCADA master...")
        st.session_state.sim_scada = SCADAMaster(
            st.session_state.sim_dnp3, 
            st.session_state.sim_system
        )
        
        log_message("📡 Creating RTU outstations...")
        # Create RTUs for key buses
        generator_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        load_buses = [3, 4, 7, 8, 15, 16, 18, 20, 21, 23]
        
        rtu_id = 1
        st.session_state.sim_rtus = []
        
        for bus in generator_buses:
            config = RTUConfig(
                rtu_id=rtu_id,
                bus_number=bus,
                station_name=f"Gen_{bus}_RTU"
            )
            rtu = IntegratedRTU(config, st.session_state.sim_system, st.session_state.sim_dnp3)
            st.session_state.sim_rtus.append(rtu)
            rtu_id += 1
        
        for bus in load_buses:
            config = RTUConfig(
                rtu_id=rtu_id,
                bus_number=bus,
                station_name=f"Load_{bus}_RTU"
            )
            rtu = IntegratedRTU(config, st.session_state.sim_system, st.session_state.sim_dnp3)
            st.session_state.sim_rtus.append(rtu)
            rtu_id += 1
        
        log_message(f"✅ Created {len(st.session_state.sim_rtus)} RTUs")
        
        log_message("🚀 Starting SCADA system...")
        st.session_state.sim_scada.start_polling()
        
        # Let system stabilize
        time.sleep(3)
        
        st.session_state.system_running = True
        log_message("✅ Real SCADA-RTU system operational!")
        
        return True
        
    except Exception as e:
        log_message(f"❌ System initialization failed: {e}")
        return False

def stop_real_system():
    """Stop the actual SCADA-RTU system"""
    try:
        if st.session_state.sim_scada:
            st.session_state.sim_scada.stop_polling()
            log_message("🛑 SCADA polling stopped")
        
        for rtu in st.session_state.sim_rtus:
            rtu.stop()
        
        st.session_state.system_running = False
        st.session_state.attack_active = False
        st.session_state.attack_interceptor_active = False
        log_message("✅ System shutdown complete")
        
    except Exception as e:
        log_message(f"❌ Shutdown error: {e}")

def activate_attack_interceptor():
    """Activate the DNP3 attack interceptor"""
    if st.session_state.sim_dnp3 and not st.session_state.attack_interceptor_active:
        st.session_state.sim_dnp3.activate_attack_interceptor()
        st.session_state.attack_interceptor_active = True
        st.session_state.attack_active = True
        log_message("🔴 ATTACK INTERCEPTOR ACTIVATED")
        log_message("🕷️ MiTM position established on DNP3 channel")

def deactivate_attack_interceptor():
    """Deactivate the DNP3 attack interceptor"""
    if st.session_state.sim_dnp3 and st.session_state.attack_interceptor_active:
        st.session_state.sim_dnp3.deactivate_attack_interceptor()
        st.session_state.attack_interceptor_active = False
        st.session_state.attack_active = False
        log_message("🟢 Attack interceptor deactivated")
        log_message("✅ Normal DNP3 communication restored")

def manipulate_rtu_data(rtu_id, voltage_offset=0.0, frequency_offset=0.0):
    """Manipulate specific RTU data through DNP3 interception"""
    if not st.session_state.attack_interceptor_active:
        log_message("❌ Attack interceptor not active!")
        return
    
    if not st.session_state.sim_dnp3:
        log_message("❌ DNP3 channel not available!")
        return
    
    log_message(f"🎯 Targeting RTU {rtu_id} for data manipulation")
    if voltage_offset != 0:
        log_message(f"⚡ Injecting voltage offset: {voltage_offset:+.3f} pu")
    if frequency_offset != 0:
        log_message(f"📊 Injecting frequency offset: {frequency_offset:+.2f} Hz")
    
    # Set the manipulation parameters in the DNP3 channel
    st.session_state.sim_dnp3.set_attack_manipulation(
        rtu_id=rtu_id,
        voltage_offset=voltage_offset,
        frequency_offset=frequency_offset
    )
    
    log_message(f"✅ Attack parameters applied to RTU {rtu_id}")

def get_real_measurements():
    """Get actual measurements from the running SCADA system"""
    if not st.session_state.system_running or not st.session_state.sim_scada:
        return {}
    
    try:
        # Get recent measurements from SCADA
        recent_measurements = st.session_state.sim_scada.get_recent_measurements(limit=50)
        
        measurements_dict = {}
        for measurement in recent_measurements:
            key = f"RTU_{measurement['rtu_id']}_{measurement['point_type']}"
            measurements_dict[key] = {
                'rtu_id': measurement['rtu_id'],
                'point_type': measurement['point_type'],
                'value': measurement['value'],
                'unit': measurement.get('unit', ''),
                'timestamp': measurement.get('timestamp', time.time())
            }
        
        return measurements_dict
        
    except Exception as e:
        log_message(f"❌ Error getting measurements: {e}")
        return {}

def get_voltage_data_for_plot():
    """Get voltage data for real-time plotting"""
    if not st.session_state.system_running or not st.session_state.sim_system:
        return [], []
    
    try:
        # Get actual voltage data from power system
        bus_voltages = st.session_state.sim_system.bus_voltages
        bus_numbers = list(range(1, len(bus_voltages) + 1))
        return bus_numbers, bus_voltages.tolist()
    except:
        return [], []

def main():
    # Header
    st.markdown('<h1 class="main-header">🕷️ REAL-TIME SCADA ATTACK CONSOLE 🕷️</h1>', 
                unsafe_allow_html=True)
    st.markdown("*Connected to live IEEE 39-Bus SCADA-RTU system*")
    st.markdown("---")
    
    # Sidebar - System Control
    with st.sidebar:
        st.header("🔧 System Control")
        
        if not st.session_state.system_running:
            if st.button("🚀 Initialize Real System", type="primary", use_container_width=True):
                with st.spinner("Initializing SCADA-RTU system..."):
                    if initialize_real_system():
                        st.success("✅ System Online!")
                        st.rerun()
                    else:
                        st.error("❌ Initialization Failed!")
        else:
            st.success("✅ System Online")
            
            if st.button("🛑 Stop System", use_container_width=True):
                stop_real_system()
                st.rerun()
        
        st.markdown("---")
        
        # Attack Control
        st.header("🕷️ Attack Control")
        
        if st.session_state.system_running:
            if not st.session_state.attack_interceptor_active:
                if st.button("🔴 ACTIVATE ATTACK", type="primary", use_container_width=True):
                    activate_attack_interceptor()
                    st.rerun()
            else:
                st.markdown('<div class="status-active">⚡ ATTACK ACTIVE ⚡</div>', 
                           unsafe_allow_html=True)
                
                if st.button("🛑 STOP ATTACK", use_container_width=True):
                    deactivate_attack_interceptor()
                    st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.header("📊 System Status")
        if st.session_state.system_running and st.session_state.sim_scada:
            try:
                status = st.session_state.sim_scada.get_system_status()
                st.metric("RTUs Online", f"{len([r for r in st.session_state.sim_rtus if r.is_running])}/20")
                st.metric("Successful Polls", status.get('successful_polls', 0))
                st.metric("Total Measurements", status.get('total_measurements', 0))
                st.metric("Active Alarms", status.get('total_alarms', 0))
            except:
                st.metric("RTUs Online", "N/A")
                st.metric("Successful Polls", "N/A")
                st.metric("Total Measurements", "N/A")
                st.metric("Active Alarms", "N/A")
        else:
            st.metric("System", "OFFLINE")
    
    if not st.session_state.system_running:
        st.info("👆 Please initialize the real SCADA-RTU system using the sidebar to begin attacks.")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Attack Configuration Panel
        st.markdown('<div class="attack-panel">', unsafe_allow_html=True)
        st.subheader("🎯 Real-Time Attack Configuration")
        
        # RTU selection
        available_rtus = [f"RTU {i}" for i in range(1, 21)]
        target_rtu = st.selectbox("Target RTU", ["All RTUs"] + available_rtus)
        
        # Attack parameters
        st.subheader("⚙️ Data Manipulation Parameters")
        voltage_offset = st.slider("Voltage Offset (pu)", -0.5, 0.5, 0.0, 0.01)
        frequency_offset = st.slider("Frequency Offset (Hz)", -2.0, 2.0, 0.0, 0.1)
        
        # Execute attack
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚡ Apply Manipulation", use_container_width=True):
                if st.session_state.attack_interceptor_active:
                    if target_rtu == "All RTUs":
                        for i in range(1, 21):
                            manipulate_rtu_data(i, voltage_offset, frequency_offset)
                    else:
                        rtu_id = int(target_rtu.split()[1])
                        manipulate_rtu_data(rtu_id, voltage_offset, frequency_offset)
                else:
                    st.error("❌ Attack interceptor not active!")
        
        with col_b:
            if st.button("🔄 Reset Values", use_container_width=True):
                log_message("🔄 Resetting manipulation parameters")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time measurements display
        st.subheader("📊 Live SCADA Measurements")
        
        measurements = get_real_measurements()
        if measurements:
            # Create DataFrame for display
            measurement_data = []
            for key, data in measurements.items():
                measurement_data.append({
                    'RTU': f"RTU {data['rtu_id']}",
                    'Measurement': data['point_type'],
                    'Value': f"{data['value']:.4f}",
                    'Unit': data['unit'],
                    'Time': datetime.fromtimestamp(data['timestamp']).strftime("%H:%M:%S")
                })
            
            if measurement_data:
                df = pd.DataFrame(measurement_data)
                st.dataframe(df, use_container_width=True, height=300)
        else:
            st.info("No measurements available yet. System may be starting up...")
        
        # Real-time voltage plot
        st.subheader("📈 Live Voltage Profile")
        bus_numbers, voltages = get_voltage_data_for_plot()
        
        if bus_numbers and voltages:
            fig = go.Figure()
            
            # Color based on attack status
            line_color = '#ff4444' if st.session_state.attack_active else '#00ff00'
            
            fig.add_trace(go.Scatter(
                x=bus_numbers,
                y=voltages,
                mode='lines+markers',
                name='Bus Voltages',
                line=dict(color=line_color, width=2),
                marker=dict(size=4)
            ))
            
            # Add voltage limits
            fig.add_hline(y=1.05, line_dash="dash", line_color="red", 
                         annotation_text="Upper Limit (1.05 pu)")
            fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                         annotation_text="Lower Limit (0.95 pu)")
            
            attack_status = " (UNDER ATTACK)" if st.session_state.attack_active else " (NORMAL)"
            fig.update_layout(
                title=f"IEEE 39-Bus Real-Time Voltage Profile{attack_status}",
                xaxis_title="Bus Number",
                yaxis_title="Voltage (pu)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for voltage data...")
    
    with col2:
        # Attack Console
        st.subheader("🖥️ Attack Console")
        
        # Console output
        console_html = '<div class="console-output">'
        for log_entry in st.session_state.attack_log[-25:]:  # Show last 25 entries
            console_html += f"{log_entry}<br>"
        console_html += '</div>'
        
        st.markdown(console_html, unsafe_allow_html=True)
        
        # Quick attack buttons
        st.subheader("⚡ Quick Attacks")
        
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("🔥 Voltage Spike", use_container_width=True):
                if st.session_state.attack_interceptor_active:
                    manipulate_rtu_data(1, voltage_offset=0.2)
                else:
                    st.error("Activate attack first!")
            
            if st.button("📉 Voltage Drop", use_container_width=True):
                if st.session_state.attack_interceptor_active:
                    manipulate_rtu_data(5, voltage_offset=-0.15)
                else:
                    st.error("Activate attack first!")
        
        with quick_col2:
            if st.button("🌊 Freq Drift", use_container_width=True):
                if st.session_state.attack_interceptor_active:
                    manipulate_rtu_data(3, frequency_offset=1.5)
                else:
                    st.error("Activate attack first!")
            
            if st.button("💥 All RTUs", use_container_width=True):
                if st.session_state.attack_interceptor_active:
                    for i in range(1, 6):  # Attack first 5 RTUs
                        manipulate_rtu_data(i, voltage_offset=0.1)
                else:
                    st.error("Activate attack first!")
        
        # System health
        st.subheader("🩺 System Health")
        if st.session_state.attack_active:
            st.error("🚨 UNDER ATTACK")
            st.metric("DNP3 Integrity", "COMPROMISED")
        else:
            st.success("✅ System Secure")
            st.metric("DNP3 Integrity", "INTACT")
    
    # Auto-refresh when system is running
    if st.session_state.system_running:
        time.sleep(3)  # Refresh every 3 seconds
        st.rerun()

if __name__ == "__main__":
    main()
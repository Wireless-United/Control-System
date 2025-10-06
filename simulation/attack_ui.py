#!/usr/bin/env python3
"""
Real-time SCADA Attack Interface
Streamlit application for conducting DNP3 attacks on live SCADA system
"""

import streamlit as st
import time
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# Add simulation directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system status access
from system_status import read_system_status, is_system_running
from integrated_scada import scada_master
from mock_dnp3 import dnp3_channel
from mock_dnp3 import dnp3_channel

# Page configuration
st.set_page_config(
    page_title="SCADA Attack Interface",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_attack_system():
    """Get attack system interface"""
    if not is_system_running():
        return None, "System not running. Please start main_cyber_demo.py first."
    
    try:
        # Return a simple interface object that provides the needed methods
        class AttackInterface:
            def get_system_status(self):
                return read_system_status()
            
            def activate_attack_mode(self, voltage_offset=0.0, frequency_offset=0.0):
                try:
                    # Activate attack interceptor first
                    dnp3_channel.activate_attack_interceptor()
                    
                    # Set attack parameters for all RTUs (1-20)
                    for rtu_id in range(1, 21):
                        dnp3_channel.set_attack_manipulation(
                            rtu_id=rtu_id,
                            voltage_offset=voltage_offset,
                            frequency_offset=frequency_offset
                        )
                    return True
                except Exception as e:
                    print(f"Attack activation failed: {e}")
                    return False
            
            def deactivate_attack_mode(self):
                try:
                    dnp3_channel.deactivate_attack_interceptor()
                    return True
                except Exception as e:
                    print(f"Attack deactivation failed: {e}")
                    return False
            
            def get_real_time_data(self):
                try:
                    measurements_dict = scada_master.get_measurements()
                    
                    # Format measurements for UI
                    rtu_data = {}
                    for key, measurement in measurements_dict.items():
                        rtu_id = measurement.rtu_id
                        if rtu_id not in rtu_data:
                            rtu_data[rtu_id] = {
                                'rtu_id': rtu_id,
                                'bus_number': rtu_id + 29 if rtu_id <= 10 else [3, 4, 7, 8, 15, 16, 18, 20, 21, 23][rtu_id - 11],
                                'timestamp': measurement.timestamp,
                                'voltage_magnitude': 0.0,
                                'voltage_angle': 0.0,
                                'active_power': 0.0,
                                'reactive_power': 0.0,
                                'frequency': 50.0,
                                'status': measurement.quality
                            }
                        
                        # Populate measurement values
                        if 'voltage_magnitude' in measurement.point_name:
                            rtu_data[rtu_id]['voltage_magnitude'] = measurement.value
                        elif 'voltage_angle' in measurement.point_name:
                            rtu_data[rtu_id]['voltage_angle'] = measurement.value
                        elif 'active_power' in measurement.point_name:
                            rtu_data[rtu_id]['active_power'] = measurement.value
                        elif 'reactive_power' in measurement.point_name:
                            rtu_data[rtu_id]['reactive_power'] = measurement.value
                        elif 'frequency' in measurement.point_name:
                            rtu_data[rtu_id]['frequency'] = measurement.value
                    
                    return list(rtu_data.values())
                except Exception as e:
                    print(f"Error getting real-time data: {e}")
                    return []
        
        return AttackInterface(), None
    except Exception as e:
        return None, f"Error connecting to system: {e}"

def main():
    """Main attack interface application"""
    
    # Header with warning
    st.title("🎯 SCADA Attack Interface")
    st.error("⚠️ **WARNING: This is a cybersecurity research tool. Use only in authorized environments.**")
    st.markdown("---")
    
    # Get system interface
    attack_system, error = get_attack_system()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please start the main system first: `python main_cyber_demo.py`")
        return
    
    if not attack_system:
        st.warning("⚠️ No attack system available")
        return
    
    # Get current system status
    try:
        status = attack_system.get_system_status()
    except Exception as e:
        st.error(f"Failed to get system status: {e}")
        return
    
    # System status display
    st.subheader("🖥️ Target System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        system_status = status.get("status", "UNKNOWN")
        if system_status == "ONLINE":
            st.success(f"System: {system_status}")
        else:
            st.error(f"System: {system_status}")
    
    with col2:
        active_rtus = status.get("active_rtus", 0)
        total_rtus = status.get("total_rtus", 0)
        st.info(f"RTUs: {active_rtus}/{total_rtus}")
    
    with col3:
        attack_active = status.get("dnp3_attacks_active", False)
        if attack_active:
            st.error("Attacks: ACTIVE")
        else:
            st.success("Attacks: INACTIVE")
    
    with col4:
        scada_running = status.get("scada_running", False)
        if scada_running:
            st.success("SCADA: RUNNING")
        else:
            st.error("SCADA: STOPPED")
    
    st.markdown("---")
    
    # Attack configuration
    st.subheader("⚙️ Attack Configuration")
    
    # Only allow attacks if system is online
    if status.get("status") != "ONLINE":
        st.error("❌ Cannot launch attacks - target system is not online")
        return
    
    # Attack tabs
    tab1, tab2, tab3 = st.tabs(["🔋 Voltage Attack", "⚡ Frequency Attack", "🎛️ Combined Attack"])
    
    with tab1:
        st.markdown("### Voltage Manipulation Attack")
        st.info("Manipulate voltage measurements seen by SCADA operator")
        
        voltage_offset = st.slider(
            "Voltage Offset (pu)", 
            min_value=-0.2, 
            max_value=0.2, 
            value=0.0, 
            step=0.01,
            help="Positive values increase apparent voltage, negative values decrease it"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Launch Voltage Attack", type="primary"):
                try:
                    success = attack_system.activate_attack_mode(
                        voltage_offset=voltage_offset,
                        frequency_offset=0.0
                    )
                    if success:
                        st.success(f"✅ Voltage attack launched! Offset: {voltage_offset:+.3f} pu")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to launch voltage attack")
                except Exception as e:
                    st.error(f"❌ Attack failed: {e}")
        
        with col2:
            if st.button("🛑 Stop Voltage Attack"):
                try:
                    success = attack_system.deactivate_attack_mode()
                    if success:
                        st.success("✅ Voltage attack stopped")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop attack")
                except Exception as e:
                    st.error(f"❌ Stop failed: {e}")
    
    with tab2:
        st.markdown("### Frequency Manipulation Attack")
        st.info("Manipulate frequency measurements seen by SCADA operator")
        
        frequency_offset = st.slider(
            "Frequency Offset (Hz)", 
            min_value=-2.0, 
            max_value=2.0, 
            value=0.0, 
            step=0.1,
            help="Positive values increase apparent frequency, negative values decrease it"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Launch Frequency Attack", type="primary"):
                try:
                    success = attack_system.activate_attack_mode(
                        voltage_offset=0.0,
                        frequency_offset=frequency_offset
                    )
                    if success:
                        st.success(f"✅ Frequency attack launched! Offset: {frequency_offset:+.1f} Hz")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to launch frequency attack")
                except Exception as e:
                    st.error(f"❌ Attack failed: {e}")
        
        with col2:
            if st.button("🛑 Stop Frequency Attack"):
                try:
                    success = attack_system.deactivate_attack_mode()
                    if success:
                        st.success("✅ Frequency attack stopped")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop attack")
                except Exception as e:
                    st.error(f"❌ Stop failed: {e}")
    
    with tab3:
        st.markdown("### Combined Attack")
        st.info("Simultaneously manipulate voltage and frequency measurements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            combined_voltage_offset = st.slider(
                "Voltage Offset (pu)", 
                min_value=-0.2, 
                max_value=0.2, 
                value=0.0, 
                step=0.01,
                key="combined_voltage"
            )
        
        with col2:
            combined_frequency_offset = st.slider(
                "Frequency Offset (Hz)", 
                min_value=-2.0, 
                max_value=2.0, 
                value=0.0, 
                step=0.1,
                key="combined_frequency"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Launch Combined Attack", type="primary"):
                try:
                    success = attack_system.activate_attack_mode(
                        voltage_offset=combined_voltage_offset,
                        frequency_offset=combined_frequency_offset
                    )
                    if success:
                        st.success(f"✅ Combined attack launched!")
                        st.info(f"Voltage offset: {combined_voltage_offset:+.3f} pu")
                        st.info(f"Frequency offset: {combined_frequency_offset:+.1f} Hz")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to launch combined attack")
                except Exception as e:
                    st.error(f"❌ Attack failed: {e}")
        
        with col2:
            if st.button("🛑 Stop All Attacks"):
                try:
                    success = attack_system.deactivate_attack_mode()
                    if success:
                        st.success("✅ All attacks stopped")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop attacks")
                except Exception as e:
                    st.error(f"❌ Stop failed: {e}")
    
    st.markdown("---")
    
    # Real-time attack monitoring
    st.subheader("📊 Attack Impact Monitoring")
    
    # Get current measurements to show attack impact
    try:
        measurements = attack_system.get_real_time_data()
        
        if measurements:
            # Convert to DataFrame for analysis
            df_data = []
            for measurement in measurements:
                df_data.append({
                    'RTU_ID': measurement.get('rtu_id', 'Unknown'),
                    'Bus': measurement.get('bus_number', 0),
                    'Voltage_Magnitude': measurement.get('voltage_magnitude', 0.0),
                    'Frequency': measurement.get('frequency', 50.0),
                    'Status': measurement.get('status', 'Unknown')
                })
            
            df = pd.DataFrame(df_data)
            
            if not df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Voltage impact chart
                    voltage_fig = go.Figure()
                    voltage_fig.add_trace(go.Scatter(
                        x=df['RTU_ID'],
                        y=df['Voltage_Magnitude'],
                        mode='markers+lines',
                        name='Current Voltage',
                        marker=dict(
                            size=8,
                            color=df['Voltage_Magnitude'],
                            colorscale='RdYlGn_r' if attack_active else 'RdYlGn',
                            cmin=0.95,
                            cmax=1.05
                        )
                    ))
                    
                    voltage_fig.add_hline(y=1.05, line_dash="dash", line_color="red")
                    voltage_fig.add_hline(y=0.95, line_dash="dash", line_color="red")
                    voltage_fig.add_hline(y=1.0, line_dash="dot", line_color="blue")
                    
                    voltage_fig.update_layout(
                        title=f"Voltage Impact {'(UNDER ATTACK)' if attack_active else '(NORMAL)'}",
                        xaxis_title="RTU ID",
                        yaxis_title="Voltage (pu)",
                        height=300
                    )
                    
                    st.plotly_chart(voltage_fig, config={"displayModeBar": True}, theme="streamlit")
                
                with col2:
                    # Frequency impact chart
                    frequency_fig = go.Figure()
                    frequency_fig.add_trace(go.Scatter(
                        x=df['RTU_ID'],
                        y=df['Frequency'],
                        mode='markers+lines',
                        name='Current Frequency',
                        marker=dict(
                            size=8,
                            color='red' if attack_active else 'blue'
                        )
                    ))
                    
                    frequency_fig.add_hline(y=50.0, line_dash="dot", line_color="blue", 
                                          annotation_text="Nominal (50 Hz)")
                    frequency_fig.add_hline(y=49.5, line_dash="dash", line_color="orange")
                    frequency_fig.add_hline(y=50.5, line_dash="dash", line_color="orange")
                    
                    frequency_fig.update_layout(
                        title=f"Frequency Impact {'(UNDER ATTACK)' if attack_active else '(NORMAL)'}",
                        xaxis_title="RTU ID",
                        yaxis_title="Frequency (Hz)",
                        height=300
                    )
                    
                    st.plotly_chart(frequency_fig, config={"displayModeBar": True}, theme="streamlit")
                
                # Attack statistics
                if attack_active:
                    st.subheader("🎯 Attack Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        affected_rtus = len(df)
                        st.metric("RTUs Under Attack", affected_rtus)
                    
                    with col2:
                        voltage_anomalies = ((df['Voltage_Magnitude'] > 1.05) | (df['Voltage_Magnitude'] < 0.95)).sum()
                        st.metric("Voltage Anomalies", voltage_anomalies)
                    
                    with col3:
                        frequency_anomalies = ((df['Frequency'] > 50.5) | (df['Frequency'] < 49.5)).sum()
                        st.metric("Frequency Anomalies", frequency_anomalies)
                    
                    with col4:
                        st.metric("Attack Duration", f"{time.time():.0f}s", delta="Active")
        
        else:
            st.warning("⚠️ No measurement data available for impact analysis")
    
    except Exception as e:
        st.error(f"Failed to get measurement data: {e}")
    
    # Sidebar information
    st.sidebar.title("🎯 Attack Control")
    st.sidebar.markdown("---")
    
    # Current attack status
    st.sidebar.subheader("Current Status")
    if attack_active:
        st.sidebar.error("🔴 ATTACK ACTIVE")
    else:
        st.sidebar.success("🟢 NO ACTIVE ATTACKS")
    
    # Attack presets
    st.sidebar.subheader("Quick Attack Presets")
    
    if st.sidebar.button("⚡ Voltage Spike (+5%)"):
        try:
            attack_system.activate_attack_mode(voltage_offset=0.05, frequency_offset=0.0)
            st.sidebar.success("Voltage spike activated!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")
    
    if st.sidebar.button("📉 Voltage Drop (-5%)"):
        try:
            attack_system.activate_attack_mode(voltage_offset=-0.05, frequency_offset=0.0)
            st.sidebar.success("Voltage drop activated!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")
    
    if st.sidebar.button("⚡ Frequency Drift (+1Hz)"):
        try:
            attack_system.activate_attack_mode(voltage_offset=0.0, frequency_offset=1.0)
            st.sidebar.success("Frequency drift activated!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")
    
    if st.sidebar.button("🛑 Emergency Stop"):
        try:
            attack_system.deactivate_attack_mode()
            st.sidebar.success("All attacks stopped!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")
    
    # Auto-refresh for real-time monitoring
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()
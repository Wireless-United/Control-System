#!/usr/bin/env python3
"""
Real-time SCADA Monitoring UI
Streamlit application for monitoring live SCADA measurements
"""

import streamlit as st
import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add simulation directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system status access
from system_status import read_system_status, is_system_running
from integrated_scada import scada_master

# Page configuration
st.set_page_config(
    page_title="SCADA Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_system_data():
    """Get real-time data from the running system"""
    if not is_system_running():
        return None, "System not running. Please start main_cyber_demo.py first."
    
    try:
        # Get status from file
        status = read_system_status()
        if not status:
            return None, "Unable to read system status."
        
        # Get real-time data from system status file instead of SCADA master
        # This ensures we see the same data that the main system sees
        from main_cyber_demo import get_demo_instance
        
        # First try using the demo instance
        demo = get_demo_instance()
        if demo:
            data = demo.get_real_time_data()
        else:
            # If demo instance not available (UI running in separate process),
            # Get data directly from system status file
            data = status.get("real_time_data", [])
            
        # If no data, return empty list
        if not data:
            data = []
            
        return {"status": status, "data": data}, None
        
        return {"status": status, "data": data}, None
    except Exception as e:
        return None, f"Error connecting to system: {e}"

def format_measurement_data(measurements):
    """Format measurement data for display"""
    if not measurements:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df_data = []
    for measurement in measurements:
        df_data.append({
            'RTU_ID': measurement.get('rtu_id', 'Unknown'),
            'Bus': measurement.get('bus_number', 0),
            'Timestamp': measurement.get('timestamp', datetime.now()),
            'Voltage_Magnitude': measurement.get('voltage_magnitude', 0.0),
            'Voltage_Angle': measurement.get('voltage_angle', 0.0),
            'Active_Power': measurement.get('active_power', 0.0),
            'Reactive_Power': measurement.get('reactive_power', 0.0),
            'Frequency': measurement.get('frequency', 50.0),
            'Status': measurement.get('status', 'Unknown')
        })
    
    return pd.DataFrame(df_data)

def create_voltage_chart(df):
    """Create voltage magnitude chart"""
    if df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add voltage magnitude trace
    fig.add_trace(go.Scatter(
        x=df['RTU_ID'],
        y=df['Voltage_Magnitude'],
        mode='markers+lines',
        name='Voltage Magnitude (pu)',
        marker=dict(
            size=8,
            color=df['Voltage_Magnitude'],
            colorscale='RdYlGn',
            colorbar=dict(title="Voltage (pu)"),
            cmin=0.95,
            cmax=1.05
        ),
        line=dict(width=2)
    ))
    
    # Add voltage limits
    fig.add_hline(y=1.05, line_dash="dash", line_color="red", 
                  annotation_text="Upper Limit (1.05 pu)")
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                  annotation_text="Lower Limit (0.95 pu)")
    fig.add_hline(y=1.0, line_dash="dot", line_color="blue", 
                  annotation_text="Nominal (1.0 pu)")
    
    fig.update_layout(
        title="Real-time Voltage Monitoring",
        xaxis_title="RTU ID",
        yaxis_title="Voltage Magnitude (pu)",
        showlegend=True,
        height=400
    )
    
    return fig

def create_power_chart(df):
    """Create power flow chart"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Active Power (MW)', 'Reactive Power (MVAR)']
    )
    
    # Active Power
    fig.add_trace(
        go.Bar(x=df['RTU_ID'], y=df['Active_Power'], name='Active Power'),
        row=1, col=1
    )
    
    # Reactive Power
    fig.add_trace(
        go.Bar(x=df['RTU_ID'], y=df['Reactive_Power'], name='Reactive Power'),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Real-time Power Flow Monitoring",
        showlegend=False,
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔍 Real-time SCADA Monitoring System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ System Control")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 2)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Get system data
    system_data, error = get_system_data()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please start the main system first: `python main_cyber_demo.py`")
        return
    
    if not system_data:
        st.warning("⚠️ No data available")
        return
    
    # System status
    status = system_data["status"]
    measurements = system_data["data"]
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "System Status", 
            status.get("status", "UNKNOWN"),
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        active_rtus = status.get("active_rtus", 0)
        total_rtus = status.get("total_rtus", 0)
        st.metric(
            "RTUs Online", 
            f"{active_rtus}/{total_rtus}",
            delta=None,
            delta_color="normal"
        )
    
    with col3:
        attack_status = "ACTIVE" if status.get("dnp3_attacks_active", False) else "INACTIVE"
        st.metric(
            "Attack Status", 
            attack_status,
            delta=None,
            delta_color="inverse" if attack_status == "ACTIVE" else "normal"
        )
    
    with col4:
        measurements_count = len(measurements) if measurements else 0
        st.metric(
            "Measurements", 
            measurements_count,
            delta=None,
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Check if we have measurement data
    if not measurements:
        st.warning("⚠️ No measurement data available")
        return
    
    # Format measurement data
    df = format_measurement_data(measurements)
    
    if df.empty:
        st.warning("⚠️ Unable to format measurement data")
        return
    
    # Charts
    st.subheader("📊 Real-time Measurements")
    
    # Voltage monitoring
    voltage_chart = create_voltage_chart(df)
    st.plotly_chart(voltage_chart, config={"displayModeBar": True}, theme="streamlit")
    
    # Power monitoring
    power_chart = create_power_chart(df)
    st.plotly_chart(power_chart, config={"displayModeBar": True}, theme="streamlit")
    
    # Detailed data table
    st.subheader("📋 Detailed Measurements")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_rtus = st.multiselect(
            "Filter by RTU ID",
            options=df['RTU_ID'].unique(),
            default=df['RTU_ID'].unique()[:10]  # Show first 10 by default
        )
    
    with col2:
        show_all_columns = st.checkbox("Show All Columns", value=False)
    
    # Filter data
    filtered_df = df[df['RTU_ID'].isin(selected_rtus)] if selected_rtus else df
    
    # Select columns to display
    if show_all_columns:
        display_df = filtered_df
    else:
        display_columns = ['RTU_ID', 'Bus', 'Voltage_Magnitude', 'Active_Power', 'Reactive_Power', 'Frequency', 'Status']
        display_df = filtered_df[display_columns]
    
    # Display table
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    # System information
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 System Info")
    
    if status.get("timestamp"):
        st.sidebar.text(f"Last Update: {status['timestamp'][:19]}")
    
    st.sidebar.text(f"SCADA Running: {'Yes' if status.get('scada_running', False) else 'No'}")
    
    # Highlight attack status more prominently
    attack_active = status.get("dnp3_attacks_active", False)
    if attack_active:
        st.sidebar.error("⚠️ ATTACKS ACTIVE ⚠️")
    else:
        st.sidebar.success("✅ No attacks detected")
    
    # Voltage statistics
    if not df.empty:
        st.sidebar.markdown("**Voltage Statistics:**")
        st.sidebar.text(f"Min: {df['Voltage_Magnitude'].min():.3f} pu")
        st.sidebar.text(f"Max: {df['Voltage_Magnitude'].max():.3f} pu")
        st.sidebar.text(f"Avg: {df['Voltage_Magnitude'].mean():.3f} pu")
        
        # Alarm count
        high_voltage_count = (df['Voltage_Magnitude'] > 1.05).sum()
        low_voltage_count = (df['Voltage_Magnitude'] < 0.95).sum()
        st.sidebar.text(f"High Voltage Alarms: {high_voltage_count}")
        st.sidebar.text(f"Low Voltage Alarms: {low_voltage_count}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
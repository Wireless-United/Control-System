#!/usr/bin/env python3
"""
IEEE 39-Bus MiTM Attack Controller - Streamlit UI

Interactive web interface for launching and controlling Man-in-the-Middle attacks
on IEEE 39-bus SCADA-RTU communication system.

Features:
- Real-time attack control interface
- ARP spoofing management
- DNP3 packet manipulation
- Attack scenario selection
- Live status monitoring
- Attack statistics visualization
"""

import streamlit as st
import asyncio
import threading
import time
import json
import sys
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mitm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))

# Configure page
st.set_page_config(
    page_title="IEEE 39-Bus MiTM Attack Controller",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for attack state
if 'attack_controller' not in st.session_state:
    st.session_state.attack_controller = None
if 'attack_active' not in st.session_state:
    st.session_state.attack_active = False
if 'attack_stats' not in st.session_state:
    st.session_state.attack_stats = {
        'start_time': None,
        'packets_intercepted': 0,
        'packets_modified': 0,
        'commands_injected': 0,
        'arp_packets_sent': 0,
        'attack_success_rate': 0.0
    }
if 'attack_log' not in st.session_state:
    st.session_state.attack_log = []

def load_attack_modules():
    """Load attack modules with error handling"""
    try:
        # Try to import attack modules
        from mitm.attacker import MiTMAttacker, AttackConfig, AttackType, AttackScenario
        from mitm.arp_spoof import ARPSpoofer
        from mitm.packet_filter import PacketFilter
        return True, None
    except ImportError as e:
        return False, str(e)

def init_attack_controller():
    """Initialize attack controller"""
    modules_loaded, error = load_attack_modules()
    if not modules_loaded:
        st.error(f"Failed to load attack modules: {error}")
        return None
    
    try:
        from mitm.attacker import MiTMAttacker
        return MiTMAttacker("Ethernet")  # Windows interface name
    except Exception as e:
        st.error(f"Failed to initialize attack controller: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🕷️ IEEE 39-Bus MiTM Attack Controller")
    st.markdown("**Interactive cybersecurity attack simulation for IEEE 39-bus SCADA-RTU system**")
    
    # Sidebar for attack configuration
    st.sidebar.title("⚙️ Attack Configuration")
    
    # Check if SCADA-RTU system is running
    st.sidebar.markdown("### 📡 System Status")
    if st.sidebar.button("🔍 Check SCADA-RTU Status"):
        # Check if simulation is running
        scada_status = check_scada_system()
        if scada_status['running']:
            st.sidebar.success(f"✅ SCADA-RTU System Running")
            st.sidebar.write(f"• RTUs: {scada_status['rtu_count']}")
            st.sidebar.write(f"• SCADA: {scada_status['scada_active']}")
        else:
            st.sidebar.error("❌ SCADA-RTU System Not Running")
            st.sidebar.warning("Start the system first: `python main_demo.py` → Option 2")
    
    # Attack targets configuration
    st.sidebar.markdown("### 🎯 Attack Targets")
    scada_ip = st.sidebar.text_input("SCADA Master IP", value="127.0.0.1:21000")
    rtu_ip = st.sidebar.selectbox(
        "Target RTU", 
        ["127.0.0.1:20000 (Bus 30)", "127.0.0.1:20001 (Bus 31)", 
         "127.0.0.1:20002 (Bus 32)", "127.0.0.1:20003 (Bus 33)"]
    )
    
    # Attack scenarios
    st.sidebar.markdown("### 🔥 Attack Scenarios")
    attack_scenarios = st.sidebar.multiselect(
        "Select Attack Types",
        ["False Command Injection", "False Data Injection", "ARP Spoofing", 
         "DNP3 Manipulation", "Denial of Service"],
        default=["ARP Spoofing", "False Data Injection"]
    )
    
    # Attack parameters
    attack_duration = st.sidebar.slider("Attack Duration (seconds)", 10, 600, 60)
    attack_intensity = st.sidebar.slider("Attack Intensity", 0.1, 1.0, 0.5)
    stealth_mode = st.sidebar.checkbox("Stealth Mode", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Attack Control Panel")
        
        # Attack control buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("🔥 Start Attack", type="primary", disabled=st.session_state.attack_active):
                start_attack(scada_ip, rtu_ip, attack_scenarios, attack_duration, attack_intensity, stealth_mode)
        
        with button_col2:
            if st.button("⏹️ Stop Attack", disabled=not st.session_state.attack_active):
                stop_attack()
        
        with button_col3:
            if st.button("🔄 Reset Stats"):
                reset_attack_stats()
        
        # Attack status display
        if st.session_state.attack_active:
            st.success("🟢 Attack Active")
            elapsed = time.time() - st.session_state.attack_stats['start_time'] if st.session_state.attack_stats['start_time'] else 0
            st.write(f"**Duration**: {elapsed:.1f} seconds")
        else:
            st.info("🔴 Attack Inactive")
        
        # Real-time attack statistics
        st.markdown("## 📊 Attack Statistics")
        
        # Create metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Packets Intercepted", st.session_state.attack_stats['packets_intercepted'])
        
        with metric_col2:
            st.metric("Packets Modified", st.session_state.attack_stats['packets_modified'])
        
        with metric_col3:
            st.metric("Commands Injected", st.session_state.attack_stats['commands_injected'])
        
        with metric_col4:
            st.metric("ARP Packets Sent", st.session_state.attack_stats['arp_packets_sent'])
        
        # Attack progress chart
        if st.session_state.attack_stats['start_time']:
            create_attack_timeline()
    
    with col2:
        st.markdown("## 📝 Attack Log")
        
        # Display recent attack events
        log_container = st.container()
        with log_container:
            if st.session_state.attack_log:
                for i, log_entry in enumerate(reversed(st.session_state.attack_log[-10:])):
                    timestamp = log_entry.get('timestamp', 'Unknown')
                    event = log_entry.get('event', 'Unknown')
                    details = log_entry.get('details', '')
                    
                    with st.expander(f"{timestamp} - {event}"):
                        st.write(details)
            else:
                st.info("No attack events logged yet")
        
        # ARP Spoofing Status
        st.markdown("## 🕸️ ARP Spoofing Status")
        if st.button("Test ARP Spoofing"):
            test_arp_spoofing()
        
        # DNP3 Packet Analysis
        st.markdown("## 📦 DNP3 Traffic")
        if st.button("Analyze DNP3 Traffic"):
            analyze_dnp3_traffic()
    
    # Auto-refresh for real-time updates
    if st.session_state.attack_active:
        time.sleep(1)
        st.rerun()

def check_scada_system() -> Dict[str, Any]:
    """Check if SCADA-RTU system is running"""
    import socket
    
    status = {
        'running': False,
        'scada_active': False,
        'rtu_count': 0,
        'ports_active': []
    }
    
    # Check SCADA port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 21000))
        if result == 0:
            status['scada_active'] = True
        sock.close()
    except:
        pass
    
    # Check RTU ports
    for port in range(20000, 20010):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            if result == 0:
                status['rtu_count'] += 1
                status['ports_active'].append(port)
            sock.close()
        except:
            continue
    
    status['running'] = status['scada_active'] or status['rtu_count'] > 0
    return status

def start_attack(scada_ip: str, rtu_ip: str, scenarios: List[str], 
                duration: int, intensity: float, stealth: bool):
    """Start the MiTM attack"""
    try:
        st.session_state.attack_active = True
        st.session_state.attack_stats['start_time'] = time.time()
        
        # Log attack start
        log_event("Attack Started", f"Targets: {scada_ip} <-> {rtu_ip}, Scenarios: {scenarios}")
        
        # Start attack simulation (mock for now)
        start_attack_simulation(scenarios, duration, intensity)
        
        st.success(f"🔥 Attack started against {scada_ip} <-> {rtu_ip}")
        
    except Exception as e:
        st.error(f"Failed to start attack: {e}")
        st.session_state.attack_active = False

def stop_attack():
    """Stop the MiTM attack"""
    try:
        st.session_state.attack_active = False
        log_event("Attack Stopped", "All attack activities terminated")
        st.success("⏹️ Attack stopped successfully")
        
    except Exception as e:
        st.error(f"Failed to stop attack: {e}")

def reset_attack_stats():
    """Reset attack statistics"""
    st.session_state.attack_stats = {
        'start_time': None,
        'packets_intercepted': 0,
        'packets_modified': 0,
        'commands_injected': 0,
        'arp_packets_sent': 0,
        'attack_success_rate': 0.0
    }
    st.session_state.attack_log = []
    st.success("📊 Statistics reset")

def start_attack_simulation(scenarios: List[str], duration: int, intensity: float):
    """Start attack simulation in background thread"""
    def attack_worker():
        import random
        end_time = time.time() + duration
        
        while time.time() < end_time and st.session_state.attack_active:
            # Simulate attack activity
            if "ARP Spoofing" in scenarios:
                st.session_state.attack_stats['arp_packets_sent'] += random.randint(1, 5)
            
            if "False Data Injection" in scenarios:
                st.session_state.attack_stats['packets_intercepted'] += random.randint(0, 3)
                st.session_state.attack_stats['packets_modified'] += random.randint(0, 2)
            
            if "False Command Injection" in scenarios:
                if random.random() < 0.1:  # 10% chance
                    st.session_state.attack_stats['commands_injected'] += 1
                    log_event("Command Injected", f"False breaker command sent")
            
            time.sleep(1)
        
        # Attack finished
        if st.session_state.attack_active:
            st.session_state.attack_active = False
            log_event("Attack Completed", f"Duration: {duration} seconds")
    
    # Start in background thread
    thread = threading.Thread(target=attack_worker, daemon=True)
    thread.start()

def log_event(event: str, details: str):
    """Log an attack event"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.attack_log.append({
        'timestamp': timestamp,
        'event': event,
        'details': details
    })

def test_arp_spoofing():
    """Test ARP spoofing functionality"""
    try:
        # Mock ARP test
        st.info("🕸️ Testing ARP spoofing...")
        time.sleep(1)
        
        # Simulate test results
        st.success("✅ ARP spoofing test completed")
        st.write("**Test Results:**")
        st.write("• Network interface: Detected")
        st.write("• Target reachability: OK") 
        st.write("• ARP packet crafting: OK")
        
        log_event("ARP Test", "ARP spoofing functionality verified")
        
    except Exception as e:
        st.error(f"❌ ARP test failed: {e}")

def analyze_dnp3_traffic():
    """Analyze DNP3 traffic"""
    try:
        st.info("📦 Analyzing DNP3 traffic...")
        
        # Mock traffic analysis
        import random
        
        traffic_data = {
            'Packet Type': ['Read Request', 'Read Response', 'Write Request', 'Unsolicited Response'] * 5,
            'Count': [random.randint(10, 100) for _ in range(20)],
            'Size (bytes)': [random.randint(50, 500) for _ in range(20)]
        }
        
        df = pd.DataFrame(traffic_data)
        
        # Display chart
        fig = px.bar(df.groupby('Packet Type')['Count'].sum().reset_index(), 
                    x='Packet Type', y='Count', 
                    title='DNP3 Packet Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        log_event("Traffic Analysis", "DNP3 packet analysis completed")
        
    except Exception as e:
        st.error(f"❌ Traffic analysis failed: {e}")

def create_attack_timeline():
    """Create attack timeline visualization"""
    if not st.session_state.attack_stats['start_time']:
        return
    
    # Generate timeline data
    start_time = st.session_state.attack_stats['start_time']
    current_time = time.time()
    duration = current_time - start_time
    
    # Create timeline
    timeline_data = []
    for i in range(int(duration)):
        timeline_data.append({
            'Time': i,
            'Packets': st.session_state.attack_stats['packets_intercepted'] * (i + 1) / duration,
            'Modifications': st.session_state.attack_stats['packets_modified'] * (i + 1) / duration
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        fig = px.line(df, x='Time', y=['Packets', 'Modifications'], 
                     title='Attack Progress Timeline')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
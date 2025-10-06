#!/usr/bin/env python3
"""
Power Grid Cyber Attack Console
Streamlit UI for DNP3 MiTM attacks on IEEE 39-bus system
"""

import streamlit as st
import time
import random
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Power Grid Cyber Attack Console",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
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
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.4; }
    }
    
    .metric-card {
        background: #262626;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
        margin: 0.5rem 0;
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
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'attack_active' not in st.session_state:
    st.session_state.attack_active = False
    st.session_state.attack_log = []
    st.session_state.system_online = True
    st.session_state.compromised_rtus = set()
    st.session_state.attack_start_time = None

def log_attack(message):
    """Add message to attack log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.attack_log.append(f"[{timestamp}] {message}")
    if len(st.session_state.attack_log) > 50:
        st.session_state.attack_log.pop(0)

def simulate_attack(attack_type, target_rtu, params):
    """Simulate cyber attack execution"""
    st.session_state.attack_active = True
    st.session_state.attack_start_time = datetime.now()
    
    if attack_type == "Data Manipulation":
        log_attack(f"🔴 ATTACK INITIATED: {attack_type}")
        log_attack(f"🎯 Target: {target_rtu}")
        log_attack(f"📊 Injecting false voltage readings...")
        log_attack(f"⚡ Voltage offset: {params.get('voltage_offset', 0)} pu")
        
        if target_rtu == "All RTUs":
            st.session_state.compromised_rtus.update([f"RTU {i}" for i in range(1, 21)])
            log_attack("💥 ALL RTUs COMPROMISED!")
        else:
            st.session_state.compromised_rtus.add(target_rtu)
            log_attack(f"💥 {target_rtu} COMPROMISED!")
    
    elif attack_type == "Communication Disruption":
        log_attack(f"🔴 ATTACK INITIATED: Communication Jamming")
        log_attack(f"📡 Disrupting DNP3 protocol...")
        log_attack(f"🔇 Packet loss: {params.get('packet_loss', 0)}%")
        
    elif attack_type == "False Data Injection":
        log_attack(f"🔴 ATTACK INITIATED: False Data Injection")
        log_attack(f"💉 Injecting malicious measurements...")
        log_attack(f"📈 Injection rate: {params.get('injection_rate', 0)}%")
    
    log_attack("✅ Attack deployed successfully!")

def stop_attack():
    """Stop current attack"""
    st.session_state.attack_active = False
    st.session_state.compromised_rtus.clear()
    log_attack("🛑 ATTACK TERMINATED")
    log_attack("🔧 Restoring normal operations...")

def generate_mock_data():
    """Generate mock system data for demonstration"""
    bus_numbers = list(range(1, 40))
    voltages = [0.98 + random.uniform(-0.05, 0.05) for _ in bus_numbers]
    
    # Simulate attack effects
    if st.session_state.attack_active:
        for i in range(len(voltages)):
            if random.random() < 0.3:  # 30% chance of corruption
                voltages[i] += random.uniform(-0.2, 0.2)
    
    return bus_numbers, voltages

def main():
    # Header
    st.markdown('<h1 class="main-header">🕷️ POWER GRID CYBER ATTACK CONSOLE 🕷️</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - System Control
    with st.sidebar:
        st.header("🔧 Attack Control")
        
        # Attack status
        if st.session_state.attack_active:
            st.markdown('<div class="status-active">⚡ ATTACK ACTIVE ⚡</div>', 
                       unsafe_allow_html=True)
            
            # Time since attack started
            if st.session_state.attack_start_time:
                elapsed = datetime.now() - st.session_state.attack_start_time
                st.write(f"Duration: {elapsed.seconds}s")
            
            if st.button("🛑 STOP ATTACK", use_container_width=True, type="primary"):
                stop_attack()
                st.rerun()
        else:
            st.markdown('<div class="status-online">🟢 READY TO ATTACK</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status
        st.header("📊 System Status")
        st.metric("RTUs Online", "20/20")
        st.metric("RTUs Compromised", len(st.session_state.compromised_rtus))
        st.metric("Attack Success Rate", "100%" if st.session_state.attack_active else "N/A")
        
        # Target System Info
        st.markdown("---")
        st.header("🎯 Target System")
        st.write("**IEEE 39-Bus Power System**")
        st.write("- 39 buses")
        st.write("- 20 RTU outstations")
        st.write("- DNP3 protocol")
        st.write("- SCADA master station")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Attack Configuration Panel
        st.markdown('<div class="attack-panel">', unsafe_allow_html=True)
        st.subheader("🎯 Attack Configuration")
        
        attack_type = st.selectbox(
            "Select Attack Vector",
            ["Data Manipulation", "Communication Disruption", "False Data Injection", "Protocol Fuzzing"],
            help="Choose the type of cyber attack to execute"
        )
        
        target_rtu = st.selectbox(
            "Target RTU",
            ["All RTUs"] + [f"RTU {i}" for i in range(1, 21)],
            help="Select specific RTU or attack all simultaneously"
        )
        
        # Attack parameters
        st.subheader("⚙️ Attack Parameters")
        
        params = {}
        if attack_type == "Data Manipulation":
            params['voltage_offset'] = st.slider("Voltage Offset (pu)", -0.5, 0.5, 0.1, 0.01)
            params['frequency_offset'] = st.slider("Frequency Offset (Hz)", -2.0, 2.0, 0.5, 0.1)
            params['noise_level'] = st.slider("Measurement Noise", 0.0, 1.0, 0.2, 0.05)
            
        elif attack_type == "Communication Disruption":
            params['packet_loss'] = st.slider("Packet Loss (%)", 0, 100, 50, 5)
            params['delay_ms'] = st.slider("Communication Delay (ms)", 0, 5000, 1000, 100)
            
        elif attack_type == "False Data Injection":
            params['injection_rate'] = st.slider("Injection Rate (%)", 0, 100, 25, 5)
            params['corruption_level'] = st.slider("Data Corruption", 0.0, 1.0, 0.3, 0.1)
        
        # Launch attack button
        if not st.session_state.attack_active:
            if st.button("🚀 LAUNCH ATTACK", type="primary", use_container_width=True):
                simulate_attack(attack_type, target_rtu, params)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time voltage monitoring
        st.subheader("📊 Real-Time System Monitoring")
        
        # Generate and display voltage data
        bus_numbers, voltages = generate_mock_data()
        
        fig = go.Figure()
        
        # Normal voltage trace
        normal_color = '#00ff00' if not st.session_state.attack_active else '#ffaa00'
        fig.add_trace(go.Scatter(
            x=bus_numbers,
            y=voltages,
            mode='lines+markers',
            name='Bus Voltages',
            line=dict(color=normal_color, width=2),
            marker=dict(size=4)
        ))
        
        # Add voltage limits
        fig.add_hline(y=1.05, line_dash="dash", line_color="red", 
                     annotation_text="Upper Limit")
        fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                     annotation_text="Lower Limit")
        
        fig.update_layout(
            title=f"IEEE 39-Bus Voltage Profile {'(UNDER ATTACK)' if st.session_state.attack_active else ''}",
            xaxis_title="Bus Number",
            yaxis_title="Voltage (pu)",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attack Console
        st.subheader("🖥️ Attack Console")
        
        # Console output
        console_html = '<div class="console-output">'
        for log_entry in st.session_state.attack_log[-15:]:  # Show last 15 entries
            console_html += f"{log_entry}<br>"
        console_html += '</div>'
        
        st.markdown(console_html, unsafe_allow_html=True)
        
        # Quick attack buttons
        st.subheader("⚡ Quick Attacks")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔥 Voltage Spike", use_container_width=True):
                if not st.session_state.attack_active:
                    simulate_attack("Data Manipulation", "RTU 1", {'voltage_offset': 0.3})
                    st.rerun()
            
            if st.button("📊 False Readings", use_container_width=True):
                if not st.session_state.attack_active:
                    simulate_attack("False Data Injection", "RTU 5", {'injection_rate': 75})
                    st.rerun()
        
        with col_b:
            if st.button("🔇 Comm Jam", use_container_width=True):
                if not st.session_state.attack_active:
                    simulate_attack("Communication Disruption", "All RTUs", {'packet_loss': 90})
                    st.rerun()
            
            if st.button("💥 Full Assault", use_container_width=True):
                if not st.session_state.attack_active:
                    simulate_attack("Data Manipulation", "All RTUs", {'voltage_offset': 0.5})
                    st.rerun()
        
        # System impact
        st.subheader("📈 Attack Impact")
        if st.session_state.attack_active:
            st.error("🚨 SYSTEM UNDER ATTACK")
            st.metric("Data Integrity", "COMPROMISED", delta="-85%")
            st.metric("System Stability", "CRITICAL", delta="-70%")
        else:
            st.success("✅ System Secure")
            st.metric("Data Integrity", "100%")
            st.metric("System Stability", "NORMAL")
    
    # Instructions at bottom
    st.markdown("---")
    st.info("""
    **🛡️ ATTACK SIMULATION INSTRUCTIONS:**
    1. **Initialize the system**: Run `python main_demo.py` and select option 2 (cybersecurity simulation)
    2. **Configure attack**: Select attack type, target RTU, and parameters above
    3. **Launch attack**: Click the red attack button to begin the simulation
    4. **Monitor impact**: Watch real-time voltage data and system alarms
    5. **Stop attack**: Use the stop button to restore normal operations
    
    **⚠️ This is a simulation environment for cybersecurity research and education only.**
    """)
    
    # Auto-refresh when attack is active
    if st.session_state.attack_active:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
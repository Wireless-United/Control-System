# 🎯 SCADA Cybersecurity Demonstration System

## 🚀 Quick Start

### 1. Start the Main System

```bash
cd "n:\PraneshAvv\SEMESTER-5\control systems\Control-System\simulation"
..\.venv\Scripts\python.exe main_cyber_demo.py
```

### 2. Launch SCADA Monitor (In New Terminal)

```bash
cd "n:\PraneshAvv\SEMESTER-5\control systems\Control-System\simulation"
..\.venv\Scripts\streamlit.exe run scada_monitor_ui.py --server.port 8501
```

### 3. Launch Attack Interface (In New Terminal)

```bash
cd "n:\PraneshAvv\SEMESTER-5\control systems\Control-System\simulation"
..\.venv\Scripts\streamlit.exe run attack_ui.py --server.port 8502
```

## 🌐 Web Interface URLs

- **SCADA Monitor**: http://localhost:8501
- **Attack Interface**: http://localhost:8502

## ⚡ What This Demonstrates

### Real Power System Simulation

- IEEE 39-bus power system with actual power flow
- 20 RTU outstations (10 generators, 10 loads)
- Real voltage, frequency, and power measurements

### SCADA Network

- Master station polling RTUs every 2 seconds
- Real-time measurement collection
- Alarm generation for abnormal conditions

### Cybersecurity Attacks

- **Voltage Manipulation**: Modify voltage readings seen by operators
- **Frequency Manipulation**: Alter frequency measurements
- **DNP3 Protocol Attacks**: Intercept and manipulate data in transit

### Attack Impact Visualization

- Real-time charts showing manipulated measurements
- Comparison between normal and attacked states
- Attack statistics and monitoring

## 🎯 Attack Scenarios to Try

1. **Voltage Spike Attack**: +5% voltage manipulation
2. **Voltage Drop Attack**: -5% voltage manipulation
3. **Frequency Drift**: ±1 Hz frequency manipulation
4. **Combined Attack**: Simultaneous voltage and frequency manipulation

## 📊 What You'll See

### SCADA Monitor Shows:

- Real-time voltage magnitudes for all 20 RTUs
- Power flow data (active/reactive power)
- System status and alarms
- Measurement data tables

### Attack Interface Shows:

- Current system status
- Attack configuration controls
- Real-time impact visualization
- Attack statistics and monitoring

### Attack Effects:

- When you activate attacks in the Attack UI, the SCADA Monitor will show the manipulated measurements
- Operators see false readings while actual system remains unaffected
- Demonstrates how cyber attacks can mislead operators

## 🔧 Troubleshooting

### If System Won't Start:

1. Ensure virtual environment is activated
2. Check that all dependencies are installed
3. Run the working test first: `python test_cyber.py`

### If UIs Won't Connect:

1. Ensure main system is running first
2. Check that streamlit is installed: `pip install streamlit plotly`
3. Use full paths to streamlit executable

### For Best Results:

1. Start main system and wait for "SYSTEM IS NOW OPERATIONAL"
2. Then launch both UIs in separate terminals
3. Monitor SCADA readings while conducting attacks
4. Use Ctrl+C to stop any component

## 💡 Educational Value

This demonstration shows:

- How SCADA systems work in power grids
- Vulnerabilities in industrial control protocols
- Impact of data manipulation attacks
- Real-time monitoring and attack detection
- Cybersecurity risks in critical infrastructure

Perfect for cybersecurity research, education, and testing attack detection systems!

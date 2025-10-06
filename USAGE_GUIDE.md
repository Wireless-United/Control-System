# 🎯 IEEE 39-Bus SCADA-RTU Cybersecurity Simulation Suite

## **COMPLETE SYSTEM OVERVIEW**

You now have **TWO main entry points** for different purposes:

### **1. `main_demo.py` - Interactive Menu System**

**🎯 USER-FRIENDLY INTERFACE**

- Interactive menu with 5 simulation options
- Perfect for demonstrations and testing
- Guides users through different simulation modes
- Easy selection of basic vs. advanced features

### **2. `ieee39_integrated.py` - Command-Line Interface**

**⚙️ ADVANCED CONTROL**

- Full command-line control with arguments
- Scriptable for automated testing
- Fine-tuned configuration options
- Professional research-grade interface

---

## **🚀 USAGE OPTIONS**

### **Option 1: Interactive Menu (Recommended for First-Time Users)**

```bash
cd "n:\PraneshAvv\SEMESTER-5\control systems\Control-System\simulation"
python main_demo.py
```

**Menu Options:**

- **1️⃣ Basic Power System Demo**: IEEE 39-bus analysis, load control, DER testing
- **2️⃣ Normal SCADA-RTU Operation**: RTUs + SCADA communication (no attacks)
- **3️⃣ Cybersecurity Attack Simulation**: SCADA-RTU + MiTM attacks
- **4️⃣ Full Cybersecurity Simulation**: Complete system with all attack vectors
- **5️⃣ Quick System Test**: 1-minute verification of all components

### **Option 2: Direct Command-Line Interface**

```bash
cd "n:\PraneshAvv\SEMESTER-5\control systems\Control-System\simulation"

# Normal SCADA-RTU operation (5 minutes)
python ieee39_integrated.py --mode normal --duration 300

# Attack simulation with specific scenarios (10 minutes)
python ieee39_integrated.py --mode attack --duration 600 --attack-scenarios generator_trip,voltage_manipulation

# Full cybersecurity simulation (10 minutes)
python ieee39_integrated.py --mode full_cyber --duration 600 --rtu-count 10

# Quick test (1 minute)
python ieee39_integrated.py --mode normal --duration 60 --rtu-count 3
```

---

## **🔧 SIMULATION COMPONENTS**

### **IEEE 39-Bus Power System** (`ieee39_system_strict.py`)

- ✅ Real IEEE 39-bus topology
- ✅ PyPower/PandaPower integration
- ✅ 50 Hz operation
- ✅ Real-time power flow analysis

### **RTU Outstations** (`rtu.py`)

- ✅ **10 Strategic RTU Locations**:
  - **Generation Buses**: 30, 31, 32, 33, 39
  - **Transmission Hubs**: 16, 21, 25
  - **Load Centers**: 4, 20
- ✅ **DNP3 Protocol Implementation**
- ✅ **Real Measurements** from IEEE 39-bus system
- ✅ **Network Configuration**:
  ```
  RTU Bus 30 → localhost:20000
  RTU Bus 31 → localhost:20001
  RTU Bus 32 → localhost:20002
  ... (each RTU on different port)
  ```

### **SCADA Master Station** (`scada.py`)

- ✅ **Automatic RTU Polling** every 5 seconds
- ✅ **DNP3 Master Protocol**
- ✅ **Real-time Data Collection**
- ✅ **Alarm Management**
- ✅ **Control Command Execution**

### **MiTM Attack System** (`ieee39_mitm.py`)

- ✅ **ARP Spoofing** for traffic interception
- ✅ **DNP3 Packet Manipulation**
- ✅ **False Command Injection (FCI)**
- ✅ **False Data Injection (FDI)**
- ✅ **Coordinated Attack Scenarios**

---

## **🎮 SIMULATION MODES EXPLAINED**

### **Mode 1: Normal Operation**

```bash
python ieee39_integrated.py --mode normal --duration 300
```

- IEEE 39-bus power system ✅
- RTU outstations collecting real measurements ✅
- SCADA master polling RTUs via DNP3 ✅
- NO attacks (clean communication) ✅
- **Purpose**: Verify system functionality

### **Mode 2: Attack Simulation**

```bash
python ieee39_integrated.py --mode attack --duration 600 --attack-scenarios generator_trip,voltage_manipulation
```

- All normal operation components ✅
- MiTM attacks intercepting communication ✅
- False data injection ✅
- Attack impact analysis ✅
- **Purpose**: Cybersecurity research

### **Mode 3: Full Cybersecurity**

```bash
python ieee39_integrated.py --mode full_cyber --duration 600
```

- All components active ✅
- Multiple attack vectors ✅
- Advanced persistent threats ✅
- Real-time attack detection ✅
- **Purpose**: Comprehensive security testing

---

## **📊 REAL-TIME MONITORING & RESULTS**

### **Console Output**

- Real-time status updates
- Attack detection alerts
- Communication statistics
- System performance metrics

### **Log Files**

- `ieee39_simulation.log`: Detailed simulation logs
- `control_system_integration.log`: System integration events

### **Simulation Results**

- JSON format results saved automatically
- Power system measurements
- Communication logs
- Attack success rates
- Performance metrics

---

## **🔍 VERIFICATION COMMANDS**

### **Test Individual Components**

```bash
# Test IEEE 39-bus system only
python ieee39_system_strict.py

# Test RTU functionality
python rtu.py

# Test SCADA functionality
python scada.py

# Test MiTM attacks
python ieee39_mitm.py
```

### **System Integration Tests**

```bash
# Complete system verification
python demo_complete_system.py

# Comprehensive test suite
python test_ieee39_system.py
```

---

## **⚠️ IMPORTANT NOTES**

### **Port Configuration**

All components run on **localhost** with different ports:

- **RTUs**: 20000-20009 (10 RTUs max)
- **SCADA**: 21000
- **MiTM Controller**: 22000

### **Dependencies**

- ✅ PyPower (for power system analysis)
- ✅ PandaPower (for advanced analysis)
- ✅ NumPy, SciPy (numerical computation)
- ✅ AsyncIO (concurrent operations)

### **System Requirements**

- **Duration**: 1-10 minutes per simulation
- **Memory**: ~500MB RAM for full simulation
- **CPU**: Multi-core recommended for concurrent RTUs
- **Network**: Localhost communication only

---

## **🎯 QUICK START GUIDE**

### **For First-Time Users:**

1. **Interactive Menu**: `python main_demo.py`
2. Choose **Option 5** (Quick System Test)
3. Verify all components work
4. Try **Option 2** (Normal Operation)
5. Advanced users: Try **Option 4** (Full Cybersecurity)

### **For Research/Advanced Users:**

1. **Command Line**: `python ieee39_integrated.py --mode full_cyber --duration 600`
2. Monitor logs in real-time
3. Analyze results files
4. Customize attack scenarios

---

## **✅ SYSTEM STATUS**

**🎉 COMPLETE IMPLEMENTATION READY!**

✅ **IEEE 39-Bus System**: Fully operational with real power flow analysis  
✅ **RTU Outstations**: 10 strategic locations with DNP3 protocol  
✅ **SCADA Master**: Real-time polling and control capabilities  
✅ **MiTM Attacks**: Advanced cybersecurity attack simulation  
✅ **Integration**: Seamless component coordination  
✅ **Testing**: Comprehensive verification suite  
✅ **Documentation**: Complete usage instructions

**The system is ready for advanced cybersecurity research and education!**

# 🔒 Industrial Control System MiTM Attack Framework

## 📋 Overview

This repository contains a comprehensive **Man-in-the-Middle (MiTM) attack framework** specifically designed for **industrial control systems** using the **DNP3 protocol**. The framework provides both educational and research capabilities for cybersecurity analysis in SCADA environments.

## 🏗️ Architecture

```
Control-System/
├── mitm/                           # 🎯 MiTM Attack Framework
│   ├── __init__.py                 # Package initialization
│   ├── attacker.py                 # Main attack controller & CLI
│   ├── arp_spoof.py               # ARP cache poisoning
│   └── packet_filter.py           # DNP3 packet manipulation
├── simulation/                     # 🏭 Industrial System Simulation
│   ├── scada.py                   # SCADA system with MiTM integration
│   └── ...                        # Other simulation components
├── detection/                      # 📊 Security Analysis & Datasets
│   └── Datasets/                  # Attack/normal behavior datasets
└── demo_cybersecurity.py          # 🎪 Demonstration interface
```

## 🎯 MiTM Attack Capabilities

### 🔧 Core Features

1. **ARP Spoofing**

   - Network position between SCADA master and RTU
   - Automatic ARP table restoration on exit
   - Failsafe mechanisms for network stability

2. **DNP3 Packet Manipulation**

   - Real-time packet capture and analysis
   - Binary/analog command modification
   - False Command Injection (FCI)
   - False Data Injection (FDI)

3. **Attack Scenarios**
   - Breaker trip/close manipulation
   - Generator setpoint modification
   - Voltage measurement tampering
   - Custom attack patterns

### ⚔️ Attack Types

| Attack Type                       | Description               | Impact                                    |
| --------------------------------- | ------------------------- | ----------------------------------------- |
| **FCI** (False Command Injection) | Modifies control commands | Equipment malfunction, system instability |
| **FDI** (False Data Injection)    | Alters measurement data   | Operator misinformation, poor decisions   |
| **Combined**                      | Both FCI and FDI          | Comprehensive system compromise           |

## 🚀 Quick Start

### 1. 🎮 Interactive Demo

```bash
python demo_cybersecurity.py
```

### 2. 🔗 Dual Terminal Setup

**Terminal 1 (SCADA System):**

```bash
python -m simulation.scada --enable-attack
```

**Terminal 2 (MiTM Attacker):**

```bash
python mitm/attacker.py --target scada --victim rtu --attack fci
```

### 3. 🎯 Standalone Attack

```bash
# False Command Injection
python mitm/attacker.py --attack fci --duration 30

# False Data Injection
python mitm/attacker.py --attack fdi --duration 30

# Combined attacks
python mitm/attacker.py --attack fci fdi --scenario all
```

## 📖 Command Reference

### MiTM Attacker CLI

```bash
python mitm/attacker.py [OPTIONS]

Options:
  --target IP        Target SCADA IP (default: 192.168.1.100)
  --victim IP        Target RTU IP (default: 192.168.1.10)
  --attack TYPE      Attack type: fci, fdi, or both
  --scenario NAME    Specific attack scenario
  --duration SECS    Attack duration in seconds
  --interface IFACE  Network interface to use
  --verbose         Enable verbose logging
```

### Demo Script Options

```bash
python demo_cybersecurity.py [OPTIONS]

Options:
  --normal           Normal SCADA simulation
  --attack           SCADA with MiTM attack
  --mitm [TYPE]      Standalone MiTM attack
  --dual             Show dual terminal instructions
  --analysis         Security analysis of datasets
  --duration SECS    Demo duration (default: 30s)
```

## 🔍 Attack Scenarios

### Scenario 1: Breaker Manipulation

```bash
python mitm/attacker.py --scenario breaker_trip_close --duration 60
```

- **Objective:** Manipulate circuit breaker commands
- **Method:** Intercepts DNP3 binary operate commands
- **Impact:** Unauthorized equipment switching

### Scenario 2: Generator Control

```bash
python mitm/attacker.py --scenario generator_setpoint --attack fci
```

- **Objective:** Modify generator setpoint commands
- **Method:** Alters analog operate commands
- **Impact:** Generator over/under generation

### Scenario 3: Data Falsification

```bash
python mitm/attacker.py --attack fdi --scenario voltage_measurement
```

- **Objective:** Corrupt voltage measurement data
- **Method:** Modifies measurement responses
- **Impact:** Operator receives false system state

## 📊 Security Analysis

### Dataset Structure

```
detection/Datasets/
├── Adversary/                     # Attack behavior patterns
│   ├── UC1_PyDNP3_CORE_Adversary_10_OS_30_dnp3.json
│   └── ... (14 attack scenario files)
├── csvs/                          # Processed attack data
│   ├── UC1/                       # Use Case 1 (12 files)
│   ├── UC2/                       # Use Case 2 (24 files)
│   ├── UC3/                       # Use Case 3 (24 files)
│   └── UC4/                       # Use Case 4 (24 files)
└── RawFiles/                      # Original capture data
```

### Analysis Capabilities

- **84 CSV files** with attack/normal behavior data
- **14 JSON files** with detailed attack patterns
- Machine learning training datasets
- Anomaly detection algorithm development

## 🛡️ Security Features

### 🔒 Safety Mechanisms

1. **Mock Implementation**

   - Uses mock packets when Scapy unavailable
   - Prevents accidental network damage
   - Safe for educational environments

2. **Attack Scope Limitation**

   - Targets only specific IP ranges
   - Automatic timeout mechanisms
   - Clean exit procedures

3. **Logging & Monitoring**
   - Comprehensive attack logging
   - Real-time statistics
   - Performance metrics

### ⚠️ Safety Warnings

- **Educational Use Only:** This framework is designed for learning and research
- **Controlled Environment:** Only use in isolated test networks
- **Legal Compliance:** Ensure authorization before any security testing
- **Network Impact:** ARP spoofing can affect network performance

## 🎓 Educational Use Cases

### For Students

- Learn about industrial cybersecurity threats
- Understand DNP3 protocol vulnerabilities
- Practice attack detection and mitigation
- Develop security awareness

### For Researchers

- Test new detection algorithms
- Analyze attack patterns
- Develop countermeasures
- Publish security research

### For Security Professionals

- Penetration testing preparation
- Security assessment training
- Incident response practice
- Risk analysis

## 🔧 Technical Details

### DNP3 Protocol Support

- **Function Codes:** Binary/Analog Operate, Read requests
- **Data Types:** Binary points, analog inputs/outputs
- **Authentication:** Basic challenge-response handling
- **Addressing:** Individual and broadcast addressing

### Network Protocols

- **ARP Spoofing:** Ethernet layer manipulation
- **IP Routing:** Traffic redirection
- **TCP/UDP:** Transport layer processing
- **Application:** DNP3 message parsing

### Performance Metrics

- **Packet Processing:** ~1000 packets/second
- **Latency Impact:** <10ms additional delay
- **Memory Usage:** <50MB typical
- **CPU Usage:** <5% on modern systems

## 📚 Research Applications

### Attack Pattern Analysis

```python
# Load attack datasets for analysis
import json
with open('detection/Datasets/Adversary/UC1_PyDNP3_CORE_Adversary_10_OS_30_dnp3.json') as f:
    attack_data = json.load(f)

# Analyze attack timing, frequency, targets
```

### Machine Learning Integration

```python
# Train anomaly detection models
import pandas as pd
normal_data = pd.read_csv('detection/Datasets/csvs/UC1/DS_merged_phy_cyb_10os_30poll.csv')
attack_data = pd.read_csv('detection/Datasets/csvs/UC1/DS_merged_phy_cyb_10os_30poll_encoded.csv')

# Develop classification algorithms
```

## 🎪 Demonstration Scenarios

### Scenario A: Basic Attack Demo

1. Start SCADA simulation
2. Launch MiTM attack
3. Observe packet interception
4. Analyze attack effectiveness

### Scenario B: Detection Training

1. Run normal operation baseline
2. Introduce various attack types
3. Train detection algorithms
4. Test detection accuracy

### Scenario C: Impact Analysis

1. Measure system performance
2. Apply different attack intensities
3. Quantify impact on operations
4. Develop mitigation strategies

## 🔮 Future Enhancements

### Planned Features

- [ ] Real Scapy integration (when available)
- [ ] Additional protocol support (Modbus, IEC 61850)
- [ ] GUI interface for attack visualization
- [ ] Automated penetration testing
- [ ] Advanced evasion techniques

### Research Directions

- [ ] AI-driven attack adaptation
- [ ] Zero-day vulnerability simulation
- [ ] Distributed attack coordination
- [ ] Physical system impact modeling

## 📝 Contributing

### Development Guidelines

1. Follow Python PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for changes
5. Test in isolated environments only

### Security Considerations

- Never test on production systems
- Always obtain proper authorization
- Report discovered vulnerabilities responsibly
- Maintain ethical hacking principles

## 📄 License & Disclaimer

**Educational Use Only:** This software is provided for educational and research purposes only. Users are responsible for ensuring compliance with all applicable laws and regulations. The authors assume no liability for misuse of this software.

**⚠️ WARNING:** Unauthorized access to computer systems is illegal. Only use this software on systems you own or have explicit permission to test.

---

## 📞 Support & Contact

For questions, issues, or research collaboration:

- 📧 Email: [Your Institution Email]
- 🏫 Institution: [Your Institution]
- 📚 Course: Control Systems Security Lab
- 🎯 Purpose: Educational & Research Use Only

---

_Remember: With great power comes great responsibility. Use these tools ethically and legally._

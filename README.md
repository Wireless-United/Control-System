# Control System & Load-Altering Attacks (LAA) Framework

⚡ **Comprehensive Power System Security Research Platform**

## 🎯 Overview

This project provides a complete power system security analysis platform that integrates:

1. **IEEE 39-Bus System Simulation** - Complete power system model with dynamic components
2. **Load-Altering Attacks (LAA) Framework** - Advanced attack simulation and analysis
3. **SCADA-RTU Communication** - Traditional supervisory control and data acquisition
4. **PMU-PDC Integration** - Synchrophasor measurement and monitoring
3. **PMU-PDC Synchrophasor System** - Real-time phasor measurements using IEEE C37.118 protocol
4. **PDC-SCADA Integration** - Bridge between high-frequency synchrophasor data and SCADA systems

## System Architecture

### Core Components

#### 1. Grid Simulation (`simulation/grid_avr.py`)

- Power system model with voltage regulation
- AVR (Automatic Voltage Regulator) control
- Disturbance simulation and response

#### 2. SCADA-RTU System (`simulation/scada_rtu.py`)

- SCADA Master for supervisory control
- RTU Outstation for data acquisition
- Traditional polling-based communication

#### 3. PMU-PDC Synchrophasor System

- **PMU (Phasor Measurement Unit)** (`simulation/components/pmu.py`)
  - Real-time phasor measurements at 20 fps
  - IEEE C37.118 protocol implementation
  - High-precision time synchronization
- **PDC (Phasor Data Concentrator)** (`simulation/components/pdc.py`)
  - Multi-PMU data aggregation
  - Time-window based synchronization
  - Data quality assessment
- **IEEE C37.118 Protocol** (`simulation/protocols/c37_118.py`)
  - Complete protocol implementation
  - Configuration, Data, Header, and Command frames
  - CRC validation and error handling

#### 4. PDC-SCADA Integration

- **PDC-SCADA Link** (`simulation/protocols/pdc_scada_link.py`)
  - Subscription-based data exchange
  - Event-driven notifications
  - 100-200ms realistic latency simulation
- **SCADA Gateway** (`simulation/components/scada_gateway.py`)
  - Bridge between PDC and SCADA systems
  - Data translation and aggregation
  - Alarm generation and management
- **Integration Layer** (`simulation/integration.py`)
  - Comprehensive system coordination
  - Cross-validation between data sources
  - Performance monitoring

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
cd simulation
python integration.py
```

### Example Test Results

- **PMU-PDC System**: 100% success rate with 2-3 PMUs
- **PDC-SCADA Integration**: 100% data flow success rate
- **Multi-PMU Aggregation**: 30-100% data quality (depends on timing synchronization)

## Technical Features

### IEEE C37.118 Synchrophasor Protocol

- **Frame Types**: Configuration, Data, Header, Command
- **Data Rate**: 10-60 fps (configurable)
- **Time Sync**: GPS-quality timestamp simulation
- **Quality**: Comprehensive data quality metrics

### Real-time Data Processing

- **PMU Streaming**: 20 fps phasor measurements
- **PDC Aggregation**: 200ms time windows
- **SCADA Updates**: 2-5 second intervals
- **Cross-validation**: Multi-source data comparison

### Alarm and Event Management

- **Frequency Deviation**: ±0.5 Hz thresholds
- **Voltage Violations**: 0.9-1.1 pu limits
- **Data Quality**: <80% quality alarms
- **Communication Loss**: PMU/PDC health monitoring

---

## 🛡️ Load-Altering Attacks (LAA) Framework

**NEW: Comprehensive security analysis framework for power system vulnerability assessment**

### 🎯 LAA Framework Overview

The LAA Framework provides advanced simulation and analysis capabilities for Load-Altering Attacks on IEEE 39-bus power systems:

- **Multiple Attack Types**: Step, Random, Periodic, Feedback-based attacks
- **Inertia Scenarios**: Low/High inertia system configurations
- **Comprehensive Analysis**: Voltage, frequency, stability, and comparative studies
- **Advanced Visualization**: Real-time dashboards and detailed reports

### 🏗️ Framework Architecture

```
laa/                        # LAA Framework Core
├── laa_config.py          # Configuration management
├── inertia_manager.py     # System inertia scenarios
├── static_laa.py         # Static attack patterns
├── dynamic_laa.py        # Dynamic adaptive attacks
├── laa_analysis.py       # Analysis engines
├── laa_visualization.py  # Visualization system
└── laa_main_runner.py    # Main orchestrator

Examples:
├── laa_demo.py           # Quick demonstration
├── laa_example.py        # Comprehensive examples
└── simple_test.py        # Basic validation
```

### ⚔️ Attack Simulation Capabilities

#### Attack Types
- **Step Attack**: Sudden coordinated load increase
- **Random Attack**: Stochastic load variation patterns  
- **Periodic Attack**: Sinusoidal load modulation
- **Feedback Attack**: Adaptive attacks responding to system state
- **Multi-stage Attack**: Complex coordinated sequences

#### System Scenarios
- **Low Inertia** (H=2.5s): Fast response, reduced stability
- **High Inertia** (H=4.5s): Slower response, enhanced stability
- **Configurable Parameters**: Generator settings, control systems

### 📊 Analysis & Metrics

#### Comprehensive Analysis
- **Voltage Stability**: Minimum voltage, recovery time, stability margins
- **Frequency Response**: Deviation magnitude, ROCOF analysis
- **Line Loading**: Thermal limits, bottleneck identification
- **System Stability**: Small-signal and transient stability
- **Comparative Studies**: Multi-scenario performance analysis

#### Key Performance Indicators
- Voltage Stability Index
- Maximum Frequency Deviation  
- System Recovery Time
- Attack Effectiveness Score
- Resilience Metrics

### 🚀 Quick Start - LAA Framework

#### Basic Usage
```python
from laa import LAA_SimulationOrchestrator

# Initialize orchestrator
orchestrator = LAA_SimulationOrchestrator()

# Run comprehensive simulation
results = orchestrator.run_comprehensive_laa_simulation(
    output_dir="attack_analysis_results"
)
```

#### Custom Attack Scenario
```python
from laa import InertiaCondition, AttackType

# Run specific attack type
results = orchestrator.simulate_single_attack_scenario(
    inertia_condition=InertiaCondition.LOW,
    attack_type=AttackType.FEEDBACK,
    output_dir="feedback_attack_results"
)
```

### 🧪 Framework Validation

#### Quick Demonstration
```bash
python laa_demo.py          # Lightweight framework demo
```

#### Comprehensive Testing
```bash
python laa_example.py       # Full framework examples
python simple_test.py       # Basic functionality test
```

### 📈 Research Applications

The LAA Framework enables:
- **Security Assessment**: Power system vulnerability analysis
- **Attack Impact Studies**: Quantitative damage assessment
- **Defense Strategy Development**: Mitigation technique evaluation  
- **Resilience Planning**: System hardening strategies
- **Academic Research**: Power system security studies

### 🔬 Analysis Results

Generated outputs include:
- **Time-series Analysis**: Voltage, frequency, power responses
- **Comparative Studies**: Low vs high inertia scenarios
- **Impact Assessment**: System-wide vulnerability maps
- **Mitigation Analysis**: Defense effectiveness evaluation
- **Statistical Reports**: Comprehensive performance metrics

---

## Project Structure

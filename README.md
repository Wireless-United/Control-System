# Control-System

⚡ Comprehensive Control System Simulation with PMU-PDC and SCADA Integration

## Overview

This project provides a complete power system control simulation that integrates multiple communication layers:

1. **Grid + AVR (Automatic Voltage Regulation)** - Power system simulation with voltage control
2. **SCADA-RTU Communication** - Traditional supervisory control and data acquisition
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

## Quick Start

### Test Individual Components

1. **Test PMU-PDC Communication:**

```bash
cd simulation
python test_multi_pmu.py
```

2. **Test PDC-SCADA Integration:**

```bash
cd simulation
python test_pdc_scada_integration.py
```

3. **Run Complete Integration:**

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

## Project Structure

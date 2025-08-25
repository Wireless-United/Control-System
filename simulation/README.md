# Cyber-Physical Grid Simulation

A modular cyber-## Installationation project that integrates IEEE-39 bus system with AVR (Automatic Voltage Regulator) control using PROFINET-like communication.

## Project Structure

```
simulation/
├── grid.py                 # Main simulation entry point
├── scada.py               # SCADA-RTU demonstration with DNP3
├── test_setup.py          # Test script to verify installation
├── requirements.txt       # Required Python packages
├── components/            # Grid components
│   ├── __init__.py
│   ├── bus.py            # Bus component
│   ├── generator.py      # Generator component
│   ├── load.py           # Load component
│   ├── avr.py            # Automatic Voltage Regulator
│   ├── scada_master.py   # SCADA Master (DNP3)
│   └── rtu.py            # Remote Terminal Unit (DNP3)
├── endpoints/             # Control system endpoints
│   ├── __init__.py
│   ├── sensor.py         # Voltage/current sensors
│   ├── controller.py     # AVR control logic
│   └── actuator.py       # Generator excitation control
└── protocols/             # Communication protocols
    ├── __init__.py
    ├── profinet.py       # PROFINET-like fieldbus protocol
    └── dnp3.py           # DNP3 protocol for SCADA-RTU communication
```

## Features

### Grid Simulation

- **IEEE-39 Bus System**: Full 39-bus power system using pandapower
- **Dynamic Load Changes**: Ability to modify load demand during simulation
- **Real-time Power Flow**: Continuous power flow calculations

### AVR Control System

- **Distributed Control**: Sensor → Controller → Actuator architecture
- **PID Control**: Proportional-Integral-Derivative voltage regulation
- **Multiple Generators**: Support for multiple AVR-controlled generators

### Communication

- **PROFINET Protocol**: Low-latency fieldbus communication (2ms typical)
- **DNP3 Protocol**: SCADA-RTU communication with polling (100-500ms latency)
- **Asynchronous Messaging**: Non-blocking communication between endpoints
- **Message Prioritization**: High/normal/low priority message handling

## Demonstrations

### Basic Grid Simulation

Run the main grid simulation with PROFINET AVR control:

```bash
python grid.py
```

### SCADA-RTU System

Run the SCADA demonstration with DNP3 communication:

```bash
python scada.py
```

1. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python test_setup.py
   ```

## Usage

### Basic Grid Simulation

Run the main grid simulation with PROFINET AVR control:

```bash
python grid.py
```

### SCADA-RTU System

Run the SCADA demonstration with DNP3 communication:

```bash
python scada.py
```

### What the Grid Simulation Does

1. **System Initialization**:

   - Creates IEEE-39 bus power system
   - Deploys AVR systems on selected generators
   - Starts PROFINET communication network

2. **Control Loop Operation**:

   - **Sensors** measure bus voltage at 50 Hz
   - **Controllers** apply PID control logic at 10 Hz
   - **Actuators** adjust generator excitation with 10ms response time
   - All communication via PROFINET protocol

3. **Dynamic Events**:

   - Increases load demand at specific buses
   - Demonstrates AVR response to voltage deviations
   - Shows voltage regulation in action

4. **Logging Output**:
   - System status every 10 seconds
   - Voltage measurements and control actions
   - PROFINET communication statistics

### Expected Output

```
2024-XX-XX XX:XX:XX - INFO - Creating IEEE-39 bus system...
2024-XX-XX XX:XX:XX - INFO - Deploying AVR system for Generator 0 at Bus 30
2024-XX-XX XX:XX:XX - INFO - Starting grid simulation...
...
=== System Status at t=10.0s ===
Gen 0 at Bus 30: V=1.020pu, Vref=1.020pu, Error=0.000pu, Exc=1.000pu
Gen 1 at Bus 31: V=1.019pu, Vref=1.020pu, Error=0.001pu, Exc=1.005pu
PROFINET: 156/156 delivered, avg latency: 2.1ms

=== SIMULATING LOAD INCREASE ===
Changed Load 0 demand by ΔP=20.0MW, new demand: P=117.0MW, Q=44.3MVAR
   Bus 3 voltage before load change: 1.021 pu
...
```

### What the SCADA System Does

1. **SCADA Master Initialization**:

   - Creates DNP3 SCADA master with 5-second polling interval
   - Registers multiple RTUs for monitoring different grid locations
   - Starts DNP3 network with 100-500ms communication delays

2. **RTU Operation**:

   - **Generation RTUs**: Monitor generator buses (30, 31)
   - **Load Center RTU**: Monitors major load bus (3)
   - **Data Collection**: Voltage, power, breaker status
   - **Control Capabilities**: Breaker control, voltage setpoints

3. **Communication Comparison**:

   - **PROFINET AVR**: 2ms latency, 50Hz sensor updates, local control
   - **DNP3 SCADA**: 100-500ms latency, 5-second polling, supervisory control
   - **Demonstrates**: Fast vs. slow control loop differences

4. **Operator Scenarios**:
   - Remote voltage setpoint adjustment via SCADA
   - Breaker operation commands
   - Generator start/stop control
   - All actions logged with communication delays

### Expected SCADA Output

````
🏭 SCADA-RTU Communication Demonstration using DNP3
============================================================
16:45:23 - INFO - DNP3 network scada_dnp3 started
16:45:23 - INFO - SCADA Master Control Center SCADA started with 5-second polling interval
16:45:23 - INFO - RTU Generation RTU 1 initialized for Bus 30
...
=== System Status at t=10.0s ===
SCADA Master: 18 measurements, 0 alarms
DNP3 Network: 6/6 delivered, avg latency: 287.3ms
RTU Generation RTU 1: Online, last communication: 0.3s ago
...
🎯 === SCENARIO 1: Operator Voltage Setpoint Change ===
16:45:38 - INFO - 📊 RTU 10: Voltage setpoint changed to 1.040 pu for Gen 0
16:45:38 - INFO - 📊 Updated AVR setpoint for Gen 0 to 1.040 pu
...## Configuration

### AVR Parameters

- **Voltage Setpoint**: 1.02 pu (default)
- **PID Gains**: Kp=5.0, Ki=0.5, Kd=0.1
- **Excitation Limits**: 0.5 to 1.5 pu

### Communication Parameters

- **PROFINET Latency**: 2ms typical
- **Sensor Sampling**: 50 Hz
- **Control Rate**: 10 Hz
- **Actuator Response**: 10ms

### Simulation Parameters

- **Time Step**: 1 second
- **Total Runtime**: ~40 seconds (with load changes)

## Extending the Simulation

### Adding More Generators

```python
# Deploy AVR on additional generators
for gen_id in [3, 4, 5]:
    grid.deploy_avr_system(gen_id, voltage_setpoint=1.01)
````

### Changing AVR Parameters

```python
# Modify AVR gains
controller.avr.kp = 10.0  # More aggressive control
controller.avr.ki = 1.0
controller.avr.kd = 0.2
```

### Adding Other Protocols

The modular design allows easy addition of other fieldbus protocols:

- DNP3 for SCADA communication
- IEEE C37.118 for PMU data
- Modbus for industrial devices

## Technical Details

### Grid Model

- **Base System**: IEEE-39 bus test case
- **Voltage Levels**: 345 kV transmission system
- **Generator Control**: Voltage-controlled buses
- **Load Model**: Constant power loads

### Control System

- **Architecture**: Distributed control with local AVR loops
- **Communication**: Message-based asynchronous protocol
- **Real-time**: Sub-100ms control loop performance

### Software Architecture

- **Asyncio**: For concurrent operation
- **Pydantic**: For data validation and modeling
- **Pandapower**: For power system analysis
- **Modular Design**: Easy to extend and modify

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `python test_setup.py` to verify installation
2. **Power Flow Convergence**: Check load/generation balance
3. **Communication Delays**: Adjust PROFINET latency if needed

### Performance Notes

- Simulation runs in real-time by default
- Control loops operate faster than simulation time step
- Message queuing prevents blocking in communication

## Next Steps

This simulation provides the foundation for:

- **SCADA Integration**: Add DNP3 protocol for wider-area control
- **PMU Networks**: Implement IEEE C37.118 for synchrophasor data
- **Cybersecurity Testing**: Add attack/defense scenarios
- **Machine Learning**: Integrate ML-based control algorithms

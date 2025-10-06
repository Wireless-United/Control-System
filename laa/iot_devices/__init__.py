#!/usr/bin/env python3
"""
IoT Devices Module for Load-Altering Attacks

This module provides IoT device models and botnet control
for executing sophisticated load-altering attacks on power systems.

Author: Pranaav
Date: October 2025
"""

from .device_models import (
    SmartThermostat,
    SmartWaterHeater,
    SmartEVCharger,
    SmartPoolPump,
    IndustrialIoTController,
    IoTDeviceType,
    IoTDeviceState,
    IoTDeviceParams
)

from .device_controller import (
    IoTBotnetController,
    BotnetStrategy,
    BotnetStats
)

from .integration import IoTLAAIntegrator

__all__ = [
    # Device models
    'SmartThermostat',
    'SmartWaterHeater',
    'SmartEVCharger',
    'SmartPoolPump',
    'IndustrialIoTController',
    'IoTDeviceType',
    'IoTDeviceState',
    'IoTDeviceParams',
    
    # Botnet controller
    'IoTBotnetController',
    'BotnetStrategy',
    'BotnetStats',
    
    # Integration
    'IoTLAAIntegrator'
]

__version__ = '1.0.0'
__author__ = 'Pranaav'

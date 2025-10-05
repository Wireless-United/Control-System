#!/usr/bin/env python3
"""
LAA Attacker Module

This module contains all configuration files and attack orchestration logic:
- laa_config.py: Core configuration classes and enums
- inertia_manager.py: System inertia scenario management  
- ieee_protocols.py: IEEE standards compliance

Author: Pranaav
Date: October 2025
"""

from .laa_config import (
    InertiaCondition, AttackType, AttackSeverity,
    SystemInertiaConfig, AttackConfig, SimulationConfig,
    LAA_Configurations, TargetBusSelections,
    create_default_simulation_setup
)

from .inertia_manager import InertiaScenarioManager

from .ieee_protocols import IEEE_Standards, IEEE_SystemParameters

__all__ = [
    'InertiaCondition', 'AttackType', 'AttackSeverity',
    'SystemInertiaConfig', 'AttackConfig', 'SimulationConfig',
    'LAA_Configurations', 'TargetBusSelections',
    'create_default_simulation_setup',
    'InertiaScenarioManager',
    'IEEE_Standards', 'IEEE_SystemParameters'
]

__version__ = "2.0.0"
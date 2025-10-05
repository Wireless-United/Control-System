#!/usr/bin/env python3
"""
LAA Framework v2.0 - Clean 3-Folder Architecture

This module initializes the Load-Altering Attacks (LAA) simulation framework
with clean organization: static/, dynamic/, attacker/ folders only.

Author: Pranaav
Date: October 2025
"""

# Framework version
__version__ = "2.0.0"
__author__ = "Pranaav"
__description__ = "Clean 3-Folder LAA Framework: static/, dynamic/, attacker/"

# Attacker module (all configurations)
from .attacker import (
    InertiaCondition, AttackType, AttackSeverity,
    SystemInertiaConfig, AttackConfig, SimulationConfig,
    LAA_Configurations, TargetBusSelections,
    create_default_simulation_setup,
    InertiaScenarioManager,
    IEEE_Standards, IEEE_SystemParameters
)

# Static attack module
from .static import (
    StaticLAAGenerator, StaticAttackExecutor, 
    AttackResult, StaticAttackSequence
)

# Dynamic attack module  
from .dynamic import (
    AsyncDynamicAttackGenerator, AsyncAttackOrchestrator, 
    AsyncAttackState
)

# Main orchestrator (simplified for 3-folder structure)
from .simple_main_runner import SimpleLAA_Orchestrator, LAA_SimulationOrchestrator

__all__ = [
    # Core orchestration
    'SimpleLAA_Orchestrator',
    'LAA_SimulationOrchestrator',  # Alias for backward compatibility
    'InertiaScenarioManager',
    
    # Static attacks
    'StaticLAAGenerator',
    'StaticAttackExecutor',
    'AttackResult',
    'StaticAttackSequence',
    
    # Dynamic attacks
    'AsyncDynamicAttackGenerator',
    'AsyncAttackOrchestrator', 
    'AsyncAttackState',
    
    # Configuration classes (from attacker)
    'InertiaCondition',
    'AttackType',
    'AttackSeverity',
    'SystemInertiaConfig',
    'AttackConfig',
    'SimulationConfig',
    'LAA_Configurations',
    'TargetBusSelections',
    'create_default_simulation_setup',
    
    # IEEE Standards (from attacker)
    'IEEE_Standards',
    'IEEE_SystemParameters'
]
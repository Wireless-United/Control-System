#!/usr/bin/env python3
"""
LAA Visualization Package

This package provides comprehensive IEEE-compliant visualization capabilities
for the Load Altering Attack (LAA) framework. It includes:

- IEEE-standard color schemes and plot formatting
- Frequency, voltage, power, and stability analysis
- Real-time attack visualization 
- Comprehensive reporting capabilities
- Integration with all LAA framework modules

IEEE Standards Compliance:
- IEEE 1547.1: Distributed Energy Resource standards
- IEEE C37.118: Synchrophasor measurement standards  
- IEEE 421.5: Excitation system standards
- IEEE 1110: Generator modeling standards

Author: Pranaav
Date: October 2025
"""

from .ieee_graphs import (
    IEEE_Colors,
    IEEE_PlotConfig,
    IEEE_FrequencyAnalyzer,
    IEEE_VoltageAnalyzer, 
    IEEE_PowerFlowAnalyzer,
    IEEE_StabilityAnalyzer,
    IEEE_AttackVisualization
)

from .visualization_orchestrator import (
    LAAVisualizationOrchestrator,
    quick_attack_analysis,
    create_visualization_orchestrator
)

# Package metadata
__version__ = "2.0.0"
__author__ = "Pranaav"
__license__ = "MIT"
__description__ = "IEEE-compliant visualization package for LAA framework"

# IEEE standards compliance information
IEEE_STANDARDS = {
    "1547.1": "Standard for Conformance Test Procedures for Equipment Interconnecting Distributed Energy Resources",
    "C37.118": "Standard for Synchrophasor Measurements for Power Systems", 
    "421.5": "Recommended Practice for Excitation System Models for Power System Stability Studies",
    "1110": "Guide for Synchronous Generator Modeling Practices and Parameter Verification"
}

# Package-level exports
__all__ = [
    # IEEE Graphics Components
    'IEEE_Colors',
    'IEEE_PlotConfig',
    'IEEE_FrequencyAnalyzer', 
    'IEEE_VoltageAnalyzer',
    'IEEE_PowerFlowAnalyzer',
    'IEEE_StabilityAnalyzer', 
    'IEEE_AttackVisualization',
    
    # Orchestration Components
    'LAAVisualizationOrchestrator',
    'quick_attack_analysis',
    'create_visualization_orchestrator',
    
    # Package Metadata
    '__version__',
    '__author__',
    '__license__',
    '__description__',
    'IEEE_STANDARDS'
]
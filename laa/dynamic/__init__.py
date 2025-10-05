#!/usr/bin/env python3
"""
LAA Dynamic Module

This module contains dynamic attack generation capabilities:
- async_dynamic_laa.py: Async dynamic attack generation with IEEE compliance
- sync_dynamic_laa.py: Synchronous dynamic attack implementation (if needed)

Author: Pranaav
Date: October 2025
"""

from .async_dynamic_laa import (
    AsyncDynamicAttackGenerator,
    AsyncAttackOrchestrator,
    AsyncAttackState
)

try:
    from .sync_dynamic_laa import SyncDynamicAttackGenerator
    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False

__all__ = [
    'AsyncDynamicAttackGenerator',
    'AsyncAttackOrchestrator', 
    'AsyncAttackState'
]

if SYNC_AVAILABLE:
    __all__.append('SyncDynamicAttackGenerator')
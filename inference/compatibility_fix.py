#!/usr/bin/env python3
"""
Fixed compatibility script for Python 3.13
"""

# Fix for collections.Callable deprecated in Python 3
import collections
import collections.abc
import builtins
import sys

print(f"Loading compatibility layer for Python {sys.version.split()[0]}...")

# Define a custom callable function that works in Python 3.13
def is_callable(obj):
    return hasattr(obj, '__call__')

# Apply fixes for Python 3.12+
if sys.version_info >= (3, 12):
    # Fix the callable check
    builtins.callable = is_callable
    
    # Fix collections.Callable for libraries that directly use it
    collections.Callable = collections.abc.Callable
    
    # For pyreadline and other libraries
    sys.modules['collections'].Callable = collections.abc.Callable
    
    print("Applied Python 3.12+ compatibility fixes")

# Fix for string type checking
try:
    # For libraries that use basestring (Python 2 compatibility)
    if not hasattr(builtins, 'basestring'):
        builtins.basestring = str
        print("Applied basestring compatibility fix")
except Exception as e:
    print(f"Failed to apply basestring fix: {e}")

# Suppress TensorFlow warnings
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    print("Suppressed TensorFlow warnings")
except Exception as e:
    print(f"Failed to suppress TensorFlow warnings: {e}")

print("Compatibility layer loaded successfully")
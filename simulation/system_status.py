#!/usr/bin/env python3
"""
System Status File for Inter-Process Communication
This file stores the current system status for UI applications to access
"""

import json
import os
import time
from datetime import datetime

STATUS_FILE = "system_status.json"

def write_system_status(status_data):
    """Write system status to file"""
    try:
        status_data['last_update'] = datetime.now().isoformat()
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"Error writing status file: {e}")

def read_system_status():
    """Read system status from file"""
    try:
        if not os.path.exists(STATUS_FILE):
            return None
        
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
        
        # Check if status is recent (within last 10 seconds)
        last_update = datetime.fromisoformat(status.get('last_update', ''))
        if (datetime.now() - last_update).total_seconds() > 10:
            return None
            
        return status
    except Exception as e:
        print(f"Error reading status file: {e}")
        return None

def is_system_running():
    """Check if main system is running"""
    status = read_system_status()
    return status is not None and status.get('status') == 'ONLINE'
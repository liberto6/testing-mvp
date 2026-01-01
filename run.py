#!/usr/bin/env python3
"""
Launcher script for Pipecat Voice Pipeline
Handles Python path setup automatically
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check for CONFIG_PATH environment variable
config_path = os.getenv("CONFIG_PATH")
if config_path:
    print(f"📋 Using configuration from: {config_path}")
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

# Now import and run
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

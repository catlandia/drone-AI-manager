#!/usr/bin/env python3
"""
Drone AI Manager Installer

Cross-platform installer - works on Windows, Mac, and Linux.
Just run: python install.py
"""

import subprocess
import sys
import os


def main():
    print("=" * 50)
    print("  Drone AI Manager Installer")
    print("=" * 50)
    print()
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Installing drone-ai package...")
    print()

    # Install with pip
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", ".[all]"
        ])
    except subprocess.CalledProcessError:
        print()
        print("ERROR: Installation failed!")
        print("Try running with administrator/sudo privileges")
        return 1

    print()
    print("=" * 50)
    print("  Installation Complete!")
    print("=" * 50)
    print()
    print("Quick test:")
    print('  python -c "from drone_ai import DroneDeliveryEnv; print(\'OK\')"')
    print()
    print("Run demo:")
    print("  drone-demo --no-render --episodes 1")
    print()
    print("Run with visualization:")
    print("  drone-demo --render")
    print()

    # Quick test
    print("Running quick test...")
    try:
        subprocess.check_call([
            sys.executable, "-c",
            "from drone_ai import DroneDeliveryEnv, MissionPlanner; print('Test passed!')"
        ])
    except subprocess.CalledProcessError:
        print("Warning: Quick test failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

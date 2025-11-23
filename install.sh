#!/bin/bash
# Drone AI Installer
# Usage: ./install.sh [--viz] [--dev] [--all]

set -e

echo "=================================="
echo "  Drone AI Manager Installer"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Parse arguments
EXTRAS=""
for arg in "$@"; do
    case $arg in
        --viz)
            EXTRAS="[viz]"
            echo "Installing with visualization support (pygame)"
            ;;
        --dev)
            EXTRAS="[dev]"
            echo "Installing with development tools (pytest)"
            ;;
        --all)
            EXTRAS="[all]"
            echo "Installing with all extras"
            ;;
        --help|-h)
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --viz   Install with pygame for visualization"
            echo "  --dev   Install with pytest for development"
            echo "  --all   Install all optional dependencies"
            echo "  --help  Show this help message"
            echo ""
            exit 0
            ;;
    esac
done

# Install in editable mode
echo ""
echo "Installing drone-ai package..."
pip install -e ".$EXTRAS"

echo ""
echo "=================================="
echo "  Installation Complete!"
echo "=================================="
echo ""
echo "Quick test:"
echo "  python -c \"from drone_ai import DroneDeliveryEnv; print('OK')\""
echo ""
echo "Run demo:"
echo "  drone-demo --help"
echo "  drone-demo --no-render --episodes 1"
echo ""
echo "Run with visualization (requires --viz):"
echo "  drone-demo --render"
echo ""

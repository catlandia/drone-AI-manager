@echo off
REM Drone AI Manager Installer for Windows
REM Just double-click to install!

echo ==================================
echo   Drone AI Manager Installer
echo ==================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

echo Installing drone-ai package...
echo.

REM Install with all dependencies
pip install -e ".[all]"

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo Try running as Administrator or check your Python installation
    pause
    exit /b 1
)

echo.
echo ==================================
echo   Installation Complete!
echo ==================================
echo.
echo Quick test - run this command:
echo   python -c "from drone_ai import DroneDeliveryEnv; print('OK')"
echo.
echo Run demo:
echo   drone-demo --no-render --episodes 1
echo.
echo Run with visualization:
echo   drone-demo --render
echo.
pause

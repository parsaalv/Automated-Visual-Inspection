@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Robotic Simulator - Virtual Environment & Launcher
echo ========================================================
echo.

:: Navigate to the script's directory
cd /d "%~dp0"

:: 1. Check or Create Virtual Environment (.venv)
echo [1/4] Checking Virtual Environment (.venv)...
if not exist ".venv" (
    echo [INFO] Virtual environment '.venv' not found. Creating .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment '.venv'. Make sure Python is installed.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
    set NEED_INSTALL=1
) else (
    echo [OK] Virtual environment '.venv' found.
)

:: 2. Activate Virtual Environment
echo [2/4] Activating Virtual Environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment in '.venv\Scripts\activate.bat'.
    pause
    exit /b 1
)

:: 3. Install Requirements inside .venv if needed
if defined NEED_INSTALL (
    echo [INFO] Installing Simulator requirements into .venv...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Package installation failed inside .venv.
        pause
        exit /b 1
    )
) else (
    python -c "import pybullet, cv2, numpy, torch, ultralytics, anomalib, gdown" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [INFO] Missing requirements inside .venv. Installing dependencies...
        pip install -r requirements.txt
    ) else (
        echo [OK] All required packages are installed in .venv.
    )
)

:: 4. Check and Download anomalib_outputs model weights folder
echo.
echo [3/4] Checking model asset: anomalib_outputs...
if not exist "anomalib_outputs" (
    echo [INFO] 'anomalib_outputs' folder not found. Downloading from Google Drive...
    python -m gdown --folder "https://drive.google.com/drive/folders/1ugl2caHKBjgp5Yr9JpxZ0Sm9MXbW9h_V" -O anomalib_outputs
    if %errorlevel% neq 0 (
        echo [WARNING] Automatic download of anomalib_outputs failed. Please download manually from:
        echo https://drive.google.com/drive/folders/1ugl2caHKBjgp5Yr9JpxZ0Sm9MXbW9h_V
    )
) else (
    echo [OK] 'anomalib_outputs' folder present.
)

:: 5. Check and Download industrial_parts_det-2 dataset/weights folder
echo.
echo [4/4] Checking model asset: industrial_parts_det-2...
if not exist "industrial_parts_det-2" (
    echo [INFO] 'industrial_parts_det-2' folder not found. Downloading from Google Drive...
    python -m gdown --folder "https://drive.google.com/drive/folders/1qxHBJZB8lC476KrMKF-iHAyKYWUKUoGm" -O industrial_parts_det-2
    if %errorlevel% neq 0 (
        echo [WARNING] Automatic download of industrial_parts_det-2 failed. Please download manually from:
        echo https://drive.google.com/drive/folders/1qxHBJZB8lC476KrMKF-iHAyKYWUKUoGm
    )
) else (
    echo [OK] 'industrial_parts_det-2' folder present.
)

echo.
echo ========================================================
echo   Launching Simulator inside .venv (main.py)...
echo ========================================================
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Simulation exited with error code %errorlevel%.
)

pause

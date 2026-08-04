@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Model Training - Environment Setup ^& Jupyter Launcher
echo ========================================================
echo.

cd /d "%~dp0"

:: 1. Check or Create Virtual Environment (.venv)
echo [1/3] Checking Virtual Environment (.venv)...
if not exist ".venv" (
    echo [INFO] Virtual environment '.venv' not found. Creating .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment '.venv'. Make sure Python is installed.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [OK] Virtual environment '.venv' found.
)

:: 2. Activate Virtual Environment & Install Requirements
echo [2/3] Activating environment and installing requirements...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [INFO] Upgrading pip and installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

:: Register the kernel for Jupyter
echo [INFO] Registering IPython kernel...
python -m ipykernel install --user --name=rv_model_env --display-name "Python (RV Model Env)"

:: 3. Launch Jupyter Notebook
echo.
echo [3/3] Launching Jupyter Notebook...
echo Please open 'RV_ProjectV1_Gem.ipynb' from the Jupyter interface.
echo Ensure you select the 'Python (RV Model Env)' kernel.
echo.

jupyter notebook

pause

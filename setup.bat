@echo off
chcp 65001 >nul
title VisionaryAI — Setup

echo.
echo  ================================================
echo   VisionaryAI Desktop - Setup (Windows)
echo  ================================================
echo.

:: Check Node
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found.
    echo         Download from: https://nodejs.org
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('node -v') do echo [OK] Node.js %%v

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo         Download from: https://python.org
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo [OK] %%v

echo.
echo [1/3] Installing Python API dependencies (FastAPI only - fast install)...
cd backend
pip install fastapi uvicorn[standard] python-multipart pydantic --quiet
echo [OK] FastAPI installed

echo.
echo [1b/3] Installing ML dependencies (TensorFlow - may take 5-15 min)...
pip install tensorflow numpy pillow opencv-python-headless scikit-learn matplotlib --quiet
echo [OK] ML dependencies installed
cd ..

echo.
echo [2/3] Installing Node.js dependencies (no C++ compiler needed)...
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] npm install failed. See error above.
    pause & exit /b 1
)
echo [OK] Node.js dependencies installed

echo.
echo [3/3] Installing React frontend dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo [WARN] Frontend npm install had issues, trying to continue...
)
cd ..
echo [OK] Frontend dependencies installed

echo.
echo  ================================================
echo   Setup complete!
echo.
echo   IMPORTANT: Copy your best_model.keras into:
echo   backend\best_model.keras
echo.
echo   Then run:  npm run dev
echo  ================================================
echo.
pause

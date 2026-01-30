@echo off
title EURON ML Automation

echo.
echo =============================================
echo     EURON ML Automation - Startup
echo =============================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

REM Create necessary directories
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "reports" mkdir reports

echo.
echo Starting services...
echo.

REM Start FastAPI backend
start "EURON Backend" cmd /k "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
timeout /t 5 /nobreak > nul

REM Start Streamlit frontend
start "EURON Frontend" cmd /k "cd frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0"

echo.
echo =============================================
echo     EURON ML Automation is running!
echo =============================================
echo.
echo Frontend (Streamlit): http://localhost:8501
echo Backend (FastAPI):    http://localhost:8000
echo API Docs:             http://localhost:8000/docs
echo.
echo Close the terminal windows to stop services.
echo =============================================

pause

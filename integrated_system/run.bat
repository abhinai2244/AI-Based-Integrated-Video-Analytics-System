@echo off
echo Starting AI Video Analytics System...

:: Start FastAPI backend
start "FastAPI Backend" cmd /c "cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: Start Streamlit frontend
start "Streamlit Dashboard" cmd /c "cd frontend && streamlit run dashboard.py"

echo Both services started!

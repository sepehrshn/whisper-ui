@echo off
echo Starting Whisper server on http://127.0.0.1:8000
cd /d "%~dp0server"
call .venv\Scripts\activate.bat
uvicorn main:app --reload --port 8000

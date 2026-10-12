@echo off
rem Mosaic Vectorizer: first run creates the Python environment, then starts the app.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Setting up Python environment, this takes a few minutes the first time...
  py -3.12 -m venv .venv || python -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
start "" .venv\Scripts\pythonw.exe app.py

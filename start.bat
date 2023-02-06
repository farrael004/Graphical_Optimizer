@echo off

set VENV_DIR=venv

REM Set the path to the Python interpreter
set PYTHON="%~dp0%VENV_DIR%\Scripts\Python.exe"

REM start cmd /k
start cmd /k "%~dp0%VENV_DIR%\Scripts\activate.bat && cd front-end && streamlit run main.py --server.headless=true"
start cmd /k "%PYTHON% front-end\flask_test.py"


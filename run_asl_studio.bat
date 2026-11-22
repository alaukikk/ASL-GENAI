@echo off
cd /d "%~dp0"
python asl_studio.py
if errorlevel 1 pause

@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "WEB_URL=http://127.0.0.1:5378"
set "WEB_PORT=5378"
set "TENSORRT_HOME=%~dp0TensorRT-10.15.1.29"

if exist "%TENSORRT_HOME%\bin\nvinfer_10.dll" (
    set "PATH=%TENSORRT_HOME%\lib;%TENSORRT_HOME%\bin;%PATH%"
)

echo Starting CaneSegLab Web...
echo Project dir: %~dp0
echo URL: %WEB_URL%
echo.

if exist "%PYTHON_EXE%" goto run_web

set "PYTHON_EXE=python"
python --version >nul 2>nul
if errorlevel 1 (
    echo Python not found.
    echo Please make sure .venv\Scripts\python.exe exists, or python is available in PATH.
    pause
    exit /b 1
)

:run_web
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%WEB_PORT% .*LISTENING"') do (
    echo Found existing process on port %WEB_PORT%: PID=%%P
    taskkill /PID %%P /F >nul 2>nul
)

"%PYTHON_EXE%" main.py web --open-browser

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo Web service exited with code %EXIT_CODE%.
) else (
    echo Web service stopped.
)
pause
exit /b %EXIT_CODE%

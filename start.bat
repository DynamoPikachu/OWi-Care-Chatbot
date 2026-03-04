@echo off
REM Start-Skript für Ask OWi (Windows)

cd /d "%~dp0"

set VENV_DIR=venv

REM Python finden
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python wurde nicht gefunden. Bitte installiere Python 3.10+
    pause
    exit /b 1
)

echo 🐍 Verwende: python

REM venv erstellen falls nicht vorhanden
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo 📦 Erstelle virtuelle Umgebung...
    python -m venv %VENV_DIR%
    
    echo 📥 Installiere Abhängigkeiten...
    call %VENV_DIR%\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call %VENV_DIR%\Scripts\activate.bat
)

REM Prüfe ob chroma DB existiert
if not exist "chroma" (
    echo 🗄️ Erstelle Datenbank (dies kann einige Minuten dauern^)...
    python populate_database.py
)

echo 🚀 Starte Ask OWi...
python gui.py

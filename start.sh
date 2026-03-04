#!/bin/bash
# Start-Skript für Ask OWi (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON_CMD=""

# Python finden
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python wurde nicht gefunden. Bitte installiere Python 3.10+"
    exit 1
fi

echo "🐍 Verwende: $PYTHON_CMD"

# venv erstellen falls nicht vorhanden
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Erstelle virtuelle Umgebung..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    echo "📥 Installiere Abhängigkeiten..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

# Prüfe ob chroma DB existiert
if [ ! -d "chroma" ]; then
    echo "🗄️ Erstelle Datenbank (dies kann einige Minuten dauern)..."
    python populate_database.py
fi

echo "🚀 Starte Ask OWi..."
python gui.py

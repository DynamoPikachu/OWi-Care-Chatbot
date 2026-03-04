#!/bin/bash
# Build-Skript für standalone Executable (Linux)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Aktiviere venv falls vorhanden
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Installiere PyInstaller falls nicht vorhanden
pip install pyinstaller

echo "🔨 Baue standalone Executable..."

pyinstaller --onedir \
    --name "AskOWi" \
    --windowed \
    --add-data "icons:icons" \
    --add-data "data:data" \
    --add-data "chroma:chroma" \
    --add-data "query_data.py:." \
    --add-data "get_embedding_function.py:." \
    --add-data "populate_database.py:." \
    --hidden-import=tiktoken_ext.openai_public \
    --hidden-import=tiktoken_ext \
    --collect-all chromadb \
    --collect-all langchain \
    --collect-all langchain_chroma \
    --collect-all langchain_ollama \
    --collect-all langchain_openai \
    gui.py

echo "✅ Build abgeschlossen!"
echo "📁 Die Anwendung befindet sich in: dist/AskOWi/"
echo "🚀 Starte mit: ./dist/AskOWi/AskOWi"

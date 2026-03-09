# Ask OWi - RAG Chatbot für Sondenentwöhnung

Ein KI-gestützter Chatbot zur Unterstützung von Eltern bei der Ernährungstherapie und Sondenentwöhnung von Kindern.

## 🚀 Schnellstart

### Option 1: Mit Python (empfohlen für Entwicklung)

**Voraussetzungen:**
- Python 3.10 oder höher
- LM Studio mit geladenem Modell (läuft auf `localhost:1234`)

**Linux/macOS:**
```bash
./start.sh
```

**Windows:**
```batch
start.bat
```

Das Skript:
1. Erstellt automatisch eine virtuelle Umgebung
2. Installiert alle Abhängigkeiten
3. Erstellt die Datenbank (beim ersten Start)
4. Startet die GUI

### Option 2: Standalone Executable (ohne Python)

**Build erstellen:**

Linux:
```bash
./build_linux.sh
```

Windows:
```batch
build_windows.bat
```

Nach dem Build findest du die Anwendung in `dist/AskOWi/`.

## 📁 Projektstruktur

```
├── gui.py                 # Hauptanwendung (GUI)
├── query_data.py          # RAG-Abfrage-Logik
├── populate_database.py   # Datenbank befüllen
├── get_embedding_function.py
├── start.sh / start.bat   # Start-Skripte
├── build_linux.sh / build_windows.bat  # Build-Skripte
├── data/                  # PDF-Dokumente
├── chroma/                # Vektor-Datenbank
└── icons/                 # GUI-Icons
```

## ⚙️ Konfiguration

Umgebungsvariablen (optional):
- `LMSTUDIO_API_BASE` - LM Studio API URL (Standard: `http://localhost:1234/v1`)
- `LMSTUDIO_CHAT_MODEL` - Chat-Modell (Standard: `qwen2.5-14b-instruct`)
- `LMSTUDIO_EMBEDDING_MODEL` - Embedding-Modell (Standard: `text-embedding-3-small`)

## 📦 Manuelle Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oder: venv\Scripts\activate  # Windows

pip install -r requirements.txt
python populate_database.py  # Datenbank erstellen
python gui.py                # GUI starten
```

## 🔧 Neue Dokumente hinzufügen

1. PDF-Dateien in den `data/` Ordner legen
2. Datenbank aktualisieren:
   ```bash
   python populate_database.py
   ```
   Oder mit Reset:
   ```bash
   python populate_database.py --reset
   ```

## 🧪 Testen

### Einzelne Frage testen

```bash
python query_data.py "Was ist Sondenentwöhnung?"
```

### Mehrere Fragen aus Datei testen

Erstelle eine `testprompts.txt` Datei mit Fragen (durch Leerzeilen getrennt):
```
Was ist Sondenentwöhnung?

Wie lange dauert der Prozess?

Welche Komplikationen können auftreten?
```

Dann ausführen:
```bash
python query_data.py --input-file testprompts.txt --output-file results.txt
```

Ergebnis: `results.txt` mit allen Antworten

### Mehrfache Test-Durchläufe (für Konsistenz-Tests)

Um zu prüfen, wie konsistent die Antworten des Chatbots sind, können mehrere Durchläufe durchgeführt werden:

```bash
python query_data.py --input-file testprompts.txt --output-file results.txt --iterations 5
```

Das erstellt:
- `results_1.txt` - Frage 1 mit 5 verschiedenen Antworten
- `results_2.txt` - Frage 2 mit 5 verschiedenen Antworten
- `results_3.txt` - Frage 3 mit 5 verschiedenen Antworten

Jede Datei zeigt alle Durchläufe für eine Frage, so dass du die Konsistenz der Antworten vergleichen kannst.

**Beispiel-Output:**
```
Frage 1:
Was ist Sondenentwöhnung?
============================================================

Durchlauf 1:
────────────────────────────────────────────────────────────
🤖 Assistant:
[Antwort 1]
(Quellen: datei.pdf)
Dauer: 12s
────────────────────────────────────────────────────────────

Durchlauf 2:
────────────────────────────────────────────────────────────
[Antwort 2]
...
```


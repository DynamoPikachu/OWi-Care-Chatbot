import time
import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import re
import os
import sys


QUERY_SCRIPT = "query_data.py"
PYTHON_EXECUTABLE = sys.executable


class RAGGui(tk.Tk):
    start_time = 0
    def __init__(self):
        super().__init__()

        self.title("RAG Chat")
        self.geometry("750x600")

        self._build_ui()

    def _build_ui(self):
        # Chat output
        self.chat = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input frame
        input_frame = ttk.Frame(self)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.query_entry = ttk.Entry(input_frame)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.query_entry.bind("<Return>", self.run_query)

        self.send_button = ttk.Button(
            input_frame, text="Senden", command=self.run_query
        )
        self.send_button.pack(side=tk.LEFT, padx=5)

        # Loading indicator
        self.progress = ttk.Progressbar(
            self, mode="indeterminate"
        )
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.progress.pack_forget()

    def run_query(self, event=None):
        global start_time
        start_time = time.time()
        query = self.query_entry.get().strip()
        if not query:
            return

        self.query_entry.delete(0, tk.END)
        self._append_chat(f"🧑 Du:\n{query}\n\n")

        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.progress.start()

        self.send_button.config(state=tk.DISABLED)

        thread = threading.Thread(
            target=self._execute_query, args=(query,), daemon=True
        )
        thread.start()

    def _execute_query(self, query):
        try:
            result = subprocess.run(
                [PYTHON_EXECUTABLE, QUERY_SCRIPT, query],
                capture_output=True,
                text=True,
            )
            output = result.stdout.encode("latin1", errors="ignore").decode("utf-8", errors="ignore") \
                     + result.stderr.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")

            response, sources = self._parse_output(output)

        except Exception as e:
            response = f"❌ Fehler:\n{e}"
            sources = []

        self.after(0, self._display_response, response, sources)

    def _parse_output(self, output: str):
        response = output
        sources = []

        # --- Response extrahieren ---
        response_match = re.search(
            r"Response:\s*(.*?)\s*Sources:",
            output,
            re.DOTALL | re.IGNORECASE,
        )

        if response_match:
            response = response_match.group(1).strip()

        # --- Sources extrahieren ---
        sources_match = re.search(
            r"Sources:\s*\[(.*?)\]",
            output,
            re.DOTALL | re.IGNORECASE,
        )

        if sources_match:
            raw_sources = sources_match.group(1)
            sources = sorted(set(re.findall(r"([^\\/]+\.pdf)", raw_sources)))

        return response, sources

    def _display_response(self, response, sources):
        self.progress.stop()
        self.progress.pack_forget()
        self.send_button.config(state=tk.NORMAL)

        self._append_chat("🤖 Assistant:\n")
        self._append_chat(response + "\n")

        if sources:
            self._append_chat(
                f"\n(Quellen: {', '.join(sorted(sources))})\n"
            )

        global start_time
        self._append_chat("\n" + f"Dauer: {int(time.time() - start_time)}s")
        self._append_chat("\n" + "─" * 60 + "\n\n")

    def _append_chat(self, text):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)


if __name__ == "__main__":
    if not os.path.exists(QUERY_SCRIPT):
        raise FileNotFoundError(f"{QUERY_SCRIPT} nicht gefunden")

    app = RAGGui()
    app.mainloop()

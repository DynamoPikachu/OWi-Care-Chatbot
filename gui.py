import time
import tkinter as tk
import subprocess
import threading
import re
import os
import sys
import json


QUERY_SCRIPT = "query_data.py"
PYTHON_EXECUTABLE = sys.executable


class RAGGui(tk.Tk):
    start_time = 0
    def __init__(self):
        super().__init__()

        self.title("Ask OWi")
        self.geometry("430x760")
        self.minsize(360, 640)

        self.chat_history = []  # Speichert den Chatverlauf als Liste von {"role": ..., "content": ...}

        self.logo_img = "icons/logo.png"
        self.avatar_img = "icons/logo.png"
        self.cam_img = None
        self.send_img = None
        self.typing_frames_raw = []
        self.typing_frames = []
        self.typing_frame_index = 0
        self.typing_job = None
        self.typing_label = None
        self.typing_image_id = None
        self.typing_widget = None
        self._load_images()

        self._build_ui()

    def _build_ui(self):
        self.configure(bg="#f6f4f1")

        header = tk.Frame(self, bg="#f6f4f1")
        header.pack(fill=tk.X, pady=(18, 8))

        if self.logo_img:
            logo_label = tk.Label(header, image=self.logo_img, bg="#f6f4f1")
            logo_label.pack()
        else:
            self._header_fallback_logo(header)

        name = tk.Label(
            header,
            text="Ask OWi",
            font=("Segoe UI", 18, "bold"),
            fg="#5b5a56",
            bg="#f6f4f1",
        )
        name.pack(pady=(8, 0))

        status = tk.Label(
            header,
            text="Online",
            font=("Segoe UI", 10),
            fg="#8c8a86",
            bg="#f6f4f1",
        )
        status.pack()

        time_label = tk.Label(
            header,
            text=self._format_time(),
            font=("Segoe UI", 9),
            fg="#9f9c97",
            bg="#f6f4f1",
        )
        time_label.pack(pady=(10, 0))

        # Messages area (scrollable)
        self._build_messages_area()

        # Input area
        input_wrap = tk.Frame(self, bg="#f6f4f1")
        input_wrap.pack(fill=tk.X, padx=16, pady=(6, 14))

        cam_btn = tk.Button(
            input_wrap,
            text="Cam" if not self.cam_img else "",
            image=self.cam_img if self.cam_img else None,
            font=("Segoe UI", 9, "bold"),
            fg="#b7b4ae",
            bg="#efede9",
            activebackground="#e5e2dc",
            bd=0,
            relief=tk.FLAT,
            width=36,
            height=36,
        )
        cam_btn.pack(side=tk.LEFT, padx=(0, 8))

        entry_frame = tk.Frame(input_wrap, bg="#efede9")
        entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.query_entry = tk.Entry(
            entry_frame,
            font=("Segoe UI", 11),
            fg="#a6a39d",
            bg="#efede9",
            relief=tk.FLAT,
            insertbackground="#6a6762",
        )
        self.query_entry.pack(fill=tk.BOTH, padx=12, pady=10)
        self.query_entry.insert(0, "Enter your message...")
        self.query_entry.bind("<FocusIn>", self._clear_placeholder)
        self.query_entry.bind("<FocusOut>", self._restore_placeholder)
        self.query_entry.bind("<Return>", self.run_query)

        self.send_button = tk.Button(
            input_wrap,
            text=">" if not self.send_img else "",
            image=self.send_img if self.send_img else None,
            font=("Segoe UI", 12, "bold"),
            fg="#b7b4ae",
            bg="#efede9",
            activebackground="#e5e2dc",
            bd=0,
            relief=tk.FLAT,
            width=36,
            height=36,
            command=self.run_query,
        )
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))

        self.after(0, self._prepare_typing_frames_from_entry)

    def run_query(self, event=None):
        global start_time
        start_time = time.time()
        query = self.query_entry.get().strip()
        if not query or query == "Enter your message...":
            return

        self.query_entry.delete(0, tk.END)
        self._add_message(query, side="right")
        
        # Füge User-Nachricht zum Chatverlauf hinzu
        self.chat_history.append({"role": "user", "content": query})
        
        self._show_typing()

        self.send_button.config(state=tk.DISABLED)

        thread = threading.Thread(
            target=self._execute_query, args=(query,), daemon=True
        )
        thread.start()

    def _execute_query(self, query):
        try:
            # Übergebe den Chatverlauf als JSON-String via --history Argument
            history_json = json.dumps(self.chat_history[:-1], ensure_ascii=False)  # Ohne die aktuelle Nachricht
            result = subprocess.run(
                [PYTHON_EXECUTABLE, QUERY_SCRIPT, query, "--history", history_json],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output = (result.stdout or "") + (result.stderr or "")

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
        # Suche nach Sources: [...] im Output
        sources_match = re.search(
            r"Sources:\s*\[([^\]]*)\]",
            output,
            re.DOTALL | re.IGNORECASE,
        )

        if sources_match:
            raw_sources = sources_match.group(1)
            # Finde alle Einträge zwischen Anführungszeichen
            all_entries = re.findall(r"'([^']+)'", raw_sources)
            # Extrahiere nur den Dateinamen ohne Pfad
            for entry in all_entries:
                # Entferne Pfad (alles vor dem letzten / oder \)
                filename = entry.split("/")[-1].split("\\")[-1]
                # Entferne Chunk-IDs (z.B. :1:2 am Ende)
                filename = filename.split(":")[0]
                if filename.lower().endswith(".pdf"):
                    sources.append(filename)
            sources = sorted(set(sources))

        return response, sources

    def _display_response(self, response, sources):
        self._hide_typing()
        self.send_button.config(state=tk.NORMAL)

        # Füge Assistant-Antwort zum Chatverlauf hinzu
        self.chat_history.append({"role": "assistant", "content": response})

        self._add_message(response, side="left", sources=sources)

    def _build_messages_area(self):
        container = tk.Frame(self, bg="#f6f4f1")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self.canvas = tk.Canvas(container, bg="#f6f4f1", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(container, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.messages_frame = tk.Frame(self.canvas, bg="#f6f4f1")
        self.canvas.create_window((0, 0), window=self.messages_frame, anchor="nw")

        self.messages_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _add_message(self, text, side="left", sources=None):
        outer = tk.Frame(self.messages_frame, bg="#f6f4f1")
        outer.pack(fill=tk.X, pady=6)

        if side == "left":
            row = tk.Frame(outer, bg="#f6f4f1")
            row.pack(anchor="w")

            if self.avatar_img:
                avatar = tk.Label(row, image=self.avatar_img, bg="#f6f4f1")
                avatar.pack(side=tk.LEFT, padx=(6, 8))
            else:
                self._avatar_fallback(row)

            bubble = self._rounded_bubble(
                row,
                text=text,
                wraplength=260,
                font=("Segoe UI", 11),
                fg="#5f5b55",
                bg="#e9e4d8",
            )
            bubble.pack(side=tk.LEFT, anchor="w")

            if sources:
                src_label = tk.Label(
                    outer,
                    text="Quellen: " + ", ".join(sorted(sources)),
                    font=("Segoe UI", 8),
                    fg="#9f9c97",
                    bg="#f6f4f1",
                    wraplength=300,
                    justify=tk.LEFT,
                )
                src_label.pack(anchor="w", padx=56, pady=(4, 0))
        else:
            # Rechte Sprechblase - am rechten Rand positioniert
            row = tk.Frame(outer, bg="#f6f4f1")
            row.pack(anchor="e", fill=tk.X)

            bubble = self._rounded_bubble(
                row,
                text=text,
                wraplength=260,
                font=("Segoe UI", 11),
                fg="#ffffff",
                bg="#7a9ec9",
            )
            bubble.pack(side=tk.RIGHT, anchor="e", padx=(50, 6))

        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _show_typing(self):
        self.typing_widget = tk.Frame(self.messages_frame, bg="#f6f4f1")
        self.typing_widget.pack(fill=tk.X, pady=6)

        row = tk.Frame(self.typing_widget, bg="#f6f4f1")
        row.pack(anchor="w")

        if self.avatar_img:
            avatar = tk.Label(row, image=self.avatar_img, bg="#f6f4f1")
            avatar.pack(side=tk.LEFT, padx=(6, 8))
        else:
            self._avatar_fallback(row)

        if self.typing_frames:
            self.typing_frame_index = 0
            bubble, image_id = self._rounded_image_bubble(
                row,
                image=self.typing_frames[0],
                bg="#e9e4d8",
            )
            bubble.pack(side=tk.LEFT, anchor="w")
            self.typing_label = bubble
            self.typing_image_id = image_id
            self._animate_typing()
        else:
            bubble = self._rounded_bubble(
                row,
                text="...",
                wraplength=80,
                font=("Segoe UI", 14, "bold"),
                fg="#7b7873",
                bg="#e9e4d8",
            )
            bubble.pack(side=tk.LEFT, anchor="w")

        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _hide_typing(self):
        if self.typing_job:
            self.after_cancel(self.typing_job)
            self.typing_job = None
        self.typing_label = None
        self.typing_image_id = None
        if self.typing_widget:
            self.typing_widget.destroy()
            self.typing_widget = None

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _format_time(self):
        return time.strftime("%b %d, %I:%M %p").lower()

    def _strip_markdown(self, text: str) -> str:
        """Entfernt Markdown-Formatierung aus dem Text."""
        # Fett: **text** oder __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        # Kursiv: *text* oder _text_
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
        # Code: `text`
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Überschriften: # text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Links: [text](url)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        return text

    def _rounded_bubble(self, parent, text, wraplength, font, fg, bg):
        # Entferne Markdown-Formatierung
        text = self._strip_markdown(text)
        
        pad_x = 14
        pad_y = 10
        radius = 16

        canvas = tk.Canvas(parent, bg=parent.cget("bg"), highlightthickness=0)
        text_id = canvas.create_text(
            pad_x,
            pad_y,
            text=text,
            font=font,
            fill=fg,
            anchor="nw",
            width=wraplength,
        )
        bbox = canvas.bbox(text_id)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        width = text_w + pad_x * 2
        height = text_h + pad_y * 2

        self._rounded_rect(canvas, 0, 0, width, height, radius, fill=bg, outline=bg)
        canvas.tag_raise(text_id)
        canvas.config(width=width, height=height)
        return canvas

    def _rounded_rect(self, canvas, x1, y1, x2, y2, r, fill, outline):
        r = max(2, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        canvas.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline=outline)
        canvas.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline=outline)
        canvas.create_oval(x1, y1, x1 + 2 * r, y1 + 2 * r, fill=fill, outline=outline)
        canvas.create_oval(x2 - 2 * r, y1, x2, y1 + 2 * r, fill=fill, outline=outline)
        canvas.create_oval(x1, y2 - 2 * r, x1 + 2 * r, y2, fill=fill, outline=outline)
        canvas.create_oval(x2 - 2 * r, y2 - 2 * r, x2, y2, fill=fill, outline=outline)

    def _rounded_image_bubble(self, parent, image, bg):
        pad = 10
        radius = 16
        width = image.width() + pad * 2
        height = image.height() + pad * 2

        canvas = tk.Canvas(parent, bg=parent.cget("bg"), highlightthickness=0)
        self._rounded_rect(canvas, 0, 0, width, height, radius, fill=bg, outline=bg)
        image_id = canvas.create_image(width // 2, height // 2, image=image)
        canvas.config(width=width, height=height)
        return canvas, image_id

    def _load_gif_frames(self, path):
        frames = []
        idx = 0
        while True:
            try:
                frame = tk.PhotoImage(file=path, format=f"gif -index {idx}")
            except tk.TclError:
                break
            frames.append(frame)
            idx += 1
        return frames

    def _scale_gif_frames(self, frames, target_height):
        if not frames:
            return []
        first = frames[0]
        factor = max(1, int(round(first.height() / float(target_height))))
        if factor <= 1:
            return frames
        return [frame.subsample(factor, factor) for frame in frames]

    def _prepare_typing_frames_from_entry(self):
        if not self.typing_frames_raw:
            return
        self.update_idletasks()
        target_height = max(16, int((self.query_entry.winfo_height() - 12) * 3))
        self.typing_frames = self._scale_gif_frames(
            self.typing_frames_raw,
            target_height,
        )

    def _animate_typing(self):
        if not self.typing_label or not self.typing_frames:
            return
        self.typing_frame_index = (self.typing_frame_index + 1) % len(self.typing_frames)
        self.typing_label.itemconfig(
            self.typing_image_id,
            image=self.typing_frames[self.typing_frame_index],
        )
        self.typing_job = self.after(120, self._animate_typing)

    def _scale_image(self, img, target):
        factor = max(1, int(round(img.width() / float(target))))
        return img.subsample(factor, factor)

    def _load_images(self):
        logo_path = "logo.png"
        if os.path.exists(os.path.join("icons", "logo.png")):
            logo_path = os.path.join("icons", "logo.png")
        if os.path.exists(logo_path):
            try:
                img = tk.PhotoImage(file=logo_path)
                self.logo_img = self._scale_image(img, 120)
                self.avatar_img = self._scale_image(img, 32)
            except tk.TclError:
                self.logo_img = None
                self.avatar_img = None
        if os.path.exists(os.path.join("icons", "cam.png")):
            try:
                img = tk.PhotoImage(file=os.path.join("icons", "cam.png"))
                self.cam_img = self._scale_image(img, 24)
            except tk.TclError:
                self.cam_img = None
        if os.path.exists(os.path.join("icons", "sent.png")):
            try:
                img = tk.PhotoImage(file=os.path.join("icons", "sent.png"))
                self.send_img = self._scale_image(img, 24)
            except tk.TclError:
                self.send_img = None
        if os.path.exists(os.path.join("icons", "loader_dots.gif")):
            self.typing_frames_raw = self._load_gif_frames(
                os.path.join("icons", "loader_dots.gif")
            )

    def _header_fallback_logo(self, parent):
        canvas = tk.Canvas(parent, width=96, height=96, bg="#f6f4f1", highlightthickness=0)
        canvas.pack()
        canvas.create_oval(6, 6, 90, 90, fill="#8aa7e6", outline="#8aa7e6")
        canvas.create_text(48, 48, text="OW", fill="white", font=("Segoe UI", 18, "bold"))

    def _avatar_fallback(self, parent):
        canvas = tk.Canvas(parent, width=32, height=32, bg="#f6f4f1", highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=(6, 8))
        canvas.create_oval(2, 2, 30, 30, fill="#8aa7e6", outline="#8aa7e6")
        canvas.create_text(16, 16, text="O", fill="white", font=("Segoe UI", 10, "bold"))

    def _clear_placeholder(self, event):
        if self.query_entry.get() == "Enter your message...":
            self.query_entry.delete(0, tk.END)
            self.query_entry.config(fg="#6a6762")

    def _restore_placeholder(self, event):
        if not self.query_entry.get().strip():
            self.query_entry.insert(0, "Enter your message...")
            self.query_entry.config(fg="#a6a39d")


if __name__ == "__main__":
    if not os.path.exists(QUERY_SCRIPT):
        raise FileNotFoundError(f"{QUERY_SCRIPT} nicht gefunden")

    app = RAGGui()
    app.mainloop()

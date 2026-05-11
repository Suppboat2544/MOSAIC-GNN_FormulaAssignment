"""
MOSAIC-GNN Desktop Application
A standalone Tkinter GUI that talks to the FastAPI backend.
Starts the backend automatically if it is not already running.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import (
    BooleanVar, DoubleVar, END, Event, Frame, IntVar,
    StringVar, Text, Tk, Toplevel, ttk, filedialog, messagebox,
)
import tkinter as tk

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]          # Formula_assignment/
BACKEND_DIR = ROOT / "platform" / "backend"
VENV_PY     = ROOT / ".venv" / "bin" / "python"
PYTHON      = str(VENV_PY) if VENV_PY.exists() else sys.executable
BACKEND_URL = "http://127.0.0.1:8000"
API         = f"{BACKEND_URL}/api/v1"

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":       "#0f172a",
    "surface":  "#1e293b",
    "surface2": "#172032",
    "border":   "#334155",
    "accent":   "#7c3aed",
    "accent2":  "#9333ea",
    "text":     "#f1f5f9",
    "muted":    "#94a3b8",
    "dim":      "#475569",
    "success":  "#34d399",
    "error":    "#f87171",
    "warn":     "#fbbf24",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_requests():
    """Import requests lazily (it is in the venv)."""
    try:
        import requests
        return requests
    except ImportError:
        return None


def _backend_alive() -> bool:
    r = _ensure_requests()
    if r is None:
        return False
    try:
        r.get(f"{BACKEND_URL}/health", timeout=2)
        return True
    except Exception:
        return False


def _start_backend() -> subprocess.Popen | None:
    """Spin up the FastAPI server as a subprocess."""
    cmd = [
        PYTHON, "-m", "uvicorn",
        "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except Exception as exc:
        print(f"[desktop] could not start backend: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Styled widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def label(parent, text, size=11, weight="normal", color=None, **kw):
    return tk.Label(
        parent, text=text,
        font=("Helvetica", size, weight),
        fg=color or C["text"],
        bg=kw.pop("bg", C["bg"]),
        **kw,
    )


def btn(parent, text, command=None, accent=True, width=12, **kw):
    bg = C["accent"] if accent else C["surface"]
    fg = C["text"]
    b = tk.Button(
        parent, text=text, command=command,
        font=("Helvetica", 10, "bold"),
        bg=bg, fg=fg,
        activebackground=C["accent2"],
        activeforeground=C["text"],
        relief="flat", cursor="hand2",
        padx=14, pady=7, width=width,
        borderwidth=0,
        **kw,
    )
    return b


def entry(parent, textvariable=None, placeholder="", width=30, **kw):
    e = tk.Entry(
        parent,
        textvariable=textvariable,
        font=("Helvetica", 10),
        bg=C["surface"],
        fg=C["text"],
        insertbackground=C["text"],
        relief="flat",
        bd=6,
        width=width,
        **kw,
    )
    if placeholder and textvariable is None:
        e.insert(0, placeholder)
        e.config(fg=C["dim"])
        def _on_focus_in(event):
            if e.get() == placeholder:
                e.delete(0, END)
                e.config(fg=C["text"])
        def _on_focus_out(event):
            if not e.get():
                e.insert(0, placeholder)
                e.config(fg=C["dim"])
        e.bind("<FocusIn>",  _on_focus_in)
        e.bind("<FocusOut>", _on_focus_out)
    return e


def card(parent, **kw):
    return tk.Frame(parent, bg=C["surface"], bd=0, highlightbackground=C["border"],
                    highlightthickness=1, **kw)


def separator(parent, padx=10, pady=0):
    tk.Frame(parent, height=1, bg=C["border"]).pack(fill="x", padx=padx, pady=pady)


def _style_ax(ax):
    """Apply the dark-theme style to a matplotlib Axes object."""
    GRID = "#334155"
    TEXT = "#94a3b8"
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.set_xticks([])
    ax.set_yticks([])


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────────────────────────

class MosaicApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("MOSAIC-GNN  ·  Formula Assignment Platform")
        self.configure(bg=C["bg"])
        self.geometry("1100x720")
        self.minsize(900, 600)

        # state
        self._job_id   = StringVar()
        self._filename = StringVar()
        self._model    = StringVar(value="Model10")
        self._sample_type = StringVar(value="DOM")
        self._mz_min   = StringVar()
        self._mz_max   = StringVar()
        self._ion_mode = StringVar(value="both")
        self._mz_col   = StringVar()   # user-selected m/z column
        self._int_col  = StringVar()   # user-selected intensity column
        self._col_names: list[str] = []  # columns from uploaded file
        self._status   = StringVar(value="Idle")
        self._backend_proc: subprocess.Popen | None = None
        self._results: list[dict] = []

        self._setup_style()
        self._build_ui()
        self._ensure_backend_thread()

    # ── ttk style (for Notebook, Treeview, Progressbar) ──────────────────────

    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TProgressbar",
                    troughcolor=C["surface"],
                    background=C["accent"],
                    thickness=6)

        s.configure("Treeview",
                    background=C["surface"],
                    foreground=C["text"],
                    rowheight=28,
                    fieldbackground=C["surface"],
                    borderwidth=0)
        s.configure("Treeview.Heading",
                    background=C["surface2"],
                    foreground=C["muted"],
                    font=("Helvetica", 9, "bold"),
                    relief="flat")
        s.map("Treeview",
              background=[("selected", C["accent"])],
              foreground=[("selected", C["text"])])

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── 1. Top bar ───────────────────────────────────────────────────────
        top = tk.Frame(self, bg=C["surface"], height=54)
        top.pack(fill="x")
        top.pack_propagate(False)

        logo = tk.Frame(top, bg=C["accent"], width=32, height=32)
        logo.place(relx=0, rely=0.5, anchor="w", x=16)
        label(logo, "M", size=13, weight="bold", bg=C["accent"]).place(
            relx=0.5, rely=0.5, anchor="center")

        label(top, "MOSAIC-GNN", size=13, weight="bold",
              bg=C["surface"]).place(x=58, rely=0.5, anchor="w")
        label(top, "Formula Assignment Platform", size=9,
              color=C["muted"], bg=C["surface"]).place(x=175, rely=0.5, anchor="w")

        tk.Frame(self, height=1, bg=C["border"]).pack(fill="x")

        # ── 2. Body: sidebar + content area ──────────────────────────────────
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = tk.Frame(body, bg=C["surface2"], width=152)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Frame(sidebar, height=14, bg=C["surface2"]).pack()
        tk.Label(sidebar, text="WORKFLOW", font=("Helvetica", 7, "bold"),
                 bg=C["surface2"], fg=C["dim"]).pack(anchor="w", padx=14)
        tk.Frame(sidebar, height=6, bg=C["surface2"]).pack()

        self._nav_frames: dict = {}
        for page_name, num, title, sub in [
            ("upload",  "1", "Upload",  "Add data"),
            ("predict", "2", "Predict", "Run model"),
            ("results", "3", "Results", "Analyse"),
        ]:
            self._nav_frames[page_name] = self._build_nav_item(
                sidebar, num, title, sub, page_name)

        # Spacer
        tk.Frame(sidebar, bg=C["surface2"]).pack(fill="both", expand=True)

        # Backend status footer
        tk.Frame(sidebar, height=1, bg=C["border"]).pack(fill="x")
        footer = tk.Frame(sidebar, bg=C["surface2"], pady=8)
        footer.pack(fill="x")
        self._status_lbl = tk.Label(
            footer, text="⬤  Connecting…",
            font=("Helvetica", 8), bg=C["surface2"], fg=C["warn"],
            anchor="w", wraplength=132,
        )
        self._status_lbl.pack(anchor="w", padx=12)
        tk.Label(footer, text="v1.0.0", font=("Helvetica", 7),
                 bg=C["surface2"], fg=C["dim"]).pack(anchor="w", padx=12, pady=(2, 0))

        # Vertical divider
        tk.Frame(body, width=1, bg=C["border"]).pack(side="left", fill="y")

        # ── Content area ─────────────────────────────────────────────────────
        content_area = tk.Frame(body, bg=C["bg"])
        content_area.pack(side="left", fill="both", expand=True)

        self._page_upload  = tk.Frame(content_area, bg=C["bg"])
        self._page_predict = tk.Frame(content_area, bg=C["bg"])
        self._page_results = tk.Frame(content_area, bg=C["bg"])
        self._pages = {
            "upload":  self._page_upload,
            "predict": self._page_predict,
            "results": self._page_results,
        }

        self._build_upload_tab(self._page_upload)
        self._build_predict_tab(self._page_predict)
        self._build_results_tab(self._page_results)

        # ── 3. Status bar ─────────────────────────────────────────────────────
        tk.Frame(self, height=1, bg=C["border"]).pack(fill="x")
        sb = tk.Frame(self, bg=C["surface2"], height=26)
        sb.pack(fill="x")
        sb.pack_propagate(False)
        self._sb_lbl = label(sb, "Ready", size=9, color=C["dim"], bg=C["surface2"])
        self._sb_lbl.pack(side="left", padx=12)
        self._pbar = ttk.Progressbar(
            sb, style="TProgressbar", mode="indeterminate", length=120)
        self._pbar.pack(side="right", padx=12, pady=5)

        # Show first page
        self._show_page("upload")

    # ── Sidebar navigation helpers ────────────────────────────────────────────

    def _build_nav_item(self, parent, num: str, title: str, sub: str, page_name: str):
        """Build one sidebar nav item. Returns a tuple used by _set_nav_active."""
        BG = C["surface2"]
        outer = tk.Frame(parent, bg=BG, cursor="hand2")
        outer.pack(fill="x")

        accent = tk.Frame(outer, width=3, bg=BG)
        accent.pack(side="left", fill="y")

        content = tk.Frame(outer, bg=BG, padx=10, pady=10)
        content.pack(side="left", fill="both", expand=True)

        row = tk.Frame(content, bg=BG)
        row.pack(anchor="w")

        badge = tk.Canvas(row, width=24, height=24, bg=BG, highlightthickness=0)
        badge.pack(side="left", padx=(0, 9))
        badge.create_oval(1, 1, 23, 23, fill=C["dim"], outline="")
        badge.create_text(12, 12, text=num, fill="white",
                          font=("Helvetica", 8, "bold"))

        name_lbl = tk.Label(row, text=title, font=("Helvetica", 10, "bold"),
                            bg=BG, fg=C["muted"])
        name_lbl.pack(side="left")

        sub_lbl = tk.Label(content, text=sub, font=("Helvetica", 8),
                           bg=BG, fg=C["dim"])
        sub_lbl.pack(anchor="w", padx=(33, 0))

        bg_widgets = [outer, content, row, name_lbl, sub_lbl]

        def _click(e, n=page_name):
            self._show_page(n)
        for w in [outer, content, row, badge, name_lbl, sub_lbl]:
            w.bind("<Button-1>", _click)

        return outer, badge, accent, bg_widgets, name_lbl, sub_lbl

    def _show_page(self, name: str):
        """Show the named content page and hide the others."""
        for n, frame in self._pages.items():
            if n == name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        self._set_nav_active(name)

    def _set_nav_active(self, name: str):
        """Update sidebar visual state to highlight the active page."""
        nav_order = ["upload", "predict", "results"]
        for n, (outer, badge, accent, bg_widgets, name_lbl, sub_lbl) in self._nav_frames.items():
            active = n == name
            bg = C["surface"] if active else C["surface2"]
            for w in bg_widgets:
                try:
                    w.config(bg=bg)
                except Exception:
                    pass
            badge.config(bg=bg)
            badge.delete("all")
            idx = nav_order.index(n)
            badge.create_oval(1, 1, 23, 23,
                              fill=C["accent"] if active else C["dim"],
                              outline="")
            badge.create_text(12, 12, text=str(idx + 1), fill="white",
                              font=("Helvetica", 8, "bold"))
            accent.config(bg=C["accent"] if active else C["surface2"])
            name_lbl.config(fg=C["text"] if active else C["muted"])
            sub_lbl.config(fg=C["accent"] if active else C["dim"])

    # ── Upload tab ────────────────────────────────────────────────────────────


    def _build_upload_tab(self, tab):
        # ── Page header ───────────────────────────────────────────────────────
        header = tk.Frame(tab, bg=C["bg"])
        header.pack(fill="x", padx=36, pady=(24, 0))

        tk.Label(header, text=" Step 1 of 3 ", font=("Helvetica", 8, "bold"),
                 bg=C["accent"], fg="white", padx=6, pady=2).pack(anchor="w", pady=(0, 8))
        label(header, "Upload Mass-Spec Data", size=15, weight="bold").pack(anchor="w")
        label(header, "Select a CSV, XLSX, mzML, or similar file to begin formula assignment.",
              size=9, color=C["muted"]).pack(anchor="w", pady=(2, 0))

        tk.Frame(tab, height=1, bg=C["border"]).pack(fill="x", pady=(20, 0))

        # ── Content ───────────────────────────────────────────────────────────
        outer = tk.Frame(tab, bg=C["bg"])
        outer.pack(fill="x", padx=36, pady=20)

        # Drop-zone card
        dz = card(outer)
        dz.pack(fill="x", pady=(0, 12))

        dz_body = tk.Frame(dz, bg=C["surface"], pady=36)
        dz_body.pack(fill="x")

        icon_box = tk.Frame(dz_body, bg=C["surface2"],
                            highlightbackground=C["border"], highlightthickness=1,
                            width=52, height=52)
        icon_box.pack(pady=(0, 14))
        icon_box.pack_propagate(False)
        tk.Label(icon_box, text="⬆", font=("Helvetica", 20),
                 bg=C["surface2"], fg=C["muted"]).place(relx=0.5, rely=0.5, anchor="center")

        label(dz_body, "Click to browse a file", size=12, weight="bold",
              bg=C["surface"]).pack()
        label(dz_body, "CSV  ·  XLSX  ·  TSV  ·  TXT  ·  mzML  ·  mzXML",
              size=9, color=C["muted"], bg=C["surface"]).pack(pady=(4, 2))
        label(dz_body, "Max 200 MB  ·  one file at a time",
              size=8, color=C["dim"], bg=C["surface"]).pack()

        def _browse():
            path = filedialog.askopenfilename(
                title="Select mass-spec file",
                filetypes=[
                    ("All supported", "*.csv *.xlsx *.xls *.tsv *.txt *.mzml *.mzxml"),
                    ("CSV",   "*.csv"),
                    ("Excel", "*.xlsx *.xls"),
                    ("TSV",   "*.tsv"),
                    ("TXT",   "*.txt"),
                    ("mzML",  "*.mzml"),
                    ("mzXML", "*.mzxml"),
                ],
            )
            if path:
                self._upload_file(path)

        for w in [dz, dz_body] + list(dz_body.winfo_children()):
            w.bind("<Button-1>", lambda e: _browse())
        dz.config(cursor="hand2")
        dz.bind("<Enter>", lambda e: dz.config(highlightbackground=C["accent"]))
        dz.bind("<Leave>", lambda e: dz.config(highlightbackground=C["border"]))

        btn(outer, "  Browse & Upload  ", command=_browse, width=22).pack(
            anchor="w", pady=(0, 14))

        tk.Frame(outer, height=1, bg=C["border"]).pack(fill="x", pady=(0, 12))

        # File info card
        self._file_card = card(outer)
        self._file_card.pack(fill="x")
        info_body = tk.Frame(self._file_card, bg=C["surface"], padx=14, pady=12)
        info_body.pack(fill="x")
        self._file_info = tk.Label(
            info_body,
            text="No file selected — choose a file above to get started.",
            font=("Helvetica", 9), bg=C["surface"], fg=C["dim"], anchor="w",
        )
        self._file_info.pack(anchor="w")

    # ── Predict tab ───────────────────────────────────────────────────────────

    def _build_predict_tab(self, tab):
        # Scrollable canvas ─────────────────────────────────────────────────
        canvas = tk.Canvas(tab, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        scroll_frame = tk.Frame(canvas, bg=C["bg"])
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind("<Configure>", _on_frame_configure)

        def _on_canvas_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(e):
            canvas.yview_scroll(-1 * int(e.delta / 120), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Inner content ──────────────────────────────────────────────────────
        content = tk.Frame(scroll_frame, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=36, pady=24)

        # Header
        hdr = tk.Frame(content, bg=C["bg"])
        hdr.pack(fill="x", pady=(0, 20))
        tk.Label(hdr, text=" Step 2 of 3 ", font=("Helvetica", 8, "bold"),
                 bg=C["accent"], fg="white", padx=6, pady=2).pack(anchor="w", pady=(0, 8))
        label(hdr, "Configure Prediction", size=15, weight="bold").pack(anchor="w")
        label(hdr, "Choose model and parameters, then run MOSAIC-GNN inference.",
              size=9, color=C["muted"]).pack(anchor="w", pady=(2, 0))
        tk.Frame(content, height=1, bg=C["border"]).pack(fill="x", pady=(0, 20))

        MODEL_INFO = {
            "Model10":  ("DOM / NOM",  "14-dim FCL · 13 relations", "Best for dissolved / natural organic matter."),
            "Model11b": ("PFAS",       "17-dim · bypass · fine-tuned", "Best for fluorinated PFAS/AFFF compounds."),
        }

        # ── Sample type selector ─────────────────────────────────────────────
        label(content, "Sample Type", size=10, weight="bold").pack(anchor="w", pady=(0, 6))
        label(content, "Choose DOM for environmental samples or PFAS for fluorinated compounds.",
              size=8, color=C["dim"]).pack(anchor="w", pady=(0, 8))

        st_frame = tk.Frame(content, bg=C["bg"])
        st_frame.pack(fill="x")
        self._sample_type_btns: dict[str, tk.Frame] = {}

        for i, (stype, sname, sdesc) in enumerate([
            ("DOM",  "DOM  —  Environmental / Marine",
             "Dissolved organic matter. Routes to Model 10 (97.68% Top-1)."),
            ("PFAS", "PFAS  —  Fluorinated Compounds",
             "Per- & polyfluoroalkyl substances / AFFF. Routes to Model 11b (78.9% Top-1)."),
        ]):
            is_dom = stype == "DOM"
            accent_col = "#22d3ee" if is_dom else C["warn"]
            st_col = tk.Frame(
                st_frame, bg=C["surface"],
                highlightbackground=accent_col if is_dom else C["border"],
                highlightthickness=1,
                padx=12, pady=10,
            )
            st_col.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            st_frame.columnconfigure(i, weight=1)

            tk.Label(st_col, text=sname, font=("Helvetica", 9, "bold"),
                     bg=C["surface"], fg=C["text"]).pack(anchor="w")
            tk.Label(st_col, text=sdesc, font=("Helvetica", 8),
                     bg=C["surface"], fg=C["muted"], wraplength=200, justify="left").pack(anchor="w", pady=(4, 0))

            st_col.bind("<Button-1>", lambda e, s=stype: self._select_sample_type(s))
            for child in st_col.winfo_children():
                child.bind("<Button-1>", lambda e, s=stype: self._select_sample_type(s))
            st_col.config(cursor="hand2")
            self._sample_type_btns[stype] = st_col

        # ── Model (auto-selected by sample type) ──────────────────────────────
        tk.Frame(content, height=14, bg=C["bg"]).pack()
        label(content, "Model", size=10, weight="bold").pack(anchor="w", pady=(0, 4))
        self._model_info_lbl = tk.Label(
            content,
            text="\u25b6  Model 10  \u00b7  14-dim FCL  \u00b7  best for DOM / NOM",
            font=("Helvetica", 9, "bold"), bg=C["bg"], fg="#22d3ee", anchor="w",
        )
        self._model_info_lbl.pack(anchor="w", pady=(0, 2))
        tk.Label(
            content,
            text="The best-performing model for each sample type is selected automatically.",
            font=("Helvetica", 8), bg=C["bg"], fg=C["dim"], anchor="w",
        ).pack(anchor="w")
        self._model_btns = {}  # no UI cards; kept for compatibility

        # ── m/z range ────────────────────────────────────────────────────────
        tk.Frame(content, height=18, bg=C["bg"]).pack()
        label(content, "m/z Range  (optional)", size=10, weight="bold").pack(anchor="w", pady=(0, 6))
        rng = tk.Frame(content, bg=C["bg"])
        rng.pack(anchor="w")
        entry(rng, textvariable=self._mz_min, placeholder="Min  e.g. 100", width=18).pack(side="left")
        label(rng, "  —  ", color=C["dim"]).pack(side="left")
        entry(rng, textvariable=self._mz_max, placeholder="Max  e.g. 800", width=18).pack(side="left")

        # ── Ion mode ─────────────────────────────────────────────────────────
        tk.Frame(content, height=14, bg=C["bg"]).pack()
        label(content, "Ion Mode", size=10, weight="bold").pack(anchor="w", pady=(0, 6))
        ion_frame = tk.Frame(content, bg=C["bg"])
        ion_frame.pack(anchor="w")
        self._ion_btns = {}
        for mode in ["both", "positive", "negative"]:
            b = tk.Button(
                ion_frame, text=mode.capitalize(),
                font=("Helvetica", 9, "bold"),
                bg=C["accent"] if mode == "both" else C["surface"],
                fg=C["text"],
                relief="flat", cursor="hand2", padx=16, pady=6,
                borderwidth=0, highlightthickness=1,
                highlightbackground=C["accent"] if mode == "both" else C["border"],
                command=lambda m=mode: self._select_ion(m),
            )
            b.pack(side="left", padx=(0, 6))
            self._ion_btns[mode] = b

        # ── Column selector ───────────────────────────────────────────────────
        tk.Frame(content, height=14, bg=C["bg"]).pack()
        label(content, "Input Columns", size=10, weight="bold").pack(anchor="w", pady=(0, 4))
        label(content, "Select which columns contain m/z values and intensity.",
              size=8, color=C["dim"]).pack(anchor="w", pady=(0, 8))

        col_frame = tk.Frame(content, bg=C["bg"])
        col_frame.pack(fill="x")
        label(col_frame, "m/z column:", size=9, color=C["muted"], bg=C["bg"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        self._mz_col_menu = ttk.Combobox(
            col_frame, textvariable=self._mz_col, state="readonly", width=30)
        self._mz_col_menu.grid(row=0, column=1, sticky="w", pady=3)

        label(col_frame, "Intensity column:", size=9, color=C["muted"], bg=C["bg"]).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        self._int_col_menu = ttk.Combobox(
            col_frame, textvariable=self._int_col, state="readonly", width=30)
        self._int_col_menu.grid(row=1, column=1, sticky="w", pady=3)

        # ── Run button ────────────────────────────────────────────────────────
        tk.Frame(content, height=20, bg=C["bg"]).pack()
        tk.Frame(content, height=1, bg=C["border"]).pack(fill="x")
        tk.Frame(content, height=12, bg=C["bg"]).pack()
        self._run_btn = btn(content, "⚡  Run MOSAIC-GNN",
                            command=self._run_predict, width=24)
        self._run_btn.pack(anchor="w")
        self._pred_status = label(content, "", size=9, color=C["muted"])
        self._pred_status.pack(anchor="w", pady=(8, 0))

    # ── Results tab ───────────────────────────────────────────────────────────

    def _build_results_tab(self, tab):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # ── Scrollable canvas (same pattern as predict tab) ───────────────────
        outer_canvas = tk.Canvas(tab, bg=C["bg"], highlightthickness=0)
        vsb_outer = ttk.Scrollbar(tab, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=vsb_outer.set)
        vsb_outer.pack(side="right", fill="y")
        outer_canvas.pack(side="left", fill="both", expand=True)

        scroll_frame = tk.Frame(outer_canvas, bg=C["bg"])
        _win = outer_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        scroll_frame.bind("<Configure>",
                          lambda e: outer_canvas.configure(
                              scrollregion=outer_canvas.bbox("all")))
        outer_canvas.bind("<Configure>",
                          lambda e: outer_canvas.itemconfig(_win, width=e.width))

        def _mw(e):
            outer_canvas.yview_scroll(-1 * int(e.delta / 120), "units")
        outer_canvas.bind("<MouseWheel>", _mw)
        outer_canvas.bind("<Button-4>", lambda e: outer_canvas.yview_scroll(-1, "units"))
        outer_canvas.bind("<Button-5>", lambda e: outer_canvas.yview_scroll(1, "units"))

        ct = scroll_frame   # content shorthand

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(ct, bg=C["bg"])
        hdr.pack(fill="x", padx=24, pady=(16, 8))
        label(hdr, "Results", size=15, weight="bold").pack(anchor="w")
        label(hdr, "Formula assignments, visualisations, and the full results table.",
              size=9, color=C["muted"]).pack(anchor="w", pady=(2, 0))
        tk.Frame(ct, height=1, bg=C["border"]).pack(fill="x")

        # ── Stat row ──────────────────────────────────────────────────────────
        stat_row = tk.Frame(ct, bg=C["bg"])
        stat_row.pack(fill="x", padx=20, pady=(12, 8))

        STAT_ACCENTS = [C["accent"], "#22d3ee", "#34d399", "#fbbf24"]
        self._stat_vars = {}
        for i, (lbl_txt, accent_color) in enumerate(zip(
            ["Total Peaks", "Unique Formulas", "Avg Confidence", "Coverage"],
            STAT_ACCENTS,
        )):
            sc = tk.Frame(stat_row, bg=C["surface"],
                          highlightbackground=C["border"], highlightthickness=1)
            sc.pack(side="left", fill="both", expand=True,
                    padx=(0, 8 if i < 3 else 0))
            tk.Frame(sc, height=3, bg=accent_color).pack(fill="x")
            v = StringVar(value="—")
            self._stat_vars[lbl_txt] = v
            tk.Label(sc, textvariable=v, font=("Helvetica", 18, "bold"),
                     fg=C["text"], bg=C["surface"]).pack(padx=14, pady=(10, 2))
            tk.Label(sc, text=lbl_txt, font=("Helvetica", 8),
                     fg=C["muted"], bg=C["surface"]).pack(padx=14, pady=(0, 10))

        tk.Frame(ct, height=1, bg=C["border"]).pack(fill="x")

        # ── Visualisations section header ─────────────────────────────────────
        vh = tk.Frame(ct, bg=C["bg"])
        vh.pack(fill="x", padx=20, pady=(12, 8))
        label(vh, "Visualisations", size=11, weight="bold").pack(anchor="w")
        label(vh, "Element class · m/z · Confidence · ppm Error · Van Krevelen · DBE vs m/z · Heteroatom ratios",
              size=9, color=C["muted"]).pack(anchor="w", pady=(2, 0))

        # ── Row 1: Element Class + m/z Histogram ──────────────────────────────
        row1 = tk.Frame(ct, bg=C["bg"])
        row1.pack(fill="x", padx=20, pady=(0, 8))
        row1.columnconfigure(0, weight=1)
        row1.columnconfigure(1, weight=1)

        FIG_BG  = "#0f172a"
        AX_BG   = "#1e293b"

        # Element class panel
        ec_panel = tk.Frame(row1, bg=C["surface"],
                            highlightbackground=C["border"], highlightthickness=1)
        ec_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        label(ec_panel, "Element Class Distribution", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_ec = Figure(figsize=(4.5, 2.8), dpi=92, facecolor=FIG_BG)
        self._ax_ec = fig_ec.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_ec)
        self._canvas_ec = FigureCanvasTkAgg(fig_ec, master=ec_panel)
        self._canvas_ec.get_tk_widget().pack(fill="both", expand=True,
                                             padx=6, pady=(0, 8))

        # m/z histogram panel
        mz_panel = tk.Frame(row1, bg=C["surface"],
                            highlightbackground=C["border"], highlightthickness=1)
        mz_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        label(mz_panel, "m/z Distribution", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_mz = Figure(figsize=(4.5, 2.8), dpi=92, facecolor=FIG_BG)
        self._ax_mz = fig_mz.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_mz)
        self._canvas_mz = FigureCanvasTkAgg(fig_mz, master=mz_panel)
        self._canvas_mz.get_tk_widget().pack(fill="both", expand=True,
                                             padx=6, pady=(0, 8))

        # ── Row 2: Confidence distribution + ppm Error distribution ──────────
        row2 = tk.Frame(ct, bg=C["bg"])
        row2.pack(fill="x", padx=20, pady=(0, 8))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        conf_panel = tk.Frame(row2, bg=C["surface"],
                              highlightbackground=C["border"], highlightthickness=1)
        conf_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        label(conf_panel, "Confidence Score Distribution", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_conf = Figure(figsize=(4.5, 2.8), dpi=92, facecolor=FIG_BG)
        self._ax_conf = fig_conf.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_conf)
        self._canvas_conf = FigureCanvasTkAgg(fig_conf, master=conf_panel)
        self._canvas_conf.get_tk_widget().pack(fill="both", expand=True,
                                               padx=6, pady=(0, 8))

        ppm_panel = tk.Frame(row2, bg=C["surface"],
                             highlightbackground=C["border"], highlightthickness=1)
        ppm_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        label(ppm_panel, "Mass Error Distribution (ppm)", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_ppm = Figure(figsize=(4.5, 2.8), dpi=92, facecolor=FIG_BG)
        self._ax_ppm = fig_ppm.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_ppm)
        self._canvas_ppm = FigureCanvasTkAgg(fig_ppm, master=ppm_panel)
        self._canvas_ppm.get_tk_widget().pack(fill="both", expand=True,
                                              padx=6, pady=(0, 8))

        # ── Van Krevelen panel (full width) ───────────────────────────────────
        vk_panel = tk.Frame(ct, bg=C["surface"],
                            highlightbackground=C["border"], highlightthickness=1)
        vk_panel.pack(fill="x", padx=20, pady=(0, 12))
        label(vk_panel, "Van Krevelen Plot  (H/C vs O/C)", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_vk = Figure(figsize=(9, 3.4), dpi=92, facecolor=FIG_BG)
        self._ax_vk = fig_vk.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_vk)
        self._canvas_vk = FigureCanvasTkAgg(fig_vk, master=vk_panel)
        self._canvas_vk.get_tk_widget().pack(fill="both", expand=True,
                                             padx=6, pady=(0, 8))

        # ── Row 3: DBE vs m/z + Heteroatom Ratios ─────────────────────────────
        row3 = tk.Frame(ct, bg=C["bg"])
        row3.pack(fill="x", padx=20, pady=(0, 12))
        row3.columnconfigure(0, weight=1)
        row3.columnconfigure(1, weight=1)

        dbe_panel = tk.Frame(row3, bg=C["surface"],
                             highlightbackground=C["border"], highlightthickness=1)
        dbe_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        label(dbe_panel, "DBE vs m/z", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_dbe = Figure(figsize=(4.5, 2.8), dpi=92, facecolor=FIG_BG)
        self._ax_dbe = fig_dbe.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_dbe)
        self._canvas_dbe = FigureCanvasTkAgg(fig_dbe, master=dbe_panel)
        self._canvas_dbe.get_tk_widget().pack(fill="both", expand=True,
                                              padx=6, pady=(0, 8))

        het_panel = tk.Frame(row3, bg=C["surface"],
                             highlightbackground=C["border"], highlightthickness=1)
        het_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        label(het_panel, "Mean Heteroatom / C Ratios by Class", size=9, weight="bold",
              bg=C["surface"]).pack(anchor="w", padx=10, pady=(8, 2))

        fig_het = Figure(figsize=(4.5, 2.8), dpi=92, facecolor=FIG_BG)
        self._ax_het = fig_het.add_subplot(111, facecolor=AX_BG)
        _style_ax(self._ax_het)
        self._canvas_het = FigureCanvasTkAgg(fig_het, master=het_panel)
        self._canvas_het.get_tk_widget().pack(fill="both", expand=True,
                                              padx=6, pady=(0, 8))

        tk.Frame(ct, height=1, bg=C["border"]).pack(fill="x")

        # ── Formula Assignments table ──────────────────────────────────────────
        th = tk.Frame(ct, bg=C["bg"])
        th.pack(fill="x", padx=20, pady=(10, 4))
        label(th, "Formula Assignments", size=11, weight="bold").pack(anchor="w")

        tbl_frame = tk.Frame(ct, bg=C["bg"])
        tbl_frame.pack(fill="x", padx=20, pady=(0, 6))

        cols = ("m/z", "Formula", "Confidence", "Element Class", "Error (ppm)")
        self._tree = ttk.Treeview(tbl_frame, columns=cols,
                                  show="headings", selectmode="browse", height=14)
        col_widths = [140, 170, 120, 160, 120]
        for c, w in zip(cols, col_widths):
            self._tree.heading(c, text=c)
            self._tree.column(c, width=w, anchor="center")

        tvsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=tvsb.set)
        tvsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        # ── Action buttons ─────────────────────────────────────────────────────
        dl_frame = tk.Frame(ct, bg=C["bg"])
        dl_frame.pack(fill="x", padx=20, pady=(6, 18))
        btn(dl_frame, "⬇  Download CSV", command=self._download_csv,
            accent=False, width=18).pack(side="left")
        btn(dl_frame, "🌐  Open in Browser", command=self._open_browser,
            accent=False, width=18).pack(side="left", padx=8)

    # ── backend lifecycle ─────────────────────────────────────────────────────

    def _ensure_backend_thread(self):
        def _run():
            if _backend_alive():
                self._set_status("connected")
                return
            self._set_status("starting")
            proc = _start_backend()
            self._backend_proc = proc
            for _ in range(30):
                time.sleep(1)
                if _backend_alive():
                    self._set_status("connected")
                    return
            self._set_status("failed")
        threading.Thread(target=_run, daemon=True).start()

    def _set_status(self, state: str):
        mapping = {
            "connected": ("⬤  Backend ready", C["success"]),
            "starting":  ("⬤  Starting backend…", C["warn"]),
            "failed":    ("⬤  Backend offline", C["error"]),
        }
        text, color = mapping.get(state, ("⬤  Unknown", C["muted"]))
        self._status_lbl.config(text=text, fg=color)

    def _set_sb(self, text: str):
        self._sb_lbl.config(text=text)

    # ── actions ───────────────────────────────────────────────────────────────

    def _select_sample_type(self, sample_type: str):
        self._sample_type.set(sample_type)
        is_pfas = sample_type == "PFAS"
        # Highlight the active sample-type card
        for stype, frame in self._sample_type_btns.items():
            active = stype == sample_type
            accent_col = "#22d3ee" if stype == "DOM" else C["warn"]
            frame.config(highlightbackground=accent_col if active else C["border"])
        # Auto-select model and update info label
        if is_pfas:
            self._model.set("Model11b")
            self._model_info_lbl.config(
                text="▶  Model 11b  ·  bypass ·  best for PFAS / AFFF  (78.9% Top-1)",
                fg=C["warn"],
            )
        else:
            self._model.set("Model10")
            self._model_info_lbl.config(
                text="▶  Model 10  ·  14-dim FCL  ·  best for DOM / NOM",
                fg="#22d3ee",
            )

    def _select_model(self, model: str):
        self._model.set(model)


    def _select_ion(self, mode: str):
        self._ion_mode.set(mode)
        for m, b in self._ion_btns.items():
            active = m == mode
            b.config(
                bg=C["accent"] if active else C["surface"],
                highlightbackground=C["accent"] if active else C["border"],
            )

    def _upload_file(self, path: str):
        r = _ensure_requests()
        if r is None:
            messagebox.showerror("Missing package", "requests is not installed.")
            return

        self._set_sb(f"Uploading {Path(path).name}…")
        self._pbar.start(10)

        def _do():
            try:
                with open(path, "rb") as f:
                    resp = r.post(f"{API}/upload/", files={"file": (Path(path).name, f)}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                self._job_id.set(data["job_id"])
                self._filename.set(data["filename"])
                self.after(0, self._on_upload_done, path, data)
            except Exception as exc:
                self.after(0, self._on_upload_error, str(exc))

        threading.Thread(target=_do, daemon=True).start()

    def _on_upload_done(self, path: str, data: dict):
        self._pbar.stop()
        fname = Path(path).name
        size  = Path(path).stat().st_size
        size_str = f"{size/1024:.1f} KB" if size < 1_048_576 else f"{size/1048576:.1f} MB"
        self._file_info.config(
            text=f"✔  {fname}   ({size_str})   job_id: {data['job_id']}",
            fg=C["success"],
        )
        self._set_sb(f"Upload complete  ·  {fname}")
        # Fetch column names for the column picker
        self._fetch_columns(data["job_id"])
        # switch to predict page
        self._show_page("predict")

    def _fetch_columns(self, job_id: str):
        """Background fetch of column names; populates the column dropdowns."""
        r = _ensure_requests()
        if not r:
            return

        def _do():
            try:
                resp = r.get(f"{API}/upload/{job_id}/columns", timeout=15)
                resp.raise_for_status()
                info = resp.json()
                cols = info.get("columns", [])
                self.after(0, self._populate_column_menus, cols)
            except Exception:
                pass  # non-critical; dropdowns stay empty (auto-detect used)

        threading.Thread(target=_do, daemon=True).start()

    def _populate_column_menus(self, cols: list):
        self._col_names = cols
        opts = cols if cols else ["— auto-detect —"]
        self._mz_col_menu["values"] = opts
        self._int_col_menu["values"] = ["— auto-detect —"] + cols

        if cols:
            # Auto-guess m/z column
            import re
            mz_guess = next(
                (c for c in cols if re.search(r"m.?z|mass", c, re.I)),
                cols[0]
            )
            self._mz_col.set(mz_guess)
            # Auto-guess intensity column
            int_guess = next(
                (c for c in cols if re.search(r"int(ensity)?|tic|abund", c, re.I)),
                ""
            )
            self._int_col.set(int_guess if int_guess else "— auto-detect —")

    def _on_upload_error(self, msg: str):
        self._pbar.stop()
        self._file_info.config(text=f"✘  Error: {msg}", fg=C["error"])
        self._set_sb("Upload failed")

    def _run_predict(self):
        r = _ensure_requests()
        if not r:
            messagebox.showerror("Missing", "requests is not installed.")
            return
        job_id = self._job_id.get()
        if not job_id:
            messagebox.showwarning("No file", "Please upload a file first (Step 1).")
            return

        self._run_btn.config(state="disabled")
        self._pred_status.config(text="Submitting…", fg=C["muted"])
        self._pbar.start(10)
        self._set_sb("Running MOSAIC-GNN…")

        payload = {
            "job_id":      job_id,
            "model":       self._model.get(),
            "sample_type": self._sample_type.get(),
            "ion_mode":    self._ion_mode.get(),
        }
        mz_min = self._mz_min.get().strip()
        mz_max = self._mz_max.get().strip()
        if mz_min and mz_max:
            try:
                payload["mass_range"] = [float(mz_min), float(mz_max)]
            except ValueError:
                pass
        mz_col = self._mz_col.get().strip()
        int_col = self._int_col.get().strip()
        if mz_col and mz_col != "— auto-detect —":
            payload["mz_column"] = mz_col
        if int_col and int_col != "— auto-detect —":
            payload["intensity_column"] = int_col

        def _do():
            try:
                resp = r.post(f"{API}/predict/", json=payload, timeout=30)
                resp.raise_for_status()
                self.after(0, lambda: self._pred_status.config(text="Running inference…", fg=C["warn"]))
                # poll
                for _ in range(120):
                    time.sleep(2)
                    st = r.get(f"{API}/predict/{job_id}/status", timeout=10).json()
                    if st["status"] == "done":
                        self.after(0, self._on_predict_done)
                        return
                    elif st["status"] == "failed":
                        msg = st.get("message") or st.get("error") or "Inference failed"
                        self.after(0, self._on_predict_error, msg)
                        return
                self.after(0, self._on_predict_error, "Timed out waiting for result")
            except Exception as exc:
                self.after(0, self._on_predict_error, str(exc))

        threading.Thread(target=_do, daemon=True).start()

    def _on_predict_done(self):
        self._pbar.stop()
        self._run_btn.config(state="normal")
        self._pred_status.config(text="✔  Done — loading results…", fg=C["success"])
        self._set_sb("Prediction complete")
        self._load_results()
        self._show_page("results")

    def _on_predict_error(self, msg: str):
        self._pbar.stop()
        self._run_btn.config(state="normal")
        # Truncate for the small status label but show full message in a dialog
        short = msg if len(msg) <= 120 else msg[:117] + "…"
        self._pred_status.config(text=f"✘  {short}", fg=C["error"])
        self._set_sb("Prediction failed")
        # Show a scrollable error dialog with the full message
        win = tk.Toplevel(self)
        win.title("Inference Error")
        win.configure(bg=C["bg"])
        win.geometry("560x260")
        win.resizable(True, True)
        label(win, "Inference failed", size=13, weight="bold", color=C["error"]).pack(
            anchor="w", padx=20, pady=(16, 6))
        frame = tk.Frame(win, bg=C["surface"], highlightbackground=C["border"],
                         highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        txt = tk.Text(frame, wrap="word", bg=C["surface"], fg=C["error"],
                      font=("Menlo", 9), relief="flat", yscrollcommand=sb.set,
                      padx=10, pady=8, borderwidth=0)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", msg)
        txt.config(state="disabled")
        btn(win, "Close", command=win.destroy, accent=False, width=10).pack(pady=(0, 14))

    def _draw_charts(self, rows: list, class_data: list | None,
                       vk_points: list | None):
        """Redraw all three embedded matplotlib charts with fresh data."""
        from collections import defaultdict

        BG     = "#0f172a"
        AX_BG  = "#1e293b"
        TEXT   = "#94a3b8"
        GRID   = "#334155"
        ACCENT = "#7c3aed"
        CYAN   = "#22d3ee"

        def _setup(ax):
            ax.clear()
            ax.set_facecolor(AX_BG)
            ax.figure.set_facecolor(BG)

        def _no_data(ax, msg="No data"):
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color=TEXT, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)

        def _apply_style(ax, grid_axis=None):
            ax.tick_params(colors=TEXT, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            if grid_axis:
                ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.6)
            try:
                ax.figure.tight_layout(pad=0.8)
            except Exception:
                pass

        # ── Element Class bar chart ───────────────────────────────────────────
        ax = self._ax_ec
        _setup(ax)
        if class_data:
            labels = [d["class"] for d in class_data]
            counts = [d["count"] for d in class_data]
            bars = ax.barh(labels, counts, color=ACCENT, height=0.55)
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=6, color=TEXT)
            ax.set_xlabel("Count", fontsize=7)
            _apply_style(ax, grid_axis="x")
        else:
            _no_data(ax)
        self._canvas_ec.draw()

        # ── m/z Histogram ─────────────────────────────────────────────────────
        ax = self._ax_mz
        _setup(ax)
        mz_vals = [r["mz"] for r in rows if r.get("mz") is not None]
        if mz_vals:
            n_bins = min(50, max(10, len(mz_vals) // 4))
            ax.hist(mz_vals, bins=n_bins, color=CYAN, edgecolor="none", alpha=0.85)
            ax.set_xlabel("m/z", fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            _apply_style(ax, grid_axis="y")
        else:
            _no_data(ax)
        self._canvas_mz.draw()

        # ── Van Krevelen scatter ───────────────────────────────────────────────
        CLASS_COLORS = {
            "CHO": "#7c3aed", "CHON": "#22d3ee", "CHOS": "#34d399",
            "CHONS": "#f59e0b", "CHOP": "#ec4899", "CHONSP": "#f97316",
        }
        ax = self._ax_vk
        _setup(ax)
        if vk_points:
            by_class: dict = defaultdict(lambda: ([], []))
            for pt in vk_points:
                oc  = pt.get("O_C")
                hc  = pt.get("H_C")
                cls = pt.get("element_class", "Other")
                if oc is not None and hc is not None:
                    by_class[cls][0].append(oc)
                    by_class[cls][1].append(hc)
            for cls, (xs, ys) in by_class.items():
                ax.scatter(xs, ys, s=3, alpha=0.55,
                           color=CLASS_COLORS.get(cls, "#64748b"),
                           linewidths=0, label=cls)
            ax.set_xlabel("O/C", fontsize=7)
            ax.set_ylabel("H/C", fontsize=7)
            _apply_style(ax, grid_axis=None)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
            if by_class:
                ax.legend(fontsize=6, framealpha=0.3,
                          facecolor=AX_BG, edgecolor=GRID,
                          labelcolor=TEXT, markerscale=2,
                          ncol=min(len(by_class), 4))
        else:
            _no_data(ax, "No Van Krevelen data")
        self._canvas_vk.draw()

        # ── Confidence score histogram ────────────────────────────────────────
        ax = self._ax_conf
        _setup(ax)
        conf_vals = [r.get("confidence") for r in rows
                     if r.get("confidence") is not None]
        if conf_vals:
            ax.hist([v * 100 for v in conf_vals], bins=20,
                    color="#34d399", edgecolor="none", alpha=0.85)
            ax.set_xlabel("Confidence (%)", fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            _apply_style(ax, grid_axis="y")
        else:
            _no_data(ax)
        self._canvas_conf.draw()

        # ── ppm Error histogram ───────────────────────────────────────────────
        ax = self._ax_ppm
        _setup(ax)
        ppm_vals_all = [r.get("mass_error_ppm") for r in rows
                        if r.get("mass_error_ppm") is not None]
        PPM_CLIP = 20.0
        ppm_vals = [v for v in ppm_vals_all if abs(v) <= PPM_CLIP]
        if ppm_vals:
            n_outliers = len(ppm_vals_all) - len(ppm_vals)
            ax.hist(ppm_vals, bins=30, color="#fbbf24", edgecolor="none", alpha=0.85)
            ax.set_xlabel("Error (ppm)", fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            ax.axvline(0, color=ACCENT, linewidth=1.2, linestyle="--", alpha=0.8)
            ax.set_xlim(-PPM_CLIP, PPM_CLIP)
            if n_outliers:
                ax.set_title(f"{n_outliers} outlier(s) |ppm|>{PPM_CLIP:.0f} hidden",
                             fontsize=6, color="#f87171", pad=2)
            _apply_style(ax, grid_axis="y")
        else:
            _no_data(ax)
        self._canvas_ppm.draw()

        # ── DBE vs m/z scatter ────────────────────────────────────────────────
        import re as _re

        def _dbe_val(formula: str) -> float | None:
            atoms = {m[0]: int(m[1] or 1)
                     for m in _re.findall(r'([A-Z][a-z]?)(\d*)', formula)
                     if m[0]}
            C = atoms.get('C', 0); H = atoms.get('H', 0); N = atoms.get('N', 0)
            return (2 * C + 2 + N - H) / 2 if C else None

        ax = self._ax_dbe
        _setup(ax)
        if vk_points:
            xs_d, ys_d, cs_d = [], [], []
            for pt in vk_points:
                f   = pt.get("formula", "")
                mzv = pt.get("mz")
                cls = pt.get("element_class", "Other")
                if f and mzv is not None:
                    d = _dbe_val(f)
                    if d is not None:
                        xs_d.append(mzv)
                        ys_d.append(d)
                        cs_d.append(CLASS_COLORS.get(cls, "#64748b"))
            if xs_d:
                ax.scatter(xs_d, ys_d, s=2, alpha=0.4, c=cs_d, linewidths=0)
                ax.set_xlabel("m/z", fontsize=7)
                ax.set_ylabel("DBE", fontsize=7)
                _apply_style(ax, grid_axis=None)
                ax.grid(color=GRID, linewidth=0.5, alpha=0.4)
            else:
                _no_data(ax)
        else:
            _no_data(ax)
        self._canvas_dbe.draw()

        # ── Mean Heteroatom / C ratios by class ───────────────────────────────
        ax = self._ax_het
        _setup(ax)
        if vk_points:
            import numpy as _np
            nc_by: dict = defaultdict(list)
            sc_by: dict = defaultdict(list)
            for pt in vk_points:
                cls  = pt.get("element_class", "Other")
                nc_v = pt.get("N_C") or 0
                sc_v = pt.get("S_C") or 0
                nc_by[cls].append(nc_v)
                sc_by[cls].append(sc_v)
            classes = sorted(set(nc_by) | set(sc_by))
            if classes:
                nc_means = [_np.mean(nc_by[c]) for c in classes]
                sc_means = [_np.mean(sc_by[c]) for c in classes]
                x_pos    = _np.arange(len(classes))
                w = 0.35
                ax.bar(x_pos - w / 2, nc_means, width=w,
                       label="N/C", color="#22d3ee", alpha=0.85)
                ax.bar(x_pos + w / 2, sc_means, width=w,
                       label="S/C", color="#f59e0b", alpha=0.85)
                ax.set_ylabel("Mean ratio", fontsize=7)
                _apply_style(ax, grid_axis="y")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(classes, fontsize=6, rotation=30,
                                   ha="right", color=TEXT)
                ax.legend(fontsize=6, framealpha=0.3, facecolor=AX_BG,
                          edgecolor=GRID, labelcolor=TEXT)
            else:
                _no_data(ax)
        else:
            _no_data(ax)
        self._canvas_het.draw()

    def _load_results(self):
        r = _ensure_requests()
        job_id = self._job_id.get()
        if not r or not job_id:
            return

        def _do():
            try:
                data = r.get(f"{API}/results/{job_id}", timeout=30).json()
                try:
                    cls_data = r.get(
                        f"{API}/results/{job_id}/element_classes", timeout=15
                    ).json()
                except Exception:
                    cls_data = None
                try:
                    vk_data = r.get(
                        f"{API}/vankrevelen/{job_id}", timeout=15
                    ).json()
                except Exception:
                    vk_data = None
                self.after(0, self._populate_results, data, cls_data, vk_data)
            except Exception as exc:
                self.after(0, lambda: self._set_sb(f"Results load error: {exc}"))

        threading.Thread(target=_do, daemon=True).start()

    def _populate_results(self, data: dict,
                           cls_data: dict | None = None,
                           vk_data:  dict | None = None):
        rows: list[dict] = data.get("results", [])
        self._results = rows

        # clear tree
        for item in self._tree.get_children():
            self._tree.delete(item)

        total      = len(rows)
        formulas   = {r.get("formula") for r in rows if r.get("formula")}
        confs      = [r.get("confidence", 0) for r in rows if r.get("confidence") is not None]
        avg_conf   = f"{sum(confs)/len(confs)*100:.1f}%" if confs else "—"
        coverage   = f"{len(formulas)/total*100:.0f}%" if total else "—"

        self._stat_vars["Total Peaks"].set(f"{total:,}")
        self._stat_vars["Unique Formulas"].set(f"{len(formulas):,}")
        self._stat_vars["Avg Confidence"].set(avg_conf)
        self._stat_vars["Coverage"].set(coverage)

        for row in rows[:2000]:   # limit display to 2000 rows
            self._tree.insert("", END, values=(
                f"{row.get('mz', ''):.5f}" if row.get("mz") else "",
                row.get("formula", ""),
                f"{row.get('confidence', 0)*100:.1f}%" if row.get("confidence") else "",
                row.get("element_class", ""),
                f"{row.get('mass_error_ppm', 0):.3f}" if row.get("mass_error_ppm") is not None else "",
            ))

        self._set_sb(f"Loaded {total:,} results")

        # Draw charts with fresh data
        class_list = cls_data.get("element_classes", []) if cls_data else None
        vk_points  = vk_data.get("points", [])         if vk_data  else None
        self._draw_charts(rows, class_list, vk_points)

    def _download_csv(self):
        job_id = self._job_id.get()
        if not job_id:
            messagebox.showwarning("No results", "Run a prediction first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"mosaic_results_{job_id[:8]}.csv",
        )
        if not save_path:
            return
        if not self._results:
            messagebox.showinfo("Empty", "No results to save.")
            return
        import csv
        with open(save_path, "w", newline="") as f:
            if self._results:
                writer = csv.DictWriter(f, fieldnames=self._results[0].keys())
                writer.writeheader()
                writer.writerows(self._results)
        messagebox.showinfo("Saved", f"Results saved to:\n{save_path}")

    def _open_browser(self):
        webbrowser.open("http://localhost:3000")

    def destroy(self):
        if self._backend_proc:
            try:
                self._backend_proc.terminate()
            except Exception:
                pass
        super().destroy()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MosaicApp()
    app.mainloop()

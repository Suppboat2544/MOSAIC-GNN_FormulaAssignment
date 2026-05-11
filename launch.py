#!/usr/bin/env python3
"""
MOSAIC-GNN Launcher
───────────────────
Shows a popup to choose between the Web Platform or Desktop Application.
  • Web Platform   → starts FastAPI + Next.js, shows tqdm progress, opens browser
  • Desktop App    → launches the standalone Tkinter GUI directly
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "platform" / "backend"
FRONTEND_DIR= ROOT / "platform" / "frontend"
DESKTOP_APP = ROOT / "platform" / "app" / "desktop_app.py"
VENV_PY     = ROOT / ".venv" / "bin" / "python"
VENV_UV     = ROOT / ".venv" / "bin" / "uvicorn"
PYTHON      = str(VENV_PY)  if VENV_PY.exists()  else sys.executable
UVICORN     = str(VENV_UV)  if VENV_UV.exists()  else "uvicorn"
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL= "http://localhost:3000"

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":      "#0f172a",
    "surface": "#1e293b",
    "border":  "#334155",
    "accent":  "#7c3aed",
    "accent2": "#6d28d9",
    "text":    "#f1f5f9",
    "muted":   "#94a3b8",
    "dim":     "#475569",
    "success": "#34d399",
    "error":   "#f87171",
    "warn":    "#fbbf24",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _backend_alive() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=2)
        return True
    except Exception:
        return False


def _frontend_alive() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(FRONTEND_URL, timeout=2)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Launcher  (osascript on macOS, terminal fallback elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

def _osascript_dialog() -> "str | None":
    """Show a native macOS AppleScript dialog. Returns 'web', 'desktop', or None."""
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e", 'set choices to {"Web Platform  (opens in browser)", "Desktop Application  (native window)"}',
                "-e", 'set chosen to choose from list choices with title "MOSAIC-GNN Launcher" with prompt "How would you like to launch?" default items {"Web Platform  (opens in browser)"} without multiple selections allowed',
                "-e", 'if chosen is false then return "cancel"',
                "-e", "return item 1 of chosen",
            ],
            capture_output=True, text=True, timeout=120,
        )
        out = result.stdout.strip()
        if out.startswith("Web"):
            return "web"
        if out.startswith("Desktop"):
            return "desktop"
        return None          # cancelled
    except Exception:
        return None


def _terminal_menu() -> "str | None":
    """Simple numbered terminal picker — works on any platform."""
    print("\n" + "─" * 52)
    print("  MOSAIC-GNN Launcher")
    print("─" * 52)
    print("  [1]  Web Platform   (opens in browser)")
    print("  [2]  Desktop Application  (native window)")
    print("  [q]  Quit")
    print("─" * 52)
    while True:
        raw = input("  Your choice: ").strip().lower()
        if raw in ("1", "web"):
            return "web"
        if raw in ("2", "desktop"):
            return "desktop"
        if raw in ("q", "quit", "exit"):
            return None
        print("  Please enter 1, 2, or q.")


def show_launcher() -> "str | None":
    """Return 'web', 'desktop', or None (cancelled)."""
    import platform as _platform
    if _platform.system() == "Darwin":
        choice = _osascript_dialog()
        if choice is None:          # user closed dialog without picking
            choice = _terminal_menu()
        return choice
    return _terminal_menu()


# ─────────────────────────────────────────────────────────────────────────────
# Web Platform launcher  (FastAPI + Next.js + tqdm progress)
# ─────────────────────────────────────────────────────────────────────────────

def _start_web():
    try:
        from tqdm import tqdm
    except ImportError:
        # fallback — simple print progress
        class tqdm:  # type: ignore[no-redef]
            def __init__(self, total, desc="", bar_format=None, **kw):
                self._total = total
                self._n = 0
                self._desc = desc
                print(f"\n{desc}")
            def update(self, n=1):
                self._n += n
                pct = int(self._n / self._total * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {self._n}/{self._total}", end="", flush=True)
            def set_description(self, s):
                print(f"\n  {s}", end="")
            def close(self):
                print()

    STEPS = 10
    bar = tqdm(
        total=STEPS,
        desc="🚀  MOSAIC-GNN Platform",
        bar_format="  {desc}  {bar}  {n}/{total}",
        ncols=70,
        colour="magenta",  # ignored on plain tqdm fallback
    )

    def _step(desc: str, wait: float = 0.3):
        bar.set_description(desc)
        bar.update(1)
        time.sleep(wait)

    print("\n" + "═" * 58)
    print("  MOSAIC-GNN  ·  Launching Web Platform")
    print("═" * 58)

    # ── Start backend ──
    _step("Starting FastAPI backend …", 0.2)
    backend_proc = None
    if not _backend_alive():
        backend_cmd = [
            PYTHON, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ]
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        _step("Backend already running  ✔", 0.1)

    # ── Wait for backend ──
    _step("Waiting for backend to be ready …", 0.4)
    for i in range(25):
        if _backend_alive():
            break
        time.sleep(1)
        if i % 3 == 0:
            bar.set_description(f"Backend starting … ({i+1}s)")
    else:
        bar.close()
        print("\n  ✘  Backend did not start. Check that port 8000 is free.\n")
        return

    _step("Backend ready  ✔", 0.2)

    # ── Start Next.js frontend ──
    _step("Starting Next.js frontend …", 0.3)
    frontend_proc = None
    if not _frontend_alive():
        # detect npm path
        npm_candidates = [
            "/usr/local/bin/npm", "/opt/homebrew/bin/npm", "npm",
        ]
        npm = next((p for p in npm_candidates if Path(p).exists() or p == "npm"), "npm")
        frontend_proc = subprocess.Popen(
            [npm, "run", "dev", "--", "--port", "3000"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        _step("Frontend already running  ✔", 0.1)

    # ── Wait for frontend ──
    _step("Waiting for frontend to be ready …", 0.5)
    for i in range(40):
        if _frontend_alive():
            break
        time.sleep(1.5)
        if i % 4 == 0:
            bar.set_description(f"Frontend compiling … ({(i+1)*1}s)")
    else:
        bar.close()
        print("\n  ✘  Frontend did not start. Run 'npm install' inside platform/frontend/ first.\n")
        return

    _step("Frontend ready  ✔", 0.2)
    _step("Platform fully loaded  ✔", 0.2)
    bar.close()

    print("\n" + "═" * 58)
    print(f"  ✔  Backend   →  {BACKEND_URL}")
    print(f"  ✔  Frontend  →  {FRONTEND_URL}")
    print("═" * 58)
    print("  Opening browser…  (press Ctrl+C to stop)")
    print()

    time.sleep(0.8)
    webbrowser.open(FRONTEND_URL)

    # ── Keep running ──
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Shutting down…")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("  Stopped.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Desktop Application launcher
# ─────────────────────────────────────────────────────────────────────────────

def _start_desktop():
    print("\n" + "═" * 58)
    print("  MOSAIC-GNN  ·  Launching Desktop Application")
    print("═" * 58)
    print("  Starting backend (silent) …")

    if not _backend_alive():
        subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("  Opening desktop window…\n")
    subprocess.run([PYTHON, str(DESKTOP_APP)])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    choice = show_launcher()

    if choice is None:
        print("  No option selected. Exiting.")
        return

    if choice == "web":
        _start_web()
    elif choice == "desktop":
        _start_desktop()


if __name__ == "__main__":
    main()

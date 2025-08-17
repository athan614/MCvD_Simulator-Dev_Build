# analysis/ui_progress.py
"""
Lightweight progress manager used by analysis drivers.

Usage:
    pm = ProgressManager(mode="tqdm")  # or "rich" or "gui" or "none"
    bar = pm.task(total=100, description="Overall")
    bar.update(1)
    bar.close()
    pm.stop()
"""

from typing import Optional

class _NoopBar:
    def update(self, n: int = 1): pass
    def close(self): pass

class _TqdmWrapper:
    def __init__(self, total: int, description: str):
        from tqdm import tqdm  # type: ignore
        self._bar = tqdm(total=total, desc=description)
    def update(self, n: int = 1):
        self._bar.update(n)
    def close(self):
        self._bar.close()

class _RichWrapper:
    def __init__(self, backend, task_id):
        self._backend = backend
        self._task_id = task_id
    def update(self, n: int = 1):
        self._backend.update(self._task_id, advance=n)
    def close(self):
        # remove_task keeps the screen clean when nested tasks finish
        try:
            self._backend.remove_task(self._task_id)
        except Exception:
            pass

class _GuiWrapper:
    def __init__(self, root, bar, label, total: int):
        self._root = root
        self._bar = bar
        self._label = label
        self._total = max(1, int(total))
        self._count = 0
    def update(self, n: int = 1):
        try:
            self._count += n
            self._bar['value'] = min(self._count, self._total)
            self._bar['maximum'] = self._total
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            pass
    def close(self):
        # leave the row; overall window closes in ProgressManager.stop()
        pass

class ProgressManager:
    def __init__(self, mode: str = "tqdm"):
        self.mode = mode
        self._backend = None   # for rich
        self._gui = None       # (root, rows)
        if self.mode == "rich":
            try:
                from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn
                self._backend = Progress(
                    "{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•",
                    TimeElapsedColumn(),
                    "• ETA",
                    TimeRemainingColumn(),
                )
                self._backend.start()
            except Exception:
                self.mode = "tqdm"
        elif self.mode == "gui":
            # Build a tiny Tk popup; fall back silently if display not available
            try:
                import tkinter as tk
                from tkinter import ttk
                root = tk.Tk()
                root.title("Analysis Progress")
                root.geometry("420x120")
                root.resizable(False, False)
                self._gui = (root, [])  # rows
                # Header
                hdr = tk.Label(root, text="Running analyses...", font=("TkDefaultFont", 10, "bold"))
                hdr.pack(pady=6)
            except Exception:
                self.mode = "tqdm"

    def task(self, total: int, description: str = ""):
        if self.mode == "none":
            return _NoopBar()
        if self.mode == "rich" and self._backend is not None:
            task_id = self._backend.add_task(description, total=total)
            return _RichWrapper(self._backend, task_id)
        if self.mode == "gui" and self._gui is not None:
            import tkinter as tk
            from tkinter import ttk
            root, rows = self._gui
            # Row frame
            fr = tk.Frame(root)
            fr.pack(fill="x", padx=10, pady=4)
            lbl = tk.Label(fr, text=description)
            lbl.pack(anchor="w")
            pb = ttk.Progressbar(fr, orient="horizontal", length=380, mode="determinate")
            pb.pack(anchor="w", pady=2)
            rows.append((fr, pb, lbl))
            root.update_idletasks()
            root.update()
            return _GuiWrapper(root, pb, lbl, total)
        # fallback: tqdm
        try:
            return _TqdmWrapper(total=total, description=description)
        except Exception:
            return _NoopBar()

    def stop(self):
        if self.mode == "rich" and self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                pass
        if self.mode == "gui" and self._gui is not None:
            try:
                root, _ = self._gui
                root.destroy()
            except Exception:
                pass

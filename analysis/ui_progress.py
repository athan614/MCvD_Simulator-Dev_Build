from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING
import time, threading, queue, math, sys

# For better type hints
if TYPE_CHECKING:
    import tkinter as tk
    from tkinter import ttk
    TkWidget = Union[tk.Widget, ttk.Widget]
else:
    TkWidget = Any

# ---------------------- No-op / shared helpers ----------------------

class _NoopBar:
    def update(self, n: int = 1, **_): 
        self._completed = getattr(self, '_completed', 0) + n
    def set_description(self, *_args, **_kw): pass
    def close(self): pass
    @property
    def completed(self) -> int:
        return getattr(self, '_completed', 0)

def _now() -> float:
    return time.perf_counter()

def _fmt_hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

# ---------------------- GUI implementation --------------------------

@dataclass
class _GuiRow:
    total: int
    completed: int = 0
    start: float = field(default_factory=_now)
    label: str = ""
    parent: Optional[Any] = None  # Changed from Tuple to Any for parent key
    widget: Any = None  # (frame, label, bar, eta_label)
    last_eta: str = "—"
    _smoothed_rate: Optional[float] = None  # Add ETA smoothing state

class _GuiBackend:
    """
    Tkinter UI running in its own thread; receives dict events via Queue.
    """
    def __init__(self, session_meta: Optional[Dict[str, Any]] = None):
        self.session_meta = session_meta or {}
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.rows: Dict[Any, _GuiRow] = {}
        self.workers: Dict[int, str] = {}
        self._last_worker_labels: Dict[int, str] = {}
        self.root: Optional[Any] = None  # Fix: Use Any instead of TkWidget
        self.frames: Dict[str, Any] = {}  # Fix: Use Any instead of TkWidget
        self.status_var: Optional[Any] = None  # Fix: Use Any instead of tk.StringVar

    # -------------- public --------------
    def start(self) -> None:
        self.thread.start()

    def post(self, evt: Dict[str, Any]) -> None:
        self.events.put(evt)

    def stop(self) -> None:
        self.events.put({"type": "stop"})
        # give the UI thread a moment to finish
        for _ in range(100):
            if not self.thread.is_alive(): break
            time.sleep(0.01)

    # -------------- ui thread --------------
    def _run(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        self.root = root
        root.title("Tri‑Channel OECT Simulator — Progress")
        resume_str = "RESUME" if self.session_meta.get("resume") else "FRESH"
        root.title(f"Tri‑Channel OECT Simulator — {resume_str}")
        root.protocol("WM_DELETE_WINDOW", lambda: self.post({"type":"stop"}))
        root.minsize(720, 420)

        # --- Top header (flags / modes / resume) ---
        header = ttk.Frame(root, padding=8); header.pack(side="top", fill="x")
        title = ttk.Label(header, text="Tri‑Channel OECT Simulator", font=("TkDefaultFont", 12, "bold"))
        title.pack(side="left")

        self.status_var = tk.StringVar(value="Status: preparing…")
        status = ttk.Label(header, textvariable=self.status_var)
        status.pack(side="right")

        flags = self._meta_summary()
        flags_lbl = ttk.Label(root, text=flags, padding=6, relief="groove", justify="left")
        flags_lbl.pack(side="top", fill="x")

        # --- Overall / Modes / Sweeps / Distances ---
        overall_frame = ttk.LabelFrame(root, text="Overall", padding=6); overall_frame.pack(fill="x", padx=8, pady=6)
        modes_frame   = ttk.LabelFrame(root, text="Modes", padding=6); modes_frame.pack(fill="x", padx=8, pady=6)
        sweeps_frame  = ttk.LabelFrame(root, text="Sweeps & Distances", padding=6); sweeps_frame.pack(fill="both", expand=True, padx=8, pady=6)
        workers_frame = ttk.LabelFrame(root, text="Workers (optional)", padding=6); workers_frame.pack(fill="x", padx=8, pady=6)

        self.frames = {
            "overall": overall_frame,
            "mode": modes_frame,
            "sweep": sweeps_frame,
            "dist": sweeps_frame,   # share container
            "worker": workers_frame,
        }

        # periodic pump
        def pump():
            # Process a bounded number of events per tick to stay responsive
            processed = 0
            max_per_tick = 250
            try:
                while processed < max_per_tick:
                    evt = self.events.get_nowait()
                    if evt["type"] == "stop":
                        # Schedule destroy on Tk's own event loop; avoids cross-thread destroy
                        root.after(0, root.destroy)
                        return
                    self._handle_event(evt)
                    processed += 1
            except queue.Empty:
                pass
            # refresh ETAs every 200 ms
            self._refresh_etas()
            root.after(200, pump)

        root.after(100, pump)
        try:
            root.mainloop()
        except Exception as e:
            # If Tk crashes (e.g., after sleep/wake), log it but don't crash the whole program
            print(f"⚠️  GUI crashed: {e}. Simulation continues in background.")
            pass

    def _meta_summary(self) -> str:
        # Build a compact, fixed string; persists at top
        parts = []
        m = self.session_meta
        if m.get("modes"):
            parts.append(f"Modes: {', '.join(m['modes'])}")
        if "progress" in m:
            parts.append(f"Progress backend: {m['progress']}")
        if "resume" in m:
            parts.append("Session: RESUME" if m["resume"] else "Session: FRESH")
        if "with_ctrl" in m:
            parts.append(f"CTRL: {'ON' if m['with_ctrl'] else 'OFF'}")
        if "isi" in m:
            parts.append(f"ISI: {'ON' if m['isi'] else 'OFF'}")
        if "flags" in m and m["flags"]:
            parts.append("Flags: " + " ".join(m["flags"]))
        return "  •  ".join(parts) if parts else "—"

    def _ensure_row(self, key: Any, total: int, label: str, kind: str | None, parent_key: Any | None = None) -> None:
        from tkinter import ttk
        if key in self.rows:  # just update label/total if needed
            row = self.rows[key]
            row.total = max(int(total), 0)
            row.label = label
            row.parent = parent_key
            # update label text
            if row.widget:
                _, lbl, bar, _ = row.widget
                lbl.configure(text=self._label_with_stats(key))
                # ensure maximum reflects the new total right away
                bar.configure(maximum=max(int(row.total), 1))
            return

        # choose frame by kind
        frame = self.frames.get(kind or "sweep", self.frames["sweep"])
        row_frame = ttk.Frame(frame); row_frame.pack(fill="x", pady=2)
        lbl = ttk.Label(row_frame, text=label, width=36, anchor="w"); lbl.pack(side="left")
        bar = ttk.Progressbar(row_frame, mode="determinate", length=320, maximum=max(int(total),1))
        bar.pack(side="left", padx=8)
        eta_lbl = ttk.Label(row_frame, text="elapsed: —  •  eta: —"); eta_lbl.pack(side="left")
        self.rows[key] = _GuiRow(total=total, label=label, parent=parent_key,
                                widget=(row_frame, lbl, bar, eta_lbl))

    def _label_with_stats(self, key: Any) -> str:
        row = self.rows[key]
        done = row.completed
        total = max(row.total, 0)
        pct = (100.0*done/total) if total else 0.0
        return f"{row.label}  [{done}/{total} • {pct:4.1f}%]"

    def _handle_event(self, evt: Dict[str, Any]) -> None:
        t = evt.get("type")
        if t == "set_status":
            mode = evt.get("mode"); sweep = evt.get("sweep")
            txt = "Status: "
            if mode:  txt += f"{mode} "
            if sweep: txt += f"— {sweep}"
            if self.status_var:  # Fix: Add None check
                self.status_var.set(txt.strip())
            return

        if t == "create_task":
            key   = evt["key"]
            total = int(evt.get("total", 0))
            label = evt.get("label", str(key))
            kind  = evt.get("kind")   # "overall" | "mode" | "sweep" | "dist"
            parent_key = evt.get("parent", None)
            self._ensure_row(key, total, label, kind, parent_key)
            return

        if t == "update_task":
            key = evt["key"]
            inc = int(evt.get("inc", 1))
            label = evt.get("label")
            if key not in self.rows:
                # create with a guess; the next create will correct totals
                self._ensure_row(key, max(inc,1), label or str(key), None, None)
            row = self.rows[key]
            # Clamp to total (prevents >100%)
            row.completed = max(0, min(row.completed + inc, row.total))
            if label: row.label = label
            # update widgets
            if row.widget:
                _, lbl, bar, eta_lbl = row.widget
                lbl.configure(text=self._label_with_stats(key))
                bar.configure(value=row.completed, maximum=max(int(row.total),1))
                # eta is refreshed by _refresh_etas()
            # Bubble increments to parent(s)
            self._bump_parent(row.parent, inc)
            return

        if t == "close_task":
            key = evt["key"]
            if key in self.rows and self.rows[key].widget:
                frame, *_ = self.rows[key].widget
                try: frame.destroy()
                except Exception: pass
                del self.rows[key]
            return

        if t == "worker_update":
            wid = int(evt.get("worker_id", -1))
            label = evt.get("label", "")
            # De-duplicate to avoid unnecessary redraws
            if self._last_worker_labels.get(wid) != label:
                self.workers[wid] = label
                self._last_worker_labels[wid] = label
                self._refresh_workers()
            return

    def _bump_parent(self, parent_key: Any | None, inc: int) -> None:
        if parent_key is None: 
            return
        if parent_key not in self.rows:
            return
        prow = self.rows[parent_key]
        prow.completed = max(0, min(prow.completed + inc, prow.total))
        if prow.widget:
            _, lbl, bar, eta_lbl = prow.widget
            lbl.configure(text=self._label_with_stats(parent_key))
            bar.configure(value=prow.completed, maximum=max(int(prow.total),1))
        # Recurse upward if there is a grandparent
        self._bump_parent(prow.parent, inc)

    def _refresh_workers(self) -> None:
        import tkinter as tk
        from tkinter import ttk
        frame = self.frames["worker"]
        for child in list(frame.winfo_children()):
            child.destroy()
        if not self.workers:
            ttk.Label(frame, text="(not used)").pack(anchor="w")
            return
        grid = ttk.Frame(frame); grid.pack(fill="x")
        for wid in sorted(self.workers):
            row = ttk.Frame(grid); row.pack(fill="x")
            ttk.Label(row, text=f"Worker {wid:02d}:", width=12).pack(side="left")
            ttk.Label(row, text=self.workers[wid], width=64, anchor="w").pack(side="left")

    def _refresh_etas(self) -> None:
        # compute elapsed & ETA every tick with exponential smoothing
        for key, row in list(self.rows.items()):
            if not row.widget: continue
            elapsed = _now() - row.start
            rem = "—"
            if row.total > 0 and row.completed > 0:
                current_rate = row.completed / max(elapsed, 1e-6)
                
                # Exponential smoothing for rate (α = 0.3 for responsiveness vs stability)
                if row._smoothed_rate is None:
                    row._smoothed_rate = current_rate
                else:
                    alpha = 0.3
                    row._smoothed_rate = alpha * current_rate + (1 - alpha) * row._smoothed_rate
                
                remaining = (row.total - row.completed) / max(row._smoothed_rate, 1e-9)
                rem = _fmt_hms(remaining)
            
            lbl_txt = self._label_with_stats(key)
            frame, lbl, bar, eta_lbl = row.widget
            lbl.configure(text=lbl_txt)
            eta_lbl.configure(text=f"elapsed: {_fmt_hms(elapsed)}  •  eta: {rem}")

# ---------------------- Public Manager ------------------------------

class ProgressManager:
    def __init__(self, mode: str = "tqdm", gui_session_meta: Optional[Dict[str, Any]] = None):
        self.mode = (mode or "tqdm").lower()
        self._backend = None
        self._gui: Optional[_GuiBackend] = None
        self._tqdm = None
        self._rich = None
        self._rich_tasks: Dict[Any, Any] = {}   # key -> task_id
        self._tqdm_bars: Dict[Any, Any] = {}    # key -> tqdm bar

        if self.mode == "gui":
            try:
                # Quick import check for headless environments / macOS constraints
                import tkinter as _tk  # noqa: F401
                import platform, threading
                if platform.system() == "Darwin" and threading.current_thread() is not threading.main_thread():
                    # Tk on macOS must run on the main thread; fall back gracefully
                    print("⚠️  GUI not supported on macOS (must run in main thread). Falling back to 'rich'.")
                    print("    → This is expected behavior on macOS when running in background threads.")
                    self.mode = "rich"
                else:
                    self._gui = _GuiBackend(gui_session_meta or {})
                    self._gui.start()
            except Exception as e:
                # Graceful fallback
                print(f"⚠️  GUI unavailable ({e}); falling back to 'rich' progress.")
                self.mode = "rich"
        elif self.mode == "rich":
            try:
                from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
                self._rich = Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•", TimeElapsedColumn(), "•", TimeRemainingColumn()
                )
                self._rich.start()
            except Exception:
                self.mode = "tqdm"
        if self.mode == "tqdm":
            try:
                import tqdm  # noqa
                self._tqdm = __import__("tqdm").tqdm
            except Exception:
                self.mode = "none"

    # --- session header ticker ---
    def set_status(self, mode: Optional[str] = None, sweep: Optional[str] = None) -> None:
        if self._gui is not None:  # Fix: More explicit None check
            self._gui.post({"type": "set_status", "mode": mode, "sweep": sweep})

    # --- tasks ---
    def task(self, total: int, description: str, parent: Any = None, key: Any = None, kind: Optional[str] = None):
        key = key if key is not None else (description, id(self))
        total = int(total) if total is not None else 0

        if self._gui is not None:  # Fix: More explicit None check
            self._gui.post({"type":"create_task","key":key,"total":total,
                            "label":description,"kind":kind,"parent":parent})
            # return a lightweight proxy that posts updates
            mgr = self
            class _GuiProxy:
                def __init__(self):
                    self._completed = 0
                def update(self, n: int = 1, description: Optional[str] = None):
                    self._completed += n
                    if mgr._gui is not None:  # Fix: Add None check inside proxy
                        mgr._gui.post({"type":"update_task","key":key,"inc":n,"label":description})
                def set_description(self, text: str):
                    if mgr._gui is not None:  # Fix: Add None check inside proxy
                        mgr._gui.post({"type":"update_task","key":key,"inc":0,"label":text})
                def close(self):
                    if mgr._gui is not None:  # Fix: Add None check inside proxy
                        mgr._gui.post({"type":"close_task","key":key})
                @property
                def completed(self) -> int:
                    return self._completed
            return _GuiProxy()

        if self._rich:
            task_id = self._rich.add_task(description, total=total)
            progress = self._rich
            class _RichProxy:
                def __init__(self):
                    self._completed = 0
                def update(self, n: int = 1, description: Optional[str] = None):
                    self._completed += n
                    if description:
                        progress.update(task_id, advance=n, description=description)
                    else:
                        progress.update(task_id, advance=n)
                def set_description(self, text: str):
                    progress.update(task_id, description=text)
                def close(self):
                    progress.remove_task(task_id)
                @property
                def completed(self) -> int:
                    return self._completed
            # remember task so we can update totals later
            self._rich_tasks[key] = task_id
            return _RichProxy()

        if self._tqdm:
            bar = self._tqdm(total=total, desc=description, leave=True)
            mgr = self
            class _TqdmProxy:
                def __init__(self):
                    self._completed = 0
                def update(self, n: int = 1, description: Optional[str] = None):
                    self._completed += n
                    if description:
                        try:
                            bar.set_description_str(description, refresh=False)
                        except Exception:
                            pass
                    bar.update(n)
                def set_description(self, text: str):
                    try:
                        bar.set_description_str(text, refresh=False)
                    except Exception:
                        pass
                def close(self):
                    try:
                        bar.close()
                    except Exception:
                        pass
                @property
                def completed(self) -> int:
                    return self._completed
            self._tqdm_bars[key] = bar
            return _TqdmProxy()

        return _NoopBar()

    # --- worker pane ---
    def worker_update(self, worker_id: int, label: str) -> None:
        if self._gui is not None:  # Fix: More explicit None check
            self._gui.post({"type":"worker_update","worker_id":worker_id,"label":label})

    def worker_task(self, worker_id: int, total: int, label: str = "", parent: Any = None):
        """Create a worker progress bar like any task."""
        key = ("worker", worker_id)
        description = label or f"Worker {worker_id:02d}"
        return self.task(total=total, description=description, key=key, kind="worker", parent=parent)

    def update_total(self, key: Any, total: int, label: Optional[str] = None, 
                     kind: Optional[str] = None, parent: Any = None) -> None:
        """
        Update the total for an existing task or create a new task with the specified total.
        This is a public API for updating task parameters without accessing private attributes.
        """
        if self._gui is not None:
            self._gui.post({"type":"create_task","key":key,"total":int(total),
                            "label":label or "", "kind":kind, "parent":parent})
        elif self._rich and key in self._rich_tasks:
            try:
                self._rich.update(self._rich_tasks[key], total=int(total))
                if label:
                    self._rich.update(self._rich_tasks[key], description=label)
            except Exception:
                pass
        elif self._tqdm and key in self._tqdm_bars:
            try:
                bar = self._tqdm_bars[key]
                bar.total = int(total)
                if label:
                    bar.set_description_str(label, refresh=False)
                bar.refresh()
            except Exception:
                pass

    def stop(self) -> None:
        try:
            if self._gui:
                self._gui.stop()
        except Exception as e:
            # GUI might have already crashed; that's OK
            print(f"⚠️  GUI stop warning: {e}")
        finally:
            try:
                if self._rich:
                    self._rich.stop()
            except Exception:
                pass
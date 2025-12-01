from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys, os, io
from typing import Any, Optional

class _Tee(io.TextIOBase):
    def __init__(self, stream: Any, f: Any):  # Accept Any to handle sys.stdout/stderr
        self.stream, self.f = stream, f
    def write(self, s: str) -> int:
        n1 = self.stream.write(s); self.stream.flush()
        n2 = self.f.write(s); self.f.flush()
        return max(n1, n2)
    def flush(self) -> None:
        self.stream.flush(); self.f.flush()

# Keep track of the active tee file so repeated setup calls don't nest infinitely
_ACTIVE_LOG_FILE: Optional[io.TextIOBase] = None
_ACTIVE_LOG_PATH: Optional[Path] = None


def _unwrap_stream(stream: Any) -> Any:
    """Return the deepest non-tee stream to avoid recursive wrapping."""
    seen = set()
    cur = stream
    while isinstance(cur, _Tee) and getattr(cur, "stream", None) not in seen:
        seen.add(cur)
        cur = cur.stream
    return cur


def tee_logging_active() -> bool:
    """Return True when stdout/stderr are already wrapped by the tee helper."""
    return isinstance(sys.stdout, _Tee) or isinstance(sys.stderr, _Tee)


def setup_tee_logging(log_dir: Path, prefix: str = "run", fsync: bool = False) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"{prefix}_{ts}.log"
    # Close any previous log file to avoid handle leaks when reconfiguring
    global _ACTIVE_LOG_FILE, _ACTIVE_LOG_PATH
    try:
        if _ACTIVE_LOG_FILE and not _ACTIVE_LOG_FILE.closed:
            _ACTIVE_LOG_FILE.close()
    except Exception:
        pass

    f = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered

    # Always unwrap to the base console stream so repeated calls do not create deep recursion chains
    base_stdout = _unwrap_stream(getattr(sys, "__stdout__", sys.stdout))
    base_stderr = _unwrap_stream(getattr(sys, "__stderr__", sys.stderr))
    # Wrap with a tee so console output is mirrored to the file in real time
    sys.stdout = _Tee(base_stdout, f)      # type: ignore[assignment,arg-type]
    sys.stderr = _Tee(base_stderr, f)      # type: ignore[assignment,arg-type]
    _ACTIVE_LOG_FILE = f
    _ACTIVE_LOG_PATH = path
    # Optional: force OS flush on each write (slower; useful on fragile systems)
    if fsync:
        old_write = f.write
        def _write_and_fsync(s: str) -> int:
            n = old_write(s); f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass
            return n
        f.write = _write_and_fsync  # type: ignore[assignment]
    print(f"[log] Streaming log to: {path}")
    return path

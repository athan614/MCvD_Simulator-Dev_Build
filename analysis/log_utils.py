from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys, os, io
from typing import Any

class _Tee(io.TextIOBase):
    def __init__(self, stream: Any, f: Any):  # Accept Any to handle sys.stdout/stderr
        self.stream, self.f = stream, f
    def write(self, s: str) -> int:
        n1 = self.stream.write(s); self.stream.flush()
        n2 = self.f.write(s); self.f.flush()
        return max(n1, n2)
    def flush(self) -> None:
        self.stream.flush(); self.f.flush()

def setup_tee_logging(log_dir: Path, prefix: str = "run", fsync: bool = False) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"{prefix}_{ts}.log"
    f = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
    # Wrap with a tee so console output is mirrored to the file in real time
    sys.stdout = _Tee(sys.stdout, f)      # type: ignore[assignment,arg-type]
    sys.stderr = _Tee(sys.stderr, f)      # type: ignore[assignment,arg-type]
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
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ClusterLogger:
    """
    Lightweight JSON-lines logger used for graph clustering diagnostics.
    Each log entry is appended to <project>/app/cluster_logs/YYYYMMDD.log.
    """

    def __init__(self, *, base_dir: Optional[Path] = None, session_id: Optional[str] = None) -> None:
        default_dir = Path(__file__).resolve().parent / "cluster_logs"
        self.base_dir = base_dir or default_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._lock = threading.Lock()

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "event": event,
        }
        if self.session_id:
            entry["session"] = self.session_id
        entry.update(payload)
        line = json.dumps(entry, ensure_ascii=False)
        filename = datetime.utcnow().strftime("%Y%m%d") + ".log"
        path = self.base_dir / filename
        with self._lock, path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


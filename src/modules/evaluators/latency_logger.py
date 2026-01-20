from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def log_latency(
    *,
    log_path: Path,
    stage: str,
    total_s: float,
    **meta: Any,
) -> None:
    """
    Append one latency entry to a JSONL file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "stage": stage,
        "total_s": round(float(total_s), 4),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **meta,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def time_block(fn, *, log_path: Path, stage: str, **meta: Any):
    """
    Measure latency of a function call.
    """
    t0 = time.perf_counter()
    result = fn()
    total_s = time.perf_counter() - t0
    log_latency(
        log_path=log_path,
        stage=stage,
        total_s=total_s,
        **meta,
    )
    return result

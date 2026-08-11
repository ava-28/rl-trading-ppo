"""
utils/logger.py
---------------
Structured experiment logger that records all training metrics, evaluation
results, and configuration to a JSON-lines file for post-hoc analysis.

Usage
-----
  from utils.logger import ExperimentLogger

  with ExperimentLogger("results_v2/run_log.jsonl") as logger:
      logger.log_config(cfg)
      logger.log_episode(fold_id, abl_id, seed, ep=1, metrics={...})
      logger.log_eval(fold_id, abl_id, seed, metrics={...})
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


log = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Append-only JSON-lines logger for experiment runs.

    Each call to log_*() appends one JSON object to the file, tagged with
    a timestamp and run_id so multiple parallel runs can write to separate
    files and be merged later.

    Parameters
    ----------
    path   : path to the output .jsonl file
    run_id : unique identifier for this experiment run (auto-generated if None)
    """

    def __init__(self, path: str, run_id: Optional[str] = None):
        self.path   = path
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "a", buffering=1)   # line-buffered
        return self

    def __exit__(self, *args):
        if self._f:
            self._f.close()
            self._f = None

    # ------------------------------------------------------------------
    # Internal write
    # ------------------------------------------------------------------

    def _write(self, record: Dict[str, Any]):
        record["_ts"]     = time.time()
        record["_run_id"] = self.run_id
        if self._f:
            self._f.write(json.dumps(record, default=str) + "\n")
        else:
            # Fallback: open for one write
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_config(self, config: dict):
        """Record the full experiment configuration."""
        self._write({"type": "config", "config": config})

    def log_episode(
        self,
        fold_id:   str,
        abl_id:    str,
        seed:      int,
        ep:        int,
        metrics:   Dict[str, Any],
    ):
        """Record per-episode training metrics."""
        self._write({
            "type":      "episode",
            "fold_id":   fold_id,
            "ablation":  abl_id,
            "seed":      seed,
            "episode":   ep,
            **metrics,
        })

    def log_eval(
        self,
        fold_id:  str,
        abl_id:   str,
        seed:     int,
        metrics:  Dict[str, Any],
        split:    str = "test",
    ):
        """Record evaluation metrics for one seed."""
        self._write({
            "type":     "eval",
            "fold_id":  fold_id,
            "ablation": abl_id,
            "seed":     seed,
            "split":    split,
            **metrics,
        })

    def log_summary(
        self,
        fold_id: str,
        abl_id:  str,
        metrics: Dict[str, Any],
    ):
        """Record aggregate (mean ± std) summary for a fold × ablation."""
        self._write({
            "type":     "summary",
            "fold_id":  fold_id,
            "ablation": abl_id,
            **metrics,
        })


# ---------------------------------------------------------------------------
# JSONL reader (for post-hoc analysis)
# ---------------------------------------------------------------------------

def read_log(path: str) -> list:
    """Read a JSON-lines log file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def filter_log(records: list, record_type: str) -> list:
    """Filter log records by type ('config', 'episode', 'eval', 'summary')."""
    return [r for r in records if r.get("type") == record_type]

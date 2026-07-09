"""Shared pipeline state, written atomically to `results/run_state.json`.

`scripts/run_all.py` (and the modules it drives) write to this file at every
meaningful transition; `app.py` polls it to render the live dashboard. A
single shared JSON file, replaced atomically on every write, is enough for
one-process-at-a-time orchestration and avoids adding a database dependency
just to pass status between a background subprocess and a UI.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

STATE_FILENAME = "run_state.json"

# Pipeline-level stages, in order.
STAGE_ENV_CHECK = "env_check"
STAGE_SPLIT = "split"
STAGE_EXPERIMENTS = "experiments"
STAGE_COMPARE = "compare"
STAGE_FINAL_REPORT = "final_report"
STAGE_CONCLUSIONS = "conclusions"
STAGE_DONE = "done"

# Per-experiment statuses.
STATUS_PENDING = "pending"
STATUS_SMOKE_TEST = "smoke_test"
STATUS_TRAINING = "training"
STATUS_EVALUATING = "evaluating"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass
class RunState:
    path: Path
    stage: str = STAGE_ENV_CHECK
    stage_detail: str = ""
    env_check: Dict = field(default_factory=dict)
    experiments: Dict[str, Dict] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @classmethod
    def create(cls, results_dir: Path) -> "RunState":
        results_dir.mkdir(parents=True, exist_ok=True)
        return cls(path=results_dir / STATE_FILENAME)

    @classmethod
    def load(cls, results_dir: Path) -> Optional["RunState"]:
        path = results_dir / STATE_FILENAME
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        payload["path"] = path
        return cls(**payload)

    def _save(self) -> None:
        self.updated_at = time.time()
        payload = {
            "stage": self.stage,
            "stage_detail": self.stage_detail,
            "env_check": self.env_check,
            "experiments": self.experiments,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def set_stage(self, stage: str, detail: str = "") -> None:
        self.stage = stage
        self.stage_detail = detail
        self._save()

    def set_env_check(self, info: Dict) -> None:
        self.env_check = info
        self._save()

    def set_experiment_status(
        self,
        name: str,
        status: str,
        detail: str = "",
        metrics: Optional[Dict] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> None:
        record = self.experiments.setdefault(
            name, {"status": STATUS_PENDING, "detail": "", "history": []}
        )
        record["status"] = status
        record["detail"] = detail
        record["updated_at"] = time.time()
        if epoch is not None:
            record["epoch"] = epoch
        if total_epochs is not None:
            record["total_epochs"] = total_epochs
        if metrics is not None:
            record["latest_metrics"] = metrics
            record.setdefault("history", []).append({"epoch": epoch, **metrics})
        self._save()

    def fail(self, message: str) -> None:
        self.error = message
        self._save()

    def ordered_experiment_names(self) -> List[str]:
        return list(self.experiments.keys())

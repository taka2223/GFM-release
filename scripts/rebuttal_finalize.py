#!/usr/bin/env python3
"""Wait for both E1 workers, then build E2 and E1 metric summaries.

This supervisor is deliberately narrow: it only waits for the two fixed runner
PID files and invokes the repository's rebuttal reliability/metrics scripts.
Its state file makes the long-running handoff auditable after disconnects.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".rebuttal-runtime"
STATE_PATH = RUNTIME / "finalize-state.json"
RUNNER_PIDS = (RUNTIME / "runner.pid", RUNTIME / "worker2.pid")
EXPERIMENTS = ROOT / "scripts" / "rebuttal_experiments.py"
METRICS = ROOT / "scripts" / "rebuttal_metrics.py"
OUTPUT_ROOT = ROOT / "artifacts" / "rebuttal"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(stage: str, **extra: Any) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_at": now(), **extra}
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)
    print(f"[{now()}] finalize stage={stage}", flush=True)


def read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if value > 0 else None


def is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_workers() -> None:
    initial_pids = {str(path): read_pid(path) for path in RUNNER_PIDS}
    if any(pid is None for pid in initial_pids.values()):
        raise RuntimeError(f"Missing runner PID before finalize start: {initial_pids}")
    write_state("waiting_for_sweeps", runner_pids=initial_pids)
    while True:
        live = {
            str(path): pid
            for path, pid in (
                (path, read_pid(path)) for path in RUNNER_PIDS
            )
            if is_alive(pid)
        }
        if not live:
            return
        write_state("waiting_for_sweeps", live_runner_pids=live)
        time.sleep(30)


def run_stage(name: str, command: list[str], env: dict[str, str]) -> None:
    write_state(name, command=command)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def validate_outputs() -> None:
    e2_path = OUTPUT_ROOT / "summaries" / "e2_reliability" / "summary.json"
    e1_path = OUTPUT_ROOT / "summaries" / "e1_metrics" / "summary.json"
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))

    e2_complete = (
        len(e2) == 4
        and all(row["completed_runs"] == 150 for row in e2)
        and all(row["expected_runs"] == 150 for row in e2)
    )
    e1_complete = (
        len(e1) == 24
        and all(row["completed_seeds"] == 3 for row in e1)
        and all(row["expected_seeds"] == 3 for row in e1)
    )
    if not e2_complete or not e1_complete:
        raise RuntimeError(
            "Incomplete rebuttal summaries: "
            f"E2 rows={len(e2)}, E1 rows={len(e1)}, "
            f"E2 complete={e2_complete}, E1 complete={e1_complete}"
        )


def main() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env.setdefault("TMPDIR", str(RUNTIME / "tmp"))
    env.setdefault("CUDA_CACHE_PATH", str(RUNTIME / "cuda-cache"))
    env.setdefault("MPLCONFIGDIR", str(RUNTIME / "matplotlib"))

    try:
        wait_for_workers()
        run_stage(
            "resuming_any_missing_sweep_batches",
            [
                sys.executable,
                str(EXPERIMENTS),
                "--phase",
                "sweep",
                "--resume",
            ],
            env,
        )
        run_stage(
            "summarizing_e2",
            [sys.executable, str(EXPERIMENTS), "--phase", "reliability", "--resume"],
            env,
        )
        run_stage(
            "decoding_and_evaluating_e1",
            [sys.executable, str(METRICS), "--phase", "all"],
            env,
        )
        write_state("validating")
        validate_outputs()
        write_state("completed")
    except Exception as error:
        write_state("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()

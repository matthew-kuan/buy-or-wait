"""Append run records to the repository-root log.txt in the AGENTS.md section 5 shape.

The transcript entries in log.txt are written by the coding agent; this module adds *pipeline run*
records so the judge can see when output.csv was produced and which requests were flagged for
review. Entries are append-only, UTF-8, LF, and never contain secrets.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_PATH = os.path.join(ROOT, "log.txt")
TOOL = "code/main.py (automated pipeline run)"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _branch() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "branch", "--show-current"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _append(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def append_session_start(mode: str = "live") -> None:
    _append(
        f"\n## [{_now()}] SESSION START\n\n"
        f"tool={TOOL}\n"
        f"Repo Root: {ROOT}\n"
        f"Branch: {_branch()}\n"
        f"Worktree: main\n"
        f"Parent Agent: none\n"
        f"Language: py\n"
        f"Time Remaining: not computed by the pipeline (mode={mode})\n"
    )


def append_run_summary(summary: dict, review: list[tuple[str, list[str]]], failures: list[str]) -> None:
    lines = [f"\n## [{_now()}] Pipeline run: output.csv written ({summary.get('rows', 0)} rows)\n",
             "User Prompt (verbatim, secrets redacted):", "(automated run of code/main.py - no user prompt)", "",
             "Agent Response Summary:",
             f"mode={summary.get('mode')} model_calls={summary.get('calls')} cache_hits={summary.get('cache_hits')} "
             f"reasks={summary.get('reasks')} fallbacks={summary.get('fallbacks')} injection_flags={summary.get('injections')} "
             f"validation_failures={summary.get('validation_failures')} pipeline_failures={len(failures)} "
             f"wall_time_s={summary.get('wall_time_s')}", "",
             "Actions:", "* wrote output.csv (repository root)", "* wrote code/evaluation/usage_report.md"]
    if failures:
        lines.append(f"* wrote code/evaluation/failures.csv ({len(failures)} rows): {', '.join(failures[:20])}")
    if review:
        lines.append("* requests marked for review:")
        for rid, reasons in review:
            lines.append(f"  - {rid}: " + "; ".join(reasons)[:300])
    lines += ["", "Context:", f"tool={TOOL}", f"branch={_branch()}", f"repo_root={ROOT}", "worktree=main", "parent_agent=none"]
    _append("\n".join(lines))

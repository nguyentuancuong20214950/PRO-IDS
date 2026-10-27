"""JSONL and CSV logging helpers for PRO-IDS testbed runs."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


SUMMARY_FIELDS: tuple[str, ...] = (
    "run_id", "label", "policy", "num_requests", "num_completed",
    "edge_ok", "edge_timeout", "edge_error",
    "cloud_ok", "cloud_timeout", "cloud_error",
    "drop_rate", "total_duration_sec",
    "latency_total_mean_ms", "latency_total_p50_ms",
    "latency_total_p95_ms", "latency_total_p99_ms",
    "latency_dispatch_mean_ms", "latency_dispatch_p95_ms",
    "latency_m2_read_mean_ms", "latency_m2_read_p95_ms",
    "edge_latency_mean_ms", "edge_latency_p95_ms",
    "cloud_latency_mean_ms", "cloud_latency_p95_ms",
    "accuracy", "f1", "fpr", "fnr", "tn", "fp", "fn", "tp",
    "m2_state_ok_count", "m2_state_null_count",
    "m2_mean_h_c_ms", "m2_mean_rtt_ms", "m2_mean_q_e", "m2_mean_q_c",
    "started_utc", "finished_utc", "git_commit", "notes",
)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert common scientific/Python objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # Avoid importing NumPy: support ndarray/generic types by their public methods.
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return sanitize_for_json(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        return sanitize_for_json(obj.item())
    return obj


def _json_default(obj: Any) -> Any:
    """Convert supported non-standard JSON objects to native Python values."""
    converted = sanitize_for_json(obj)
    if converted is obj:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    return converted


def iso_now() -> str:
    """Return the current UTC time in second-resolution ISO-8601 form."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def iso_filename_token() -> str:
    """Return a compact UTC timestamp suitable for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class JsonlWriter:
    """Append flushed JSON records to a JSON Lines file."""

    def __init__(self, path: Path) -> None:
        """Create parent directories and open the JSONL file for appending."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open(
            "a", encoding="utf-8", newline=""
        )

    def write(self, record: dict) -> None:
        """Write one sanitized JSON object and flush it immediately."""
        safe_record = sanitize_for_json(record)
        line = json.dumps(
            safe_record,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying JSONL file."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlWriter":
        """Return this writer for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the writer when leaving a context."""
        self.close()


class SummaryCsvWriter:
    """Append rows to a run-summary CSV with a fixed schema."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        """Create or validate the summary CSV header and open it for appending."""
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        if exists:
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                existing = next(csv.reader(stream), [])
            if existing != self.fieldnames:
                raise ValueError(
                    f"CSV header mismatch for {self.path}: "
                    f"existing={existing!r}, requested={self.fieldnames!r}"
                )
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction="raise",
        )
        if not exists:
            self._writer.writeheader()
            self._file.flush()

    def append(self, row: dict) -> None:
        """Validate and append one summary row, filling absent values with blanks."""
        extra = set(row) - set(self.fieldnames)
        if extra:
            raise ValueError(f"Unexpected CSV row keys: {sorted(extra)!r}")
        normalized = {
            key: ("" if row.get(key) is None else row.get(key, ""))
            for key in self.fieldnames
        }
        self._writer.writerow(normalized)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying CSV file."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "SummaryCsvWriter":
        """Return this writer for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the writer when leaving a context."""
        self.close()
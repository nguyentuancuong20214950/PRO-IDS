"""CSV-backed or synthetic feature requests for the PRO-IDS testbed."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

LOG = logging.getLogger(__name__)
_MEMORY_ROW_LIMIT = 100_000


def _label_value(value: str) -> int:
    """Convert a CICIDS-style label to binary benign/attack form."""
    return 0 if value.strip().casefold() == "benign" else 1


def _is_float(value: str) -> bool:
    """Return whether a CSV cell can be parsed as a float."""
    try:
        float(value.strip())
        return True
    except (TypeError, ValueError):
        return False


class RequestSource:
    """Iterate over (request_id, float32 feature vector, optional true label)."""

    def __init__(
        self,
        *,
        length: int,
        feature_dim: int,
        rows: np.ndarray | None = None,
        labels: list[int | None] | None = None,
        path: Path | None = None,
        has_header: bool = False,
        feature_columns: list[int] | None = None,
        label_column: int | None = None,
        csv_row_count: int | None = None,
    ) -> None:
        """Initialize an in-memory or streaming request source."""
        self._length = length
        self._feature_dim = feature_dim
        self._rows = rows
        self._labels = labels
        self._path = path
        self._has_header = has_header
        self._feature_columns = feature_columns or []
        self._label_column = label_column
        self._csv_row_count = csv_row_count or 0

    def __len__(self) -> int:
        """Return the number of requests this source will yield."""
        return self._length

    def __iter__(self) -> Iterator[tuple[str, np.ndarray, int | None]]:
        """Yield request IDs, feature vectors, and optional labels."""
        if self._rows is not None:
            row_count = len(self._rows)
            for index in range(self._length):
                source_index = index % row_count
                yield (
                    f"req_{index:06d}",
                    self._rows[source_index].copy(),
                    self._labels[source_index] if self._labels is not None else None,
                )
            return

        if self._path is None:
            return

        # Large CSVs are reopened and streamed; cycling does not retain them in memory.
        produced = 0
        while produced < self._length:
            with self._path.open("r", encoding="latin-1", newline="") as stream:
                reader = csv.reader(stream)
                if self._has_header:
                    next(reader, None)
                for row in reader:
                    if produced >= self._length:
                        break
                    try:
                        values = [float(row[column].strip()) for column in self._feature_columns]
                    except (IndexError, ValueError):
                        LOG.warning("Skipping malformed CSV row near request %d", produced)
                        continue
                    label = None
                    if self._label_column is not None:
                        try:
                            label = _label_value(row[self._label_column])
                        except IndexError:
                            LOG.warning("Missing label in CSV row near request %d", produced)
                    features = np.asarray(values, dtype=np.float32)
                    invalid_count = int((~np.isfinite(features)).sum())
                    if invalid_count:
                        LOG.warning(
                            "RequestSource: replacing %d non-finite feature value(s) with 0.0",
                            invalid_count,
                        )
                        features = np.nan_to_num(
                            features, nan=0.0, posinf=0.0, neginf=0.0
                        )
                    yield f"req_{produced:06d}", features, label
                    produced += 1
            if self._csv_row_count == 0:
                break

    @classmethod
    def from_csv(
        cls,
        path: Path,
        feature_dim: int = 77,
        num_requests: int | None = None,
        label_col: str | None = "auto",
    ) -> "RequestSource":
        """Build a source from a CSV, retaining small files and streaming large ones."""
        csv_path = Path(path).expanduser().resolve()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if num_requests is not None and num_requests < 0:
            raise ValueError("num_requests cannot be negative")
        if not csv_path.is_file():
            raise FileNotFoundError(f"input CSV does not exist: {csv_path}")

        with csv_path.open("r", encoding="latin-1", newline="") as stream:
            reader = csv.reader(stream)
            first = next(reader, None)
            if first is None:
                raise ValueError(f"CSV is empty: {csv_path}")
            has_header = not all(_is_float(cell) for cell in first[:feature_dim])
            header = first if has_header else None
            first_data = next(reader, None) if has_header else first
            if first_data is None:
                raise ValueError(f"CSV has a header but no data rows: {csv_path}")

        label_index = None
        if label_col == "auto" and header is not None:
            for index, name in enumerate(header):
                if name.strip().casefold() in {"label", "labels"}:
                    label_index = index
                    break
        elif label_col is not None and label_col != "auto" and header is not None:
            wanted = label_col.strip().casefold()
            for index, name in enumerate(header):
                if name.strip().casefold() == wanted:
                    label_index = index
                    break

        # Identify numeric feature columns from the first data row, excluding label.
        feature_columns = []
        for index, cell in enumerate(first_data):
            if index == label_index:
                continue
            if _is_float(cell):
                feature_columns.append(index)
            if len(feature_columns) == feature_dim:
                break
        if len(feature_columns) < feature_dim:
            raise ValueError(
                f"CSV has only {len(feature_columns)} numeric feature columns; "
                f"{feature_dim} required after dropping the label column"
            )

        # Count rows first. Files under the limit are loaded; larger files stream.
        with csv_path.open("r", encoding="latin-1", newline="") as stream:
            reader = csv.reader(stream)
            if has_header:
                next(reader, None)
            row_count = sum(1 for _ in reader)
        if row_count == 0:
            raise ValueError(f"CSV has no data rows: {csv_path}")
        if row_count >= _MEMORY_ROW_LIMIT and num_requests is None:
            raise ValueError(
                f"CSV has {row_count} rows; provide num_requests to stream files "
                f"with {_MEMORY_ROW_LIMIT} rows or more"
            )

        length = row_count if num_requests is None else num_requests
        LOG.warning(
            "RequestSource: features are used as-is; if the model expects standardized "
            "inputs, results will be inaccurate."
        )

        if row_count >= _MEMORY_ROW_LIMIT:
            return cls(
                length=length,
                feature_dim=feature_dim,
                path=csv_path,
                has_header=has_header,
                feature_columns=feature_columns,
                label_column=label_index,
                csv_row_count=row_count,
            )

        features_list: list[list[float]] = []
        labels: list[int | None] = []
        with csv_path.open("r", encoding="latin-1", newline="") as stream:
            reader = csv.reader(stream)
            if has_header:
                next(reader, None)
            for row_number, row in enumerate(reader, start=1):
                try:
                    features_list.append(
                        [float(row[column].strip()) for column in feature_columns]
                    )
                    labels.append(
                        _label_value(row[label_index])
                        if label_index is not None
                        else None
                    )
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric feature data in CSV row {row_number + int(has_header)}"
                    ) from exc

        features = np.asarray(features_list, dtype=np.float32)
        invalid_count = int((~np.isfinite(features)).sum())
        if invalid_count:
            LOG.warning(
                "RequestSource: replacing %d non-finite feature value(s) with 0.0",
                invalid_count,
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return cls(
            length=length,
            feature_dim=feature_dim,
            rows=features,
            labels=labels,
        )

    @classmethod
    def synthetic(
        cls,
        n: int = 1000,
        feature_dim: int = 77,
        seed: int = 42,
    ) -> "RequestSource":
        """Create deterministic random features and binary labels for plumbing tests."""
        if n < 0 or feature_dim < 1:
            raise ValueError("n must be non-negative and feature_dim must be positive")
        LOG.warning(
            "RequestSource: synthetic data — for plumbing tests only, not for quality evaluation."
        )
        rng = np.random.default_rng(seed)
        features = rng.standard_normal((n, feature_dim)).astype(np.float32)
        labels = rng.integers(0, 2, size=n).tolist()
        return cls(length=n, feature_dim=feature_dim, rows=features, labels=labels)
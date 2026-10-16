"""Stream Parquet batches instead of loading millions of spectra at once."""

from collections import Counter
from pathlib import Path
from typing import Iterator
import math

import pyarrow.parquet as pq

REFERENCE_COLUMNS = [
    "ms2_mzs", "ms2_normalized_intensities",
    "adduct", "ionization_mode", "precursor_mz",
]
SPECTRUM_COLUMNS = ["molecule_id", "spectrum_id", *REFERENCE_COLUMNS]


def rows(path: str | Path, columns: list[str], batch_size: int = 4096) -> Iterator[dict]:
    source = pq.ParquetFile(path)
    missing = set(columns) - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")
    for batch in source.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def validate_spectrum(row: dict, *, require_ids: bool = True) -> None:
    names = ["adduct", "ionization_mode"]
    if require_ids:
        names += ["molecule_id", "spectrum_id"]
    for name in names:
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"Missing or invalid {name}")
    if row["ionization_mode"] not in {"positive", "negative"}:
        raise ValueError(f"Invalid ionization_mode: {row['ionization_mode']!r}")
    mass = row["precursor_mz"]
    if mass is None or not math.isfinite(mass) or mass <= 0:
        raise ValueError("Invalid precursor_mz")
    mzs, intensities = row["ms2_mzs"], row["ms2_normalized_intensities"]
    if mzs is None or intensities is None or len(mzs) != len(intensities):
        raise ValueError("Peak arrays must be present and aligned")


def inspect(path: str | Path) -> dict:
    source = pq.ParquetFile(path)
    names = source.schema_arrow.names
    columns = [c for c in ["adduct", "ionization_mode", "instrument_type", "ingest_lib"] if c in names]
    counts = {name: Counter() for name in columns}
    for row in rows(path, columns):
        for name in columns:
            value = row[name]
            counts[name]["<null>" if value is None else str(value)] += 1
    return {
        "path": str(path), "rows": source.metadata.num_rows,
        "columns": names, "counts": {name: dict(c.most_common()) for name, c in counts.items()},
    }

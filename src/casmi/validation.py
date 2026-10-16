"""Reproducible reference/query splits; no row-random validation default."""

from contextlib import ExitStack
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .chemistry import require_metric_rdkit, structure_key


def split_dataset(source, output_dir, fraction=0.1, seed=42, query_library=None) -> dict:
    require_metric_rdkit()
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    parquet = pq.ParquetFile(source)
    if "normalized_smiles" not in parquet.schema_arrow.names:
        raise ValueError("Split requires training labels")
    if query_library is not None and "ingest_lib" not in parquet.schema_arrow.names:
        raise ValueError("Cross-library split requires ingest_lib")
    destination = Path(output_dir)
    # Never replace an existing experiment directory.
    destination.mkdir(parents=True, exist_ok=False)
    paths = {name: destination / f"{name}.parquet" for name in ["reference", "validation"]}
    counts = {name: 0 for name in paths}
    try:
        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(pq.ParquetWriter(path, parquet.schema_arrow))
                for name, path in paths.items()
            }
            for batch in parquet.iter_batches(batch_size=4096):
                labels = batch.column(batch.schema.get_field_index("normalized_smiles")).to_pylist()
                if query_library is not None:
                    libraries = batch.column(batch.schema.get_field_index("ingest_lib")).to_pylist()
                    selected = [value == query_library for value in libraries]
                else:
                    selected = []
                    for label in labels:
                        key = structure_key(label)
                        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
                        selected.append(int.from_bytes(digest[:8], "big") / 2**64 < fraction)
                table = pa.Table.from_batches([batch])
                for name, mask in [
                    ("validation", selected), ("reference", [not value for value in selected]),
                ]:
                    subset = table.filter(pa.array(mask))
                    if len(subset):
                        writers[name].write_table(subset)
                        counts[name] += len(subset)
        if not all(counts.values()):
            raise ValueError("Split produced an empty partition; change the fraction or library")
    except BaseException:
        for path in paths.values():
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise
    return {
        "mode": "cross_library" if query_library else "structure_disjoint",
        "seed": seed, "fraction": fraction if query_library is None else None,
        "query_library": query_library, "spectra": counts,
        "paths": {name: str(path) for name, path in paths.items()},
    }

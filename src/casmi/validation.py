"""Reproducible reference/query splits; no row-random validation default."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .chemistry import require_metric_rdkit, structure_key


def split_dataset(
    source, output_dir, fraction=0.1, seed=42, query_library=None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    require_metric_rdkit()
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    parquet = pq.ParquetFile(source)
    if "normalized_smiles" not in parquet.schema_arrow.names:
        raise ValueError("Split requires training labels")
    if query_library is not None and "ingest_lib" not in parquet.schema_arrow.names:
        raise ValueError("Cross-library split requires ingest_lib")
    schema = parquet.schema_arrow
    generated_ids = [name for name in ["molecule_id", "spectrum_id"] if name not in schema.names]
    for name in generated_ids:
        schema = schema.append(pa.field(name, pa.string()))
    schemas = {
        "validation": schema,
        # Reference indexing does not need anonymous IDs. A cross-library
        # split should not canonicalize reference molecules merely to add them.
        "reference": parquet.schema_arrow if query_library is not None else schema,
    }
    destination = Path(output_dir)
    # Never replace an existing experiment directory.
    destination.mkdir(parents=True, exist_ok=False)
    paths = {name: destination / f"{name}.parquet" for name in ["reference", "validation"]}
    counts = {name: 0 for name in paths}
    try:
        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(pq.ParquetWriter(path, schemas[name]))
                for name, path in paths.items()
            }
            row_offset = 0
            next_report = 100_000
            for batch in parquet.iter_batches(batch_size=4096):
                keys = None
                if query_library is not None:
                    libraries = batch.column(batch.schema.get_field_index("ingest_lib")).to_pylist()
                    selected = [value == query_library for value in libraries]
                else:
                    labels = batch.column(batch.schema.get_field_index("normalized_smiles")).to_pylist()
                    keys = [structure_key(label) for label in labels]
                    selected = []
                    for key in keys:
                        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
                        selected.append(int.from_bytes(digest[:8], "big") / 2**64 < fraction)
                table = pa.Table.from_batches([batch])
                for name, mask in [
                    ("validation", selected), ("reference", [not value for value in selected]),
                ]:
                    subset = table.filter(pa.array(mask))
                    if len(subset):
                        if name == "validation" or query_library is None:
                            if "molecule_id" in generated_ids:
                                if keys is None:
                                    subset_keys = [structure_key(label) for label in
                                                   subset["normalized_smiles"].to_pylist()]
                                else:
                                    subset_keys = [key for key, keep in zip(keys, mask) if keep]
                                subset = subset.append_column(
                                    "molecule_id", pa.array(subset_keys, type=pa.string()),
                                )
                            if "spectrum_id" in generated_ids:
                                subset = subset.append_column("spectrum_id", pa.array([
                                    f"local_spectrum_{row_offset + i}"
                                    for i, keep in enumerate(mask) if keep
                                ], type=pa.string()))
                        writers[name].write_table(subset)
                        counts[name] += len(subset)
                row_offset += len(batch)
                if progress is not None and (
                    row_offset >= next_report or row_offset == parquet.metadata.num_rows
                ):
                    progress({
                        "processed_spectra": row_offset,
                        "total_spectra": parquet.metadata.num_rows,
                        "partitions": dict(counts),
                    })
                    next_report = row_offset + 100_000
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
        "generated_id_columns": generated_ids,
        "paths": {name: str(path) for name, path in paths.items()},
    }

"""Disk-backed reference library: indexed mass filtering, streamed peak blobs."""

from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Callable

import numpy as np
import pyarrow.dataset as ds
from rdkit import rdBase

from .chemistry import neutral_mass, require_metric_rdkit, structure_info
from .data import REFERENCE_COLUMNS, rows, validate_spectrum
from .spectra import SearchConfig, cosine, prepare

SCHEMA_VERSION = 1


def build_library(
    train_path, output, config: SearchConfig, *, include_libraries=None,
    exclude_libraries=None, max_spectra: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    require_metric_rdkit()
    if isinstance(include_libraries, str) or isinstance(exclude_libraries, str):
        raise ValueError("Library selection must be a list of library names")
    included, excluded = set(include_libraries or []), set(exclude_libraries or [])
    if max_spectra is not None and max_spectra < 1:
        raise ValueError("max_spectra must be positive")
    selection = None
    if included:
        selection = ds.field("ingest_lib").isin(sorted(included))
    if excluded:
        keep = ~ds.field("ingest_lib").isin(sorted(excluded))
        selection = keep if selection is None else selection & keep
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite library: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    # Exclusive file creation protects an existing/interrupted build.
    with temporary.open("xb"):
        pass
    counts = Counter()
    seen_structures = set()
    try:
        with closing(sqlite3.connect(temporary)) as db:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE structures (
                    structure_key TEXT PRIMARY KEY, smiles TEXT NOT NULL,
                    neutral_mass REAL NOT NULL
                );
                CREATE TABLE spectra (
                    id INTEGER PRIMARY KEY, structure_key TEXT NOT NULL,
                    precursor_mz REAL NOT NULL, adduct TEXT NOT NULL,
                    mode TEXT NOT NULL, mzs BLOB NOT NULL, weights BLOB NOT NULL
                );
            """)
            for row in rows(
                train_path, REFERENCE_COLUMNS + ["normalized_smiles"],
                filter_expression=selection,
            ):
                counts["input_spectra"] += 1
                if progress is not None and counts["input_spectra"] % 10_000 == 0:
                    progress({"phase": "reading", "counts": dict(counts)})
                try:
                    validate_spectrum(row, require_ids=False)
                    key, smiles, mass = structure_info(row["normalized_smiles"])
                    mz, weights = prepare(
                        row["ms2_mzs"], row["ms2_normalized_intensities"],
                        row["precursor_mz"], config,
                    )
                except (ValueError, TypeError, OverflowError):
                    counts["invalid_rows"] += 1
                    continue
                if key not in seen_structures:
                    db.execute("INSERT INTO structures VALUES (?, ?, ?)", (key, smiles, mass))
                    seen_structures.add(key)
                if not len(mz):
                    counts["empty_spectra"] += 1
                    continue
                db.execute(
                    "INSERT INTO spectra VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                    (key, row["precursor_mz"], row["adduct"], row["ionization_mode"],
                     mz.astype("<f8").tobytes(), weights.astype("<f8").tobytes()),
                )
                counts["indexed_spectra"] += 1
                if counts["input_spectra"] % 10_000 == 0:
                    db.commit()
                if max_spectra is not None and counts["indexed_spectra"] >= max_spectra:
                    break
            if not counts["indexed_spectra"]:
                raise ValueError("No usable reference spectra were found")
            if progress is not None:
                progress({"phase": "creating_indexes", "counts": dict(counts)})
            db.executescript("""
                CREATE INDEX spectra_lookup ON spectra(adduct, mode, precursor_mz);
                CREATE INDEX structures_mass ON structures(neutral_mass);
            """)
            counts["structures"] = len(seen_structures)
            metadata = {
                "schema_version": SCHEMA_VERSION, "rdkit": rdBase.rdkitVersion,
                "config": config.to_dict(), "counts": dict(counts),
                "source": str(Path(train_path).resolve()),
                "source_bytes": Path(train_path).stat().st_size,
                "selection": {
                    "include_libraries": sorted(included),
                    "exclude_libraries": sorted(excluded),
                    "max_spectra": max_spectra,
                },
            }
            db.execute("INSERT INTO metadata VALUES ('build', ?)", (json.dumps(metadata),))
            db.commit()
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("Reference database integrity check failed")
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if progress is not None:
        progress({"phase": "complete", "counts": dict(counts)})
    return metadata


class SpectralLibrary:
    def __init__(self, path):
        require_metric_rdkit()
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            record = self.db.execute("SELECT value FROM metadata WHERE key='build'").fetchone()
            if record is None:
                raise ValueError("Missing library metadata")
            self.metadata = json.loads(record[0])
            if self.metadata["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unsupported library schema; rebuild the index")
            if self.metadata["rdkit"] != rdBase.rdkitVersion:
                raise ValueError("RDKit version differs from index; rebuild the index")
            self.config = SearchConfig(**self.metadata["config"])
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def _nearest_structures(self, mass: float, limit: int):
        # Two indexed scans avoid sorting the entire structure table per query.
        below = self.db.execute(
            "SELECT structure_key, smiles, neutral_mass FROM structures "
            "WHERE neutral_mass <= ? ORDER BY neutral_mass DESC LIMIT ?", (mass, limit),
        ).fetchall()
        above = self.db.execute(
            "SELECT structure_key, smiles, neutral_mass FROM structures "
            "WHERE neutral_mass > ? ORDER BY neutral_mass ASC LIMIT ?", (mass, limit),
        ).fetchall()
        return sorted(below + above, key=lambda row: (abs(row[2] - mass), row[0]))[:limit]

    def predict(self, spectra: list[dict], top_k: int = 25) -> tuple[list[str], dict]:
        if not spectra or not 1 <= top_k <= 25:
            raise ValueError("Provide spectra and a top_k between 1 and 25")
        if len({row["molecule_id"] for row in spectra}) != 1:
            raise ValueError("Prediction requires spectra from exactly one molecule")
        config = self.config
        total_scores = defaultdict(float)
        masses = []
        informative = 0
        compared = 0
        for row in spectra:
            validate_spectrum(row)
            expected_mode = "positive" if row["adduct"].endswith("+") else "negative"
            if row["ionization_mode"] != expected_mode:
                raise ValueError("Adduct and ionization mode disagree")
            masses.append(neutral_mass(row["precursor_mz"], row["adduct"]))
            mz, weights = prepare(
                row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"], config,
            )
            if not len(mz):
                continue
            informative += 1
            tolerance = max(config.precursor_da, row["precursor_mz"] * config.precursor_ppm / 1e6)
            candidates = self.db.execute(
                "SELECT structure_key, mzs, weights FROM spectra "
                "WHERE adduct=? AND mode=? AND precursor_mz BETWEEN ? AND ?",
                (row["adduct"], row["ionization_mode"],
                 row["precursor_mz"] - tolerance, row["precursor_mz"] + tolerance),
            )
            # Best reference match per structure per query spectrum. Repeated
            # reference acquisitions cannot increase a score just by summing.
            best = {}
            for key, mz_blob, weight_blob in candidates:
                compared += 1
                score, matches = cosine(
                    mz, weights, np.frombuffer(mz_blob, dtype="<f8"),
                    np.frombuffer(weight_blob, dtype="<f8"), config.fragment_da,
                )
                if matches >= config.min_matched_peaks and score > best.get(key, 0):
                    best[key] = score
            for key, score in best.items():
                total_scores[key] += score
        ranked_keys = sorted(total_scores, key=lambda key: (-total_scores[key], key))[:top_k]
        guesses = [self.db.execute(
            "SELECT smiles FROM structures WHERE structure_key=?", (key,),
        ).fetchone()[0] for key in ranked_keys]
        spectral_count = len(guesses)
        mass = statistics.median(masses)
        mass_fallback = []
        for key, smiles, candidate_mass in self._nearest_structures(mass, top_k):
            if len(guesses) == top_k:
                break
            if key not in ranked_keys:
                guesses.append(smiles)
                mass_fallback.append(abs(candidate_mass - mass) / mass * 1e6)
        if not guesses:
            raise ValueError("Library contains no candidate structures")
        return guesses, {
            "spectra": len(spectra), "informative_spectra": informative,
            "reference_spectra_compared": compared, "spectral_candidates": spectral_count,
            "mass_only_candidates": len(mass_fallback),
            "mass_only_error_ppm": mass_fallback,
            "neutral_mass_estimate": mass,
            "neutral_mass_spread_ppm": (max(masses) - min(masses)) / mass * 1e6,
            "best_spectral_score": max(total_scores.values(), default=0) / max(informative, 1),
        }

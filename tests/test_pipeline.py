import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi.chemistry import PROTON, structure_key
from casmi.cli import predict_file
from casmi.data import inspect
from casmi.evaluation import evaluate_files
from casmi.library import SpectralLibrary, build_library
from casmi.spectra import SearchConfig
from casmi.submission import read_submission
from casmi.validation import split_dataset


def spectrum(identifier, smiles="CCO", mzs=None, adduct="[M+H]+", mass=46.041864812):
    return {
        "molecule_id": identifier.split("_")[0], "spectrum_id": identifier,
        "ms2_mzs": mzs if mzs is not None else [15.0, 25.0, 35.0],
        "ms2_normalized_intensities": [1.0, 0.5, 0.25],
        "adduct": adduct, "ionization_mode": "positive",
        "precursor_mz": mass + PROTON,
        "normalized_smiles": smiles, "ingest_lib": "reference",
    }


def parquet(path, records):
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def actual_training_row(identifier, smiles="CCO", library="reference"):
    """Match the user's observed 18-column schema, without anonymous IDs."""
    row = spectrum(identifier, smiles)
    del row["molecule_id"]
    del row["spectrum_id"]
    row.update({
        "ingest_lib": library, "inchikey": structure_key(smiles) + "-TESTKEY",
        "inchikey14": structure_key(smiles), "molecular_formula": "C2H6O",
        "instrument_type": "timsTOF", "adduct_orig": "[M+H]+",
        "precursor_error_ppm": 0.0, "num_peaks": 3,
        "base_peak_intensity": 10000.0, "collision_energy_ev": [20.0],
        "collision_energy_orig": "20", "collision_energy_orig_units": "eV",
    })
    assert len(row) == 18
    return row


def test_end_to_end_parquet_to_submission_and_score(tmp_path):
    train = parquet(tmp_path / "train.parquet", [
        spectrum("ethanol_1"),
        spectrum("ethanol_2"),
        # Same formula/mass, distinct structure and different fragments.
        spectrum("ether_1", "COC", [16.0, 26.0, 36.0]),
    ])
    query = [spectrum("unknown_1"), spectrum("unknown_2")]
    test = parquet(tmp_path / "test.parquet", query)
    index = tmp_path / "library.sqlite"
    built = build_library(train, index, SearchConfig())
    assert built["counts"]["indexed_spectra"] == 3
    output = tmp_path / "submission.csv"
    run = predict_file(index, test, output)
    assert run["molecules"] == 1
    assert run["molecules_without_spectral_match"] == 0
    predictions = read_submission(output, {"unknown"})
    assert structure_key(predictions["unknown"][0]) == structure_key("CCO")
    assert len({structure_key(s) for s in predictions["unknown"]}) == len(predictions["unknown"])
    assert evaluate_files(output, test)["mrr_at_25"] == 1.0
    assert output.with_suffix(".report.json").exists()
    assert inspect(train)["rows"] == 3
    checked = subprocess.run([
        sys.executable, "-m", "casmi", "check-submission", "--submission", str(output),
        "--test", str(test),
    ], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert '"valid": true' in checked.stdout


def test_aggregation_uses_each_spectrum_and_does_not_sum_reference_duplicates(tmp_path):
    index = tmp_path / "library.sqlite"
    train = parquet(tmp_path / "train.parquet", [
        spectrum("ethanol_1"), spectrum("ethanol_2"),
        spectrum("ether_1", "COC", [16.0, 26.0, 36.0]),
    ])
    build_library(train, index, SearchConfig())
    reference = SpectralLibrary(index)
    try:
        _, details = reference.predict([
            spectrum("unknown_1"), spectrum("unknown_2", mzs=[16.0, 26.0, 36.0]),
        ])
        assert details["spectral_candidates"] == 2
        assert details["best_spectral_score"] == pytest.approx(0.5)
    finally:
        reference.close()


def test_no_match_has_explicit_mass_fallback_and_different_adduct_is_not_compared(tmp_path):
    index = tmp_path / "library.sqlite"
    build_library(parquet(tmp_path / "train.parquet", [spectrum("a_1")]), index, SearchConfig())
    reference = SpectralLibrary(index)
    try:
        query = spectrum("query_1", adduct="[M+Na]+")
        query["precursor_mz"] = 46.041864812 + 22.989220702
        guesses, details = reference.predict([query])
        assert guesses
        assert details["reference_spectra_compared"] == 0
        assert details["spectral_candidates"] == 0
        assert details["mass_only_candidates"] == 1
        assert details["mass_only_error_ppm"][0] < 0.1
    finally:
        reference.close()


def test_structure_split_keeps_tautomers_together_and_is_repeatable(tmp_path):
    records = [spectrum(f"mol{i}_1", "C" * i) for i in range(1, 21)]
    records += [spectrum("tautomer1_1", "CC(=O)C"), spectrum("tautomer2_1", "C=C(O)C")]
    source = parquet(tmp_path / "train.parquet", records)
    for name in ["split1", "split2"]:
        split_dataset(source, tmp_path / name, fraction=0.5)
    key_sets = {}
    for name in ["reference", "validation"]:
        first = pq.read_table(tmp_path / "split1" / f"{name}.parquet").to_pylist()
        second = pq.read_table(tmp_path / "split2" / f"{name}.parquet").to_pylist()
        assert first == second
        key_sets[name] = {structure_key(row["normalized_smiles"]) for row in first}
    assert not key_sets["reference"] & key_sets["validation"]


def test_cross_library_split_and_no_overwrite(tmp_path):
    query = spectrum("query_1")
    query["ingest_lib"] = "enveda-np-examples"
    source = parquet(tmp_path / "train.parquet", [spectrum("reference_1"), query])
    result = split_dataset(source, tmp_path / "split", query_library="enveda-np-examples")
    assert result["spectra"] == {"reference": 1, "validation": 1}
    with pytest.raises(FileExistsError):
        split_dataset(source, tmp_path / "split")
    index = tmp_path / "library.sqlite"
    build_library(source, index, SearchConfig())
    original = index.read_bytes()
    with pytest.raises(FileExistsError):
        build_library(source, index, SearchConfig())
    assert index.read_bytes() == original


def test_kaggle_notebook_code_cells_execute_with_local_fixture(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    index = tmp_path / "library.sqlite"
    train = parquet(tmp_path / "train.parquet", [spectrum("reference_1")])
    # Hide labels entirely in the inference input, as in the actual test schema.
    query = spectrum("unknown_1")
    del query["normalized_smiles"]
    parquet(tmp_path / "test.parquet", [query])
    build_library(train, index, SearchConfig())
    notebook = json.loads((repository / "notebooks/kaggle_inference.ipynb").read_text())
    script = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    runtime = os.environ.copy()
    runtime.pop("CASMI_WHEELHOUSE", None)
    runtime.update({
        "CASMI_SOURCE_ROOT": str(repository), "CASMI_DATA_ROOT": str(tmp_path),
        "CASMI_INDEX_PATH": str(index), "CASMI_OUTPUT": str(tmp_path / "submission.csv"),
    })
    executed = subprocess.run(
        [sys.executable, "-c", script], env=runtime, capture_output=True, text=True,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    guesses = read_submission(tmp_path / "submission.csv", {"unknown"})
    assert structure_key(guesses["unknown"][0]) == structure_key("CCO")


def test_reference_index_accepts_actual_training_schema_without_ids(tmp_path):
    source = parquet(tmp_path / "train.parquet", [actual_training_row("a")])
    index = tmp_path / "library.sqlite"
    assert build_library(source, index, SearchConfig())["counts"]["indexed_spectra"] == 1
    query = spectrum("anonymous_1")
    del query["normalized_smiles"]
    test = parquet(tmp_path / "test.parquet", [query])
    output = tmp_path / "submission.csv"
    assert predict_file(index, test, output)["molecules_without_spectral_match"] == 0
    guesses = read_submission(output, {"anonymous"})
    assert structure_key(guesses["anonymous"][0]) == structure_key("CCO")


def test_actual_schema_cross_library_split_can_predict_and_evaluate(tmp_path):
    records = [
        actual_training_row("reference"),
        actual_training_row("query1", library="enveda-np-examples"),
        actual_training_row("query2", library="enveda-np-examples"),
    ]
    source = parquet(tmp_path / "train.parquet", records)
    original_schema = pq.ParquetFile(source).schema_arrow
    split = split_dataset(source, tmp_path / "split", query_library="enveda-np-examples")
    assert split["generated_id_columns"] == ["molecule_id", "spectrum_id"]
    reference, validation = split["paths"]["reference"], split["paths"]["validation"]
    query_rows = pq.read_table(validation).to_pylist()
    assert len({row["molecule_id"] for row in query_rows}) == 1
    assert len({row["spectrum_id"] for row in query_rows}) == 2
    assert pq.ParquetFile(source).schema_arrow == original_schema
    index = tmp_path / "library.sqlite"
    build_library(reference, index, SearchConfig())
    output = tmp_path / "submission.csv"
    assert predict_file(index, validation, output)["molecules"] == 1
    assert evaluate_files(output, validation)["mrr_at_25"] == 1.0


def test_actual_schema_structure_split_groups_identities_across_batches(tmp_path):
    # More than one 4096-row batch verifies generated IDs use global positions.
    records = [actual_training_row(str(i), "C" * (i % 20 + 1)) for i in range(4100)]
    source = parquet(tmp_path / "train.parquet", records)
    split = split_dataset(source, tmp_path / "split", fraction=0.5)
    reference = pq.read_table(split["paths"]["reference"]).to_pylist()
    validation = pq.read_table(split["paths"]["validation"]).to_pylist()
    assert len({row["spectrum_id"] for row in reference + validation}) == 4100
    assert not {row["molecule_id"] for row in reference} & {row["molecule_id"] for row in validation}
    for row in reference + validation:
        assert row["molecule_id"] == structure_key(row["normalized_smiles"])


def test_cross_library_split_does_not_canonicalize_references_and_reports_progress(tmp_path, monkeypatch):
    import casmi.validation as validation_module

    records = [
        actual_training_row("reference", smiles="COC"),
        actual_training_row("query", library="enveda-np-examples"),
    ]
    source = parquet(tmp_path / "train.parquet", records)
    calls, progress = [], []

    def tracked_key(smiles):
        calls.append(smiles)
        assert smiles != "COC", "Reference chemistry must not be processed during this split"
        return structure_key(smiles)

    monkeypatch.setattr(validation_module, "structure_key", tracked_key)
    result = split_dataset(
        source, tmp_path / "split", query_library="enveda-np-examples", progress=progress.append,
    )
    assert calls == ["CCO"]
    assert progress == [{
        "processed_spectra": 2, "total_spectra": 2,
        "partitions": {"reference": 1, "validation": 1},
    }]
    assert pq.ParquetFile(result["paths"]["reference"]).schema_arrow == pq.ParquetFile(source).schema_arrow
    assert "molecule_id" in pq.ParquetFile(result["paths"]["validation"]).schema_arrow.names

import pytest

from casmi.chemistry import ADDUCT_DELTAS, neutral_mass, structure_key
from casmi.evaluation import evaluate
from casmi.submission import read_submission, write_submission


def test_mrr_first_correct_rank_including_invalid_guesses():
    predictions = {
        "first": ["CCO"],
        "second": ["CC", "CCO", "CCO"],
        "last": ["CC"] * 24 + ["CCO"],
        "miss": ["not a smiles"],
    }
    result = evaluate(predictions, {identifier: "CCO" for identifier in predictions})
    assert result["mrr_at_25"] == pytest.approx((1 + 0.5 + 0.04) / 4)
    assert result["top_1_accuracy"] == 0.25
    assert result["top_5_accuracy"] == 0.5
    assert result["top_25_accuracy"] == 0.75


def test_matching_ignores_stereochemistry_and_tautomer_form():
    assert structure_key("N[C@@H](C)C(=O)O") == structure_key("N[C@H](C)C(=O)O")
    assert structure_key("CC(=O)C") == structure_key("C=C(O)C")
    assert structure_key("CCO") != structure_key("COC")


def test_all_test_adducts_recover_neutral_mass():
    for adduct, delta in ADDUCT_DELTAS.items():
        assert neutral_mass(348.123 + delta, adduct) == pytest.approx(348.123)
    with pytest.raises(ValueError, match="Unsupported"):
        neutral_mass(100, "[M+2H]2+")


def test_submission_roundtrip_and_coverage(tmp_path):
    path = tmp_path / "submission.csv"
    expected = {"a": ["CCO", "CC"]}
    write_submission(expected, path)
    assert read_submission(path, {"a"}) == expected
    with pytest.raises(ValueError, match="ID mismatch"):
        read_submission(path, {"a", "b"})
    with pytest.raises(FileExistsError):
        write_submission(expected, path)


@pytest.mark.parametrize("body", [
    "molecule_id,smiles\na,CCO\na,CC\n",
    "molecule_id,smiles\na,\n",
    "molecule_id,smiles\n",
    "molecule_id,wrong\na,CC\n",
    "molecule_id,smiles\na," + ";".join(["CC"] * 26) + "\n",
])
def test_rejects_malformed_submission(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises(ValueError):
        read_submission(path)

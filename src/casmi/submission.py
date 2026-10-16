"""Submission format checks and CSV I/O; each list position is preserved."""

import csv
from pathlib import Path

from .chemistry import structure_key


def read_submission(path, expected_ids=None, check_chemistry: bool = True) -> dict[str, list[str]]:
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"molecule_id", "smiles"} <= set(reader.fieldnames):
            raise ValueError("Submission requires molecule_id and smiles columns")
        for row in reader:
            identifier, field = row["molecule_id"], row["smiles"]
            if not identifier or not identifier.strip() or not field or not field.strip():
                raise ValueError("Submission contains empty molecule_id or smiles")
            if identifier in result:
                raise ValueError(f"Repeated molecule_id: {identifier}")
            guesses = field.split(";")
            if len(guesses) > 25 or any(not guess.strip() for guess in guesses):
                raise ValueError(f"Expected 1–25 nonempty guesses for {identifier}")
            if check_chemistry:
                for guess in guesses:
                    structure_key(guess)
            result[identifier] = guesses
    if not result:
        raise ValueError("Submission is empty")
    if expected_ids is not None:
        missing, extra = set(expected_ids) - result.keys(), result.keys() - set(expected_ids)
        if missing or extra:
            raise ValueError(f"Molecule ID mismatch: {len(missing)} missing, {len(extra)} extra")
    return result


def write_submission(predictions: dict[str, list[str]], path) -> None:
    if not predictions:
        raise ValueError("No predictions to write")
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite submission: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate all rows before creating an output; callers deduplicate candidates.
    for identifier, guesses in predictions.items():
        if not isinstance(identifier, str) or not identifier.strip() or not 1 <= len(guesses) <= 25:
            raise ValueError("Invalid molecule ID or number of guesses")
        for guess in guesses:
            if ";" in guess:
                raise ValueError("A candidate must contain exactly one SMILES")
            structure_key(guess)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "smiles"])
        for identifier, guesses in sorted(predictions.items()):
            writer.writerow([identifier, ";".join(guesses)])

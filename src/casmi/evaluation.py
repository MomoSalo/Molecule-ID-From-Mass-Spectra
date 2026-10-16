"""Local MRR@25 based on the supplied competition matching description.

Confirm parity against the official metric notebook when it is available.
"""

from .chemistry import require_metric_rdkit, structure_key
from .data import rows
from .submission import read_submission


def evaluate(predictions: dict[str, list[str]], truth: dict[str, str]) -> dict:
    require_metric_rdkit()
    if not truth or predictions.keys() != truth.keys():
        raise ValueError("Predictions and nonempty truth must cover exactly the same molecules")
    ranks = []
    for identifier, label in truth.items():
        guesses = predictions[identifier]
        if not 1 <= len(guesses) <= 25:
            raise ValueError("Each molecule requires 1–25 candidates")
        target = structure_key(label)
        rank = None
        for position, guess in enumerate(guesses, start=1):
            try:
                correct = structure_key(guess) == target
            except (ValueError, TypeError):
                correct = False
            if correct:
                rank = position
                break
        ranks.append(rank)
    count = len(ranks)
    return {
        "molecules": count,
        "mrr_at_25": sum(1 / rank for rank in ranks if rank is not None) / count,
        **{f"top_{k}_accuracy": sum(r is not None and r <= k for r in ranks) / count
           for k in [1, 5, 25]},
    }


def evaluate_files(submission, labels) -> dict:
    truth = {}
    for row in rows(labels, ["molecule_id", "normalized_smiles"]):
        identifier, label = row["molecule_id"], row["normalized_smiles"]
        if identifier in truth and structure_key(truth[identifier]) != structure_key(label):
            raise ValueError(f"Conflicting labels for molecule_id {identifier}")
        truth[identifier] = label
    predictions = read_submission(submission, truth.keys(), check_chemistry=False)
    return evaluate(predictions, truth)

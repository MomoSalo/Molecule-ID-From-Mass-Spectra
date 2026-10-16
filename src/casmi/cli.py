"""Command-line entry points shared by local development and Kaggle."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from .chemistry import require_metric_rdkit
from .data import SPECTRUM_COLUMNS, inspect, rows
from .evaluation import evaluate_files
from .library import SpectralLibrary, build_library
from .spectra import SearchConfig
from .submission import read_submission, write_submission
from .validation import split_dataset


def predict_file(library, test, output, top_k=25, report=None) -> dict:
    require_metric_rdkit()
    output = Path(output)
    report = Path(report) if report else output.with_suffix(".report.json")
    if output.resolve() == report.resolve():
        raise ValueError("Submission and report must use different paths")
    for path in [output, report]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    started = time.perf_counter()
    grouped = defaultdict(list)
    seen_spectra = set()
    for row in rows(test, SPECTRUM_COLUMNS):
        identifier = row["spectrum_id"]
        if identifier in seen_spectra:
            raise ValueError(f"Repeated spectrum_id: {identifier}")
        seen_spectra.add(identifier)
        grouped[row["molecule_id"]].append(row)
    if not grouped:
        raise ValueError("Test file contains no spectra")
    predictions, diagnostics = {}, {}
    reference = SpectralLibrary(library)
    try:
        for identifier, spectra in grouped.items():
            predictions[identifier], diagnostics[identifier] = reference.predict(spectra, top_k)
        run = {
            "molecules": len(grouped), "spectra": len(seen_spectra),
            "molecules_without_spectral_match": sum(
                item["spectral_candidates"] == 0 for item in diagnostics.values()
            ),
            "seconds": time.perf_counter() - started,
            "library_metadata": reference.metadata,
            "per_molecule": diagnostics,
        }
    finally:
        reference.close()
    write_submission(predictions, output)
    read_submission(output, grouped.keys())
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(run, handle, indent=2)
        handle.write("\n")
    return {key: value for key, value in run.items() if key not in {"per_molecule", "library_metadata"}}


def parser():
    root = argparse.ArgumentParser(description="CASMI 2026 offline spectral-search baseline")
    commands = root.add_subparsers(dest="command", required=True)
    inspect_cmd = commands.add_parser("inspect", help="Stream dataset metadata and category counts")
    inspect_cmd.add_argument("parquet")

    split = commands.add_parser("split", help="Make structure-disjoint or cross-library validation files")
    split.add_argument("--train", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--query-library", help="Hold out this library's spectra instead of structures")

    build = commands.add_parser("build-library", help="Build an indexed SQLite reference library")
    build.add_argument("--train", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--precursor-ppm", type=float, default=10.0)
    build.add_argument("--precursor-da", type=float, default=0.005)
    build.add_argument("--fragment-da", type=float, default=0.02)
    build.add_argument("--intensity-floor", type=float, default=0.001)
    build.add_argument("--max-peaks", type=int, default=128)
    build.add_argument("--min-matched-peaks", type=int, default=3)

    predict = commands.add_parser("predict", help="Combine all spectra per molecule and write a submission")
    predict.add_argument("--library", required=True)
    predict.add_argument("--test", required=True)
    predict.add_argument("--output", default="outputs/submission.csv")
    predict.add_argument("--top-k", type=int, choices=range(1, 26), default=25)
    predict.add_argument("--report")

    evaluate = commands.add_parser("evaluate", help="Compute local MRR@25 and top-k accuracy")
    evaluate.add_argument("--submission", required=True)
    evaluate.add_argument("--labels", required=True, help="Labeled validation Parquet")

    check = commands.add_parser("check-submission", help="Check CSV, valid SMILES, and test ID coverage")
    check.add_argument("--submission", required=True)
    check.add_argument("--test", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect(args.parquet)
        elif args.command == "split":
            result = split_dataset(args.train, args.output_dir, args.fraction, args.seed, args.query_library)
        elif args.command == "build-library":
            config = SearchConfig(**{
                field: getattr(args, field) for field in SearchConfig.__dataclass_fields__
            })
            result = build_library(args.train, args.output, config)
        elif args.command == "predict":
            result = predict_file(args.library, args.test, args.output, args.top_k, args.report)
        elif args.command == "evaluate":
            result = evaluate_files(args.submission, args.labels)
        else:
            require_metric_rdkit()
            expected = {row["molecule_id"] for row in rows(args.test, ["molecule_id"])}
            checked = read_submission(args.submission, expected)
            result = {"valid": True, "molecules": len(checked)}
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))

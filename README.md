# Enveda CASMI 2026: molecule identification from MS/MS

Develop and test the Python code in this repository; run the final, offline
inference notebook on Kaggle. The task is to return up to 25 ranked SMILES per
`molecule_id`, using every available spectrum of that molecule.

This is a **spectral-library retrieval baseline**, not a complete competition
solution. It does not yet include formula prediction, PubChem/COCONUT candidate
retrieval, learned ranking, or de novo structure generation.

## Install

Python 3.11 or newer is required. RDKit is pinned to `2026.3.3` (reported by
RDKit as `2026.03.3`) to match the supplied competition evaluation description.

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pytest -q
uv run --locked casmi --help
```

Alternatively, create a virtual environment and use `pip install -e '.[dev]'`.
`uv.lock` locks development dependencies; `requirements.txt` is a hash-locked
runtime export for offline packaging. After intentional dependency changes:

```bash
uv lock
uv export --frozen --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt
```

The repository contains no competition data. Download it through your own
authorized Kaggle access after accepting the rules, then place the files under
`data/raw/`. Data, indexes, wheels, virtual environments, and outputs are ignored
by Git. Do not commit credentials or licensed datasets.

## Inspect the data

```bash
uv run --locked casmi inspect data/raw/train.parquet
uv run --locked casmi inspect data/raw/test.parquet
```

This streams category counts and prints the schema column names and row count.
The actual training file has 18 columns and no `molecule_id` or `spectrum_id`.
Reference indexing reads only peak arrays, adduct, mode, precursor m/z, and
`normalized_smiles`. Use `inchikey14` to group structures during initial exploration.
Test inference uses the anonymous `molecule_id` and `spectrum_id` supplied with
the test data. Extra columns are allowed.

## Validate before using the full training set

### Quick start: avoid copying all references

For initial development, extract only the 1,184 held-out natural-product spectra,
then build a small reference index directly from the original training file.
PyArrow filters by source library before rows are converted to Python objects;
only selected reference structures are canonicalized and indexed.

```bash
uv run --locked casmi extract-validation --train data/raw/train.parquet \
  --output data/splits/natural-product-queries.parquet
uv run --locked casmi build-library --train data/raw/train.parquet \
  --include-library gnps --max-spectra 5000 \
  --output artifacts/gnps-smoke.sqlite
uv run --locked casmi predict --library artifacts/gnps-smoke.sqlite \
  --test data/splits/natural-product-queries.parquet --output outputs/gnps-smoke.csv
uv run --locked casmi evaluate --submission outputs/gnps-smoke.csv \
  --labels data/splits/natural-product-queries.parquet
```

The cap selects the first usable spectra in source order, **not a representative
random sample**. This is a functional smoke test; reference coverage will be lower
and scores must not be compared directly with full-library validation. Remove
`--max-spectra` and use a new index/output path to evaluate all GNPS references.
Other libraries can be added with repeated `--include-library` options. Candidate
availability among all libraries does not imply availability in a small subset.

To build the full reference library without a second reference Parquet copy:

```bash
uv run --locked casmi build-library --train data/raw/train.parquet \
  --exclude-library enveda-np-examples --output artifacts/full-validation.sqlite
```

Do not include the held-out library in its own reference index. Include/exclude
filters combine by intersection (exclusions take precedence). Build metadata records
library selections and any cap, and the CLI reports indexing progress. Full indexing
still requires chemical processing of the included structures; the fast path avoids
unnecessary work rather than promising a fixed runtime on Kaggle.

In notebooks, the equivalent functions are:

```python
from casmi.validation import extract_validation
from casmi.library import build_library
from casmi.spectra import SearchConfig

extract_validation(train_path, "/kaggle/working/queries.parquet")
build_library(
    train_path, "/kaggle/working/gnps-smoke.sqlite", SearchConfig(),
    include_libraries=["gnps"], max_spectra=5000, progress=print,
)
```

The older split command below remains useful when both standalone Parquet
partitions are desired. It is not required for cross-library retrieval validation.

### Cross-library spectral retrieval

Hold out `enveda-np-examples` as queries and use other libraries as references.
The same structure may intentionally occur in both partitions: this evaluates
matching independently acquired spectra, relevant to novelty class 1. It does
not measure generalization to unseen structures. Check actual reference coverage;
not every held-out molecule is guaranteed to remain in the reference library.

```bash
uv run --locked casmi split --train data/raw/train.parquet \
  --query-library enveda-np-examples --output-dir data/splits/cross-library
uv run --locked casmi build-library \
  --train data/splits/cross-library/reference.parquet \
  --output artifacts/cross-library.sqlite
uv run --locked casmi predict --library artifacts/cross-library.sqlite \
  --test data/splits/cross-library/validation.parquet \
  --output outputs/cross-library.csv
uv run --locked casmi evaluate --submission outputs/cross-library.csv \
  --labels data/splits/cross-library/validation.parquet
```

### Structure-disjoint stress test

```bash
uv run --locked casmi split --train data/raw/train.parquet \
  --fraction 0.1 --seed 42 --output-dir data/splits/structure-disjoint
```

Use the same build/predict/evaluate commands with this split's paths. Assignment
is deterministic by canonical-tautomer InChIKey14; equivalent tautomers and
stereoisomers remain together. This is deliberately difficult for the current
retrieval-only baseline: it cannot output identities excluded from its structure
inventory. Low identification scores here are expected, not an installation bug.
Future class-2 evaluation must retain held-out structures in an external candidate
database while withholding their reference spectra. Class-3 evaluation must
exclude the correct structure from that candidate database too.

Split files retain labels for local scoring. If the source has no IDs, the split
command adds a local `molecule_id` from the canonical-tautomer InChIKey14 and a
unique `spectrum_id` from each source row's position to the validation partition.
All spectra of an equivalent structure share the same local molecule ID. Existing
IDs are preserved, and the original training file is unchanged. These IDs are only
for local validation; competition inference retains the supplied test IDs.

Cross-library splitting leaves reference rows in their original schema and only
canonicalizes held-out structures when validation IDs are needed. Structure-disjoint
splitting must canonicalize all structures to assign their partition and generates
IDs in both outputs. The CLI reports progress to stderr. In a notebook, pass
`progress=print` to `split_dataset` for progress every approximately 100,000 spectra.
RDKit can emit diagnostics about enumeration limits or rejected intermediate
tautomers; these messages alone do not establish a failed run. A Python traceback
or an incomplete output is a separate issue to investigate.

Prediction reads only the query input columns, so labels are never consumed by
inference. Use the competition scorer
as the authority once its metric notebook is available; local evaluation currently
implements the pasted description, not verified parity with that notebook.

## Build the full reference library and predict

```bash
uv run --locked casmi build-library --train data/raw/train.parquet \
  --output artifacts/library.sqlite
uv run --locked casmi predict --library artifacts/library.sqlite \
  --test data/raw/test.parquet --output outputs/submission.csv
uv run --locked casmi check-submission --submission outputs/submission.csv \
  --test data/raw/test.parquet
```

Use new output paths for new experiments: commands refuse to overwrite existing
indexes, split directories, submissions, or reports. A reference build streams
Parquet and writes a temporary database, then creates lookup indexes and checks
database integrity before publishing the file. Build metadata records source
path, byte count, preprocessing settings, dependency version, and skipped-row
counts. Inspect those counts: invalid labels/rows and empty spectra are skipped.

The full dataset and index require multiple GB of disk. Building the complete
index and canonicalizing hundreds of thousands of structures may take significant
time. Start with validation, and prepare the final index ahead of the Kaggle run.
No full-dataset timing or accuracy has been established yet.

### What the baseline actually does

1. Drop nonfinite/nonpositive peaks and peaks above precursor + 2 Da.
2. Apply a relative-intensity floor (default 0.1%); keep at most 128 strongest peaks.
3. Square-root intensities and normalize to unit L2 norm.
4. Retrieve references with the **same adduct and ionization mode**, within a
   precursor window of `max(0.005 Da, 10 ppm)`.
5. Compute greedy, one-to-one peak cosine with a 0.02 Da fragment tolerance,
   requiring at least three matched peaks.
6. Take the best reference match per candidate structure per query spectrum,
   then average scores over all nonempty query spectra. Missing matches contribute
   zero. This combines molecule evidence without summing duplicate references.
7. Rank spectral candidates first. Fill remaining slots with distinct training
   structures closest to the molecule's median adduct-corrected neutral mass.

**Mass-only filling is a weak fallback, not evidence of identification.** It is
not constrained to a mass window and may return chemically implausible candidates
when the library lacks a good match. The adjacent `*.report.json` records, per
molecule, spectral versus fallback counts, fallback mass errors, mass disagreement
across adducts, comparisons performed, and the best spectral score. Review it along
with validation MRR@25. The baseline cannot generate novel structures.

Collision energies and instrument metadata are retained in source data but not
used by this first model. It does not compare different adducts directly, perform
deisotoping, optimize mass tolerances, or correct mislabeled training spectra.
Those are measurable future improvements.

### Matching and scoring

Candidates are tautomer-canonicalized with the pinned RDKit, stereochemistry is
removed, and InChIKey14 is compared. No extra salt stripping or neutralization is
applied. Equivalent candidates are deduplicated in library construction.

Evaluation reports MRR@25 and top-1/5/25 identification accuracy. Only the first
correct rank contributes; invalid SMILES score as wrong guesses without shifting
later ranks. The separate submission checker rejects invalid SMILES as an extra
local quality check, as well as checking IDs, candidate counts, and CSV format.

The visible competition test file contains training examples; scores obtained
by looking them up are not an estimate of hidden-test performance.

## Kaggle: thin offline notebook

See [notebooks/kaggle_inference.ipynb](notebooks/kaggle_inference.ipynb).
It runs this package's commands using the same prediction/checking functions.
Upload the source snapshot (`src/` and `requirements.txt`) and the prepared index
as permitted Kaggle notebook inputs. Configure their mount paths in the first
code cell. The index belongs in a data artifact, not the Git repository.

The notebook's default mode requires the exact runtime packages to be installed.
If Kaggle's environment differs, attach a wheel directory as another notebook
input and set `WHEELHOUSE` in the notebook. Prepare wheels on an Internet-enabled
machine compatible with **the Kaggle Python version, OS, and architecture**:

```bash
python -m pip download --only-binary=:all: --require-hashes \
  -r requirements.txt --dest artifacts/wheels
```

The notebook can install those dependencies with `--no-index --require-hashes`;
no downloads or source builds happen during submission. Do not assume wheels from
a different Python version will work. Verify this on Kaggle before finalizing.

Final requirements from the supplied overview: Internet disabled, runtime at most
nine hours, and output `/kaggle/working/submission.csv`. The notebook reads the
provided test file dynamically, with no hardcoded IDs or training-label lookup.
An offline run in the actual Kaggle environment is still required to establish
compatibility and full-data runtime.

## Next development milestones

- Inspect the actual Parquet files and reproduce the official metric notebook.
- Benchmark reference search, report candidate coverage and validate tolerances.
- Predict formulas and add offline PubChem/COCONUT candidate retrieval.
- Train or adapt a spectrum-to-structure ranker and incorporate collision energy.
- Integrate a permitted de novo generator and calibrate combined ranking for MRR.

Public external data/models are allowed according to the supplied overview;
check the full rules and each resource's eligibility/license before including it.

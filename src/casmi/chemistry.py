"""Competition-style structure identity and supported precursor adducts."""

from functools import lru_cache
import math

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

EXPECTED_RDKIT = "2026.03.3"
_TAUTOMERS = rdMolStandardize.TautomerEnumerator()

# All ten listed test adducts are singly charged. Values are mass additions
# to the neutral molecule, including the charge carrier's electron mass.
PROTON = 1.007276466621
WATER = 18.01056468403
ADDUCT_DELTAS = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 18.033825553,
    "[M-H2O+H]+": PROTON - WATER,
    "[M-2H2O+H]+": PROTON - 2 * WATER,
    "[M+Na]+": 22.989220702,
    "[M+K]+": 38.963157906,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -WATER - PROTON,
    "[M+CH2O2-H]-": 46.005479304 - PROTON,
    "[M+Cl]-": 34.969401262,
}


def require_metric_rdkit() -> None:
    if rdBase.rdkitVersion != EXPECTED_RDKIT:
        raise RuntimeError(
            f"Matching requires RDKit {EXPECTED_RDKIT}; found {rdBase.rdkitVersion}. "
            "Install the project-pinned version before evaluation or indexing."
        )


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    if adduct not in ADDUCT_DELTAS:
        raise ValueError(f"Unsupported test adduct: {adduct!r}")
    value = float(precursor_mz) - ADDUCT_DELTAS[adduct]
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Precursor/adduct does not imply a positive finite neutral mass")
    return value


@lru_cache(maxsize=350_000)
def structure_info(smiles: str) -> tuple[str, str, float]:
    """Return canonical-tautomer InChIKey14, nonstereo SMILES, exact mass.

    Do not add salt stripping, neutralization, or other standardization: those
    operations were not specified in the supplied competition description.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = _TAUTOMERS.Canonicalize(mol)
    Chem.RemoveStereochemistry(canonical)
    key = Chem.MolToInchiKey(canonical)
    if not key:
        raise ValueError(f"Cannot compute InChIKey for {smiles!r}")
    return (
        key.split("-")[0],
        Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=False),
        float(Descriptors.ExactMolWt(canonical)),
    )


def structure_key(smiles: str) -> str:
    return structure_info(smiles)[0]

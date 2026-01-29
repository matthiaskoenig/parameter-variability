"""General settings for projects."""

from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR = Path(__file__).parent / "models" / "sbml"

# simple chain model
RESULTS_SIMPLE_CHAIN = RESULTS_DIR / "simple_chain"
RESULTS_SIMPLE_CHAIN.mkdir(exist_ok=True, parents=True)
MODEL_SIMPLE_CHAIN = MODELS_DIR / "simple_chain.xml"

# simple pk model
RESULTS_SIMPLE_PK = RESULTS_DIR / "simple_pk"
RESULTS_SIMPLE_PK.mkdir(exist_ok=True, parents=True)
MODEL_SIMPLE_PK = MODELS_DIR / "simple_pk.xml"

# icg model
RESULTS_ICG = RESULTS_DIR / "icg_body_flat"
RESULTS_ICG.mkdir(exist_ok=True, parents=True)
MODEL_ICG = MODELS_DIR / "icg_body_flat.xml"

MODELS: dict[str, Path] = {
    "simple_chain": MODEL_SIMPLE_CHAIN,
    "simple_pk": MODEL_SIMPLE_PK,
    "icg_body_flat": MODEL_ICG,
}

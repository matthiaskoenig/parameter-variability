"""General settings for projects."""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = Path(__file__).parent / "models" / "sbml"

MODEL_SIMPLE_CHAIN = str(MODELS_DIR / "simple_chain.xml")
MODEL_SIMPLE_PK = str(MODELS_DIR / "simple_pk.xml")

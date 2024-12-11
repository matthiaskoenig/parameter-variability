"""General settings for projects."""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = Path(__file__).parent / "models" / "sbml"

# simple chain model
RESULTS_SIMPLE_CHAIN = RESULTS_DIR / "simple_chain"
RESULTS_SIMPLE_CHAIN.mkdir(exist_ok=True, parents=True)
MODEL_SIMPLE_CHAIN = str(MODELS_DIR / "simple_chain.xml")

# simple pk model
RESULTS_SIMPLE_PK = RESULTS_DIR / "simple_pk"
RESULTS_SIMPLE_PK.mkdir(exist_ok=True, parents=True)
MODEL_SIMPLE_PK = str(MODELS_DIR / "simple_pk.xml")

BASE_DIR: Path = Path(__file__).parent.parent.parent
BAYES_DIR: Path = BASE_DIR / "src" / "parameter_variability" / "bayes"

MEASUREMENT_UNIT_COLUMN = "measurementUnit"
MEASUREMENT_TIME_UNIT_COLUMN = "timeUnit"

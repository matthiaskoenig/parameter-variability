# simple pk
from parvar import RESULTS_SIMPLE_PK
from parvar.optimization.run_optimization import run_optimizations

optimizations = {
    "all": {
        "prior_type": [
            # "prior_biased",
            "exact_prior"
        ],
        # "n_t": [11, 21, 41, 81],
        "noise_cv": [
            # 0.0,
            # 0.001,
            0.001
        ],
    },
    "timepoints": {
        "timepoints": [5, 11, 81],
    },
}
run_optimizations(optimizations=optimizations, results_path=RESULTS_SIMPLE_PK)

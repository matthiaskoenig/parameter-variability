from pathlib import Path

from pymetadata.console import console

from parvar import RESULTS_ICG #, RESULTS_SIMPLE_CHAIN
from parvar.analysis.run_optimization import run_optimizations


if __name__ == "__main__":
    # example for single optimization
    # yaml_path = Path(
    #     # "/home/mkoenig/git/parameter-variability/results/icg_body_flat/xps/all/1dBZb5eYAp3xwgqjGXyb/petab.yaml"
    #     "/home/mkoenig/git/parameter-variability/results/simple_chain/xps/all/1bdPtwOJcMVwsx0KbV6V/petab.yaml"
    # )
    # results = optimize_petab_xp(yaml_file=yaml_path)
    # console.print(results)

    xps_selection = {
        'all': {
            "prior_type": [
                # "prior_biased",
                "exact_prior"
            ],
            # "n_t": [11, 21, 41, 81],
            "noise_cv": [
                # 0.0,
                # 0.001,
                0.01
            ],
        }
    }

    run_optimizations(results_path=RESULTS_ICG, xps_selection=xps_selection)

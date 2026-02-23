from pathlib import Path

from pymetadata.console import console

from parvar import RESULTS_ICG, RESULTS_SIMPLE_CHAIN
from parvar.analysis.run_optimization import (
    xps_selector,
    optimize_petab_xp,
    optimize_petab_xps,
)


if __name__ == "__main__":
    # example for single optimization
    # yaml_path = Path(
    #     # "/home/mkoenig/git/parameter-variability/results/icg_body_flat/xps/all/1dBZb5eYAp3xwgqjGXyb/petab.yaml"
    #     "/home/mkoenig/git/parameter-variability/results/simple_chain/xps/all/1bdPtwOJcMVwsx0KbV6V/petab.yaml"
    # )
    # results = optimize_petab_xp(yaml_file=yaml_path)
    # console.print(results)

    # optimize icg
    console.rule("Selection", align="center")
    xp_ids = xps_selector(
        results_dir=RESULTS_ICG,
        xp_type="all",
        conditions={
            "prior_type": [
                #"prior_biased",
                "exact_prior"
            ],
            # "n_t": [11, 21, 41, 81],
            "noise_cv": [
                # 0.0,
                # 0.001,
                0.01
            ],
        },
    )
    console.print(xp_ids)
    optimize_petab_xps(results_dir=RESULTS_ICG, exp_type='all', xp_ids=xp_ids)


    if False:

        # optimize simple chain
        xp_ids = xps_selector(
            results_dir=RESULTS_SIMPLE_CHAIN,
            xp_type="all",
            conditions={
                "prior_type": ["prior_biased", "exact_prior"],
                # "n_t": [11, 21, 41, 81],
                "noise_cv": [0.0, 0.001, 0.01],
            },
        )
        optimize_petab_xps(
            results_dir=RESULTS_SIMPLE_CHAIN, exp_type="all", xp_ids=xp_ids
        )

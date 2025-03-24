from pathlib import Path
from parameter_variability import RESULTS_SIMPLE_CHAIN
from parameter_variability.bayes.pypesto.simple_chain.petab_optimization import (
    PyPestoSampler
)
from parameter_variability.console import console


def optimize_petab_xp(yaml_file: Path) -> None:
    """Optimize single petab problem using PyPesto."""
    pypesto_sampler = PyPestoSampler(
        yaml_file=yaml_file
    )
    pypesto_sampler.load_problem()
    pypesto_sampler.optimizer()
    pypesto_sampler.bayesian_sampler(n_samples=1000)
    pypesto_sampler.results_hdi()

    #

    # pypesto_sampler.results_median()

    # model_id =
    # experiment_collection =
    # experiment_key =
    # => petab_file

    # sampling


    # Save results
    # res = {}
    # res['real_mean'] = prior_real[]

def optimize_petab_xps(exp_type: str):
    yaml_files = sorted(
        [f for f in (RESULTS_SIMPLE_CHAIN / exp_type).glob("**/petab.yaml")])
    console.print(yaml_files)
    for yaml_file in yaml_files:
        console.rule(yaml_file.name, style="white", align="left")
        optimize_petab_xp(yaml_file)

if __name__ == "__main__":

    # optimize_petab_xps(exp_type="prior")
    optimize_petab_xps(exp_type="n")
    # optimize_petab_xps(exp_type="Nt")



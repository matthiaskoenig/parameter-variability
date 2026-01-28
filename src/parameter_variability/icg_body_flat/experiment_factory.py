"""Factory to create the various sampling experiments."""

from typing import Optional
from rich.progress import track
from itertools import product
from parameter_variability.console import console
from parameter_variability import RESULTS_ICG

from parameter_variability.petab_factory import create_petabs
from parameter_variability.run_optimization import (
    xps_selector, optimize_petab_xps
)
from parameter_variability.experiment import *
from parameter_variability.utils import uuid_alphanumeric


# -------------------------------------------------------------------------------------
# General definitions for ICG model
# -------------------------------------------------------------------------------------

# Observables in model
# FIXME: this should be venous plasma and liver
observables_icg: list[Observable] = [
    Observable(
        id="Cre_plasma_icg",
        starting_value=0,
    ),
    Observable(
        id="Cgi_plasma_icg",
        starting_value=0,
    )
]

# True parameters for sampling
pars_true_icg: dict[str, Parameter] = {
    'BW_MALE': Parameter(id="BW", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 75.0, "scale": 10})),  # bodyweight [kg] (loc: mean;
    'LI__ICGIM_Vmax_MALE': Parameter(id="LI__ICGIM_Vmax", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 0.0369598840327503, "scale": 0.01})),
    'BW_FEMALE': Parameter(id="BW", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 65.0, "scale": 10})),  # bodyweight [kg] (loc: mean;
    'LI__ICGIM_Vmax_FEMALE': Parameter(id="LI__ICGIM_Vmax", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 0.02947, "scale": 0.01}))
}

# Biased parameters
pars_biased_icg: dict[str, Parameter] = {
    'BW_MALE': Parameter(id="BW", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 10.0, "scale": 0.2})),
    'LI__ICGIM_Vmax_MALE': Parameter(id="LI__ICGIM_Vmax", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 10.0, "scale": 0.2})),
    'BW_FEMALE': Parameter(id="BW", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 30.0, "scale": 20})),
    'LI__ICGIM_Vmax_FEMALE': Parameter(id="LI__ICGIM_Vmax", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 0.02, "scale": 0.2}))
}

# True sampling
true_sampling: dict[str, Sampling] = {
    'MALE': Sampling(
        n_samples=100,
        steps=20,
        parameters=[pars_true_icg['BW_MALE'],
                    pars_true_icg['LI__ICGIM_Vmax_MALE']],
        noise=Noise(
            add_noise=True,
            cv=0.05
        ),
        observables=observables_icg
    ),
    'FEMALE': Sampling(
        n_samples=100,
        steps=20,
        parameters=[pars_true_icg['BW_FEMALE'],
                    pars_true_icg['LI__ICGIM_Vmax_FEMALE']],
        noise=Noise(
            add_noise=True,
            cv=0.05
        ),
        observables=observables_icg
    )
}

exp_base = PETabExperiment(
    id='empty',
    model='icg_body_flat',
    prior_type='empty',
    dosage={"IVDOSE_icg": 10.0},
    groups=[
        Group(
            id='MALE',
            sampling=true_sampling['MALE'],
            estimation=Estimation(
                parameters=[pars_true_icg['BW_MALE'],
                            pars_true_icg['LI__ICGIM_Vmax_MALE']]
            )
        ),
        Group(
            id='FEMALE',
            sampling=true_sampling['FEMALE'],
            estimation=Estimation(
                parameters=[pars_true_icg['BW_FEMALE'],
                            pars_true_icg['LI__ICGIM_Vmax_FEMALE']]
            )
        )
    ]
)
# -------------------------------------------------------------------------------------


def icg_experiment_factory(
    n_samples: Optional[list[int]] = None,
    n_timepoints: Optional[list[int]] = None,
    noise_cvs: Optional[list[float]] = None,
    prior_types: list[str] = None,
) -> PETabExperimentList:
    """Factory for creating all ICG experiments.

    :param xps_path: Path to xps file
    :param n_samples: Number of samples to generate
    :param prior_types: List of prior types
    :param n_timepoints: Number of timepoints to generate
    :param noise_cvs: List of CV noise values

    :return: PETabExperimentList
    """
    # handle default values
    if n_samples is None:
        n_samples = [20]
        console.print(f"Using default number of samples: {n_samples}", style="warning")
    if n_timepoints is None:
        n_timepoints = [20]
        console.print(f"Using default number of timepoints: {n_timepoints}", style="warning")
    if noise_cvs is None:
        noise_cvs = [0.1]
        console.print(f"Using default CVS: {noise_cvs}", style="warning")
    if prior_types is None:
        prior_types = ["no_prior"]
        console.print(f"Using default priors: {prior_types}", style="warning")

    # check prior types
    supported_prior_types = ['no_prior', 'prior_biased', 'exact_prior']
    for prior_type in prior_types:
        if prior_type not in supported_prior_types:
            raise ValueError(f"Unsupported prior type: {prior_type}")

    # create all experiments
    experiments = []

    console.rule()
    console.print(f"{n_samples=}", style="info")
    console.print(f"{n_timepoints=}", style="info")
    console.print(f"{noise_cvs=}", style="info")
    console.print(f"{prior_types=}", style="info")
    console.rule()

    tuples = list(product(prior_types, n_samples, n_timepoints, noise_cvs))
    for kt in track(range(len(tuples)), description="Creating experiment definitions..."):

        # current settings
        (prior_type, n_sample, n_timepoint, cv) = tuples[kt]

        # copy base experiment
        exp_n = exp_base.model_copy(deep=True)
        exp_n.id = uuid_alphanumeric()
        exp_n.prior_type = prior_type

        for g in exp_n.groups:
            g.sampling.n_samples = n_sample
            g.sampling.steps = n_timepoint - 1
            g.sampling.noise.cv = cv

            if exp_n.prior_type == 'no_prior':
                g.estimation= Estimation(parameters=[])

            elif exp_n.prior_type == 'prior_biased':
                pars_id = [par for par in pars_biased_icg if g.id in par]
                g.estimation = Estimation(parameters=[pars_biased_icg[par] for par in pars_id])

            elif exp_n.prior_type == 'exact_prior':
                pars_id = [par for par in pars_true_icg if g.id in par]
                g.estimation = Estimation(parameters=[pars_true_icg[par] for par in pars_id])

        experiments.append(exp_n)


    exp_list = PETabExperimentList(
        experiments=experiments
    )

    return exp_list


if __name__ == "__main__":
    # Set up experiments
    definitions = {
        "all": {
            #"n_samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
            "prior_types": ['prior_biased', 'exact_prior'],
            "n_timepoints": [11, 21, 41, 81],
            "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
        },
        # "samples": {
        #     "n_samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
        # },
        # "prior_types": {
        #     "prior_types": ['no_prior', 'prior_biased', 'exact_prior'],
        # },
        # "timepoints": {
        #     "n_timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
        # },
        # "cvs": {
        #     "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
        # },
    }
    for key, definition in definitions.items():
        console.rule(f"{key.upper()}", style="bold white", align="center")
        xps = icg_experiment_factory(**definition)
        # xps.to_yaml_file(RESULTS_ICG / f"xps_{key}.yaml")
        create_petabs(xps, directory=RESULTS_ICG / f"xps_{key}", show_plot=False)
        console.print()

    # Optimizer
    console.rule("Optimization", align="center")
    xp_ids = xps_selector(
        results_dir=RESULTS_ICG,
        xp_type='all',
        conditions={
            'prior_type': ['prior_biased', 'exact_prior'],
            'n_t': [11, 21, 41, 81],
            'noise_cv': [0.0, 0.001, 0.01]
        })
    console.print(xp_ids)

    optimize_petab_xps(
        results_dir=RESULTS_ICG,
        exp_type='all',
        xp_ids=xp_ids
    )


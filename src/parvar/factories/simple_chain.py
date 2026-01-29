"""Factory to create the various sampling experiments."""

from itertools import product
from typing import Optional

from pymetadata.console import console
from rich.progress import track

from parvar.analysis.experiment import *
from parvar.analysis.petab_factory import create_petabs_for_definitions
from parvar.analysis.run_optimization import xps_selector, optimize_petab_xps
from parvar.analysis.utils import uuid_alphanumeric


observables_simple_chain: list[Observable] = [
    Observable(id="S1", starting_value=1),
    Observable(id="S2", starting_value=0),
]

# Define the true values of the parameters for distribution sampling
pars_true: dict[str, Parameter] = {
    "MALE": Parameter(
        id="k1",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.0, "scale": 1}
        ),
    ),
    "FEMALE": Parameter(
        id="k1",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 2.0, "scale": 1}
        ),
    ),
}

pars_biased: dict[str, Parameter] = {
    "MALE": Parameter(
        id="k1",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 10.0, "scale": 0.2}
        ),
    ),
    "FEMALE": Parameter(
        id="k1",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 10.0, "scale": 0.2}
        ),
    ),
}

true_sampling: dict[str, Sampling] = {
    "MALE": Sampling(
        n_samples=20,
        steps=20,
        parameters=[pars_true["MALE"]],
        noise=Noise(add_noise=True, cv=0.05),
        observables=observables_simple_chain,
    ),
    "FEMALE": Sampling(
        n_samples=20,
        steps=20,
        parameters=[pars_true["FEMALE"]],
        noise=Noise(add_noise=True, cv=0.05),
        observables=observables_simple_chain,
    ),
}

exp_base = PETabExperiment(
    id="empty",
    model="simple_chain",
    prior_type="empty",
    groups=[
        Group(
            id="MALE",
            sampling=true_sampling["MALE"],
            estimation=Estimation(parameters=[pars_true["MALE"]]),
        ),
        Group(
            id="FEMALE",
            sampling=true_sampling["FEMALE"],
            estimation=Estimation(parameters=[pars_true["FEMALE"]]),
        ),
    ],
)


def factory(
    n_samples: Optional[list[int]] = None,
    n_timepoints: Optional[list[int]] = None,
    noise_cvs: Optional[list[float]] = None,
    prior_types: list[str] = None,
) -> PETabExperimentList:
    """Factory for simple chain experiments."""
    # handle default values
    if n_samples is None:
        n_samples = [20]
        console.print(f"Using default number of samples: {n_samples}", style="warning")
    if n_timepoints is None:
        n_timepoints = [20]
        console.print(
            f"Using default number of timepoints: {n_timepoints}", style="warning"
        )
    if noise_cvs is None:
        noise_cvs = [0.1]
        console.print(f"Using default CVS: {noise_cvs}", style="warning")
    if prior_types is None:
        prior_types = ["no_prior"]
        console.print(f"Using default priors: {prior_types}", style="warning")

    # check prior types
    supported_prior_types = ["no_prior", "prior_biased", "exact_prior"]
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
    for kt in track(
        range(len(tuples)), description="Creating experiment definitions..."
    ):
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

            if exp_n.prior_type == "no_prior":
                g.estimation = Estimation(parameters=[])

            elif exp_n.prior_type == "prior_biased":
                pars_id = [par for par in pars_biased if g.id in par]
                g.estimation = Estimation(
                    parameters=[pars_biased[par] for par in pars_id]
                )

            elif exp_n.prior_type == "exact_prior":
                pars_id = [par for par in pars_true if g.id in par]
                g.estimation = Estimation(
                    parameters=[pars_true[par] for par in pars_id]
                )

        experiments.append(exp_n)

    exp_list = PETabExperimentList(experiments=experiments)

    return exp_list


definitions = {
    "all": {
        # "n_samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
        "prior_types": ["prior_biased", "exact_prior"],
        "n_timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    },
    "samples": {
        "n_samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
    },
    "prior_types": {
        "prior_types": ["no_prior", "prior_biased", "exact_prior"],
    },
    "timepoints": {
        "n_timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
    },
    "cvs": {
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    },
}


if __name__ == "__main__":
    from parvar import RESULTS_SIMPLE_CHAIN

    # select subset
    # definitions = {k:v for k,v in definitions if k=="timepoints"}
    create_petabs_for_definitions(
        definitions, factory, results_path=RESULTS_SIMPLE_CHAIN
    )

    xps_ids = xps_selector(
        results_dir=RESULTS_SIMPLE_CHAIN,
        xp_type='all',
        conditions={
            "prior_type": ["prior_biased", "exact_prior"],
            "n_t": [11, 21, 41, 81],
            "noise_cv": [0.1, 0.2, 0.5],
        }
    )

    console.print(xps_ids)

    optimize_petab_xps(
        results_dir=RESULTS_SIMPLE_CHAIN,
        exp_type='all',
        xp_ids=xps_ids,
    )


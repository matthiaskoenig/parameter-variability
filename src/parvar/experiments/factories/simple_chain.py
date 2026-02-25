"""Factory to create the various sampling experiments."""

from parvar.experiments.experiment import *
from parvar.experiments.petab_factory import create_petabs_for_definitions


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


factory_data = {
    "exp_base": exp_base,
    "pars_true": pars_true,
    "pars_biased": pars_biased,
}

definitions = {
    "all": {
        # "samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
        "prior_types": ["prior_biased", "exact_prior"],
        "timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    },
    "samples": {
        "samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
    },
    "prior_types": {
        "prior_types": ["no_prior", "prior_biased", "exact_prior"],
    },
    "timepoints": {
        "timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
    },
    "cvs": {
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    },
}

# optimizations = {
#     "all": {
#         "prior_type": [
#             # "prior_biased",
#             "exact_prior"
#         ],
#         # "n_t": [11, 21, 41, 81],
#         "noise_cv": [
#             # 0.0,
#             # 0.001,
#             0.1
#         ],
#     },
#     "timepoints": {
#         "timepoints": [5, 11, 81],
#     },
# }


if __name__ == "__main__":
    from parvar import RESULTS_SIMPLE_CHAIN

    # select subset
    # definitions = {k:v for k,v in definitions if k=="timepoints"}
    create_petabs_for_definitions(
        results_path=RESULTS_SIMPLE_CHAIN,
        definitions=definitions,
        factory_data=factory_data,
    )

    # run_optimizations(optimizations=optimizations, results_path=RESULTS_SIMPLE_CHAIN)

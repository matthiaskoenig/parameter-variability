"""Factory to create the various sampling experiments."""

from parvar.experiments.experiment import *


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

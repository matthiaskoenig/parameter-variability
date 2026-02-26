from parvar.experiments.experiment import *


observables_simple_pk: list[Observable] = [
    Observable(
        id="y_cent",
        starting_value=0,
    ),
    Observable(
        id="y_gut",
        starting_value=1,
    ),
    Observable(
        id="y_peri",
        starting_value=0,
    ),
]

pars_true: dict[str, Parameter] = {
    "CL_MALE": Parameter(
        id="CL",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.0, "scale": 1}
        ),
    ),
    "CL_FEMALE": Parameter(
        id="CL",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 0.5, "scale": 1}
        ),
    ),
    "k_abs_MALE": Parameter(
        id="k_abs",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 0.5, "scale": 1}
        ),
    ),
    "k_abs_FEMALE": Parameter(
        id="k_abs",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.0, "scale": 1}
        ),
    ),
}

pars_biased: dict[str, Parameter] = {
    "CL_MALE": Parameter(
        id="CL",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.5, "scale": 1}
        ),
    ),
    "CL_FEMALE": Parameter(
        id="CL",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.5, "scale": 1}
        ),
    ),
    "k_abs_MALE": Parameter(
        id="k_abs",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.5, "scale": 1}
        ),
    ),
    "k_abs_FEMALE": Parameter(
        id="k_abs",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 1.5, "scale": 1}
        ),
    ),
}

true_sampling: dict[str, Sampling] = {
    "MALE": Sampling(
        n_samples=20,
        steps=20,
        parameters=[
            pars_true["CL_MALE"],
            pars_true["k_abs_MALE"],
        ],
        noise=Noise(add_noise=True, cv=0.05),
        observables=observables_simple_pk,
    ),
    "FEMALE": Sampling(
        n_samples=20,
        steps=20,
        parameters=[
            pars_true["CL_FEMALE"],
            pars_true["k_abs_FEMALE"],
        ],
        noise=Noise(add_noise=True, cv=0.05),
        observables=observables_simple_pk,
    ),
}


exp_base = PETabExperiment(
    id="empty",
    model="simple_pk",
    prior_type="empty",
    groups=[
        Group(
            id="MALE",
            sampling=true_sampling["MALE"],
            estimation=Estimation(
                parameters=[
                    pars_true["CL_MALE"],
                    pars_true["k_abs_MALE"],
                ]
            ),
        ),
        Group(
            id="FEMALE",
            sampling=true_sampling["FEMALE"],
            estimation=Estimation(
                parameters=[
                    pars_true["CL_FEMALE"],
                    pars_true["k_abs_FEMALE"],
                ]
            ),
        ),
    ],
)

factory_data = {
    "exp_base": exp_base,
    "pars_true": pars_true,
    "pars_biased": pars_biased,
}

"""Factory to create the various sampling experiments."""

from parvar.experiments.experiment import *
from parvar.experiments.petab_factory import create_petabs_for_definitions

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
    ),
]

# True parameters for sampling
pars_true_icg: dict[str, Parameter] = {
    "BW_MALE": Parameter(
        id="BW",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 75.0, "scale": 10}
        ),
    ),  # bodyweight [kg] (loc: mean;
    "LI__ICGIM_Vmax_MALE": Parameter(
        id="LI__ICGIM_Vmax",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 0.0369598840327503, "scale": 0.01},
        ),
    ),
    "BW_FEMALE": Parameter(
        id="BW",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 65.0, "scale": 10}
        ),
    ),  # bodyweight [kg] (loc: mean;
    "LI__ICGIM_Vmax_FEMALE": Parameter(
        id="LI__ICGIM_Vmax",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 0.02947, "scale": 0.01}
        ),
    ),
}

# Biased parameters
pars_biased_icg: dict[str, Parameter] = {
    "BW_MALE": Parameter(
        id="BW",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 10.0, "scale": 0.2}
        ),
    ),
    "LI__ICGIM_Vmax_MALE": Parameter(
        id="LI__ICGIM_Vmax",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 10.0, "scale": 0.2}
        ),
    ),
    "BW_FEMALE": Parameter(
        id="BW",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 30.0, "scale": 20}
        ),
    ),
    "LI__ICGIM_Vmax_FEMALE": Parameter(
        id="LI__ICGIM_Vmax",
        distribution=Distribution(
            type=DistributionType.LOGNORMAL, parameters={"loc": 0.02, "scale": 0.2}
        ),
    ),
}

# True sampling
true_sampling: dict[str, Sampling] = {
    "MALE": Sampling(
        n_samples=100,
        steps=20,
        parameters=[pars_true_icg["BW_MALE"], pars_true_icg["LI__ICGIM_Vmax_MALE"]],
        noise=Noise(add_noise=True, cv=0.05),
        observables=observables_icg,
    ),
    "FEMALE": Sampling(
        n_samples=100,
        steps=20,
        parameters=[pars_true_icg["BW_FEMALE"], pars_true_icg["LI__ICGIM_Vmax_FEMALE"]],
        noise=Noise(add_noise=True, cv=0.05),
        observables=observables_icg,
    ),
}

exp_base = PETabExperiment(
    id="empty",
    model="icg_body_flat",
    prior_type="empty",
    dosage={"IVDOSE_icg": 10.0},
    groups=[
        Group(
            id="MALE",
            sampling=true_sampling["MALE"],
            estimation=Estimation(
                parameters=[
                    pars_true_icg["BW_MALE"],
                    pars_true_icg["LI__ICGIM_Vmax_MALE"],
                ]
            ),
        ),
        Group(
            id="FEMALE",
            sampling=true_sampling["FEMALE"],
            estimation=Estimation(
                parameters=[
                    pars_true_icg["BW_FEMALE"],
                    pars_true_icg["LI__ICGIM_Vmax_FEMALE"],
                ]
            ),
        ),
    ],
)


definitions = {
    "all": {
        # "samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
        "prior_types": ["prior_biased", "exact_prior"],
        # "timepoints": [11, 21, 41, 81],
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

factory_data = {
    "exp_base": exp_base,
    "pars_true": pars_true_icg,
    "pars_biased": pars_biased_icg,
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
#             0.01
#         ],
#     },
#     "timepoints": {
#         "timepoints": [5, 11, 81],
#     },
# }


if __name__ == "__main__":
    from parvar import RESULTS_ICG

    # select subset
    # definitions = {k:v for k,v in definitions if k=="timepoints"}
    create_petabs_for_definitions(
        results_path=RESULTS_ICG, definitions=definitions, factory_data=factory_data
    )

    # run_optimizations(
    #     {k: v for (k, v) in optimizations.items() if k == "all"},
    #     results_path=RESULTS_ICG,
    # )

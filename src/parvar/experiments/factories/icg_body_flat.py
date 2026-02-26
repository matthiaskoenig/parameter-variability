"""Factory to create the various sampling experiments."""

from parvar.experiments.experiment import *

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


factory_data = {
    "exp_base": exp_base,
    "pars_true": pars_true_icg,
    "pars_biased": pars_biased_icg,
}

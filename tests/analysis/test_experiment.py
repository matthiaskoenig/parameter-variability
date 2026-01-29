from parvar.analysis.experiment import *


def example_experiment() -> PETabExperiment:
    """Example of a PETabExperiment."""

    # Define the true values of the parameters for distribution sampling
    true_par: dict[str, Parameter] = {
        "BW_MALE": Parameter(
            id="BW",
            distribution=Distribution(
                type=DistributionType.LOGNORMAL, parameters={"loc": 75.0, "scale": 10}
            ),
        ),
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
        ),
        "LI__ICGIM_Vmax_FEMALE": Parameter(
            id="LI__ICGIM_Vmax",
            distribution=Distribution(
                type=DistributionType.LOGNORMAL,
                parameters={"loc": 0.02947, "scale": 0.01},
            ),
        ),
    }

    observables: list[Observable] = [
        Observable(
            id="Cve_plasma_icg",
            starting_value=0,
        ),
    ]

    # example experiment
    petab_experiment = PETabExperiment(
        id="noise",
        model="icg_body_flat",
        prior_type="exact",
        dosage={"IVDOSE_icg": 10.0},
        groups=[
            Group(
                id="MALE",
                sampling=Sampling(
                    n_samples=100,
                    steps=20,
                    parameters=[true_par["BW_MALE"], true_par["LI__ICGIM_Vmax_MALE"]],
                    noise=Noise(add_noise=True, cv=0.05),
                    observables=observables,
                ),
                estimation=Estimation(
                    parameters=[true_par["BW_MALE"], true_par["LI__ICGIM_Vmax_MALE"]]
                ),
            ),
            Group(
                id="FEMALE",
                sampling=Sampling(
                    n_samples=100,
                    steps=20,
                    parameters=[
                        true_par["BW_FEMALE"],
                        true_par["LI__ICGIM_Vmax_FEMALE"],
                    ],
                    noise=Noise(add_noise=True, cv=0.05),
                    observables=observables,
                ),
                estimation=Estimation(
                    parameters=[
                        true_par["BW_FEMALE"],
                        true_par["LI__ICGIM_Vmax_FEMALE"],
                    ]
                ),
            ),
        ],
    )
    return petab_experiment


def test_experiment():
    pass


def test_experiment_list() -> None:
    """Test creation of experiment list."""
    exp_list = PETabExperimentList(
        experiments=[
            example_experiment(),
            example_experiment(),
        ]
    )
    assert exp_list

    df = exp_list.to_dataframe()
    assert not df.empty

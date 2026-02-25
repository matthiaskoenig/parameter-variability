from parvar.experiments.experiment import *
from pymetadata.console import console


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


def example_experiment_list() -> PETabExperimentList:
    """Example experiments."""
    return PETabExperimentList(
        experiments=[
            example_experiment(),
            example_experiment(),
        ]
    )


def test_petab_experiment() -> None:
    exp = example_experiment()
    assert exp


def test_petab_experiment_list() -> None:
    """Test creation of experiment list."""
    exp_list = example_experiment_list()
    assert exp_list

    df = exp_list.to_dataframe()
    assert not df.empty


def test_experiment_list_to_json():
    experiment_list = example_experiment_list()
    json = experiment_list.to_json()
    assert json


def test_experiment():
    """Test the complete workflow."""
    experiment = example_experiment()
    experiment.print_schema()
    experiment.print_json()
    experiment.print_yaml()

    console.rule("Reading data", style="white", align="left")
    yaml = experiment.to_yaml()
    experiment_new = PETabExperiment.from_yaml(yaml)
    assert experiment_new


def test_print_schema():
    experiment = example_experiment()
    experiment.print_schema()


def test_to_json():
    experiment = example_experiment()
    json = experiment.to_json()
    assert json


def test_print_json():
    experiment = example_experiment()
    experiment.print_json()


def test_to_yaml():
    experiment = example_experiment()
    yaml = experiment.to_yaml()
    assert yaml


def test_print_yaml():
    experiment = example_experiment()
    experiment.print_yaml()


def test_write_read_yaml():
    experiment = example_experiment()
    yaml = experiment.to_yaml()
    experiment_new = PETabExperiment.from_yaml(yaml)
    assert experiment_new

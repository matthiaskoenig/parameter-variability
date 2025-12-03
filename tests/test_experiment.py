from pymetadata.console import console

from parameter_variability.bayes.pypesto.experiment import (
    example_experiment,
    example_experiment_list,
    PETabExperiment,
)

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

from pathlib import Path
from typing import List, Union
import numpy as np
import xarray as xr
import pandas as pd
import yaml
from pydantic_yaml import parse_yaml_raw_as

import parameter_variability.bayes.pypesto.simple_chain.petab_factory as pf
from parameter_variability.bayes.pypesto.simple_chain.petab_optimization import (
    PyPestoSampler
)

from parameter_variability.bayes.pypesto.simple_chain.experiment_factory import (
    PETabExperimentList, PETabExperiment, Group)
from parameter_variability.console import console
from parameter_variability import MODEL_SIMPLE_CHAIN
from parameter_variability import RESULTS_DIR, MODELS

def create_petab_for_experiment(model_id: str,
                                experiment: PETabExperiment,
                                xp_settings: dict[str, dict] = None,
                                n_samples: dict[str, int] = None):
    """Create all the petab problems for the given model and experiment."""

    # create results directory
    xp_path: Path = RESULTS_DIR / experiment.model / f"xp_{experiment.id}"
    xp_path.mkdir(parents=True, exist_ok=True)

    # get absolute model path
    sbml_path: Path = MODELS[experiment.model]

    # TODO: save the settings as JSON
    console.print(experiment.groups)

    # if xp_key == 'exact':
    #     prior_real = prior_estim = xp_settings
    # else:
    # prior_real = xp_settings['real']
    # prior_estim = xp_settings['estim']

    # if n_samples is None:
    #     n_samples = {"k1_MALE": 100, "k1_FEMALE": 100}

    # create samples
    groups: List[Group] = experiment.groups
    samples_dsn: dict[pf.Category, pf.LognormParameters] = {}
    for group in groups:
        samples = pf.LognormParameters(
            mu=group.get_parameter('sampling', 'k1', 'loc'),
            sigma=group.get_parameter('sampling', 'k1', 'scale'),
            n=group.sampling.n_samples
        )
        samples_dsn[pf.Category[group.id]] = samples

    samples_k1 = pf.create_male_female_samples(samples_dsn)

    console.print(samples_k1)
    # TODO: plot the samples
    # simulate samples to get data for measurement table
    simulator = pf.ODESampleSimulator(model_path=MODEL_SIMPLE_CHAIN)
    dsets: dict[pf.Category, xr.Dataset] = {}
    for (category, data), group in zip(samples_k1.items(), groups):
        # simulate samples for category

        sim_settings = pf.SimulationSettings(start=0.0,
                                             end=group.sampling.tend,
                                             steps=group.sampling.steps)
        parameters = pd.DataFrame({"k1": data})
        dset = simulator.simulate_samples(parameters,
                                          simulation_settings=sim_settings)
        dsets[category] = dset

        # serialize to netCDF
        dset.to_netcdf(xp_path / f"{category}.nc")


    # save the plot
    pf.plot_simulations(dsets, fig_path=xp_path / "simulations.png")

    # create petab path
    # TODO: feed prior_estim from the new format
    petab_path = xp_path / "petab"
    yaml_file = pf.create_petab_example(petab_path, dsets, param='k1',
                                        compartment_starting_values={'S1': 1, 'S2': 0},
                                        groups=groups,
                                        sbml_path=sbml_path)

    return yaml_file


if __name__ == '__main__':
    # model
    model_id: str = "simple_chain"

    # 1. large collection of petab problems
    # TODO: vary number of samples (for ode) [1, 5, 10, 20, ]
    # TODO: unbalanced samples (10, 100)
    # TODO: vary number of timepoints (for ode) [3, 20}
    # TODO: vary the noise;
    # distance between the means;
    # biased priors;

    # 2. single function to run single experiment and store resuls

    # definition of the different problems for the model
    experiments = {
        'exact': {'k1_MALE': [1.0, 0.2],
                  'k1_FEMALE': [10.0, 0.2]},
        'uninformative': {'real': {'k1_MALE': [1.0, 0.2],
                                   'k1_FEMALE': [10.0, 0.2]},
                          'estim': {'k1_MALE': [0.0, 1],
                                    'k1_FEMALE': [0.0, 1]}},
        'high_variance': {'real': {'k1_MALE': [1.0, 0.2],
                                   'k1_FEMALE': [10.0, 0.4]},
                          'estim': {'k1_MALE': [0, 100],
                                    'k1_FEMALE': [0, 100]}},
        'biased': {'real': {'k1_MALE': [1.0, 0.2],
                            'k1_FEMALE': [10.0, 0.2]},
                   'estim': {'k1_MALE': [10.0, 0.2],
                             'k1_FEMALE': [1.0, 0.2]}}
    }

    xps_file_path = Path(__file__).parents[0] / 'xps.yaml'
    with open(xps_file_path, "r") as xps_file:
        exps_str: str = xps_file.read()

    exps: PETabExperimentList = parse_yaml_raw_as(
        PETabExperimentList, exps_str
    )

    for xps in exps.experiments:

        yaml_files: dict[str, Path] = {}
        console.print(xps)

        console.rule(title=xps.id, style="bold white")
        yaml_file = create_petab_for_experiment(model_id=xps.model,
                                                experiment=xps)
        yaml_files[xps.id] = yaml_file

        exit()
        pypesto_sampler = PyPestoSampler(
            yaml_file=yaml_file
        )

        pypesto_sampler.load_problem()

        pypesto_sampler.optimizer()

        pypesto_sampler.bayesian_sampler(n_samples=1000)

        pypesto_sampler.results_hdi()

    console.print(yaml_files)

        # model_id =
        # experiment_collection =
        # experiment_key =
        # => petab_file

    # sampling


    # Save results
    # res = {}
    # res['real_mean'] = prior_real[]



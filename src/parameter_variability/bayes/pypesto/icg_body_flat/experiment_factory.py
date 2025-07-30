"""Factory to create the various sampling experiments."""
from parameter_variability.bayes.pypesto.experiment import *
from pathlib import Path
from pydantic_yaml import parse_yaml_file_as, to_yaml_str
import yaml
from pymetadata.console import console


from pathlib import Path
from typing import List, Union
import numpy as np
import xarray as xr
import pandas as pd
import yaml
from pydantic_yaml import parse_yaml_raw_as

import parameter_variability.bayes.pypesto.icg_body_flat.petab_factory as pf

from parameter_variability.console import console
from parameter_variability import MODEL_ICG, RESULTS_ICG
from parameter_variability import RESULTS_DIR, MODELS


def create_petab_for_experiment(experiment: PETabExperiment,
                                directory: Path):
    """Create all the petab problems for the given model and experiment."""

    # create results directory
    xp_path: Path = directory / f"{experiment.id}"
    xp_path.mkdir(parents=True, exist_ok=True)

    # get absolute model path
    sbml_path: Path = MODELS[experiment.model]


    # create samples
    groups: List[Group] = experiment.groups
    samples_dsn: dict[pf.Category, dict[pf.PKPDParameters, pf.LognormParameters]] = {}
    for group in groups:
        parameters = group.get_parameter_list('sampling')
        samples_par: dict[pf.PKPDParameters, pf.LognormParameters] = {}
        for par in parameters:
            samples = pf.LognormParameters(
                mu=group.get_parameter('sampling', par.id, 'loc'),
                sigma=group.get_parameter('sampling', par.id, 'scale'),
                n=group.sampling.n_samples
            )
            samples_par[pf.PKPDParameters[par.id]] = samples

        samples_dsn[pf.Category[group.id]] = samples_par

    samples_pkpd_par = pf.create_samples_parameters(samples_dsn)
    pf.plot_samples(samples_pkpd_par, fig_path=xp_path / 'samples.png')

    console.print(samples_pkpd_par)
    # simulate samples to get data for measurement table
    simulator = pf.ODESampleSimulator(model_path=sbml_path)
    dsets: dict[pf.Category, xr.Dataset] = {}
    for (category, data), group in zip(samples_pkpd_par.items(), groups):
        # simulate samples for category

        sim_settings = pf.SimulationSettings(start=0.0,
                                             end=group.sampling.tend,
                                             steps=group.sampling.steps,
                                             model_changes=experiment.model_changes)
        parameters = pd.DataFrame({par_id: samples for par_id, samples in data.items()})
        dset = simulator.simulate_samples(parameters,
                                          simulation_settings=sim_settings)
        dsets[category] = dset

        # serialize to netCDF
        dset.to_netcdf(xp_path / f"{category}.nc")

    # save the plot
    pf.plot_simulations(dsets, fig_path=xp_path / "simulations.png")
    # create petab path
    # TODO: Feed the param and the sbml_path inputs accordingly.
    #   feed the model_icg inside to get all the model parameters r.getIds
    #   https://libroadrunner.readthedocs.io/en/latest/PythonAPIReference/cls_RoadRunner.html#RoadRunner.getIds
    petab_path = xp_path / "petab"
    params = [par.id for par in experiment.groups[0].get_parameter_list('sampling')]
    yaml_file = pf.create_petab_example(
        dfs=dsets,
        groups=groups,
        petab_path=petab_path,
        param=params,
        sbml_path=sbml_path,
        initial_values=None
    )

    return yaml_file


# Define the true values of the parameters for distribution sampling
true_par: dict[str, Parameter] = {
    'MALE_BW': Parameter(id="BW", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 75.0, "scale": 10})),  # bodyweight [kg] (loc: mean;
    'MALE_LI__ICGIM_Vmax': Parameter(id="LI__ICGIM_Vmax", distribution=Distribution(
        type=DistributionType.LOGNORMAL,
        parameters={"loc": 0.0369598840327503, "scale": 0.01}))
}

true_sampling: dict[str, Sampling] = {
    'MALE': Sampling(
        n_samples=100,
        steps=20,
        parameters=[true_par['MALE_BW'],
                    true_par['MALE_LI__ICGIM_Vmax']]
    )
}

def create_prior_experiments(xps_path: Path) -> PETabExperimentList:
    """Create experiments to check for prior effect."""
    # exact prior
    exp_exact = PETabExperiment(
        id='prior_exact',
        model='icg_body_flat',
        model_changes= {"IVDOSE_icg": 10.0},
        groups=[
            Group(
                id='MALE',
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[true_par['MALE_BW'],
                                true_par['MALE_LI__ICGIM_Vmax']]
                )
            )
        ]
    )

    # No prior
    exp_noprior = PETabExperiment(
        id="prior_noprior",
        model='icg_body_flat',
        model_changes= {"IVDOSE_icg": 10.0},
        groups=[
            Group(
                id="MALE",
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[],
                )
            )
        ]
    )

    # biased prior
    pars_biased: dict[str, Parameter] = {
        'MALE_BW': Parameter(id="BW", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 10.0, "scale": 0.2})),
        'MALE_LI__ICGIM_Vmax': Parameter(id="LI__ICGIM_Vmax", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 10.0, "scale": 0.2}))
    }
    exp_biased = PETabExperiment(
        id="prior_biased",
        model="icg_body_flat",
        model_changes= {"IVDOSE_icg": 10.0},
        groups=[
            Group(
                id="MALE",
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[
                        pars_biased['MALE_BW'],
                        pars_biased['MALE_LI__ICGIM_Vmax']
                    ],
                )
            )
        ]
    )

    exp_list = PETabExperimentList(
        experiments=[exp_exact, exp_noprior, exp_biased]
    )
    exp_list.to_yaml(xps_path)
    return exp_list


def create_samples_experiments(xps_path: Path) -> PETabExperimentList:
    """Create experiments to check for number of samples n effect."""
    exp = PETabExperiment(
        id='n',
        model='icg_body_flat',
        model_changes= {"IVDOSE_icg": 10.0},
        groups=[
            Group(
                id='MALE',
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[true_par['MALE_BW'],
                                true_par['MALE_LI__ICGIM_Vmax']]
                )
            )
        ]
    )
    experiments = []
    for n in [1, 2, 3, 4, 5, 10, 20, 40, 80]:
        exp_n = exp.model_copy(deep=True)
        exp_n.id = f"n_{n}"
        for g in exp_n.groups:
            g.sampling.n_samples = n
        experiments.append(exp_n)

    exp_list = PETabExperimentList(
        experiments=experiments
    )
    exp_list.to_yaml(xps_path)
    return exp_list


def create_timepoints_experiments(xps_path: Path) -> PETabExperimentList:
    """Create experiments to check for number of timepoints Nt effect."""
    exp = PETabExperiment(
        id='Nt',
        model='icg_body_flat',
        model_changes= {"IVDOSE_icg": 10.0},
        groups=[
            Group(
                id='MALE',
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[true_par['MALE_BW'],
                                true_par['MALE_LI__ICGIM_Vmax']]
                )
            )
        ]
    )
    experiments = []
    for Nt in [2, 3, 4, 5, 11, 21, 41, 81]:
        exp_n = exp.model_copy(deep=True)
        exp_n.id = f"Nt_{Nt}"
        for g in exp_n.groups:
            g.sampling.steps = Nt - 1
        experiments.append(exp_n)

    exp_list = PETabExperimentList(
        experiments=experiments
    )
    exp_list.to_yaml(xps_path)
    return exp_list


def create_petabs(exps: PETabExperimentList, directory: Path) -> list[Path]:
    """Create Petab files."""
    directory.mkdir(parents=True, exist_ok=True)
    yaml_files: list[Path] = []
    for xp in exps.experiments:
        console.rule(title=xp.id, style="bold white")
        console.print(xp)
        yaml_file = create_petab_for_experiment(experiment=xp, directory=directory)
        yaml_files.append(yaml_file)

        # Dump PETabExperiment into YAML file
        with open(directory / f"{xp.id}" / "xp.yaml", "w") as f:
            ex_m = xp.model_dump(mode='json')
            yaml.dump(ex_m, f, sort_keys=False, indent=2)

    return yaml_files


if __name__ == "__main__":
    # FIXME: what subset should be written in the measurements table [Cve_icg], [LI__icg_bi],

    PARAMETERS = ['LI__ICGIM_Vmax', 'BW']
    # vary priors
    xps_prior = create_prior_experiments(xps_path=RESULTS_ICG / "xps_prior.yaml")
    create_petabs(xps_prior, directory=RESULTS_ICG / "prior")

    # vary samples
    xps_samples = create_samples_experiments(xps_path=RESULTS_ICG / "xps_n.yaml")
    create_petabs(xps_samples, directory=RESULTS_ICG / "n")

    # vary number of timepoints
    xps_timepoints = create_timepoints_experiments(xps_path=RESULTS_ICG / "xps_Nt.yaml")
    create_petabs(xps_timepoints, directory=RESULTS_ICG / "Nt")

    # TODO: vary the noise;

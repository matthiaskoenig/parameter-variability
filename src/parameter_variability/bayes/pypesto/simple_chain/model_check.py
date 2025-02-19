from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd


import parameter_variability.bayes.pypesto.simple_chain.petab_factory as pf
from parameter_variability.bayes.pypesto.simple_chain.petab_optimization import (
    PyPestoSampler
)
from parameter_variability.console import console
from parameter_variability import MODEL_SIMPLE_CHAIN
from parameter_variability import RESULTS_DIR, MODELS

def create_petab_for_experiment(model_id: str, xp_key: str, xp_settings: dict[str, dict]):
    """Create all the petab problems for the given model and experiment."""

    # create results directory
    xp_path: Path = RESULTS_DIR / model_id / f"xp_{xp_key}"
    xp_path.mkdir(parents=True, exist_ok=True)

    # get absolute model path
    sbml_path: Path = MODELS[model_id]

    # TODO: save the settings as JSON
    console.print(f"{xp_settings=}")


    if xp_key == 'exact':
        prior_real = prior_estim = xp_settings
    else:
        prior_real = xp_settings['real']
        prior_estim = xp_settings['estim']

    # create samples
    samples_k1: dict[pf.Category, np.ndarray] = pf.create_male_female_samples(
        {
            # Category.MALE: LognormParameters(mu=1.5, sigma=1.0, n=50),  # mu_ln=0.2216, sigma_ln=0.60640
            # Category.FEMALE: LognormParameters(mu=3.0, sigma=0.5, n=100),  # mu_ln=1.0849, sigma_ln=0.16552
            pf.Category.MALE: pf.LognormParameters(mu=prior_real["k1_MALE"][0],
                                                   sigma=prior_real["k1_MALE"][1],
                                                   n=100),
            pf.Category.FEMALE: pf.LognormParameters(mu=prior_real["k1_FEMALE"][0],
                                                     sigma=prior_real["k1_FEMALE"][1],
                                                     n=100),

            # Category.OLD: LognormParameters(mu=10.0, sigma=3, n=20),
            # Category.YOUNG: LognormParameters(mu=1.5, sigma=1, n=40),
        }
    )
    # TODO: plot the samples

    # simulate samples to get data for measurement table
    simulator = pf.ODESampleSimulator(model_path=MODEL_SIMPLE_CHAIN)
    sim_settings = pf.SimulationSettings(start=0.0, end=20.0, steps=300)
    dsets: dict[pf.Category, xr.Dataset] = {}
    for category, data in samples_k1.items():
        # simulate samples for category
        parameters = pd.DataFrame({"k1": data})
        dset = simulator.simulate_samples(parameters,
                                          simulation_settings=sim_settings)
        dsets[category] = dset

        # serialize to netCDF
        dset.to_netcdf(xp_path / f"{category}.nc")


    # save the plot
    pf.plot_simulations(dsets, fig_path=xp_path / "simulations.png")

    # create petab path
    petab_path = xp_path / "petab"
    yaml_file = pf.create_petab_example(petab_path, dsets, param='k1',
                            compartment_starting_values={'S1': 1, 'S2': 0},
                            prior_par=prior_estim,
                            sbml_path=sbml_path)

    return xp_key, yaml_file


if __name__ == '__main__':
    # model
    model_id: str = "simple_chain"

    # 1. large collection of petab problems
    # TODO: vary number of samples (for ode) [1, 5, 10, 20, }
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

    yaml_files: dict[str, Path] = {}
    for xp_key, xp_settings in experiments.items():
        console.rule(title=xp_key, style="bold white")
        xp_key, yaml_file = create_petab_for_experiment(model_id=model_id, xp_key=xp_key, xp_settings=xp_settings)
        yaml_files[xp_key] = yaml_file

    console.print(yaml_files)

        # model_id =
        # experiment_collection =
        # experiment_key =
        # => petab_file

        # # sampling
        # pypesto_sampler = PyPestoSampler(
        #     yaml_file=xp_path / "petab.yaml",
        #     fig_path=xp_path / "figs"
        # )
        #
        # pypesto_sampler.load_problem()
        #
        # pypesto_sampler.optimizer()
        #
        # pypesto_sampler.bayesian_sampler(n_samples=1000)
        #
        # pypesto_sampler.results_hdi()
        #
        # # Save results
        # # res = {}
        # # res['real_mean'] = prior_real[]



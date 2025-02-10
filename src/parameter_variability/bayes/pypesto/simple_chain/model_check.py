import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import parameter_variability.bayes.pypesto.simple_chain.petab_factory as pf
from parameter_variability.bayes.pypesto.simple_chain.petab_optimization import (
    PyPestoSampler
)
from parameter_variability.console import console
from parameter_variability import MODEL_SIMPLE_CHAIN


if __name__ == '__main__':
    prior_pars = {
        'exact': {'k1_MALE':   [1.0, 0.2],
                  'k1_FEMALE': [10.0, 0.2]},
        'uninformative': {'real': {'k1_MALE':   [1.0, 0.2],
                                   'k1_FEMALE': [10.0, 0.2]},
                          'estim': {'k1_MALE':   [0.0, 1],
                                    'k1_FEMALE': [0.0, 1]}},
        'high_variance': {'real': {'k1_MALE':   [1.0, 0.2],
                                   'k1_FEMALE': [10.0, 0.4]},
                          'estim': {'k1_MALE':   [0, 100],
                                    'k1_FEMALE': [0, 100]}},
        'biased': {'real': {'k1_MALE':   [1.0, 0.2],
                            'k1_FEMALE': [10.0, 0.2]},
                   'estim': {'k1_MALE':   [10.0, 0.2],
                             'k1_FEMALE': [1.0, 0.2]}}
    }

    fig_path: Path = Path(__file__).parent / "results"
    fig_path.mkdir(parents=True, exist_ok=True)

    petab_path = fig_path / "petab"
    petab_path.mkdir(parents=True, exist_ok=True)

    for par in prior_pars:
        prior_p = prior_pars[par]

        if par == 'exact':
            prior_real = prior_estim = prior_p

        else:
            prior_real = prior_p['real']
            prior_estim = prior_p['estim']

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

        simulator = pf.ODESampleSimulator(model_path=MODEL_SIMPLE_CHAIN)
        sim_settings = pf.SimulationSettings(start=0.0, end=20.0, steps=300)
        dsets: dict[pf.Category, xr.Dataset] = {}
        for category, data in samples_k1.items():
            # simulate samples for category
            parameters = pd.DataFrame({"k1": data})
            dset = simulator.simulate_samples(parameters,
                                              simulation_settings=sim_settings)
            dsets[category] = dset

        pf.plot_simulations(dsets)

        pf.create_petab_example(petab_path, dsets, param='k1',
                                compartment_starting_values={'S1': 1, 'S2': 0},
                                prior_par=prior_estim)

        pypesto_sampler = PyPestoSampler(
            yaml_file=Path(__file__).parent / "petab.yaml",
            fig_path=Path(__file__).parents[5] / "results" / "simple_chain"
        )

        pypesto_sampler.load_problem()

        pypesto_sampler.optimizer()

        pypesto_sampler.bayesian_sampler(n_samples=1000)

        # Save results
        # res = {}
        # res['real_mean'] = prior_real[]



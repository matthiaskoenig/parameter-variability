import json
from itertools import cycle
from typing import Union, List, Callable, Dict, Any
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib import pyplot as plt
from pytensor.compile.ops import as_op
from scipy import stats
from dataclasses import dataclass
from pathlib import Path
import roadrunner
from parameter_variability.console import console
from parameter_variability import MODEL_SIMPLE_PK, RESULTS_DIR
from parameter_variability.bayes.sampler import SampleSimulator, Sampler
import arviz as az


@dataclass
class BayesModel:
    """Perform Bayesian Inference on Parameter of ODE model"""
    ode_mod: Union[str, Path]
    parameter: List[str]
    compartment: str
    steps: int
    prior_parameters: Dict[str, float]
    f_prior_dsn: Callable
    tune: int
    draws: int
    chains: int
    init_vals: Dict[str, np.ndarray]

    def ls_soln(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        pass

    def setup(self, data: xr.Dataset, plot_graph: bool) -> pm.Model:
        """Initialization of Priors and Likelihood"""
        ode_model = roadrunner.RoadRunner(self.ode_mod)
        n_sim = data.sizes['sim']

        # TODO: add dimensions for several thetas and sims
        @as_op(itypes=[pt.dmatrix], otypes=[pt.dmatrix])  # otypes=[pt.dmatrix]
        def pytensor_forward_model_matrix(theta):
            dfs = []
            for i in range(n_sim):
                ode_model.resetAll()
                for par, val in zip(self.parameter, theta):  # each prior parameter
                    ode_model.setValue(par, val[i])

                sim = ode_model.simulate(start=0, end=10, steps=self.steps)
                sim_df = pd.DataFrame(sim, columns=sim.colnames)
                dfs.append(sim_df)

            dset = xr.concat([d.to_xarray() for d in dfs], dim="sim")

            return dset[self.compartment].to_numpy()

        with pm.Model() as model:

            k = self.f_prior_dsn(self.parameter[0], mu=self.prior_parameters['loc'],
                                 sigma=self.prior_parameters['s'],
                                 initval=self.init_vals['k'],  # FIXME: Use dist to sample an initval
                                 shape=(n_sim,), dims='sim')

            sigma = pm.HalfNormal("sigma", sigma=1, shape=(n_sim,)

            # ODE solution function
            ode_soln = pytensor_forward_model_matrix(pm.math.stack([k]))

            # likelihood
            pm.LogNormal(
                name=self.compartment,
                mu=ode_soln,  # FIXME: Write the correct MU from the papers
                sigma=sigma,
                observed=data[self.compartment]
            )

        if plot_graph:
            pm.model_to_graphviz(model)  # FIXME: Not plotting locally in SciView

        return model

    def sampler(self, model: pm.Model) -> az.InferenceData:
        """Definition of the Sampling Process """

        vars_list = list(model.values_to_rvs.keys())[:-1]
        print(f'Variables: {vars_list}\n')
        with model:
            for i in range(3):
                try:
                    trace = pm.sample(step=[pm.Slice(vars_list)],
                                      tune=self.tune, draws=self.draws,
                                      chains=self.chains)

                    break

                except pm.exceptions.SamplingError as e:
                    print(f'{e}\nTrying again. Trial {i}')
                    continue

        return trace

    def plot_trace(self, trace: az.InferenceData) -> None:
        """Trace plots of the parameters sampled"""
        console.print(az.summary(trace, stat_focus='median'))

        az.plot_trace(sample, compact=True, kind='trace')
        plt.suptitle('Trace plots')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    console.rule("Sampling")
    sampler = Sampler(
        model=MODEL_SIMPLE_PK,
        parameter="k",
        f_distribution=stats.lognorm,
        distribution_parameters={
            "loc": np.log(2.5),
            "s": 1,
        }
    )
    console.print(sampler)

    true_thetas = sampler.sample(n=2)
    console.print(f"{true_thetas=}")

    console.rule('Thetas PDF')
    sampler.plot(true_thetas)

    console.rule("Simulation", align="left", style="white")
    simulator = SampleSimulator(
        model=MODEL_SIMPLE_PK,
        thetas=true_thetas,
    )
    df = simulator.simulate(start=0, end=10, steps=10)
    console.print(df)

    console.rule('Model Setup')
    bayes_model = BayesModel(ode_mod=MODEL_SIMPLE_PK,
                             parameter=['k'],
                             compartment='[y_gut]',
                             steps=10,
                             prior_parameters={
                                 'loc': np.log(1.0),
                                 's': 0.5
                             },
                             f_prior_dsn=pm.LogNormal,
                             tune=2000, draws=4000, chains=4,
                             init_vals={'k': np.array([2.00, 2.00])})

    mod = bayes_model.setup(df, plot_graph=True)
    console.print(mod)

    console.rule('Sampling starts')
    sample = bayes_model.sampler(mod)

    console.rule('Results')
    bayes_model.plot_trace(sample)


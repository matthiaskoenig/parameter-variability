import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import roadrunner
import xarray as xr
from matplotlib import pyplot as plt
from pytensor.compile.ops import as_op
from scipy import stats

from parameter_variability import MODEL_SIMPLE_PK, RESULTS_DIR
from parameter_variability.bayes.sampler import Sampler, SampleSimulator, DistDefinition
from parameter_variability.console import console


@dataclass
class BayesModel:
    """Perform Bayesian Inference on Parameter of ODE model"""

    ode_mod: Union[str, Path]
    parameter: List[str]
    compartment: str
    prior_parameters: Dict[str, Dict[str, float]]
    f_prior_dsn: Callable

    def ls_soln(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        pass

    def setup(self, data: xr.Dataset, end: int, steps: int,
              init_vals: Dict[str, np.ndarray]) -> pm.Model:
        """Initialization of Priors and Likelihood"""
        n_sim = data.sizes['sim']
        ode_model = roadrunner.RoadRunner(self.ode_mod)

        @as_op(itypes=[pt.dmatrix], otypes=[pt.dmatrix])  # otypes=[pt.dmatrix]
        def pytensor_forward_model_matrix(theta):
            dfs = []

            for i in range(n_sim):
                ode_model.resetAll()
                for par_name, value in zip(self.parameter, theta):
                    ode_model.setValue(par_name, value[i])
                sim = ode_model.simulate(start=0, end=end, steps=steps)
                sim_df = pd.DataFrame(sim, columns=sim.colnames)
                dfs.append(sim_df)

            dset = xr.concat([d.to_xarray() for d in dfs], dim="sim")

            return dset[self.compartment].to_numpy()

        with pm.Model() as model:
            k = self.f_prior_dsn(
                'k',
                mu=self.prior_parameters['k']['loc'],
                sigma=self.prior_parameters['k']['s'],
                initval=init_vals['k'], shape=(n_sim,), dims='sim'
            )

            sigma = pm.HalfNormal("sigma", sigma=1, shape=(n_sim,))

            # ODE solution function
            ode_soln = pytensor_forward_model_matrix(pm.math.stack([k]))

            # likelihood
            pm.LogNormal(
                name=self.compartment,
                mu=ode_soln,
                sigma=sigma,
                observed=data[self.compartment],
            )

        return model

    def sampler(self, model: pm.Model, tune: int, draws: int, chains: int) -> az.InferenceData:
        """Definition of the Sampling Process"""

        vars_list = list(model.values_to_rvs.keys())[:-1]
        print(f"Variables: {vars_list}\n")
        with model:
            trace = pm.sample(
                step=[pm.Slice(vars_list)],
                tune=tune,
                draws=draws,
                chains=chains
            )

        return trace

    def plot_trace(self, trace: az.InferenceData) -> None:
        """Trace plots of the parameters sampled"""
        console.print(az.summary(trace, stat_focus="median"))

        az.plot_trace(trace, compact=True, kind="trace")
        plt.suptitle("Trace plots")
        plt.tight_layout()
        plt.show()


def bayes_analysis(sampler: Sampler, bayes_model: BayesModel,
                   n: int, end: int = 20, steps: int = 100) -> None:

    console.rule("Sampling", align="left", style="white")
    console.print(sampler)
    thetas = sampler.sample(n=n)
    console.print(f"{thetas=}")
    sampler.plot_samples(thetas, distributions=sampler.distributions)

    console.rule("Simulation", align="left", style="white")
    simulator = SampleSimulator(
        model=MODEL_SIMPLE_PK,
        thetas=thetas,
    )
    data = simulator.simulate(start=0, end=end, steps=steps)
    console.print(data)

    data_err = simulator.apply_errors_to_data(data, variables=["[y_gut]", "[y_cent]"])
    simulator.plot_data(
        data=data,
        data_err=data_err,
        variables=["[y_gut]", "[y_cent]", "[y_peri]"]
    )

    mod = bayes_model.setup(data_err, end, steps, init_vals={'k': np.repeat(2.0, n)})
    console.print(mod)

    console.rule(f'Sampling starts for {n=}')
    sample = bayes_model.sampler(mod, tune=2000, draws=4000, chains=4)

    console.rule(f'Results for {n=}')
    bayes_model.plot_trace(sample)


if __name__ == "__main__":

    sampler = Sampler(
        model=MODEL_SIMPLE_PK,
        distributions=[
            DistDefinition(
                parameter="k",
                f_distribution=stats.lognorm,
                distribution_parameters={
                    "loc": np.log(2.5),
                    "s": 1,
                },
            )
        ]
    )

    bayes_model = BayesModel(ode_mod=MODEL_SIMPLE_PK,
                             parameter=['k'],
                             compartment='[y_gut]',
                             prior_parameters={'k': {
                                 'loc': np.log(1.0),
                                 's': 0.5
                             }},
                             f_prior_dsn=pm.LogNormal
                             )

    bayes_analysis(sampler, bayes_model, n=1)
    bayes_analysis(sampler, bayes_model, n=2)

    # console.print(sampler)
    #
    # true_thetas = sampler.sample(n=1)
    # console.print(f"{true_thetas=}")
    #
    # console.rule("Thetas PDF")
    # sampler.plot_samples(true_thetas)
    #
    # console.rule("Simulation", align="left", style="white")
    # simulator = SampleSimulator(
    #     model=MODEL_SIMPLE_PK,
    #     thetas=true_thetas,
    # )
    # df = simulator.simulate(start=0, end=10, steps=10)
    # console.print(df)
    #
    # console.rule("Model Setup")
    # bayes_model = BayesModel(
    #     ode_mod=MODEL_SIMPLE_PK,
    #     parameter="k",
    #     compartment="[y_gut]",
    #     steps=10,
    #     prior_parameters={"loc": np.log(2.0), "s": 0.5},
    #     f_prior_dsn=pm.LogNormal,
    #     tune=2000,
    #     draws=4000,
    #     chains=4,
    #     init_vals={"k": np.array([2.00])},
    # )
    #
    # mod = bayes_model.setup(df)
    # console.print(mod)
    #
    # console.rule("Sampling starts")
    # sample = bayes_model.sampler(mod)
    #
    # console.rule("Results")
    # bayes_model.plot_trace(sample)

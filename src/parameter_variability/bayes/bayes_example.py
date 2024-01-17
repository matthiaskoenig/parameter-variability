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
from parameter_variability.bayes.sampler import Sampler, SampleSimulator
from parameter_variability.console import console


@dataclass
class BayesModel:
    """Perform Bayesian Inference on Parameter of ODE model"""

    ode_mod: Union[str, Path]
    parameter: str
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

    def setup(self, data: xr.Dataset) -> pm.Model:
        """Initialization of Priors and Likelihood"""
        ode_model = roadrunner.RoadRunner(self.ode_mod)

        @as_op(itypes=[pt.dvector], otypes=[pt.dvector])  # otypes=[pt.dmatrix]
        def pytensor_forward_model_matrix(theta):
            ode_model.resetAll()
            for par_name, value in zip(self.parameter, theta):
                ode_model.setValue(par_name, value)
            sim = ode_model.simulate(start=0, end=10, steps=self.steps)
            sim_df = pd.DataFrame(sim, columns=sim.colnames)
            return sim_df[self.compartment].to_numpy()

        with pm.Model() as model:
            theta = self.f_prior_dsn(
                self.parameter,
                mu=self.prior_parameters["loc"],
                sigma=self.prior_parameters["s"],
                initval=self.init_vals[self.parameter][0],
            )

            sigma = pm.HalfNormal("sigma", sigma=1)

            # ODE solution function
            ode_soln = pytensor_forward_model_matrix(pm.math.stack([theta]))

            # likelihood
            pm.LogNormal(
                name=self.compartment,
                mu=ode_soln,
                sigma=sigma,
                observed=data[self.compartment],
            )

        return model

    def sampler(self, model: pm.Model) -> az.InferenceData:
        """Definition of the Sampling Process"""

        vars_list = list(model.values_to_rvs.keys())[:-1]
        print(f"Variables: {vars_list}\n")
        with model:
            trace = pm.sample(
                step=[pm.Slice(vars_list)],
                tune=self.tune,
                draws=self.draws,
                chains=self.chains,
            )

        return trace

    def plot_trace(self, trace: az.InferenceData) -> None:
        """Trace plots of the parameters sampled"""
        console.print(az.summary(trace, stat_focus="median"))

        az.plot_trace(sample, compact=True, kind="trace")
        plt.suptitle("Trace plots")
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
        },
    )
    console.print(sampler)

    true_thetas = sampler.sample(n=1)
    console.print(f"{true_thetas=}")

    console.rule("Thetas PDF")
    sampler.plot_samples(true_thetas)

    console.rule("Simulation", align="left", style="white")
    simulator = SampleSimulator(
        model=MODEL_SIMPLE_PK,
        thetas=true_thetas,
    )
    df = simulator.simulate(start=0, end=10, steps=10)
    console.print(df)

    console.rule("Model Setup")
    bayes_model = BayesModel(
        ode_mod=MODEL_SIMPLE_PK,
        parameter="k",
        compartment="[y_gut]",
        steps=10,
        prior_parameters={"loc": np.log(2.0), "s": 0.5},
        f_prior_dsn=pm.LogNormal,
        tune=2000,
        draws=4000,
        chains=4,
        init_vals={"k": np.array([2.00])},
    )

    mod = bayes_model.setup(df)
    console.print(mod)

    console.rule("Sampling starts")
    sample = bayes_model.sampler(mod)

    console.rule("Results")
    bayes_model.plot_trace(sample)

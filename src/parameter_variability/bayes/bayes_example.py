import json
from typing import Union, List, Callable, Dict, Any
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from matplotlib import pyplot as plt
from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
from scipy.stats import lognorm
from dataclasses import dataclass
from pathlib import Path
import roadrunner
from parameter_variability.console import console
from parameter_variability import MODEL_SIMPLE_PK


@dataclass
class BayesModel:
    ode_mod: Union[str, Path]
    parameter: str
    compartment: str
    prior_parameters: Dict[str, float]
    f_prior_dsn: Callable


    def read_data(self, path: str):
        return xr.open_dataset(path)

    def model_bayes_setup(self, data: xr.Dataset):
        ode_model = roadrunner.RoadRunner(self.ode_mod)

        @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
        def pytensor_forward_model_matrix(theta, compartment, steps=10):
            ode_model.resetAll()
            for par_name, value in zip(self.parameter, theta):
                ode_model.setValue(par_name, value)
            sim = ode_model.simulate(start=0, end=10, steps=steps)
            sim_df = pd.DataFrame(sim, columns=sim.colnames)
            return sim_df[compartment]

        with pm.Model() as model:
            theta = self.f_prior_dsn(self.compartment, mu=self.prior_parameters['loc'],
                                     sigma=self.prior_parameters['s'], initval=1)

            sigma = pm.HalfNormal("sigma", sigma=1)

            # ODE solution function
            ode_soln = pytensor_forward_model_matrix(theta, self.compartment)

            # likelihood
            pm.LogNormal(
                name=self.compartment,
                mu=ode_soln,
                sigma=sigma,
                observed=data[self.compartment],
            )

            pm.model_to_graphviz(model=model)

        return model



# class BayesModel(Sampler):
#     def __init__(
#         self, prior_loc, prior_scale, compartment="[y_gut]", n_post=2000, **kwargs
#     ):
#         super().__init__(
#             loc=kwargs["true_loc"],
#             scale=kwargs["true_scale"],
#             name=kwargs["parameter_name"],
#             n=kwargs["n_sampler"],
#             steps=kwargs["sampler_steps"],
#             model_path=kwargs["model_path"],
#         )
#
#         self.prior_loc = prior_loc
#         self.prior_scale = prior_scale
#         self.n_post = n_post
#         self.compartment = compartment
#         print("\n-------------------- Model Initialized --------------------")
#
#         self.model_bayes = None
#         self.post_samples = None
#
#         self.model_bayes_setup()
#
#     def model_bayes_setup(self):
#         # Forward training
#         @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
#         def pytensor_forward_model_matrix(theta):
#             self.model.resetAll()
#             for par_name, value in zip(self.name, theta):
#                 self.model.setValue(par_name, value)
#             sim = self.model.simulate(start=0, end=10, steps=self.steps)
#             sim_df = pd.DataFrame(sim, columns=sim.colnames)
#             return sim_df[self.compartment]

if __name__ == "__main__":

    bayes_model = BayesModel(ode_mod=MODEL_SIMPLE_PK,
                             parameter='k',
                             compartment='[y_gut]',
                             prior_parameters={
                                 'loc': np.log(1.5),
                                 's': 2
                             },
                             f_prior_dsn=pm.LogNormal)

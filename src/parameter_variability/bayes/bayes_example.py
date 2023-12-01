import json

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from config import parse_args
from matplotlib import pyplot as plt
from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
from scipy.stats import lognorm
from simulation2_sampling import Sampler


class BayesModel(Sampler):
    def __init__(
        self, prior_loc, prior_scale, compartment="[y_gut]", n_post=2000, **kwargs
    ):
        super().__init__(
            loc=kwargs["true_loc"],
            scale=kwargs["true_scale"],
            name=kwargs["parameter_name"],
            n=kwargs["n_sampler"],
            steps=kwargs["sampler_steps"],
            model_path=kwargs["model_path"],
        )

        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.n_post = n_post
        self.compartment = compartment
        print("\n-------------------- Model Initialized --------------------")

        self.model_bayes = None
        self.post_samples = None

        self.model_bayes_setup()

    def model_bayes_setup(self):
        # Forward training
        @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
        def pytensor_forward_model_matrix(theta):
            self.model.resetAll()
            for par_name, value in zip(self.name, theta):
                self.model.setValue(par_name, value)
            sim = self.model.simulate(start=0, end=10, steps=self.steps)
            sim_df = pd.DataFrame(sim, columns=sim.colnames)
            return sim_df[self.compartment]

        with pm.Model() as model:
            k = pm.LogNormal("k", mu=self.prior_loc, sigma=self.prior_scale, initval=1)

            sigma = pm.HalfNormal("sigma", sigma=1)

            # ODE solution function
            ode_soln = pytensor_forward_model_matrix(k)

            # likelihood
            pm.LogNormal(
                name=self.compartment,
                mu=ode_soln,
                sigma=sigma,
                observed=self.df_sampler[self.compartment],
            )

            pm.model_to_graphviz(model=model)

        print("\n------------------ Model Setup Finalized ------------------")

        self.model_bayes = model


if __name__ == "__main__":
    args = parse_args()

    BayesModel(
        # Sampler Params
        true_loc=args.sampler_mean,
        true_scale=args.sampler_variance,
        n_sampler=args.n,
        parameter_name=args.parameter,
        sampler_steps=args.steps,
        model_path=args.model_path,
        # Model params
        prior_loc=args.prior_mean,
        prior_scale=args.prior_variance,
        n_post=args.n_post,
        compartment=args.compartment,
    )

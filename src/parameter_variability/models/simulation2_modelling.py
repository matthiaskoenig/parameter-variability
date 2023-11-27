import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.stats import lognorm
from simulation2_sampling import Sampler
import pymc as pm
import pytensor
import pytensor.tensor as pt


class BayesModel(Sampler):

    def __init__(self, prior_loc, prior_scale, compartment='[y_gut]', n_post=2000, **kwargs):
        super().__init__(loc=kwargs['true_loc'],
                         scale=kwargs['true_scale'],
                         name=kwargs['parameter_name'],
                         n=kwargs['n_sampler'],
                         steps=kwargs['sampler_steps'],
                         model_path=kwargs['model_path']
                         )

        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.n_post = n_post
        self.compartment = compartment

        self.model_bayes = None
        self.post_samples = None

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
            k = pm.LogNormal('k', mu=np.log(2.5), sigma=1, initval=self.init_val['k'])

            PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
            sigma = PositiveNormal('sigma', mu=0, sigma=1)

            # ODE solution function
            ode_soln = pytensor_forward_model_matrix(
                pm.math.stack([k])
            )

            # likelihood
            pm.LogNormal('c_obs', mu=pm.math.log(ode_soln), sigma=sigma, observed=self.df_sampler['[y_gut]'])















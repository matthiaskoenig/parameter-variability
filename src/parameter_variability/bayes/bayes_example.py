import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Union, Sequence

import arviz as az
import numpy as np
import pymc as pm

import pytensor.tensor as pt
import roadrunner
import xarray as xr
from matplotlib import pyplot as plt
from pytensor.compile.ops import as_op
from scipy import stats

from parameter_variability import MODEL_SIMPLE_PK, RESULTS_DIR
from parameter_variability.bayes.sampler import DistDefinition, Sampler, \
    SampleSimulator, sampling_analysis
from parameter_variability.console import console


@dataclass
class BayesModel:
    """Perform Bayesian Inference on Parameter of ODE model.

    This is based on SBML as a model format.
    """
    sbml_model: Union[str, Path]
    observable: str  # FIXME: should be list
    init_values: Dict[str, float]
    f_prior_dsns: Dict[str, Callable]
    prior_parameters: Dict[str, Dict[str, float]]

    def ls_soln(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        pass

    def setup(
        self, data: xr.Dataset, end: int, steps: int,
    ) -> pm.Model:
        """Initialization of Priors and Likelihood"""
        n_sim = data.sizes["sim"]
        coords = {'sim': data['sim']}
        rr_model: roadrunner.RoadRunner = roadrunner.RoadRunner(self.sbml_model)
        # minimal selection
        rr_model.timeCourseSelections = [self.observable]

        @as_op(itypes=[pt.dmatrix, pt.ivector], otypes=[pt.dmatrix])
        def pytensor_forward_model_matrix(theta: np.ndarray, h: pm.ConstantData):
            """ODE solution function.

            Run the forward simulation for the sampled parameters theta.
            """
            y = np.empty(shape=(steps+1, h.size))

            for ksim in h:
                rr_model.resetAll()
                for kkey, key in enumerate(self.prior_parameters):
                    # FIXME: make work for multiple parameter
                    rr_model.setValue(key, theta[ksim, kkey])  # theta[ksim, kkey]
                # FIXME: use timepoints which match samples
                sim = rr_model.simulate(start=0, end=end, steps=steps)
                # store data
                # y[ksim, :, kobs] = sim[self.observable[kobs]]
                y[:, ksim] = sim[self.observable]
            # console.print(y)
            # console.print(y.shape)
            return y

        with pm.Model(coords=coords) as model:
            sims = pm.ConstantData('sim_idx', data['sim'], dims='sim')
            # prior distribution
            p_prior_dsns: Dict[str, np.ndarray] = {}
            for pid in self.prior_parameters:
                dsn_pars = self.prior_parameters[pid]
                dsn_f = self.f_prior_dsns[pid]

                p_prior_dsns[pid] = dsn_f(
                    pid,
                    mu=dsn_pars["loc"],
                    sigma=dsn_pars["s"],
                    initval=np.repeat(self.init_values[pid], n_sim),
                    # shape=(n_sim,),
                    dims="sim",
                )

                # p_prior_dsns[pid] = p_prior_dsns[pid][sims]

            # errors
            sigma = pm.HalfNormal("sigma", sigma=1)

            # ODE solution function
            theta: Sequence["TensorLike"] = \
                [p_prior_dsns[pid][sims] for pid in self.prior_parameters]
            theta_tensor: np.ndarray = pm.math.stack(theta, axis=1)\

            ode_soln = pytensor_forward_model_matrix(theta_tensor, sims)

            # likelihood
            pm.LogNormal(
                name=self.observable,
                mu=ode_soln,
                sigma=sigma,
                observed=data[self.observable].transpose('time', 'sim'),
            )

        return model

    def sampler(
        self, model: pm.Model, tune: int, draws: int, chains: int
    ) -> az.InferenceData:
        """Definition of the Sampling Process"""

        vars_list = list(model.values_to_rvs.keys())[:-1]
        print(f"Variables: {vars_list}\n")
        with model:
            trace = pm.sample(
                step=[pm.Slice(vars_list)], tune=tune, draws=draws, chains=chains,
                # cores=15  # Number of CPUs - 2
            )

        return trace

    def plot_trace(self, trace: az.InferenceData) -> None:
        """Trace plots of the parameters sampled"""
        console.print(az.summary(trace, stat_focus="median"))

        az.plot_trace(trace, compact=True, kind="trace")
        plt.suptitle("Trace plots")
        plt.tight_layout()
        plt.show()


def bayes_analysis(
    bayes_model: BayesModel,
    sampler: Sampler,
    tune: int = 2000, draws: int = 4000, chains: int = 4,
    n: int = 1, end: int = 20, steps: int = 100
) -> None:

    # Sampling of data (FIXME: make this work only with the data; )
    console.rule("Sampling", align="left", style="white")
    data_err = sampling_analysis(
        sampler=sampler,
        n=n,
        end=end,
        steps=steps,
    )

    console.rule(f"Setup Bayes model")
    mod = bayes_model.setup(data_err, end, steps)
    console.print(mod)

    console.rule(f"Sampling for {n=}") # FIXME: Save results to investigate later
    sample = bayes_model.sampler(mod, tune=tune, draws=draws, chains=chains)

    console.rule(f"Results for {n=}")
    bayes_model.plot_trace(sample)


if __name__ == "__main__":

    # model definition
    bayes_model = BayesModel(
        sbml_model=MODEL_SIMPLE_PK,
        observable="[y_gut]",  # FIXME: support multiple
        init_values={
            "k": 2.0,
            "CL": 1.0,
        },
        prior_parameters={
            "k": {"loc": np.log(1.0), "s": 0.5},
            "CL": {"loc": np.log(1.0), "s": 0.5}
        },
        f_prior_dsns={
            "k": pm.LogNormal,
            "CL": pm.LogNormal,
        },
    )
    console.print(f"{bayes_model=}")

    # example sampler
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
            ),
            DistDefinition(
                parameter="CL",
                f_distribution=stats.lognorm,
                distribution_parameters={
                    "loc": np.log(2.5),
                    "s": 1,
                },
            )
        ],
    )
    console.print(f"{sampler=}")

    bayes_analysis(
        bayes_model=bayes_model,
        tune=2000,
        draws=4000,
        chains=4,
        sampler=sampler,
        n=5
    )

    # FIXME: bias in the sampling
    # FIXME: make work for multiple parameters
    # FIXME: make work for multiple observables

    # bayes_analysis(sampler, bayes_model, n=2)

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

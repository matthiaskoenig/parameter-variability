from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union, Sequence

import arviz as az
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import pymc as pm

import pytensor.tensor as pt
import roadrunner
import xarray as xr
from matplotlib import pyplot as plt
from pytensor.compile.ops import as_op
from scipy import stats

from parameter_variability import MODEL_SIMPLE_PK, RESULTS_DIR
from parameter_variability.bayes.pymc_only.sampler import DistDefinition, Sampler, SampleSimulator, sampling_analysis
from parameter_variability.console import console


@dataclass
class BayesModel:
    """Perform Bayesian Inference on Parameter of ODE model.

    Attributes
    ----------
    sbml_model:
        path to the SBML xml file
    observable:
        name of the compartment observed concentration e.g. [y_gut] or [y_cent]
    init_values:
        initial values on where to start the MCMC samplers
    f_prior_dsns:
        dict of PyMC prior distributions
    prior_parameters:
        parameters for the PyMC prior distributions

    """
    sbml_model: Union[str, Path]
    observable: str  # FIXME: should be list
    init_values: Dict[str, float]
    f_prior_dsns: Dict[str, Callable]
    prior_parameters: Dict[str, Dict[str, float]]

    def ls_soln(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        # TODO: Use the ls solution for the init values
        pass

    def setup(
        self, data: xr.Dataset, end: int, steps: int, plot_model: bool = True,
        use_true_thetas: bool = False
    ) -> pm.Model:
        """Initialization of Priors and Likelihood

        Parameters
        ----------
        data:
            simulations dataframe
        end:
            end of the SBML forward simulation
        steps:
            steps on the SBML forward simulation
        plot_model:
            PLot model diagram after set up
        use_true_thetas:
            Use the true thetas as initial values

        Returns
        -------
        model:
            PyMC-based Bayesian model ready to be smapled

        """
        coords: Dict[str, ArrayLike] = {'sim': data['sim'], 'time': data['time']}
        rr_model: roadrunner.RoadRunner = roadrunner.RoadRunner(self.sbml_model)
        # minimal selection
        rr_model.timeCourseSelections = [self.observable]

        @as_op(itypes=[pt.dmatrix, pt.ivector], otypes=[pt.dmatrix])
        def pytensor_forward_model_matrix(theta: np.ndarray, sims: pt.TensorConstant):
            """ODE solution function.

            Run the forward simulation for the sampled parameters theta.

            Parameters
            ----------
            theta:
                draws coming from the MCMC sampler
            sims:
                amount of simulations in the data

            Returns
            -------
            y:
                forward simulation based on the SBML model

            """
            y = np.empty(shape=(steps+1, sims.size))

            for ksim in sims:
                rr_model.resetAll()
                for kkey, key in enumerate(self.prior_parameters):
                    rr_model.setValue(key, theta[ksim, kkey])

                sim = rr_model.simulate(start=0, end=end, steps=steps)
                # store data
                # y[ksim, :, kobs] = sim[self.observable[kobs]]
                y[:, ksim] = sim[self.observable]

            return y

        with pm.Model(coords=coords) as model:
            # TODO: Add correlation matrix between priors and/or sims
            # Simulation array for forward modelling
            simulations = pm.ConstantData('simulations', data['sim'], dims='sim')
            # prior distribution
            p_prior_dsns: Dict[str, np.ndarray] = {}
            for pid in self.prior_parameters:
                dsn_pars = self.prior_parameters[pid]
                dsn_f = self.f_prior_dsns[pid]

                if use_true_thetas:
                    init = self.init_values[pid]
                else:
                    init = np.repeat(self.init_values[pid], data['sim'].size)

                p_prior_dsns[pid] = dsn_f(
                    pid,
                    mu=dsn_pars["loc"],
                    sigma=dsn_pars["s"],
                    initval=init,
                    # shape=(data['sim'].size,),
                    dims="sim",
                )

            # errors
            sigma = pm.HalfNormal("sigma", sigma=1)

            # ODE solution function
            theta: Sequence[pt.TensorLike] = \
                [p_prior_dsns[pid] for pid in self.prior_parameters]
            theta_tensor: np.ndarray = pm.math.stack(theta, axis=1)

            ode_soln = pytensor_forward_model_matrix(theta_tensor, simulations)

            # likelihood
            pm.LogNormal(
                name=self.observable,
                mu=ode_soln,
                sigma=sigma,
                observed=data[self.observable].transpose('time', 'sim'),
                dims=('time', 'sim')
            )

        if plot_model:
            pm.model_to_graphviz(model) \
                .render(directory=RESULTS_DIR/'graph',
                        filename='bayes_model_graph.gv',
                        view=True)

        return model

    def sampler(
        self, model: pm.Model, tune: int, draws: int, chains: int
    ) -> az.InferenceData:
        """Definition of the Sampling Process

        Parameters
        ----------
        model:
            PyMC-based Model
        tune:
            amount of draws to be initially discarded
        draws:
            amount of draws to keep
        chains:
            number of chains to draw samples from

        Returns
        -------
        trace:
            MCMC draws from the specified model

        """

        vars_list = list(model.values_to_rvs.keys())[:-1]
        print(f"Variables: {vars_list}\n")
        with model:
            trace = pm.sample(
                step=[pm.Slice(vars_list)], tune=tune, draws=draws, chains=chains,
                # cores=15  # Number of CPUs - 2
            )

        return trace

    def plot_trace(self, trace: az.InferenceData) -> None:
        """Trace plots of the parameters sampled

        Parameters
        ----------
        trace:
            MCMC draws from the specified model

        Returns
        -------
        plot:
            Diagnostic plot of the samples

        """
        console.print(az.summary(trace, stat_focus="median"))
        az.plot_trace(trace, compact=True, kind="trace")
        plt.suptitle("Trace plots")
        plt.tight_layout()
        plt.show()

    def plot_simulations(
        self, data: xr.Dataset, trace: az.InferenceData,
        num_samples: int, forward_end: int, forward_steps: int
    ) -> None:
        """Plot observable with the simulations based on the MCMC samples

        Parameters
        ----------
        data:
            simulations dataframe
        trace:
            MCMC draws from the specified model
        num_samples:
            amount of samples from the trace to be used on the SBML forward model
        forward_end:
            end of the SBML forward simulation
        forward_steps:
            steps on the SBML forward simulation

        Returns
        -------
        plot:
            Plot comparing the observed data with other possible simulations
            based on the Bayesian sampler

        """

        sims = data['sim'].values
        n_sim = data['sim'].size
        rr_model: roadrunner.RoadRunner = roadrunner.RoadRunner(self.sbml_model)

        f, axes = plt.subplots(
            nrows=n_sim,
            ncols=1,
            dpi=300,
            figsize=(5, 5 * n_sim),
            layout="constrained",
        )
        axes = [axes] if n_sim == 1 else axes

        trace_ex = az.extract(trace, num_samples=num_samples)

        for s, ax in zip(sims, axes):

            df_s = data.sel(sim=s).to_dataframe().reset_index()
            trace_s = trace_ex.sel(sim=s).to_dataframe().reset_index(drop=True)
            # plot observable
            ax.plot(
                df_s["time"],
                df_s[self.observable],
                alpha=0.7,
                color="tab:blue",
                marker="o",
                linestyle="None",
            )

            # plot sims
            for _, row in trace_s.iterrows():
                rr_model.resetAll()
                for key in self.prior_parameters:
                    rr_model.setValue(key, row[key])
                sim = rr_model.simulate(start=0, end=forward_end, steps=forward_steps)
                sim = pd.DataFrame(sim, columns=sim.colnames)

                ax.plot(
                    sim['time'],
                    sim[self.observable],
                    alpha=0.2,
                    lw=1,
                    linestyle='solid'
                )

            ax.set_xlabel("Time [min]")
            ax.set_ylabel("Concentation [mM]")
            ax.set_title(f"Compartment: {self.observable}\nSimulation: {s}")

        plt.show()


def bayes_analysis(
    bayes_model: BayesModel,
    sampler: Sampler,
    tune: int = 2000, draws: int = 4000, chains: int = 4,
    n: int = 1, end: int = 20, steps: int = 100,
    use_true_thetas: bool = False
) -> None:
    """Wrap up for the Bayesian analysis

    Parameters
    ----------
    bayes_model:
        object with the PyMC setup
    sampler:
        object with the forward simulation
    tune:
        amount of draws to be initially discarded
    draws:
        amount of draws to keep
    chains:
        number of chains to draw samples from
    n:
        number of simulations to create for the toy example
    end:
        end of the SBML forward simulation
    steps:
        steps on the SBML forward simulation
    use_true_thetas:
            Use the true thetas as initial values

    Returns
    -------
    result:
        Result of the analysis

    """

    # Sampling of data (FIXME: make this work only with the data; )
    console.rule("Sampling", align="left", style="white")
    data_err, true_thetas = sampling_analysis(
        sampler=sampler,
        n=n,
        end=end,
        steps=steps,
    )

    console.rule(f"Setup Bayes model")
    if use_true_thetas:
        bayes_model.init_values = true_thetas

    mod = bayes_model.setup(data_err, end, steps, plot_model=True,
                            use_true_thetas=use_true_thetas)
    console.print(mod)

    console.rule(f"Sampling for {n=}") # FIXME: Save results to investigate later
    sample = bayes_model.sampler(mod, tune=tune, draws=draws, chains=chains)

    console.rule(f"Results for {n=}")
    bayes_model.plot_trace(sample)

    console.rule(f'Simulation for {n=}')
    bayes_model.plot_simulations(data_err, sample, num_samples=25,
                                 forward_end=end, forward_steps=steps)


if __name__ == "__main__":

    # model definition
    bayes_model = BayesModel(
        sbml_model=MODEL_SIMPLE_PK,
        observable="[y_gut]",  # Other options are '[y_cent]' and '[y_peri]'
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
        chains=3,
        sampler=sampler,
        n=5
    )

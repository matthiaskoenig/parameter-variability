"""Module for sampling parameters from models.

Samples parameters from models and runs corresponding forward simulations.
Using a SBML Model, declare a random distribution to generate solutions to the model as simulations.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import roadrunner
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from parameter_variability import MODEL_SIMPLE_PK
from parameter_variability.console import console


@dataclass
class DistDefinition:
    """Definition of distribution via scipy callables."""

    parameter: str
    f_distribution: Callable
    distribution_parameters: Dict[str, float]


@dataclass
class Sampler:
    """Samples parameters from a SBML model.
    FIXME: include covariance structure in the sampling.
    """

    model: Union[str, Path]
    distributions: List[DistDefinition]

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """Sample from the random distributions to feed the SBML sampler

        Parameters
        ----------
        n:
            amount of parameters to draw from the distributions

        Returns
        -------
        samples:
            dictionary with an array of shape n
        """
        samples: Dict[str, np.ndarray] = {}
        for dist in self.distributions:
            dsn = dist.f_distribution(**dist.distribution_parameters)
            samples[dist.parameter] = dsn.rvs(size=n)
        return samples

    @staticmethod
    def plot_samples(
        thetas: Dict[str, np.ndarray],
        distributions: Optional[List[DistDefinition]] = None,
    ) -> Figure:
        """Plot the samples

        Parameters
        ----------
        thetas:
            dictionary of the parameters sampled
        distributions:

        """
        n_pars = len(thetas)

        fontdict = {"fontweight": "bold"}
        f, axes = plt.subplots(
            nrows=n_pars,
            ncols=1,
            dpi=300,
            figsize=(5, 5 * n_pars),
            layout="constrained",
        )
        if n_pars == 1:
            axes = [axes]

        for k, sid in enumerate(thetas):
            ax = axes[k]
            ax.set_xlabel(sid, **fontdict)

            theta_values = thetas[sid]

            # plot histogram
            ax.hist(
                theta_values,
                color="tab:orange",
                density=True,
            )

            # plot values
            for kv, value in enumerate(theta_values):
                label = (
                    f"{sid} samples (n={len(theta_values)})"
                    if kv == 0
                    else "__nolabel__"
                )
                ax.axvline(
                    value,
                    linestyle="--",
                    color="tab:blue",
                    alpha=0.5,
                    label=label,
                )

            # plot the distribution
            if distributions:
                dist = distributions[k]
                dsn = dist.f_distribution(**dist.distribution_parameters)
                theta = np.linspace(dsn.ppf(0.001), dsn.ppf(0.999), 500)

                ax.plot(
                    theta,
                    dsn.pdf(theta),
                    color="black",
                    label=f"{sid} distribution",
                )

        for ax in axes:
            ax.set_ylabel("Probability Density Function", **fontdict)
            ax.legend()

        return f


@dataclass
class SampleSimulator:
    """Simulation of parameter samples."""

    model: Union[str, Path]
    thetas: Dict[str, np.ndarray]

    def simulate(self, start, end, steps, **kwargs) -> xr.Dataset:
        """Simulate data as xarray dataset."""
        self.model = roadrunner.RoadRunner(self.model)
        n_sim = list(self.thetas.values())[0].size

        dfs: List[pd.DataFrame] = []
        for ksim in range(n_sim):
            self.model.resetAll()

            for parameter, values in self.thetas.items():
                self.model.setValue(parameter, values[ksim])

            sim = self.model.simulate(start=start, end=end, steps=steps, **kwargs)
            df = pd.DataFrame(sim, columns=sim.colnames).set_index("time")
            dfs.append(df)

        # create xarray
        dset = xr.concat([df.to_xarray() for df in dfs],
                         dim=pd.Index(np.arange(n_sim), name='sim'))

        return dset

    def save_data(self, data: xr.Dataset, results_path: Path):
        """Store dataset as netCDF."""
        data.to_netcdf(results_path)

    def load_data(self, results_path):
        """Load dataset from netCDF."""
        return xr.open_dataset(results_path)

    @staticmethod
    def apply_errors_to_data(
        data: xr.Dataset, variables: List[str], error_scale: float = 0.01
    ) -> xr.Dataset:
        """Applies errors to the simulation data."""
        n_sim = data.sizes["sim"]
        n_time = data.sizes["time"]

        # normal distributed errors (independent of data)
        errors_dsn = stats.halfnorm(loc=0, scale=error_scale)
        errors = errors_dsn.rvs((n_sim, n_time))

        variables = variables[0] if len(variables) <= 1 else variables

        # additive errors
        data_err = data.copy()
        data_err[variables] = data_err[variables] + errors

        return data_err

    @staticmethod
    def plot_data(data: xr.Dataset, data_err: xr.Dataset, variables: List[str]) -> None:
        n_vars = len(variables)
        fontdict = {"fontweight": "bold"}

        f, axes = plt.subplots(
            nrows=n_vars,
            ncols=1,
            dpi=300,
            figsize=(5, 5 * n_vars),
            layout="constrained",
        )
        axes = [axes] if n_vars == 1 else axes

        sims = data.sim.values
        for kp, var in enumerate(variables):
            ax = axes[kp]

            for s in sims:
                # plot data
                df = data.isel(sim=s)[var].to_dataframe().reset_index()
                ax.plot(df["time"], df[var], alpha=0.7, color="black")

                df_err = data_err.isel(sim=s)[var].to_dataframe().reset_index()
                ax.plot(
                    df_err["time"],
                    df_err[var],
                    alpha=0.7,
                    color="tab:blue",
                    marker="o",
                    linestyle="None",
                )

            ax.set_xlabel("Time [min]")
            ax.set_ylabel("Concentation [mM]")
            ax.set_title(f"Compartment: {var}")

        plt.show()


def sampling_analysis(
    sampler: Sampler, n: int, end=20, steps=100, seed: Optional[int] = None
) -> tuple[xr.Dataset, Dict[str, np.ndarray]]:
    """Creates samples from sampler with control plots."""

    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

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
        data=data, data_err=data_err, variables=["[y_gut]", "[y_cent]", "[y_peri]"]
    )
    return data_err, thetas


if __name__ == "__main__":
    seed = 1234
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
        ],
    )
    sampling_analysis(sampler, n=1, seed=seed)
    sampling_analysis(sampler, n=100, seed=seed)

    sampler2p = Sampler(
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
                    "loc": np.log(20),
                    "s": 1.5,
                },
            ),
        ],
    )
    sampling_analysis(sampler2p, n=25, seed=seed)

    # from parameter_variability import RESULTS_DIR
    # testing loading and saving of data
    # dset_path = RESULTS_DIR / "test.nc"
    # simulator.save_data(data, dset_path)
    # data2 = simulator.load_data(dset_path)
    # console.print(data2)

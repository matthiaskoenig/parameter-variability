"""Module for sampling parameters from models.

Samples parameters from models and runs corresponding forward simulations.
Using a SBML Model, declare a random distribution to generate solutions to the model as simulations.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Union, List, Callable, Dict, Any

import numpy as np
import pandas as pd
import roadrunner
from matplotlib import pyplot as plt
from scipy import stats
from dataclasses import dataclass
import xarray as xr

from parameter_variability.console import console
from parameter_variability import MODEL_SIMPLE_PK


@dataclass
class Sampler:
    """Samples parameters from a SBML model."""

    model: Union[str, Path]
    parameter: str
    f_distribution: Callable
    distribution_parameters: Dict[str, float]

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """Sample from sampler."""
        dsn = self.f_distribution(**self.distribution_parameters)
        return {self.parameter: dsn.rvs(size=n)}


@dataclass
class SampleSimulator:
    """Simulation of parameter samples."""
    model: Union[str, Path]
    thetas: Dict[str, np.ndarray]

    def simulate(self, start, end, steps, **kwargs) -> xr.Dataset:
        """Simulate data."""
        self.model = roadrunner.RoadRunner(self.model)
        n_sim = len(list(self.thetas.values())[0])

        dfs: List[pd.DataFrame] = []
        for i in range(n_sim):
            self.model.resetAll()
            for parameter, values in self.thetas.items():
                self.model.setValue(parameter, values[i])

            sim = self.model.simulate(start=start, end=end, steps=steps, **kwargs)

            # Adding the errors: y_i = yhat_i + errors_i
            df = pd.DataFrame(sim, columns=sim.colnames)
            dfs.append(df)

        # create xarray
        dset = xr.concat([df.to_xarray() for df in dfs], dim="thetas")

        return dset

    def apply_errors(data: xr.Dataset, variables: List[str]) -> xr.Dataset:
        """Applies errors to the simulation."""

        # FIXME: error has to be applied on the state variables;


        pass
        # # error distribution
        # errors_dsn = stats.halfnorm
        # errors_dsn = errors_dsn()  # ??? how large are the errors
        # self.errors_distribution = errors_dsn
        # TODO: implement
        # sim[:, 1:] = sim[:, 1:] + errors.reshape((step_correction, 1))
        # errors = self.errors_distribution.rvs(size=step_correction)
        return data

    def save_data(self, data: xr.Dataset, results_path: Path):
        """Store dataset as netCDF."""
        data.to_netcdf(results_path)

    def load_data(self, results_path):
        """Load dataset from netCDF."""
        return xr.open_dataset(results_path)


# def plot(self):
#     # TODO: add generalization to several parameters
#     fig, ax = plt.subplots(2, 1, figsize=(8, 12))
#
#     # PDF of theta and the value drawn
#     theta = np.linspace(
#         self.true_distribution.ppf(0.001), self.true_distribution.ppf(0.999), 500
#     )
#     ax[0].plot(
#         theta,
#         self.true_distribution.pdf(theta),
#         color="crimson",
#         label=f"$k \sim lognorm({self.loc:.2f}, {self.scale:.2f})$",
#     )
#     ax[0].axvline(self.thetas, linestyle="--", color="green", label="$k$ drawn")
#     ax[0].set_xlabel("$k$")
#     ax[0].set_ylabel("Probability Density Function")
#     ax[0].legend()
#
#     # Generated data
#     self.df_sampler.plot(
#         x="time", y=["[y_gut]", "[y_cent]", "[y_peri]"], ax=ax[1], style=".-"
#     )
#     ax[1].set_xlabel("Time [min]")
#     ax[1].set_ylabel("Concentration [mM]")
#     ax[1].legend()
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    from parameter_variability import RESULTS_DIR

    console.rule("Sampling", align="left", style="white")
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

    thetas = sampler.sample(n=10)
    console.print(f"{thetas=}")

    console.rule("Simulation", align="left", style="white")
    simulator = SampleSimulator(
        model=MODEL_SIMPLE_PK,
        thetas=thetas,
    )
    data = simulator.simulate(start=0, end=10, steps=10)
    console.print(data)
    dset_path = RESULTS_DIR / "test.nc"
    simulator.save_data(data, dset_path)
    data2 = simulator.load_data(dset_path)
    console.print(data2)

    simulator.apply_errors(data, variables=["[y_gut]", "[y_gut]", "[y_peri]"])

    console.rule("Plotting", align="left", style="white")


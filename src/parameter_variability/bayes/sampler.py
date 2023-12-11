"""Module for sampling parameters from models.

Samples parameters from models and runs corresponding forward simulations.
Using a SBML Model, declare a random distribution to generate solutions to the model as simulations.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Union, List, Callable, Dict, Any

import matplotlib.pyplot
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

    def plot(self, thetas: Dict[str, np.array]):
        dsn = self.f_distribution(**self.distribution_parameters)
        theta_values = list(thetas.values())[0]

        _, ax = plt.subplots()

        theta = np.linspace(
            dsn.ppf(0.001), dsn.ppf(0.999), 500
        )
        ax.plot(
                theta, dsn.pdf(theta), color="crimson",
                label=f"${self.parameter} \sim "
                      f"lognorm({self.distribution_parameters['loc']:.2f}, "
                      f"{self.distribution_parameters['s']:.2f})$"
        )

        for i, value in enumerate(theta_values, start=1):

            if i < len(theta_values):
                ax.axvline(value, linestyle="--", color="green", alpha=0.5)
            else:
                ax.axvline(value, linestyle="--", color="green", alpha=0.5,
                           label=f"${self.parameter}$ drawn (n={i})")

        ax.set_xlabel(f"${self.parameter}$")
        ax.set_ylabel("Probability Density Function")
        ax.legend()

        plt.show()


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
            df = pd.DataFrame(sim, columns=sim.colnames).set_index('time')
            dfs.append(df)

        # create xarray
        dset = xr.concat([df.to_xarray() for df in dfs], dim="sim")

        return dset

    def apply_errors(self, data: xr.Dataset, variables: List[str],
                     error_scale: float = 0.05) -> xr.Dataset:
        """Applies errors to the simulation."""
        n_sim = data.sizes['sim']
        n_time = data.sizes['time']

        errors_dsn = stats.halfnorm(loc=0, scale=error_scale)
        errors = errors_dsn.rvs((n_sim, n_time))

        variables = variables[0] if len(variables) <= 1 else variables

        data[variables] = data[variables] + errors

        return data

    def plot(self, data: xr.Dataset, variables: List[str]) -> None:

        sims = data.sim.values
        for var in variables:
            _, ax = plt.subplots()
            for s in sims:
                df_s = data.isel(sim=s)[var].to_dataframe().reset_index()
                df_s.plot(x='time', y=var, ax=ax, label=s)
            ax.set_xlabel('Time [min]')
            ax.set_ylabel('Concentation [mM]')
            ax.set_title(f'Compartment: {var}')

        plt.show()

    def save_data(self, data: xr.Dataset, results_path: Path):
        """Store dataset as netCDF."""
        data.to_netcdf(results_path)

    def load_data(self, results_path):
        """Load dataset from netCDF."""
        return xr.open_dataset(results_path)


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

    console.rule('Thetas PDF', align='left', style='white')
    sampler.plot(thetas)

    console.rule("Simulation", align="left", style="white")
    simulator = SampleSimulator(
        model=MODEL_SIMPLE_PK,
        thetas=thetas,
    )
    data = simulator.simulate(start=0, end=10, steps=10)
    console.print(data)

    console.rule("Errors", align="left", style="white")
    data = simulator.apply_errors(data, variables=['[y_gut]', '[y_cent]'])
    console.print(data)

    console.rule("Simulation plots", align="left", style="white")
    simulator.plot(data, variables=['[y_gut]', '[y_cent]', '[y_peri]'])

    dset_path = RESULTS_DIR / "test.nc"
    simulator.save_data(data, dset_path)
    data2 = simulator.load_data(dset_path)
    console.print(data2)




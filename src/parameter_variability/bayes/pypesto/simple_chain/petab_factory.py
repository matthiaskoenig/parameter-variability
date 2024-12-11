from pathlib import Path
from typing import Optional

import xarray
from matplotlib import pyplot as plt
import roadrunner
import pandas as pd
import numpy as np
from dataclasses import dataclass
import xarray as xr
from parameter_variability import BAYES_DIR, MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN
from parameter_variability.console import console
from dataclasses import dataclass
from enum import Enum
from scipy.stats import lognorm


class Category(str, Enum):
    """Categories."""
    MALE = "male"
    FEMALE = "female"
    OLD = "old"
    YOUNG = "young"


colors = {
    Category.MALE: "tab:blue",
    Category.FEMALE: "tab:red",
    Category.OLD: "tab:orange",
    Category.YOUNG: "tab:green",
}

@dataclass
class LognormParameters:
    """Lognormal parameters"""
    n: int  # number of samples
    sigma: float  # standard deviation
    mu: float  # mean

@dataclass
class SimulationSettings:
    """Lognormal parameters"""
    start: float
    end: float
    steps: int

def create_male_female_samples(parameters: dict[Category, LognormParameters], seed: Optional[int] = 1234):
    """Create the male and female samples."""
    if seed is not None:
        np.random.seed(seed)

    # samples
    samples: dict[Category, np.ndarray] = {}
    for category, lnpars in parameters.items():
        s = lnpars.sigma
        m = lnpars.mu

        sigma_ln = np.sqrt(np.log(1 + (s / m) ** 2))
        mu_ln = np.log(m) - sigma_ln ** 2 / 2
        samples[category] =np.random.lognormal(mu_ln, sigma_ln, size=lnpars.n)

    return samples

def plot_samples(samples: dict[Category, np.ndarray]):
    console.print(samples_k1)

    # plot distributions
    f, ax = plt.subplots(dpi=300, layout="constrained")
    for category, data in samples.items():
        ax.hist(
            data, density=True, bins='auto', histtype='stepfilled', alpha=0.5,
            color=colors[category], label=f"hist {category.name}"
        )
    for category, data in samples.items():
        mean = np.mean(data)
        std = np.std(data)
        ax.axvline(x=mean, color=colors[category], label=f"mean {category.name}")
        ax.axvline(x=mean + std, color=colors[category], linestyle="--",
                   label=f"__nolabel__")
        ax.axvline(x=mean - std, color=colors[category], linestyle="--",
                   label=f"__nolabel__")

    ax.set_xlabel("parameter")
    ax.set_ylabel("density")
    ax.legend()
    plt.show()

class ODESampleSimulator:
    """Performs simulations with given model and samples."""

    def __init__(self, model_path: Path, abs_tol: float = 1E-6, rel_tol: float = 1E-6):
        """Load model and integrator settings."""
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        integrator: roadrunner.Integrator = self.r.integrator
        integrator.setSetting("absolute_tolerance", abs_tol)
        integrator.setSetting("relative_tolerance", rel_tol)


    def simulate_samples(self, parameters: pd.DataFrame, simulation_settings: SimulationSettings) -> xr.Dataset:
        """Simulate samples with roadrunner."""
        n = parameters.shape[0]
        pids = parameters.columns
        dfs = []
        for _, row in parameters.iterrows():
            self.r.resetAll()

            # set the parameter values
            for pid in pids:
                value = row[pid]
                # console.print(f"{pid}: {value}")
                self.r.setValue(pid, value)

            # simulate
            s = self.r.simulate(
                start=simulation_settings.start,
                end=simulation_settings.end,
                steps=simulation_settings.steps,
            )
            # convert result to data frame
            df = pd.DataFrame(s, columns=s.colnames).set_index("time")
            dfs.append(df)

        dset = xr.concat([df.to_xarray() for df in dfs], dim=pd.Index(np.arange(n), name='sim'))
        return dset


def plot_simulations(dsets: dict[Category, xarray.Dataset]):
    """Plot simulations."""
    console.print(dsets)

    # plot distributions
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=300, layout="constrained")

    alpha = 0.7
    for category, dset in dsets.items():
        color = colors[category]
        nsim = len(dset["sim"])
        for k in range(nsim):
            ax1.plot(
                dset["time"],
                dset["[S1]"].isel(sim=k),
                alpha=alpha,
                color=color
            )
            ax2.plot(
                dset["time"],
                dset["[S2]"].isel(sim=k),
                alpha=alpha,
                color=color
            )

    ax1.set_ylabel("[S1]")
    ax2.set_ylabel("[S2]")
    for ax in ax1, ax2:
        ax.set_xlabel("time")
    plt.show()


def create_petab_example(model_path: Path, fig_path: Path):
    # TODO: create the complete PETab problem
    # Create all files and copy all the files
    shutils.copy()


    # create the measurement table from given data !
    df = example_simulation(model_path, fig_path)

    data = []
    for k, row in df.iterrows():
        for col in ['y_gut', 'y_cent', 'y_peri']:
            data.append({
                "observableId": f"{col}_observable",
                "preequilibrationConditionId": None,
                "simulationConditionId": "model1_data1",
                "measurement": 	row[f"{col}_data"],
                MEASUREMENT_UNIT_COLUMN: "mmole/l",
                "time": row["time"],
                MEASUREMENT_TIME_UNIT_COLUMN: "second",
                "observableParameters": None,
                "noiseParameters": None,
            })
    measurement_df = pd.DataFrame(data)
    measurement_df.to_csv(model_path.parent / "measurements_simple_pk.tsv", sep="\t", index=False)

    model_path: Path = Path(__file__).parent / "simple_pk.xml"
    fig_path: Path = Path(__file__).parent / "results"
    fig_path.mkdir(exist_ok=True)
    create_petab_example(model_path, fig_path)

    import petab

    yaml_path = BAYES_DIR / "pypesto" / "simple_pk" / "simple_pk.yaml"
    problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
    console.print(problem)
    errors_exist = petab.lint.lint_problem(problem)
    console.print(f"PEtab errors: {errors_exist}")


if __name__ == '__main__':
    from parameter_variability import MODEL_SIMPLE_CHAIN

    # samples
    samples_k1: dict[Category, np.ndarray] = create_male_female_samples(
        {
            Category.MALE: LognormParameters(mu=1.5, sigma=1.0, n=50),
            Category.FEMALE: LognormParameters(mu=3.0, sigma=0.5, n=100),
            Category.OLD: LognormParameters(mu=10.0, sigma=3, n=20),
            Category.YOUNG: LognormParameters(mu=1.5, sigma=1, n=40),
        }
    )
    plot_samples(samples_k1)

    # simulations
    simulator = ODESampleSimulator(model_path=MODEL_SIMPLE_CHAIN)
    sim_settings = SimulationSettings(start=0.0, end=20.0, steps=300)
    dsets: dict[Category, xarray.Dataset] = {}
    for category, data in samples_k1.items():
        # simulate samples for category
        parameters = pd.DataFrame({"k1": data})
        dset = simulator.simulate_samples(parameters, simulation_settings=sim_settings)
        dsets[category] = dset

    plot_simulations(dsets)














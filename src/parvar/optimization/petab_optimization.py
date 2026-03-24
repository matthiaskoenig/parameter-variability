"""Optimization using petab and pypesto."""

import functools
import logging
import multiprocessing
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import arviz as az
import numpy as np
import pandas as pd
import petab
import pypesto
import xarray as xr
from matplotlib import pyplot as plt
from petab.v1 import Problem
from pypesto import petab as pt
from pypesto.visualize import model_fit
from pymetadata.console import console

from parvar.experiments.utils import get_group_from_pid, get_parameter_from_pid


@dataclass
class PyPestoSampler:
    """Create Petab Problem based on yaml"""

    yaml_file: Path
    fig_path: Path = None
    petab_problem: petab.v1.Problem = None
    pypesto_problem: pypesto.Problem = None
    result: pypesto.Result = None

    def load_problem(self):
        self.petab_problem: Problem = petab.v1.Problem.from_yaml(self.yaml_file)
        importer = pt.PetabImporter(self.petab_problem)

        self.pypesto_problem = importer.create_problem(verbose=True)
        self.results_path = self.yaml_file.parents[0] / "results"
        self.fig_path = self.results_path / "figs"
        self.fig_path.mkdir(parents=True, exist_ok=True)

    def print_optimization_results(self):
        """Print output of the optimization results."""

        console.rule("results", style="white")
        console.print(self.result.summary())

        # match to parameters
        console.print("Parameters:")
        console.print(self.petab_problem.parameter_df)
        parameter_names = self.petab_problem.parameter_df.parameterName

        # best fit
        console.print("Best fit:")
        best_fit: dict = self.result.optimize_result.list[0]
        # console.print(best_fit)
        popts = deepcopy(best_fit["x"])
        for k, scale in enumerate(self.petab_problem.parameter_df.parameterScale):
            # backtransformations
            if scale == "lin":
                continue
            elif scale == "log10":
                popts[k] = 10 ** popts[k]
            elif scale == "log":
                popts[k] = np.exp(popts[k])

        console.print(dict(zip(parameter_names, popts)))

    def plot_optimization_results(self):
        """Create plots of the optimization results."""
        model_fit.visualize_optimized_model_fit(
            petab_problem=self.petab_problem,
            result=self.result,
            pypesto_problem=self.pypesto_problem,
        )
        # plt.show()
        pypesto.visualize.waterfall(self.result)
        plt.savefig(str(self.fig_path) + "/01_waterfall.png")

        pypesto.visualize.parameters(self.result)
        plt.savefig(str(self.fig_path) + "/02_parameters.png")

        # pypesto.visualize.parameters_correlation_matrix(result)
        console.rule("Parameter_hist", style="white")
        # pypesto.visualize.parameter_hist(result=result, parameter_name="kabs")
        # plt.savefig(str(fig_path) + '/03_parameters_hist.png')

        pypesto.visualize.optimization_scatter(self.result)
        plt.savefig(str(self.fig_path) + "/04_opt_scatter.png")

    def run_optimization(
        self,
        maxiter: Optional[float] = 1e4,
        fatol: Optional[float] = 1e-12,
        frtol: Optional[float] = 1e-12,
        optim: Optional[str] = "fides",
        startpoint_method: Optional[str] = "uniform",
        n_starts: Optional[int] = 100,
        seed: Optional[int] = 1,
        create_optimization_plots: bool = True,
        engine: str = "SingleCoreEngine",
    ):
        # Optimization
        if optim == "fides":
            optimizer = pypesto.optimize.FidesOptimizer(
                options={"maxiter": maxiter, "fatol": fatol, "frtol": frtol},
                verbose=logging.WARN,
            )
        else:
            raise ValueError(f"Optimizer {optim} not supported.")

        if startpoint_method == "uniform":
            startpoint_method = pypesto.startpoint.uniform
        else:
            raise ValueError(f"Startpoint method '{startpoint_method}' not supported.")

        # save optimizer trace
        # history_options = pypesto.HistoryOptions(trace_record=True)
        opt_options = pypesto.optimize.OptimizeOptions()
        # console.print(opt_options)

        if engine == "MultiProcessEngine":
            engine = pypesto.engine.MultiProcessEngine()
        elif engine == "SingleCoreEngine":
            engine = pypesto.engine.SingleCoreEngine()
        else:
            raise ValueError(f"Engine {engine} not supported.")

        # Set seed for reproducibility
        if seed:
            np.random.seed(seed)

        console.rule("Optimization", style="white")

        # optimize
        self.result = pypesto.optimize.minimize(
            problem=self.pypesto_problem,
            optimizer=optimizer,
            n_starts=n_starts,
            startpoint_method=startpoint_method,
            engine=engine,
            options=opt_options,
        )

        self.print_optimization_results()

        if create_optimization_plots:
            self.plot_optimization_results()

    def bayesian_sampler(
        self,
        n_samples: int,
        n_chains: int,
    ):
        sampler: Callable = pypesto.sample.AdaptiveMetropolisSampler()
        sampler_w_chains = pypesto.sample.AdaptiveParallelTemperingSampler(
            internal_sampler=sampler,
            n_chains=n_chains,
        )
        self.result = pypesto.sample.sample(
            problem=self.pypesto_problem,
            sampler=sampler_w_chains,
            n_samples=n_samples,
            result=self.result,
        )

        pypesto.sample.effective_sample_size(result=self.result)

        # Plots
        pypesto.visualize.sampling_fval_traces(self.result)
        plt.tight_layout()
        plt.savefig(str(self.fig_path) + "/06_sampling_fval_traces.png")

        pypesto.visualize.sampling_parameter_traces(
            self.result, use_problem_bounds=False, size=(12, 5)
        )
        plt.savefig(str(self.fig_path) + "/07_traces.png")

        pypesto.visualize.sampling_parameter_cis(
            self.result, alpha=[99, 95, 90], size=(10, 5)
        )
        plt.savefig(str(self.fig_path) + "/08_cis.png")

        pypesto.visualize.sampling_1d_marginals(self.result)
        plt.savefig(str(self.fig_path) + "/09_marginals.png")
        # plt.show()

    def get_posterior(self) -> az.InferenceData:
        """trace.

        Interval of high density using arviz, an open source project aiming to
        provide tools for Exploratory Analysis of Bayesian Models that do
        not depend on the inference library used.
        """
        # TODO: Fix arviz data ingestion

        trace_ = self.result.sample_result.trace_x
        # ds = xr.Dataset(trace_)#,
        # coords={'chain': np.arange(trace_.shape[0]),
        #         'draw': np.arange(trace_.shape[1]),
        #         'k1': self.petab_problem.parameter_df.parameterName.array})
        trace = az.convert_to_inference_data(trace_)
        trace.posterior = trace.posterior.rename({"x_dim_0": "parameter"})
        trace.posterior["parameter"] = (
            self.petab_problem.parameter_df.parameterName.values
        )
        return trace

    def results_hdi(self) -> dict:
        hdi = az.hdi(self.get_posterior())
        console.print("HDI for each parameter: ")
        results = {}
        for pid in hdi["parameter"]:
            # console.print(pid.to_numpy())
            hdi_values = hdi.sel(parameter=pid).x.values
            console.print(f"{pid.item()}: {hdi_values}")
            results[pid.item()] = hdi_values
        return results

    def results_dict(self, settings: dict) -> dict:
        dset: xr.Dataset = self.get_posterior().posterior
        # get information on real parameter

        # calculate statistics on posterior
        results = {}
        for pid in dset["parameter"]:
            values: np.ndarray = dset.sel(parameter=pid).x.values
            results[pid.item()] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "ess": self.result.sample_result.effective_sample_size,
                "n_samples": settings["n_samples"],
                "values": values,
            }

        hdi = self.results_hdi()
        for pid, hdi_values in hdi.items():
            results[pid]["hdi_low"] = float(hdi_values[0])
            results[pid]["hdi_high"] = float(hdi_values[1])

        return results

    def results_df(self, uid: str, settings: dict):
        # collect results for parameters
        results = []
        results_petab = self.results_dict(settings=settings)
        for pid, stats in results_petab.items():
            results.append(
                {
                    "id": uid,
                    "group": get_group_from_pid(pid),
                    "parameter": get_parameter_from_pid(pid),
                    "pid": pid,
                    **stats,
                }
            )

        return pd.DataFrame(results)


def optimize_experiment(
    yaml_path: Path,
    results_dir: Path,
    caching: bool = True,
    engine: str = "SingleCoreEngine",
) -> bool:
    """Optimize a single petab problem using PyPesto."""
    uid = yaml_path.parent.name

    console.print()
    console.rule(style="white bold")
    console.print(uid, style="white bold")
    console.rule(style="white bold")
    console.print(yaml_path)

    results_path = results_dir / f"{uid}_results.tsv"
    error_path = results_dir / f"{uid}_errors.tsv"

    if caching and results_path.exists():
        console.print("Cached results: optimization results already exist.")
        return True

    settings = dict(
        maxiter=1e4,
        fatol=1e-12,
        frtol=1e-12,
        optim="fides",
        startpoint_method="uniform",
        n_starts=100,
        seed=1234,
        create_optimization_plots=True,
        engine=engine,
        n_samples=5000,
        n_chains=4,
    )

    try:
        pypesto_sampler = PyPestoSampler(yaml_file=yaml_path)
        pypesto_sampler.load_problem()
        pypesto_sampler.run_optimization(
            maxiter=settings["maxiter"],
            fatol=settings["fatol"],
            frtol=settings["frtol"],
            optim=settings["optim"],
            startpoint_method=settings["startpoint_method"],
            n_starts=settings["n_starts"],
            seed=settings["seed"],
            create_optimization_plots=settings["create_optimization_plots"],
            engine=settings["engine"],
        )
        pypesto_sampler.bayesian_sampler(
            n_samples=settings["n_samples"],
            n_chains=settings["n_chains"],
        )
        pypesto_sampler.results_hdi()

        # save DataFrame
        df = pypesto_sampler.results_df(uid, settings)
        df.to_csv(results_path, sep="\t")
        console.print(f"Results saved to {results_path}", style="green bold")
        return True

    except Exception as e:
        stack_trace = traceback.format_exc()
        with open(error_path, "w") as ferr:
            ferr.write(stack_trace)
            console.print(e)

        console.print(f"Errors saved to {results_path}", style="red bold")

        return False


def optimize_experiments(
    results_dir: Path,
    yaml_paths: list[Path],
    caching: bool = True,
    engine: str = "MultiProcessEngine",
) -> None:
    """Optimize PETab problems serial.

    For remote execution and multiprocessing see below.
    """
    for yaml_path in yaml_paths:
        optimize_experiment(
            yaml_path=yaml_path,
            results_dir=results_dir,
            caching=caching,
            engine=engine,
        )


def optimize_experiments_multicore(
    results_dir: Path,
    yaml_paths: list[Path],
    caching: bool = True,
    engine: str = "SingleCoreEngine",
) -> None:
    """Optimize PETab problems multicore usage.

    Necessary to ensure that no parallelization.
    """

    n_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=n_cores) as pool:
        pool.map(
            functools.partial(
                optimize_experiment,
                results_dir=results_dir,
                caching=caching,
                engine=engine,
            ),
            yaml_paths,
        )

"""Optimization using petab and pypesto."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional, Dict, Union

import arviz as az
import numpy as np
import pandas as pd
import petab
import pypesto
import pypesto.visualize.model_fit as model_fit
import xarray as xr
from matplotlib import pyplot as plt
from petab.v1 import Problem
from pypesto import petab as pt
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

    def optimizer(self, plot: bool = True):
        # FIXME: this should also be general settings
        optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}

        # FIXME: add optimizer option "Fides"
        optimizer = pypesto.optimize.FidesOptimizer(
            options=optimizer_options, verbose=logging.WARN
        )
        # FIXME: add option for the startpoint method.
        startpoint_method = pypesto.startpoint.uniform
        # save optimizer trace
        # history_options = pypesto.HistoryOptions(trace_record=True)
        opt_options = pypesto.optimize.OptimizeOptions()
        console.print(opt_options)

        # FIXME: this should be in settings;
        n_starts = 100  # usually a value >= 100 should be used

        # FIXME: flag for the engine;
        engine = pypesto.engine.MultiProcessEngine()

        # Set seed for reproducibility
        # FIXME: this should be an optional seed setting
        seed = 1
        np.random.seed(seed)

        console.rule("Optimization", style="white")
        # TODO: store much more information on the optimization run;
        # runtime, settings, success, optimization options

        # THis is performing the optimization
        self.result = pypesto.optimize.minimize(
            problem=self.pypesto_problem,
            optimizer=optimizer,
            n_starts=n_starts,
            startpoint_method=startpoint_method,
            engine=engine,
            options=opt_options,
        )

        def print_optimization_results(result: pypesto.Result):
            """Print output of the optimization results."""

            console.rule("results", style="white")
            console.print(result.summary())

            # match to parameters
            console.print("Parameters:")
            console.print(self.petab_problem.parameter_df)
            parameter_names = self.petab_problem.parameter_df.parameterName

            # best fit
            console.print("Best fit:")
            best_fit: dict = result.optimize_result.list[0]
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

        print_optimization_results(self.result)

        if plot:
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

    def bayesian_sampler(
        self,
        sampler: Callable = pypesto.sample.AdaptiveMetropolisSampler(),
        n_samples: int = 10_000,
    ):
        self.result = pypesto.sample.sample(
            problem=self.pypesto_problem,
            sampler=sampler,
            n_samples=n_samples,
            result=self.result,
        )

        pypesto.sample.effective_sample_size(result=self.result)
        ess = self.result.sample_result.effective_sample_size
        print(
            f"Effective sample size per computation time: "
            f"{round(ess / self.result.sample_result.time, 2)}"
        )

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

    def results_dict(self) -> dict:
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
                "values": values,
            }

        hdi = self.results_hdi()
        for pid, hdi_values in hdi.items():
            results[pid]["hdi_low"] = float(hdi_values[0])
            results[pid]["hdi_high"] = float(hdi_values[1])

        console.rule(style="blue bold")
        console.print(results)
        console.rule(style="blue bold")

        return results

    # def results_median(self):
    #     medians = self.get_posterior().median()
    #     console.print('Medians: ')
    #     console.print(medians.sel(parameter='k1_MALE'))
    #     exit()
    #     for par in medians['parameter']:
    #         console.print(par.to_numpy())
    #         console.print(medians.sel(parameter=par).data_vars.variables)


def xps_selector(
    results_path: Path, xp_type: str, conditions: Optional[Dict[str, list]]
) -> Union[list[str], pd.DataFrame]:
    """Select the xps that match the desired conditions."""
    df = pd.read_csv(results_path / "xps" / xp_type / "results.tsv", sep="\t")

    if not conditions:  # empty dict -> no filtering
        return df

    if "timepoints" in conditions:
        conditions["timepoints"] = [t - 1 for t in conditions["timepoints"]]

    combinations = list(product(*(conditions[col] for col in conditions)))

    matching_indices = set()

    for comb in combinations:
        comb_dict = dict(zip(conditions.keys(), comb))
        mask = pd.Series(True, index=df.index)
        for col, val in comb_dict.items():
            mask &= df[col].eq(val)
        matching_indices.update(df[mask].index)

    df_res = df.loc[list(matching_indices)].sort_index()

    if df_res.empty:
        raise console.print(
            "No XPs were selected. Check if conditions are correct", style="warning"
        )

    return df_res["id"].unique().tolist()


def optimize_petab_xp(yaml_file: Path) -> list[dict]:
    """Optimize single petab problem using PyPesto."""

    # FIXME: add settings dictionary to this function
    # n_samples

    pypesto_sampler = PyPestoSampler(yaml_file=yaml_file)
    pypesto_sampler.load_problem()
    pypesto_sampler.optimizer()
    pypesto_sampler.bayesian_sampler(n_samples=1000)
    pypesto_sampler.results_hdi()
    # pypesto_sampler.results_median()

    results = []
    results_petab = pypesto_sampler.results_dict()
    for pid, stats in results_petab.items():
        results.append(
            {
                "id": yaml_file.parent.name,
                "group": get_group_from_pid(pid),
                "parameter": get_parameter_from_pid(pid),
                "pid": pid,
                **stats,
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(yaml_file.parent / "optimization_results.tsv", sep="\t")

    return results


def optimize_petab_xps(results_path: Path, xp_type: str, xp_ids: list[str]):
    """Optimize the given PEtab problems."""

    xp_path = results_path / "xps" / xp_type
    yaml_files: list[Path] = []
    for xp in xp_path.iterdir():
        if xp.is_dir() and xp.name in xp_ids:
            for yaml_file in xp.glob("**/petab.yaml"):
                yaml_files.append(yaml_file)

    yaml_files = sorted(yaml_files)

    infos = []
    for yaml_file in yaml_files:
        console.rule(yaml_file.name, style="white", align="left")
        results: list[dict] = optimize_petab_xp(yaml_file)
        infos.extend(results)

    df = pd.DataFrame(infos)
    df.to_csv(
        results_path / "xps" / xp_type / "bayes_results.tsv", sep="\t", index=False
    )
    console.print(df)
    return df


def run_optimizations(optimizations: dict[str, dict], results_path: Path) -> None:
    for xp_type, conditions in optimizations.items():
        console.rule(f"Selection for {xp_type}", align="center")

        xp_ids = xps_selector(
            results_path=results_path,
            xp_type=xp_type,
            conditions=conditions,
        )

        console.print(xp_ids)

        optimize_petab_xps(
            results_path=results_path,
            xp_type=xp_type,
            xp_ids=xp_ids,
        )

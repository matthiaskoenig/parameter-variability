"""Optimization using petab and pypesto."""
import arviz
import petab
from pathlib import Path
import numpy as np
from petab.v1 import Problem
from pypesto import petab as pt
import pypesto
from parameter_variability.console import console
from matplotlib import pyplot as plt
from copy import deepcopy
import logging
from dataclasses import dataclass
import pypesto.visualize.model_fit as model_fit
from typing import Callable
import arviz as az
import xarray as xr

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

        self.fig_path = self.yaml_file.parents[0] / "figs"
        self.fig_path.mkdir(parents=True, exist_ok=True)

    def optimizer(self, plot: bool = True):
        optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}

        optimizer = pypesto.optimize.FidesOptimizer(
            options=optimizer_options, verbose=logging.WARN
        )
        startpoint_method = pypesto.startpoint.uniform
        # save optimizer trace
        # history_options = pypesto.HistoryOptions(trace_record=True)
        opt_options = pypesto.optimize.OptimizeOptions()
        console.print(opt_options)

        n_starts = 100  # usually a value >= 100 should be used
        engine = pypesto.engine.MultiProcessEngine()

        # Set seed for reproducibility
        np.random.seed(1)

        console.rule("Optimization", style="white")
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

            ax = model_fit.visualize_optimized_model_fit(
                petab_problem=self.petab_problem,
                result=self.result,
                pypesto_problem=self.pypesto_problem
            )
            # plt.show()
            pypesto.visualize.waterfall(self.result)
            plt.savefig(str(self.fig_path) + '/01_waterfall.png')

            pypesto.visualize.parameters(self.result)
            plt.savefig(str(self.fig_path) + '/02_parameters.png')

            # pypesto.visualize.parameters_correlation_matrix(result)
            console.rule("Parameter_hist", style="white")
            # pypesto.visualize.parameter_hist(result=result, parameter_name="kabs")
            # plt.savefig(str(fig_path) + '/03_parameters_hist.png')

            pypesto.visualize.optimization_scatter(self.result)
            plt.savefig(str(self.fig_path) + '/04_opt_scatter.png')




    def bayesian_sampler(self,
                         sampler: Callable = pypesto.sample.AdaptiveMetropolisSampler(),
                         n_samples: int = 10_000):
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
        plt.savefig(str(self.fig_path) + '/06_sampling_fval_traces.png')

        pypesto.visualize.sampling_parameter_traces(
            self.result, use_problem_bounds=False, size=(12, 5)
        )
        plt.savefig(str(self.fig_path) + '/07_traces.png')

        pypesto.visualize.sampling_parameter_cis(self.result, alpha=[99, 95, 90],
                                                 size=(10, 5))
        plt.savefig(str(self.fig_path) + '/08_cis.png')

        pypesto.visualize.sampling_1d_marginals(self.result)
        plt.savefig(str(self.fig_path) + '/09_marginals.png')
        # plt.show()

    def get_posterior(self) -> az.InferenceData:
        """High density interval (HDI).

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
        trace.posterior = trace.posterior.rename({'x_dim_0': 'parameter'})
        trace.posterior['parameter'] = self.petab_problem.parameter_df.parameterName.values
        return trace

    def results_hdi(self):
        hdi = az.hdi(self.get_posterior())
        console.print('HDI for each parameter: ')
        for par in hdi['parameter']:
            console.print(par.to_numpy())
            console.print(hdi.sel(parameter=par).data_vars.variables)

    # def results_median(self):
    #     medians = self.get_posterior().median()
    #     console.print('Medians: ')
    #     console.print(medians.sel(parameter='k1_MALE'))
    #     exit()
    #     for par in medians['parameter']:
    #         console.print(par.to_numpy())
    #         console.print(medians.sel(parameter=par).data_vars.variables)


# if __name__ == '__main__':
#     pypesto_sampler = PyPestoSampler(
#         yaml_file=Path(__file__).parent / "petab.yaml",
#         fig_path=Path(__file__).parents[5] / "results" / "simple_chain"
#     )
#
#     pypesto_sampler.load_problem()
#
#     pypesto_sampler.optimizer()
#
#     pypesto_sampler.bayesian_sampler(n_samples=1000)
#
#     pypesto_sampler.results_hdi()
#
#     console.print('end')

"""Optimization using petab and pypesto."""
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
from arviz import hdi

console.rule("Load PEtab", style="white")
petab_yaml: Path = Path(__file__).parent / "petab.yaml"
petab_problem: Problem = petab.Problem.from_yaml(petab_yaml)
importer = pt.PetabImporter(petab_problem)


# TODO: Create an AggregatedObjective and added to the problem
#   Prior creation in create_problem https://github.com/ICB-DCM/pyPESTO/blob/main/pypesto/petab/importer.py and
#   Prior definition https://pypesto.readthedocs.io/en/latest/example/prior_definition.html

problem = importer.create_problem(verbose=True)

# check tbe observables df
console.rule("observables", style="white")
console.print(petab_problem.observable_df)

# Check the measurement dataframe
console.rule("measurements", style="white")
console.print(petab_problem.measurement_df)

# check the condition dataframe
console.rule("conditions", style="white")
console.print(petab_problem.condition_df)

# change things in the model
console.rule(style="white")
# console.print(problem.objective.amici_model.requireSensitivitiesForAllParameters())

# print(
#     f"Absolute tolerance before change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
# )
# problem.objective.amici_solver.setAbsoluteTolerance(1e-15)
# print(
#     f"Absolute tolerance after change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
# )

optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}

optimizer = pypesto.optimize.FidesOptimizer(
    options=optimizer_options, verbose=logging.WARN
)
startpoint_method = pypesto.startpoint.uniform
# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)
opt_options = pypesto.optimize.OptimizeOptions()
console.print(opt_options)

n_starts = 100  # usually a value >= 100 should be used
engine = pypesto.engine.MultiProcessEngine()

# Set seed for reproducibility
np.random.seed(1)

console.rule("Optimization", style="white")
result = pypesto.optimize.minimize(
    problem=problem,
    optimizer=optimizer,
    n_starts=n_starts,
    startpoint_method=startpoint_method,
    engine=engine,
    options=opt_options,
)

def print_optimization_results(result):
    """Print output of the optimization results."""

    console.rule("results", style="white")
    console.print(result.summary())

    # match to parameters
    console.print("Parameters:")
    console.print(petab_problem.parameter_df)
    parameter_names = petab_problem.parameter_df.parameterName

    # best fit
    console.print("Best fit:")
    best_fit: dict = result.optimize_result.list[0]
    console.print(best_fit)
    popts = deepcopy(best_fit["x"])
    for k, scale in enumerate(petab_problem.parameter_df.parameterScale):
        # backtransformations
        if scale == "lin":
            continue
        elif scale == "log10":
            popts[k] = 10 ** popts[k]
        elif scale == "log":
            popts[k] = np.exp(popts[k])

    console.print(dict(zip(parameter_names, popts)))

print_optimization_results(result)


fig_path: Path = Path(__file__).parents[5] / "results" / "simple_chain"

# see https://pypesto.readthedocs.io/en/latest/example/amici.html#2.-Optimization

ax = model_fit.visualize_optimized_model_fit(
    petab_problem=petab_problem, result=result, pypesto_problem=problem
)
plt.show()
pypesto.visualize.waterfall(result)
plt.savefig(str(fig_path) + '/01_waterfall.png')

pypesto.visualize.parameters(result)
plt.savefig(str(fig_path) + '/02_parameters.png')

# pypesto.visualize.parameters_correlation_matrix(result)
console.rule("Parameter_hist", style="white")
# pypesto.visualize.parameter_hist(result=result, parameter_name="kabs")
# plt.savefig(str(fig_path) + '/03_parameters_hist.png')

pypesto.visualize.optimization_scatter(result)
plt.savefig(str(fig_path) + '/04_opt_scatter.png')

# console.rule("Profile Likelihood", style="white")
# result = pypesto.profile.parameter_profile(
#     problem=problem,
#     result=result,
#     optimizer=optimizer,
#     engine=engine,
#     # profile_index=[0, 1],
# )
#
# pypesto.visualize.profiles(result)
# plt.savefig(str(fig_path) + '/05_profiles.png')



console.rule("Bayesian Sampler", style="white")
sampler = pypesto.sample.AdaptiveMetropolisSampler()
result = pypesto.sample.sample(
    problem=problem,
    sampler=sampler,
    n_samples=10_000,
    # n_samples=1_000,
    result=result,
)

pypesto.sample.effective_sample_size(result=result)
ess = result.sample_result.effective_sample_size
print(
    f"Effective sample size per computation time: {round(ess/result.sample_result.time,2)}"
)

pypesto.visualize.sampling_fval_traces(result)
plt.tight_layout()
plt.savefig(str(fig_path) + '/06_sampling_fval_traces.png')

pypesto.visualize.sampling_parameter_traces(
    result, use_problem_bounds=False, size=(12, 5)
)
plt.savefig(str(fig_path) + '/07_traces.png')

pypesto.visualize.sampling_parameter_cis(result, alpha=[99, 95, 90], size=(10, 5))
plt.savefig(str(fig_path) + '/08_cis.png')

pypesto.visualize.sampling_1d_marginals(result)
plt.savefig(str(fig_path) + '/09_marginals.png')
plt.show()

trace = result.sample_result.trace_x





# for i in range(result.sample_result.trace_x.shape[2]):
#     trace = result.sample_result.trace_x[:, :, i]
#     sigma_ln = np.var(trace)
#
#     transformed_trace = np.exp(trace + sigma_ln/2)
#     result.sample_result.trace_x[:, :, i] = transformed_trace
#
# pypesto.visualize.sampling_parameter_cis(result, alpha=[99, 95, 90], size=(10, 5))
# plt.savefig(str(fig_path) + '/08_cis_t.png')
#
# pypesto.visualize.sampling_1d_marginals(result)
# plt.savefig(str(fig_path) + '/09_marginals_t.png')
# plt.show()

# Get results and transform them

# @dataclass
# class PyPestoSampler:
#     """Create Petab Problem based on yaml"""
#     yaml_file: Path
#     fig_path: Path
#     petab_problem: petab.Problem = None
#     pypesto_problem: pypesto.Problem = None
#     result: pypesto.Result = None
#
#     def load_problem(self):
#         self.petab_problem: Problem = petab.Problem.from_yaml(self.yaml_file)
#         importer = pt.PetabImporter(self.petab_problem)
#         self.pypesto_problem = importer.create_problem(verbose=True)
#
#     def optimizer(self, plot: bool = True):
#         optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}
#
#         optimizer = pypesto.optimize.FidesOptimizer(
#             options=optimizer_options, verbose=logging.WARN
#         )
#         startpoint_method = pypesto.startpoint.uniform
#         # save optimizer trace
#         # history_options = pypesto.HistoryOptions(trace_record=True)
#         opt_options = pypesto.optimize.OptimizeOptions()
#         console.print(opt_options)
#
#         n_starts = 100  # usually a value >= 100 should be used
#         engine = pypesto.engine.MultiProcessEngine()
#
#         # Set seed for reproducibility
#         np.random.seed(1)
#
#         console.rule("Optimization", style="white")
#         self.result = pypesto.optimize.minimize(
#             problem=self.pypesto_problem,
#             optimizer=optimizer,
#             n_starts=n_starts,
#             startpoint_method=startpoint_method,
#             engine=engine,
#             options=opt_options,
#         )
#
#         def print_optimization_results(result: pypesto.Result):
#             """Print output of the optimization results."""
#
#             console.rule("results", style="white")
#             console.print(result.summary())
#
#             # match to parameters
#             console.print("Parameters:")
#             console.print(petab_problem.parameter_df)
#             parameter_names = petab_problem.parameter_df.parameterName
#
#             # best fit
#             console.print("Best fit:")
#             best_fit: dict = result.optimize_result.list[0]
#             console.print(best_fit)
#             popts = deepcopy(best_fit["x"])
#             for k, scale in enumerate(petab_problem.parameter_df.parameterScale):
#                 # backtransformations
#                 if scale == "lin":
#                     continue
#                 elif scale == "log10":
#                     popts[k] = 10 ** popts[k]
#                 elif scale == "log":
#                     popts[k] = np.exp(popts[k])
#
#             console.print(dict(zip(parameter_names, popts)))
#
#         print_optimization_results(self.result)
#
#         if plot:
#
#             ax = model_fit.visualize_optimized_model_fit(
#                 petab_problem=self.petab_problem,
#                 result=self.result,
#                 pypesto_problem=self.pypesto_problem
#             )
#             plt.show()
#             pypesto.visualize.waterfall(result)
#             plt.savefig(str(self.fig_path) + '/01_waterfall.png')
#
#             pypesto.visualize.parameters(result)
#             plt.savefig(str(self.fig_path) + '/02_parameters.png')
#
#             # pypesto.visualize.parameters_correlation_matrix(result)
#             console.rule("Parameter_hist", style="white")
#             # pypesto.visualize.parameter_hist(result=result, parameter_name="kabs")
#             # plt.savefig(str(fig_path) + '/03_parameters_hist.png')
#
#             pypesto.visualize.optimization_scatter(result)
#             plt.savefig(str(self.fig_path) + '/04_opt_scatter.png')
#
#
#
#
#     def bayesian_sampler(self,
#                          sampler: Callable = pypesto.sample.AdaptiveMetropolisSampler()):
#         self.result = pypesto.sample.sample(
#             problem=self.pypesto_problem,
#             sampler=sampler,
#             n_samples=10_000,
#             result=self.result,
#         )
#
#         pypesto.sample.effective_sample_size(result=self.result)
#         ess = result.sample_result.effective_sample_size
#         print(
#             f"Effective sample size per computation time: {round(ess / result.sample_result.time, 2)}"
#         )
#
#         pypesto.visualize.sampling_fval_traces(self.result)
#         plt.tight_layout()
#         plt.savefig(str(self.fig_path) + '/06_sampling_fval_traces.png')
#
#         pypesto.visualize.sampling_parameter_traces(
#             result, use_problem_bounds=False, size=(12, 5)
#         )
#         plt.savefig(str(self.fig_path) + '/07_traces.png')
#
#         pypesto.visualize.sampling_parameter_cis(self.result, alpha=[99, 95, 90],
#                                                  size=(10, 5))
#         plt.savefig(str(self.fig_path) + '/08_cis.png')
#
#         pypesto.visualize.sampling_1d_marginals(self.result)
#         plt.savefig(str(self.fig_path) + '/09_marginals.png')
#         plt.show()
#
#
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
#     pypesto_sampler.bayesian_sampler()



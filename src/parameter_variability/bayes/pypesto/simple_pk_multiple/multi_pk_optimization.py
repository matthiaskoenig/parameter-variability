"""Optimization using petab and pypesto."""
import petab
from pathlib import Path
import numpy as np
import pypesto.petab
from parameter_variability.console import console
from matplotlib import pyplot as plt
import logging
import pypesto.visualize.model_fit as model_fit

# 1. Create problem --------------------------
console.rule("1. Create Problem", style="white")
petab_yaml: Path = Path(__file__).parent / "multi_pk.yaml"
petab_problem = petab.Problem.from_yaml(petab_yaml)
importer = pypesto.petab.PetabImporter(petab_problem)
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
console.print(problem.objective.amici_model.requireSensitivitiesForAllParameters())

print(
    f"Absolute tolerance before change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
)
problem.objective.amici_solver.setAbsoluteTolerance(1e-15)
print(
    f"Absolute tolerance after change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
)


# 2. Choose optimizer ----------------------------------
console.rule('2. Choose optimizer', style='white')
optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}
optimizer = pypesto.optimize.FidesOptimizer(
    options=optimizer_options, verbose=logging.WARN
)
startpoint_method = pypesto.startpoint.uniform
# # save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)
opt_options = pypesto.optimize.OptimizeOptions()
console.print(opt_options)
#
n_starts = 100  # usually a value >= 100 should be used
engine = pypesto.engine.MultiProcessEngine()
#
# # Set seed for reproducibility
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
console.rule("results", style="white")
console.print(result.summary())
fig_path: Path = Path(__file__).parent / "results"
#
# see https://pypesto.readthedocs.io/en/latest/example/amici.html#2.-Optimization


# 3. Evaluate model fit ------------------------------------
console.rule('3. Evaluate Model Fit', style='white')
# ax = model_fit.visualize_optimized_model_fit(
#     petab_problem=petab_problem, result=result, pypesto_problem=problem
# )
# plt.show()
# pypesto.visualize.waterfall(result)
# plt.savefig(str(fig_path) + '/01_model-fit_waterfall.png')
#
# pypesto.visualize.parameters(result)
# plt.savefig(str(fig_path) + '/02_model-fit_parameters.png')

# pypesto.visualize.parameters_correlation_matrix(result)
# console.rule("Parameter_hist", style="white")
# pypesto.visualize.parameter_hist(result=result, parameter_name="kabs")
# plt.savefig(str(fig_path) + '/03_parameters_hist.png')

# pypesto.visualize.optimization_scatter(result)
# plt.savefig(str(fig_path) + '/04_model-fit_opt_scatter.png')


# 4. Determine Parameter Uncertainty ----------------------------
console.rule("4. Determine Parameter Uncertainty", style="white")

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


sampler = pypesto.sample.AdaptiveMetropolisSampler()
result = pypesto.sample.sample(
    problem=problem,
    sampler=sampler,
    n_samples=5000,
    result=result,
)

pypesto.sample.effective_sample_size(result=result)
ess = result.sample_result.effective_sample_size
print(
    f"Effective sample size per computation time: {round(ess/result.sample_result.time,2)}"
)

pypesto.visualize.sampling_fval_traces(result)
plt.tight_layout()
plt.savefig(str(fig_path) + '/06_param-uncert_sampling_fval_traces.png')

pypesto.visualize.sampling_parameter_traces(
    result, use_problem_bounds=False, size=(12, 5), stepsize=10
)
plt.savefig(str(fig_path) + '/07_param-uncert_traces.png')

pypesto.visualize.sampling_scatter(result, stepsize=10, show_bounds=False)
plt.savefig(str(fig_path) + '/08_param-scatter.png')

pypesto.visualize.sampling_parameter_cis(result, alpha=[99, 95, 90], size=(10, 5))
plt.savefig(str(fig_path) + '/09_param-uncert_cis.png')

pypesto.visualize.sampling_1d_marginals(result, stepsize=10, plot_type='both')
plt.savefig(str(fig_path) + '/10_param-uncert_marginals.png')
plt.show()


# 5. Prediction Uncertainty ------------------------------------------
console.rule('5. Prediction Uncertainty', style='white')


# 6. Model Selection
# https://github.com/PEtab-dev/petab_select


# from pypesto.petab.importer import PetabImporter
# from petab import Problem
# from pypesto import optimize, sample, HistoryOptions, engine, startpoint
#
# class PyPestoWorkflow:
#
#     def __init__(self, petab_yaml: Path):
#         self.petab_yaml = petab_yaml
#
#         self.importer = None
#         self.result = None
#
#     def create_problem(
#         self,
#         verbose: bool = True,
#         check: bool = False
#     ) -> pypesto.problem.base.Problem:
#         petab_problem = Problem.from_yaml(self.petab_yaml)
#         self.importer: PetabImporter = PetabImporter(petab_problem)
#         problem = importer.create_problem(verbose=verbose)
#
#         if check:
#             # check tbe observables df
#             console.rule("observables", style="white")
#             console.print(petab_problem.observable_df)
#
#             # Check the measurement dataframe
#             console.rule("measurements", style="white")
#             console.print(petab_problem.measurement_df)
#
#             # check the condition dataframe
#             console.rule("conditions", style="white")
#             console.print(petab_problem.condition_df)
#
#         return problem
#
#     def optimize(self, problem: pypesto.problem.base.Problem):
#         objective = self.importer.create_objective()
#
#         objective.amici_solver.setAbsoluteTolerance(1e-15)
#
#         console.print(objective.amici_model.requireSensitivitiesForAllParameters())
#
#         optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}
#         optimizer = optimize.FidesOptimizer(
#             options=optimizer_options, verbose=logging.WARN
#         )
#
#         history_options = HistoryOptions(trace_record=True)
#         opt_options = optimize.OptimizeOptions()
#
#         sampler = sample.AdaptiveMetropolisSampler()
#
#         n_starts = 100  # usually a value >= 100 should be used
#         engine = engine.MultiProcessEngine()
#         startpoint_method = startpoint.uniform
#
#         self.result = optimize.minimize(
#             problem=problem,
#             optimizer=optimizer,
#             n_starts=n_starts,
#             startpoint_method=startpoint_method,
#             engine=engine,
#             options=opt_options,
#         )
#
#         self.result = sample.sample(
#             problem=problem,
#             sampler=sampler,
#             n_samples=10000,
#             result=self.result,
#         )
#
#         burn = sample.geweke_test(self.result)  # cutting burin-in samples
#         # at least first 100 samples are cut for burn in
#         if burn < 100:
#             self.result.sample_result.burn_in = 100




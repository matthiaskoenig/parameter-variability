""""
# not working with MKL
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_def.so:/usr/lib/x86_64-linux-gnu/libmkl_avx2.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libiomp5.so
"""
# see https://github.com/AMICI-dev/AMICI/issues/2176
# import os
# os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libmkl_def.so:/usr/lib/x86_64-linux-gnu/libmkl_avx2.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libiomp5.so"

from matplotlib import pyplot as plt
import importlib
import os
import sys

import amici
import amici.plotting
import numpy as np

import pypesto
import pypesto.optimize as optimize
import pypesto.visualize as visualize
from amici import AmiciModel

#
from parameter_variability import MODEL_SIMPLE_PK
from parameter_variability.console import console

console.rule("Compile the model", style="white")
# sbml file we want to import
sbml_file = str(MODEL_SIMPLE_PK)
# name of the model that will also be the name of the python module
model_name = "model_simple_pk"
# directory to which the generated model code is written
model_output_dir = "tmp/" + model_name
#
# import sbml model, compile and generate amici module
# sbml_importer = amici.SbmlImporter(sbml_file)
# sbml_importer.sbml2amici(model_name, model_output_dir, verbose=False)


console.rule("Load the model", style="white")
# load amici module (the usual starting point later for the analysis)
sys.path.insert(0, os.path.abspath(model_output_dir))
model_module = importlib.import_module(model_name)
model: AmiciModel = model_module.getModel()
parameters = model.getParameters()
parameter_ids = model.getParameterIds()
console.print(dict(zip(parameter_ids, parameters)))


model.requireSensitivitiesForAllParameters()
model.setTimepoints(np.linspace(0, 10, 11))  # [min]
model.setParameterScale(amici.ParameterScaling.log10)

# log10 paramaters; log10(p) => 10^(-0.3) = p
# {'amici_k': 1.0, 'CL': 1.0, 'Q': 1.0}
model.setParameters([0.0, 0.0, 0.0])

solver = model.getSolver()
solver.setSensitivityMethod(amici.SensitivityMethod.forward)
solver.setSensitivityOrder(amici.SensitivityOrder.first)

# how to run amici now:
rdata = amici.runAmiciSimulation(model, solver, None)
amici.plotting.plotStateTrajectories(rdata)

# create artifical data
edata = amici.ExpData(rdata, 0.2, 0.0)

console.rule("Optimize", style="white")
# create objective function from amici model
# pesto.AmiciObjective is derived from pesto.Objective,
# the general pesto objective function class
objective = pypesto.AmiciObjective(model, solver, [edata], 1)

# create optimizer object which contains all information for doing the optimization
optimizer = optimize.ScipyOptimizer(method="ls_trf")

# create problem object containing all information on the problem to be solved
problem = pypesto.Problem(objective=objective, lb=[-2, -2, -2], ub=[2, 2, 2])

# do the optimization
result = optimize.minimize(
    problem=problem, optimizer=optimizer, n_starts=100, filename=None
)

console.rule("Visualization")
visualize.waterfall(result)
visualize.parameters(result)
visualize.optimization_scatter(result=result)
plt.show()

console.rule("Profiles")
import pypesto.profile as profile

profile_options = profile.ProfileOptions(
    min_step_size=0.0005,
    delta_ratio_max=0.05,
    default_step_size=0.005,
    ratio_min=0.01,
)

result = profile.parameter_profile(
    problem=problem,
    result=result,
    optimizer=optimizer,
    profile_index=np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]),
    result_index=0,
    profile_options=profile_options,
    filename=None,
)
plt.show()




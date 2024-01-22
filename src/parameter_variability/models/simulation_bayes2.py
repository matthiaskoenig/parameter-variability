import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
import roadrunner
from scipy.stats import lognorm

import pymc as pm
import pytensor
import pytensor.tensor as pt

from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op

rng = np.random.default_rng(1234)

with open('data/twoCpt.data.json', 'r') as f:
    two_compartment_model = json.load(f)

df = pd.DataFrame({'time': two_compartment_model['time'][1:],
                   'c_obs': two_compartment_model['cObs']})


def init_par():
    return {'CL': lognorm.rvs(scale=np.log(10), s=0.25, size=1)[0],
            'Q': lognorm.rvs(scale=np.log(15), s=0.5, size=1)[0],
            'v_c': lognorm.rvs(scale=np.log(35), s=0.25, size=1)[0],
            'v_p': lognorm.rvs(scale=np.log(105), s=0.5, size=1)[0],
            'k': lognorm.rvs(scale=np.log(2.5), s=1, size=1)[0]}


fig, ax = plt.subplots(1,1)


def plot_data(ax, title='Gut Plasma Concentration'):
    ax.plot(df['time'], df['c_obs'], '-.')
    ax.set_xlabel('time [min]')
    ax.set_ylabel('concentration')
    ax.set_title(title)
    ax.grid(True)

    return ax


r: roadrunner.RoadRunner = roadrunner.RoadRunner("model2.xml")
r.resetAll()


# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    runner: roadrunner.RoadRunner = roadrunner.RoadRunner("model2.xml")
    runner.resetAll()
    for par_name, value in zip(['k', 'CL', 'Q', 'Vcent', 'Vperi'], theta):
        runner.setValue(par_name, value)
    sim = runner.simulate(start=0, end=np.max(df['time']), steps=df.shape[0])
    sim_df = pd.DataFrame(sim, columns=sim.colnames)
    return sim_df['[y_gut]']


with pm.Model() as model:
    k = pm.LogNormal('k', mu=np.log(2.5), sigma=1, initval=sim_soln[0])
    cl = pm.LogNormal("CL", mu=np.log(10), sigma=0.25, initval=sim_soln[1])
    q = pm.LogNormal('Q', mu=np.log(15), sigma=0.5, initval=sim_soln[2])
    vc = pm.LogNormal('Vcent', mu=np.log(35), sigma=0.5, initval=sim_sol[3])
    vp = pm.LogNormal('Vperi', mu=np.log(2.5), sigma=1, initval=sim_soln[4])

    PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
    sigma = PositiveNormal('sigma', mu=0, sigma=1)

    # ODE solution function
    ode_soln = pytensor_forward_model_matrix(
        pm.math.stack([k, cl, q, vc, vp])
    )

    # likelihood
    pm.LogNormal('c_obs', mu=pm.math.log(ode_soln/vc), sigma=sigma, observed=df['c_obs'])


def plot_model(ax, y, time, alpha, lw, title):
    ax.plot(time, y, color="b", alpha=alpha, lw=lw)
    ax.set_title(title, fontsize=16)
    return ax

#%%
def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ["alpha", "beta", "gamma", "delta", "xto", "yto"]
    row = trace_df.iloc[row_idx, :][cols].values

    # alpha, beta, gamma, delta, Xt0, Yt0
    time = np.arange(1900, 1921, 0.01)
    theta = row
    x_y = odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,))
    plot_model(ax, x_y, time=time, lw=lw, alpha=alpha);

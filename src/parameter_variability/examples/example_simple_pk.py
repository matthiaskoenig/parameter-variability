"""Example simulations and parameter scans with simple PK model."""

import numpy as np
import pandas as pd
import roadrunner
from matplotlib import pyplot as plt
from sbmlutils.console import console


# ------------------------------------------------------------------------------
# plt.style.use('science')
import matplotlib

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 25

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# ------------------------------------------------------------------------------

def reference_simulation(r: roadrunner.RoadRunner) -> None:
    """Reference simulation."""

    console.rule("Reference simulation", style="white")
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=300)

    # simulation
    r.resetAll()
    s = r.simulate(start=0, end=10, steps=400)  # [min]
    df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)

    f.suptitle("Reference simulation")
    for sid in ["[y_gut]", "[y_cent]", "[y_peri]"]:
        ax.plot(df.time, df[sid], label=sid)

        # ax.legend()

        ax.set_xlabel("time [min]", fontweight="bold")
        ax.set_ylabel("concentration [mM]", fontweight="bold")

    ax.legend()
    # ax.grid(True)
    plt.tight_layout()
    plt.show()


def parameter_scan(r: roadrunner.RoadRunner) -> None:
    """Scanning model parameters."""
    console.rule("Parameter scan", style="white")

    ks = np.linspace(0, 10, num=11)
    cls = np.linspace(0, 10, num=11)
    qs = np.linspace(0, 10, num=11)

    results = {}

    for parameter, par_name in zip([ks, cls, qs], ["kabs", "CL", "Q"]):
        results_par = []
        for value in parameter:
            # reset to a clean state
            r.resetAll()
            r.setValue(par_name, value)
            s = r.simulate(start=0, end=10, steps=400)
            # pretty slow (memory copy)
            df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
            # console.print(df)
            results_par.append(df)

        results[par_name] = results_par

    f, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 13), dpi=300)
    for parameter, ax in zip(results, axes):
        for sid in ["[y_cent]", "[y_gut]", "[y_peri]"]:
            for df in results[parameter]:
                ax.plot(
                    df.time,
                    df[sid],
                    label=sid,
                    linestyle="-",
                    # marker="o",
                    markeredgecolor="black",
                )
        # ax.legend()
        ax.set_title(parameter)
        ax.set_xlabel("time [min]")
        ax.set_ylabel("concentration [mM]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load_model
    from parameter_variability import MODEL_SIMPLE_PK

    r: roadrunner.RoadRunner = roadrunner.RoadRunner(MODEL_SIMPLE_PK)
    reference_simulation(r)
    parameter_scan(r)

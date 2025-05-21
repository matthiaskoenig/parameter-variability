"""Example simulations and parameter scans with simple PK model."""

import numpy as np
import pandas as pd
import roadrunner
from matplotlib import pyplot as plt
from sbmlutils.console import console
from parameter_variability import RESULTS_SIMPLE_PK


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
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=300, layout="constrained")

    # simulation
    r.resetAll()

    # set dose
    r.setValue("IVDOSE_icg", 10)  # [mg]

    s = r.simulate(start=0, end=30, steps=200)  # [min]
    df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)

    f.suptitle("Reference simulation")
    for sid in ["[Cve_icg]"]:
        ax.plot(df.time, df[sid], label=sid)

        # ax.legend()

        ax.set_xlabel("time [min]", fontweight="bold")
        ax.set_ylabel("concentration [mM]", fontweight="bold")

    ax.legend()
    # ax.grid(True)
    plt.tight_layout()
    plt.show()
    f.savefig(RESULTS_SIMPLE_PK / "simple_pk_simulation.png", bbox_inches="tight")


def parameter_scan(r: roadrunner.RoadRunner) -> None:
    """Scanning model parameters."""
    console.rule("Parameter scan", style="white")

    bw = np.linspace(50, 150, num=11)
    vmax = np.linspace(0.01, 0.5, num=11)


    results = {}

    for parameter, par_name in zip([bw, vmax], ["BW", "LI__ICGIM_Vmax"]):
        results_par = []
        for value in parameter:
            # reset to a clean state
            r.resetAll()
            r.setValue("IVDOSE_icg", 10)  # [mg]
            r.setValue(par_name, value)
            s = r.simulate(start=0, end=10, steps=400)
            # pretty slow (memory copy)
            df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
            # console.print(df)
            results_par.append(df)

        results[par_name] = results_par

    f, axes = plt.subplots(nrows=len(results), ncols=1, figsize=(7, 7*len(results)), dpi=300)
    for parameter, ax in zip(results, axes):
        for sid in ["[Cve_icg]"]:  #  venous plasma concentration ICG
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
    f.savefig(RESULTS_SIMPLE_PK / "icg_scan.png", bbox_inches="tight")


if __name__ == "__main__":
    # load_model
    from parameter_variability import MODEL_ICG

    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(MODEL_ICG))
    reference_simulation(r)
    parameter_scan(r)

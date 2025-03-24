"""Running test simulation with example model."""
from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner
from matplotlib import pyplot as plt
from sbmlutils.console import console

import matplotlib

SMALL_SIZE = 13
MEDIUM_SIZE = 18
BIGGER_SIZE = 25

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from parameter_variability import MODEL_SIMPLE_CHAIN, RESULTS_SIMPLE_CHAIN


def example_simulation_chain() -> None:
    """Example simulation and visualization."""
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(MODEL_SIMPLE_CHAIN))
    ks = np.logspace(-1, 1, num=11)

    results = []
    for k1 in ks:
        # reset to a clean state
        r.resetAll()
        r.setValue("k1", k1)
        s = r.simulate(start=0, end=20, steps=400)
        # pretty slow (memory copy)
        df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
        # console.print(df)
        results.append(df)

    f, ax = plt.subplots(nrows=1, ncols=1, dpi=300, layout="constrained")
    for sid in ["[S1]", "[S2]"]:
        for kdf, df in enumerate(results):
            ax.plot(
                df.time,
                df[sid],
                label=sid if kdf == 0 else "__nolabel__",
                linestyle="-" if sid == "[S1]" else "--",
                # marker="o",
                # markeredgecolor="black",
                color="black" if sid == "[S2]" else "black",
            )
        # ax.legend()

    ax.legend()
    f.suptitle("Effect of parameter k1")
    ax.set_xlabel("Time")
    ax.set_ylabel("Plasma Concentration")
    plt.show()
    fig_path: Path = RESULTS_SIMPLE_CHAIN / "simple_chain_simulation.png"
    f.savefig(fig_path)
    console.print(f"file://{fig_path}")

if __name__ == "__main__":
    example_simulation_chain()

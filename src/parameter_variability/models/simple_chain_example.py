"""Running test simulation with example model."""

import numpy as np
import pandas as pd
import roadrunner
from matplotlib import pyplot as plt
from sbmlutils.console import console

from parameter_variability import MODEL_SIMPLE_CHAIN, RESULTS_SIMPLE_CHAIN


def example_simulation_chain() -> None:
    """Example simulation and visualization."""
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(MODEL_SIMPLE_CHAIN)
    ks = np.linspace(0, 10, num=11)

    results = []
    for k1 in ks:
        # reset to a clean state
        r.resetAll()
        r.setValue("k1", k1)
        s = r.simulate(start=0, end=10, steps=400)
        # pretty slow (memory copy)
        df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
        # console.print(df)
        results.append(df)

    f, ax = plt.subplots(nrows=1, ncols=1, dpi=300, layout="constrained")
    for sid in ["[S1]", "[S2]"]:
        for df in results:
            ax.plot(
                df.time,
                df[sid],
                label=sid,
                linestyle="-",
                # marker="o",
                markeredgecolor="black",
            )
        # ax.legend()

    ax.set_xlabel("time [A.U.]")
    ax.set_ylabel("concentration [substance_units/compartment_units]")
    plt.show()
    f.savefig(RESULTS_SIMPLE_CHAIN / "simple_chain_simulation.png")


if __name__ == "__main__":
    example_simulation_chain()

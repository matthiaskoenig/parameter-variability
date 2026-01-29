"""Running test simulation with example model."""

from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner

from parvar import MODEL_SIMPLE_CHAIN, RESULTS_SIMPLE_CHAIN
from parvar.plots import plt, DPI


def example_simple_chain() -> None:
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

    f, ax = plt.subplots(dpi=DPI, layout="constrained", figsize=(6, 6))
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
    f.suptitle("Parameter scan k1")
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_ylabel("Plasma Concentration", fontweight="bold")
    plt.show()
    results_path = RESULTS_SIMPLE_CHAIN / "example"
    results_path.mkdir(parents=True, exist_ok=True)
    fig_path: Path = results_path / "simple_chain_simulation.png"
    f.savefig(fig_path)


if __name__ == "__main__":
    example_simple_chain()

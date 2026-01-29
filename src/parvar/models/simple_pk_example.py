"""Example simulations and parameter scans with simple PK model."""

from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner
from pymetadata.console import console

from parvar.plots import plt


def reference_simulation(r: roadrunner.RoadRunner, results_path: Path) -> None:
    """Reference simulation."""

    console.rule("Reference simulation", style="white")
    f, ax = plt.subplots(figsize=(6, 6), dpi=300, layout="constrained")

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
    f.savefig(results_path / "simple_pk_simulation.png")


def parameter_scan(r: roadrunner.RoadRunner, results_path: Path) -> None:
    """Scanning model parameters."""
    console.rule("Parameter scan", style="white")

    ks = np.linspace(0, 10, num=11)
    cls = np.linspace(0, 10, num=11)
    qs = np.linspace(0, 10, num=11)

    results = {}

    for parameter, par_name in zip([ks, cls, qs], ["k", "CL", "Q"]):
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
    f.suptitle("Parameter scan")
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
        ax.set_xlabel("time [min]", fontweight="bold")
        ax.set_ylabel("concentration [mM]", fontweight="bold")
    plt.tight_layout()
    plt.show()
    f.savefig(results_path / "simple_pk_scan.png")


def example_simple_pk() -> None:
    """Run simple pk example."""
    from parvar import RESULTS_SIMPLE_PK, MODEL_SIMPLE_PK

    results_path = RESULTS_SIMPLE_PK / "example"
    results_path.mkdir(parents=True, exist_ok=True)
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(MODEL_SIMPLE_PK))
    reference_simulation(r, results_path=results_path)
    parameter_scan(r, results_path=results_path)


if __name__ == "__main__":
    example_simple_pk()

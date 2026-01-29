"""Example simulations and parameter scans with simple PK model."""

from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner
from matplotlib import pyplot as plt
from pymetadata.console import console

from parvar.plots import DPI


def reference_simulation(r: roadrunner.RoadRunner, results_path: Path) -> None:
    """Reference simulation."""

    console.rule("Reference simulation", style="white")
    f, ax = plt.subplots(figsize=(6, 6), dpi=DPI, layout="constrained")

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
    f.savefig(results_path / "icg_simulation.png")


def parameter_scan(r: roadrunner.RoadRunner, results_path: Path) -> None:
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

    f, axes = plt.subplots(
        nrows=len(results),
        ncols=1,
        figsize=(6, 6 * len(results)),
        dpi=DPI,
        layout="constrained",
    )
    f.suptitle("Parameter Scan")
    for parameter, ax in zip(results, axes):
        for sid in ["[Cve_icg]"]:  # venous plasma concentration ICG
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
    f.savefig(results_path / "icg_scan.png")


def example_icg() -> None:
    """Run ICG example."""
    from parvar import RESULTS_ICG, MODEL_ICG

    results_path = RESULTS_ICG / "example"
    results_path.mkdir(parents=True, exist_ok=True)
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(MODEL_ICG))
    reference_simulation(r, results_path=results_path)
    parameter_scan(r, results_path=results_path)


if __name__ == "__main__":
    example_icg()

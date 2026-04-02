from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.utils import append_server_result, join_optimization_results


def runtime_boxplot(
    df: pd.DataFrame,
    column: str = "samples",
    save_path: Path = None,
    ax: plt.Axes = None,
    show_plot: bool = False,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    groups = [grp["optim_duration"].values for _, grp in df.groupby(column)]
    labels = [key for key, _ in df.groupby(column)]

    ax.boxplot(
        groups,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1", color="#2471A3"),
        medianprops=dict(color="#E74C3C", linewidth=2),
        whiskerprops=dict(color="#2471A3"),
        capprops=dict(color="#2471A3"),
        flierprops=dict(marker="o", color="#AAA", markersize=4),
    )

    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12) if show_plot else None

    if save_path:
        plt.savefig(save_path / f"{column}_runtime_boxplot.png")

    if show_plot:
        plt.tight_layout()
        plt.show()


def runtime_boxplots(df: pd.DataFrame, save_path: Path = None) -> None:
    fig = plt.figure(figsize=(16, 5))

    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        wspace=0.35,
        top=0.95,
        bottom=0.1,  # margin for legend + xlabel
        left=0.07,
        right=0.97,
    )
    for i, c in enumerate(["prior_type", "samples", "timepoints", "noise_cv"]):
        ax = fig.add_subplot(gs[0, i])
        runtime_boxplot(df, column=c, ax=ax)

    fig.supylabel("Runtime (s)", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / "runtime_boxplot.png")

    plt.show()


if __name__ == "__main__":
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=results_path, xp_type="all")

        # runtime_boxplot(results, show_plot=True)
        runtime_boxplots(results)

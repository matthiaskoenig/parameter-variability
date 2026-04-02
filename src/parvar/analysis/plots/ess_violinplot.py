from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.utils import append_server_result, join_optimization_results
from parvar.plots import value_labels, axis_labels


def ess_violinplot(
    df: pd.DataFrame,
    column: str = "samples",
    save_path: Path = None,
    ax: plt.Axes = None,
    show_plot: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    groups = [grp["ess"].values for _, grp in df.groupby(column)]
    labels = [key for key, _ in df.groupby(column)]

    if column == "prior_type":
        labels = [value_labels[column][lab] for lab in labels]

    parts = ax.violinplot(
        groups, positions=range(len(labels)), showmedians=True, showextrema=True
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)

    for i, grp in enumerate(groups):
        ax.scatter([i] * len(grp), grp, color="navy", alpha=0.5, s=15, zorder=3)

    parts["cmedians"].set_color("navy")
    parts["cmins"].set_color("steelblue")
    parts["cmaxes"].set_color("steelblue")
    parts["cbars"].set_color("steelblue")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_xlabel(axis_labels[column], fontsize=10)
    ax.set_ylabel(
        "Effective Sample Size", fontsize=10
    ) if save_path or show_plot else None

    if save_path:
        plt.savefig(save_path / f"{column}_ess_violinplot.png")

    if show_plot:
        plt.tight_layout()
        plt.show()


def ess_violinplots(
    df: pd.DataFrame,
    save_path: Path = None,
) -> None:
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
        ess_violinplot(df, column=c, ax=ax)

    fig.supylabel("Effective Sample Size", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / "ess_violinplot.png")

    plt.show()


if __name__ == "__main__":
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=results_path, xp_type="all")

        ess_violinplot(results, column="prior_type", show_plot=True)
        # ess_violinplots(results)

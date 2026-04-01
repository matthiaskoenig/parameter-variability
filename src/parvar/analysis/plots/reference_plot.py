from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from pymetadata.console import console

from parvar import RESULTS_SIMPLE_PK, RESULTS_SIMPLE_CHAIN, RESULTS_ICG
from parvar.analysis.utils import append_server_result, join_optimization_results

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import lines as mlines
from parvar.plots import colors


def reference_plot(
    df: pd.DataFrame,
    reference: Dict[str, Any],
    save_path: Path = None,
) -> None:
    pars = df["parameter"].unique()
    groups = df["group"].unique()
    console.print(groups)
    # prior_types = df["prior_type"].unique()
    # noise_cv  = df['noise_cv'].unique()

    def errorbar_calc(df: pd.DataFrame) -> np.ndarray:
        median = df["bayes_sampler_median"]
        yerr_lower = median - df["hdi_low"]
        yerr_upper = df["hdi_high"] - median
        yerr = np.vstack([yerr_lower, yerr_upper])

        return yerr

    fig = plt.figure(
        dpi=360,
        figsize=(len(pars) * 4, 10),
    )

    gs = gridspec.GridSpec(
        4,
        len(pars),
        figure=fig,
        hspace=0.45,
        wspace=0.30,
        top=0.93,
        bottom=0.12,  # ← legend lives in this margin
        left=0.07,
        right=0.97,
    )

    pc_y = 0
    legend_handles = []

    for p in pars:
        # offset = 0
        df_p = df[df["parameter"] == p]

        if pc_y > 1:
            pc_y = 0

        # for g in groups:

        def plot_(ax: plt.Axes, df_: pd.DataFrame, col: str, handles: list[Any]):
            for g in groups:
                df_g = df_[df_["group"] == g]
                console.print(df_g)
                ax.plot(
                    df_g[col],
                    df_g["bayes_sampler_median"],
                    ".",
                    label=f"{g} (estimated)",
                    color=colors[g],
                )
                ax.errorbar(
                    df_g[col],
                    df_g["bayes_sampler_median"],
                    yerr=errorbar_calc(df_g),
                    color=colors[g],
                    linestyle="",
                    # linestyle="dashed"
                )
                console.print(df_g["sample_loc"].tolist())
                ax.axhline(
                    y=df_g["sample_loc"].tolist()[0],
                    label=f"{g} (real)",
                    color=colors[g],
                    linestyle="dashed",
                )

                handles.extend(
                    [
                        mlines.Line2D(
                            [],
                            [],
                            marker="o",
                            color=colors[g],
                            label=f"{g} (estimated)",
                        ),
                        mlines.Line2D(
                            [],
                            [],
                            color=colors[g],
                            linestyle="dashed",
                            label=f"{g} (real)",
                        ),
                    ]
                )

        # Prior plot

        df_prior = df_p[
            # (df_p['group'] == g) &
            (df_p["timepoints"] == reference["timepoints"])
            & (df_p["samples"] == reference["samples"])
            & (df_p["noise_cv"] == reference["noise_cv"])
        ]
        ax_prior = fig.add_subplot(gs[0, pc_y])
        plot_(ax_prior, df_prior, "prior_type", legend_handles)
        ax_prior.set_title(p)

        # Timepoint plot
        df_timepoint = df_p[
            # (df_p['group'] == g) &
            (df_p["prior_type"] == reference["prior_type"])
            & (df_p["samples"] == reference["samples"])
            & (df_p["noise_cv"] == reference["noise_cv"])
        ]
        ax_timepoint = fig.add_subplot(gs[1, pc_y])
        plot_(ax_timepoint, df_timepoint, "timepoints", legend_handles)

        # Samples plot
        df_samples = df_p[
            # (df_p['group'] == g) &
            (df_p["timepoints"] == reference["timepoints"])
            & (df_p["prior_type"] == reference["prior_type"])
            & (df_p["noise_cv"] == reference["noise_cv"])
        ]
        ax_samples = fig.add_subplot(gs[2, pc_y])
        plot_(ax_samples, df_samples, "samples", legend_handles)

        # Noise Plot
        df_noise = df_p[
            # (df_p['group'] == g) &
            (df_p["timepoints"] == reference["timepoints"])
            & (df_p["prior_type"] == reference["prior_type"])
            & (df_p["samples"] == reference["samples"])
        ]
        ax_noise = fig.add_subplot(gs[3, pc_y])
        plot_(ax_noise, df_noise, "noise_cv", legend_handles)

        pc_y += 1
    fig.supylabel("Parameter Value")

    console.print(fig.axes[3].get_position())

    # row_axes = [(ax1, ax2), (ax3, ax4), (ax5, ax6)]
    row_labels = ["Priors", "Timepoints", "Samples", "Coefficient Variation"]

    for ax, label in zip(fig.axes, row_labels):
        pos = ax.get_position()  # Bbox in figure fraction
        mid_x = (len(pars) / 2) * (pos.x0 + pos.x1)  # horizontal centre of the row
        label_y = pos.y0 - 0.03  # just below the row's bottom
        fig.text(
            mid_x,
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=12,
            color="#444444",
        )

    seen = set()
    unique_handles = []
    for h in legend_handles:
        label = h.get_label()
        if label not in seen:
            seen.add(label)
            unique_handles.append(h)
    fig.legend(
        handles=unique_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor="#FFFFFF",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path / "reference_plot.png")

    plt.show()


if __name__ == "__main__":
    reference = {
        "prior_type": "prior_biased_2",
        "timepoints": 9,
        "samples": 40,
        "noise_cv": 0.001,
    }

    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=results_path, xp_type="all")
        console.print(results.info())

        reference_plot(df=results, reference=reference)

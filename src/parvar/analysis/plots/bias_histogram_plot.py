from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from parvar import RESULTS_SIMPLE_PK, RESULTS_ICG, RESULTS_SIMPLE_CHAIN
from parvar.analysis.utils import append_server_result, join_optimization_results
from matplotlib import gridspec
from matplotlib import patches as mpatches
from parvar.plots import colors, parameter_labels, value_labels, axis_labels


def bias_histogram(
    df: pd.DataFrame,
    column: str = "prior_type",
    save_path: Path = None,
) -> None:
    pars = df["parameter"].unique()
    groups = df["group"].unique()
    vals = df[column].unique()

    # Only accept maximum of 4 values
    if len(vals) > 4:
        vals = vals[-4:]

    def point_bias(df: pd.DataFrame) -> pd.DataFrame:
        df["point_bias"] = (
            np.abs(df["sample_loc"] - df["bayes_sampler_median"]) / df["sample_loc"]
        )
        return df

    df = point_bias(df)

    fig = plt.figure(
        dpi=360,
        figsize=(len(pars) * 4.2, len(vals) * 1.5),
    )

    gs = gridspec.GridSpec(
        len(vals),
        len(pars),
        figure=fig,
        hspace=0.3,
        wspace=0.30,
        top=0.93,
        bottom=0.3,  # ← legend lives in this margin
        left=0.07,
        right=0.97,
    )

    pc_x = 0
    pc_y = 0
    legend_handles = []

    for p in pars:
        for val in vals:
            df_p = df[(df["parameter"] == p) & (df[column] == val)]

            ax = fig.add_subplot(gs[pc_x, pc_y])
            for g in groups:
                df_g = df_p[df_p["group"] == g]
                ax.hist(
                    df_g["point_bias"],
                    density=True,
                    bins="auto",
                    histtype="stepfilled",
                    alpha=0.5,
                    color=colors[g],
                )

                if column == "prior_type" and pc_y == 0:
                    ax.set_ylabel(value_labels[column][val], fontsize=8)
                elif pc_y == 0:
                    ax.set_ylabel(val, fontsize=8)

                legend_handles.append(mpatches.Patch(label=g, color=colors[g]))

                # if p in ["BW", "LI__ICGIM_Vmax"] and val == "prior_biased_1":
                #     console.print(df_g[["point_bias"]])

            if pc_x == 0:
                ax.set_title(parameter_labels[p])

            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

            pc_x += 1
            if pc_x == len(vals):
                pc_x = 0

        fig.supxlabel("Point Bias", fontsize=11, y=0.2)

        if len(pars) == 1:
            ylabel_x = -0.1
        else:
            ylabel_x = -0.01

        fig.supylabel(axis_labels[column], fontsize=11, x=ylabel_x, y=0.6)

        pc_y += 1

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
        bbox_to_anchor=(0.5, 0.1),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / f"{column}_bias_histogram.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=results_path, xp_type="all")

        bias_histogram(df=results, column="samples")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from parvar import RESULTS_SIMPLE_PK, RESULTS_ICG, RESULTS_SIMPLE_CHAIN
from parvar.analysis.utils import append_server_result, join_optimization_results
from matplotlib import gridspec
from matplotlib import patches as mpatches

from parvar.plots import colors


def bias_histogram(df: pd.DataFrame) -> None:
    pars = df["parameter"].unique()
    groups = df["group"].unique()

    def point_bias(df: pd.DataFrame) -> pd.DataFrame:
        df["point_bias"] = (
            np.abs(df["sample_loc"] - df["bayes_sampler_median"]) / df["sample_loc"]
        )
        return df

    df = point_bias(df)

    fig = plt.figure(
        dpi=360,
        figsize=(len(pars) * 4, 3),
    )

    gs = gridspec.GridSpec(
        1,
        len(pars),
        figure=fig,
        hspace=0.45,
        wspace=0.30,
        top=0.93,
        bottom=0.3,  # ← legend lives in this margin
        left=0.07,
        right=0.97,
    )

    pc_y = 0
    legend_handles = []

    for p in pars:
        df_p = df[df["parameter"] == p]

        ax = fig.add_subplot(gs[0, pc_y])
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

            legend_handles.append(mpatches.Patch(label=g, color=colors[g]))

            ax.set_title(p)
        fig.supxlabel("Point Bias", y=0.02)

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
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor="#FFFFFF",
        bbox_to_anchor=(0.5, 0.1),
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    plt.show()


if __name__ == "__main__":
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=results_path, xp_type="all")

        bias_histogram(df=results)

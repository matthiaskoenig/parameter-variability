import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from parvar import RESULTS_SIMPLE_PK, RESULTS_ICG, RESULTS_SIMPLE_CHAIN
from parvar.analysis.utils import (
    # append_server_result,
    join_optimization_results,
    reference_df_filter,
)
from matplotlib import gridspec
from matplotlib import patches as mpatches
from parvar.plots import colors, parameter_labels, value_labels, axis_labels


def bias_histogram(
    df: pd.DataFrame,
    column: str = "prior_type",
    save_path: Path = None,
) -> None:
    xp_id = df["id"].unique()
    model = df["model"].unique()
    pars = df["parameter"].unique()
    groups = df["group"].unique()
    vals = df[column].unique()

    # Only accept maximum of 4 values
    if len(vals) > 4:
        vals = vals[-4:]

    df = reference_df_filter(column, df, reference)

    def point_bias(df: pd.DataFrame, array: np.array) -> np.array:
        return (df["sample_loc"].to_numpy() - array) / df["sample_loc"].to_numpy()

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
    logs: list[dict] = []
    for p in pars:
        for val in vals:
            df_p = df[(df["parameter"] == p) & (df[column] == val)]

            ax = fig.add_subplot(gs[pc_x, pc_y])
            for g in groups:
                df_g = df_p[df_p["group"] == g]

                array_string = df_g["bayes_sampler_values"].values[0]
                bayes_samples = np.array(ast.literal_eval(array_string)).flatten()

                bayes_samples_diff = point_bias(df_g, bayes_samples)
                # console.print(type(bayes_samples_diff))
                # exit()
                ax.hist(
                    bayes_samples_diff,
                    density=True,
                    bins="auto",
                    histtype="stepfilled",
                    alpha=0.5,
                    color=colors[g],
                )

                ax.axvline(
                    np.median(bayes_samples_diff),
                    color=colors[g],
                    linewidth=1.0,
                    linestyle="--",
                    zorder=5,
                    alpha=0.7,
                )
                if save_path:
                    log = {
                        "id": xp_id,
                        "model": model,
                        "parameter": p,
                        "groups": g,
                        "prior_type": p,
                        "column": column,
                        "median_point_bias": np.median(bayes_samples_diff),
                    }
                    logs.append(log)

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

        fig.supxlabel("Point Bias", fontsize=11, y=0.1)

        if len(pars) == 1:
            ylabel_x = -0.1
        else:
            ylabel_x = -0.01

        fig.supylabel(axis_labels[column], fontsize=11, x=ylabel_x, y=0.6)

        pc_y += 1

    # seen = set()
    # unique_handles = []
    # for h in legend_handles:
    #     label = h.get_label()
    #     if label not in seen:
    #         seen.add(label)
    #         unique_handles.append(h)
    # fig.legend(
    #     handles=unique_handles,
    #     loc="lower center",
    #     ncol=2,
    #     fontsize=9,
    #     bbox_to_anchor=(0.5, 0.1),
    # )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / f"{column}_bias_histogram.png", bbox_inches="tight")
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(save_path / f"{column}_bias_results.tsv", sep="\t", index=False)

    plt.show()


if __name__ == "__main__":
    reference = {
        "prior_type": "exact_prior",
        "timepoints": 10,
        "samples": 20,
        "noise_cv": 0.1,
    }

    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        # results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=r, xp_type="timepoints")

        bias_histogram(df=results, column="timepoints")

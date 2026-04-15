import ast
from pathlib import Path

from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.utils import join_optimization_results
from parvar.plots import colors, parameter_labels, axis_labels, value_labels

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pymetadata.console import console


def grouped_boxplot(
    df: pd.DataFrame,
    reference: dict = None,
    column: str = "samples",
    save_path: Path = None,
) -> None:
    pars = df["parameter"].unique()
    values = df[column].unique()
    groups = df["group"].unique()
    console.print(reference)
    n_pars = len(pars)
    n_x = len(values)
    n_groups = len(groups)

    reference_cp = reference.copy()
    reference_cp.pop(column)

    mask = pd.Series([True] * len(df), index=df.index)

    for col, val in reference_cp.items():
        mask &= df[col] == val

    df = df[mask]

    group_width = 0.7  # total width occupied by all boxes at one x tick
    box_width = group_width / n_groups
    offsets = np.linspace(
        -(group_width - box_width) / 2,
        (group_width - box_width) / 2,
        n_groups,
    )

    fig, axes = plt.subplots(
        1,
        n_pars,
        figsize=(6 * n_pars, 5),
        sharey=False,
    )

    if n_pars == 1:
        axes = [axes]

    for ax, par in zip(axes, pars):
        df_p = df[df["parameter"] == par]

        for g_idx, group in enumerate(groups):
            df_g = df_p[df_p["group"] == group]

            positions = []
            box_data = []
            for v_idx, val in enumerate(values):
                array_string = df_g[df_g[column] == val]["bayes_sampler_values"].values[
                    0
                ]
                cell = np.array(ast.literal_eval(array_string)).flatten().tolist()
                # if cell:
                positions.append(v_idx + offsets[g_idx])
                box_data.append(cell)

            if not box_data:
                continue

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                notch=False,
                vert=True,
                manage_ticks=False,
            )

            color = colors[group]

            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("white")
                patch.set_linewidth(1.2)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_color(color)
                    line.set_linewidth(1.2)
            for median in bp["medians"]:
                median.set_color("white")
                median.set_linewidth(2)
            for flier in bp["fliers"]:
                flier.set(
                    marker="o",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=4,
                    alpha=0.6,
                )

            ax.axhline(
                df_g["sample_loc"].to_list()[0],
                color=color,
                linewidth=1.8,
                linestyle="--",
                zorder=5,
            )
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01

            if par != "LI__ICGIM_Vmax":  # exclude bc text clashes
                ax.text(
                    n_x - 0.45,
                    df_g["sample_loc"].to_list()[0] + y_offset,
                    f"True {group} {par} = {df_g['sample_loc'].to_list()[0]}",
                    color="black",
                    fontsize=8,
                    va="bottom",
                    ha="right",
                )

        ax.set_xticks(range(n_x))
        if column == "prior_type":
            ax.set_xticklabels(
                [value_labels[column][val] for val in values], fontsize=10
            )
        else:
            ax.set_xticklabels(values, fontsize=10)
        ax.set_xlim(-0.5, n_x - 0.5)
        ax.set_title(parameter_labels[par], fontsize=13, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.supxlabel(axis_labels[column], fontsize=11)
    fig.supylabel("Posterior Median", fontsize=11)
    legend_handles: list = [
        mpatches.Patch(facecolor=colors[g], alpha=0.75, edgecolor="white", label=g)
        for g in groups
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                color=colors[g],
                linewidth=1.8,
                linestyle="--",
                label=f"True {g} parameter",
            )
            for g in groups
        ]
    )
    leg = fig.legend(
        handles=legend_handles,
        # title="Groups:",
        loc="lower center",
        ncol=n_groups,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(0.5, -0.1),
    )

    plt.tight_layout()
    fig.canvas.draw()

    # 3. Measure the legend's true height and push subplots up by that amount
    renderer = fig.canvas.get_renderer()
    leg_frac = (
        leg.get_window_extent(renderer).height / fig.get_window_extent(renderer).height
    )
    fig.subplots_adjust(bottom=leg_frac + 0.01)

    if save_path:
        plt.savefig(save_path / f"{column}_boxplot.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    reference = {
        "prior_type": "exact_prior",
        "timepoints": 10,
        "samples": 20,
        "noise_cv": 0.1,
    }
    # [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        # results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=r, xp_type="timepoints")
        # console.print(results['bayes_sampler_values'][0])

        grouped_boxplot(results, reference, column="timepoints")

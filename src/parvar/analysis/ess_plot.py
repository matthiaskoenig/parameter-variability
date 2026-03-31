import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.utils import append_server_result, join_optimization_results


def ess_violinplot(
    df: pd.DataFrame,
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
        groups = [grp["ess"].values for _, grp in df.groupby(c)]
        labels = [key for key, _ in df.groupby(c)]

        parts = ax.violinplot(
            groups, positions=range(len(labels)), showmedians=True, showextrema=True
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)

        parts["cmedians"].set_color("navy")
        parts["cmins"].set_color("steelblue")
        parts["cmaxes"].set_color("steelblue")
        parts["cbars"].set_color("steelblue")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(c, fontsize=12)
        ax.set_xlabel(c, fontsize=10)
        # ax.set_ylabel("Effective Sample Size", fontsize=10)

        # ax.boxplot(
        #     groups,
        #     labels=labels,
        #     patch_artist=True,
        #     boxprops=dict(facecolor="#AED6F1", color="#2471A3"),
        #     medianprops=dict(color="#E74C3C", linewidth=2),
        #     whiskerprops=dict(color="#2471A3"),
        #     capprops=dict(color="#2471A3"),
        #     flierprops=dict(marker="o", color="#AAA", markersize=4),
        # )

        # ax.set_xlabel(c, fontsize=12)

    fig.supylabel("Effective Sample Size", fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")
        results = join_optimization_results(results_path=results_path, xp_type="all")

        ess_violinplot(results)

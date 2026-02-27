from pathlib import Path

import pandas as pd
from console import console

from parvar import RESULTS_SIMPLE_PK
from parvar.analysis.utils import extract_key_from_dict

from matplotlib import pyplot as plt
from parvar.plots import colors


def join_optimization_results(
    results_path: Path,
    xp_type: str,
) -> pd.DataFrame:
    directories: Path = results_path / "xps" / xp_type
    filenames = [d for d in directories.glob("**/optimization_results.tsv")]

    df_ls = []
    for filename in filenames:
        df = pd.read_csv(filename, sep="\t")
        df_ls.append(df)

    df_bayes = pd.concat(df_ls)
    df_xp = pd.read_csv(directories / "definitions.tsv", sep="\t")

    df_join = df_xp.merge(df_bayes, on=["id", "group", "parameter"], how="inner")
    df_join["sample_loc"] = extract_key_from_dict(df_join["dsn_par"], "loc")
    df_join["sample_scale"] = extract_key_from_dict(df_join["dsn_par"], "scale")

    col_rename = {
        "mean": "bayes_sampler_mean",
        "median": "bayes_sampler_median",
        "n_samples": "bayes_sampler_n_samples",
        "values": "bayes_sampler_values",
    }

    df_join.rename(columns=col_rename, inplace=True)

    col_order = [
        "id",
        "model",
        "prior_type",
        "group",
        "parameter",
        "samples",
        "timepoints",
        "noise_cv",
        "sample_loc",
        "sample_scale",
        "bayes_sampler_mean",
        "bayes_sampler_median",
        "bayes_sampler_n_samples",
        "bayes_sampler_values",
        "hdi_high",
        "hdi_low",
    ]

    df = df_join[col_order]

    return df


def prior_types_plot(
    df: pd.DataFrame,
) -> None:
    pars = df["parameter"].unique()
    # groups = df["group"].unique()
    prior_types = df["prior_type"].unique()
    fig, axs = plt.subplots(nrows=len(prior_types), ncols=len(pars))

    # offset = lambda p: transforms.ScaledTranslation(p / 72., 0,
    #                                                 plt.gcf().dpi_scale_trans)
    # trans = plt.gca().transData
    for ax, p in zip(axs, pars):
        for g in ["FEMALE"]:
            df_gp = df[(df["group"] == g) & (df["parameter"] == p)]
            console.print(df_gp[["group", "parameter", "timepoints"]])
            ax.plot(
                df_gp["timepoints"],
                df_gp["sample_loc"],
                ".",
                label=g,
                color=colors[g],
            )
            # ax.plot(
            #     df_gp['prior_type'],
            #     df_gp['bayes_sampler_median'],
            #     '*',
            #     label=g,
            #     color=colors[g]
            #     # transform=trans + offset(0.5)
            # )
            ax.plot(
                df_gp["timepoints"],
                df_gp["bayes_sampler_median"],
                "*",
                color=colors[g],
                # yerr=[df_gp['hdi_low'],df_gp['hdi_high']],
            )
            console.print(f"{p}: {df_gp[['hdi_low', 'hdi_high']].to_numpy().tolist()}")
            ax.errorbar(
                df_gp["timepoints"],
                df_gp["bayes_sampler_median"],
                uplims=df_gp["hdi_high"].to_list(),
                # yerr=df_gp[['hdi_low','hdi_high']].to_numpy().tolist(),
                color=colors[g],
            )
            ax.set_title(p)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    results = join_optimization_results(
        results_path=RESULTS_SIMPLE_PK, xp_type="timepoints"
    )

    # results.to_csv(analysis.results_path / "xps" / "join.tsv", sep="\t", index=False)
    # console.print(
    #     results[[
    #     "prior_type",
    #     "group",
    #     "parameter",
    #     "sample_loc",
    #     "bayes_sampler_median"
    #     ]])
    prior_types_plot(results)

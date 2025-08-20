from pathlib import Path
import matplotlib

SMALL_SIZE = 13
MEDIUM_SIZE = 18
BIGGER_SIZE = 25

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
import pandas as pd
from matplotlib import pyplot as plt
from parameter_variability import RESULTS_ICG
from parameter_variability.bayes.pypesto.icg_body_flat.petab_optimization import (
    PyPestoSampler
)
from parameter_variability.console import console
from parameter_variability.bayes.pypesto.icg_body_flat.experiment_factory import (
    true_par
)

def optimize_petab_xp(yaml_file: Path) -> list[dict]:
    """Optimize single petab problem using PyPesto."""
    pypesto_sampler = PyPestoSampler(yaml_file=yaml_file)
    pypesto_sampler.load_problem()
    pypesto_sampler.optimizer()
    pypesto_sampler.bayesian_sampler(n_samples=1000)
    pypesto_sampler.results_hdi()
    # pypesto_sampler.results_median()

    results = []
    results_petab = pypesto_sampler.results_dict()
    for pid, stats in results_petab.items():
        results.append({
            "xp": yaml_file.parent.name,
            "pid": pid,
            **stats,
        })

    return results


def optimize_petab_xps(exp_type: str):
    yaml_files = sorted(
        [f for f in (RESULTS_ICG / exp_type).glob("**/petab.yaml")])
    console.print(yaml_files)

    infos = []
    for yaml_file in yaml_files:
        console.rule(yaml_file.name, style="white", align="left")
        results: list[dict] = optimize_petab_xp(yaml_file)
        infos.extend(results)

    df = pd.DataFrame(infos)
    df.to_csv(RESULTS_ICG / f"xps_{exp_type}.tsv", sep="\t", index=False)
    console.print(df)
    return df

colors = {
    "MALE": "tab:blue",
    "FEMALE": "tab:red",
}

def visualize_timepoints_samples():
    """Visualize the results."""

    for xp_key in ["n", "Nt"]:

        # data processing
        df = pd.read_csv(RESULTS_ICG / f"xps_{xp_key}.tsv", sep="\t")
        df[xp_key] = df.xp.str.split("_").str[-1]
        df[xp_key] = df[xp_key].astype(int)
        df["category"] = df.pid.str.split("_").str[-1]
        df = df.sort_values(by=xp_key)
        console.print(df)

        # visulazation

        f, ax = plt.subplots(dpi=300, layout="constrained", figsize=(6, 6))
        # plot the mean
        # FIXME: hard coded
        ax.axhline(y=1.0, label="MALE (exact)", linestyle="--", color=colors["MALE"])
        ax.axhline(y=2.0, label="FEMALE (exact)", linestyle="--", color=colors["FEMALE"])

        for category in df.category.unique():
            df_cat = df[df.category == category]
            ax.errorbar(
                x=df_cat[xp_key], y=df_cat["median"],
                yerr=df_cat["std"],
                # yerr=[df_cat["hdi_low"], df_cat["hdi_high"]],
                label=category,
                marker="o",
                color=colors[category],
                # linestyle="",
                markeredgecolor="black",
            )

        ax.set_xlabel(xp_key)
        ax.set_ylabel("Parameter k1")
        ax.legend()
        f.savefig(RESULTS_ICG / f"xps_{xp_key}.png", bbox_inches="tight")

    plt.show()

def visualize_priors():
    """Visualize the different priors."""

    # data processing
    df = pd.read_csv(RESULTS_ICG / f"xps_prior.tsv", sep="\t")
    df["category"] = df.pid.str.split("_").str[-1]
    df["prior"] = df.xp.str.split("_").str[-1]
    df["parameters"] = df.pid.str.split("_").str[:-1].str.join("_") # FIXME: Category names edge cases

    # visualization
    from matplotlib import pyplot as plt
    f, ax = plt.subplots(nrows=len(df['prior'].unique()),
                         ncols=len(df['parameters'].unique()),
                         dpi=300, layout="constrained", figsize=(6, 6))

    # plot the mean
    # FIXME: hard coded
    console.print(df.head())
    console.print(df['category'].unique())
    cats = df.groupby(['pid', 'category']).size().reset_index()
    for par in df['parameters'].unique():
        for cat in df['category'].unique():
            ax.axhline(y=true_par[f"{par}_{cat}"].distribution.parameters['loc'],
                       label=f"{par} (exact)", linestyle="--", color=colors[category])
    # ax.axhline(y=2.0, label="FEMALE (exact)", linestyle="--", color=colors["FEMALE"])
    exit()
    # plot the data
    for par, category in zip(cats['pid'], cats['category']):
        for k, prior in enumerate(["exact", "biased"]):  # ["exact", "biased", "noprior"]
            # TODO: plot x parameters in different subplots
            df_cat = df[(df['prior'] == prior) &
                        (df['category'] == category) &
                        (df['pid'] == par)]
            ax.errorbar(
                x=prior, y=df_cat["median"],
                yerr=df_cat["std"],
                # yerr=[df_cat["hdi_low"], df_cat["hdi_high"]],
                label=par,
                marker="o",
                color=colors[category],
                # linestyle="",
                markeredgecolor="black",
            )

            # FIXME: boxplot
            # ax.boxplot(df)



    ax.set_xlabel("n (samples)")
    ax.set_ylabel("Parameter k1")
    ax.legend()
    plt.show()
    f.savefig(RESULTS_ICG / f"xps_prior.png", bbox_inches="tight")


if __name__ == "__main__":

    # optimize_petab_xps(exp_type="prior")
    # optimize_petab_xps(exp_type="n")
    # optimize_petab_xps(exp_type="Nt")

    # visualize_timepoints_samples() # Only for n and Nt exps
    visualize_priors()


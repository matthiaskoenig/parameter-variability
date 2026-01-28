from pathlib import Path
from typing import Dict, Optional
from itertools import product

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
from parameter_variability import RESULTS_ICG
from parameter_variability.petab_optimization import (
    PyPestoSampler
)
from parameter_variability.console import console
# from parameter_variability.bayes.pypesto.icg_body_flat.experiment_factory import (
#     pars_true_icg
# )

def xps_selector(
    results_dir: Path,
    xp_type: str,
    conditions: Optional[Dict[str, list]]
) -> list[str]:
    """Select the xps that match the desired conditions."""
    df = pd.read_csv(results_dir / f'xps_{xp_type}' / 'results.tsv', sep='\t')

    if not conditions:  # empty dict -> no filtering
        return df

    if 'n_t' in conditions:
        conditions['n_t'] = [t - 1 for t in conditions['n_t']]

    combinations = list(product(*(conditions[col] for col in conditions)))

    matching_indices = set()

    for comb in combinations:
        comb_dict = dict(zip(conditions.keys(), comb))
        mask = pd.Series(True, index=df.index)
        for col, val in comb_dict.items():
            mask &= df[col].eq(val)
        matching_indices.update(df[mask].index)


    df_res = df.loc[list(matching_indices)].sort_index()

    if df_res.empty:
        raise console.print("No XPs were selected. Check if conditions are correct",
                            style="warning")

    return df_res['id'].unique().tolist()


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


def optimize_petab_xps(results_dir: Path, exp_type: str, xp_ids: list[str]):
    """Optimize the given PEtab problems."""

    xp_path = results_dir / f'xps_{exp_type}'
    yaml_files: list[Path] = []
    for xp in xp_path.iterdir():
        if xp.is_dir() and xp.name in xp_ids:
            for yaml_file in xp.glob("**/petab.yaml"):
                yaml_files.append(yaml_file)

    yaml_files = sorted(yaml_files)

    infos = []
    for yaml_file in yaml_files:
        console.rule(yaml_file.name, style="white", align="left")
        results: list[dict] = optimize_petab_xp(yaml_file)
        infos.extend(results)

    df = pd.DataFrame(infos)
    df.to_csv(RESULTS_ICG / f'xps_{exp_type}' / f"bayes_results.tsv", sep="\t", index=False)
    console.print(df)
    return df

colors = {
    "MALE": "tab:blue",
    "FEMALE": "tab:red",
}

# def visualize_timepoints_samples():
#     """Visualize the results."""
#
#     for xp_key in ["n", "Nt"]:
#
#         # data processing
#         df = pd.read_csv(RESULTS_ICG / f"xps_{xp_key}.tsv", sep="\t")
#         df[xp_key] = df.xp.str.split("_").str[-1]
#         df[xp_key] = df[xp_key].astype(int)
#         df["category"] = df.pid.str.split("_").str[-1]
#         df["parameters"] = df.pid.str.split("_").str[:-1].str.join("_")  # FIXME: Category names edge cases
#         df = df.sort_values(by=xp_key)
#         console.print(df)
#
#         # visulazation
#         f, axs = plt.subplots(nrows=len(df['parameters'].unique()),
#                               dpi=300, layout="constrained",
#                               figsize=(6, 4 * len(df['parameters'].unique())))
#
#         # plot the mean
#         for cat in df['category'].unique():
#             for ax, par in zip(axs, df['parameters'].unique()):
#                 ax.axhline(y=pars_true_icg[f"{par}_{cat}"].distribution.parameters['loc'],
#                            label=f"{par} (exact)", linestyle="--", color=colors[cat])
#
#                 df_cat = df[(df['category'] == cat) & (df['parameters'] == par)]
#                 ax.errorbar(
#                     x=df_cat[xp_key], y=df_cat["median"],
#                     yerr=df_cat["std"],
#                     # yerr=[df_cat["hdi_low"], df_cat["hdi_high"]],
#                     label=cat,
#                     marker="o",
#                     color=colors[cat],
#                     # linestyle="",
#                     markeredgecolor="black",
#                 )
#
#                 ax.set_xlabel(xp_key)
#                 ax.set_ylabel(f"Parameter {par}")
#                 ax.legend()
#         f.savefig(RESULTS_ICG / f"xps_{xp_key}.png", bbox_inches="tight")
#
#         plt.show()
#
# def visualize_priors():
#     """Visualize the different priors."""
#
#     # data processing
#     df = pd.read_csv(RESULTS_ICG / f"xps_prior.tsv", sep="\t")
#     df["category"] = df.pid.str.split("_").str[-1]
#     df["prior"] = df.xp.str.split("_").str[-1]
#     df["parameters"] = df.pid.str.split("_").str[:-1].str.join("_") # FIXME: Category names edge cases
#
#     # visualization
#     from matplotlib import pyplot as plt
#     f, axs = plt.subplots(nrows=len(df['parameters'].unique()),
#                           dpi=300, layout="constrained",
#                           figsize=(6, 4*len(df['parameters'].unique())))
#
#     # plot the mean
#     for cat in df['category'].unique():
#         for ax, par in zip(axs, df['parameters'].unique()):
#             ax.axhline(y=pars_true_icg[f"{par}_{cat}"].distribution.parameters['loc'],
#                        label=f"{par} (exact)", linestyle="--", color=colors[cat])
#             for k, prior in enumerate(["exact", "biased"]):  # ["exact", "biased", "noprior"]
#                 # TODO: plot x parameters in different subplots
#                 df_cat = df[(df['prior'] == prior) &
#                             (df['category'] == cat) &
#                             (df['parameters'] == par)]
#                 console.print(df_cat)
#                 ax.errorbar(
#                     x=prior, y=df_cat["median"],
#                     yerr=df_cat["std"],
#                     # yerr=[df_cat["hdi_low"], df_cat["hdi_high"]],
#                     label=f"{par}_{cat}",
#                     marker="o",
#                     color=colors[cat],
#                     # linestyle="",
#                     markeredgecolor="black",
#                 )
#
#             # FIXME: boxplot
#             # ax.boxplot(df)
#
#
#
#             ax.set_xlabel("n (samples)")
#             ax.set_ylabel(f"Parameter {par}")
#             ax.legend()
#     plt.show()
#     f.savefig(RESULTS_ICG / f"xps_prior.png", bbox_inches="tight")


if __name__ == "__main__":
    # FIXME: Remove AMICI Models everytime the Observed Compartments are redefined
    #   or for every run

    # xp_ids = xps_selector(
    #     xp_type='all',
    #     conditions={
    #         'prior_type': ['prior_biased', 'exact_prior'],
    #         'n_t': [11, 21, 41, 81],
    #         'noise_cv': [0.0, 0.001, 0.01]
    #     })
    # console.print(xp_ids)

    # optimize_petab_xps(
    #     exp_type='all',
    #     xp_ids=xp_ids
    # )

    yaml_path = Path("/home/mkoenig/git/parameter-variability/results/icg_body_flat/xps_all/1a4bKyK5fJxx1C1O5n0D/petab.yaml")
    results = optimize_petab_xp(yaml_file=yaml_path)
    console.print(results)

    # visualize_timepoints_samples() # Only for n and Nt exps
    # visualize_priors()

